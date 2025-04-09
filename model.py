import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = F.avg_pool2d(x, x.size()[2:])
        attention = F.relu(self.conv1(attention))
        attention = self.sigmoid(self.conv2(attention))
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.attention = AttentionBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MusicNoteClassifier(nn.Module):
    def __init__(self, num_notes=24, num_octaves=11):
        super(MusicNoteClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 512, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.15)
        
        self.layer1 = self._make_layer(512, 1024, 6)
        self.layer2 = self._make_layer(1024, 2048, 6, stride=2)
        self.layer3 = self._make_layer(2048, 4096, 6, stride=2)
        
        self.dropout = nn.Dropout(0.2)
        self.fc_input_size = 4096 * 8 * 8
        
        self.fc1_octave = nn.Linear(self.fc_input_size, 8192)
        self.fc2_octave = nn.Linear(8192, 4096)
        self.fc3_octave = nn.Linear(4096, 2048)
        self.fc4_octave = nn.Linear(2048, 1024)
        self.fc5_octave = nn.Linear(1024, num_octaves)
        
        self.fc1_note = nn.Linear(self.fc_input_size + num_octaves, 16384)
        self.fc2_note = nn.Linear(16384, 8192)
        self.fc3_note = nn.Linear(8192, 4096)
        self.fc4_note = nn.Linear(4096, 2048)
        self.fc5_note = nn.Linear(2048, 1024)
        self.fc6_note = nn.Linear(1024, num_notes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(-1, self.fc_input_size)
        
        octave_features = F.relu(self.fc1_octave(x))
        octave_features = self.dropout(octave_features)
        octave_features = F.relu(self.fc2_octave(octave_features))
        octave_features = self.dropout(octave_features)
        octave_features = F.relu(self.fc3_octave(octave_features))
        octave_features = self.dropout(octave_features)
        octave_features = F.relu(self.fc4_octave(octave_features))
        octave_features = self.dropout(octave_features)
        octave_output = self.fc5_octave(octave_features)
        
        octave_probs = F.softmax(octave_output, dim=1)
        combined_features = torch.cat([x, octave_probs], dim=1)
        
        note_features = F.relu(self.fc1_note(combined_features))
        note_features = self.dropout(note_features)
        note_features = F.relu(self.fc2_note(note_features))
        note_features = self.dropout(note_features)
        note_features = F.relu(self.fc3_note(note_features))
        note_features = self.dropout(note_features)
        note_features = F.relu(self.fc4_note(note_features))
        note_features = self.dropout(note_features)
        note_features = F.relu(self.fc5_note(note_features))
        note_features = self.dropout(note_features)
        note_output = self.fc6_note(note_features)
        
        return note_output, octave_output
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            note_output, octave_output = self(x)
            note_pred = torch.argmax(note_output, dim=1)
            octave_pred = torch.argmax(octave_output, dim=1)
        return note_pred, octave_pred 