import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.3)
        
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
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MusicNoteClassifier(nn.Module):
    def __init__(self, num_notes=12, num_octaves=11):
        super(MusicNoteClassifier, self).__init__()
        
        # Couche initiales + filtres
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.4)
        
        # Blocs résiduels + filtres
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        self.dropout = nn.Dropout(0.6)
        
        # Calcul de la taille après les couches de convolution
        # Input: [batch_size, 1, 128, 128]
        # Après conv1: [batch_size, 64, 64, 64]
        # Après maxpool: [batch_size, 64, 32, 32]
        # Après layer1: [batch_size, 128, 32, 32]
        # Après layer2: [batch_size, 256, 16, 16]
        # Après layer3: [batch_size, 512, 8, 8]
        self.fc_input_size = 512 * 8 * 8
        
        # Fully connected pour classification des notes avec plus de neurones
        self.fc1 = nn.Linear(self.fc_input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_notes)
        
        # Couches fully connected classification octaves
        self.fc1_octave = nn.Linear(self.fc_input_size, 1024)
        self.fc2_octave = nn.Linear(1024, num_octaves)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Extraction caractéristiques
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(-1, self.fc_input_size)
        
        # Classification notes avec couches, dropout ++
        note_features = F.relu(self.fc1(x))
        note_features = self.dropout(note_features)
        note_features = F.relu(self.fc2(note_features))
        note_features = self.dropout(note_features)
        note_output = self.fc3(note_features)
        
        # Classification des octaves
        octave_features = F.relu(self.fc1_octave(x))
        octave_features = self.dropout(octave_features)
        octave_output = self.fc2_octave(octave_features)
        
        return note_output, octave_output
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            note_output, octave_output = self(x)
            note_pred = torch.argmax(note_output, dim=1)
            octave_pred = torch.argmax(octave_output, dim=1)
        return note_pred, octave_pred 