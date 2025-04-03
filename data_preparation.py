import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MusicNoteDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_length=128):
        self.root_dir = root_dir
        self.transform = transform
        self.target_length = target_length
        self.samples = []
        self.note_encoder = LabelEncoder()
        self.octave_encoder = LabelEncoder()
        
        for octave in range(11):
            octave_dir = os.path.join(root_dir, f"Octave {octave}")
            if not os.path.exists(octave_dir):
                continue
                
            for filename in os.listdir(octave_dir):
                if filename.endswith(".wav"):
                    note_name = filename.split(".")[0]
                    self.samples.append({
                        'path': os.path.join(octave_dir, filename),
                        'note': note_name[:-1],
                        'octave': int(note_name[-1])
                    })
        
        self.note_encoder.fit([sample['note'] for sample in self.samples])
        self.octave_encoder.fit([sample['octave'] for sample in self.samples])
        
        unique_notes = np.unique([sample['note'] for sample in self.samples])
        unique_octaves = np.unique([sample['octave'] for sample in self.samples])
        print(f"Notes uniques: {unique_notes}")
        print(f"Octaves uniques: {unique_octaves}")
        print(f"Nombre de classes de notes: {len(self.note_encoder.classes_)}")
        print(f"Nombre de classes d'octaves: {len(self.octave_encoder.classes_)}")
        
        print(f"Dataset créé avec {len(self.samples)} échantillons")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        audio, sr = librosa.load(sample['path'], sr=None)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        if mel_spec_norm.shape[1] != self.target_length:
            mel_spec_norm = librosa.util.fix_length(mel_spec_norm, size=self.target_length, axis=1)
        
        assert mel_spec_norm.shape == (128, self.target_length), f"Taille incorrecte: {mel_spec_norm.shape}"
        
        mel_spec_tensor = torch.FloatTensor(mel_spec_norm)
        
        note_label = self.note_encoder.transform([sample['note']])[0]
        octave_label = self.octave_encoder.transform([sample['octave']])[0]
        
        assert note_label < len(self.note_encoder.classes_), f"Label de note invalide: {note_label}"
        assert octave_label < len(self.octave_encoder.classes_), f"Label d'octave invalide: {octave_label}"
        
        return {
            'spectrogram': mel_spec_tensor,
            'note_label': torch.LongTensor([note_label])[0],
            'octave_label': torch.LongTensor([octave_label])[0]
        }
    
    def get_encoders(self):
        return self.note_encoder, self.octave_encoder

def create_data_loaders(root_dir, batch_size=32, train_split=0.8, target_length=128):
    dataset = MusicNoteDataset(root_dir, target_length=target_length)
    
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(train_split * len(dataset)))
    train_indices, val_indices = indices[:split], indices[split:]
    
    print(f"Nombre d'échantillons d'entraînement: {len(train_indices)}")
    print(f"Nombre d'échantillons de validation: {len(val_indices)}")
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.get_encoders() 