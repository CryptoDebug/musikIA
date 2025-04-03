import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import os
from data_preparation import create_data_loaders
from model import MusicNoteClassifier

def train_model(model, train_loader, val_loader, num_epochs=1000, device='cuda'):
    model = model.to(device)
    
    # Enhanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Combined learning rate schedulers
    scheduler1 = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_val_note_acc = 0
    best_val_octave_acc = 0
    best_epoch = 0
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Early stopping parameters
    patience = 30
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_note_correct = 0
        train_octave_correct = 0
        total = 0
        
        # Training loop with gradient clipping
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            spectrograms = batch['spectrogram'].to(device)
            note_labels = batch['note_label'].to(device)
            octave_labels = batch['octave_label'].to(device)
            
            optimizer.zero_grad()
            note_output, octave_output = model(spectrograms)
            
            # Enhanced loss weighting
            note_loss = criterion(note_output, note_labels) * 8.0  # Increased weight for notes
            octave_loss = criterion(octave_output, octave_labels)
            loss = note_loss + octave_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            _, note_pred = torch.max(note_output, 1)
            _, octave_pred = torch.max(octave_output, 1)
            train_note_correct += (note_pred == note_labels).sum().item()
            train_octave_correct += (octave_pred == octave_labels).sum().item()
            total += note_labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_note_acc = 100 * train_note_correct / total
        train_octave_acc = 100 * train_octave_correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_note_correct = 0
        val_octave_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                spectrograms = batch['spectrogram'].to(device)
                note_labels = batch['note_label'].to(device)
                octave_labels = batch['octave_label'].to(device)
                
                note_output, octave_output = model(spectrograms)
                
                note_loss = criterion(note_output, note_labels) * 8.0
                octave_loss = criterion(octave_output, octave_labels)
                loss = note_loss + octave_loss
                
                val_loss += loss.item()
                
                _, note_pred = torch.max(note_output, 1)
                _, octave_pred = torch.max(octave_output, 1)
                val_note_correct += (note_pred == note_labels).sum().item()
                val_octave_correct += (octave_pred == octave_labels).sum().item()
                total += note_labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_note_acc = 100 * val_note_correct / total
        val_octave_acc = 100 * val_octave_correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Note Acc: {train_note_acc:.2f}%, Train Octave Acc: {train_octave_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Note Acc: {val_note_acc:.2f}%, Val Octave Acc: {val_octave_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Update learning rate
        scheduler1.step(val_loss)
        scheduler2.step()
        
        # Save best model
        if val_note_acc > best_val_note_acc or val_octave_acc > best_val_octave_acc:
            if val_note_acc > best_val_note_acc:
                best_val_note_acc = val_note_acc
            if val_octave_acc > best_val_octave_acc:
                best_val_octave_acc = val_octave_acc
            best_epoch = epoch
            best_val_loss = val_loss
            
            model_path = f'models/best_model_epoch{epoch+1}_note{val_note_acc:.1f}_octave{val_octave_acc:.1f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_note_acc': val_note_acc,
                'val_octave_acc': val_octave_acc,
            }, model_path)
            print(f"Nouveau meilleur modèle sauvegardé dans {model_path}!")
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nMeilleures performances (Epoch {best_epoch+1}):")
    print(f"Note Accuracy: {best_val_note_acc:.2f}%")
    print(f"Octave Accuracy: {best_val_octave_acc:.2f}%")

def main():
    train_loader, val_loader, (note_encoder, octave_encoder) = create_data_loaders('notes_db')
    
    num_notes = len(note_encoder.classes_)
    num_octaves = len(octave_encoder.classes_)
    print(f"Création du modèle avec {num_notes} notes et {num_octaves} octaves")
    model = MusicNoteClassifier(num_notes=num_notes, num_octaves=num_octaves)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_model(model, train_loader, val_loader, device=device)
    
    # Save encoders in models directory
    os.makedirs('models', exist_ok=True)
    import joblib
    joblib.dump(note_encoder, 'models/note_encoder.pkl')
    joblib.dump(octave_encoder, 'models/octave_encoder.pkl')

if __name__ == '__main__':
    main() 