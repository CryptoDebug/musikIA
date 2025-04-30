import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import os
from data_preparation import create_data_loaders
from model import MusicNoteClassifier, FocalLoss
import gc
from colorama import Fore, Style

def train_model(model, train_loader, val_loader, num_epochs=400, device='cuda'):
    model = model.to(device)
    note_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    octave_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.0001)
    
    best_val_note_acc = 0
    best_epoch = 0
    best_model_path = None
    os.makedirs('models', exist_ok=True)
    patience = 50
    counter = 0
    
    print(f"\n{Fore.CYAN}Début de l'entraînement{Style.RESET_ALL}")
    print(f"Nombre d'époques maximum: {num_epochs}")
    print(f"Taille du dataset: {len(train_loader.dataset)} échantillons d'entraînement, {len(val_loader.dataset)} échantillons de validation")
    print(f"Device utilisé: {device}")
    print(f"Learning rate initial: {optimizer.param_groups[0]['lr']}")
    print(f"Patience early stopping: {patience} époques\n")
    
    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        
        model.train()
        train_loss = 0.0
        train_note_correct = 0
        train_octave_correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f'Entraînement - Époque {epoch+1}/{num_epochs}'):
            spectrograms = batch['spectrogram'].to(device)
            note_labels = batch['note_label'].to(device)
            octave_labels = batch['octave_label'].to(device)
            
            optimizer.zero_grad()
            note_output, octave_output = model(spectrograms)
            
            note_loss = note_criterion(note_output, note_labels) * 15.0
            octave_loss = octave_criterion(octave_output, octave_labels) * 1.0
            loss = note_loss + octave_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, note_pred = torch.max(note_output, 1)
            _, octave_pred = torch.max(octave_output, 1)
            train_note_correct += (note_pred == note_labels).sum().item()
            train_octave_correct += (octave_pred == octave_labels).sum().item()
            total += note_labels.size(0)
            
            del spectrograms, note_labels, octave_labels, note_output, octave_output, loss
            torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader)
        train_note_acc = 100 * train_note_correct / total
        train_octave_acc = 100 * train_octave_correct / total
        
        model.eval()
        val_loss = 0.0
        val_note_correct = 0
        val_octave_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation - Époque {epoch+1}/{num_epochs}'):
                spectrograms = batch['spectrogram'].to(device)
                note_labels = batch['note_label'].to(device)
                octave_labels = batch['octave_label'].to(device)
                
                note_output, octave_output = model(spectrograms)
                
                note_loss = note_criterion(note_output, note_labels) * 15.0
                octave_loss = octave_criterion(octave_output, octave_labels) * 1.0
                loss = note_loss + octave_loss
                
                val_loss += loss.item()
                _, note_pred = torch.max(note_output, 1)
                _, octave_pred = torch.max(octave_output, 1)
                val_note_correct += (note_pred == note_labels).sum().item()
                val_octave_correct += (octave_pred == octave_labels).sum().item()
                total += note_labels.size(0)
                
                del spectrograms, note_labels, octave_labels, note_output, octave_output, loss
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(val_loader)
        val_note_acc = 100 * val_note_correct / total
        val_octave_acc = 100 * val_octave_correct / total
        
        print(f"\n{Fore.GREEN}Époque {epoch+1}/{num_epochs}{Style.RESET_ALL}")
        print(f"Entraînement - Perte: {train_loss:.4f} | Notes: {train_note_acc:.2f}% | Octaves: {train_octave_acc:.2f}%")
        print(f"Validation   - Perte: {val_loss:.4f} | Notes: {val_note_acc:.2f}% | Octaves: {val_octave_acc:.2f}%")
        print(f"Learning rate actuel: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        if val_note_acc > best_val_note_acc:
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_val_note_acc = val_note_acc
            best_epoch = epoch
            
            best_model_path = f'models/best_model_epoch{epoch+1}_note{val_note_acc:.1f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_note_acc': val_note_acc,
                'val_octave_acc': val_octave_acc,
            }, best_model_path)
            print(f"{Fore.YELLOW}✨ Nouveau meilleur modèle sauvegardé! (Précision notes: {val_note_acc:.1f}%){Style.RESET_ALL}")
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"\n{Fore.RED}Early stopping déclenché après {epoch+1} époques{Style.RESET_ALL}")
            break
    
    print(f"\n{Fore.CYAN}Entraînement terminé!{Style.RESET_ALL}")
    print(f"Meilleures performances (Époque {best_epoch+1}):")
    print(f"Précision notes: {best_val_note_acc:.2f}%")
    print(f"Modèle sauvegardé dans: {best_model_path}")

def main():
    train_loader, val_loader, (note_encoder, octave_encoder) = create_data_loaders('notes_db')
    
    num_notes = len(note_encoder.classes_)
    num_octaves = len(octave_encoder.classes_)
    print(f"Création du modèle avec {num_notes} notes et {num_octaves} octaves")
    model = MusicNoteClassifier(num_notes=num_notes, num_octaves=num_octaves)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_model(model, train_loader, val_loader, device=device)
    
    os.makedirs('models', exist_ok=True)
    import joblib
    joblib.dump(note_encoder, 'models/note_encoder.pkl')
    joblib.dump(octave_encoder, 'models/octave_encoder.pkl')

if __name__ == '__main__':
    main() 