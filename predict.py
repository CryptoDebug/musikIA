import torch
import librosa
import numpy as np
import joblib
from model import MusicNoteClassifier

def load_model(model_path, note_encoder_path, octave_encoder_path):
    note_encoder = joblib.load(note_encoder_path)
    octave_encoder = joblib.load(octave_encoder_path)    
    model = MusicNoteClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, note_encoder, octave_encoder

def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_tensor = torch.FloatTensor(mel_spec_norm)
    
    return mel_spec_tensor

def predict_note(model, audio_path, note_encoder, octave_encoder, device='cuda'):
    mel_spec = preprocess_audio(audio_path)
    mel_spec = mel_spec.to(device)
    note_pred, octave_pred = model.predict(mel_spec.unsqueeze(0))
    note_name = note_encoder.inverse_transform([note_pred.item()])[0]
    octave = octave_encoder.inverse_transform([octave_pred.item()])[0]
    
    return note_name, octave

def main():
    model_path = 'best_model.pth' # Chemin vers le modèle
    note_encoder_path = 'note_encoder.pkl' # Chemin vers l'encodeur de notes
    octave_encoder_path = 'octave_encoder.pkl' # Chemin vers l'encodeur d'octaves
    audio_path = 'audio_to_transcribe.wav' # Fichier audio à transcrire
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, note_encoder, octave_encoder = load_model(model_path, note_encoder_path, octave_encoder_path)
    model = model.to(device)
    
    note_name, octave = predict_note(model, audio_path, note_encoder, octave_encoder, device)
    
    print(f"Note prédite : {note_name}{octave}")

if __name__ == '__main__':
    main() 