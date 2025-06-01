import librosa
import matplotlib.pyplot as plt
import numpy as np

def load_and_process(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :128]
    return mel_db


def display_spectrogram(spectrogram, label):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title("Spectrogram - " + label)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.show()
