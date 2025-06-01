import random
from tensorflow.keras.models import load_model
from spectrogram import load_and_process, display_spectrogram
from model import create_model
import os
import numpy as np

def predict_and_display(model, filepath):
    if not os.path.exists(filepath):
        print("File: "+ filepath+" does not exist")
        return

    spec = load_and_process(filepath)
    display_spectrogram(spec, os.path.basename(filepath))
    spec = spec[np.newaxis, ..., np.newaxis]

    pred = model.predict(spec)[0][0]
    if pred >= 0.5:
        label = "Dog"
    else:
        label = "Cat"

    filename = os.path.basename(filepath)
    rounded_pred = round(pred, 2)
    print(filename + " → Prediction: " + label + " (" + str(round(pred * 100, 2)) + "% chance that its dog)")



if __name__ == "__main__":
    model = load_model("cat_dog_sound_model.h5")
    test_files = []

    for filename in os.listdir("test_sounds"):
        if filename.lower().endswith(".wav"):
            full_path = os.path.join("test_sounds", filename)
            test_files.append(full_path)

    random.shuffle(test_files)

    for filepath in test_files:
        predict_and_display(model, filepath)
