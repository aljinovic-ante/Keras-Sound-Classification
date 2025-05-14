from spectrogram import load_and_process, display_spectrogram
import os
import numpy as np

cat_spectrograms = []
dog_spectrograms = []

for i in range(1, 40):
    cat_filename = 'cat' + str(i) + '.wav'
    dog_filename = 'dog' + str(i) + '.wav'
    cat_path = os.path.join('cat', cat_filename)
    dog_path = os.path.join('dog', dog_filename)
    if os.path.exists(cat_path):
        cat_spectrograms.append(load_and_process(cat_path))
    if os.path.exists(dog_path):
        dog_spectrograms.append(load_and_process(dog_path))

display_spectrogram(cat_spectrograms[0], "Cat 1")
display_spectrogram(dog_spectrograms[0], "Dog 1")