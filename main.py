from spectrogram import load_and_process, display_spectrogram
from model import create_model
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback

def load_data():
    cat_spectrograms = []
    dog_spectrograms = []

    for i in range(1, 38):
        cat_path = 'cat\cat'+str(i)+'.wav'
        dog_path = 'dog\dog'+str(i)+'.wav'
        #print(cat_path)
        #print(dog_path)
        if os.path.exists(cat_path):
            cat_spectrograms.append(load_and_process(cat_path))
        if os.path.exists(dog_path):
            dog_spectrograms.append(load_and_process(dog_path))
    return cat_spectrograms, dog_spectrograms

def train():
    cat_spectrograms, dog_spectrograms = load_data()

    X = np.array(cat_spectrograms + dog_spectrograms)
    y = np.array([0] * len(cat_spectrograms) + [1] * len(dog_spectrograms))

    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(input_shape=X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=8, callbacks=[early_stop])

    loss, acc = model.evaluate(X_test, y_test)
    print("Accuracy: "+str(round(acc*100, 2))+"%")

    model.save("cat_dog_sound_model.h5")

    return model

if __name__ == "__main__":
    trained_model = train()
