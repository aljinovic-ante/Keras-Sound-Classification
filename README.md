# 🐱🐶 Cat vs. Dog Sound Classifier

This project is a deep learning model that classifies audio recordings as either **cat** or **dog** sounds using spectrograms and a Convolutional Neural Network (CNN).

---

## 📁 Project Structure

```
├── cat/                      # Folder with cat .wav files
├── dog/                      # Folder with dog .wav files
├── test_sounds/              # Folder with test .wav files for prediction
├── spectrogram.py            # Spectrogram processing & visualization
├── model.py                  # CNN model definition
├── main.py                   # Model training script
├── predict.py                # Script to make predictions on new audio
├── cat_dog_sound_model.h5    # Trained model (generated after training)
```

---

## 🧠 How It Works

1. Audio files are loaded and converted into Mel spectrograms using `librosa`.
2. These spectrograms are resized to a uniform size of `128 x 128`.
3. A CNN model is trained to recognize cat vs. dog sounds from the spectrograms.
4. The trained model can predict new audio samples and visualize the spectrogram.

---

## 🚀 Training

Make sure you have folders `cat/` and `dog/` with `.wav` files named like `cat1.wav`, `dog2.wav`, etc.

Run the training script:

```bash
python main.py
```

This will:
- Load spectrograms from the cat and dog folders.
- Train the CNN model.
- Save the trained model as `cat_dog_sound_model.h5`.

---

## 🧪 Prediction

Put your `.wav` test files in the `test_sounds/` folder.

Then run:

```bash
python predict.py
```

This will:
- Load the trained model.
- Predict whether each test file is a cat or dog sound.
- Display its spectrogram and print the result.

---

## 🖼️ Example

The `display_spectrogram` function shows a spectrogram like this:

# Dog:
![image](https://github.com/user-attachments/assets/ddfb173f-d523-40e4-b36d-258cf2d70d42)
# And it makes a prediction based on that spectrogram:
![image](https://github.com/user-attachments/assets/bea76af9-7986-46de-9d2a-7719fd22e684)


# Cat:
![image](https://github.com/user-attachments/assets/6489cf17-a7f6-48dc-aa31-3a2c25b88e37)
# And it makes a prediction based on that spectrogram:
![image](https://github.com/user-attachments/assets/930d5fb1-2c29-4753-8aac-9cdd20ef1b60)




---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install tensorflow librosa matplotlib scikit-learn numpy
```

---

## 📌 Notes

- Binary classification: `0 = Cat`, `1 = Dog`.
- Model uses a CNN with:
  - 2 convolutional blocks
  - Dropout for regularization
  - Sigmoid activation in final layer for probability output

