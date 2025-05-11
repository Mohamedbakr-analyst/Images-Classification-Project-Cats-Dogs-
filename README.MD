# Image Classifier using CNN and Tkinter Interface

## 🧠 Overview

This project is a complete image classification pipeline using Convolutional Neural Networks (CNN) in TensorFlow/Keras. It includes:

* Preprocessing image data from a custom dataset
* Building and training a CNN model
* Evaluating performance (loss, accuracy, confusion matrix)
* Saving the model for later use
* Building a user-friendly GUI with **Tkinter** for uploading and classifying new images in Arabic

---

## 📁 Dataset Structure

The dataset should be structured as follows:

```
dataset/
├── Cats/
│   ├── image1.jpg
│   └── image2.jpg
├── Dogs/
│   ├── image1.jpg
│   └── image2.jpg
...
```

Each subfolder represents a separate class.

---

## 🔧 Dependencies

* Python 3.7+
* TensorFlow
* NumPy
* OpenCV
* Matplotlib
* Pillow
* scikit-learn
* Seaborn
* Tkinter (usually comes with Python)

To install required packages:

```bash
pip install tensorflow numpy opencv-python matplotlib pillow scikit-learn seaborn
```

---

## 🚀 Model Architecture

```python
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
```

### Compile and Train:

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

## 📊 Evaluation

* Accuracy and Loss curves
* Confusion matrix
* Classification report

![Loss Plot](Images/Loss%20Plot.png)
![Accuracy Plot](Images/Accuracy%20Plot.png)
![Confusion Matrix](Images/Confusion%20Matrix.png)

---

## 🖼️ Tkinter GUI

* Upload an image
* View prediction result in Arabic
* Display uploaded image

![GUI Dog Screenshot](images/DOG.png)
![GUI Cat Screenshot](images/CAT.png)

---

## 💾 Saving and Loading the Model

```python
model.save("image_classifier_model.keras")
...
model = load_model("image_classifier_model.keras")
```

---

## 🇸🇦 Arabic Support

The GUI is localized in Arabic for accessibility and ease of use.

---

## 📤 How to Run

```bash
python main.py
```

Make sure your current directory contains:

* The dataset folder
* The saved model (or script to train it)
* The script containing GUI logic

---

## 🙋 Future Work

* Add support for more image formats
* Improve GUI design
* Implement model updates with new training data

---

## 📌 Author

\[Mohamed Bakr Mohamed]
Email: [mohamedbakrawais@gmail.com](mohamedbakrawais@gmail.com)

---

