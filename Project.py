import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


dataset_path = "dataset/"
class_names = os.listdir(dataset_path)


for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    print(f"Class: {class_name}, Number of images: {len(os.listdir(class_path))}")


for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    sample_image = os.path.join(class_path, os.listdir(class_path)[0])
    img = cv2.imread(sample_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.title(f"Sample Image from Class: {class_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def preprocess_data(dataset_path, class_names):
    images = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


X, y = preprocess_data(dataset_path, class_names)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


model.save("image_classifier_model.keras")


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_val, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report(y_val, y_pred_classes, target_names=class_names))

root = tk.Tk()
root.title("تصنيف الصور بالذكاء الاصطناعي")
root.geometry("400x500")

model = load_model("image_classifier_model.keras")

def upload_and_classify():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    result_label.config(text=f"النتيجة: {class_names[predicted_class]}")

    img_display = Image.open(file_path)
    img_display = img_display.resize((200, 200))
    img_display = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_display)
    image_label.image = img_display

upload_btn = tk.Button(root, text="اختر صورة", command=upload_and_classify, font=("Arial", 14))
upload_btn.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="النتيجة: ", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
