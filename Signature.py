import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load Dataset
data_dir = "dataset"
img_size = 128

data = []
labels = []

for person_name in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person_name)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(person_name)

label_names = list(set(labels))
label_map = {name: idx for idx, name in enumerate(label_names)}
labels = [label_map[name] for name in labels]

data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
labels = to_categorical(np.array(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
model.fit(data_gen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

model.save("signature_recognition_model.h5")

def predict_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    img = cv2.resize(img, (img_size, img_size))
    img = img.reshape(1, img_size, img_size, 1) / 255.0
    prediction = model.predict(img)
    predicted_label = label_names[np.argmax(prediction)]
    return predicted_label

new_signature = ""
try:
    owner = predict_signature(new_signature)
    print(f"Tanda tangan ini milik: {owner}")
except ValueError as e:
    print(e)
