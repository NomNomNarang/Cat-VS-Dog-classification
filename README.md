# 🐱🐶 **Cats vs Dogs Image Classifier**

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg" height="120"/>
  &nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/RedDog.jpg" height="120"/>
</p>

<p align="center">
  <b>A Deep Learning CNN built with TensorFlow & Keras.</b><br>
  Classifies images as <b>Cat</b> 🐱 or <b>Dog</b> 🐶 with high accuracy.
</p>

---

## 🎯 **Project Status**

<p align="center">
  <img src="https://img.shields.io/badge/Project-Complete-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-CNN-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=TensorFlow"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Working-yellow?style=for-the-badge&logo=googlecolab"/>
</p>

---

## 🧠 **Overview**

This project builds a **Convolutional Neural Network (CNN)** to classify images of **cats and dogs**.
Implemented end-to-end in **Google Colab**, the model handles:

✔ Dataset loading from Google Drive
✔ Preprocessing & normalization
✔ CNN model building
✔ Training & validation
✔ Confusion matrix evaluation
✔ Predicting real images (user uploaded)

---

## 📂 **Dataset Structure**

```
CatsVsDog/
│
├── train/
│   ├── Cat/
│   └── Dog/
│
└── test/
    ├── Cat/
    └── Dog/
```

Images are automatically labeled based on folder names.

---

## 🧹 **Data Preprocessing**

### 🔹 Load Dataset

```python
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_path,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)
```

### 🔹 Normalize

```python
def process(image, label):
    return tf.cast(image / 255., tf.float32), label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
```

---

## 🧠 **Model Architecture**

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:800/1*U7ZdfZI5LvqfKmQYzfoalA.png" width="550"/>
</p>

### 🏗 CNN Model

```python
model = keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### ⚙ Compile

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 📈 **Training**

```python
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

Outputs:

* Training accuracy
* Validation accuracy
* Loss curves

---

## 📊 **Confusion Matrix**

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212550021-3483c1e9-ecde-4e6a-a58c-6da52e7ad5f5.png" width="450"/>
</p>

Generated using:

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

---

## 🔮 **Predict on Uploaded Image**

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png" height="140"/>
</p>

### Steps:

```python
img = tf.keras.utils.load_img(img_path, target_size=(256,256))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)[0][0]
```

### Prediction Logic:

```python
if prediction > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
```

---

## 🚀 **Future Enhancements**

* Add data augmentation
* Use EfficientNet / MobileNet (Transfer Learning)
* Add learning rate scheduler
* Save and deploy model using Streamlit or Flask
* Build a web UI for uploading images

---

## 🏁 **Conclusion**

This project demonstrates how to:

✔ Build a CNN from scratch
✔ Handle image preprocessing
✔ Train on a real dataset
✔ Evaluate with a confusion matrix
✔ Make predictions on normal uploaded images

A great beginner-friendly introduction to deep learning and computer vision!

---

## 💙 **Author**

Made by **Namita Narang**
If you want a **PPT, project report, or deployment code**, just let me know!

