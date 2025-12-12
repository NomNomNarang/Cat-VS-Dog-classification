

# <p align="center"><img src="https://readme-typing-svg.herokuapp.com?size=32&duration=2700&color=FF8CE6&center=true&vCenter=true&width=900&lines=Cats+vs+Dogs+Classifier+%F0%9F%90%B1%E2%AD%90%EF%B8%8F%F0%9F%90%B6;Deep+Learning+CNN+Project+%F0%9F%A4%96;Built+Using+TensorFlow+%26+Keras+⚡" /></p>

---

<p align="center">
  <img src="https://media.giphy.com/media/WXB88TeARFVvi/giphy.gif" width="200"/>
  <img src="https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy.gif" width="200"/>
</p>

<p align="center">
  <b>A minimal yet powerful CNN to classify images as Cat 🐱 or Dog 🐶.</b><br>
  <i>Designed to be aesthetic, clean, and easy to understand.</i>
</p>

---

## <p align="center">✨ Modern Badges</p>

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-COMPLETE-42f57b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Running-f9ab00?style=for-the-badge&logo=googlecolab"/>
  <img src="https://img.shields.io/badge/Model-CNN-blueviolet?style=for-the-badge"/>
</p>



# 🧠 **Overview**

This project builds a **Convolutional Neural Network (CNN)** to classify images into:

* 🐱 Cat
* 🐶 Dog

Everything — from dataset loading to final prediction — is done inside **Google Colab**.

It includes:

✔ Dataset loading
✔ Preprocessing pipeline
✔ CNN model
✔ Model evaluation
✔ Prediction on your own images

---

# 💫 **Dataset Structure**

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

Folder names → automatic labels (0 = Cat, 1 = Dog)

---

# 🧹 **Preprocessing & Data Pipeline**

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

### 🔹 Normalize Images

```python
def process(image, label):
    return tf.cast(image / 255., tf.float32), label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
```

---

# 🧱 **Model Architecture**

<p align="center">
  <img src="https://media.giphy.com/media/QTfX9Ejfra3ZmNxh6B/giphy.gif" width="250">
</p>

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

---

# 🚀 **Training**

```python
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

This prints:

* Training accuracy
* Validation accuracy
* Loss curves

---

# 🔮 **Predict on Uploaded Images**

Upload any **normal image** (cat/dog) → model predicts instantly.

### 🔹 Preprocess & Predict

```python
img = tf.keras.utils.load_img(img_path, target_size=(256,256))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, 0)
prediction = model.predict(img_array)[0][0]
```

### 🔹 Output

```python
if prediction > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
```


<p align="center">
  <img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" width="300">
</p>

---

# 🌟 **Future Enhancements**

* Add data augmentation
* Use EfficientNet/MobileNet for higher accuracy
* Visualize with Grad-CAM
* Deploy with Streamlit

---

# 💖 **Author**

Made with ✨ by **Namita Narang**

