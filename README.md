
# 🐱🐶 **Cats vs Dogs Image Classifier**

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=32&duration=3500&color=FF7DD1&center=true&vCenter=true&width=900&lines=Cats+vs+Dogs+Classifier+%F0%9F%90%B1+%E2%9D%A4%EF%B8%8F+%F0%9F%90%B6;A+Deep+Learning+CNN+Project;Built+Using+TensorFlow+%26+Keras" />
</p>

---

<p align="center">
  <img src="https://media.giphy.com/media/yFQ0ywscgobJK/giphy.gif" width="180">
  <img src="https://media.giphy.com/media/ICOgUNjpvO0PC/giphy.gif" width="180">
  <img src="https://media.giphy.com/media/11s7Ke7jcNxCHS/giphy.gif" width="180">
</p>

<p align="center">
  <b>A cute yet powerful CNN that classifies images as Cat 🐱 or Dog 🐶.</b><br>
  Built end-to-end in Google Colab.
</p>

---

## <p align="center">✨ **Badges**</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Complete-00C853?style=for-the-badge&logo=checkmarx&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Active-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-CNN-blueviolet?style=for-the-badge"/>
</p>

---

# 🧠 **Overview**

This deep learning project builds a **CNN model** to classify images into two categories:

* 🐱 **Cat**
* 🐶 **Dog**

It includes:

* Data loading
* Preprocessing
* CNN architecture
* Training
* Prediction on uploaded images

---

# 🎨 **Pixel Art Animal Divider**

<p align="center">
  <img src="https://i.imgur.com/qk4d7Yx.png" width="500">
</p>

---

# 📂 **Dataset Structure**

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

# 🧹 **Preprocessing Pipeline**

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

# 🧱 **Model Architecture (CNN)**

<p align="center">
  <img src="https://media.giphy.com/media/13borq7Zo2kulO/giphy.gif" width="160"><br>
  <b>Convolution → BatchNorm → Pool → Dense → Sigmoid</b>
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

This outputs:

* Training accuracy
* Validation accuracy
* Loss curves

---

# 🐾 **Predicting on Uploaded Images**

Upload any **normal real-life image**, and the model will classify it.

### Preprocess & predict:

```python
img = tf.keras.utils.load_img(img_path, target_size=(256,256))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, 0)
prediction = model.predict(img_array)[0][0]
```

### Output:

```python
if prediction > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
```

---

# 🐱💬 Pixel-Art Ending Banner

<p align="center">
  <img src="https://i.imgur.com/eVHu1xE.gif" width="300">
</p>

---

# 🌟 **Future Enhancements**

* Add data augmentation
* Use EfficientNet / MobileNet
* Deploy using Streamlit
* Add Grad-CAM heatmaps

---

# 💖 **Author**

Made with ❤️ by **Namita Narang**

