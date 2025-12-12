
# <p align="center"><img src="https://readme-typing-svg.herokuapp.com?size=32&duration=3500&color=F776D6&center=true&vCenter=true&width=900&lines=Cats+vs+Dogs+Image+Classifier+%F0%9F%90%B1%E2%9D%A4%EF%B8%8F%F0%9F%90%B6;Convolutional+Neural+Network+Using+TensorFlow+%F0%9F%A4%96;Deep+Learning+Project+Showcase+%F0%9F%8E%89" /></p>

---

<p align="center">
  <img src="https://media.giphy.com/media/yFQ0ywscgobJK/giphy.gif" width="200" />
  &nbsp;&nbsp;
  <img src="https://media.giphy.com/media/WXB88TeARFVvi/giphy.gif" width="200" />
</p>

<p align="center">
  <b>A Deep Learning CNN that classifies images as Cat 🐱 or Dog 🐶</b><br>
  Built using TensorFlow, Keras, and Google Colab.
</p>

---

## <p align="center">✨ **Project Badges**</p>

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-COMPLETE-brightgreen?style=for-the-badge&logo=vercel&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Active-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-CNN-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-High-00C853?style=for-the-badge&logo=databricks&logoColor=white"/>
</p>

---

# 🧠 **Overview**

This project builds a **Convolutional Neural Network (CNN)** to classify whether an input image contains a **cat** or a **dog**.
It covers an end-to-end machine learning workflow:

* 📥 Data loading & preprocessing
* 🧹 Normalization
* 🏗 CNN model creation
* 🎯 Training & validation
* 📊 Confusion matrix
* 🔮 Prediction on uploaded images

---

# 🌈 **Aesthetic Animated Section Divider**

<p align="center">
  <img src="https://i.imgur.com/EB4s1C8.gif" width="700"/>
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

# 🚀 **Model Architecture (CNN)**

<p align="center">
<img src="https://media.giphy.com/media/QTfX9Ejfra3ZmNxh6B/giphy.gif" width="300">
</p>

### **Layers Used**

* Convolution (Conv2D)
* Batch Normalization
* MaxPooling
* Dense Layers
* Dropout
* Sigmoid Output

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

# ⚙️ **Training Results**

<p align="center">
  <img src="https://media.giphy.com/media/VgE1Q5o0GyrY0/giphy.gif" width="400">
</p>

The model is trained for 10 epochs with validation monitoring.

---

# 📊 **Confusion Matrix**

<p align="center">
<img src="https://i.imgur.com/ZQJH3LO.gif" width="450">
</p>

Generated using:

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

---

# 🔮 **Prediction on Uploaded Images**

Upload any image (cat/dog) and the model predicts using:

```python
img = tf.keras.utils.load_img(img_path, target_size=(256,256))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, 0)
prediction = model.predict(img_array)[0][0]
```

---

# ✨ **Final Output Example**

<p align="center">
  <img src="https://media.giphy.com/media/13borq7Zo2kulO/giphy.gif" width="250">
</p>

```
Prediction: 🐱 Cat (0)  or  🐶 Dog (1)
```

---

# 🌟 **Future Enhancements**

* Use **EfficientNet / MobileNet** (Transfer Learning)
* Add **data augmentation**
* Deploy model using **Streamlit** or **Flask**
* Add **Grad-CAM visualization**

---

# 💖 **Credits**

Made with ❤️ by **Namita Narang**
If you want a **PPT**, **project report**, or **aesthetic diagrams**, I can generate everything!

---
