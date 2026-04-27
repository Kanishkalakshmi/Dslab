import tensorflow as tf
import numpy as np
data = tf.keras.preprocessing.image_dataset_from_directory(
    "EBHI-SEG",
    image_size=(224, 224),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 folders = 6 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(data, epochs=15)


# Load and preprocess image
img = tf.keras.utils.load_img("nor.png", target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)

# Get class
class_names = data.class_names
print("Prediction:", class_names[np.argmax(pred)])