# type: ignore
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalise the images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Define the Convolutional Neural Network model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalise the images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Define the Convolutional Neural Network model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3,3),activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

#Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Train the model
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

#Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# Removed the incorrectly indented lines and the duplicate model definition.



import numpy as np
import PIL
from PIL import Image
# Load the image and convert it to grayscale
# image = Image.open(r"C:\Users\aadim\DigitForRecognition.png").convert("L")
image = Image.open(r".\DigitForRecognition.png").convert("L")

# Resize the Image to 28x28 Pixels
image = image.resize((28, 28), Image.Resampling.BICUBIC) # Corrected line: Pass the size as a tuple (width, height) and the resampling filter.

# Convert the image to a NumPy array and normalise the pixel values
image_array = np.array(image) / 255.0
image_array = image_array.reshape(1, 28, 28, 1)

# Make a prediction
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)

# Print the predicted class
print("Predicted class:", predicted_class)