import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train.reshape(-1, 784)

# Generator
generator = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(100,)),
    layers.Dense(784, activation="tanh")
])

# Generate random noise
noise = tf.random.normal([1, 100])

# Generate synthetic image
image = generator(noise)
image = tf.reshape(image, (28, 28))

# Display image
plt.imshow(image, cmap="gray")
plt.title("Synthetic Image")
plt.axis("off")
plt.show()

print("Synthetic image generated successfully")
