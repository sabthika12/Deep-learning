import tensorflow as tf
from tensorflow.keras import layers, models

# Load dog and cat dataset
train = tf.keras.utils.image_dataset_from_directory(
    "dogs_cats",
    image_size=(64, 64),
    batch_size=32,
    label_mode="binary"
)

# Build CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train, epochs=3)

print("Dog vs Cat classification completed")
