# Import necessary libraries
import sys
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Enable GPU acceleration in Google Colab (if not already enabled)
# In Colab: Runtime > Change runtime type > Hardware accelerator > GPU

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_HEIGHT, IMG_WIDTH = 32, 32  # Keep original size
BATCH_SIZE = 64  # Increased batch size
EPOCHS = 20  # Increased number of epochs
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
SUBSET_SIZE = 10000  # Use a subset of the data to reduce memory usage

# Load the CIFAR-10 dataset
(x_train_full, y_train_full), (x_val_full, y_val_full) = cifar10.load_data()

# Select a smaller subset for training and validation
x_train = x_train_full[:SUBSET_SIZE]
y_train = y_train_full[:SUBSET_SIZE]
x_val = x_val_full[:SUBSET_SIZE // 5]
y_val = y_val_full[:SUBSET_SIZE // 5]

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

# Define data augmentation function
def augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT + 4, IMG_WIDTH + 4)
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure pixel values are within [0,1]
    return image, label

# Create tf.data.Dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=SUBSET_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.repeat()  # Repeat indefinitely
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create tf.data.Dataset for validation
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the CNN model (simplified)
from tensorflow.keras.regularizers import l2

model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model with optimizer and learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Steps per epoch
steps_per_epoch = len(x_train) // BATCH_SIZE
validation_steps = len(x_val) // BATCH_SIZE

# Train the model using the tf.data.Dataset
print("Starting model training...")

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Save the trained model
model.save('enhanced_game_state_detector.keras')
print("Model training complete and saved.")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o')
plt.plot(history.history['val_accuracy'], marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(EPOCHS))
plt.legend(['Train', 'Validation'], loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(EPOCHS))
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()
