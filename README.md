# IMAGE-Classification
#Its a CNN model to train and recognize the images 


import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.getcwd()

dataset_dir = '/Users/turjo/PycharmProjects/PythonProject3/cats and dogs'


# Create training and validation datasets

batch_size = 32  # defining how many images will be loaded at once in each batch
img_height = 150  # dimensions of each images
img_width = 150 # dimensions of each images

base_dir = '/Users/turjo/PycharmProjects/PythonProject3/data'

# Define directories for train, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


for subset in [train_dir, val_dir, test_dir]:
    os.makedirs(subset, exist_ok=True)


for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        # Get all images in the class
        all_images = os.listdir(class_path)
        random.shuffle(all_images)

        # split the training validation adn test datasets
        num_images = len(all_images)
        train_split = int(0.6 * num_images)  # 60% for training
        val_split = int(0.8 * num_images)  # 20% for validation, leaving 20% for testing

        train_images = all_images[:train_split]
        val_images = all_images[train_split:val_split]
        test_images = all_images[val_split:]

        
        for image_set, subset_dir in zip([train_images, val_images, test_images], [train_dir, val_dir, test_dir]):
            class_subset_dir = os.path.join(subset_dir, class_name)
            os.makedirs(class_subset_dir, exist_ok=True)
            for image in image_set:
                shutil.copy(os.path.join(class_path, image), os.path.join(class_subset_dir, image))


# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,

)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,

)

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# fit the data into the model
epochs = 70
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,  # Number of epochs
    verbose=2
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")


# Plot the results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(epochs), acc, label='Training Accuracy')
plt.plot(np.arange(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy (With Augmentation)')

plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), loss, label='Training Loss')
plt.plot(np.arange(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (With Augmentation)')
plt.show()

# ------------------------------------------------------------
# Make a Prediction on a New Image
# -----------------------------------------------------------

from tensorflow.keras.preprocessing import image

img_path = '/Users/turjo/PycharmProjects/PythonProject3/flower_photos/daisy/153210866_03cc9f2f36.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)  # Convert the image to a numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (model expects a batch)
img_array = img_array / 255.0  # Rescale the image to [0, 1] as done during training

# Make the prediction
predictions = model.predict(img_array)

# Get the predicted class (index with the highest probability)
predicted_class_index = np.argmax(predictions, axis=-1)
class_names = train_generator.class_indices  # Access class names from the training generator
class_names = {v: k for k, v in class_names.items()}  # Reverse the dictionary for easy lookup

# Print the predicted class label
predicted_class_name = class_names[predicted_class_index[0]]
print(f"The predicted class for the image is: {predicted_class_name}")

# Create a figure with 1 row and 2 columns (side-by-side)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Show the original image in the first subplot
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis('off')  # Hide axis for better image presentation

# Show the predicted class image in the second subplot
# Optionally, you can display the predicted class on a blank canvas or put it on the image
ax[1].imshow(img)
ax[1].set_title(f"Prediction: {predicted_class_name}")
ax[1].axis('off')  # Hide axis for better image presentation

# Show the combined plot
plt.show()




