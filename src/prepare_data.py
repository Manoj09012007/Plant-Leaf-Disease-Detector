import tensorflow as tf

DATASET_PATH = "dataset/raw/plantvillage/raw/color"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Get disease categories
class_names = train_dataset.class_names

# Improve loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

print("\nDataset preprocessing completed!")

print("Number of classes:", len(class_names))
print("Disease categories:", class_names)

for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

import json

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Class names saved successfully!")
