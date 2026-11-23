import os

from prepare_data import (
    train_dataset,
    validation_dataset,
    class_names
)

from model import model

# Create models folder
os.makedirs("models", exist_ok=True)

# Training settings
EPOCHS = 5

# Train the CNN
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# Save trained model
model.save("models/plant_disease_model.keras")

print("Model training completed!")
print("Model saved successfully!")

