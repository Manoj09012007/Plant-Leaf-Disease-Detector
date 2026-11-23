from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

print("\nDataset loaded successfully!")
print(dataset)

print("\nTraining images:", len(dataset["train"]))
print("Testing images:", len(dataset["test"]))

print("\nAvailable columns:")
print(dataset["train"].column_names)
