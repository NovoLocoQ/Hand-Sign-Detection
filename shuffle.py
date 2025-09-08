import pandas as pd

# Load dataset
df = pd.read_csv("hand_sign_dataset.csv")

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save shuffled dataset
df.to_csv("hand_sign_dataset_shuffled.csv", index=False)

print("Shuffled dataset saved as hand_sign_dataset_shuffled.csv")
