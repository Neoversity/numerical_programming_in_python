import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Step 1: Generate a synthetic dataset with moderate noise (same)
X, y = make_regression(n_samples=100, n_features=1, noise=5)
y = y.reshape(-1, 1)


# Step 2: Split the dataset into training and test sets (same)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 3: Define a more complex neural network model (same)
model = Sequential(
    [
        Dense(1024, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(1024, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(512, activation="relu"),
        Dense(1),
    ]
)


# Compile the model (same)
model.compile(
    optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mse"]
)


# Step 4: Train the model for many epochs (too many)
history = model.fit(
    X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=0
)


# Combine history from both training phases (same)
history_combined_loss = history.history["loss"]
history_combined_val_loss = history.history["val_loss"]


# Step 5: Visualize the loss on both the training and test datasets
plt.figure(figsize=(12, 10))  # Adjust for readability


# Plot loss for overfitting
plt.plot(
    range(1, len(history_combined_loss) + 1), history_combined_loss, label="Train Loss"
)
plt.plot(
    range(1, len(history_combined_val_loss) + 1),
    history_combined_val_loss,
    label="Test Loss",
)
plt.title("Overfitting (Too Many Epochs)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
