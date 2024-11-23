import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# Load the dataset
data = pd.read_csv(os.getcwd() + '/Datasets/Battery Database.csv')
#data.drop('Unnamed: 0', axis=1, inplace=True)

# Separate features and target
X = data.drop(columns=['RUL', 'Cycle_Group', 'Battery_ID'], errors='ignore')
y = data['RUL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Feedforward Neural Network model
def create_ffnn(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(512, activation='linear', input_dim=input_dim),
        Dropout(0.3),  # Dropout for regularization
        Dense(256, activation='linear'),
        Dropout(0.2),
        Dense(128, activation='linear'),
        Dropout(0.2),
        Dense(128, activation='linear'),
        Dropout(0.2),
        Dense(128, activation='linear'),
        Dropout(0.2),
        Dense(128, activation='linear'),
        Dense(1, activation='linear')  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Learning Rate Scheduler: Custom Step Decay
def step_decay(epoch, lr):
    """Reduces the learning rate by a factor of 0.5 every 10 epochs."""
    if epoch > 0 and epoch % 10 == 0:
        return lr * 0.5
    return lr

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
lr_scheduler = LearningRateScheduler(step_decay, verbose=1)  # Apply custom scheduler

# Create the model
model = create_ffnn(input_dim=X_train_scaled.shape[1])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

# Print the metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Explained Variance Score: {explained_var:.2f}")

# Save the model
model.save('feedforward_rul_prediction_model_with_scheduler.h5')
print("Model saved as 'feedforward_rul_prediction_model_with_scheduler.h5'")

# Plot Training and Validation Loss
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

