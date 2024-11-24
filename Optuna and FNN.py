## Battery RUL prediction with Feedforward Neural Network
## The battery data is from 14 Li-on batteries were used. Data source: https://www.batteryarchive.org/list.html 
## The code uses optuna to find the best hyper parameters for the given data set
## The Best params are found using trails and the best params are derived within 10 trails
## These best params were provided to the regression object 
## The algorithm plots the overall validation loss as well


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import optuna
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(os.getcwd() + '/Datasets/Battery Database.csv')

# Drop the cycle index column
data.drop('Unnamed: 0', axis=1, inplace=True) 

# Separate features and target
X = data.drop(columns=['RUL'], errors='ignore')
y = data['RUL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Feedforward Neural Network model
def create_ffnn(input_dim, learning_rate=0.001, num_layers=2, num_units=64, dropout_rate=0.3):
    model = Sequential()
    # Adding layers based on hyperparameters
    for _ in range(num_layers):
        model.add(Dense(num_units, activation='linear'))
        model.add(Dropout(dropout_rate))  # Dropout for regularization
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# Define the objective function for Optuna to optimize hyperparameters
def objective(trial):
    # Suggest hyperparameters for optimization
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_layers = trial.suggest_int("num_layers", 1, 5)  # Number of hidden layers
    num_units = trial.suggest_int("num_units", 32, 512)  # Number of units in each hidden layer
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)  # Dropout rate

    # Create and train the model
    model = create_ffnn(input_dim=X_train_scaled.shape[1], learning_rate=learning_rate,
                        num_layers=num_layers, num_units=num_units, dropout_rate=dropout_rate)

    # Train the model
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
    
    # Return the validation loss as the objective value
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Create a study object to start optimization
study = optuna.create_study(direction="minimize")
print(f"Starting optimization with 10 trials")
study.optimize(objective, n_trials=10)
print(f"Finished optimization after 10 trials.")

# Print the best hyperparameters from the optimization
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Create the model with the best hyperparameters
model = create_ffnn(input_dim=X_train_scaled.shape[1], 
                    learning_rate=best_params['learning_rate'], 
                    num_layers=best_params['num_layers'], 
                    num_units=best_params['num_units'], 
                    dropout_rate=best_params['dropout_rate'])

# Train the model with the best hyperparameters
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Explained Variance Score: {explained_var:.2f}")


# Plot the training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
