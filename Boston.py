# Import libraries for data manipulation, machine learning, and visualization
import pandas as pd                        # For data handling using DataFrames
import numpy as np                         # For numerical operations
import tensorflow as tf                    # For building and training neural networks
from sklearn.model_selection import train_test_split  # To split dataset into training and testing sets
from sklearn.preprocessing import StandardScaler      # To normalize features before training
import matplotlib.pyplot as plt            # For plotting training metrics and predictions
import warnings                            # To manage warning messages
warnings.filterwarnings('ignore')          # Ignore any warnings to keep output clean

# Load the Boston Housing dataset from CSV
data = pd.read_csv('Boston.csv')           # Read dataset into a pandas DataFrame
data.shape                                 # Display the number of rows and columns in the dataset

# Preprocessing (Optional)
print(data.isnull().sum())                 # Print number of missing values in each column
data = data.dropna()                       # Remove rows with any missing values

print(data.duplicated().sum())             # Print number of duplicate rows
data = data.drop_duplicates()              # Remove duplicate rows from the dataset

print(f"Shape after removing Null & duplicates: {data.shape}")  # Print new shape of cleaned dataset

print(data.columns)                        # Display column names of the dataset

# Assuming the target column is named 'medv' (adjust if different)
X = data.drop('medv', axis=1).values       # Features: drop the target column 'medv'
y = data['medv'].values                    # Target: the column 'medv' as a NumPy array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                                  # Features and target
    test_size=0.2,                         # 20% of data for testing
    random_state=42                        # Seed to ensure reproducible split
)

# Scale the features
scaler = StandardScaler()                 # Create a StandardScaler instance for normalization
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and scale
X_test_scaled = scaler.transform(X_test)        # Scale test data using same parameters

# Build the neural network model
model = tf.keras.Sequential([                      # Sequential model (layers added in order)
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 64 neurons
    tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer with 32 neurons
    tf.keras.layers.Dense(1)                       # Output layer for regression (1 neuron, linear activation)
])

# Compile the model
model.compile(
    optimizer='adam',       # Optimizer: Adam (adaptive gradient descent)
    loss='mse',             # Loss function: Mean Squared Error (good for regression)
    metrics=['mae']         # Metric to track: Mean Absolute Error (easier to interpret)
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,    # Training features and labels
    epochs=100,                 # Train for 100 iterations through the dataset
    batch_size=32,              # Use mini-batches of 32 samples
    validation_split=0.2,       # 20% of training data used for validation
    verbose=1                   # Show progress bar during training
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)  # Evaluate on unseen test data
print(f"\nTest Mean Absolute Error: ${test_mae:.2f}k")                   # Print the MAE formatted in thousands

# Make predictions
predictions = model.predict(X_test_scaled)  # Predict house prices using the test data

# Plot training history
plt.figure(figsize=(12, 4))  # Create a figure with 2 subplots side by side

# Plot loss (MSE) over epochs
plt.subplot(1, 2, 1)                                  # First subplot
plt.plot(history.history['loss'], label='Training Loss')          # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')    # Plot validation loss
plt.title('Model Loss')                              # Title of plot
plt.xlabel('Epoch')                                  # Label x-axis
plt.ylabel('Loss')                                   # Label y-axis
plt.legend()                                         # Add legend

# Plot MAE over epochs
plt.subplot(1, 2, 2)                                  # Second subplot
plt.plot(history.history['mae'], label='Training MAE')            # Plot training MAE
plt.plot(history.history['val_mae'], label='Validation MAE')      # Plot validation MAE
plt.title('Model MAE')                                # Title of plot
plt.xlabel('Epoch')                                   # Label x-axis
plt.ylabel('MAE')                                     # Label y-axis
plt.legend()                                          # Add legend

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))                            # New figure for actual vs predicted plot
plt.scatter(y_test, predictions, alpha=0.5)           # Scatter plot of actual vs predicted prices
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal reference line
plt.xlabel('Actual Price')                            # Label x-axis
plt.ylabel('Predicted Price')                         # Label y-axis
plt.title('Actual vs Predicted Prices')               # Plot title
plt.tight_layout()                                    # Automatically adjust subplot parameters
plt.show()                                            # Display the plots


"""
### **Key Terms Related to the Code**  

---

### **1. Data Handling & Preprocessing**
- **Pandas (`pd`)**  
  - Library for data manipulation (DataFrames, cleaning, filtering).  
  - Used here to load CSV data (`pd.read_csv`), check for missing values (`isnull()`), and remove duplicates (`drop_duplicates`).  

- **NumPy (`np`)**  
  - Library for numerical operations (arrays, math functions).  
  - Converts DataFrame columns to NumPy arrays (`.values`) for ML models.  

- **Train-Test Split (`train_test_split`)**  
  - Splits data into training (80%) and testing (20%) sets for model evaluation.  
  - `random_state=42` ensures reproducibility.  

- **Feature Scaling (`StandardScaler`)**  
  - Normalizes features to have zero mean and unit variance.  
  - Critical for neural networks to ensure stable training.  

---

### **2. Neural Network (TensorFlow/Keras)**
- **Sequential Model**  
  - A linear stack of layers (input → hidden → output).  
  - Layers:  
    - **Dense (Fully Connected)**: Each neuron connects to all inputs.  
    - **ReLU Activation**: Introduces non-linearity (`max(0, x)`).  
    - **Output Layer**: Single neuron (linear activation) for regression.  

- **Compilation**  
  - **Optimizer**: `adam` (adaptive learning rate for gradient descent).  
  - **Loss Function**: `mse` (Mean Squared Error) for regression tasks.  
  - **Metrics**: `mae` (Mean Absolute Error) for interpretability.  

- **Training (`model.fit`)**  
  - **Epochs**: 100 passes through the dataset.  
  - **Batch Size**: 32 samples per gradient update (mini-batch).  
  - **Validation Split**: 20% of training data used for validation during training.  

---

### **3. Evaluation & Visualization**
- **Model Evaluation (`model.evaluate`)**  
  - Computes test loss (MSE) and MAE on unseen data (`X_test_scaled`).  

- **Predictions (`model.predict`)**  
  - Generates house price predictions for test data.  

- **Matplotlib (`plt`)**  
  - Plots:  
    1. **Training History**: Loss (MSE) and MAE over epochs.  
    2. **Actual vs. Predicted**: Scatter plot with a reference line (perfect predictions).  

---

### **4. Key Concepts**
- **Regression Task**  
  - Predicts continuous values (e.g., house prices).  
  - Uses MSE/MAE (not accuracy, which is for classification).  

- **Overfitting Detection**  
  - Monitored via validation loss (stops improving or starts rising).  

- **Data Leakage Prevention**  
  - Scaling (`StandardScaler`) is fit only on training data (`fit_transform`), then applied to test data (`transform`).  

---

### **Why This Code Matters**
- **End-to-End ML Pipeline**: From data loading → preprocessing → modeling → evaluation.  
- **Neural Network Basics**: Demonstrates layer design, activation functions, and training loops.  
- **Best Practices**: Feature scaling, train-test split, and validation monitoring.  

### **Execution Flow**
1. **Load & Clean Data** → 2. **Split & Scale** → 3. **Build Model** → 4. **Train** → 5. **Evaluate** → 6. **Visualize**.  

This template is widely applicable to regression problems (e.g., stock prices, sensor readings).
"""