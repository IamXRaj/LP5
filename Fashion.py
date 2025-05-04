# Import required libraries
import pandas as pd                          # Used for data manipulation and analysis
import numpy as np                           # Provides support for arrays and numerical operations
import tensorflow as tf                      # TensorFlow library for deep learning
from tensorflow import keras                 # Keras API from TensorFlow for building neural networks
import matplotlib.pyplot as plt              # Used for plotting graphs and visualizations
import warnings                              # Used to handle warning messages
warnings.filterwarnings('ignore')            # Ignore warnings in the output

# Load train and test data from CSV files
train_data = pd.read_csv('fashion-mnist_train.csv')    # Load training dataset
test_data = pd.read_csv('fashion-mnist_test.csv')      # Load testing dataset
print(f"Original train shape: {train_data.shape}, test shape: {test_data.shape}")  # Print shape of datasets

# Preprocessing (Optional)
# Identify null values
print(train_data.isnull().sum())             # Print count of missing values per column in train dataset
print(test_data.isnull().sum())              # Print count of missing values per column in test dataset

train_data = train_data.dropna()             # Drop rows with missing values in training dataset
test_data = test_data.dropna()               # Drop rows with missing values in test dataset

print(train_data.duplicated().sum())         # Print number of duplicate rows in train dataset
print(test_data.duplicated().sum())          # Print number of duplicate rows in test dataset

train_data = train_data.drop_duplicates()    # Drop duplicate rows in training dataset
test_data = test_data.drop_duplicates()      # Drop duplicate rows in test dataset

print("shape after removing Null & duplicates")        # Print after-cleaning info
print(f"Train : {train_data.shape}")                   # Print cleaned train shape
print(f"Test : {test_data.shape}")                     # Print cleaned test shape

# Extract features and labels
x_train = train_data.drop('label', axis=1).values       # Features (images) from training data
y_train = train_data['label'].values                    # Labels (targets) from training data
x_test = test_data.drop('label', axis=1).values         # Features (images) from test data
y_test = test_data['label'].values                      # Labels (targets) from test data

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0             # Convert pixel values to float and normalize (0-1)
x_test = x_test.astype('float32') / 255.0               # Normalize test set similarly

# Reshape for CNN input (28x28 images with 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)                 # Reshape train data to 4D tensor for CNN
x_test = x_test.reshape(-1, 28, 28, 1)                   # Reshape test data similarly
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")  # Print new shape

print(train_data['label'].nunique())                    # Print number of unique classes (should be 10)

# Class names for visualization (Optional)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',    # Human-readable class names
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Visualize sample images
plt.figure(figsize=(10, 10))                            # Set figure size for plotting
for i in range(25):                                     # Plot first 25 images
    plt.subplot(5, 5, i+1)                              # Create 5x5 grid
    plt.xticks([])                                      # Remove x-axis ticks
    plt.yticks([])                                      # Remove y-axis ticks
    plt.grid(False)                                     # Remove grid lines
    plt.imshow(x_train[i], cmap=plt.cm.binary)          # Display image in grayscale
    plt.xlabel(class_names[y_train[i]])                 # Label each image with its class name
plt.savefig('sample_images.png')                        # Save the plot as an image file

# Build the CNN model
model = keras.Sequential([                              # Create a Sequential model
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Conv layer with 32 filters, 3x3 kernel
    keras.layers.MaxPooling2D((2,2)),                   # Max pooling layer with 2x2 pool size
    keras.layers.Dropout(0.25),                         # Dropout layer for regularization (25% dropout)
    keras.layers.Conv2D(64, (3,3), activation='relu'),  # Second conv layer with 64 filters
    keras.layers.MaxPooling2D((2,2)),                   # Second pooling layer
    keras.layers.Dropout(0.25),                         # Dropout layer
    keras.layers.Conv2D(128, (3,3), activation='relu'), # Third conv layer with 128 filters
    keras.layers.Flatten(),                             # Flatten output into 1D vector for Dense layers
    keras.layers.Dense(128, activation='relu'),         # Dense layer with 128 units and ReLU activation
    keras.layers.Dropout(0.25),                         # Dropout layer
    keras.layers.Dense(10, activation='softmax')        # Output layer with 10 classes and softmax for classification
])

# Compile the model
model.compile(optimizer='adam',                         # Adam optimizer for adaptive learning rate
              loss='sparse_categorical_crossentropy',   # Loss function for multi-class classification
              metrics=['accuracy'])                     # Metric: classification accuracy

# Train the model
history = model.fit(x_train, y_train,                   # Training data
                    epochs=10,                          # Train for 10 epochs
                    batch_size=32,                      # Use batches of 32 samples
                    validation_data=(x_test, y_test),   # Use test data for validation during training
                    verbose=1)                          # Show training progress

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)  # Get test loss and accuracy
print(f'\nTest Loss: {test_loss:.4f}')                            # Print formatted test loss
print(f'Test Accuracy: {test_acc:.2%}')                          # Print accuracy as percentage

# Plot training and validation accuracy and loss
plt.figure(figsize=(10, 5))                             # Create a new figure for plots

plt.subplot(1, 2, 1)                                    # First subplot for accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')       # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Plot validation accuracy
plt.xlabel('Epochs')                                   # Label x-axis
plt.ylabel('Accuracy')                                 # Label y-axis
plt.title('Training and Validation Accuracy')          # Title of the plot
plt.legend()                                           # Add legend

plt.subplot(1, 2, 2)                                    # Second subplot for loss
plt.plot(history.history['loss'], label='Training Loss')            # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')      # Plot validation loss
plt.xlabel('Epochs')                                   # Label x-axis
plt.ylabel('Loss')                                     # Label y-axis
plt.title('Training and Validation Loss')              # Title of the plot
plt.legend()                                           # Add legend


"""
### **Key Terms Related to the Code**  

---

### **1. Data Handling & Preprocessing**
- **Fashion MNIST Dataset**  
  - A dataset of 70,000 grayscale images (28×28 pixels) across 10 fashion categories.  
  - Common benchmark for image classification tasks.  

- **Pandas (`pd`)**  
  - Used for loading CSV data (`read_csv`), checking for missing values (`isnull()`), and removing duplicates (`drop_duplicates`).  

- **NumPy (`np`)**  
  - Converts DataFrame columns to NumPy arrays (`.values`).  
  - Handles array operations (reshaping, normalization).  

- **Normalization**  
  - Pixel values scaled to `[0, 1]` (`/ 255.0`) for better neural network training.  

- **Reshaping for CNN**  
  - Images reshaped to `(28, 28, 1)` (height, width, channels) for convolutional layers.  

---

### **2. Convolutional Neural Network (CNN)**
- **Keras Sequential Model**  
  - Linear stack of layers for building CNNs.  

- **Layers**  
  - **Conv2D**:  
    - Learns spatial features using filters (e.g., `32` filters of size `3×3`).  
    - `ReLU` activation introduces non-linearity.  
  - **MaxPooling2D**:  
    - Reduces spatial dimensions (e.g., `2×2` pooling).  
  - **Dropout**:  
    - Regularization technique to prevent overfitting (e.g., `0.25` dropout rate).  
  - **Flatten**:  
    - Converts 3D feature maps to 1D for dense layers.  
  - **Dense**:  
    - Fully connected layers (`128` neurons with `ReLU`, `10` neurons with `softmax` for classification).  

- **Compilation**  
  - **Optimizer**: `adam` (adaptive learning rate).  
  - **Loss Function**: `sparse_categorical_crossentropy` (multi-class classification).  
  - **Metric**: `accuracy` (fraction of correctly classified images).  

---

### **3. Training & Evaluation**
- **Training (`model.fit`)**  
  - **Epochs**: 10 full passes through the dataset.  
  - **Batch Size**: 32 samples per gradient update.  
  - **Validation Data**: Test set used to monitor performance during training.  

- **Evaluation (`model.evaluate`)**  
  - Computes test loss and accuracy on unseen data.  

- **Visualization**  
  - **Sample Images**: Displays 25 training images with labels.  
  - **Training Curves**: Plots accuracy/loss over epochs to detect overfitting.  

---

### **4. Key Concepts**
- **Image Classification**  
  - Task: Assign one of 10 fashion categories to each image.  
  - Output: Probability distribution via `softmax` activation.  

- **Overfitting Prevention**  
  - Techniques: Dropout, validation monitoring.  

- **Data Pipeline**  
  1. **Load Data** → 2. **Clean** → 3. **Normalize** → 4. **Reshape** → 5. **Train** → 6. **Evaluate**.  

---

### **Why This Code Matters**
- **End-to-End CNN Example**: From data loading to model evaluation.  
- **Best Practices**: Normalization, dropout, and visualization.  
- **Scalable Template**: Adaptable to other image datasets (e.g., CIFAR-10).  

### **Execution Flow**
1. **Data Preparation**: Load, clean, normalize, and reshape images.  
2. **Model Building**: Define CNN architecture.  
3. **Training**: Fit model while monitoring validation metrics.  
4. **Evaluation**: Test accuracy/loss and visualize results.  

This code is a **foundational template** for image classification tasks using CNNs in Keras.
"""
