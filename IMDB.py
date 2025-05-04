# Import required libraries
import pandas as pd                                  # Pandas for data manipulation and loading CSVs
import numpy as np                                   # NumPy for numerical operations
import tensorflow as tf                              # TensorFlow for building and training models
from sklearn.model_selection import train_test_split # Used to split data into training and testing sets
import matplotlib.pyplot as plt                      # Used to visualize training progress
import warnings                                      # Used to manage warning messages
warnings.filterwarnings('ignore')                    # Ignore warnings in output

# Load the IMDB dataset from CSV
data = pd.read_csv('imdb_dataset.csv')               # Load dataset from CSV file
print(f"Original shape: {data.shape}")               # Print initial shape of the dataset

# Preprocessing (Optional)
print(data.isnull().sum())                           # Count and print null values per column
print(data.duplicated().sum())                       # Print total number of duplicate rows
# Drop rows with missing values in 'review' or 'sentiment', and remove duplicates based on 'review'
data = data.dropna(subset=['review', 'sentiment']).drop_duplicates(subset=['review'])
print(f"Shape after removing Null & duplicates: {data.shape}")  # Print shape after cleaning

# Encode labels ('positive' -> 1, 'negative' -> 0)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})  # Map string labels to integers
data['sentiment'] = (data['sentiment'] == 'positive').astype(int)          # Ensures only 1s and 0s (redundant but safe)
print(data['sentiment'].value_counts())                      # Print number of positive and negative reviews

# Separate features and target
X = data['review'].values                       # Feature: reviews (text)
y = data['sentiment'].values                    # Target: sentiment (0 or 1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)       # random_state ensures reproducibility
print(f"Training set shape: {X_train.shape}Test set shape: {X_test.shape}")  # Print split shapes

# Text vectorization
max_words = 10000                                           # Max number of unique words to consider
max_len = 200                                               # Max sequence length (padding/truncation)
vectorizer = tf.keras.layers.TextVectorization(             # Create text vectorization layer
    max_tokens=max_words,                                   # Limit vocabulary size
    output_mode='int',                                      # Convert text to integers
    output_sequence_length=max_len                          # Fix sequence length to 200 tokens
)
vectorizer.adapt(X_train)                                   # Learn the vocabulary from training text

# Build the neural network model
model = tf.keras.Sequential([                               # Create a sequential model
    vectorizer,                                             # First layer: text vectorization
    tf.keras.layers.Embedding(max_words, 16, input_length=max_len),  # Word embedding (dense vector of size 16)
    tf.keras.layers.GlobalAveragePooling1D(),              # Pooling layer that averages over word embeddings
    tf.keras.layers.Dense(16, activation='relu'),          # Hidden dense layer with ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')         # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',                      # Adam optimizer for efficient gradient descent
    loss='binary_crossentropy',           # Binary classification loss function
    metrics=['accuracy']                  # Track accuracy during training and evaluation
)

# Train the model
history = model.fit(
    X_train, y_train,                     # Input features and labels
    epochs=10,                            # Train for 10 full passes over the training data
    batch_size=32,                        # Train in mini-batches of 32 samples
    validation_split=0.2,                 # Use 20% of training data for validation
    verbose=1                             # Display training progress
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # Evaluate on test set (no output)
print(f'\nTest Loss: {test_loss:.4f}')                           # Print test loss (rounded to 4 decimal places)
print(f'Test Accuracy: {test_acc:.2%}')                          # Print test accuracy as percentage

# Plot results
plt.figure(figsize=(12, 4))                                      # Set figure size for plots

# Plot training and validation loss
plt.subplot(1, 2, 1)                                             # First subplot
plt.plot(history.history['loss'], label='Training Loss')        # Plot training loss over epochs
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss over epochs
plt.title('Model Loss')                                         # Title for loss graph
plt.xlabel('Epoch')                                             # X-axis label
plt.ylabel('Loss')                                              # Y-axis label
plt.legend()                                                    # Add legend to distinguish lines

# Plot training and validation accuracy
plt.subplot(1, 2, 2)                                             # Second subplot
plt.plot(history.history['accuracy'], label='Training Accuracy')       # Training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Validation accuracy
plt.title('Model Accuracy')                                     # Title for accuracy graph
plt.xlabel('Epoch')                                             # X-axis label
plt.ylabel('Accuracy')                                          # Y-axis label
plt.legend()                                                    # Add legend



"""
### **Key Terms Related to the Code**  

---

### **1. Data Handling & Preprocessing**
- **Pandas (`pd`)**  
  - Used for loading CSV data (`read_csv`), checking for missing values (`isnull()`), and removing duplicates (`drop_duplicates`).  
- **Label Encoding**  
  - Converts text labels (`'positive'`, `'negative'`) to binary integers (`1`, `0`).  
- **Train-Test Split (`train_test_split`)**  
  - Splits data into 80% training and 20% testing sets (`test_size=0.2`).  

---

### **2. Text Vectorization**
- **TextVectorization Layer**  
  - Converts raw text to integer sequences:  
    - `max_tokens=10000`: Limits vocabulary to 10,000 most frequent words.  
    - `output_sequence_length=200`: Pads/truncates sequences to 200 tokens.  
- **Adapting the Vectorizer**  
  - `vectorizer.adapt(X_train)`: Learns vocabulary from training data only (prevents data leakage).  

---

### **3. Neural Network Architecture**
- **Embedding Layer**  
  - Maps integer tokens to dense vectors (16 dimensions) for semantic representation.  
- **GlobalAveragePooling1D**  
  - Reduces sequence of word embeddings to a single vector by averaging.  
- **Dense Layers**  
  - `ReLU` activation for hidden layer (16 units).  
  - `Sigmoid` activation for output layer (binary classification).  

---

### **4. Model Training & Evaluation**
- **Compilation**  
  - **Optimizer**: `adam` (adaptive learning rate).  
  - **Loss**: `binary_crossentropy` (measures binary classification error).  
  - **Metric**: `accuracy` (fraction of correct predictions).  
- **Training (`model.fit`)**  
  - `epochs=10`: 10 passes over the training data.  
  - `validation_split=0.2`: 20% of training data used for validation.  
- **Evaluation**  
  - Reports test loss and accuracy (e.g., `Test Accuracy: 85.00%`).  

---

### **5. Visualization**
- **Training Curves**  
  - **Loss Plot**: Tracks training/validation loss to detect overfitting.  
  - **Accuracy Plot**: Shows model performance improvement over epochs.  

---

### **6. Key Concepts**
- **Binary Text Classification**  
  - Predicts sentiment (positive/negative) from movie reviews.  
- **Overfitting Prevention**  
  - Validation monitoring and fixed sequence length (`max_len=200`).  
- **Data Pipeline**  
  1. **Load Data** → 2. **Clean** → 3. **Encode** → 4. **Vectorize** → 5. **Train** → 6. **Evaluate**.  

---

### **Why This Code Matters**
- **End-to-End NLP Example**: From raw text to trained model.  
- **Best Practices**: Text preprocessing, embedding, and validation.  
- **Scalable Template**: Adaptable to other binary text classification tasks (e.g., spam detection).  

### **Execution Flow**
1. **Data Preparation**: Load, clean, and split text/labels.  
2. **Text Processing**: Convert words to integer sequences.  
3. **Model Building**: Define and compile neural network.  
4. **Training**: Fit model while tracking validation metrics.  
5. **Evaluation**: Test performance and visualize results.  

This code is a **foundational template** for sentiment analysis using TensorFlow/Keras.

"""