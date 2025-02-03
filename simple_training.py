import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('res/8_cluster.csv')

# Separate features and target
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Encode categorical variables if necessary
X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

# Define a simpler model
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),  # Input layer
    keras.layers.Dense(32, activation='relu'),  # Fewer neurons in the first hidden layer
    keras.layers.Dense(16, activation='relu'),  # One hidden layer with fewer neurons
    keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer for multi-class classification
])


# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
               metrics=[
                    'accuracy'
                    #keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
              ])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training history with accuracy, loss, categorical accuracy, and top-3 accuracy
plt.figure(figsize=(14, 10))

# Accuracy plot with test accuracy line
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = label_encoder.inverse_transform(np.unique(y_encoded))
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Save the model
model.save('models/mod_reclustered.h5')
print("Model saved to 'models/mod_reclustered.h5'")

# Load the model (optional)
loaded_model = keras.models.load_model('models/mod_reclustered.h5')
print("Model loaded from 'models/mod_reclustered.h5'")

# Evaluate the loaded model (optional)
loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)
print(f'Loaded Model Test Accuracy: {loaded_test_accuracy:.4f}')
