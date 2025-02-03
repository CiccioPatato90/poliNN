from db import Database
import utils as ut
import evaluate
from constants import *
from preprocess.split import Data
from tensorflow.python.keras.models import load_model
import numpy as np

# Load pre-trained model
MODEL_PATH = 'trained_model.h5'  # Set the correct path
model = load_model(MODEL_PATH)

# Initialize database
db = Database('res/records.db')

data_instance = Data()

# Extract training and test data from database
X_train, y_train, X_test, y_test = data_instance.extract_train_test_db(db)

# Convert to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Debug print for verification
ut.print_debug(X_train, info="X_train", num_elements=2)
ut.print_debug(X_test, info="X_test", num_elements=2)
ut.print_debug(y_train, info="y_train", num_elements=2)
ut.print_debug(y_test, info="y_test", num_elements=2)

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
print(f'Test Loss: {loss}')

# Predict using the trained model
pred = model.predict(X_train)  # Query model with all training data
ut.print_debug(pred, info="pred_one_hot", num_elements=1)

pred = np.argmax(pred, axis=1) 
print("800th argmax ", pred[800] if len(pred) > 800 else "Index out of range")

ut.print_debug(pred, info="pred", num_elements=1)

# Extract false positives and false negatives
n_false_positives = np.zeros(4)
n_false_negatives = np.zeros(4)

y_compare = np.argmax(y_train, axis=1)  # Using training labels
for true_label, pred_label in zip(y_compare, pred):
    if true_label != pred_label:
        n_false_positives[pred_label] += 1
        n_false_negatives[true_label] += 1

# Print false positive/negative statistics
for i in range(4):
    print(f"False Positives for class {i} : {n_false_positives[i]}")
    print(f"False Negatives for class {i} : {n_false_negatives[i]}")

# Compute evaluation metrics
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

np.set_printoptions(precision=4, suppress=True)

# Compute confusion matrix
cm = metrics.confusion_matrix(y_compare, pred, normalize='all')
print("Confusion Matrix:")
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)