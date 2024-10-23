#from imports import *
from db import Database
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import utils as ut
import evaluate
from constants import *
import matplotlib.pyplot as plt


db = Database('res/records.db')
data = db.fetch_all()
num_records = len(data)
#db.close_conn()

# Split data into features and labels
dfs = np.split(data, [24], axis=1)

# FORMAT: [0.06  0.097 0.064 0.059 0.059 0.097 0.061 0.057 0.057 0.095 0.058 0.059
# 0.07  0.087 0.054 0.055 0.089 0.067 0.068 0.066 0.09  0.058 0.056 0.087]
X = dfs[0]
# FORMAT: [1.] [1.] [1.] [2.] [3.]
y = dfs[1]

#numpy arrays
#print("FEATURES (NUMPY) :", X[1:2])
#print("LABELS (NUMPY) :", y[1:2])

from sklearn.preprocessing import LabelEncoder
#from keras import utils as ut
import utils as ut

# Assuming 'y' is your LABEL column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # NOT NEEDED
#print(y_encoded[10:13])

#[1, 0, 0, 0] ... [0, 0, 0, 1]
#y_categorical = ut.to_categorical(y)  # One-hot encode
#print("converting to one-hot encoding: ", y_categorical[300:301])

model = Sequential()
model.add(Dense(4, input_dim=len(X[0]), activation=None))
model.add(Dense(4, activation='softmax'))

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
                        verbose=1, mode='auto', restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from preprocess.split import Data
# Initialize the Data object
data_instance = Data()
# Call the init method example
if True:
    data_instance.init_pickle([0,1,2,3])
    #exit()

data_instance.info([0,1,2,3], METADATA_FILE)

data_instance.analyze([0,1,2,3], "res/analysis.txt")

data_instance.info([0,1,2,3], "res/cleaned.txt")

#exit()
data_instance.split(proportion=0.8)

#evaluate.cross_correlation([(2,3), (1, 2), (0,2)])

#exit()

#data_instance.print_results()

train, test = data_instance.extract_train_test()

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Convert to NumPy arrays if necessary
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

if True:
    ut.print_debug(X_train, info="X_train", num_elements=5)
    ut.print_debug(X_test, info="X_test", num_elements=5)
    ut.print_debug(y_train, info="y_train", num_elements=5)
    ut.print_debug(y_test, info="y_test", num_elements=5)

#X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y,random_state=42)
#train, test = train_test_split(data, test_size=0.2, stratify=y,random_state=42)

# Riscrivere la funzione di split del dataset poichè le proporzioni non vengono mantenute
# SHOW CLASS DISTRIBUTION
#ut.print_debug(train, info="train", num_elements=5)
#ut.print_debug(test, info="test", num_elements=5)

evaluate.bar_chart_binary(y, 'pre')
evaluate.bar_chart_one_hot(y_train, 'train')
evaluate.bar_chart_one_hot(y_test, 'test')

model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
print(f'Test Loss: {loss}')

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1) 
ut.print_debug(pred, info="pred", num_elements=15)


from sklearn import metrics
y_compare = np.argmax(y_test,axis=1) 
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))


import numpy as np
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Compute confusion matrix
cm = confusion_matrix(y_compare, pred)
print(cm)
np.set_printoptions(precision=2)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)


