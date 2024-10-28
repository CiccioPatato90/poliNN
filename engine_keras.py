from db import Database
import utils as ut
import evaluate
from constants import *
from preprocess.split import Data
from classes import Model

from sklearn.model_selection import train_test_split

import numpy as np


db = Database('res/records.db')

if False:
    #create db relation to efficiently fetch one_hot_encoding at db level
    db.create_label_conversion()
    exit()

data_instance = Data()

if False:
    # 1. fetch data divided by cluster from db
    # 2. calculate metadata (mean, std, var ..) + populate total_values
    # 3. cache result
    data_instance.init_pickle([0,1,2,3], db)

    # exiting bc need to do it only once, or upon db data change
    exit()

# analyze cluster 
# generate metadata info (res/metadata.txt)
if False:
    data_instance.info([0,1,2,3], METADATA_FILE)
    data_instance.analyze([0,1,2,3], "res/analysis.txt", clean=True, outliers_threshold=15)
    data_instance.info([0,1,2,3], "res/cleaned.txt", cache_dir=NO_OUTLIERS_CACHE_DIR)

    #pass cache dir to tell it where to save the test and train splits
    data_instance.sklearn_split(db,test_size=0.3, random=False, debug=True, cache_dir=NO_OUTLIERS_CACHE_DIR)
    exit()

evaluate.cross_correlation([(0,3), (0, 1), (0,2)], cache_dir=NO_OUTLIERS_CACHE_DIR)

if True:
    X_train, y_train, X_test, y_test = data_instance.extract_train_test_db(db)
    
    """
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    """

    # Convert to NumPy arrays if necessary
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

if True:
    ut.print_debug(X_train, info="X_train", num_elements=2)
    ut.print_debug(X_test, info="X_test", num_elements=2)
    ut.print_debug(y_train, info="y_train", num_elements=2)
    ut.print_debug(y_test, info="y_test", num_elements=2)


# Riscrivere la funzione di split del dataset poichè le proporzioni non vengono mantenute
# SHOW CLASS DISTRIBUTION
#evaluate.bar_chart_binary(y, 'pre')
evaluate.bar_chart_one_hot(y_train, 'train')
evaluate.bar_chart_one_hot(y_test, 'test')

#model init, need to pass input and output dim from Data()
model = Model(input_dim=len(X_train[0]), output_dim=4)

model.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
print(f'Test Loss: {loss}')

pred = model.model.predict(X_test)
ut.print_debug(pred, info="pred_one_hot", num_elements=1)
print("800th one_hot ", pred[800])
pred = np.argmax(pred,axis=1) 
print("800th argmax ", pred[800])
ut.print_debug(pred, info="pred", num_elements=1)
# Extract false positives and false negatives for each class

n_false_positives = np.zeros(4)
n_false_negatives = np.zeros(4)
# Loop over true and predicted labels
y_compare = np.argmax(y_test,axis=1) 
for true_label, pred_label in zip(y_compare, pred):
    if true_label != pred_label:
        # Increase false positives for the predicted class
        n_false_positives[pred_label] += 1
        # Increase false negatives for the true class
        n_false_negatives[true_label] += 1

# Print the results
for i in range(4):
    print(f"False Positives for class {i} : {n_false_positives[i]}")
    print(f"False Negatives for class {i} : {n_false_negatives[i]}")


from sklearn import metrics

score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))


import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Compute confusion matrix
cm = metrics.confusion_matrix(y_compare, pred, normalize='all')
print(cm)
np.set_printoptions(precision=2)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)


