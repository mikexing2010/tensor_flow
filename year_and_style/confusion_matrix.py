"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

print(__doc__)
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    ### TODO: fill this in with print outputs
    y_pred = [ 8,  3,  9,  5,  9,  8,  9,  7, 10,  7,  9, 11,  5,  7, 10,  8,  3, 10, 11,  5,  9,  9,  7,  7,  9,
  3,  9,  8,  8, 10,  9,  9,  9,  9, 10,  5,  4, 11, 11,  7]
    y_test =  [ 5,  3,  9,  5, 10,  6, 10,  4, 10,  8,  8,  8,  5,  7,  9,  5,  3,  9, 10,  5, 10,  7,  8,  6,  7,
  4, 11,  7,  8,  9,  9,  8,  7, 11,  9,  6,  4,  9,  8, 5]

    y_pred =	[8,	    6,  8,	9,	5,	5,	9,	8,	11,	10,	9,	4,	6,	10,	8,	11,	8,	9,	5,	9,	8,	9,	7,	8,	8,	8,	9,	5,	8,	3,	8,	3,	11,	10,	8,	4,	7,	11,	7,	9,	6,	7,	5,	7,	8,	3,	9,	6,	6,	10,	5,	9,	8,	7,	4,	7,	8,	7,	8,	8,	8,	6,	9,	11,	5,	3,	8]
    y_test = 	[11,	9,	5,	8,	8,	5,	10,	10,	6,	11,	7,	4,	5,	7,	5,	11,	8,	6,	5,	6,	8,	11,	6,	8,	9,10,	9,	10,	10,	9,	9,	3,	10,	6,	9,	3,	5,	3,	9,	5,	10,	6,	10,	4,	10,	8,	8,	8,	5,	10,	5,	10,	7,	8,	6,	7,	4,	11,	7,	8,	9,	9,	8,	7,	11,	9,	5]

    preds = "8,\s8,\s7,\s7,\s8,\s5,\s8,\s8,\s7,\s9,\s7,\s2,\s6,\s10,\s8,\s6,\s8,\s6,\s11,\s7,\s6,\s5,\s9,\s9,\s11,\s8,\s11,\s6,\s8,\s11,\s10,\s10,\s7,\s9,\s3,\s8,\s6,\s6,\s4,\s4,\s8,\s9,\s9,\s8,\s10,\s4,\s2,\s8,\s10,\s8,\s8,\s6,\s5,\s7,\s10,\s8,\s4,\s9,\s8,\s6,\s4,\s8,\s9,\s6,\s9,\s4,\s7,\s6,\s7,\s4,\s8,\s4,\s8,\s11,\s11,\s6,\s4,\s11,\s11,\s7"
    labels =	"11,\s9,\s5,\s8,\s8,\s5,\s10,\s10,\s6,\s11,\s7,\s4,\s5,\s7,\s5,\s5,\s8,\s7,\s11,\s8,\s6,\s5,\s6,\s8,\s11,\s6,\s8,\s9,\s10,\s9,\s10,\s10,\s9,\s9,\s3,\s7,\s10,\s6,\s9,\s3,\s5,\s3,\s9,\s5,\s10,\s6,\s10,\s4,\s10,\s8,\s8,\s8,\s5,\s7,\s9,\s5,\s3,\s9,\s10,\s5,\s10,\s7,\s8,\s6,\s7,\s4,\s11,\s7,\s8,\s9,\s9,\s8,\s7,\s11,\s9,\s6,\s4,\s9,\s8,\s5"
    y_pred = map(int,preds.replace("\s"," ").split(", "))   ####sigmoid
    y_test = map(int,labels.replace("\s"," ").split(", "))   ###sigmoid


    y_pred = [6, 10, 9, 9, 5, 5, 10, 9, 6, 9, 11, 3, 6, 10, 5, 11, 8, 6, 10, 7, 10, 5, 5, 9, 11, 7, 5, 7, 10, 9, 8, 11, 7, 6, 3, 11, 9, 5, 8, 3, 6, 11, 9, 10, 9, 7, 6, 5, 11, 10, 9, 10, 5, 10, 10, 11, 3, 5, 3, 5, 4, 5, 10, 5, 6, 3, 10, 6, 7, 11, 9, 4, 8, 9, 11, 5, 4, 11, 11, 5]   ###relu
    y_test = [11, 9, 5, 8, 8, 5, 10, 10, 6, 11, 7, 4, 5, 7, 5, 5, 8, 7, 11, 8, 6, 5, 6, 8, 11, 6, 8, 9, 10, 9, 10, 10, 9, 9, 3, 7, 10, 6, 9, 3, 5, 3, 9, 5, 10, 6, 10, 4, 10, 8, 8, 8, 5, 7, 9, 5, 3, 9, 10, 5, 10, 7, 8, 6, 7, 4, 11, 7, 8, 9, 9, 8, 7, 11, 9, 6, 4, 9, 8, 5]  ##relu

    print y_pred
    print y_test
    class_names = np.arange(13)
    print class_names

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig("./file_nonnorm_confusion_matrix_relu")
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.savefig("./file_norm_confusion_matrix_relu")