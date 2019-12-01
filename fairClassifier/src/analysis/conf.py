import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
      if normalize:
        title = 'Normalized confusion matrix'
      else:
        title = 'Confusion matrix, without normalization'
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['Low Risk','High Risk'], yticklabels=['Low Risk','High Risk'],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    print('at end')
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plotMatrix(y_test, y_pred, class_names, normalize):
  np.set_printoptions(precision=2)
  # Plot non-normalized confusion matrix
  y_test = y_test.values.tolist()
  ytest = []
  for i in y_test:
    ytest.append(i[0])
  ypredh = y_pred.tolist()
  ypred = []
  for val in ypredh:
    ypred.append(val > 0.5)

  #plot_confusion_matrix(ytest, ypred, classes=class_names, title='Confusion matrix, without normalization')
  plot_confusion_matrix(ytest, ypred, classes=class_names, normalize=True,title='Normalized confusion matrix')
  plt.show()
