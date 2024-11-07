from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import RocCurveDisplay

def sens_spec_at_diff_th(y, y_pred):    
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    plt.plot(thresholds[1:], tpr[1:], label='sens')
    plt.plot(thresholds[1:], 1-fpr[1:], label='spec')
    plt.axis("square")
    plt.grid(which='major', linewidth=1.2)
    plt.grid(which='minor', linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    return plt

def plot_auc(y,y_pred):
    RocCurveDisplay.from_predictions(y, y_pred)
    plt.grid(which='major', linewidth=1.2)
    plt.grid(which='minor', linewidth=0.3)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.minorticks_on()
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    return plt


