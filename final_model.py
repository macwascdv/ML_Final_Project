import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import Pipeline
import pickle
###############################################

def prediction(X_test):
    """
    This function imports model and uses it to predict labels
    ARGS:
        X_test - path to file with X_test
    
    Returns:
    predict - predicted labels in array """

    encoder = lambda x:1 if x == -1 else 0
    decoder = lambda x:-1 if x == 1 else 1

    LR = LogisticRegression(C= 1.0, class_weight= 'balanced', penalty= 'l1', solver= 'liblinear')
    SC = StandardScaler()
    KPCA = KernelPCA(gamma= 0.03, kernel= 'linear')
    pipe = Pipeline(steps=[('scaler',SC),('decomposition',KPCA),('estimator',LR)])

    X_test = pd.read_csv(X_test,header=None, low_memory = False)

    #Loading model and predicting labels
    model = pickle.load(open('finalized_model.sav', 'rb'))
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame([y_pred]).T
    y_pred= y_pred[0].apply(decoder)
    return y_pred
if __name__=="__main__":
    y_pred = prediction('test_data.csv')
    y_pred.to_csv('test_labels.csv', index= False)