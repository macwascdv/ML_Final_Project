import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, TomekLinks, \
                                    EditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from typing import Dict, List


def create_params_grid(scalers: Dict, decompositions: Dict, estimators: Dict) -> List:
    
    params_grid_without_samplings=[]
    
    for _,i in scalers.items():
        for _,j in decompositions.items():
            for _,k in estimators.items():
                params_grid_without_samplings.append({**i,**j,**k})
    return params_grid_without_samplings


decompositions = { 'pca': {'decomposition': [PCA()],
                            'decomposition__n_components': [2,5,10,100,.95,.7], #sorted(set(np.logspace(0.4, 2, 10, dtype='int', endpoint=False))),
                            'decomposition__whiten': [False, True],
                            'decomposition__svd_solver': ['auto', 'full', 'arpack', 'randomized']},
                   'kpca': {'decomposition': [KernelPCA()],
                            'decomposition__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                            'decomposition__gamma': np.linspace(0.03, 0.05, 5)}
                 }

scalers = { 'StandardScaler': {'scaler': [StandardScaler()]},
            'Normalizer': {'scaler': [Normalizer()]},
            'MinMaxScaler': {'scaler': [MinMaxScaler()]}
          }

oversamplings = { 'RandomOverSampler': {'sampler': [RandomOverSampler()]},
              'SMOTE': {'sampler': [SMOTE()],
                        'sampler__k_neighbors': [2,3,4,5,7,10],
                        'sampler__m_neighbors': [2,3,4,5,7,10]},
              'SVMSMOTE': {'sampler': [SVMSMOTE()],
                           'sampler__k_neighbors': [2,3,4,5,7,10],
                           'sampler__m_neighbors': [2,3,4,5,7,10]},
              'BorderlineSMOTE': {'sampler': [BorderlineSMOTE()],
                                  'sampler__k_neighbors': [2,3,4,5,7,10],
                                  'sampler__m_neighbors': [2,3,4,5,7,10],
                                  'sampler__kind': ['borderline-1', 'borderline-2']},
                 'ADASYN':  {'sampler': [ADASYN()],
                              'sampler__n_neighbors': [2,3,4,5,7,10]}
                                  
            }

undersamplings = { 'RandomUnderSampler' : {'sampler': [RandomUnderSampler()]},
                   'CondensedNearestNeighbour' : {'sampler': [CondensedNearestNeighbour()]},
                   'TomekLinks' : {'sampler': [TomekLinks()]},
                   'EditedNearestNeighbours' : {'sampler': [EditedNearestNeighbours()],
                                                'sampler__n_neighbors': [2,3,4,5,7,10]},
                   'NeighbourhoodCleaningRule': {'sampler': [NeighbourhoodCleaningRule()],
                                                 'sampler__n_neighbors': [2,3,4,5,7,10]},
                   'OneSidedSelection': {'sampler': [OneSidedSelection()],
                                         'sampler__n_neighbors': [2,3,4,5,7,10]} 
                 }

combosamplings = {'SMOTEENN': {'sampler': [SMOTEENN()]},
                  'SMOTETomek': {'sampler': [SMOTETomek()]}
                 }
samplings = {**oversamplings, **undersamplings, **combosamplings}

estimators = {'LogisticRegression': {'estimator': [LogisticRegression()],
                                     'estimator__penalty':['l1', 'l2', 'elasticnet'],
                                     'estimator__class_weight': ['balanced'],
                                     'estimator__C': np.logspace(1, 2, 10),
                                     'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
            'LogisticRegression2':{'estimator': [LogisticRegression()],
                                     'estimator__penalty':[ 'l2'],
                                     'estimator__C': np.logspace(1, 2, 10),
                                     'estimator__solver': ['newton-cg', 'lbfgs', 'sag']},
              'LogisticRegression3':{'estimator': [LogisticRegression()],
                                     'estimator__penalty':[ 'elasticnet'],
                                     'estimator__C': np.logspace(1, 2, 10),
                                     'estimator__solver': ['saga']},
              'LinearDiscriminantAnalysis': {'estimator': [LinearDiscriminantAnalysis()],
                                             'estimator__solver': ['svd', 'lsqr', 'eigen']},
              'QuadraticDiscriminantAnalysis' :{'estimator': [QuadraticDiscriminantAnalysis()]},
              'GaussianNB': {'estimator': [GaussianNB()]},
              'BernoulliNB': {'estimator': [BernoulliNB()],
                              'estimator__alpha': [np.logspace(0,1,10, endpoint=False)/10]},
              'DecisionTreeClassifier': {'estimator': [DecisionTreeClassifier()],
                                         'estimator__criterion': ['gini', 'entropy'],
                                         'estimator__class_weight': ['balanced'],
                                         'estimator__splitter': ['best', 'random'],
                                         'estimator__max_depth': [2,5,10,None]},
              'SVC': {'estimator': [SVC()],
                      'estimator__C': np.logspace(1, 2, 10),
                      'estimator__class_weight': ['balanced'],
                      'estimator__kernel': ["linear", "rbf", "poly"],#['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                      'estimator__gamma': ['scale', 'auto']},
              'LinearSVC': {'estimator': [LinearSVC()],
                            'estimator__penalty': ['l1','l2'],
                            'estimator__loss': ['hinge', 'squared_hinge'],
                            'estimator__class_weight': ['balanced'],
                            'estimator__C': np.logspace(1, 2, 10)},
              'NuSVC': {'estimator': [NuSVC()],
                        'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                        'estimator__gamma': ['scale', 'auto'],
                        'estimator__class_weight': ['balanced']},
              'OneClassSVM': {'estimator': [OneClassSVM()],
                              'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                              'estimator__gamma': ['scale', 'auto']},
              'RandomForestClassifier': {'estimator': [RandomForestClassifier()],
                                         'estimator__n_estimators': [10,20,50,100,500,1000],
                                         'estimator__criterion': ['gini', 'entropy'],
                                         'estimator__max_depth': [2,5,10,None],
                                         'estimator__max_features': ['auto', 'sqrt', 'log2'],
                                         'estimator__class_weight': ['balanced', 'balanced_subsample']},
              'IsolationForest': {'estimator': [IsolationForest()],
                                  'estimator__n_estimators': [10,20,50,100,500,1000]},
              'GradientBoostingClassifier': {'estimator': [GradientBoostingClassifier()],
                                             'estimator__n_estimators': [10,20,50,100,500,1000],
                                             'estimator__learning_rate': np.logspace(0,1,10, endpoint=False)/10,
                                             'estimator__criterion': ['friedman_mse', 'mse', 'mae'],
                                             'estimator__max_features': ['auto', 'sqrt', 'log2']}
             }

if __name__ == '__main__':
    param_grid = create_params_grid(scalers, decompositions, estimators)
    print(param_grid)