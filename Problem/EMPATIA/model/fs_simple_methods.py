from sklearn.feature_selection import (SequentialFeatureSelector, mutual_info_regression)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso


"""
This file contains functions related to simple methods of feature selection
"""



def create_wrapper_list(loader, n_variables, label_to_predict = 'EmocionReportada'):

    wrapper = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=20),
                                        n_features_to_select=n_variables,
                                        scoring = 'f1',
                                        n_jobs=5)

    
    X, y, _, _ = loader.split_data_labels(loader.dataset)
    wrapper.fit(X, y[label_to_predict].ravel())
    feature_list = list(wrapper.get_support())
    lista_valores = [1 if variable else 0 for variable in feature_list]

    return lista_valores

def create_filter_list(loader, threshold, label_to_predict = 'EmocionReportada'):
    X, y, _, _ = loader.split_data_labels(loader.dataset)
    mi = mutual_info_regression(X, y[label_to_predict].ravel())
    lista_valores = [1 if valor>=threshold else 0 for valor in mi]
    return lista_valores

def create_LASSO_list(loader, alpha, label_to_predict = 'EmocionReportada'):
    
    X, y, _, _ = loader.split_data_labels(loader.dataset, labels_names = loader.labels)       
    linlasso = Lasso(alpha=alpha, max_iter = 10000).fit(X, y[label_to_predict].ravel())
    lista_valores = [1 if abs(valor)>0 else 0 for valor in linlasso.coef_]
    
    return lista_valores

def create_LASSO_foreach(loader, alpha, label_to_predict = 'EmocionReportada'):
    """
    Returns a list of lists, each list contains a list of the features selected
    by the LASSO technique for each volunteer.
    """
    lista_caracteristicas = []
    for X, y, _, _ in loader.volunteer_generator():
        linlasso = Lasso(alpha=alpha, max_iter = 10000).fit(X, y[label_to_predict].ravel())
        lista_valores = [1 if abs(valor)>0 else 0 for valor in linlasso.coef_]
        lista_caracteristicas.append(lista_valores)
    return lista_caracteristicas