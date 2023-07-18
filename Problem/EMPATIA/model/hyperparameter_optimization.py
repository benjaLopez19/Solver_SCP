import json

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from Problem.EMPATIA.database.basic_loader import BasicLoader

def calculate_optimal_hyperparameters(loader, label_to_predict: str = 'EmocionReportada'):
    grid_params = { 'n_neighbors': list(range(11, 39, 1)),
                    'weights': ['uniform', 'distance'],
                    'metric' : ['minkowski','euclidean','manhattan']}

    best_params = []
    for X_tr, y_tr, _, y_te in loader.LASO_generator():
        print('Volunteer: ', y_te['vol_id'].unique()[0])
        gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=5, n_jobs=-1)
        gs_results = gs.fit(X_tr, y_tr[label_to_predict].ravel())
        best_params.append(gs_results.best_params_)
    return best_params

def save_parameters(best_params, name):
    with open(name + '.json', 'w') as file:
        json.dump(best_params, file)

def load_parameters(directory, name):
    with open(directory + 'optimal_parameters/' + name + '.json', 'r') as file:
        best_params = json.load(file)
    return best_params