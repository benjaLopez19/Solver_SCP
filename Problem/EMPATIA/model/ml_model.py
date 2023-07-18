import numpy as np
import pandas as pd
from math import sqrt

from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import (KNeighborsMixin, NeighborsBase,
                                     _check_weights, _get_weights)
from sklearn.utils.validation import _num_samples
from sklearn.base import ClassifierMixin

# Other classifiers
from sklearn.tree import DecisionTreeClassifier
from Problem.EMPATIA.database.basic_loader import BasicLoader
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


'''
KNN sklearn class modification to be able to include class_weight as parameter. Following functions
 are  included:
 
 (1): get_class_mode: To get the class mode and obtain the prediction considering: weights (distance 
      to the neighbors) and class_weight (new)
 (2): modifiedKNeighborsClassifier: copy of the original KNN sklearn class with some modifications when
      doing the predictions
'''

def get_class_mode(a, weight = None, class_weight = 1.0):

    """
    Gets the class mode (y_pred) considering: (1) weights (distance to the neighbors) and (2) class_weight
    Args:
        a (np.array): Training data.
        weight (np.array): Training labels.
        class_weight (np.array): Test data.

    Returns:
        y_pred (np.array): Predicted labels.
    """
    class_mode = np.empty((0,1),int)
    for n_rows in range(0,a.shape[0]):
        temp_a = np.ravel(a[n_rows,:])

        if weight is not None:           
            temp_weight = np.ravel(weight[n_rows,:])
            if temp_a.shape != temp_weight.shape:
                raise ValueError(
                "weights and neigbours different"
                )

            unique_class = np.unique(temp_a)  # get ALL unique values
            counts = np.zeros(len(np.ravel(unique_class)))
            for score in unique_class:
                ind = temp_a == score

                if len(np.ravel(unique_class)) == 2:
                    counts[score] = np.sum(temp_weight[ind])
                else:
                    counts = np.sum(temp_weight[ind])                          
        else:
            unique_class, counts = np.unique(temp_a, return_counts=True)
        
        if(len(unique_class) ==2):
            if counts[0] >= (counts[1]*class_weight):
                row_mode = 0
            else:
                row_mode = 1
        elif (len(unique_class) ==1):
            row_mode = unique_class
        else:
            raise ValueError(
                "wrong number of classes identified"
            )

        class_mode = np.append(class_mode,row_mode)
    
    return class_mode.ravel()

class modifiedKNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):
    """Classifier implementing the k-nearest neighbors vote.

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.


    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        class_weight = 1,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights
        self.class_weight=class_weight

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        self.weights = _check_weights(self.weights)

        return self._fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        
        for k, classes_k in enumerate(classes_):
            if weights is None:
                    mode = get_class_mode(_y[neigh_ind, k], class_weight = self.class_weight)
            else:
                    mode = get_class_mode(_y[neigh_ind, k], weight = weights, class_weight = self.class_weight)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()
        return y_pred


'''
Training and model predictions
'''

def train_and_predict(X_tr, y_tr, X_te, neighbors = None, 
                      class_weight = 1.6, 
                      label_to_predict = 'EmocionReportada',
                      parameters = None):
    """
    Trains a KNN model with the given training data X_tr and y_tr, and predicts the labels (y_te) of the test data X_te.
    Args:
        X_tr (np.array): Training data.
        y_tr (np.array): Training labels.
        X_te (np.array): Test data.
        neighbors (int): Number of neighbors to consider.
        missclassification_cost (int): sensitivity towards the positive class.
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict.
        parameters (dict): parameters to use in the model. if None, specified
        and default values will be used.
    Returns:
        y_pred (np.array): Predicted labels.
    """
    if neighbors is None:
        neighbors = int(sqrt(X_tr.shape[0]))
    
    y_pred = []
    
    try: KNN = modifiedKNeighborsClassifier(n_neighbors=parameters['n_neighbors'], 
                                            weights=parameters['weights'], 
                                            class_weight=parameters['class_weight'], 
                                            metric=parameters['metric'])
    
    except TypeError: KNN = modifiedKNeighborsClassifier(n_neighbors=neighbors, 
                                                         weights='distance',
                                                         class_weight = class_weight)
    
    #Entrenamos el KNN
    KNN.fit(X_tr, y_tr[label_to_predict].ravel())
    y_pred = KNN.predict(X_te)

    return y_pred

def get_metrics_voluntaria(loader: BasicLoader, 
                neighbors = 20, 
                misclassification_cost = 1.6, 
                label_to_predict = 'y',
                selected_features = np.ones(57),
                optimal_parameters = None,
                threshold = 0.3,
                id = 0, 
                opt_params = 0):
    """
    Trains and validates the dataset in loader using the LASO techique, and reduces 
    the dimensionality of the data using the selected features.
    Args:
        loader (BasicLoader): loader of the dataset
        neighbors (int): number of neighbors to consider in KNN
        missclassification_cost (int): sensitivity towards the positive class
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features. 1 if
        the feature is selected, 0 otherwise
        optimal_parameters (dict): dictionary with the optimal parameters for the KNN,
        if None, specified and default parameters are used

    Returns:
        numpy array: array with length 4 containing (TN, FP, FN, TP) in that order.
    """


    matrix = np.zeros((2, 2))
    matrix_merge = np.zeros((2, 2))


    # selects the features to use
    reduced_dataset = loader.select_features(selected_features)
    metrics_df = pd.DataFrame({'vol_id':[],'tn':[],'fp':[],'fn':[],'tp':[],'f1':[],'acc':[],'tn_merge':[],'fp_merge':[],'fn_merge':[],'tp_merge':[],'f1_merge':[],'acc_merge':[]})
    
    i = 0;
    # Iterate using the LASO technique
    
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator_voluntaria(reduced_dataset, id):
            
        # Train and predict
        try: parameters = optimal_parameters[opt_params]
        except TypeError: parameters = None

        y_pred = train_and_predict(X_tr, y_tr, X_te, neighbors, misclassification_cost, label_to_predict, parameters)
        # Add the confusion matrix to the total
        indv_confusion_matrix = confusion_matrix(y_te[label_to_predict].ravel(), y_pred)
        tn, fp, fn, tp = indv_confusion_matrix.ravel()
        f1 = f1_score(y_te[label_to_predict].ravel(), y_pred)
        acc = accuracy_score(y_te[label_to_predict].ravel(), y_pred)
        matrix += indv_confusion_matrix

        scores_merge = get_video_merge_metrics(label_to_predict = label_to_predict, thlevel = threshold, 
                            df_yte = y_te, y_pred = y_pred)
        tn_merge, fp_merge, fn_merge, tp_merge = scores_merge[0].ravel()
        # matrix_merge += metrics_merge

        d = {'vol_id':[y_te['vol_id'].unique()[0]],'tn':[tn],'fp':[fp],'fn':[fn],
            'tp':[tp],'f1':[f1],'acc':[acc],'tn_merge':[tn_merge],'fp_merge':[fp_merge],'fn_merge':[fn_merge],
            'tp_merge':[tp_merge],'f1_merge':[scores_merge[1]],'acc_merge':[scores_merge[2]]
            }
            
        temp_metrics_df = pd.DataFrame(data=d)
        if metrics_df.empty: metrics_df = temp_metrics_df
        else: metrics_df = pd.concat([metrics_df, temp_metrics_df])

    
    
    # Return the confusion matrix (TN, FP, FN, TP) for no merged and merged labels
    return metrics_df

def get_metrics(loader: BasicLoader, 
                neighbors = 20, 
                misclassification_cost = 1.6, 
                label_to_predict = 'y',
                selected_features = np.ones(57),
                optimal_parameters = None,
                threshold = 0.3):
    """
    Trains and validates the dataset in loader using the LASO techique, and reduces 
    the dimensionality of the data using the selected features.
    Args:
        loader (BasicLoader): loader of the dataset
        neighbors (int): number of neighbors to consider in KNN
        missclassification_cost (int): sensitivity towards the positive class
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features. 1 if
        the feature is selected, 0 otherwise
        optimal_parameters (dict): dictionary with the optimal parameters for the KNN,
        if None, specified and default parameters are used

    Returns:
        numpy array: array with length 4 containing (TN, FP, FN, TP) in that order.
    """


    matrix = np.zeros((2, 2))
    matrix_merge = np.zeros((2, 2))


    # selects the features to use
    reduced_dataset = loader.select_features(selected_features)

    i = 0;
    # Iterate using the LASO technique
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator(reduced_dataset):
        # Train and predict
        try: parameters = optimal_parameters[i]
        except TypeError: parameters = None

        y_pred = train_and_predict(X_tr, y_tr, X_te, neighbors, misclassification_cost, label_to_predict, parameters)
        # Add the confusion matrix to the total
        matrix += confusion_matrix(y_te[label_to_predict].ravel(), y_pred)

        metrics_merge = get_video_merge_metrics(label_to_predict = label_to_predict, thlevel = threshold, 
                            df_yte = y_te, y_pred = y_pred)   
        matrix_merge += metrics_merge
        i += 1
    
    tn, fp, fn, tp = matrix.ravel()
    tn_merge, fp_merge, fn_merge, tp_merge = matrix_merge.ravel()
    # Return the confusion matrix (TN, FP, FN, TP) for no merged and merged labels
    return tn, fp, fn, tp, tn_merge, fp_merge, fn_merge, tp_merge

def get_video_merge_metrics(label_to_predict = "EmocionReportada",
                            thlevel = 0.3, 
                            df_yte = pd.DataFrame,
                            y_pred = np.array):

    """
    Obtains quality metrics of the predicted vs expected labels. Predicted labels are computed
    by video if the mean of the labels reach a threshold a 1 label is assigned otherwise 0.

    Args:
        loader (BasicLoader): loader of the dataset
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features.
        1 if the feature is selected, 0 otherwise
    Returns:
        tuple: (list, list, ConfusionMatrix): list of f1 and acc for each volunteer, and confusion matrix
    """

    df_yte['y_pred'] = y_pred
    df_yte = df_yte.groupby(['vol_id', 'trial_id']).mean()
    df_yte['y_merge'] = np.where(df_yte['y_pred']< thlevel, 0, 1)

    f1_merge = f1_score(df_yte[label_to_predict].ravel(), df_yte['y_merge'].ravel())
    acc_merge = accuracy_score(df_yte[label_to_predict].ravel(), df_yte['y_merge'].ravel())
    confusion_matrix_merge = confusion_matrix(df_yte[label_to_predict].ravel(), df_yte['y_merge'].ravel())

    # return confusion_matrix_merge, f1_merge, acc_merge
    return confusion_matrix_merge


'''
Other classifiers
'''

def get_metrics_svm(loader, kernel = 'linear', 
                    label_to_predict = 'EmocionReportada',
                    selected_features = np.ones(57), 
                    misclassification_cost = 1):
    """
    Trains and validates the dataset in loader using the LASO techique and SVM model.
    Args:
        loader (BasicLoader): loader of the dataset
        kernel (str): kernel to use in the SVM model
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features.
        1 if the feature is selected, 0 otherwise
        misclassification_cost (int): sensitivity towards the positive class
    Returns:
        tuple: (list, list, ConfusionMatrix): list of f1 and acc for each volunteer, and confusion matrix 
    """
    f1_list = []
    acc_list = []
    matrix = np.zeros((2, 2))

    # selects the features to use
    reduced_dataset = loader.select_features(selected_features)
    print('-'*len(set(loader.dataset['vol_id'])))
    # Iterate using the LASO technique
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator(reduced_dataset):
        
        # Train and predict
        SVM = SVC(class_weight={1: misclassification_cost}, kernel = kernel)
        SVM.fit(X_tr, y_tr[label_to_predict].ravel())
        y_pred = SVM.predict(X_te)
        # Add the confusion matrix to the total
        matrix += confusion_matrix(y_te[label_to_predict].ravel(), y_pred)
        f1_list.append(f1_score(y_te[label_to_predict].ravel(), y_pred))
        acc_list.append(accuracy_score(y_te[label_to_predict].ravel(), y_pred))
        print('*', end  = '')
    return f1_list, acc_list, matrix

def get_metrics_tree(loader, max_depth = None, 
                    misclassification_cost = 1, 
                    label_to_predict = "EmocionReportada", 
                    selected_features = np.ones(57)):
    """
    Trains and validates the dataset in loader using the LASO techique and Decision Tree Classifier model.
    Args:
        loader (BasicLoader): loader of the dataset
        max_depth (int): maximum depth of the tree
        misclassification_cost (int): sensitivity towards the positive class
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features.
        1 if the feature is selected, 0 otherwise
    Returns:
        tuple: (list, list, ConfusionMatrix): list of f1 and acc for each volunteer, and confusion matrix
    """
    f1_list = []
    acc_list = []
    matrix = np.zeros((2, 2))

    # selects the features to use
    reduced_dataset = loader.select_features(selected_features)
    print('-'*len(set(loader.dataset['vol_id'])))
    # Iterate using the LASO technique
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator(reduced_dataset):            
        # Train and predict
        tree = DecisionTreeClassifier(class_weight={1: misclassification_cost}, max_depth = max_depth)
        tree.fit(X_tr, y_tr[label_to_predict].ravel())
        y_pred = tree.predict(X_te)
        # Add the confusion matrix to the total
        matrix += confusion_matrix(y_te[label_to_predict].ravel(), y_pred)
        f1_list.append(f1_score(y_te[label_to_predict].ravel(), y_pred))
        acc_list.append(accuracy_score(y_te[label_to_predict].ravel(), y_pred))
        print('*', end  = '')
    return f1_list, acc_list, matrix

def get_metrics_boost(loader, label_to_predict = "EmocionReportada",
                        selected_features = np.ones(57)):
    
    """
    Trains and validates the dataset in loader using the LASO techique and AdaBoost model.
    Note this model does not have misclassification cost, so sensitivity towards imbalanced data
    may affect the results.
    Args:
        loader (BasicLoader): loader of the dataset
        label_to_predict (str): in case y_tr contains more than one label, this is the one to predict
        selected_features (np.array): array of n elements, where n is the number of features.
        1 if the feature is selected, 0 otherwise
    Returns:
        tuple: (list, list, ConfusionMatrix): list of f1 and acc for each volunteer, and confusion matrix
    """
    
    f1_list = []
    acc_list = []
    matrix = np.zeros((2, 2))

    # selects the features to use
    reduced_dataset = loader.select_features(selected_features)
    print('-'*len(set(loader.dataset['vol_id'])))
    # Iterate using the LASO technique
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator(reduced_dataset):
        # Train and predict
        boost = AdaBoostClassifier()
        boost.fit(X_tr, y_tr[label_to_predict].ravel())
        y_pred = boost.predict(X_te)
        # Add the confusion matrix to the total
        matrix += confusion_matrix(y_te[label_to_predict].ravel(), y_pred)
        f1_list.append(f1_score(y_te[label_to_predict].ravel(), y_pred))
        acc_list.append(accuracy_score(y_te[label_to_predict].ravel(), y_pred))
        print('*', end  = '')
    return f1_list, acc_list, matrix


'''
Other functions
'''

def generate_scores(loader: BasicLoader, 
                    crossvalidation_k = 10, 
                    neighbors = None, 
                    misclassification_cost = 1, 
                    label_to_predict = 'EmocionReportada', 
                    selected_volunteers = [],
                    selected_features = np.ones(57)):

    """
        Generates the scores of the KNN model Using CROSS VALIDATION. and LASO techniques.
        Args:
            loader (BasicLoader): loader of the data
            crossvalidation_k (int): number of folds for cross validation
            neighbors (int): number of neighbors to consider in KNN, 
                if None, KNN will take the sqrt of the number of samples in the training set.
            missclassification_cost (int): sensitivity towards the positive class.
            label_to_predict (str): in case y_tr contains more than one label, this is the one to predict.
            selected_volunteers (list): list of volunteers to use in the training set. if empty, all volunteers will be used.
            selected_features (np.array): array of 57 elements, 1 if the feature is selected, 0 otherwise.
        Returns:
            scores (dict): dictionary with the scores of the model, with the following keys:
                'LASO_f1_list': list of f1 scores for each volunteer in the training set using LASO method.
                'LASO_f1_mean': mean of the f1 scores
                'LASO_f1_std': standard deviation of the f1 scores
                'LASO_acc_list': list of accuracy scores 
                'LASO_acc_mean': mean of the accuracy scores
                'LASO_acc_std': standard deviation of the accuracy scores
                'LASO_matrix_list': list of confusion matrices for each volunteer
                'LASO_matrix': sum of the confusion matrices for each volunteer in the training set using LASO method.
                'CV_f1_list': list of f1 scores for each fold in the cross validation
                'CV_f1_mean': mean of the f1 scores
                'CV_f1_std': standard deviation of the f1 scores
                'CV_acc_list': list of accuracy scores
                'CV_acc_mean': mean of the accuracy scores
                'CV_acc_std': standard deviation of the accuracy scores
                'CV_matrix': sum of the confusion matrices for each fold in the cross validation
            
    """
    f1_list = []
    acc_list = []
    matrix = np.zeros((2, 2))
    confusion_matrix_list = []

    cv_f1_list = []
    cv_acc_list = []
    cv_matrix = np.zeros((2, 2))

    reduced_dataset = loader.select_features(selected_features)
    #Elegimos la partición correcta
    for X_tr, y_tr, X_te, y_te in loader.LASO_generator(reduced_dataset):
        
        # If we want to use only a subset of the volunteers, we stop the iteration
        #  if the volunteer is not in the list of selected volunteers
        if selected_volunteers != []:
            if y_te['vol_id'].unique()[0] not in selected_volunteers:
                continue
            
        # We train the model with the train set and get the predictions using the test set
        # both sets are splitted using the LASO technique.
        y_pred = train_and_predict(X_tr, y_tr, X_te, 
                                   neighbors, 
                                   misclassification_cost, 
                                   label_to_predict)

        # We save the different scores (f1, acc and confusion matrix) for each volunteer
        y_obj = y_te[label_to_predict].ravel()
        f1_list.append(f1_score(y_obj, y_pred))
        acc_list.append(accuracy_score(y_obj, y_pred))
        cm = confusion_matrix(y_obj, y_pred)
        matrix += cm
        confusion_matrix_list.append((cm, y_te["vol_id"]))

        # Make the cross validation to the training set, and save the scores.
        kf = KFold(n_splits = crossvalidation_k)
        f1_iteration_list = []
        acc_iteration_list = []
        for train_idx, test_idx in kf.split(X_tr):
            # Locate the train and test sets given the indexes
            cv_X_train, cv_X_test = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
            cv_y_train, cv_y_test = y_tr.iloc[train_idx], y_tr.iloc[test_idx]

            # Train the model and get the predictions
            cv_y_pred = train_and_predict(cv_X_train, 
                                          cv_y_train, 
                                          cv_X_test, 
                                          neighbors, 
                                          misclassification_cost, 
                                          label_to_predict)

            # We save the different scores (f1, acc and confusion matrix) for each fold
            y_obj = cv_y_test[label_to_predict].ravel()
            f1_iteration_list.append(f1_score(y_obj, cv_y_pred))
            acc_iteration_list.append(accuracy_score(y_obj, cv_y_pred))
            cv_matrix += confusion_matrix(y_obj, cv_y_pred)

        # Calculate the average of the crossvalidation scores
        cv_f1_list.append(np.mean(f1_iteration_list))
        cv_acc_list.append(np.mean(acc_iteration_list))

    return {'LASO_f1_list': f1_list,
            'LASO_f1_mean': np.mean(f1_list),
            'LASO_f1_std': np.std(f1_list),
            'LASO_acc_list': acc_list,
            'LASO_acc_mean': np.mean(acc_list),
            'LASO_acc_std': np.std(acc_list),
            'LASO_matrix_list': confusion_matrix_list,
            'LASO_matrix': matrix,
            'CV_f1_list': cv_f1_list,
            'CV_f1_mean': np.mean(cv_f1_list),
            'CV_f1_std': np.std(cv_f1_list),
            'CV_acc_list': cv_acc_list,
            'CV_acc_mean': np.mean(cv_acc_list),
            'CV_acc_std': np.std(cv_acc_list),
            'CV_matrix': cv_matrix} 

def get_class(nearest_neighbors, y_tr, misclasification_cost = 1, label_to_predict = "EmocionReportada"):
    """
    Given a list of nearest neighbors and the value of those neighbors, returns the class of the sample but
    taking into account the missclassification cost. e.g. if the nearest neighbors are [0, 1, 1, 0, 0]:
    If the missclassification cost is 1, the class will be 0 (3 zeros > 2 ones * 1). 
    If the missclassification cost is 1.6, the class will be 1 (3 zeros < 2 ones * 1.6).
    Args:
        nearest_neighbors (list): list with the position of nearest neighbors in the training set
        y_tr (pandas dataframe): training set labels
        missclasification_cost (float): tendency towards positive class
        label_to_predict (str): name of the column to predict in the training set
    Returns:
        int: class of the sample (1 or 0)
    """
    positive = 0
    negative = 0
    for neighbor in nearest_neighbors:
        if y_tr[label_to_predict].values[neighbor] == 1:
            positive += 1
        else:
            negative += 1
    positive *= misclasification_cost
    if positive >= negative:
        return 1
    else:
        return 0