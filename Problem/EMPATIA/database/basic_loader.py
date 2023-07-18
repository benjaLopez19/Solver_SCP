#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Loader dataset script

Developer: Fernando Hernandez
Dev_Mail: 100318091@alumnos.uc3m.es
"""

from typing import (TypeVar,
                    List,
                    Generic,
                    Callable)
import random 
import pandas as pd
import numpy as np
import sys

from .emotional_dataset import EmotionalDataset

#from ml_helpers.cv_splitter import CV_Splitter
from sklearn.preprocessing import StandardScaler

# LABELS_NAME = ['vol_id',
#                'trial_id',
#                'Tanda',
#                'EmocionReportada',
#                'EmocionTarget',
#                'PAD',
#                'Arousal',
#                'Valencia',
#                'Dominancia']
LABELS_NAME = ['vol_id',
               'trial_id',
               'y']
L = TypeVar('L')      # Declare type variable

class BasicLoader(Generic[L]):
    """
    Emotional Dataset Loader. Class to create a data loader.
    It will be useful to load the dataset.
    """

    def __init__(self, dataset: EmotionalDataset = None,
                 transformation: Callable = None,
                 norm: bool = False,
                 shuffle: bool = False,
                 frac: float = 1.,
                 path: str = None,
                 labels_names: List[str] = LABELS_NAME):
        """
        Args:
            dataset (Dataset): Base dataset
            transformation (Callable): transformation to apply to the data
            norm (bool): wether or not to normalize the data
            shuffle (bool): wheter or not to shuffle the data
            frac (float): in case only a sample set from the original is required.
            e.g. 0.5 will return half of the total dataset
            path (str): path to the dataset, if none, it will be loaded from the EmotionalDataset
            labels_names (list): list with the names of the labels
        """
        self.labels = labels_names
        self.emotional_dataset = dataset
        self.transform = transformation
        self.norm = norm
        if path is None:
            self.dataset = self.load_data()
        else:
            self.dataset = self.load_csv(path)

        # to shuffle the data
        if shuffle or frac < 1.:
            self.dataset = self.dataset.sample(frac=frac)
        
        if self.norm:
            self.dataset = self.indiv_normalize(self.dataset, self.labels)

        if self.transform:            
            self.dataset = self.transform(self.dataset)
    
    @staticmethod
    def normalize(data, labels_name: list = LABELS_NAME) -> pd.DataFrame:
        """
        Normalize data
        Args:
            data (DataFrame): Data to normalize
            labels_name (list): list with labels names that will not be normalized
        Returns:    
            DataFrame: Normalized data
        """
        set_labels = set(labels_name)
        set_columns = set(data.columns)
        set_features = set_columns - set_labels
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data[list(set_features)])
        data[list(set_features)] = features_scaled
        
        return data
    
    @staticmethod
    def indiv_normalize(data, labels_name: list = LABELS_NAME) -> pd.DataFrame:
        """
        Individual normalize data (per volunteer)
        Args:
            data (DataFrame): Data to normalize
            labels_name (list): list with labels names that will not be normalized
        Returns:    
            DataFrame: Normalized data by volunteer
        """
        set_labels = set(labels_name)
        set_columns = set(data.columns)
        set_features = set_columns - set_labels
        scaler = StandardScaler()

        if ('vol_id' in labels_name):
            pattern = 'vol_id' 
        else:
            pattern = 'voluntaria'
        set_vol = set(data[pattern])
        
        data = data.reset_index(drop = True)
        
        for vol in set_vol:           
            features_scaled = scaler.fit_transform(data.loc[data[pattern]==vol, list(set_features)])        
            data.loc[data[pattern]==vol, list(set_features)] = features_scaled

        return data
                                             
    def LASO_generator(self, df_data = None):
        """
        Leave half subject out generator
        Args:
            df_data (DataFrame): data to split, if None, self.dataset will be used,
            It must contain the same labels as self.dataset, but may have a subset of
            features.
        Returns:
            generator: generator with a tuple with the train and test set
            for each volunteer following the LASO strategy
        """

        if df_data is None:
            df_data = self.dataset
        set_vol = set(df_data["vol_id"])
        set_trial = set(df_data["trial_id"])
        
        df_data = df_data.reset_index(drop = True)
        
        for vol in set_vol:
            test_trial_set = list(set_trial)[len(set_trial)//2:]

            test_selected = df_data[['vol_id', 'trial_id']].isin({'vol_id':[vol], 'trial_id':test_trial_set}).all(axis=1)
            test_set = df_data[test_selected]
            train_set = df_data[~test_selected]
            
            yield self.split_data_labels(train_set, test_set, self.labels)
            
    def LASO_generator_voluntaria(self, df_data = None, id = 0):
        """
        Leave half subject out generator
        Args:
            df_data (DataFrame): data to split, if None, self.dataset will be used,
            It must contain the same labels as self.dataset, but may have a subset of
            features.
        Returns:
            generator: generator with a tuple with the train and test set
            for each volunteer following the LASO strategy
        """

        if df_data is None:
            df_data = self.dataset
        set_trial = set(df_data["trial_id"])
        
        df_data = df_data.reset_index(drop = True)
        
        
        test_trial_set = list(set_trial)[len(set_trial)//2:]

        test_selected = df_data[['vol_id', 'trial_id']].isin({'vol_id':[id], 'trial_id':test_trial_set}).all(axis=1)
        test_set = df_data[test_selected]
        train_set = df_data[~test_selected]
            
        yield self.split_data_labels(train_set, test_set, self.labels)

    def LOSO_generator(self, df_data = None):
        """
        Leave one subject out generator
        Args:
            df_data (DataFrame): data to split, if None, self.dataset will be used,
            It must contain the same labels as self.dataset, but may have a subset of
            features.
        Returns:
            generator: generator with a tuple with the train and test set
            for each volunteer following the LOSO strategy
        """

        if df_data is None:
            df_data = self.dataset
        set_vol = set(df_data["vol_id"])
        set_trial = set(df_data["trial_id"])
        
        df_data = df_data.reset_index(drop = True)
        
        for vol in set_vol:
            # Incluimos todos los trials de una voluntaria en la parte del testeo
            # Lo convertimos en divisiones de testeo de LOSO
            test_trial_set = list(set_trial)

            test_selected = df_data[['vol_id', 'trial_id']].isin({'vol_id':[vol], 'trial_id':test_trial_set}).all(axis=1)
            test_set = df_data[test_selected]
            train_set = df_data[~test_selected]
            
            yield self.split_data_labels(train_set, test_set, self.labels)

    def volunteer_generator(self, df_data = None):
        """
        Volunteer generator
        Args:
            df_data (DataFrame): data to split, if None, self.dataset will be used,
        Returns:
            generator: generator with a tuple with the data and labels for each volunteer
        """
        if df_data is None:
            df_data = self.dataset            
        set_vol = set(df_data["vol_id"])
        for vol in set_vol:
            yield self.split_data_labels(df_data[df_data["vol_id"] == vol])
            

    def split_data_labels(self, train_set = None, test_set = None, labels_names = LABELS_NAME):


        """
        Splits train and test sets into data (X) and labels (y) . if test set is None, it only
        splits the train set.
        Args:
            train_set (DataFrame): train set to split, if none, self.dataset will be used
            test_set (DataFrame): test set to split, if None, X_test and y_test will be None
            labels_names (list): list with the names of the labels
        Returns:
            tuple: tuple with the train and test set splitted into data and labels
            (X_train, y_train, X_test, y_test)

        """
        if train_set is None:
            train_set = self.dataset

        Y_train = train_set[labels_names]
        X_train = train_set.drop(labels_names, axis=1)
        
        if test_set is None:
            return (X_train.reset_index(drop=True), 
                    Y_train.reset_index(drop=True),
                    None, None)

        Y_test = test_set[labels_names]
        X_test = test_set.drop(labels_names, axis=1)

        return (X_train.reset_index(drop=True), Y_train.reset_index(drop=True),
                X_test.reset_index(drop=True), Y_test.reset_index(drop=True))

    def load_data(self) -> pd.DataFrame:
        """
        Args:
            shuffle (bool): shuffle the result dataframe before return it
            sample_frac (float): in case only a sample set from the original
                                 is required, a fraction can be specified, ex.,
                                 0.5 will return half of the total dataset.
        """

        data_list = []

        # iterate over each volunteer
        for data_file_name, df_labels in self.emotional_dataset:
            # read data from each meta data file
            df_data = pd.read_csv(data_file_name.squeeze())

            df_labels = df_labels.reset_index()
            
            

            df_data = df_data.merge(right=df_labels,
                                    how='left',
                                    on=['vol_id', 'trial_id'])
            
            # save each volunteer
            data_list.append(df_data)

        # final dataframes creation
        df_data = pd.concat(data_list,
                            axis=0,
                            ignore_index=False)

        return df_data

    def load_csv(self, directory) -> pd.DataFrame:
        return pd.read_csv(directory)

    def select_features(self, selected_features):
        """
        Selects the features from the dataset
        Args:
            selected_features (list): list with the selected features, 
            1 if the feature is selected, 0 otherwise. Note that the
            length of the list must be equal to the number of features
        Returns:
            DataFrame: dataset with the selected features only
        """
        counter = 0
        data_copy = self.dataset.copy()
        for feature in selected_features:
            if feature == 0:
                data_copy.drop(data_copy.columns[counter], axis=1, inplace=True)
            else:
                counter += 1
        return data_copy
    
    def exclude_volunteers(self, exclude_vol):
        """
        Selects the volunteers to drop out from the dataset
        Args:
            exclude_vol (list): list with the selected volunteers that 
            we want to take out of the dataset because they do not reach  
            fear self-report filter.
        Returns:
            DataFrame: dataset with the rest of the volunteers
        """
        data_copy = self.dataset.copy()

        set_vol = set(data_copy["vol_id"])
        set_exclude = set(exclude_vol)
        set_selected = set_vol - set_exclude

        data_copy = data_copy.loc[data_copy['vol_id'].isin(list(set_selected))]
        return data_copy

    def select_volunteers(self, include_vol):
        """
        Selects the volunteers to keep in the dataset
        Args:
            include_vol (list): list with the selected volunteers that 
            we want to keep in the dataset.
        Returns:
            DataFrame: dataset with the selected volunteers
        """
        data_copy = self.dataset.copy()
        set_selected = set(include_vol)


        data_copy = data_copy.loc[data_copy['vol_id'].isin(list(set_selected))]
        return data_copy