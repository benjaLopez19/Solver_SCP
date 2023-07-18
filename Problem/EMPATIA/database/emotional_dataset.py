#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Dataset class script

Developer: Fernando Hernandez
Dev_Mail: 100318091@alumnos.uc3m.es
"""

from typing import Union, List, Tuple
from collections.abc import Callable
import pathlib
from pathlib import Path
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler

class EmotionalDataset():
    """Emotional Dataset. Class representing the emotional dataset.
    It will be useful to load the dataset.
    """

    def __init__(self,
                 root_data_dir: Union[str, pathlib.PosixPath],
                 root_labels_dir: Union[str, pathlib.PosixPath],
                 filter_idx: List[int] = None,
                 filter_trials: List[int] = None,
                 transform: Callable = None,
                 regex_pattern: str = r'_vol(?P<vol_id>[0-9]{2,3})_trial(?P<trial_id>[0-9]{2})'):
        """
        Args:
            root_data_dir (string): Directory with all csv files with the data. Each file must
                have features_idx.csv format, ex. features_23.csv
            root_labels_dir (string): Directory with all csv files with the data. Each file must
                have labelName_idx.csv format, ex. fear1_23.csv
            filter_idx (list, optional): Indexes to load in case only few are necessary.
                Ex. Only want women registers.
            filter_trials (list, optional): Indexes to load in case only few are necessary.
            Ex. Only trials 4.
            transform (callable, optional): Optional transform to be applied
                 on a sample.
            regex_patter (str): regex patter to get index from the name of date files
        Raises:
            ValueError: if any file name is inconsistent with respect its volunteer id
        """

        self.regex_pattern = regex_pattern

        self.root_data_dir = Path(root_data_dir)
        self.root_labels_dir = Path(root_labels_dir)
        self.transform = transform

        # get dataframe with the volunteer id, trial id and the path to its data file
        self.df_data = self._get_meta_data(self.root_data_dir,
                                           regex_pattern=self.regex_pattern)
        
        
        # get dataframe with the volunteer id, trial id and its labels files
        self.df_labels = self._get_meta_labels(self.root_labels_dir)

        # filter data to select the volunteers and/or trials we want
        self.filter(filter_idx, filter_trials)


        # get total number of volunteers in folder (ex 29). Will be used to
        # check we have the same amount of labels
        self.total_volunteers = self.df_data.groupby(level='vol_id').ngroups
        self.total_trials = self.df_data.groupby(level='trial_id').ngroups

        try:
            # each file must have the right id in its name
            self._check_id_consistency(self.df_data)
        except ValueError:
            raise ValueError('Labels dataframe inconsistent')


    def __len__(self) -> int:
        return len(self.df_data)

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, pd.DataFrame]: 
        # iloc is to select row by position
        # returns a tuple with the data and the labels at idx
        return (self.df_data.iloc[[idx]],
                self.df_labels.iloc[[idx], :])

    
    def __repr__(self) -> str:
        return f'EmotionalDataset with {self.total_volunteers} volunteers and {self.total_trials} trials'
    
    @staticmethod
    def _check_id_consistency(df_data: pd.DataFrame) -> None:

        """
        Function to check if each column in each row contains the volunteer id,
        and there is no inconsistency that could mess the results.
        Args:
            df_data (pandas DataFrame): Pandas dataframe to check
        Raises:
            KeyError: Not expected error. It raises when the
                      dataframe is not well made
            ValueError: If dataframe is incosistent
        """
    
        # iterate over indexes
        for vol_x, trial_x in df_data.index:

            # display index with leading zeros (ex: 4 -> 04)
            volx_trialx_formatted = f'vol{vol_x:02d}_trial{trial_x:02d}'

            try:
                # we check if all columns contain the id in leading-zeros format
                # 1. [xs] all rows with vol_id equals to id_x (volunteer id)
                if isinstance(df_aux := df_data.xs((vol_x, trial_x), drop_level=False),
                              pd.DataFrame):
                    # 2. [squeeze] transform DataFrame into Serie (only if needed)
                    df_aux = df_aux.squeeze()

                # 3. [str.contains] check string
                if not all(df_aux.str.contains(volx_trialx_formatted)):
                    # if not all values are true, we raise a ValueError exception
                    raise ValueError('Dataframe inconsistent')
            except KeyError:
                print('Unexpected keyError detected')

    @staticmethod
    def _get_meta_data(root_dir: pathlib.PosixPath,
                       regex_pattern: str) -> pd.DataFrame:
        """
        Function to iterate over folders to load the path of all volunteers data files
        Args:
            root_dir (pathlib.PosixPath): root path to start the search
            regex_pattern (str): regex patter to get index from the name of date files
        Raises:
            ValueError: when any filename is not right named
        """

        # dictionaries to save the meta data
        meta_data_dict = {}

        # Take in a list all files with csv extension
        csv_data_files = list(root_dir.glob('*.csv'))

        # idx list is built with the name of the file.
        # From features_01_14.csv extract (1, 14).
        try:

            index_vol_trial = [ tuple(map(int, re.search(regex_pattern, file.stem).groups()))
                                for file in csv_data_files ]
        

        except AttributeError as attr_error:
            raise ValueError('Incorrect filename format.'+\
                             f' Not matches {regex_pattern}') from attr_error

        # create the MultiIndex with the volunteer id and the trial id
        pandas_indexes = pd.MultiIndex.from_tuples(index_vol_trial,
                                                   names=["vol_id", "trial_id"])


        
        meta_data_dict['feature_path'] = csv_data_files
        return pd.DataFrame(meta_data_dict,
                            index=pandas_indexes,
                            dtype=str).sort_index()

    @staticmethod
    def _get_meta_labels(root_dir: pathlib.PosixPath) -> pd.DataFrame:
        """
        Function to iterate over folders to load the path of all volunteers labels files
        Args:
            root_dir (pathlib.PosixPath): root path to start the search
        """

        # iteration over labels folders (folders are aro, arosal, dEmotion ...,
        # and contains files with the right target labels)

        # labels dataframe initialization
        labels_df = pd.DataFrame()

        # files with the labels info
        csv_files = list(root_dir.glob('*.csv'))

        # read all files and concat the result in just one dataframe
        for csv_file in csv_files:
            labels_df = pd.concat([labels_df, pd.read_csv(csv_file, sep=',')])
        
        # change index names
        labels_df = labels_df.rename(columns={"Voluntaria": "vol_id", "Video": "trial_id"})

        # return the dataframe with the multiindex
        return labels_df.set_index(['vol_id', 'trial_id'])


    def filter(self, filter_idx: List[int] = None,
               filter_trials: List[int] = None):
        """
        Function to filter the data by the indexes of the volunteers and trials
        Args:
            filter_idx (list): list of volunteers indexes to filter
            filter_trials (list): list of trials indexes to filter
        """
        if filter_idx:
            self.df_data = self.df_data[~self.df_data.index.isin(filter_idx,
                                                                    level=0)]
            self.df_labels = self.df_labels[~self.df_labels.index.isin(filter_idx,
                                                                        level=0)]        
            
        if filter_trials:
            self.df_data = self.df_data[~self.df_data.index.isin(filter_trials,
                                                                    level=1)]
            self.df_labels = self.df_labels[~self.df_labels.index.isin(filter_trials,
                                                                        level=1)]
