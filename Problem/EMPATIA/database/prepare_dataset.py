import os

from Problem.EMPATIA.database.emotional_dataset import EmotionalDataset
from Problem.EMPATIA.database.basic_loader import BasicLoader

def prepare_47vol_solap(data_dir):
    """
    Reads and prepares the loader for the 100 volunteers dataset with overlap.
    Args:
        data_dir (str): path to the main directory
    Returns:
        BasicLoader: loader with the dataset
    """
    loader = BasicLoader(norm = True, 
                path = data_dir + 'otherData/data47_50Overlapping.csv',
                labels_names = ['y', 'voluntaria', 'trial'])
    loader.dataset.rename(columns = {'voluntaria': 'vol_id', 'trial': 'trial_id'}, inplace = True)
    loader.labels = ['y', 'vol_id', 'trial_id']
    cols = loader.dataset.columns.tolist()
    cols = cols[3:] + cols[:3]
    loader.dataset  = loader.dataset[cols]

    return loader

def prepare_47vol_nosolap(data_dir):
    """
    Reads and prepares the loader for the 47 volunteers dataset without overlap.
    Args:
        data_dir (str): path to the main directory
    Returns:
        BasicLoader: loader with the dataset
    """
    loader = BasicLoader(norm = True, 
                path = data_dir + 'otherData/data47_noOverlapping.csv',
                labels_names = ['y', 'voluntaria', 'trial'])
    loader.dataset.rename(columns = {'voluntaria': 'vol_id', 'trial': 'trial_id'}, inplace = True)
    loader.labels = ['y', 'vol_id', 'trial_id']
    cols = loader.dataset.columns.tolist()
    cols = cols[3:] + cols[:3]
    loader.dataset  = loader.dataset[cols]

    return loader

def prepare_100vol_solap(data_dir):
    """
    Reads and prepares the loader for the 100 volunteers dataset with overlap.
    Args:
        data_dir (str): path to the main directory
    Returns:
        BasicLoader: loader with the dataset
    """
    loader = BasicLoader(norm = True, 
                path = data_dir + 'otherData/data100_50Overlapping.csv',
                labels_names = ['y', 'voluntaria', 'trial'])
    loader.dataset.rename(columns = {'voluntaria': 'vol_id', 'trial': 'trial_id'}, inplace = True)
    loader.labels = ['y', 'vol_id', 'trial_id']
    cols = loader.dataset.columns.tolist()
    cols = cols[3:] + cols[:3]
    loader.dataset  = loader.dataset[cols]

    return loader
    
def prepare_100vol_nosolap(data_dir):
    """
    Reads and prepares the loader for the 100 volunteers dataset without overlap.
    Args:
        data_dir (str): path to the main directory
    Returns:
        BasicLoader: loader with the dataset
    """
    loader = BasicLoader(norm = True, 
                path = data_dir + 'otherData/data100_NoOverlapping.csv',
                labels_names = ['y', 'voluntaria', 'trial'])
    loader.dataset.rename(columns = {'voluntaria': 'vol_id', 'trial': 'trial_id'}, inplace = True)
    loader.labels = ['y', 'vol_id', 'trial_id']
    cols = loader.dataset.columns.tolist()
    cols = cols[3:] + cols[:3]
    loader.dataset  = loader.dataset[cols]

    return loader