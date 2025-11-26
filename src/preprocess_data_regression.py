"""
preprocess_data_regression.py
------------------------------
This module provides functionality to load and preprocess 2D Ising model 
lattice configurations for temperature regressiong tasks using machine learning.

It reads saved Ising spin configurations (.npy files) generated at different 
temperatures, labels them based on the temperature they were generated, normalizes 
the spin values, and splits the dataset into training, validation, and test sets.

Key features:
- Loads Ising configurations from .npy files by temperature.
- Assigns labels depending on which temperature the lattice was generated.
- Normalizes spins from {-1, 1} to {0, 1}.
- Expands data dimensions to match CNN input requirements.
- Performs stratified train/validation/test splitting for balanced representation.
- Automatically saves the data at preprocessed_data_reg/ as train_data.npz, val_data.npz and test_data.npz


Author - Amal Sebastian
Date - October 2025



"""

import numpy as np
import glob
from sklearn.model_selection import train_test_split
import os 



# Change directory to the location of this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




def load_and_prepare_reg_data(data_dir="../data/",L=32):
    """
    Load and preprocess 2D Ising model configurations for phase classification.

    Parameters
    ----------
    data_dir : str
        Directory containing .npy configuration files.
    L : int
        Linear lattice size.

    """    


    

    filepath = sorted(glob.glob(f"{data_dir}/ising_L{L}_T*.npy"))         
                                                                          
    X =[]
    y=[]

    for files in filepath:
        T= float(files.split("_T")[-1].replace(".npy",""))         
        configs = np.load(files)                                 # assigning labels as per the temperature
        X.append(configs)
        y.append(np.full(configs.shape[0],T))

    
    X= np.concatenate(X,axis=0)    
    y= np.concatenate(y,axis=0)

    

    X= (X+1)/2

    X = X[...,np.newaxis]


    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y =y[indices]



     # Split into train/test first, then split train into train/val
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f" Loaded {X.shape[0]} samples")
    print(f"Train: {X_train.shape},Tval: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train: {y_train.shape},Tval: {y_val.shape}, Test: {y_test.shape}")
    print(f"Class balance: Ordered={np.sum(y==0)}, Disordered={np.sum(y==1)}")

    np.savez("../preprocessed_data_reg/train_data.npz", X_train = X_train, y_train = y_train)
    np.savez("../preprocessed_data_reg/val_data.npz", X_val = X_val, y_val = y_val)
    np.savez("../preprocessed_data_reg/test_data.npz", X_test = X_test, y_test = y_test)

    

if __name__ == "__main__":
    load_and_prepare_reg_data("../data")