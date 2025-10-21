import numpy as np
import glob
from sklearn.model_selection import train_test_split

def load_and_prepare_phase_data(data_dir="../data/",L=32,T_c=2.69):

    filepath = sorted(glob.glob(f"{data_dir}/ising_L{L}_T*.npy"))         # formated string thats why brackets
                                                                          # just list of names
    X =[]
    y=[]

    for files in filepath:
        T= float(files.split("_T")[-1].replace(".npy",""))
        configs = np.load(files)
        label = 0 if T<T_c else 1
        X.append(configs)
        y.append(np.full(configs.shape[0],label))

    X= np.concatenate(X,axis=0)    
    y= np.concatenate(y,axis=0)

    X= (X+1)/2

    X = X[...,np.newaxis]

    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)

    print(f" Loaded {X.shape[0]} samples")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class balance: Ordered={np.sum(y==0)}, Disordered={np.sum(y==1)}")


    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_prepare_phase_data("../data")