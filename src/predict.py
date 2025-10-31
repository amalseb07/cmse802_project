import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def load_and_apply_model():
    print("Loading model...")

    # Load model
    ##### CHANGE THIS TO THE MODEL YOU WANT TO USE #####
    model_path = "best_model/best_cnn_model.h5"
    model = keras.models.load_model(model_path)

    # load data set
    test = np.load("../preprocessed_data/test_data.npz")
    X_test = test['X_test']
    y_test = test['y_test']

    cuts=[0.3,0.4,0.5,0.6,0.7]

    for i in range(len(cuts)):

        prediction = model.predict(X_test, verbose=0) 
        prediction = sigmoid(prediction)
        prediction = (prediction >cuts[i]).astype(int)
        np.savez(f"../predictions/predict_cut_{cuts[i]}.npz",pred= prediction)
 

if __name__ == "__main__":
    load_and_apply_model()