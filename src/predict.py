
"""
predict.py
-----------------------------

This module is used to predict the phase on the test set saved in preproccessed_data using the 
best model weights (src/best_model/best_cnn_model.h5) obtained from training the model using train_cnn_phase_classifier.py .

The predictions are done choosing different set of cuts to determine what phase it need to be labelled with. For example, a cut of 0.3
means prediction >0.3 is considered of phase 1 and <=0.3 is considered phase 0.


Key features:
- Predict the phases on the test set for different cuts
- Automatically saves the prediction result for different cuts at `predictions/../predictions/predict_cut_{cut_value}.npz`.


Author - Amal Sebastian
Date - October 2025

"""







import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




def load_and_apply_model():

    # Predicts the test data based on different cuts

    print("Loading model...")

    # Load model
    ##### CHANGE THIS TO THE MODEL YOU WANT TO USE #####
    model_path = "best_model/best_cnn_model.h5"
    model = keras.models.load_model(model_path)

    # load data set
    test = np.load("../preprocessed_data/test_data.npz")
    X_test = test['X_test']
    y_test = test['y_test']

    # different cuts to determine what should be phase 1 and what should be phase 0
    cuts=[0.3,0.4,0.5,0.6,0.7]

    for i in range(len(cuts)):

        prediction = model.predict(X_test, verbose=0) 
        prediction = (prediction >cuts[i]).astype(int)
        np.savez(f"../predictions/predict_cut_{cuts[i]}.npz",pred= prediction)
 

if __name__ == "__main__":
    load_and_apply_model()