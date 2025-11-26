"""
train_cnn_phase_classifier.py
-----------------------------
This module defines and trains a Convolutional Neural Network (CNN) to classify 
2D Ising model spin configurations into ordered and disordered phases.

The model is trained using datasets obtained from preprocess_data/, which provides normalized lattice configurations labeled according 
to the critical temperature (T_c).

Key features:
- CNN architecture for binary phase classification.
- Includes dropout regularization and early stopping.
- Automatically saves the best-performing model (based on validation loss) into `best_model/best_cnn_model.h5`.
- Saves training history for later visualization and analysis into `history/cnn_training_history.npy`..

Author - Amal Sebastian
Date - October 2025

"""






import numpy as np
from preprocess_data import load_and_prepare_phase_data
import tensorflow as tf 
from tensorflow.keras import layers,models,callbacks
from tensorflow.keras.optimizers import Adam
import os


def build_cnn():
    """
    Build a simple Convolutional Neural Network (CNN) for binary phase classification.
    
    Returns
    -------
    model : tf.keras.Model
        Compiled CNN model ready for training.
    """
    
    # --------------------------------------------------------
    # (a) Define input shape and layer structure
    # --------------------------------------------------------


    input_shape=(32,32,1)
    
    model =models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(2,2),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1,activation='sigmoid')  # Binary classification (ordered/disordered)
    ])

    return model



def train_model():
    """
    Train the CNN on Ising model configurations and save the best-performing model.
    """
    


    model = build_cnn()

    #X_train,X_val,X_test,y_train,y_val,y_test= load_and_prepare_phase_data("../data")

    print("Reading in data...")
    # Load the data from npz files
    Train = np.load("../preprocessed_data/train_data.npz")
    X_train = Train['X_train']
    y_train = Train['y_train']

    Val = np.load("../preprocessed_data/val_data.npz")
    X_val = Val['X_val']
    y_val = Val['y_val']


    # checkpoint to save the best minimum  validation point
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath="best_model/best_cnn_model.h5", save_best_only=True,save_weights_only= False,monitor="val_loss", mode="min", verbose=1
    )

    # stops if the validation doesnt improve over a patience range.
   # earlystop_cb = callbacks.EarlyStopping(
   #     monitor="val_loss", patience=50, restore_best_weights=True, verbose=1
   # )

    optimizer = Adam(learning_rate=1e-5)
     
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


        # Train model
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb],
        verbose=1
    )

    # Save training history
    np.save("history/cnn_training_history.npy", history.history)

    print(" Training complete. Best model saved as 'best_cnn_model.h5'.")

if __name__ == "__main__":
    train_model()



    
