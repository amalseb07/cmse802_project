import numpy as np
from preprocess_data import load_and_prepare_phase_data
import tensorflow as tf 
from tensorflow.keras import layers,models,callbacks
import os


def buld_cnn():
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

    model = buld_cnn()

    X_train,X_val,X_test,y_train,y_val,y_test= load_and_prepare_phase_data("../data")

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath="best_model/best_cnn_model.h5", save_best_only=True,save_weights_only= False,monitor="val_loss", mode="min", verbose=1
    )

    
    earlystop_cb = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


        # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )

    # Save training history
    np.save("history/cnn_training_history.npy", history.history)

    print(" Training complete. Best model saved as 'best_cnn_model.h5'.")

if __name__ == "__main__":
    train_model()



    
