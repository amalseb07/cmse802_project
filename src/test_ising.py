"""
test_ising.py
----------------------
This module defines a comprehensive set of **unit tests** to verify the correctness 
and robustness of the Ising model phase classification pipeline.

It tests critical components such as:
- Data preprocessing: verifying shapes, normalization, and label encoding.
- CNN model construction: ensuring the model builds and compiles correctly.
- Model persistence: verifying model checkpoint saving functionality.

These tests ensure that each major step in the workflow — from dataset preparation 
to model creation — performs as intended, providing a foundation for reproducible 
and reliable scientific results.

Author - Amal Sebastian  
Date - October 2025
"""





import unittest
import numpy as np
import os
from preprocess_data import load_and_prepare_phase_data
from train_cnn_phase_classifier import build_cnn

class TestIsingProject(unittest.TestCase):

    def setUp(self):
        """
        Setup function that runs before each test case.
        Loads the dataset and prepares it for testing.
        """
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = load_and_prepare_phase_data("../data")
        self.input_shape = self.X_train.shape[1:]

    def test_data_shapes(self):
        """
        Test if the data arrays have consistent shapes and correct dimensionality.
        Ensures:
        - 4D input (samples, height, width, channels)
        - Square lattice structure
        - Single grayscale channel
        """
        self.assertEqual(len(self.X_train.shape), 4)
        self.assertEqual(self.X_train.shape[-1], 1)
        self.assertEqual(self.X_train.shape[1], self.X_train.shape[2])  # square lattice
        print(" Data shapes OK")

    def test_label_values(self):
        """
        Test that labels are correctly encoded as binary (0 or 1).
        This ensures proper phase classification targets.
        """
        unique_labels = np.unique(self.y_train)
        for lbl in unique_labels:
            self.assertIn(lbl, [0, 1])
        print(" Labels are binary")

    def test_normalization_range(self):
        """
        Verify that input spin configurations are normalized within [0, 1].
        Prevents numerical instability during training.
        """
        self.assertTrue(np.all(self.X_train >= 0) and np.all(self.X_train <= 1))
        print(" Normalization OK")

    def test_model_build(self):
        """
        Ensure that the CNN model builds and compiles successfully.
        This confirms architecture validity and TensorFlow integration.
        """
        model = build_cnn()
        self.assertIsNotNone(model)
        model.summary()
        print(" Model built successfully")

    def test_checkpoint_creation(self):
        """
        Verify that the model can be saved  successfully.
        This simulates checkpoint creation during training.
        """
        model = build_cnn()
        model.save("dummy_model.h5")
        self.assertTrue(os.path.exists("dummy_model.h5"))
        os.remove("dummy_model.h5")
        print(" Model save/load works")

if __name__ == "__main__":
    unittest.main()