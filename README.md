# CMSE802 Project: Machine Learning Phase Transitions in the 2D Ising Model

## Project Description
This project explores the 2D Ising model using Monte Carlo simulations and machine learning. The goal is to generate spin lattice configurations at different temperatures and use convolutional neural networks (CNNs) to classify ordered vs. disordered phases and estimate the approximate critical temperature.

## Objectives
- Generate Ising model lattice data at different temperatures using the Metropolis–Hastings Monte Carlo method.
- Train machine learning models (CNNs) to classify ordered and disordered phases.
- Perform regression to predict temperature from lattice configurations.
- Demonstrate how machine learning can identify a phase transition.


## Repository Structure
```text
cmse802_project/
│── data/ # Raw  datasets (not tracked in Git)
│── preprocessed_data/ #  processed datasets (not tracked in Git)
│── pictures/ #  important pictures from result and notebooks
│── predictions/ # prediction on test set after training model (not tracked in Git)
│── src/ # Python scripts (Monte Carlo simulation, preprocessing, CNN training)
│── notebooks/ # Jupyter notebooks for analysis and experimentation
│── results/ # Saved plots, trained models, and evaluation metrics
│── requirements.txt # Python dependencies
│── README.md # Project overview and documentation
│── .gitignore # Ignore unnecessary or large files

```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/amalseb07/cmse802_project.git
   cd cmse802_project

2. Create and Activate a python environment
   - micrmomaba create -n ising project
   - micromamba activate ising project
     
3. Install dependencies
   - pip install -r requirements.txt   
   
  
## Running the Code
## Phase classification
- To **generate Ising data**: run the Monte Carlo simulation **ising_mc.py** script in `src/`. 
-  To **preprocess the data**: run the script **preprocess_data.py** script in 'src/'. This shuffles and split the data into training, validation and test sets in preproccesed_data folder
- To **train the CNN**: run the training script **train_cnn_phase_classifier.py** in `src/` . This trains the model to classify between ordered and disordered phase by taking data from preproccesed_data folder. The best model is saved in directory src/best_model/best_cnn_model
-  To **predict on test set**: run the script **predict.py** in `src/` . The prediction is stored in folder predictions.
- To **analyze results or visualize data**: use the notebooks in `notebooks/`. The **visalize_lattice.ipynb** is used to look at the lattice generated using  **ising_mc.py**.Training loss and validation curve in **history.ipynb**. The **confusion_mattrix.ipynb** is used to see how well the predictions were made.
- The **confusion_mattrix.ipynb** is used to see how well the predictions were made.This is in the results/ directory

## Temperature Regression

-  To **preprocess the data**: run the script **preprocess_data_reg.py** script in 'src/'. This shuffles and split the data into training, validation and test sets in preproccesed_data folder
- To **train the CNN**: run the training script **train_cnn_phase_classifier_regression.py** in `src/` . This trains the model to predict temperature  by taking data from preproccesed_data folder. The best model is saved in directory src/best_model as best_cnn_model_reg
- To **analyze results or visualize data**: use the notebooks in `notebooks/`. Training loss and validation curve in **history_reg.ipynb**.
- The **regression_results.ipynb** is used to see how well the predictions were made.**finding_critical_temp.ipynb** combines predictions on phase and temperature to get to critical temperature. These are found at results/ directory.


