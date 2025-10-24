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
│── data/ # Raw and processed datasets (not tracked in Git)
│── src/ # Python scripts (Monte Carlo simulation, preprocessing, CNN training)
│── notebooks/ # Jupyter notebooks for analysis and experimentation
│── results/ # Saved plots, trained models, and evaluation metrics
│── tests/ # Unit test implementation for the project
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
- To **generate Ising data**: run the Monte Carlo simulation **ising_mc.py** script in `src/`. The generated data is preprocessed and ready for CNN taining using **preprocess_data.py** (no need to run this, goes as a header file in the train script).
- To **train the CNN**: run the training script **train_cnn_phase_classifier.py** in `src/` . This trains the model to classify between ordered and disordered phase.
- To **analyze results or visualize data**: use the notebooks in `notebooks/`. The **visalize_lattice.ipynb** is used to look at the lattice generated using  **ising_mc.py**.Training loss and validation curve in **history.ipynb**
- Final plots, chosen trained models are saved in the results/ directory
