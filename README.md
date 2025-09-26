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
   - micrmomaba create -n project_cmse
   - micromamba activate project_cmse
     
3. Install dependencies
   - pip install -r requirements.txt   
   
  
## Running the Code
- To **generate Ising data**: run the Monte Carlo simulation script in `src/`.
- To **train the CNN**: run the training script in `src/` .
- To **analyze results or visualize data**: use the notebooks in `notebooks/`.
- Final plots and trained models will be saved in `results/`.
