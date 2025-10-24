# BA Thesis: Efficient Model-Free Reinforcement Learning in Simplified Poker Environments

This repository contains the code and documentation for my Bachelor's thesis on implementing efficient model-free reinforcement learning algorithms in simplified poker environments.
The project implements Neural Fictitious Self-Play (NFSP) and augments it with a pre-trained Hand-Strength prediction to enhance learning efficiency.

# Environment Setup

Use .bat files to setup and validate environment

- `Install_environment.bat`: Installs the conda environment with all necessary dependencies from PokerRL.yml.
- `Check_Environment.bat`: Validates the conda environment to ensure pytorch and PokerRL are correctly installed.

# Project Structure

- `poker/clubs/`: Contains the poker environment implementations. (see https://github.com/MaBanser/clubs)
- `poker/envs/`: Contains the environment wrapper for PokerRL.
- `poker/agents/`: Contains the agents that act within the poker environments.
- `poker/models/`: Contains the neural network models and buffers used by the agents.

# Usage
To run training or evaluation scripts, activate the conda environment and execute the desired Python scripts located in the main project directory. Make sure to train the Hand-Strength predictor before training agents that utilize it. All configurations can be adjusted within the scripts.

# Results

![Training results NFSP vs NFSP + Hand-Strength Predictor](docs\chips_compare.png)