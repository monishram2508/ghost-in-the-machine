# NLP - Ghost in the Machine: AI Text Detection
**Author:** [Monishram Selvaraj]
**Status:** Completed (Feb 2026)

## Project Overview

This project investigates the "fingerprints" left by Large Language Models (LLMs) in generated text. We analyze the effectiveness of statistical methods versus deep learning (Transformers) in distinguishing between human and machine-authored content.

This project involves a series of subtasks that revolve around human/AI authorship detection, each of which walks through the process of training a model in a particular task. The task's central problem is not just merely about detection, but if a particularly styled prompt generates text that has statistical fingerprints, whether it can bypass detection, and how well the model can train based on this. The following is the structure of the project:

Task 0 - Dataset construction  
Task 1 - Computation of statistical values for the dataset  
Task 2 - Building and training of multiple classifiers of increasing theoretical accuracy  
Task 3 - Interpretability  
Task 4 - The Turing Test  

## Key Findings


## Repo Structure
* `notebooks/`: Jupyter notebooks for data analysis and training.
* `src/`: Helper scripts for data cleaning.
* `models/`: Saved LoRA adapters.
* `images/`: Confusion matrices and SHAP plots.

## How to Run
. Install dependencies:
   `pip install -r requirements.txt`
2. Run the training notebook:
   `jupyter notebook notebooks/Tier_C_Training.ipynb`
