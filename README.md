# NLP - Ghost in the Machine: AI Text Detection
**Author:** [Monishram Selvaraj]

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
* `notebooks/`: Jupyter notebook for all data analysis and training.
* `src/`: Scripts for all tasks.
* `models/`: Saved Tier C model (LoRA adapted).
* `plots/`: Confusion matrices, SHAP plot, and statistical plots.

## How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the training notebook:
   `jupyter notebook notebooks/overview.ipynb`

##Data & Embeddings

Large files (GloVe embeddings, generated datasets) are not tracked in Git.

To reproduce:
- Download GloVe from: <https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip>
- Place it at: data/dolma_300_2024_1.2M.100_combined.txt
