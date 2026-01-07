# Breast Cancer Classification

This repository hosts the project developed for the course **Formal Tools for Bioinformatics** at the **University of Salerno**.  
The project implements a Machine Learning pipeline for the automatic classification of tumor masses (**Malignant** vs **Benign**) based on the analysis of cellular morphological data.

## Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset.  
It can be downloaded from the following link:  
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

The file `dataset.csv` is already included in the `Dataset/` directory of this repository.

## Installation
To run the project, install the required dependencies listed in the `requirements.txt` file.

## Workflow
The analysis pipeline consists of the following steps:
1. **Preprocessing:** Data cleaning and encoding of the target variable.
2. **Split:** Division of the dataset into a Training Set (80%) and a Test Set (20%).
3. **Scaling:** Feature normalization to improve model performance.
4. **Training:** Training a **Logistic Regression** model.

## Results
The model was evaluated on a separate **Test Set** to assess its generalization capability.  
A complete report including performance metrics and the **Confusion Matrix** is available in the notebook:

`Diagnostic_analysis.ipynb`

## Authors
- Gerardo Leone — https://github.com/
- Daniele Dello Russo — https://github.com/username
- Carmine Calabrese — https://github.com/
- Mario Lezzi — https://github.com/username
