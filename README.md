# NLP Project - Track A: Multi-Label Emotion Detection

**Table of Contents**

- [Team](#team)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Running the `predict` function](#running-the-predict-function)
- [Results](#results)
  - [Output of the `predict` function](#output-of-the-predict-function)
  - [Comparison of Approaches](#comparison-of-approaches)
- [Directory Structure](#directory-structure)

## Team

| Name                                 |
| ------------------------------------ |
| Andreas A. D. Liess                  |
| Basavadeepthi Horeyala Mahadevaswamy |
| Pratiksha Jagat                      |
| Shreyas Sarathchandramohan           |
| Tushar Shandilya                     |

## Getting Started

### Requirements

- python3 (3.10.^)
- pip (23.^)

### Installation

- Packages listed in requirement.txt
- To install run in shell:

```bash

pip install -r requirements. txt

```

### Running the `predict` function

- ? Where will the input (file path) go? (cli? change a variable within main.py?)

## About

### Approaches used

- Neural Networks
  - Feed Forward NN
  - Recurring NN
- Random Forest
- Support Vector Machines
- Naive Bayes
- Transformers

### Output of the `predict` function

- A table, with each record having 5 predicted emotions corresponding to given text
- Name of best model and it's corresponding overall-accuracy of testing/validation set

## Directory Structure

- \-+`approaches-nb` (Jupyter notebooks for models)
- \-+ `approaches` (Jupyter notebooks exported to python scripts)
- \-+ `utils` (utilities and helper functions for codes)
- \-+`saved-model` (saved-models for given approaches)
- \- main.py
