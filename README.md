# Predictive Maintenance with PyTorch

An end-to-end deep learning project that predicts **machine failures from industrial sensor data** using PyTorch and the **AI4I 2020 Predictive Maintenance Dataset**.

The project focuses on building a practical binary classification system while addressing real-world machine-learning challenges such as **class imbalance, feature scaling, model validation, threshold selection, and failure detection**.

## Project Overview

Unexpected machine failures can result in production downtime, maintenance costs, and reduced equipment efficiency.

This project uses historical machine operating conditions to predict whether a machine is likely to experience a failure.

The model is trained using:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine type

The target variable is:

* `Machine failure = 0` → No failure
* `Machine failure = 1` → Failure

Failure-related columns such as `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` were excluded from the model to avoid relying on features that directly describe specific failure modes.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, containing:

* **10,000 samples**
* **14 original features**
* Binary machine-failure target
* Industrial sensor measurements and machine information

The dataset contains a significant class imbalance:

| Class      | Samples |
| ---------- | ------: |
| No Failure |   9,661 |
| Failure    |     339 |

Because failures are relatively rare, model evaluation focuses particularly on **recall, precision, F1-score, and the confusion matrix** rather than accuracy alone.

## Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Git / GitHub

## Machine Learning Workflow

The project follows an end-to-end machine-learning workflow:

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Select relevant features
4. Encode categorical machine types
5. Remove failure-related/leakage-prone features
6. Split the data into training, validation, and test sets
7. Standardize continuous features using training-set statistics
8. Create PyTorch `TensorDataset` and `DataLoader` objects
9. Build a feed-forward neural network
10. Address class imbalance using weighted cross-entropy
11. Train the model using Adam optimization
12. Monitor validation loss and use early stopping
13. Select a classification threshold using the validation set
14. Evaluate the final model on the unseen test set
15. Perform single-machine inference

## Model Architecture

The final neural network is a fully connected feed-forward classifier:

```text
Input (8 features)
       ↓
Linear (8 → 16)
       ↓
ReLU
       ↓
Linear (16 → 8)
       ↓
ReLU
       ↓
Linear (8 → 2)
       ↓
Class logits
```

The model uses:

* **CrossEntropyLoss**
* Class weighting to address imbalance
* **Adam optimizer**
* Weight decay for regularization
* Early stopping based on validation loss

Softmax is applied only when converting model logits into class probabilities.

## Data Preprocessing

The five continuous features are standardized using the **training-set mean and standard deviation**.

Categorical machine types are one-hot encoded:

```text
Type_H
Type_L
Type_M
```

The same training statistics are used during validation, testing, and inference to prevent data leakage.

## Threshold Selection

Instead of automatically using the default classification threshold of `0.50`, different thresholds are evaluated on the validation set.

This is important because the cost of missing a real machine failure can be significantly higher than generating a false alarm.

The selected threshold is then kept fixed and applied to the test set and future inference.

## Evaluation

The model is evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1 Score

For predictive maintenance, **recall is particularly important** because false negatives represent actual machine failures that the system failed to detect.

Example confusion matrix:

```text
                 Predicted
                No Failure  Failure
Actual
No Failure         TN          FP
Failure            FN          TP
```

## Project Structure

```text
predictive-maintenance-pytorch/
│
├── data/
│   └── ai4i2020.csv
│
├── model/
│   └── best_model.pth
│
├── notebooks/
│   └── predictive_maintenance.ipynb
│
├── .gitignore
└── README.md
```

## Inference

The project also includes single-machine inference.

Given new sensor measurements such as:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine type

the trained model produces a failure probability and classifies the machine as either:

```text
Machine Failure: YES
```

or

```text
Machine Failure: NO
```

The inference pipeline applies the same preprocessing and classification threshold used during model evaluation.

## Key Learning Outcomes

This project demonstrates practical experience with:

* PyTorch neural networks
* Binary/multiclass classification concepts
* `TensorDataset` and `DataLoader`
* Training and validation workflows
* GPU/CPU device management
* Feature scaling
* One-hot encoding
* Class imbalance
* Weighted loss functions
* Early stopping
* Model checkpointing
* Probability-based classification
* Threshold optimization
* Confusion matrices
* Precision, recall, and F1-score
* Single-sample inference

## Future Improvements

Potential improvements include:

* Hyperparameter optimization
* More systematic threshold optimization
* Precision-Recall and ROC curves
* Experiment tracking
* Feature importance and model explainability
* REST API deployment
* Interactive prediction interface
* Containerization with Docker
* Production-oriented monitoring

## Author

**Saleha**

This project was developed as part of a practical deep-learning learning path focused on building real-world AI/ML systems with PyTorch.
