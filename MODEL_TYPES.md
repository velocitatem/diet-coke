# Interpretable Student Models

This project supports multiple interpretable student models that can be used as alternatives to heavy transformer models like BERT. These models are designed to be lighter and more interpretable, while still providing good performance on NLP tasks through knowledge distillation.

## Available Model Types

The following model types are supported:

### 1. Decision Tree (`decision_tree`)

A simple, interpretable model that makes decisions based on a tree-like structure.

**Key parameters:**
- `criterion`: The function to measure the quality of a split ("gini" or "entropy" for classification)
- `max_depth`: Maximum depth of the tree
- `min_samples_split`: Minimum number of samples required to split an internal node
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node
- `max_features`: Number of features to consider when looking for the best split
- `class_weight`: Weights associated with classes

### 2. Random Forest (`random_forest`) 

An ensemble of decision trees, typically more accurate than a single decision tree.

**Key parameters:**
- All parameters from Decision Tree, plus:
- `n_estimators`: Number of trees in the forest

### 3. Linear Models (`linear`)

Logistic Regression for classification tasks, Ridge Regression for regression tasks.

**Classification parameters:**
- `C`: Inverse of regularization strength
- `penalty`: Type of penalty ("l1", "l2", "elasticnet", or "none")
- `solver`: Algorithm to use in the optimization problem
- `max_iter`: Maximum number of iterations
- `class_weight`: Weights associated with classes

**Regression parameters:**
- `alpha`: Regularization strength
- `solver`: Algorithm to use in the optimization problem
- `max_iter`: Maximum number of iterations

### 4. Support Vector Machines (`svm`)

SVC for classification, SVR for regression.

**Key parameters:**
- `C`: Regularization parameter
- `kernel`: Kernel type ("linear", "poly", "rbf", "sigmoid")
- `degree`: Degree of the polynomial kernel function (for poly kernel)
- `gamma`: Kernel coefficient
- `class_weight`: Weights associated with classes (for classification)

### 5. Naive Bayes (`naive_bayes`, classification only)

A simple probabilistic classifier based on applying Bayes' theorem.

**Key parameters:**
- `var_smoothing`: Portion of the largest variance of all features that is added to variances for calculation stability

## How to Configure

To select a specific model type, modify the `model_type` parameter in `configs/model/student.yaml`:

```yaml
student:
  model_type: "decision_tree"  # Change this to one of the options above
  # Other parameters...
```

Set the relevant parameters for your chosen model type in the same configuration file. The system will automatically use the parameters needed for the selected model.

## Model Capabilities

Different models provide different capabilities:

- **Tree-based models** (decision_tree, random_forest): Provide feature importance, tree visualization
- **Linear models**: Provide coefficient-based feature importance
- **All models**: Support prediction, probability estimation

Methods like `get_tree_text()` and `compute_path_lengths()` are only available for tree-based models. 