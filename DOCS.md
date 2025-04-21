# DIET COKE Project Documentation

## Project Overview

DIET COKE (Decision trees Interpreting Efficient Transformers - Compression Of Knowledge Extraction) is a knowledge distillation pipeline that transfers knowledge from a fine-tuned BERT model (teacher) to lightweight, interpretable models (student) for sentiment classification on the IMDB dataset.

### Knowledge Distillation: Teaching Small Models to Think Like Big Ones

Imagine having a brilliant professor (the BERT model) who's extremely knowledgeable but speaks slowly, uses complex language, and requires an expensive translator whenever you talk to them. Now imagine being able to capture all their wisdom into a simple pocket guidebook (the decision tree) that gives you quick answers following easy-to-understand rules. That's essentially what knowledge distillation does.

In simple terms, knowledge distillation is like teaching a smaller, simpler model to mimic the behavior of a larger, more complex model. Rather than just training the small model on the original data with hard yes/no labels, we train it to match the large model's confidence scores and nuanced predictions.

For example, when classifying movie reviews:
- A large model might say: "This review is 78% positive, 22% negative"
- A small model trained on just the labels would only learn: "This is positive" 
- But through distillation, the small model learns to output similar confidence scores: "This is 75% positive, 25% negative"

This process transfers the "dark knowledge" - the subtle patterns and uncertainties that the large model has learned - to the smaller model, allowing it to perform better than if it had been trained from scratch.

### NLP Concepts in This Project

This project leverages several key NLP concepts:

1. **Transformer Models (BERT)**: We use BERT (Bidirectional Encoder Representations from Transformers) as our teacher model. BERT revolutionized NLP by using a bidirectional approach to understanding text context and employing self-attention mechanisms to capture relationships between words. It's pre-trained on massive text corpora and then fine-tuned for our sentiment classification task.

2. **Text Vectorization (TF-IDF)**: While BERT processes raw text through tokenization and embeddings, our student models need fixed-length numerical features. We use Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, which:
   - Counts word occurrences in documents (TF)
   - Downweights common words that appear across many documents (IDF)
   - Creates sparse feature vectors representing important words in each review

3. **Sentiment Analysis**: The core NLP task in this project is binary sentiment classification of movie reviews (positive/negative). This is a foundational NLP problem that tests a model's ability to understand linguistic expressions of opinion and emotion.

4. **Softmax Temperature Scaling**: In distillation, we use temperature scaling to soften probability distributions from the teacher model. Higher temperatures make the probability distribution more uniform, revealing more of the model's uncertainty and secondary predictions, which helps the student model learn more nuanced decision boundaries.

## Why Create DIET COKE?

Large language models like BERT have transformed NLP with their impressive abilities to understand and generate human language. However, they come with significant drawbacks:

1. **Computational Requirements**: BERT models typically require GPUs for efficient inference, making them unsuitable for low-resource environments or edge devices.

2. **Latency Issues**: With hundreds of millions of parameters, large models have higher inference latency, making them problematic for real-time applications.

3. **Black Box Nature**: Transformer models operate as black boxes, making their decisions difficult to interpret or explain, which is problematic for applications requiring transparency.

4. **Storage Constraints**: The sheer size of these models (usually hundreds of MBs to GBs) makes them challenging to deploy in memory-constrained environments.

DIET COKE addresses these issues by creating lightweight, interpretable alternatives that retain much of the performance of BERT while being:

- **Fast**: Decision trees and linear models execute predictions in microseconds
- **Transparent**: Their decision processes can be visualized and explained
- **Lightweight**: Requiring kilobytes rather than gigabytes of storage
- **Accessible**: Running efficiently on CPUs without specialized hardware

## Project Structure

The project is organized into the following main components:

### Source Code (`src/`)

#### Data Processing (`src/data/`)

- **`imdb_datamodule.py`**: PyTorch Lightning data module for the IMDB dataset
  - Handles data loading, preprocessing, and batching
  - Implements TF-IDF feature extraction
  - Creates train/validation/test splits

- **`transforms.py`**: Utilities for text preprocessing
  - Text tokenization functions
  - TF-IDF vectorization
  - Softmax with temperature scaling for knowledge distillation

#### Models (`src/models/`)

- **`teacher.py`**: BERT teacher model implementation
  - PyTorch Lightning module for fine-tuning BERT
  - Implements forward pass, training, validation, and test steps
  - Provides methods for extracting logits from BERT

- **`student.py`**: Student model implementations
  - Supports multiple interpretable models:
    - `DecisionTreeModel`: Classification/regression with decision trees
    - `RandomForestModel`: Ensemble of decision trees
    - `LinearModel`: Logistic regression or ridge regression
    - `SVMModel`: Support vector machines
    - `NaiveBayesModel`: Probabilistic classifier
  - Provides model creation factory method
  - Implements model saving/loading

- **`distiller.py`**: Knowledge distillation implementation
  - Extracts logits and predictions from teacher model
  - Applies temperature scaling to soften predictions
  - Fits student model to match teacher predictions
  - Evaluates fidelity metrics between teacher and student

#### Pipeline Components

- **`train_teacher.py`**: Teacher model training script
  - Fine-tunes BERT on IMDB sentiment classification
  - Implements early stopping and model checkpointing
  - Evaluates model on test set

- **`distill_to_tree.py`**: Knowledge distillation script
  - Loads teacher model and extracts predictions
  - Processes data with TF-IDF vectorization
  - Trains student model using distillation
  - Evaluates fidelity and registers models

- **`evaluate.py`**: Model evaluation script
  - Compares teacher and student model performance
  - Calculates metrics like accuracy, F1, and confusion matrix
  - Measures model agreement and fidelity

- **`benchmark_student.py`**: Student model benchmarking
  - Tests different student model configurations
  - Evaluates performance metrics
  - Helps determine optimal hyperparameters

#### Registry (`src/registry/`)

- **`model_registry.py`**: Model registry implementation
  - Tracks trained models and their performance
  - Provides versioning and model lookup
  - Manages model metadata

- **`model_registrar.py`**: Interface for registering models
  - Records model metrics and configuration
  - Exports best models for inference
  - Creates standardized model entries

- **`cli.py`**: Command-line interface for registry operations
  - Lists registered models
  - Retrieves model information
  - Manages best model tracking

#### Utilities (`src/utils/`)

- **`logging.py`**: Logging setup and utilities
  - Configures console and file logging
  - Sets up TensorBoard logging
  - Provides logging helper functions

- **`seed.py`**: Random seed management
  - Ensures reproducibility across runs
  - Sets seeds for Python, NumPy, PyTorch, and CUDA

### Scripts (`scripts/`)

- **`run_full_pipeline.sh`**: End-to-end pipeline script
  - Trains teacher model
  - Performs knowledge distillation
  - Evaluates both models
  - Creates timestamped output directories
  - Handles error checking and logging

- **`run_training.sh`**: Teacher model training script
  - Configures training parameters
  - Runs teacher training with specified settings
  - Validates model outputs

- **`run_distillation.sh`**: Distillation-only script
  - Uses pre-trained teacher model
  - Runs distillation process with specified settings
  - Validates student model outputs

- **`setup_env.sh`**: Environment setup script
  - Creates Python virtual environment
  - Installs dependencies from requirements.txt
  - Sets up project paths

- **`activate_venv.sh`**: Virtual environment activation
  - Activates the project's virtual environment
  - Sets PYTHONPATH for proper imports

### Configuration (`configs/`)

The project uses Hydra for configuration management with a hierarchical structure:

- **`config.yaml`**: Main configuration that includes other configs
- **`data.yaml`**: Dataset configuration
  - IMDB dataset parameters
  - Text processing settings
  - Train/val/test split ratios

- **`train.yaml`**: Training parameters
  - Learning rate, batch size, epochs
  - Optimization settings
  - Early stopping criteria

- **`distill.yaml`**: Distillation parameters
  - Temperature for softening predictions
  - Balancing weights for training samples
  - Knowledge distillation hyperparameters

- **`paths.yaml`**: File path configurations
  - Model checkpoint locations
  - Output directories
  - Registry paths

- **`model/`**: Model-specific configurations
  - **`teacher.yaml`**: BERT model parameters
  - **`student.yaml`**: Student model parameters
    - Decision tree parameters (depth, split criteria)
    - Random forest parameters (n_estimators, etc.)
    - Linear model parameters (regularization, solver)
    - SVM and Naive Bayes parameters

## The Distillation Process: A Deeper Look

Knowledge distillation in DIET COKE follows these key steps:

1. **Teacher Training**: We first fine-tune a BERT model on the IMDB dataset, optimizing it for sentiment classification accuracy. This creates a powerful but complex model that serves as our teacher.

2. **Feature Extraction**: Since tree-based and linear models can't process raw text like BERT can, we transform the text data into TF-IDF vectors. This creates numerical features representing the important words in each document.

3. **Soft Target Generation**: Rather than using only the hard labels (positive/negative), we extract the teacher model's output probabilities. For example, instead of just "positive," we might get "80% positive, 20% negative." These nuanced predictions contain rich information about the model's confidence and uncertainty.

4. **Temperature Scaling**: We apply temperature scaling to the teacher's logits (pre-softmax outputs) to control how "soft" the probability distributions should be:
   - A temperature of 1 gives the normal predictions
   - Higher temperatures (T > 1) make predictions more uniform, highlighting secondary patterns
   - Lower temperatures (T < 1) make predictions more confident

5. **Student Training**: The student model (e.g., decision tree) is trained to match the teacher's soft predictions rather than just the original hard labels. This transfers the nuanced decision boundaries from the complex model to the simpler one.

6. **Fidelity Evaluation**: We measure how well the student matches the teacher using metrics like agreement percentage and distribution similarity (cross-entropy).

The magic of distillation is that by training on these soft targets, the student model often performs much better than if it had been trained on just the original hard labels. It learns not just what the final decision should be, but also inherits some of the teacher's reasoning.

## Workflow

### 1. Teacher Model Training

```bash
python src/train_teacher.py
```

This script:
1. Loads and preprocesses the IMDB dataset
2. Initializes a BERT-based teacher model
3. Fine-tunes the model using PyTorch Lightning
4. Saves the best checkpoint based on validation loss
5. Evaluates the model on the test set

**Behind the Scenes**: Fine-tuning BERT involves taking a pre-trained model that already understands language patterns and adapting it to our specific task of sentiment classification. We use a specialized [CLS] token to capture the entire review's sentiment and train the model to predict positive/negative sentiment based on this representation. Early stopping prevents overfitting by monitoring validation loss.

Configuration options can be overridden:
```bash
python src/train_teacher.py train.epochs=5 train.batch_size=16
```

### 2. Knowledge Distillation

```bash
python src/distill_to_tree.py
```

This script:
1. Loads the trained teacher model
2. Extracts TF-IDF features from the text data
3. Gets teacher model predictions with temperature scaling
4. Trains a student model to match these predictions
5. Evaluates fidelity between teacher and student
6. Saves the trained student model and vectorizer

**Behind the Scenes**: The distillation process transfers knowledge by having the student model mimic the teacher's probability distributions rather than just the hard labels. Temperature scaling controls how much of the teacher's uncertainty is preserved - higher temperatures smooth out the distributions, revealing more of the subtle patterns the teacher has learned. The student model trains on TF-IDF features to match these soft targets, effectively learning a simplified approximation of the teacher's complex decision boundary.

Configuration options can be overridden:
```bash
python src/distill_to_tree.py distill.T=4 model.student.model_type=random_forest
```

### 3. Model Evaluation

```bash
python src/evaluate.py
```

This script:
1. Loads both teacher and student models
2. Evaluates them on the test set
3. Compares performance metrics
4. Generates a detailed evaluation report

### 4. Full Pipeline

```bash
./scripts/run_full_pipeline.sh
```

This script runs the entire process:
1. Trains the teacher model
2. Performs knowledge distillation
3. Evaluates both models
4. Logs results and creates timestamped outputs

## Student Model Types: Choosing the Right Tool

Each student model type has specific strengths and use cases:

### Decision Tree

```yaml
model:
  type: decision_tree
  params:
    criterion: "gini"
    max_depth: 8
    min_samples_split: 20
    min_samples_leaf: 10
    max_features: "sqrt"
    class_weight: null
```

**Why use Decision Trees?** Decision trees create a flowchart-like structure of if-then rules based on feature values. They're the most interpretable option - you can literally follow the path from root to leaf to understand exactly why a prediction was made. This makes them ideal for applications where explaining the model's decision is critical.

**Real-world analogy:** Decision trees work like a game of "20 Questions" - asking a sequence of yes/no questions about the review's words to reach a conclusion about sentiment.

### Random Forest

```yaml
model:
  type: random_forest
  params:
    n_estimators: 100
    criterion: "gini"
    max_depth: 12
    min_samples_split: 30
    min_samples_leaf: 5
    max_features: "sqrt"
    class_weight: "balanced"
```

**Why use Random Forests?** Random forests combine many decision trees, each trained on different subsets of data and features. This ensemble approach reduces overfitting and improves accuracy compared to single trees. While slightly less interpretable than a single tree, they still provide feature importance measures and retain most of the speed benefits.

**Real-world analogy:** Random forests are like asking 100 different people to play "20 Questions" with slightly different rules, then taking a vote on the final answer - more reliable than asking just one person.

### Linear Model

```yaml
model:
  type: linear
  params:
    C: 1.0
    penalty: "l2"
    solver: "saga"
    max_iter: 1000
    class_weight: null
```

**Why use Linear Models?** Linear models (like logistic regression for classification) assign weights to each feature and sum them to make predictions. They're extremely fast, scale well to large feature spaces, and provide clear feature coefficients that show each word's impact on the prediction. L1 regularization ("lasso") can create sparse models that only use a small subset of important features.

**Real-world analogy:** Linear models work like a weighted scoring system - giving positive points for words like "brilliant" and negative points for words like "terrible," then adding up the total score to determine sentiment.

## The Science Behind Temperature Scaling

Temperature scaling is a key concept in knowledge distillation, controlling how much of the teacher model's uncertainty is transferred to the student.

The formula is simple:
```
softmax(logits / T)
```

Where:
- `logits` are the raw outputs from the teacher model
- `T` is the temperature parameter (typically between 1 and 10)

**What different temperatures mean:**

- **T = 1**: Standard softmax, giving normal predictions 
- **T = 2-4**: Moderate softening, revealing secondary patterns
- **T = 5-10**: Strong softening, creating more uniform distributions 

**Concrete example:** 
If the teacher's logits for a review are [5.0, 2.0] (strongly positive):
- At T=1: probabilities ≈ [0.95, 0.05] (very confident)
- At T=4: probabilities ≈ [0.70, 0.30] (more uncertain)

This softening is crucial because it reveals the teacher model's uncertainty, forcing the student to learn not just the primary pattern but also the nuanced, secondary patterns the teacher has discovered.

## Model Registry: Tracking Experiments

The project includes a model registry for tracking model versions and performance:

- Models are automatically registered when training completes
- Metrics are stored in JSON format
- Best models can be exported for inference
- Registry can be queried to find the best model for a task

**Why have a registry?** Machine learning development involves extensive experimentation with different architectures and hyperparameters. The registry provides a systematic way to track these experiments, compare results, and identify the most promising approaches. It facilitates reproducibility and makes it easy to deploy the best-performing models.

## Performance Metrics

The distillation process is evaluated using multiple metrics:

- **Accuracy**: Standard classification accuracy 
- **F1 Score**: Harmonic mean of precision and recall
- **Agreement**: Percentage of test samples where teacher and student predictions match
- **MSE**: Mean squared error between teacher and student probabilities
- **Cross-Entropy**: Information-theoretic measure of distribution similarity

**Why we care about fidelity:** While accuracy on the test set is important, high agreement with the teacher (fidelity) indicates successful knowledge transfer. A student might achieve good accuracy by learning different patterns than the teacher, but high fidelity ensures it's actually capturing the teacher's knowledge and reasoning.

## Case Studies: Optimal Configurations

Through extensive experimentation, we've identified optimal configurations for different use cases:

### Case 1: Maximum Accuracy with Random Forest

```yaml
model:
  type: random_forest
  params:
    n_estimators: 200
    criterion: gini
    max_depth: 12
    min_samples_split: 30
    min_samples_leaf: 5
    max_features: sqrt
    class_weight: balanced
```

Random forests consistently achieve the highest accuracy among student models, sometimes coming within 2-3% of BERT's performance while being orders of magnitude faster. The balanced class weights help address any imbalance in the dataset.

### Case 2: Maximum Interpretability with Decision Tree

```yaml
model:
  type: decision_tree
  params:
    criterion: gini
    max_depth: 6
    min_samples_split: 50
    min_samples_leaf: 20
    max_features: sqrt
    class_weight: null
```

For applications where explainability is paramount, a shallower decision tree with stricter splitting criteria creates a more concise set of rules. Though accuracy is typically 5-8% lower than random forests, the ability to visualize and explain the entire decision process is invaluable for certain domains.

### Case 3: Sparse Linear Model for Efficiency

```yaml
model:
  type: linear
  params:
    C: 10.0
    penalty: l1
    solver: liblinear
    max_iter: 1000
    class_weight: null
```

L1 regularization creates sparse linear models that only use a small subset of important features. This configuration typically selects only 500-1000 words out of the entire vocabulary, creating extremely lightweight models suitable for memory-constrained environments.

## Tips for Optimal Performance

1. **Model Selection**:
   - Random Forest generally provides the best balance of performance and interpretability
   - Linear models with L1 regularization provide excellent sparsity
   - Decision trees offer the simplest interpretability

2. **Distillation Temperature**:
   - Higher temperature (T=4-10) creates softer probability distributions
   - Helps transfer nuanced knowledge from teacher to student

3. **Feature Engineering**:
   - TF-IDF parameters affect model performance significantly
   - Adjust max_features to control vocabulary size
   - Consider ngram_range to capture phrases

4. **Hyperparameter Tuning**:
   - For trees, balance max_depth with min_samples_split
   - For linear models, test different regularization strengths
   - For all models, class_weight can help with imbalanced data

## Conclusion: The Future of Model Compression

DIET COKE demonstrates how knowledge distillation can create interpretable, efficient models that retain much of the performance of large transformer models. This approach bridges the gap between cutting-edge deep learning and practical, deployable machine learning systems.

As AI deployments continue to move toward edge devices and applications requiring transparency, techniques like knowledge distillation will play an increasingly important role in making advanced NLP capabilities accessible in resource-constrained environments.

By compressing knowledge from transformers into decision trees, we get the best of both worlds: the power of deep learning with the efficiency and interpretability of traditional machine learning.