# BERT to Decision Tree Distillation

This repository implements a knowledge distillation pipeline that transfers knowledge from a fine-tuned BERT model (teacher) to a lightweight Decision Tree (student) for sentiment classification on the IMDB dataset.

## Why Distillation?

While large language models like BERT achieve excellent performance, they:
- Are computationally expensive (inference time)
- Have high memory requirements
- Are not easily interpretable

Decision trees address these limitations by providing:
- Fast inference
- Low memory footprint
- Interpretable decisions

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/nlp-distil-bert-tree.git
cd nlp-distil-bert-tree

# Set up virtual environment and install dependencies
./scripts/setup_env.sh --dev

# Activate virtual environment
source venv/bin/activate

# Fine-tune BERT teacher model
python src/train_teacher.py

# Distill knowledge to Decision Tree
python src/distill_to_tree.py

# Evaluate both models
python src/evaluate.py
```

## Methodology

1. **Teacher model**: Fine-tune BERT on IMDB sentiment classification
2. **Feature extraction**: Convert text to TF-IDF features
3. **Distillation**: Train Decision Tree on TF-IDF features to match BERT's soft predictions
4. **Evaluation**: Compare accuracy, interpretability, and inference speed

## Configuration

Configurations are managed with Hydra. Override any parameter like this:

```bash
# Change distillation temperature
python src/distill_to_tree.py distill.T=4 distill.alpha=0.3

# Use a smaller subset of training data
python src/train_teacher.py data.dataset.n_train=1000
```

## Results

Check the TensorBoard logs for detailed metrics:

```bash
tensorboard --logdir outputs/
``` 