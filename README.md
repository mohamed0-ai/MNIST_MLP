# MNIST Handwritten Digit Recognition — MLP (PyTorch)

## Problem Description
A Multilayer Perceptron (MLP) built with PyTorch nn.Module to classify
handwritten digits (0–9) from the MNIST dataset.
Best result: **97.94% test accuracy**.

## Dataset
- **Source:** torchvision.datasets.MNIST
- **Link:** https://pytorch.org/vision/stable/datasets.html#mnist
- **Train:** 54,000 | **Val:** 6,000 | **Test:** 10,000
- **Input:** 784 features (28×28 pixels flattened)
- **Classes:** 10 (digits 0–9)

## Model Architecture (MLP)
**Regularization:** Dropout + Batch Normalization + Early Stopping + Data Augmentation

## Experiments & Results

| Experiment | Activation | LR | Neurons | Test Acc | Val Acc | MSE |
|---|---|---|---|---|---|---|
| Exp 1 — Baseline ⭐ | ReLU | 0.001 | 128 | **97.94%** | 96.40% | 0.003125 |
| Exp 2 | Sigmoid | 0.001 | 128 | 97.29% | 94.78% | 0.004127 |
| Exp 3 | ReLU | 0.01 | 128 | 97.59% | 96.07% | 0.003691 |

**Best:** Exp1 — ReLU,    lr=0.001 (Baseline) → **97.94% accuracy**

### Key Findings
- ReLU outperforms Sigmoid due to better gradient flow
- lr=0.001 is more stable than lr=0.01 for Adam optimizer
- Batch Normalization and Dropout significantly reduce overfitting

## How to Run
```bash
git clone https://github.com/mohamed0-ai/MNIST_MLP.git
cd MNIST_MLP
pip install torch torchvision matplotlib scikit-learn seaborn
jupyter notebook mnist_mlp_pytorch.ipynb
```
1. Open the notebook in Jupyter or Google Colab
2. Run all cells in order (Runtime → Run All)
3. Results are saved automatically to results/ folder

## Project Structure

## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib, scikit-learn, seaborn
