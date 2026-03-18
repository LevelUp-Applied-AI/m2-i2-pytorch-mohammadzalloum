[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field

---

## Project Overview

This project uses PyTorch to build a neural network that predicts housing prices in Jordanian Dinars (`price_jod`) from apartment features in a tabular dataset.

The model uses these 5 input features:
- `area_sqm`
- `bedrooms`
- `floor`
- `age_years`
- `distance_to_center_km`

The target variable is:
- `price_jod`

---

## Model Architecture

The model architecture is:

`Linear(5 -> 32) -> ReLU -> Linear(32 -> 1)`

This means the model takes 5 input features, passes them through a hidden layer with 32 units and a ReLU activation, then outputs one predicted housing price.

---

## Training Configuration

The model was trained with the following settings:

- **Epochs:** 101
- **Optimizer:** Adam
- **Learning Rate:** 0.01
- **Loss Function:** MSELoss

Before training, the input features were standardized using the mean and standard deviation of each feature column.

---

## Training Outcome

The model trained successfully and generated `predictions.csv`.

The loss decreased during training:

- **Epoch 0 Loss:** `1950601088.0000`
- **Epoch 50 Loss:** `1949127040.0000`
- **Epoch 100 Loss:** `1943024384.0000`
- **Final Loss:** `1943024384.0000`

This shows that the model was learning during training.

---

## Observation

The loss decreased gradually across training rather than dropping sharply at the beginning. This suggests that the model learned steadily over the full training run.

---

## Output Files

Running the script produces:
- `predictions.csv` — contains the columns `actual` and `predicted`

---

## How to Run

```bash
python train.py
```
