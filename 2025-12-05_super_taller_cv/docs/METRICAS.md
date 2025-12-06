# Detailed Metrics Documentation

## Performance Evaluation Framework

### Metrics Hierarchy

```
OVERALL METRICS
├── Classification Metrics
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   └── F1-Score
├── Advanced Metrics
│   ├── Confusion Matrix
│   ├── ROC-AUC
│   ├── PR-AUC
│   └── Cohen's Kappa
└── System Metrics
    ├── Inference Latency
    ├── Throughput
    ├── GPU Memory
    └── CPU Usage
```

---

## Classification Metrics

### 1. Accuracy

**Definition:** Proportion of correct predictions among total predictions.

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

**Range:** 0 to 1 (0% to 100%)  
**When to Use:** Balanced datasets  
**Interpretation:**
- 0.95 = 95% of predictions are correct
- Sensitive to class imbalance

**Example:**
```
Total predictions: 1000
Correct: 950
Accuracy = 950/1000 = 0.95 (95%)
```

---

### 2. Precision

**Definition:** Of positive predictions, how many were actually positive.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Range:** 0 to 1  
**When to Use:** High cost of false positives  
**Interpretation:**
- 0.90 = 90% of predicted positives are correct
- High precision = few false alarms

**Example (10-class classification):**
```
For Class "Cat":
Model predicts "Cat": 100 times
Actually "Cat": 90 times
Precision = 90/100 = 0.90 (90%)
```

---

### 3. Recall (Sensitivity)

**Definition:** Of actual positives, how many were predicted positive.

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Range:** 0 to 1  
**When to Use:** High cost of false negatives  
**Interpretation:**
- 0.85 = 85% of actual positives detected
- High recall = few missed cases

**Example:**
```
Actual "Cat" images: 100
Correctly detected: 85
Recall = 85/100 = 0.85 (85%)
```

---

### 4. F1-Score

**Definition:** Harmonic mean of Precision and Recall.

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Range:** 0 to 1  
**When to Use:** Balanced precision-recall tradeoff  
**Interpretation:**
- Single metric combining both precision and recall
- Useful for imbalanced datasets
- 0.87 = Good balance between precision and recall

**Example:**
```
Precision = 0.90
Recall = 0.85
F1 = 2 × (0.90 × 0.85) / (0.90 + 0.85) = 0.874
```

---

## Advanced Metrics

### 5. Confusion Matrix

**Definition:** Matrix showing true/false positives and negatives for each class.

**Example (3-class):**
```
           Predicted
        Cat  Dog  Bird
Actual Cat   89    5    6
       Dog    4   91    5
       Bird   2    3   95
```

**Interpretation:**
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Row normalization: Recall per class
- Column normalization: Precision per class

**Calculation:**
```
Cat Precision = 89/(89+4+2) = 0.956
Cat Recall = 89/(89+5+6) = 0.89
```

---

### 6. ROC Curve & AUC

**ROC (Receiver Operating Characteristic):**
- Plots True Positive Rate (Recall) vs False Positive Rate
- Measures model's ability to discriminate between classes

**AUC (Area Under Curve):**
$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR})$$

**Range:** 0 to 1
- 0.5 = Random classifier
- 1.0 = Perfect classifier
- 0.9 = Excellent discrimination

**Interpretation:**
```
AUC = 0.92 means:
- 92% probability model ranks random positive
  higher than random negative
```

---

### 7. Precision-Recall Curve

**Definition:** Plots Precision vs Recall at different thresholds.

**Use Case:** Imbalanced datasets where F1-score or Precision matters most

**PR-AUC (Area Under PR Curve):**
```
PR-AUC = 0.88
- Good discrimination in imbalanced setting
```

---

## Cross-Validation Results

### K-Fold Cross-Validation (K=5)

**Process:**
```
Dataset → Fold 1 (Test) + Folds 2-5 (Train)
       → Fold 2 (Test) + Folds 1,3-5 (Train)
       → Fold 3 (Test) + Folds 1-2,4-5 (Train)
       → Fold 4 (Test) + Folds 1-3,5 (Train)
       → Fold 5 (Test) + Folds 1-4 (Train)
```

**Example Results:**
```
Fold 1: 0.92
Fold 2: 0.89
Fold 3: 0.91
Fold 4: 0.93
Fold 5: 0.90

Mean: 0.910
Std Dev: 0.015
95% CI: [0.895, 0.925]
```

**Interpretation:**
- Mean 0.91 = Expected accuracy on unseen data
- Std Dev 0.015 = Model stability
- Tight CI = Consistent performance across folds

---

## System Performance Metrics

### Inference Metrics

#### Latency (ms)
**Definition:** Time for single prediction

```
Custom CNN: 50 ms
ResNet50: 30 ms
MobileNetV2: 20 ms
```

**Benchmark (single image, GPU):**
```
Latency = (Forward Pass Time) + (Post-processing)
```

#### Throughput (images/sec)
$$\text{Throughput} = \frac{1}{\text{Latency (sec)}}$$

**Example:**
```
Latency = 30 ms = 0.03 sec
Throughput = 1/0.03 = 33.3 images/sec
Batch of 32: 32/0.03 = 1,067 images/sec
```

#### Memory Usage

**GPU Memory (VRAM):**
```
Model Loading: 50-100 MB (ResNet50)
Batch Processing (32 images): 2-4 GB
Peak Usage: ~6 GB
```

**CPU Memory:**
```
Data Preprocessing: 1-2 GB
Model in CPU: 100-500 MB
Peak Usage: ~3 GB
```

---

## Benchmark Comparison

### Model Performance Summary

| Metric | Custom CNN | ResNet50 | MobileNetV2 |
|--------|-----------|----------|------------|
| **Accuracy** | 0.884 | 0.924 | 0.901 |
| **Precision** | 0.876 | 0.918 | 0.894 |
| **Recall** | 0.884 | 0.924 | 0.901 |
| **F1-Score** | 0.879 | 0.920 | 0.897 |
| **ROC-AUC** | 0.956 | 0.978 | 0.968 |
| **PR-AUC** | 0.823 | 0.891 | 0.854 |

---

### System Resource Comparison

| Resource | Custom CNN | ResNet50 | MobileNetV2 |
|----------|-----------|----------|------------|
| **Model Size** | 50 MB | 100 MB | 15 MB |
| **Parameters** | 12M | 23M | 3.5M |
| **Latency (GPU)** | 50 ms | 30 ms | 20 ms |
| **Throughput** | 20 img/s | 33 img/s | 50 img/s |
| **VRAM (batch=32)** | 3.5 GB | 4.5 GB | 2 GB |

---

## Per-Class Metrics

### Example: 10-Class Classification

```
             Precision  Recall  F1-Score  Support
Class 0          0.90     0.89      0.89       98
Class 1          0.92     0.91      0.91      101
Class 2          0.85     0.87      0.86       95
Class 3          0.88     0.86      0.87       99
Class 4          0.93     0.94      0.93      100
Class 5          0.87     0.88      0.87       97
Class 6          0.91     0.90      0.90      102
Class 7          0.89     0.91      0.90       98
Class 8          0.86     0.85      0.85       96
Class 9          0.90     0.89      0.89      104

Macro Avg        0.889    0.890     0.889     990
Weighted Avg     0.890    0.891     0.890     990
```

**Interpretation:**
- Class 4: Best performance (0.94 recall)
- Class 2: Needs improvement (0.85 precision)
- Macro avg: True average if class importance equal
- Weighted avg: Accounts for class imbalance

---

## Training Metrics

### Training Curve Analysis

**Healthy Training:**
```
Epoch  Train_Loss  Train_Acc  Val_Loss  Val_Acc
1      2.301       0.145      2.287     0.152
10     1.245       0.524      1.198     0.562
20     0.623       0.785      0.718     0.768
30     0.312       0.894      0.421     0.879
50     0.089       0.968      0.356     0.924
```

**Indicators:**
- ✓ Both train and val loss decreasing
- ✓ Gap between train/val widening (normal overfitting)
- ✓ Val loss plateauing (early stopping point)

**Overfitting Signs:**
```
Epoch  Train_Loss  Train_Acc  Val_Loss  Val_Acc
45     0.012       0.994      0.521     0.898  <- Gap widening
50     0.008       0.996      0.634     0.891  <- Val loss increasing
```

**Underfitting Signs:**
```
Epoch  Train_Loss  Train_Acc  Val_Loss  Val_Acc
30     1.234       0.456      1.289     0.445  <- Both high
50     0.987       0.534      1.156     0.502  <- Little improvement
```

---

## Statistical Significance

### Confidence Intervals (95%)

```
Model A Accuracy: 0.924 ± 0.021 = [0.903, 0.945]
Model B Accuracy: 0.901 ± 0.025 = [0.876, 0.926]

Models overlap → No significant difference
(at 95% confidence level)
```

### Paired t-test

```
H0: μA - μB = 0
Ha: μA - μB ≠ 0

t-statistic = 1.89
p-value = 0.071

p > 0.05 → Fail to reject H0
Conclusion: No significant difference
```

---

## Metrics Reporting Format

### JSON Export Example

```json
{
  "timestamp": "2025-12-05T14:30:00",
  "dataset": {
    "name": "CIFAR-10",
    "train_samples": 50000,
    "test_samples": 10000,
    "num_classes": 10
  },
  "models": {
    "custom_cnn": {
      "accuracy": 0.884,
      "precision": 0.876,
      "recall": 0.884,
      "f1_score": 0.879,
      "roc_auc": 0.956,
      "pr_auc": 0.823,
      "inference_latency_ms": 50,
      "throughput_img_per_sec": 20,
      "model_size_mb": 50,
      "parameters": 12000000
    },
    "resnet50_finetuned": {
      "accuracy": 0.924,
      "precision": 0.918,
      "recall": 0.924,
      "f1_score": 0.920,
      "roc_auc": 0.978,
      "pr_auc": 0.891,
      "inference_latency_ms": 30,
      "throughput_img_per_sec": 33,
      "model_size_mb": 100,
      "parameters": 23000000
    }
  },
  "cross_validation": {
    "method": "StratifiedKFold",
    "k_folds": 5,
    "scores": [0.92, 0.89, 0.91, 0.93, 0.90],
    "mean": 0.910,
    "std_dev": 0.015
  }
}
```

---

## References

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [TensorFlow Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
- [Machine Learning Metrics Guide](https://developers.google.com/machine-learning/glossary)

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025
