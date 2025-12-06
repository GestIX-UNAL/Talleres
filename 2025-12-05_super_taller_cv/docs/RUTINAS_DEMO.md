# Demo Execution Routines

## Quick Start Guide

### 1. Environment Setup

```bash
# Navigate to project directory
cd 2025-12-05_super_taller_cv

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution Routines

### Routine 1: Complete End-to-End Demo

**Purpose:** Run entire workflow from data preparation to dashboard

**Command:**
```bash
python python/training/run_complete_demo.py
```

**Output:**
- 3 trained models (Custom CNN, ResNet50, MobileNetV2)
- Training history plots
- Model comparison visualizations
- Confusion matrices and ROC curves
- JSON/CSV prediction exports
- Performance metrics reports
- Interactive dashboard (ready to launch)

**Duration:** ~10-15 minutes (GPU) / ~30-45 minutes (CPU)

**Expected Results:**
```
Phase 1: Data Preparation ✓
Phase 2: Custom CNN Training ✓
Phase 3: Fine-tuned Models Training ✓
Phase 4: Model Comparison ✓
Phase 5: Results Export ✓
Phase 6: Dashboard Launch ✓
```

---

### Routine 2: Train Custom CNN Only

**Purpose:** Quick training of custom CNN from scratch

**Python Script:**
```python
from python.training.cnn_trainer import CustomCNNTrainer, create_sample_dataset
from sklearn.model_selection import train_test_split

# Create dataset
X, y = create_sample_dataset(num_samples=2000, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize
X_train, X_test = X_train/255.0, X_test/255.0

# Train
trainer = CustomCNNTrainer()
trainer.build_model()
trainer.compile_model()
trainer.train(X_train, y_train, X_test, y_test, epochs=20)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save
trainer.save_model("results/models/custom_cnn.h5")
trainer.plot_training_history("results/metrics/training_history.png")
```

**Duration:** ~5 minutes (GPU)

---

### Routine 3: Train Fine-tuned Models

**Purpose:** Transfer learning with pre-trained models

**Python Script:**
```python
from python.training.finetuning_trainer import FineTuningTrainer
from sklearn.model_selection import train_test_split
from python.training.cnn_trainer import create_sample_dataset

# Prepare data
X, y = create_sample_dataset(num_samples=2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = X_train/255.0, X_test/255.0

# Train ResNet50
trainer_resnet = FineTuningTrainer(model_name="resnet50", num_classes=10)
trainer_resnet.build_model(freeze_base=True)
trainer_resnet.compile_model()
trainer_resnet.train(X_train, y_train, X_test, y_test, epochs=20)

# Optional: Fine-tune more layers
trainer_resnet.fine_tune_additional_layers(X_train, y_train, X_test, y_test, 
                                            num_unfreeze_layers=50, epochs=5)

# Evaluate and save
metrics = trainer_resnet.evaluate(X_test, y_test)
trainer_resnet.save_model("results/models/resnet50_finetuned.h5")
```

**Duration:** ~8 minutes (GPU)

---

### Routine 4: Compare Multiple Models

**Purpose:** Load and compare all trained models

**Python Script:**
```python
from python.training.model_comparison import ModelComparator
from sklearn.model_selection import train_test_split
from python.training.cnn_trainer import create_sample_dataset

# Prepare test data
X, y = create_sample_dataset(num_samples=2000)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
X_test = X_test / 255.0

# Initialize comparator
comparator = ModelComparator()

# Load models
comparator.add_model("results/models/custom_cnn.h5", "Custom CNN", "custom_cnn")
comparator.add_model("results/models/resnet50_finetuned.h5", "ResNet50", "resnet50")
comparator.add_model("results/models/mobilenetv2_finetuned.h5", "MobileNetV2", "mobilenetv2")

# Evaluate
results = comparator.evaluate_all_models(X_test, y_test)
print(results)

# Generate visualizations
comparator.plot_metrics_comparison("results/metrics/comparison.png")
comparator.plot_confusion_matrices("results/metrics/")
comparator.plot_roc_curves("results/metrics/roc_curves.png")
comparator.save_results_json("results/metrics/comparison.json")
```

**Duration:** ~5 minutes

---

### Routine 5: Launch Performance Dashboard

**Purpose:** Interactive real-time metrics visualization

**Python Script:**
```python
from python.dashboards.performance_dashboard import PerformanceDashboard

# Initialize dashboard
dashboard = PerformanceDashboard(
    metrics_file="results/metrics/model_comparison.json"
)

# Launch server
dashboard.run(host='127.0.0.1', port=8050, debug=True)
```

**Access:**
- Open browser: `http://localhost:8050`
- Interactive charts and metrics
- Real-time updates
- System information

**Duration:** Continuous until Ctrl+C

---

### Routine 6: Export Predictions

**Purpose:** Generate prediction outputs in multiple formats

**Python Script:**
```python
from python.utils.visualization_utils import ResultsExporter, VisualizationUtils
import numpy as np

# Make predictions (example)
X_test = np.random.rand(100, 224, 224, 3)
predictions = [(np.random.randint(0, 10), np.random.rand()) for _ in range(100)]
class_names = [f"Class_{i}" for i in range(10)]

# Export to JSON
ResultsExporter.export_predictions_json(
    predictions, class_names, 
    "results/predictions/predictions.json"
)

# Export to CSV
ResultsExporter.export_predictions_csv(
    predictions, class_names,
    "results/predictions/predictions.csv"
)

# Create visualizations
y_true = np.random.randint(0, 10, 100)
VisualizationUtils.plot_class_distribution(
    y_true,
    save_path="results/visualizations/class_distribution.png"
)
```

**Duration:** ~2 minutes

---

### Routine 7: Cross-Validation Analysis

**Purpose:** Evaluate model robustness using k-fold validation

**Python Script:**
```python
from python.training.model_comparison import cross_validate_model
from sklearn.model_selection import train_test_split
from python.training.cnn_trainer import create_sample_dataset

# Prepare data
X, y = create_sample_dataset(num_samples=2000)
X = X / 255.0

# Cross-validate
cv_scores = cross_validate_model(
    "results/models/custom_cnn.h5",
    X, y,
    cv_folds=5
)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

**Duration:** ~10-15 minutes

---

## Batch Execution Scripts

### Run All Training

**File:** `run_all_training.sh` (Linux/Mac) or `.bat` (Windows)

```bash
#!/bin/bash

echo "Starting Complete Subsystem 5 Training Pipeline..."
echo "=================================================="

# Activate environment
source venv/bin/activate

# Run demo
python python/training/run_complete_demo.py

echo "All training completed!"
```

### Generate All Visualizations

```bash
#!/bin/bash

python << 'EOF'
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path.cwd()))

from python.training.model_comparison import ModelComparator
from python.training.cnn_trainer import create_sample_dataset
from sklearn.model_selection import train_test_split

# Generate all visualizations
print("Generating visualizations...")

X, y = create_sample_dataset(2000)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
X_test = X_test / 255.0

comparator = ModelComparator()
# Add your models...
# Generate visualizations...

print("Visualizations generated!")
EOF
```

---

## Execution Timeline

### Estimated Execution Times

**GPU (NVIDIA RTX 3080):**
```
Phase 1 (Data Prep):        1 min
Phase 2 (Custom CNN):       3 min
Phase 3 (Fine-tuning):      4 min
Phase 4 (Comparison):       2 min
Phase 5 (Export):           1 min
Phase 6 (Dashboard setup):  0.5 min
─────────────────────────
Total:                      11.5 min
```

**CPU (Intel i7):**
```
Phase 1 (Data Prep):        1 min
Phase 2 (Custom CNN):       12 min
Phase 3 (Fine-tuning):      18 min
Phase 4 (Comparison):       5 min
Phase 5 (Export):           2 min
─────────────────────────
Total:                      38 min
```

---

## Output Structure

After running demo, results directory structure:

```
results/
├── models/
│   ├── custom_cnn_v1.h5           (50 MB)
│   ├── resnet50_finetuned.h5      (100 MB)
│   └── mobilenetv2_finetuned.h5   (15 MB)
├── metrics/
│   ├── model_comparison.json       (Summary results)
│   ├── training_history_custom_cnn.png
│   ├── training_history_resnet50.png
│   ├── training_history_mobilenetv2.png
│   ├── comparison_report.txt
│   └── performance.log
├── predictions/
│   ├── predictions_custom_cnn.json
│   ├── predictions_custom_cnn.csv
│   ├── predictions_resnet50.json
│   ├── predictions_resnet50.csv
│   ├── predictions_mobilenetv2.json
│   └── predictions_mobilenetv2.csv
└── visualizations/
    ├── metrics_comparison.png
    ├── class_distribution.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── metrics_summary.png
    └── confusion_matrices/
        ├── confusion_matrices.png
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```python
# Reduce batch size
trainer.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)

# Or reduce image size
input_shape = (160, 160, 3)  # Instead of (224, 224, 3)
```

### Issue: GPU Not Detected

**Check TensorFlow GPU Support:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Install CUDA:**
```bash
# For NVIDIA GPU
conda install cuda-toolkit=11.8
```

### Issue: Models Not Found

**Ensure models exist:**
```bash
ls -la results/models/
```

**Re-run training:**
```bash
python python/training/run_complete_demo.py
```

---

## Performance Optimization Tips

1. **Use GPU:** Install CUDA and cuDNN for 5-10x speedup
2. **Reduce Resolution:** Use 160x160 instead of 224x224
3. **Smaller Batch:** Reduce batch_size to 16 or 8 for OOM
4. **Mixed Precision:** Enable automatic mixed precision training
5. **Data Caching:** Pre-cache augmented data on disk

---

## Next Steps

1. ✅ Run complete demo
2. ✅ Review results and visualizations
3. ✅ Analyze model comparison report
4. ✅ Launch interactive dashboard
5. ✅ Export predictions for downstream tasks
6. 🔄 Iterate and improve models
7. 📦 Package for production deployment

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025
