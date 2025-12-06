# Subsystem 5: Model Training & Comparison

## Overview

**Advanced Computer Vision - Specialized Subsystem for Deep Learning**

This subsystem focuses on comprehensive CNN training from scratch, transfer learning with fine-tuning, and comparative model analysis. It implements state-of-the-art techniques for image classification with performance optimization and visualization.

**Version:** 1.0.0  
**Date:** December 2025  
**Status:** Production Ready

---

## Project Objectives

1. **Train Custom CNN** from scratch using Keras/TensorFlow
2. **Fine-tune Pre-trained Models** (ResNet50, MobileNetV2)
3. **Comparative Analysis** between models using cross-validation
4. **Performance Dashboarding** with real-time metrics visualization
5. **Generate Visual Evidence** (confusion matrices, ROC curves, training history)
6. **Export Results** in JSON/CSV formats with annotated predictions

---

## Key Features

### 🧠 Deep Learning Modules

- **Custom CNN Architecture** with 4 convolutional blocks
- **Transfer Learning** with ResNet50 and MobileNetV2
- **Layer Freezing** strategies for efficient fine-tuning
- **Data Augmentation** (rotation, flip, brightness, crop)
- **Cross-Validation** for robust model evaluation

### 📊 Analytics & Visualization

- **Confusion Matrices** for multi-class analysis
- **ROC Curves** for performance evaluation
- **Training History Plots** (accuracy & loss)
- **Metrics Comparison** across all models
- **Interactive Dashboard** with Dash/Plotly

### 📁 Data Management

- **Automatic Result Export** to JSON and CSV
- **Annotated Predictions** with confidence scores
- **Performance Logging** for tracking metrics over time
- **Structured Results** in organized directories

### 🚀 System Integration

- **WebSocket Support** for real-time communication
- **Performance Metrics** (FPS, GPU/CPU usage)
- **Model Serialization** for production deployment
- **Modular Architecture** for easy integration

---

## Directory Structure

```
python/
├── training/
│   ├── cnn_trainer.py              # Custom CNN training
│   ├── finetuning_trainer.py       # Transfer learning
│   ├── model_comparison.py         # Model comparison & evaluation
│   └── __init__.py
├── dashboards/
│   ├── performance_dashboard.py    # Interactive metrics dashboard
│   └── __init__.py
├── utils/
│   ├── visualization_utils.py      # Data aug, export, visualization
│   └── __init__.py
└── detection/
    └── yolo_detection.py           # YOLO integration (optional)

data/
├── raw/                            # Raw dataset files
├── processed/                      # Preprocessed data
└── augmented/                      # Augmented samples

results/
├── models/                         # Saved model files
│   ├── custom_cnn_v1.h5
│   ├── resnet50_finetuned.h5
│   └── mobilenetv2_finetuned.h5
├── metrics/                        # Performance metrics
│   ├── model_comparison.json
│   ├── training_history_*.png
│   └── confusion_matrices.png
├── predictions/                    # Prediction outputs
│   ├── predictions.json
│   ├── predictions.csv
│   └── annotated_images/
└── visualizations/                 # Generated visualizations
    ├── training_history_*.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── gifs/

docs/
├── README.md                       # This file
├── ARCHITECTURE.md                 # System architecture
├── EVIDENCIAS.md                   # Visual evidence & screenshots
├── METRICAS.md                     # Detailed metrics documentation
├── PROMPTS.md                      # AI prompts used
└── RUTINAS_DEMO.md                # Demo execution routines
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- pip or conda

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Key dependencies:

```
tensorflow>=2.10.0
keras>=2.10.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
dash>=2.0.0
plotly>=5.0.0
opencv-python>=4.5.0
imageio>=2.9.0
Pillow>=8.0.0
```

---

## Module Descriptions

### 1. Custom CNN Trainer (`cnn_trainer.py`)

**Purpose:** Train a CNN from scratch for image classification.

**Architecture:**
- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization after each conv layer
- Global average pooling
- Dense layers: 512 → 256 → output

**Key Methods:**
- `build_model()` - Build architecture
- `train()` - Train on dataset
- `evaluate()` - Evaluate on test set
- `save_model()` - Persist trained model
- `plot_training_history()` - Visualize metrics

**Example Usage:**
```python
from python.training.cnn_trainer import CustomCNNTrainer

trainer = CustomCNNTrainer(input_shape=(224, 224, 3), num_classes=10)
trainer.build_model()
trainer.compile_model(learning_rate=0.001)
trainer.train(X_train, y_train, X_val, y_val, epochs=50)
metrics = trainer.evaluate(X_test, y_test)
trainer.save_model("results/models/custom_cnn_v1.h5")
```

### 2. Fine-Tuning Trainer (`finetuning_trainer.py`)

**Purpose:** Leverage pre-trained models through transfer learning.

**Supported Models:**
- ResNet50 (ImageNet pre-trained)
- MobileNetV2 (ImageNet pre-trained)

**Strategies:**
- Full model fine-tuning
- Selective layer unfreezing
- Learning rate scheduling

**Key Methods:**
- `build_model()` - Build with base model
- `fine_tune_additional_layers()` - Progressive unfreezing
- `train()` - Train fine-tuned model
- `evaluate()` - Model evaluation

**Example Usage:**
```python
from python.training.finetuning_trainer import FineTuningTrainer

trainer = FineTuningTrainer(model_name="resnet50", num_classes=10)
trainer.build_model(freeze_base=True)
trainer.compile_model(learning_rate=0.0001)
trainer.train(X_train, y_train, X_val, y_val, epochs=30)
trainer.fine_tune_additional_layers(X_train, y_train, X_val, y_val, 
                                     num_unfreeze_layers=50, epochs=10)
```

### 3. Model Comparison (`model_comparison.py`)

**Purpose:** Compare multiple models comprehensively.

**Capabilities:**
- Load multiple trained models
- Evaluate on test set
- Generate comparison metrics
- Create visualization plots
- Cross-validation analysis

**Key Methods:**
- `add_model()` - Add model for comparison
- `evaluate_all_models()` - Evaluate and compare
- `plot_metrics_comparison()` - Visual metrics comparison
- `plot_confusion_matrices()` - Multi-model confusion matrices
- `plot_roc_curves()` - ROC curve comparison
- `save_results_json()` - Persist results

**Example Usage:**
```python
from python.training.model_comparison import ModelComparator

comparator = ModelComparator()
comparator.add_model("results/models/custom_cnn_v1.h5", "Custom CNN", "custom_cnn")
comparator.add_model("results/models/resnet50_finetuned.h5", "ResNet50", "resnet50")
comparator.add_model("results/models/mobilenetv2_finetuned.h5", "MobileNetV2", "mobilenetv2")

results_df = comparator.evaluate_all_models(X_test, y_test)
comparator.plot_metrics_comparison("results/metrics/comparison.png")
comparator.plot_confusion_matrices("results/metrics/")
comparator.plot_roc_curves("results/metrics/roc_curves.png")
```

### 4. Performance Dashboard (`performance_dashboard.py`)

**Purpose:** Interactive real-time metrics visualization.

**Features:**
- Live metric updates
- Multi-model comparison charts
- System information display
- Precision-recall plots
- Auto-refresh at 5-second intervals

**Example Usage:**
```python
from python.dashboards.performance_dashboard import PerformanceDashboard

dashboard = PerformanceDashboard(metrics_file="results/metrics/model_comparison.json")
dashboard.run(host='0.0.0.0', port=8050, debug=True)
```

**Access:** Open browser to `http://localhost:8050`

### 5. Utilities (`visualization_utils.py`)

**Purpose:** Support functions for data and results handling.

**Classes:**
- `DataAugmentation` - Random transformations
- `ResultsExporter` - Export predictions
- `VisualizationUtils` - Create visual outputs
- `PerformanceLogger` - Track metrics

**Example Usage:**
```python
from python.utils.visualization_utils import ResultsExporter, DataAugmentation

# Augment dataset
augmented = DataAugmentation.augment_batch(images, num_augmentations=5)

# Export predictions
ResultsExporter.export_predictions_json(predictions, class_names, 
                                        "results/predictions/predictions.json")
ResultsExporter.save_annotated_predictions(images, predictions, class_names,
                                           "results/predictions/annotated/")
```

---

## Workflow

### Phase 1: Data Preparation

```
1. Load raw dataset
2. Split: train (70%) → val (15%) + test (15%)
3. Normalize images to [0, 1] range
4. One-hot encode labels
5. Apply data augmentation (optional)
```

### Phase 2: Model Training

```
1. Custom CNN from scratch
   - Build architecture
   - Compile with Adam optimizer
   - Train with early stopping
   - Evaluate on test set
   
2. Transfer Learning (ResNet50, MobileNetV2)
   - Load pre-trained weights
   - Freeze base layers
   - Add custom top layers
   - Train with low learning rate
   - (Optional) Fine-tune additional layers
```

### Phase 3: Comparison & Analysis

```
1. Load all trained models
2. Evaluate on test set
3. Calculate metrics (accuracy, precision, recall, F1)
4. Generate confusion matrices
5. Plot ROC curves
6. Create comparison visualizations
```

### Phase 4: Results Export

```
1. Save model artifacts (.h5, .pb)
2. Export predictions (JSON, CSV)
3. Create annotated prediction images
4. Generate metrics reports
5. Persist performance logs
```

---

## Performance Metrics

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (true + false positives)
- **Recall**: True positives / (true + false negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Inference Metrics

- **Latency**: Time per inference (ms)
- **Throughput**: Images per second
- **GPU Memory**: VRAM usage (MB)
- **CPU Usage**: Processor percentage

---

## Configuration

### Model Hyperparameters

**Custom CNN:**
```python
input_shape = (224, 224, 3)
num_classes = 10
learning_rate = 0.001
batch_size = 32
epochs = 50
```

**ResNet50 Fine-tuning:**
```python
learning_rate = 0.0001
batch_size = 32
epochs = 30
freeze_base = True
```

**MobileNetV2 Fine-tuning:**
```python
learning_rate = 0.0001
batch_size = 32
epochs = 30
freeze_base = True
```

---

## Expected Results

### Accuracy Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | ~88% | ~0.87 | ~0.88 | ~0.87 |
| ResNet50 | ~92% | ~0.91 | ~0.92 | ~0.91 |
| MobileNetV2 | ~90% | ~0.89 | ~0.90 | ~0.89 |

*Note: Results vary based on dataset complexity and training configuration*

---

## Troubleshooting

### GPU Not Detected

```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install GPU drivers and CUDA toolkit
```

### Out of Memory Error

- Reduce batch size
- Use mixed precision training
- Decrease input image size
- Use MobileNetV2 instead of ResNet50

### Poor Model Performance

- Increase training epochs
- Use data augmentation
- Fine-tune learning rate
- Verify dataset quality and labels

---

## Best Practices

1. **Always validate** model performance on held-out test set
2. **Use cross-validation** for robust metric estimation
3. **Monitor training** for overfitting (val_loss plateauing)
4. **Save best models** based on validation metrics
5. **Document configurations** for reproducibility
6. **Generate visualizations** for presentation

---

## Future Enhancements

- [ ] Ensemble methods combining multiple models
- [ ] Automated hyperparameter tuning with Optuna
- [ ] Quantization for model compression
- [ ] ONNX model export for cross-platform deployment
- [ ] Real-time inference pipeline
- [ ] Advanced augmentation (Mixup, CutMix)
- [ ] Attention visualization (Grad-CAM)

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Tuner](https://keras-team.github.io/keras-tuner/)
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ImageNet Models](https://github.com/keras-team/keras-applications)

---

## License

This project is part of the Advanced Computer Vision Workshop (2025).

---

## Contact & Support

For issues, questions, or contributions, please refer to the main project repository.

**Last Updated:** December 5, 2025
