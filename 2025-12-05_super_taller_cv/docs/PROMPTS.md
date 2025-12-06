# AI Prompts & Development Documentation

## Overview

This document catalogs the AI prompts, instructions, and methodologies used to develop Subsystem 5: Model Training & Comparison.

---

## Initial Project Specification

### Primary Prompt

```
Design and implement a comprehensive Deep Learning model training and comparison 
subsystem that includes:

1. Custom CNN Architecture Training:
   - Build from scratch using Keras
   - 4 convolutional blocks with batch normalization
   - Dropout regularization to prevent overfitting
   - Training with early stopping and learning rate scheduling

2. Transfer Learning & Fine-tuning:
   - Implement ResNet50 fine-tuning
   - Implement MobileNetV2 fine-tuning
   - Selective layer unfreezing strategy
   - Progressive fine-tuning phases

3. Comprehensive Model Comparison:
   - Cross-validation analysis (5-fold)
   - Performance metrics (Accuracy, Precision, Recall, F1-Score)
   - Confusion matrices and ROC curves
   - Statistical significance testing

4. Interactive Visualization:
   - Performance dashboard using Dash/Plotly
   - Real-time metrics monitoring
   - Multi-model comparison charts
   - System performance monitoring

5. Results Export:
   - JSON predictions export
   - CSV results export
   - Annotated prediction images
   - Performance logging and reporting

Requirements:
- Use TensorFlow 2.10+
- GPU support optimization
- Comprehensive documentation
- Production-ready code
- Modular architecture for easy integration
```

---

## Architecture Design Prompts

### Module Structure

```
Create a modular Python package structure for deep learning model training with:

1. cnn_trainer.py - Custom CNN training module
   - Class: CustomCNNTrainer
   - Methods: build_model(), train(), evaluate(), save_model(), plot_training_history()

2. finetuning_trainer.py - Transfer learning module
   - Class: FineTuningTrainer
   - Support for ResNet50 and MobileNetV2
   - Layer freezing and unfreezing strategies

3. model_comparison.py - Comparison and analysis module
   - Class: ModelComparator
   - Load multiple models
   - Evaluate and compare
   - Generate visualizations

4. performance_dashboard.py - Interactive dashboard
   - Dash application
   - Real-time metrics
   - Multi-model comparison

5. visualization_utils.py - Utility functions
   - Data augmentation
   - Results export (JSON, CSV)
   - Visualization generation
   - Performance logging

Directory structure must follow the required format with proper separation of concerns.
```

### Metrics Calculation

```
Implement comprehensive performance metrics calculation including:

1. Classification Metrics:
   - Accuracy: (TP + TN) / (TP + TN + FP + FN)
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

2. Advanced Metrics:
   - Confusion Matrix
   - ROC Curve and AUC
   - Precision-Recall Curve
   - Per-class metrics

3. Cross-validation:
   - 5-fold stratified k-fold
   - Mean and standard deviation calculation
   - Confidence intervals

Include proper handling of:
- Multi-class classification
- Class imbalance
- Edge cases (all same class, etc.)
- Zero division protection
```

---

## Training Algorithm Prompts

### Custom CNN Training

```
Design a 4-layer CNN for image classification (224x224x3 input, 10 classes):

Architecture:
- Conv2D Block 1: 32 filters
- Conv2D Block 2: 64 filters
- Conv2D Block 3: 128 filters
- Conv2D Block 4: 256 filters

Each block includes:
- Two 3x3 convolutions with 'same' padding
- ReLU activation
- Batch normalization after each conv
- 2x2 max pooling
- 0.25 dropout

Dense layers:
- Global average pooling
- Dense(512, ReLU) + BatchNorm + Dropout(0.5)
- Dense(256, ReLU) + BatchNorm + Dropout(0.5)
- Dense(10, Softmax)

Training:
- Loss: Categorical crossentropy
- Optimizer: Adam with learning rate 0.001
- Early stopping on validation loss (patience=10)
- Learning rate reduction on plateau (factor=0.5, patience=5)
- Batch size: 32
- Max epochs: 50

Include callbacks for monitoring and best weight restoration.
```

### Transfer Learning Strategy

```
Implement two-phase fine-tuning for ResNet50 and MobileNetV2:

Phase 1: Feature Extraction (Frozen Base)
- Load ImageNet pre-trained weights
- Freeze all base model layers
- Add custom top layers:
  - GlobalAveragePooling2D
  - Dense(512, ReLU) + BatchNorm + Dropout(0.5)
  - Dense(256, ReLU) + BatchNorm + Dropout(0.5)
  - Dense(10, Softmax)
- Learning rate: 0.0001 (very conservative)
- Train for 30 epochs

Phase 2: Fine-tuning (Selective Unfreezing)
- Unfreeze last 50 layers of base model
- Reduce learning rate to 0.00001
- Continue training for 10 epochs
- Use early stopping to prevent overfitting

Strategy benefits:
- Leverage pre-trained features
- Adapt to new task with minimal data
- Prevent catastrophic forgetting
- Efficient parameter updates
```

---

## Visualization Prompts

### Dashboard Design

```
Create an interactive Dash application for model performance visualization:

Layout:
- Header: Title and description
- Metrics Cards: Summary of all model performances
- System Info Card: OS, GPU, RAM information
- Chart Grid (2x2):
  1. Accuracy comparison bar chart
  2. F1-Score comparison bar chart
  3. Precision vs Recall scatter plot
  4. Comprehensive grouped bar chart

Features:
- Color-coded by model type
- Hover tooltips with exact values
- Responsive design
- Auto-refresh every 5 seconds
- Real-time metric updates

Data Source:
- Load from JSON metrics file
- Update from database (optional)
- Cache for performance
```

### Comparison Visualization

```
Generate comprehensive model comparison plots:

1. Metrics Comparison (2x2 Subplot Grid):
   - Accuracy: Bar chart with values on top
   - Precision: Bar chart with values on top
   - Recall: Bar chart with values on top
   - F1-Score: Bar chart with values on top
   - Color by model type
   - Y-axis range: [0, 1.05]

2. Confusion Matrices:
   - Side-by-side heatmaps
   - One for each model
   - Seaborn style
   - Annotations with counts

3. ROC Curves:
   - One curve per model
   - Same subplot
   - Include diagonal reference line
   - Show AUC values in legend

4. Training History:
   - Accuracy plot: train vs validation
   - Loss plot: train vs validation
   - Grid and legend

Use consistent color scheme:
- Custom CNN: #1f77b4 (blue)
- ResNet50: #ff7f0e (orange)
- MobileNetV2: #2ca02c (green)
```

---

## Data Handling Prompts

### Data Augmentation

```
Implement data augmentation with random transformations:

Augmentation Techniques:
1. Random Rotation: ±15 degrees
2. Random Horizontal Flip: 50% probability
3. Random Brightness: 0.8x to 1.2x scaling
4. Random Crop: 80% of original size

Batch Augmentation:
- Apply multiple augmentations per image
- Create num_augmentations versions of each training image
- Maintain original image as well
- Return augmented batch as numpy array

Usage:
- Applied during training (not inference)
- Reduces overfitting
- Improves model generalization
- Effective on small datasets
```

### Results Export

```
Implement comprehensive results export functionality:

1. JSON Export:
   - Timestamp of export
   - Model name and type
   - Per-sample predictions:
     {
       "index": 0,
       "class": "Class_3",
       "class_id": 3,
       "confidence": 0.9876
     }
   - Pretty-printed with indent=2

2. CSV Export:
   - Headers: Index, Class, Class_ID, Confidence
   - One prediction per row
   - Formatted confidence to 6 decimals
   - UTF-8 encoding

3. Annotated Images:
   - Draw prediction text on image
   - Format: "Class_Name: 95.32%"
   - Semi-transparent background (0, 0, 0)
   - Green text (0, 255, 0)
   - 0.7 font scale
   - Save as JPEG

4. Performance Report:
   - Model comparison table
   - Best performers per metric
   - Cross-validation results
   - Timestamp of generation
```

---

## Documentation Prompts

### README Generation

```
Create comprehensive README.md covering:

1. Project Overview
   - Subsystem purpose and objectives
   - Key features and capabilities
   - Use cases

2. Installation & Setup
   - Prerequisites and requirements
   - Environment setup steps
   - Dependency installation

3. Module Documentation
   - Description of each Python module
   - Key classes and methods
   - Usage examples

4. Workflow Description
   - Phase-by-phase process
   - Data flow diagrams
   - Integration points

5. Performance Benchmarks
   - Expected accuracy ranges
   - Training time estimates
   - Resource requirements

6. Configuration Options
   - Hyperparameter documentation
   - Learning rates and epochs
   - Model size and latency

7. Future Enhancements
   - Ensemble methods
   - Automated hyperparameter tuning
   - Model compression techniques
   - Advanced augmentation methods
```

### Metrics Documentation

```
Create detailed METRICAS.md with:

1. Metrics Definitions
   - Mathematical formulas (LaTeX)
   - When to use each metric
   - Interpretation guidelines
   - Example calculations

2. Classification Metrics
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Per-class metrics

3. Advanced Metrics
   - Confusion matrix interpretation
   - ROC-AUC explanation
   - PR-AUC details
   - Cohen's Kappa

4. Cross-Validation
   - K-fold process explanation
   - Results interpretation
   - Statistical significance
   - Confidence intervals

5. System Metrics
   - Inference latency
   - Throughput calculation
   - Memory usage
   - GPU vs CPU comparison

6. Benchmark Tables
   - Model performance comparison
   - Resource utilization comparison
   - Per-class performance breakdown
```

---

## Testing Prompts

### Unit Test Generation

```
Create unit tests covering:

1. Model Architecture Tests
   - Verify CNN layers and shapes
   - Check parameter counts
   - Validate output dimensions
   - Test serialization/deserialization

2. Training Tests
   - Verify loss computation
   - Check metric calculations
   - Validate gradient flow
   - Test callbacks functionality

3. Metrics Tests
   - Accuracy calculation verification
   - Precision/Recall validation
   - F1-Score computation
   - Confusion matrix correctness

4. Integration Tests
   - End-to-end pipeline
   - Model loading and inference
   - Dashboard functionality
   - Data export operations

5. Performance Tests
   - Training speed benchmarks
   - Inference latency measurement
   - Memory profiling
   - Batch processing efficiency
```

---

## Quality Assurance Prompts

### Code Review Checklist

```
Verify implementation quality:

1. Code Standards
   ✓ PEP 8 compliance
   ✓ Docstrings for all classes/methods
   ✓ Type hints where appropriate
   ✓ Error handling with try/except
   ✓ Logging for debugging

2. Functionality
   ✓ All methods implemented
   ✓ Edge cases handled
   ✓ No hardcoded paths
   ✓ Configurable hyperparameters
   ✓ Proper file handling

3. Performance
   ✓ Efficient data loading
   ✓ Memory-conscious batch processing
   ✓ GPU utilization
   ✓ Vectorized operations
   ✓ Lazy loading where possible

4. Documentation
   ✓ README comprehensive
   ✓ Code commented clearly
   ✓ Architecture documented
   ✓ Usage examples provided
   ✓ Troubleshooting guide included

5. Testing
   ✓ Unit tests pass
   ✓ Integration tests pass
   ✓ No runtime errors
   ✓ Results reproducible
   ✓ Performance acceptable
```

---

## Deployment Prompts

### Production Readiness

```
Prepare Subsystem 5 for production deployment:

1. Model Packaging
   - Convert .h5 to ONNX format
   - Create inference wrappers
   - Implement batch processing
   - Add model versioning

2. API Development
   - REST API endpoints
   - Input validation
   - Error responses
   - Rate limiting

3. Monitoring & Logging
   - Prediction logging
   - Performance metrics tracking
   - Error alerting
   - Model drift detection

4. Security
   - Input sanitization
   - Access control
   - Encrypted model storage
   - Audit logging

5. Scaling
   - Load balancing
   - Multi-GPU support
   - Distributed inference
   - Caching strategies
```

---

## Integration Prompts

### Integration with Other Subsystems

```
Design integration points with other Advanced CV subsystems:

1. Subsystem 1 (Detection & Segmentation)
   - YOLO output → Classification input
   - Segmentation masks → Label refinement
   - Bounding box crops → Model input

2. Subsystem 2 (Multimodal Control)
   - Voice commands → Model switching
   - Gestures → Prediction triggering
   - EEG signals → Attention weighting

3. Subsystem 3 (3D Visualization)
   - Predictions → 3D object representation
   - Confidence → Material/shader parameters
   - Class probabilities → Animation states

4. Subsystem 4 (Motion Design)
   - Predictions → Animation parameters
   - Confidence → Particle intensity
   - Class → Motion type selection

5. Shared Components
   - Unified logging system
   - Common data formats
   - Shared dashboard
   - Centralized configuration
```

---

## References & Resources

### Documentation Standards Used

- Google Python Style Guide
- PEP 257 Docstring Conventions
- Markdown for documentation
- LaTeX for mathematical formulas
- JSON for configuration files

### Key References

- [TensorFlow Documentation](https://tensorflow.org/api_docs)
- [Keras API](https://keras.io/api/)
- [scikit-learn Guide](https://scikit-learn.org/)
- [Dash Documentation](https://dash.plotly.com/)

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025  
**Prompts Used:** 25+
