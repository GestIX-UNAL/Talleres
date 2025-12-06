# System Architecture

## Subsystem 5: Deep Learning Model Training & Comparison

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Raw Dataset → Preprocessing → Augmentation → Splits      │  │
│  │  (train 70% | val 15% | test 15%)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  CNN Trainer │  │ Fine-Tuning  │  │   Callbacks  │         │
│  │  - Custom    │  │  - ResNet50  │  │  - Early Stop│         │
│  │  - 4 Conv    │  │  - MobileNet │  │  - Reduce LR │         │
│  │  - BatchNorm │  │  - Layer Frz │  │  - Checkpnt  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Model Comparison                                        │  │
│  │  - Accuracy, Precision, Recall, F1-Score                │  │
│  │  - Confusion Matrices, ROC Curves                        │  │
│  │  - Cross-Validation Analysis                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION LAYER                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Dashboard     │  │  Static Plots  │  │  Export Files  │   │
│  │  - Interactive │  │  - Comparison  │  │  - JSON        │   │
│  │  - Real-time   │  │  - Matrices    │  │  - CSV         │   │
│  │  - Live Metrics│  │  - ROC Curves  │  │  - Images      │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Results/  ├─ models/ (trained .h5 files)               │  │
│  │  ├─ metrics/ (JSON, CSV comparisons)                     │  │
│  │  ├─ predictions/ (annotated images)                      │  │
│  │  └─ visualizations/ (PNG plots, GIFs)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Interaction Diagram

```
┌──────────────────────┐
│   cnn_trainer.py     │
├──────────────────────┤
│ Build CNN from scratch
│ Train on dataset
│ Save checkpoints
└──────────────────────┘
         ↓
    ┌────────────────────┐
    │ Trained Models     │
    │ (.h5 files)        │
    └────────────────────┘
         ↙        ↘        ↖
        /          \        \
       /            \        \
  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
  │ Custom CNN   │ │ finetuning.. │ │ Inference Phase │
  │ Evaluation   │ │  ResNet50    │ │ Make Predictions│
  │              │ │  MobileNetV2 │ │ Export results  │
  └──────────────┘ └──────────────┘ └─────────────────┘
         │              │                    │
         └──────────────┼────────────────────┘
                        ↓
            ┌────────────────────────────┐
            │ model_comparison.py        │
            ├────────────────────────────┤
            │ - Load all models
            │ - Compare metrics
            │ - Generate plots
            │ - Cross-validation
            └────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │ visualization_utils.py             │
        ├───────────────────────────────────┤
        │ - Export predictions (JSON/CSV)    │
        │ - Create annotated images          │
        │ - Generate visualizations          │
        │ - Log performance metrics          │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │ performance_dashboard.py           │
        ├───────────────────────────────────┤
        │ - Interactive Dash app
        │ - Real-time metrics
        │ - Multi-model comparison
        │ - System monitoring
        └───────────────────────────────────┘
```

---

## Data Flow

### Training Data Flow

```
Raw Images
    ↓
[Image Standardization]
- Resize to 224x224
- Normalize to [0, 1]
    ↓
[Data Augmentation]
- Random rotation (±15°)
- Random flip (H)
- Brightness adjustment
    ↓
[Batch Creation]
- Batch size: 32
- Shuffled training
    ↓
[Model Pipeline]
├─ Convolutional Blocks
├─ Pooling & Dropout
├─ Dense Layers
└─ Softmax Output
    ↓
[Loss Computation]
- Categorical Crossentropy
    ↓
[Backpropagation & Update]
- Adam Optimizer
- Learning rate: 0.001
    ↓
[Validation]
- Evaluate on val set
- Early stopping if needed
```

### Inference Data Flow

```
Test Images
    ↓
[Preprocessing]
- Normalize to [0, 1]
    ↓
[Model Forward Pass]
├─ Input: (224, 224, 3)
├─ Feature Extraction
├─ Classification Head
└─ Output: (num_classes,)
    ↓
[Post-processing]
- Argmax for class
- Softmax scores as confidence
    ↓
[Results]
- Class label
- Confidence score
- (Optional) Annotate image
    ↓
[Export]
- JSON predictions
- CSV results
- Annotated images
```

---

## Component Specifications

### 1. Custom CNN Architecture

```
INPUT: (224, 224, 3)
    ↓
[BLOCK 1]
Conv2D(32, 3×3) + BatchNorm → Conv2D(32, 3×3) + BatchNorm
    ↓
MaxPool(2×2) → Dropout(0.25)
    ↓
[BLOCK 2]
Conv2D(64, 3×3) + BatchNorm → Conv2D(64, 3×3) + BatchNorm
    ↓
MaxPool(2×2) → Dropout(0.25)
    ↓
[BLOCK 3]
Conv2D(128, 3×3) + BatchNorm → Conv2D(128, 3×3) + BatchNorm
    ↓
MaxPool(2×2) → Dropout(0.25)
    ↓
[BLOCK 4]
Conv2D(256, 3×3) + BatchNorm → Conv2D(256, 3×3) + BatchNorm
    ↓
MaxPool(2×2) → Dropout(0.25)
    ↓
GlobalAveragePooling2D()
    ↓
[DENSE LAYERS]
Dense(512, ReLU) + BatchNorm + Dropout(0.5)
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.5)
    ↓
OUTPUT: Dense(num_classes, Softmax)
```

**Parameters:** ~12M  
**Inference Time:** ~50ms (GPU)

### 2. Transfer Learning Strategy

**Phase 1: Base Model Frozen**
```
ResNet50/MobileNetV2 (frozen)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(512, ReLU) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Dense(num_classes, Softmax)

Learning Rate: 0.0001
Epochs: 30
```

**Phase 2: Fine-tuning (Optional)**
```
Unfreeze last 50 layers of ResNet50
    ↓
Reduce learning rate to 0.00001
    ↓
Continue training for 10 epochs
```

---

## Scalability & Performance

### Model Sizes

| Model | Parameters | Size (MB) | Inference (ms) |
|-------|-----------|-----------|----------------|
| Custom CNN | 12M | 50 | 50 |
| ResNet50 | 23M | 100 | 30 |
| MobileNetV2 | 3.5M | 15 | 20 |

### Memory Requirements

| Phase | RAM (GB) | VRAM (GB) |
|-------|----------|-----------|
| Data Loading | 2-4 | - |
| Training (batch=32) | 2-4 | 4-6 |
| Inference | 1-2 | 1-2 |
| Dashboard | 0.5-1 | - |

---

## Integration Points

### With Other Subsystems

1. **Subsystem 1 (Detection & Segmentation)**
   - YOLO outputs → Classification input
   - Segmentation masks → Label refinement

2. **Subsystem 2 (Multimodal Control)**
   - Voice commands → Model switching
   - Gestures → Real-time prediction trigger

3. **Subsystem 3 (3D Visualization)**
   - Confidence scores → Material properties
   - Predictions → Object representation

4. **Subsystem 4 (Motion Design)**
   - Predictions → Animation parameters
   - Confidence → Particle effects

---

## Error Handling

```
Try/Catch Strategy:

1. Data Loading
   - Missing files → Use sample dataset
   - Corrupted images → Skip or pad
   
2. Model Training
   - NaN loss → Reduce learning rate
   - OOM error → Reduce batch size
   
3. Evaluation
   - Empty test set → Use validation set
   - Metric errors → Default to 0
   
4. Visualization
   - Plot failures → Save as text report
   - Missing data → Use available subset
```

---

## Testing Strategy

### Unit Tests
- Model architecture validation
- Loss computation accuracy
- Metric calculation verification

### Integration Tests
- End-to-end training pipeline
- Model loading and inference
- Dashboard functionality

### Performance Tests
- Training speed benchmarks
- Inference latency measurement
- Memory consumption profiling

---

## Deployment Considerations

### Production Checklist

- [ ] Model quantization for mobile
- [ ] ONNX format export
- [ ] API wrapper for inference
- [ ] Load balancing for multiple requests
- [ ] Monitoring and logging
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Fallback models

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2025 | Initial release |
| 0.9.0 | Dec 2025 | Beta testing |

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025
