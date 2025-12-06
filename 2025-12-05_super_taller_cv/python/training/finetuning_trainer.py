"""
Fine-tuning Module - Transfer Learning
Subsystem 5: Model Training & Comparison

This module implements fine-tuning with pre-trained models (ResNet50, MobileNetV2).
Includes layer freezing, learning rate scheduling, and comparative analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from sklearn.model_selection import train_test_split
from pathlib import Path


class FineTuningTrainer:
    """Fine-tuning trainer with pre-trained models."""
    
    def __init__(self, model_name="resnet50", num_classes=10, input_shape=(224, 224, 3)):
        """
        Initialize fine-tuning trainer.
        
        Args:
            model_name: 'resnet50' or 'mobilenetv2'
            num_classes: Number of classification classes
            input_shape: Input image shape
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.base_model = None
        
    def build_model(self, freeze_base=True, num_freeze_layers=None):
        """
        Build fine-tuning model.
        
        Args:
            freeze_base: Whether to freeze base model weights
            num_freeze_layers: Number of layers to freeze (None = all)
        """
        # Load pre-trained base model
        if self.model_name == "resnet50":
            self.base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == "mobilenetv2":
            self.base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze layers
        if freeze_base:
            self.base_model.trainable = False
        else:
            # Fine-tune approach: freeze only first N layers
            if num_freeze_layers:
                for layer in self.base_model.layers[:-num_freeze_layers]:
                    layer.trainable = False
        
        # Build complete model
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """
        Train the fine-tuned model.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_additional_layers(self, X_train, y_train, X_val, y_val, 
                                     num_unfreeze_layers=50, epochs=10, batch_size=32):
        """
        Additional fine-tuning phase: unfreeze more layers.
        
        Args:
            num_unfreeze_layers: Number of layers from base model to unfreeze
        """
        # Unfreeze layers
        for layer in self.base_model.layers[-num_unfreeze_layers:]:
            layer.trainable = True
        
        # Re-compile with lower learning rate
        self.compile_model(learning_rate=0.00001)
        
        # Continue training
        print(f"Fine-tuning {num_unfreeze_layers} additional layers...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        test_loss, test_acc, test_prec, test_rec = self.model.evaluate(X_test, y_test)
        return {
            "loss": test_loss,
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec
        }
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def save_model(self, save_path):
        """Save fine-tuned model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Fine-tuned model saved to {save_path}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Model Accuracy - {self.model_name.upper()} Fine-tuned')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'Model Loss - {self.model_name.upper()} Fine-tuned')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def get_model_summary(self):
        """Get model architecture summary."""
        return self.model.summary()


if __name__ == "__main__":
    print("=" * 80)
    print("FINE-TUNING MODULE - SUBSYSTEM 5")
    print("=" * 80)
    
    # Create sample dataset
    print("\n[1] Creating sample dataset...")
    from cnn_trainer import create_sample_dataset
    X, y = create_sample_dataset(num_samples=2000, num_classes=10, img_size=224)
    
    # Split data
    print("[2] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalize
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    # Test ResNet50 fine-tuning
    print("\n[3] Training ResNet50 with fine-tuning...")
    trainer_resnet = FineTuningTrainer(model_name="resnet50", num_classes=10)
    trainer_resnet.build_model(freeze_base=True)
    trainer_resnet.compile_model(learning_rate=0.0001)
    trainer_resnet.train(X_train, y_train, X_val, y_val, epochs=15, batch_size=32)
    
    # Evaluate
    print("\n[4] Evaluating ResNet50...")
    metrics = trainer_resnet.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test Loss: {metrics['loss']:.4f}")
    
    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "models"
    trainer_resnet.save_model(str(results_dir / "resnet50_finetuned.h5"))
    trainer_resnet.plot_training_history(str(results_dir / "training_history_resnet50.png"))
    
    # Test MobileNetV2
    print("\n[5] Training MobileNetV2 with fine-tuning...")
    trainer_mobile = FineTuningTrainer(model_name="mobilenetv2", num_classes=10)
    trainer_mobile.build_model(freeze_base=True)
    trainer_mobile.compile_model(learning_rate=0.0001)
    trainer_mobile.train(X_train, y_train, X_val, y_val, epochs=15, batch_size=32)
    
    # Evaluate
    print("\n[6] Evaluating MobileNetV2...")
    metrics = trainer_mobile.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test Loss: {metrics['loss']:.4f}")
    
    # Save
    trainer_mobile.save_model(str(results_dir / "mobilenetv2_finetuned.h5"))
    trainer_mobile.plot_training_history(str(results_dir / "training_history_mobilenetv2.png"))
    
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED SUCCESSFULLY")
    print("=" * 80)
