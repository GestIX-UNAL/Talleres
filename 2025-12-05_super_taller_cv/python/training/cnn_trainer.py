"""
CNN Training Module - From Scratch
Subsystem 5: Model Training & Comparison

This module implements a custom CNN trained from scratch using TensorFlow/Keras.
Includes cross-validation, metrics analysis, and visualization.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns
from pathlib import Path


class CustomCNNTrainer:
    """Custom CNN trained from scratch for image classification."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, model_name="custom_cnn"):
        """
        Initialize CNN trainer.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of classification classes
            model_name: Name for the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.metrics_log = {
            "train_accuracy": [],
            "val_accuracy": [],
            "train_loss": [],
            "val_loss": [],
            "timestamp": str(datetime.now())
        }
        
    def build_model(self):
        """Build custom CNN architecture from scratch."""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape, name='conv1_1'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', name='fc2'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model.
        
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
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
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
        """Save trained model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load pre-trained model."""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy - Custom CNN')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss - Custom CNN')
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


def create_sample_dataset(num_samples=1000, num_classes=10, img_size=224):
    """
    Create a sample dataset for demonstration.
    In production, use real dataset (CIFAR-10, ImageNet subset, etc.)
    """
    X = np.random.rand(num_samples, img_size, img_size, 3).astype('float32')
    y = keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    return X, y


if __name__ == "__main__":
    print("=" * 80)
    print("CUSTOM CNN TRAINING MODULE - SUBSYSTEM 5")
    print("=" * 80)
    
    # Create sample dataset
    print("\n[1] Creating sample dataset...")
    X, y = create_sample_dataset(num_samples=2000, num_classes=10, img_size=224)
    
    # Split data
    print("[2] Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalize
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize trainer
    print("\n[3] Initializing CNN trainer...")
    trainer = CustomCNNTrainer(input_shape=(224, 224, 3), num_classes=10, model_name="custom_cnn_v1")
    
    # Build model
    print("[4] Building custom CNN model...")
    trainer.build_model()
    trainer.compile_model(learning_rate=0.001)
    print(trainer.get_model_summary())
    
    # Train model
    print("\n[5] Training model...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
    
    # Evaluate
    print("\n[6] Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test Loss: {metrics['loss']:.4f}")
    
    # Save model
    print("\n[7] Saving model...")
    results_dir = Path(__file__).parent.parent.parent / "results" / "models"
    trainer.save_model(str(results_dir / "custom_cnn_v1.h5"))
    
    # Plot history
    print("[8] Generating training history plot...")
    trainer.plot_training_history(str(results_dir / "training_history_custom_cnn.png"))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
