"""
Utilities Module
Subsystem 5: Common utilities for data handling and visualization

Image processing, data augmentation, results export, and visualization utilities.
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import imageio
from PIL import Image, ImageDraw, ImageFont


class DataAugmentation:
    """Data augmentation utilities."""
    
    @staticmethod
    def apply_random_rotation(image, max_angle=15):
        """Apply random rotation."""
        h, w = image.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def apply_random_flip(image, prob=0.5):
        """Apply random horizontal flip."""
        if np.random.random() < prob:
            return cv2.flip(image, 1)
        return image
    
    @staticmethod
    def apply_random_brightness(image, factor=0.2):
        """Apply random brightness adjustment."""
        brightness = np.random.uniform(1-factor, 1+factor)
        return np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_random_crop(image, crop_size=0.8):
        """Apply random crop."""
        h, w = image.shape[:2]
        crop_h = int(h * crop_size)
        crop_w = int(w * crop_size)
        top = np.random.randint(0, h - crop_h)
        left = np.random.randint(0, w - crop_w)
        return image[top:top+crop_h, left:left+crop_w]
    
    @staticmethod
    def augment_batch(images, num_augmentations=5):
        """Create augmented versions of images."""
        augmented = []
        for img in images:
            augmented.append(img)
            for _ in range(num_augmentations):
                aug_img = DataAugmentation.apply_random_rotation(img)
                aug_img = DataAugmentation.apply_random_flip(aug_img)
                aug_img = DataAugmentation.apply_random_brightness(aug_img)
                augmented.append(aug_img)
        return np.array(augmented)


class ResultsExporter:
    """Export results and predictions."""
    
    @staticmethod
    def export_predictions_json(predictions, class_names, output_path):
        """Export predictions to JSON format."""
        results = {
            "timestamp": str(datetime.now()),
            "predictions": []
        }
        
        for idx, (pred, conf) in enumerate(predictions):
            results["predictions"].append({
                "index": int(idx),
                "class": class_names[pred] if isinstance(class_names, list) else f"Class {pred}",
                "class_id": int(pred),
                "confidence": float(conf)
            })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions exported to {output_path}")
    
    @staticmethod
    def export_predictions_csv(predictions, class_names, output_path):
        """Export predictions to CSV format."""
        import csv
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Class', 'Class_ID', 'Confidence'])
            
            for idx, (pred, conf) in enumerate(predictions):
                class_name = class_names[pred] if isinstance(class_names, list) else f"Class {pred}"
                writer.writerow([idx, class_name, int(pred), f"{float(conf):.6f}"])
        
        print(f"Predictions exported to {output_path}")
    
    @staticmethod
    def annotate_image(image, prediction, confidence, class_name):
        """Annotate image with prediction."""
        annotated = image.copy()
        
        # Add background for text
        text = f"{class_name}: {confidence:.2%}"
        font_scale = 0.7
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 30
        
        # Draw rectangle background
        cv2.rectangle(annotated, (text_x-5, text_y-text_size[1]-5),
                     (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated, text, (text_x, text_y), font, font_scale,
                   (0, 255, 0), thickness)
        
        return annotated
    
    @staticmethod
    def save_annotated_predictions(images, predictions, class_names, output_dir):
        """Save annotated prediction images."""
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, (img, (pred, conf)) in enumerate(zip(images, predictions)):
            class_name = class_names[pred] if isinstance(class_names, list) else f"Class {pred}"
            annotated = ResultsExporter.annotate_image(img, pred, conf, class_name)
            
            output_path = os.path.join(output_dir, f"prediction_{idx:04d}.jpg")
            cv2.imwrite(output_path, annotated)
        
        print(f"Annotated predictions saved to {output_dir}")


class VisualizationUtils:
    """Visualization utilities."""
    
    @staticmethod
    def create_comparison_grid(images1, images2, titles=None, save_path=None):
        """Create side-by-side comparison grid."""
        num_images = len(images1)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, 4*num_images))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_images):
            # First column
            axes[idx, 0].imshow(images1[idx])
            axes[idx, 0].set_title(f"{titles[idx] if titles else 'Image'} - Original" if num_images > 1 else titles[0] if titles else 'Image 1')
            axes[idx, 0].axis('off')
            
            # Second column
            axes[idx, 1].imshow(images2[idx])
            axes[idx, 1].set_title(f"{titles[idx] if titles else 'Image'} - Processed" if num_images > 1 else titles[1] if len(titles) > 1 else 'Image 2')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def create_gif(image_paths, output_path, duration=0.5):
        """Create animated GIF from images."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        images = [imageio.imread(img_path) for img_path in image_paths]
        imageio.mimsave(output_path, images, duration=duration)
        
        print(f"GIF created: {output_path}")
    
    @staticmethod
    def plot_class_distribution(y, class_names=None, save_path=None):
        """Plot class distribution."""
        from collections import Counter
        
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
        
        counts = Counter(y)
        classes = sorted(counts.keys())
        values = [counts[c] for c in classes]
        
        if class_names:
            labels = [class_names[c] for c in classes]
        else:
            labels = [f"Class {c}" for c in classes]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def create_metrics_summary_image(metrics_dict, output_path):
        """Create a visual summary image of metrics."""
        # Create image
        img = Image.new('RGB', (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Title
        title_text = "Model Performance Summary"
        draw.text((300, 20), title_text, fill=(0, 0, 0))
        
        # Metrics
        y_offset = 80
        for i, (metric_name, metric_value) in enumerate(metrics_dict.items()):
            text = f"{metric_name}: {metric_value:.4f}"
            draw.text((50, y_offset + i*50), text, fill=(50, 50, 50))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"Metrics summary image saved to {output_path}")


class PerformanceLogger:
    """Log training and inference performance metrics."""
    
    def __init__(self, log_file):
        """Initialize performance logger."""
        self.log_file = log_file
        self.logs = []
    
    def log_training(self, epoch, loss, accuracy, val_loss, val_accuracy):
        """Log training metrics."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "training",
            "epoch": epoch,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy)
        }
        self.logs.append(log_entry)
    
    def log_inference(self, model_name, inference_time, throughput, accuracy):
        """Log inference metrics."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "inference",
            "model": model_name,
            "inference_time_ms": float(inference_time),
            "throughput_images_per_sec": float(throughput),
            "accuracy": float(accuracy)
        }
        self.logs.append(log_entry)
    
    def save(self):
        """Save logs to file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Performance logs saved to {self.log_file}")
    
    def get_summary(self):
        """Get summary of logged metrics."""
        return {
            "total_entries": len(self.logs),
            "training_entries": len([l for l in self.logs if l['type'] == 'training']),
            "inference_entries": len([l for l in self.logs if l['type'] == 'inference']),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("Utilities module loaded successfully.")
