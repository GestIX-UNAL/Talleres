import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class YOLODetector:
    def __init__(
        self, 
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: 'cpu' or 'cuda'
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Performance metrics
        self.fps_history = []
        self.detection_counts = []
        self.frame_count = 0
        
        # Color palette for bounding boxes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
    def detect_frame(
        self, 
        frame: np.ndarray,
        return_annotations: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input BGR image
            return_annotations: Whether to return detection metadata
            
        Returns:
            annotated_frame: Frame with bounding boxes
            detections: Dictionary with detection data
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = results.boxes
        detections = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'objects': [],
            'counts': {},
            'inference_time': 0
        }
        
        annotated_frame = frame.copy()
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                # Count objects by class
                detections['counts'][label] = detections['counts'].get(label, 0) + 1
                
                # Store detection
                detections['objects'].append({
                    'label': label,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'area': int((x2 - x1) * (y2 - y1))
                })
                
                # Draw bounding box
                color = tuple(map(int, self.colors[cls % len(self.colors)]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label_text = f"{label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(
                    annotated_frame, 
                    label_text, 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
        
        # Calculate metrics
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        detections['inference_time'] = inference_time
        detections['fps'] = fps
        
        self.fps_history.append(fps)
        self.detection_counts.append(len(detections['objects']))
        self.frame_count += 1
        
        # Draw metrics on frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Objects: {len(detections['objects'])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated_frame, detections if return_annotations else None
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        save_json: bool = True
    ) -> List[Dict]:
        """
        Process video file with object detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            display: Whether to display frames during processing
            save_json: Whether to save detection data as JSON
            
        Returns:
            List of detection dictionaries for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                annotated_frame, detections = self.detect_frame(frame)
                all_detections.append(detections)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Save JSON
        if save_json and output_path:
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"Saved detections to {json_path}")
        
        return all_detections
    
    def detect_webcam(self, camera_id: int = 0):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID (0 for default)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        print("Starting webcam detection. Press 'q' to quit, 's' to save frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                annotated_frame, detections = self.detect_frame(frame)
                
                # Display
                cv2.imshow('YOLO Webcam Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.fps_history:
            return {}
        
        return {
            'avg_fps': np.mean(self.fps_history),
            'min_fps': np.min(self.fps_history),
            'max_fps': np.max(self.fps_history),
            'avg_detections': np.mean(self.detection_counts),
            'total_frames': self.frame_count
        }


def main():
    """Demo usage"""
    # Initialize detector
    detector = YOLODetector(
        model_path="yolov8n.pt",  # Download automatically if not present
        conf_threshold=0.5,
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # Option 1: Webcam detection
    print("Starting webcam detection...")
    detector.detect_webcam(camera_id=0)
    
    # Option 2: Video file detection
    # detector.detect_video(
    #     video_path="input_video.mp4",
    #     output_path="output_detection.mp4",
    #     display=True
    # )
    
    # Print metrics
    metrics = detector.get_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()