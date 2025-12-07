import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json

class MediaPipeDetector:
    def __init__(
        self,
        detect_hands: bool = True,
        detect_pose: bool = True,
        detect_face: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe detector
        
        Args:
            detect_hands: Enable hand detection
            detect_pose: Enable pose detection
            detect_face: Enable face mesh detection
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize detectors
        self.detectors = {}
        
        if detect_hands:
            self.mp_hands = mp.solutions.hands
            self.detectors['hands'] = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        if detect_pose:
            self.mp_pose = mp.solutions.pose
            self.detectors['pose'] = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        if detect_face:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detectors['face'] = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        # Metrics
        self.frame_count = 0
        self.fps_history = []
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with all enabled detectors
        
        Args:
            frame: Input BGR image
            
        Returns:
            annotated_frame: Frame with landmarks drawn
            results_dict: Dictionary with all detection results
        """
        start_time = time.time()
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Store results
        results_dict = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'detections': {}
        }
        
        # Process with each detector
        annotated_frame = frame.copy()
        
        # Hands detection
        if 'hands' in self.detectors:
            hands_results = self.detectors['hands'].process(image_rgb)
            results_dict['detections']['hands'] = self._process_hands(
                annotated_frame, hands_results
            )
        
        # Pose detection
        if 'pose' in self.detectors:
            pose_results = self.detectors['pose'].process(image_rgb)
            results_dict['detections']['pose'] = self._process_pose(
                annotated_frame, pose_results
            )
        
        # Face detection
        if 'face' in self.detectors:
            face_results = self.detectors['face'].process(image_rgb)
            results_dict['detections']['face'] = self._process_face(
                annotated_frame, face_results
            )
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        
        results_dict['fps'] = fps
        results_dict['inference_time'] = inference_time
        
        # Draw FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        self.frame_count += 1
        return annotated_frame, results_dict
    
    def _process_hands(self, frame: np.ndarray, results) -> Dict:
        """Process hand detection results"""
        hands_data = {
            'detected': False,
            'num_hands': 0,
            'hands': []
        }
        
        if results.multi_hand_landmarks:
            hands_data['detected'] = True
            hands_data['num_hands'] = len(results.multi_hand_landmarks)
            
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract key points
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                # Get handedness (left/right)
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                hands_data['hands'].append({
                    'hand_id': idx,
                    'label': hand_label,
                    'confidence': hand_score,
                    'landmarks': landmarks_list,
                    'gesture': self._detect_gesture(landmarks_list)
                })
                
                # Draw label
                h, w, _ = frame.shape
                wrist = hand_landmarks.landmark[0]
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(
                    frame,
                    f"{hand_label} {hand_score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )
        
        return hands_data
    
    def _process_pose(self, frame: np.ndarray, results) -> Dict:
        """Process pose detection results"""
        pose_data = {
            'detected': False,
            'landmarks': []
        }
        
        if results.pose_landmarks:
            pose_data['detected'] = True
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                pose_data['landmarks'].append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # Calculate pose metrics
            pose_data['metrics'] = self._calculate_pose_metrics(
                results.pose_landmarks.landmark
            )
        
        return pose_data
    
    def _process_face(self, frame: np.ndarray, results) -> Dict:
        """Process face mesh results"""
        face_data = {
            'detected': False,
            'num_faces': 0
        }
        
        if results.multi_face_landmarks:
            face_data['detected'] = True
            face_data['num_faces'] = len(results.multi_face_landmarks)
            
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                )
        
        return face_data
    
    def _detect_gesture(self, landmarks: List[Dict]) -> str:
        """Detect basic hand gestures"""
        if len(landmarks) < 21:
            return "unknown"
        
        # Extract fingertip and knuckle positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Extract base positions
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        
        # Count extended fingers
        extended = 0
        if index_tip['y'] < index_base['y']:
            extended += 1
        if middle_tip['y'] < middle_base['y']:
            extended += 1
        if ring_tip['y'] < ring_base['y']:
            extended += 1
        if pinky_tip['y'] < pinky_base['y']:
            extended += 1
        
        # Simple gesture recognition
        if extended == 0:
            return "fist"
        elif extended == 1:
            return "pointing"
        elif extended == 2:
            return "peace"
        elif extended >= 4:
            return "open_palm"
        else:
            return "custom"
    
    def _calculate_pose_metrics(self, landmarks) -> Dict:
        """Calculate pose-related metrics"""
        # Get shoulder and hip positions
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate shoulder width and alignment
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        hip_width = abs(right_hip.x - left_hip.x)
        
        return {
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'body_alignment': abs(shoulder_width - hip_width)
        }
    
    def process_webcam(self, camera_id: int = 0):
        """Real-time processing from webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        print("MediaPipe detection running. Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, results = self.process_frame(frame)
                
                # Display
                cv2.imshow('MediaPipe Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """Close all detectors"""
        for detector in self.detectors.values():
            detector.close()
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.fps_history:
            return {}
        
        return {
            'avg_fps': np.mean(self.fps_history),
            'min_fps': np.min(self.fps_history),
            'max_fps': np.max(self.fps_history),
            'total_frames': self.frame_count
        }


def main():
    """Demo usage"""
    # Initialize detector
    detector = MediaPipeDetector(
        detect_hands=True,
        detect_pose=True,
        detect_face=False,
        min_detection_confidence=0.5
    )
    
    # Run webcam detection
    detector.process_webcam(camera_id=0)
    
    # Print metrics
    metrics = detector.get_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()