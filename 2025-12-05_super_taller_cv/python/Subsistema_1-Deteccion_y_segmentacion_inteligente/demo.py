import cv2
import argparse
from pathlib import Path
import json
import time

# Import detectors
from detectors.yolo_detector import YOLODetector
from detectors.mediapipe_detector import MediaPipeDetector
from detectors.clip_embeddings import CLIPEmbeddingSystem


def demo_yolo_webcam(model_path="yolov8n.pt", conf=0.5):
    """Demo YOLO detection with webcam"""
    print("\n" + "="*60)
    print("🎯 YOLO Object Detection - Webcam Demo")
    print("="*60)
    print("Press 'q' to quit, 's' to save frame")
    print()
    
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf,
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    try:
        detector.detect_webcam(camera_id=0)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    # Print metrics
    metrics = detector.get_metrics()
    print("\n📊 Performance Metrics:")
    print(f"  Average FPS: {metrics.get('avg_fps', 0):.2f}")
    print(f"  Total Frames: {metrics.get('total_frames', 0)}")
    print(f"  Avg Detections: {metrics.get('avg_detections', 0):.2f}")


def demo_yolo_image(image_path, model_path="yolov8n.pt", conf=0.5):
    """Demo YOLO detection on single image"""
    print("\n" + "="*60)
    print("🎯 YOLO Object Detection - Image Demo")
    print("="*60)
    
    # Initialize detector
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf,
        device="cpu"
    )
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}")
        return
    
    print(f"📷 Processing: {image_path}")
    
    # Detect
    annotated_frame, detections = detector.detect_frame(image)
    
    # Save result
    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated_frame)
    
    # Save JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    print(f"✅ Annotations saved to: {json_path}")
    
    # Print detection summary
    print(f"\n📋 Detection Summary:")
    print(f"  Objects detected: {len(detections.get('objects', []))}")
    print(f"  FPS: {detections.get('fps', 0):.2f}")
    
    if detections.get('counts'):
        print(f"  Object counts:")
        for obj_class, count in detections['counts'].items():
            print(f"    - {obj_class}: {count}")
    
    # Display
    cv2.imshow('YOLO Detection', annotated_frame)
    print("\n👀 Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_yolo_video(video_path, model_path="yolov8n.pt", conf=0.5):
    """Demo YOLO detection on video"""
    print("\n" + "="*60)
    print("🎯 YOLO Object Detection - Video Demo")
    print("="*60)
    
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf,
        device="cpu"
    )
    
    output_dir = Path("results/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"detected_{Path(video_path).name}"
    
    print(f"📹 Processing video: {video_path}")
    print(f"💾 Output will be saved to: {output_path}")
    
    # Process video
    all_detections = detector.detect_video(
        video_path=video_path,
        output_path=str(output_path),
        display=True,
        save_json=True
    )
    
    print(f"\n✅ Video processed successfully!")
    print(f"📊 Total frames: {len(all_detections)}")
    
    # Print metrics
    metrics = detector.get_metrics()
    print(f"\n📈 Performance Metrics:")
    print(f"  Average FPS: {metrics.get('avg_fps', 0):.2f}")
    print(f"  Min FPS: {metrics.get('min_fps', 0):.2f}")
    print(f"  Max FPS: {metrics.get('max_fps', 0):.2f}")


def demo_mediapipe_webcam():
    """Demo MediaPipe tracking with webcam"""
    print("\n" + "="*60)
    print("🤚 MediaPipe Multi-Modal Tracking - Webcam Demo")
    print("="*60)
    print("Press 'q' to quit")
    print()
    
    detector = MediaPipeDetector(
        detect_hands=True,
        detect_pose=True,
        detect_face=False,
        min_detection_confidence=0.5
    )
    
    try:
        detector.process_webcam(camera_id=0)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        detector.cleanup()
    
    # Print metrics
    metrics = detector.get_metrics()
    print("\n📊 Performance Metrics:")
    print(f"  Average FPS: {metrics.get('avg_fps', 0):.2f}")
    print(f"  Total Frames: {metrics.get('total_frames', 0)}")


def demo_clip_embeddings(image_dir):
    """Demo CLIP embeddings and visualization"""
    print("\n" + "="*60)
    print("🔍 CLIP Embeddings - Visualization Demo")
    print("="*60)
    
    # Initialize CLIP
    print("📦 Loading CLIP model...")
    clip_system = CLIPEmbeddingSystem(model_name="ViT-B/32")
    
    # Process images
    print(f"🖼️  Processing images from: {image_dir}")
    embeddings, filenames = clip_system.process_image_directory(
        image_dir=image_dir,
        output_dir="results/embeddings/"
    )
    
    print(f"✅ Processed {len(embeddings)} images")
    
    # Visualize with PCA
    print("📊 Creating PCA visualization...")
    clip_system.visualize_embeddings_pca(
        embeddings=embeddings,
        labels=filenames,
        output_path="results/embeddings/pca_visualization.png"
    )
    
    # Visualize with t-SNE
    print("📊 Creating t-SNE visualization...")
    clip_system.visualize_embeddings_tsne(
        embeddings=embeddings,
        labels=filenames,
        output_path="results/embeddings/tsne_visualization.png",
        perplexity=min(30, len(embeddings) - 1)
    )
    
    # Demo search
    print("\n🔎 Demo text-to-image search:")
    queries = [
        "a person walking",
        "a car on the road",
        "outdoor scenery"
    ]
    
    search_results = clip_system.image_search(
        query_texts=queries,
        image_embeddings=embeddings,
        image_names=filenames,
        top_k=3
    )
    
    for result in search_results:
        print(f"\n  Query: '{result['query']}'")
        for match in result['results']:
            print(f"    {match['rank']}. {match['filename']:<30} (similarity: {match['similarity']:.3f})")
    
    print("\n✅ CLIP demo completed!")


def demo_combined(mode="webcam"):
    """Demo combined detection (YOLO + MediaPipe)"""
    print("\n" + "="*60)
    print("🎯🤚 Combined Detection - YOLO + MediaPipe")
    print("="*60)
    print("Press 'q' to quit, '1' for YOLO only, '2' for MediaPipe only, '3' for both")
    print()
    
    # Initialize detectors
    yolo = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.5, device="cpu")
    mediapipe = MediaPipeDetector(detect_hands=True, detect_pose=True, detect_face=False)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    current_mode = 3  # Both
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process based on mode
            if current_mode in [1, 3]:
                frame, _ = yolo.detect_frame(frame)
            
            if current_mode in [2, 3]:
                frame, _ = mediapipe.process_frame(frame)
            
            # Display mode
            mode_text = {1: "YOLO", 2: "MediaPipe", 3: "Combined"}
            cv2.putText(
                frame,
                f"Mode: {mode_text[current_mode]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )
            
            cv2.imshow('Combined Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = 1
            elif key == ord('2'):
                current_mode = 2
            elif key == ord('3'):
                current_mode = 3
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mediapipe.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Detection & Segmentation Demo')
    parser.add_argument(
        'mode',
        choices=['yolo-webcam', 'yolo-image', 'yolo-video', 
                 'mediapipe', 'clip', 'combined', 'all'],
        help='Demo mode to run'
    )
    parser.add_argument('--input', type=str, help='Input file/directory path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("\n" + "🎯" * 30)
    print("  DETECTION & SEGMENTATION SUBSYSTEM - DEMO")
    print("🎯" * 30)
    
    # Run appropriate demo
    if args.mode == 'yolo-webcam':
        demo_yolo_webcam(model_path=args.model, conf=args.conf)
    
    elif args.mode == 'yolo-image':
        if not args.input:
            print("❌ Error: --input required for image mode")
            return
        demo_yolo_image(args.input, model_path=args.model, conf=args.conf)
    
    elif args.mode == 'yolo-video':
        if not args.input:
            print("❌ Error: --input required for video mode")
            return
        demo_yolo_video(args.input, model_path=args.model, conf=args.conf)
    
    elif args.mode == 'mediapipe':
        demo_mediapipe_webcam()
    
    elif args.mode == 'clip':
        if not args.input:
            print("❌ Error: --input directory required for CLIP mode")
            return
        demo_clip_embeddings(args.input)
    
    elif args.mode == 'combined':
        demo_combined()
    
    elif args.mode == 'all':
        print("\n🚀 Running all demos...")
        
        print("\n1️⃣  Starting YOLO Webcam Demo...")
        time.sleep(2)
        demo_yolo_webcam(model_path=args.model, conf=args.conf)
        
        print("\n2️⃣  Starting MediaPipe Demo...")
        time.sleep(2)
        demo_mediapipe_webcam()
        
        if args.input:
            print("\n3️⃣  Starting CLIP Demo...")
            time.sleep(2)
            demo_clip_embeddings(args.input)
    
    print("\n✅ Demo completed!")
    print("="*60)


if __name__ == "__main__":
    # If no arguments, show usage and run webcam demo
    import sys
    if len(sys.argv) == 1:
        print("\n🎯 Quick Start - Running YOLO Webcam Demo")
        print("For more options, run: python demo.py --help")
        print()
        demo_yolo_webcam()
    else:
        main()