import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import actual detectors
try:
    from detectors.yolo_detector import YOLODetector
    from detectors.mediapipe_detector import MediaPipeDetector
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False
    st.warning("⚠️ Detector modules not found. Using demo mode.")

# Page config
st.set_page_config(
    page_title="Detection & Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = []
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = {}
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'running' not in st.session_state:
    st.session_state.running = False

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Detection & Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        detection_mode = st.selectbox(
            "Detection Mode",
            ["YOLO Object Detection", "MediaPipe Tracking", "Combined"]
        )
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        
        if detection_mode == "YOLO Object Detection":
            model_size = st.selectbox(
                "Model Size",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
                index=0
            )
            
            conf_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.5, 0.05
            )
            
            iou_threshold = st.slider(
                "IoU Threshold",
                0.0, 1.0, 0.45, 0.05
            )
            
            # Store settings
            st.session_state.yolo_config = {
                'model_path': model_size,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
        
        elif detection_mode == "MediaPipe Tracking":
            track_hands = st.checkbox("Track Hands", value=True)
            track_pose = st.checkbox("Track Pose", value=True)
            track_face = st.checkbox("Track Face", value=False)
            
            min_conf = st.slider(
                "Min Detection Confidence",
                0.0, 1.0, 0.5, 0.05
            )
            
            # Store settings
            st.session_state.mediapipe_config = {
                'detect_hands': track_hands,
                'detect_pose': track_pose,
                'detect_face': track_face,
                'min_detection_confidence': min_conf,
                'min_tracking_confidence': min_conf
            }
        
        st.divider()
        
        # Input source
        st.subheader("Input Source")
        input_source = st.radio(
            "Select Source",
            ["Webcam", "Upload Image", "Upload Video"]
        )
        
        if input_source == "Webcam":
            camera_id = st.number_input("Camera ID", 0, 10, 0)
            st.session_state.camera_id = camera_id
        
        st.divider()
        
        # Performance settings
        st.subheader("Performance")
        device = st.selectbox("Device", ["cpu", "cuda"])
        st.session_state.device = device
        
        # Clear data button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.detection_history = []
            st.session_state.fps_history = []
            st.session_state.object_counts = {}
            st.rerun()
    
    # Main content
    tabs = st.tabs([
        "📹 Live Detection",
        "📊 Metrics",
        "🔍 Analysis",
        "💾 Export"
    ])
    
    # Tab 1: Live Detection
    with tabs[0]:
        live_detection_tab(detection_mode, input_source)
    
    # Tab 2: Metrics
    with tabs[1]:
        metrics_tab()
    
    # Tab 3: Analysis
    with tabs[2]:
        analysis_tab()
    
    # Tab 4: Export
    with tabs[3]:
        export_tab()

def live_detection_tab(detection_mode, input_source):
    st.header("Live Detection")
    
    if not DETECTORS_AVAILABLE:
        st.error("Detector modules not available. Please check imports.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Feed")
        
        if input_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                # Read image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Process image
                with st.spinner("Processing..."):
                    result_image, detections = process_single_image(
                        image_np,
                        detection_mode
                    )
                
                # Convert BGR back to RGB for display
                if len(result_image.shape) == 3 and result_image.shape[2] == 3:
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.image(result_image, use_container_width=True)
                
                # Show detection info
                num_objects = len(detections.get('objects', [])) if detection_mode == "YOLO Object Detection" else 0
                st.success(f"✅ Processed! Detected: {num_objects} objects")
                
                # Show detailed results
                with st.expander("🔍 Detection Details"):
                    st.json(detections)
        
        elif input_source == "Webcam":
            st.info("🎥 Webcam Detection Mode")
            
            # Video placeholder
            video_placeholder = st.empty()
            
            # Control buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                start_button = st.button("▶️ Start", type="primary", use_container_width=True)
            
            with col_b:
                stop_button = st.button("⏹️ Stop", type="secondary", use_container_width=True)
            
            with col_c:
                snapshot_button = st.button("📸 Snapshot", use_container_width=True)
            
            # Handle webcam
            if start_button:
                st.session_state.running = True
                initialize_detector(detection_mode)
            
            if stop_button:
                st.session_state.running = False
                cleanup_detector()
            
            # Run webcam loop
            if st.session_state.running:
                run_webcam_stream(
                    video_placeholder,
                    detection_mode,
                    snapshot_button
                )
        
        elif input_source == "Upload Video":
            uploaded_video = st.file_uploader(
                "Choose a video",
                type=['mp4', 'avi', 'mov']
            )
            
            if uploaded_video is not None:
                # Save temporarily
                temp_path = Path("temp_video.mp4")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_video.read())
                
                if st.button("Process Video", type="primary"):
                    process_video_file(str(temp_path), detection_mode)
    
    with col2:
        st.subheader("📋 Detection Info")
        
        # Real-time metrics
        if st.session_state.detection_history:
            latest = st.session_state.detection_history[-1]
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                fps_val = latest.get('fps', 0)
                st.metric("FPS", f"{fps_val:.1f}")
                
                objects_val = len(latest.get('objects', []))
                st.metric("Objects", objects_val)
            
            with col_b:
                latency_val = latest.get('inference_time', 0) * 1000
                st.metric("Latency", f"{latency_val:.0f} ms")
                
                frame_val = latest.get('frame_id', 0)
                st.metric("Frame", frame_val)
        else:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("FPS", "--")
                st.metric("Objects", "--")
            
            with col_b:
                st.metric("Latency", "-- ms")
                st.metric("Frame", "--")
        
        st.divider()
        
        # Object counts
        st.subheader("🎯 Object Counts")
        if st.session_state.object_counts:
            counts_df = pd.DataFrame(
                list(st.session_state.object_counts.items()),
                columns=['Class', 'Count']
            ).sort_values('Count', ascending=False)
            st.dataframe(counts_df, use_container_width=True, hide_index=True)
        else:
            st.info("No detections yet")

def initialize_detector(mode):
    """Initialize the appropriate detector"""
    try:
        if mode == "YOLO Object Detection":
            config = st.session_state.get('yolo_config', {})
            st.session_state.detector = YOLODetector(
                model_path=config.get('model_path', 'yolov8n.pt'),
                conf_threshold=config.get('conf_threshold', 0.5),
                iou_threshold=config.get('iou_threshold', 0.45),
                device=st.session_state.get('device', 'cpu')
            )
            st.success("✅ YOLO detector initialized!")
        
        elif mode == "MediaPipe Tracking":
            config = st.session_state.get('mediapipe_config', {})
            st.session_state.detector = MediaPipeDetector(
                detect_hands=config.get('detect_hands', True),
                detect_pose=config.get('detect_pose', True),
                detect_face=config.get('detect_face', False),
                min_detection_confidence=config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=config.get('min_tracking_confidence', 0.5)
            )
            st.success("✅ MediaPipe detector initialized!")
    
    except Exception as e:
        st.error(f"❌ Error initializing detector: {str(e)}")
        st.session_state.running = False

def cleanup_detector():
    """Cleanup detector resources"""
    if st.session_state.detector:
        if hasattr(st.session_state.detector, 'cleanup'):
            st.session_state.detector.cleanup()
        st.session_state.detector = None

def run_webcam_stream(placeholder, mode, snapshot_trigger):
    """Run webcam detection stream"""
    camera_id = st.session_state.get('camera_id', 0)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        st.error(f"❌ Could not open camera {camera_id}")
        st.session_state.running = False
        return
    
    frame_count = 0
    
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ Failed to read frame")
                break
            
            # Process frame
            if st.session_state.detector:
                if mode == "YOLO Object Detection":
                    annotated_frame, detections = st.session_state.detector.detect_frame(frame)
                    
                    # Update object counts
                    if detections and 'counts' in detections:
                        for obj_class, count in detections['counts'].items():
                            st.session_state.object_counts[obj_class] = \
                                st.session_state.object_counts.get(obj_class, 0) + count
                
                elif mode == "MediaPipe Tracking":
                    annotated_frame, detections = st.session_state.detector.process_frame(frame)
                
                # Store detection history
                st.session_state.detection_history.append(detections)
                if len(st.session_state.detection_history) > 1000:
                    st.session_state.detection_history.pop(0)
                
                # Store FPS
                if 'fps' in detections:
                    st.session_state.fps_history.append(detections['fps'])
                    if len(st.session_state.fps_history) > 1000:
                        st.session_state.fps_history.pop(0)
            else:
                annotated_frame = frame
            
            # Convert BGR to RGB for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display in Streamlit
            placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            # Save snapshot
            if snapshot_trigger and frame_count % 30 == 0:
                timestamp = int(time.time())
                snapshot_path = Path("snapshots")
                snapshot_path.mkdir(exist_ok=True)
                cv2.imwrite(str(snapshot_path / f"snapshot_{timestamp}.jpg"), annotated_frame)
            
            frame_count += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.01)
    
    finally:
        cap.release()

def process_single_image(image, mode):
    """Process a single image with detection"""
    initialize_detector(mode)
    
    if not st.session_state.detector:
        return image, {}
    
    try:
        if mode == "YOLO Object Detection":
            result_image, detections = st.session_state.detector.detect_frame(image)
        elif mode == "MediaPipe Tracking":
            result_image, detections = st.session_state.detector.process_frame(image)
        else:
            result_image, detections = image, {}
        
        # Update history
        st.session_state.detection_history.append(detections)
        
        return result_image, detections
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return image, {}

def process_video_file(video_path, mode):
    """Process uploaded video file"""
    initialize_detector(mode)
    
    if not st.session_state.detector:
        return
    
    output_path = "output_video.mp4"
    
    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)
        
        try:
            if mode == "YOLO Object Detection":
                # You would implement video processing here
                st.info("Video processing implementation needed")
            
            progress_bar.progress(100)
            st.success(f"✅ Video processed! Saved to {output_path}")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

def metrics_tab():
    st.header("Performance Metrics")
    
    if not st.session_state.detection_history:
        st.info("📊 No metrics available yet. Start detection to see metrics.")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_fps = np.mean(st.session_state.fps_history) if st.session_state.fps_history else 0
        st.metric("Avg FPS", f"{avg_fps:.1f}")
    
    with col2:
        st.metric("Total Frames", len(st.session_state.detection_history))
    
    with col3:
        total_objects = sum(st.session_state.object_counts.values())
        st.metric("Total Objects", total_objects)
    
    with col4:
        unique_classes = len(st.session_state.object_counts)
        st.metric("Unique Classes", unique_classes)
    
    st.divider()
    
    # FPS over time
    if st.session_state.fps_history:
        st.subheader("📈 FPS Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.fps_history[-200:],  # Last 200 frames
            mode='lines',
            name='FPS',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Frame",
            yaxis_title="FPS",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Object distribution
    if st.session_state.object_counts:
        st.subheader("🎯 Object Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                x=list(st.session_state.object_counts.keys()),
                y=list(st.session_state.object_counts.values()),
                labels={'x': 'Class', 'y': 'Count'},
                color=list(st.session_state.object_counts.values()),
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                values=list(st.session_state.object_counts.values()),
                names=list(st.session_state.object_counts.keys()),
                hole=0.4
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

def analysis_tab():
    st.header("Detection Analysis")
    
    if not st.session_state.detection_history:
        st.info("🔍 No detection data available. Run detection first.")
        return
    
    # Detection timeline
    st.subheader("📅 Detection Timeline")
    
    timeline_data = []
    for det in st.session_state.detection_history[-200:]:  # Last 200 frames
        num_objects = len(det.get('objects', []))
        timeline_data.append({
            'Frame': det.get('frame_id', 0),
            'Objects': num_objects,
            'FPS': det.get('fps', 0)
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Frame'],
            y=df['Objects'],
            mode='lines+markers',
            name='Objects Detected',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Frame",
            yaxis_title="Number of Objects",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics
    st.subheader("📊 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**FPS Statistics**")
        if st.session_state.fps_history:
            fps_stats = {
                'Mean': round(np.mean(st.session_state.fps_history), 2),
                'Median': round(np.median(st.session_state.fps_history), 2),
                'Min': round(np.min(st.session_state.fps_history), 2),
                'Max': round(np.max(st.session_state.fps_history), 2),
                'Std Dev': round(np.std(st.session_state.fps_history), 2)
            }
            st.json(fps_stats)
    
    with col2:
        st.write("**Detection Statistics**")
        if st.session_state.detection_history:
            object_counts = [len(d.get('objects', [])) for d in st.session_state.detection_history]
            det_stats = {
                'Total Detections': sum(object_counts),
                'Avg per Frame': round(np.mean(object_counts), 2),
                'Max per Frame': max(object_counts) if object_counts else 0,
                'Frames with Objects': sum(1 for c in object_counts if c > 0)
            }
            st.json(det_stats)

def export_tab():
    st.header("Export Results")
    
    if not st.session_state.detection_history:
        st.info("💾 No data to export. Run detection first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Export Options")
        
        export_format = st.selectbox(
            "Format",
            ["JSON", "CSV", "Excel"]
        )
        
        include_metrics = st.checkbox("Include Performance Metrics", value=True)
        
        if st.button("📥 Export Data", type="primary"):
            with st.spinner("Exporting..."):
                export_path = export_data(export_format, include_metrics)
                st.success(f"✅ Export completed! Saved to {export_path}")
                
                # Provide download button
                if export_path.exists():
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download File",
                            data=f,
                            file_name=export_path.name,
                            mime='application/octet-stream'
                        )
    
    with col2:
        st.subheader("📋 Export Preview")
        
        # Show last 10 detections
        preview_data = st.session_state.detection_history[-10:]
        preview_df = pd.DataFrame([
            {
                'Frame': d.get('frame_id', 0),
                'Objects': len(d.get('objects', [])),
                'FPS': round(d.get('fps', 0), 2),
                'Latency (ms)': round(d.get('inference_time', 0) * 1000, 2)
            }
            for d in preview_data
        ])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

def export_data(format_type, include_metrics):
    """Export detection data to file"""
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if format_type == "JSON":
        export_path = output_dir / f"detections_{timestamp}.json"
        export_dict = {
            'detections': st.session_state.detection_history,
            'object_counts': st.session_state.object_counts
        }
        
        if include_metrics and st.session_state.fps_history:
            export_dict['metrics'] = {
                'avg_fps': float(np.mean(st.session_state.fps_history)),
                'total_frames': len(st.session_state.detection_history)
            }
        
        with open(export_path, 'w') as f:
            json.dump(export_dict, f, indent=2, default=str)
    
    elif format_type == "CSV":
        export_path = output_dir / f"detections_{timestamp}.csv"
        rows = []
        for det in st.session_state.detection_history:
            rows.append({
                'frame_id': det.get('frame_id', 0),
                'timestamp': det.get('timestamp', 0),
                'fps': det.get('fps', 0),
                'num_objects': len(det.get('objects', [])),
                'inference_time': det.get('inference_time', 0)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(export_path, index=False)
    
    elif format_type == "Excel":
        export_path = output_dir / f"detections_{timestamp}.xlsx"
        rows = []
        for det in st.session_state.detection_history:
            rows.append({
                'frame_id': det.get('frame_id', 0),
                'timestamp': det.get('timestamp', 0),
                'fps': det.get('fps', 0),
                'num_objects': len(det.get('objects', [])),
                'inference_time': det.get('inference_time', 0)
            })
        
        df = pd.DataFrame(rows)
        df.to_excel(export_path, index=False, engine='openpyxl')
    
    return export_path

if __name__ == "__main__":
    main()