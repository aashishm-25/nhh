"""
CareSightAI - Hackathon Prototype
Senior Engineer Note: This script uses a continuous loop architecture for video processing.
To prevent Streamlit's top-down execution model from breaking the video stream, 
interact with the sidebar ONLY when the video is stopped.
"""

import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

# ==========================================
# 1. PAGE CONFIG & DARK THEME FIX
# ==========================================
# Forces dark theme natively even if the user hasn't set it in config.toml
st.set_page_config(
    page_title="CareSightAI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject dark theme CSS to guarantee the aesthetic for judges
dark_css = """
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #262730; color: white; }
    .stMetric, .stProgress { background-color: #262730; border-radius: 5px; padding: 10px; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ==========================================
# 2. MOCK DATA INGESTION
# ==========================================
@st.cache_data
def get_patient_data():
    data = {
        'Bed_No': ['Bed 01', 'Bed 02', 'Bed 03', 'Bed 04'],
        'Name': ['John Doe', 'Jane Smith', 'Albert White', 'Robert Johnson'],
        'Age': [45, 62, 58, 71],
        'Medical_History': ['Healthy', 'Healthy', 'Post-Op Surgery', 'Diabetic'],
        'Risk_Multiplier': [1.0, 1.0, 1.2, 1.5] # Diabetic = higher bedsore risk
    }
    return pd.DataFrame(data)

df_patients = get_patient_data()

# ==========================================
# 3. STATE INITIALIZATION
# ==========================================
# We must track these outside the loop to maintain the state machine
if 'running' not in st.session_state:
    st.session_state.running = False
    
if 'confirmed_pose' not in st.session_state:
    st.session_state.confirmed_pose = "Initializing..."
if 'pending_pose' not in st.session_state:
    st.session_state.pending_pose = None
if 'pending_pose_time' not in st.session_state:
    st.session_state.pending_pose_time = 0.0
if 'pose_start_time' not in st.session_state:
    st.session_state.pose_start_time = time.time()

# ==========================================
# 4. SIDEBAR UI
# ==========================================
st.sidebar.title("🛏️ Ward Navigation")
st.sidebar.markdown("Select a bed to view live analytics.")
selected_bed = st.sidebar.selectbox("Active Beds", df_patients['Bed_No'].tolist(), index=3)

# Control button to safely break the while loop
if st.session_state.running:
    if st.sidebar.button("⏹️ Stop Monitoring", type="secondary"):
        st.session_state.running = False
        st.experimental_rerun() # Force UI update to show start button
else:
    if st.sidebar.button("▶️ Start Monitoring", type="primary"):
        # Reset timers when starting fresh
        st.session_state.confirmed_pose = "Initializing..."
        st.session_state.pending_pose = None
        st.session_state.pose_start_time = time.time()
        st.session_state.running = True
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Demo Tip:** Record a video of someone lying flat, then turning to their side to trigger the reset!")

# ==========================================
# 5. MAIN LAYOUT
# ==========================================
st.title("CareSightAI: Contactless Pressure Ulcer Prevention")
st.markdown("Real-time pose estimation and cumulative pressure risk analysis.")

col_video, col_analytics = st.columns([1.2, 1], gap="large")

with col_video:
    st.subheader("CCTV Feed - Skeleton Overlay")
    # Placeholder for the video frame
    video_placeholder = st.empty()
    # Placeholder for pose confidence text under video
    pose_status_text = st.empty()

with col_analytics:
    st.subheader("Patient Analytics")
    patient_info_box = st.empty()
    
    st.markdown("---")
    metric_pose_box = st.empty()
    metric_time_box = st.empty()
    
    st.markdown("---")
    progress_risk_box = st.empty()
    
    st.markdown("---")
    alert_box = st.empty()

# ==========================================
# 6. MEDIAPIPE & VIDEO PROCESSING LOGIC
# ==========================================
# Only run the loop if the user has clicked Start
if st.session_state.running and selected_bed == "Bed 04":
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Custom drawing spec: Neon Cyan looks incredibly "high-tech" for judges
    neon_cyan = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
    neon_magenta = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
    
    VIDEO_PATH = "demo_video.mp4"
    
    # Fallback mechanism if video is missing
    if not os.path.exists(VIDEO_PATH):
        # Create a dummy black frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "ERROR: demo_video.mp4 NOT FOUND", (150, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        video_placeholder.image(dummy_frame, channels="BGR", use_column_width=True)
        st.session_state.running = False
        st.error("Missing demo_video.mp4. Please place your recording in the same directory as app.py.")
        st.stop()

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Get patient data for calculations
    patient_data = df_patients[df_patients['Bed_No'] == selected_bed].iloc[0]
    risk_multiplier = patient_data['Risk_Multiplier']
    base_safe_time_mins = 120.0
    max_safe_time_mins = base_safe_time_mins / risk_multiplier
    
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                
                # If video ends, loop it seamlessly for the demo
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Recolor image to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Recolor back to BGR for OpenCV display
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                current_detected_pose = "No Pose Detected"
                
                # Extract landmarks
                if results.pose_landmarks:
                    # Draw the skeleton with custom neon colors
                    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=neon_cyan,
                        connection_drawing_spec=neon_magenta
                    )
                    
                    # Get Shoulder coordinates
                    left_shoulder = results.pose_landmarks.landmark[11]
                    right_shoulder = results.pose_landmarks.landmark[12]
                    
                    # Calculate absolute X distance (Normalized 0.0 to 1.0)
                    x_dist = abs(left_shoulder.x - right_shoulder.x)
                    
                    # Rule-based logic: 
                    # If distance is small, shoulders overlap -> Lateral (Side)
                    # If distance is large, shoulders spread -> Supine (Flat)
                    # Threshold 0.1 works well for standard webcam/phone horizontal views
                    if x_dist < 0.1:
                        current_detected_pose = "Lateral (Side)"
                    else:
                        current_detected_pose = "Supine (Flat)"
                
                # ---------------------------------------------
                # STATE MACHINE & TIME TRACKING LOGIC
                # ---------------------------------------------
                now = time.time()
                
                # Only process time if a valid pose is detected
                if current_detected_pose in ["Supine (Flat)", "Lateral (Side)"]:
                    # Check if the detected pose differs from our locked-in confirmed pose
                    if current_detected_pose != st.session_state.confirmed_pose:
                        # Start tracking the "pending" new pose
                        if current_detected_pose != st.session_state.pending_pose:
                            st.session_state.pending_pose = current_detected_pose
                            st.session_state.pending_pose_time = now
                        else:
                            # Has the new pose stayed consistent for >= 3 seconds?
                            if (now - st.session_state.pending_pose_time) >= 3.0:
                                # YES -> RESET THE TIMER
                                st.session_state.confirmed_pose = current_detected_pose
                                st.session_state.pose_start_time = now
                                st.session_state.pending_pose = None
                    else:
                        # Pose is unchanged, clear pending state just in case
                        st.session_state.pending_pose = None
                else:
                    # If person leaves frame, pause the pending tracker
                    st.session_state.pending_pose = None
                
                # Calculate simulated time (Hackathon scaling: 1 real sec = 10 simulated mins)
                elapsed_real_seconds = now - st.session_state.pose_start_time
                elapsed_simulated_mins = elapsed_real_seconds * 10.0
                
                # Calculate Risk Percentage
                risk_percentage = min(1.0, elapsed_simulated_mins / max_safe_time_mins)
                is_critical = elapsed_simulated_mins >= max_safe_time_mins
                
                # ---------------------------------------------
                # UPDATE STREAMLIT UI ELEMENTS
                # ---------------------------------------------
                # 1. Video Feed
                video_placeholder.image(image, channels="BGR", use_column_width=True)
                pose_status_text.markdown(f"*Raw Landmark X-Distance:* `{abs(left_shoulder.x - right_shoulder.x):.3f}`" if results.pose_landmarks else "*Waiting for patient...*")
                
                # 2. Patient Info
                with patient_info_box.container():
                    st.markdown(f"**Name:** {patient_data['Name']}  \n"
                                f"**Age:** {patient_data['Age']}  \n"
                                f"**Medical History:** {patient_data['Medical_History']}  \n"
                                f"**Risk Multiplier:** `{risk_multiplier}x`")
                
                # 3. Metrics
                metric_pose_box.metric(label="Current Position", value=st.session_state.confirmed_pose)
                metric_time_box.metric(
                    label="Simulated Time in Position", 
                    value=f"{int(elapsed_simulated_mins)} mins",
                    delta=f"Safe limit: {int(max_safe_time_mins)} mins"
                )
                
                # 4. Progress Bar
                progress_risk_box.progress(
                    risk_percentage, 
                    text=f"Bedsore Risk Level: **{int(risk_percentage * 100)}%**"
                )
                
                # 5. Alert Logic
                if is_critical:
                    alert_box.error("🚨 ACTION REQUIRED: Reposition Patient Immediately 🚨")
                else:
                    # Clear the alert if they repositioned in time
                    alert_box.success("✅ Patient Position Stable")
                
                # Crucial: Add a tiny sleep so Streamlit doesn't consume 100% CPU and crash the laptop
                time.sleep(0.03) 

    except Exception as e:
        st.error(f"Critical CV Error: {e}")
    finally:
        # Always release the video capture when the loop ends
        cap.release()

elif selected_bed != "Bed 04":
    # Fallback UI for non-active beds
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(black_frame, "FEED OFFLINE - BED NOT MONITORED", (250, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
    video_placeholder.image(black_frame, channels="BGR", use_column_width=True)
    alert_box.warning("Camera feed inactive for this bed.")