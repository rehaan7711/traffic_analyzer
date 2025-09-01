import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import custom modules
from traffic_analyzer import TrafficAnalyzer
from vehicle_detector import VehicleDetector
from emergency_detector import EmergencyDetector
from utils import save_processed_video, create_report

# Configure Streamlit page
st.set_page_config(
    page_title="Traffic Analyzer Project",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("🚗 Traffic Analyzer Project - YOLOv8 & OpenCV")
st.markdown("""
**Analyze traffic videos with AI-powered vehicle detection and classification**
- Detect and count vehicles (cars, buses, trucks, bikes, autos)
- Identify emergency vehicles (ambulance, fire truck, police)
- Traffic density analysis (light/medium/heavy)
- Junction type detection (2-way, 3-way, 4-way)
""")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Model Selection
model_option = st.sidebar.selectbox(
    "Choose YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    index=2,
    help="Larger models are more accurate but slower"
)

# Detection Settings
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
)

# Traffic Density Thresholds
st.sidebar.subheader("Traffic Density Settings")
light_traffic_max = st.sidebar.slider("Light Traffic Max (%)", 10, 50, 40, 5)
heavy_traffic_min = st.sidebar.slider("Heavy Traffic Min (%)", 50, 90, 65, 5)

# Emergency Vehicle Settings
emergency_alert = st.sidebar.checkbox("Emergency Vehicle Alerts", value=True)
emergency_sound = st.sidebar.checkbox("Sound Alert (Emergency)", value=False)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# File Upload Section
st.header("📁 Upload Traffic Video")
uploaded_file = st.file_uploader(
    "Choose a traffic video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video of 2-way, 3-way, or 4-way traffic intersection"
)

# Main Processing Section
if uploaded_file is not None:
    # Display video info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Uploaded Video")
        st.video(uploaded_file)

    with col2:
        st.subheader("📊 Video Information")
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File Type": uploaded_file.type
        }
        st.json(file_details)

    # Process Video Button
    if st.button("🚀 Analyze Traffic Video", type="primary"):
        try:
            with st.spinner("Processing video... This may take a few minutes."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_video_path = tmp_file.name

                # Initialize analyzers
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Initializing models...")
                progress_bar.progress(10)

                # Initialize traffic analyzer
                analyzer = TrafficAnalyzer(
                    model_path=model_option,
                    confidence_threshold=confidence_threshold,
                    light_traffic_max=light_traffic_max,
                    heavy_traffic_min=heavy_traffic_min
                )

                progress_bar.progress(20)
                status_text.text("Processing video frames...")

                # Analyze video
                results = analyzer.analyze_video(tmp_video_path, progress_callback=progress_bar)

                progress_bar.progress(90)
                status_text.text("Generating reports...")

                # Store results in session state
                st.session_state.analysis_results = results

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

                # Clean up temporary file
                os.unlink(tmp_video_path)

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.exception(e)

# Display Results Section
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results

    st.header("📈 Analysis Results")

    # Summary Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Vehicles Detected",
            results['summary']['total_vehicles'],
            delta=None
        )

    with col2:
        st.metric(
            "Emergency Vehicles",
            results['summary']['emergency_vehicles'],
            delta=None,
            delta_color="inverse" if results['summary']['emergency_vehicles'] > 0 else "normal"
        )

    with col3:
        st.metric(
            "Junction Type",
            results['summary']['junction_type'],
            delta=None
        )

    with col4:
        traffic_density = results['summary']['overall_traffic_density']
        density_color = "normal"
        if traffic_density == "Heavy":
            density_color = "inverse"

        st.metric(
            "Traffic Density",
            traffic_density,
            delta=None,
            delta_color=density_color
        )

    # Vehicle Count by Type
    st.subheader("🚙 Vehicle Classification")
    vehicle_counts = results['vehicle_counts']

    col1, col2 = st.columns([1, 1])

    with col1:
        # Vehicle counts table
        df_vehicles = pd.DataFrame(list(vehicle_counts.items()), 
                                 columns=['Vehicle Type', 'Count'])
        st.dataframe(df_vehicles, use_container_width=True)

    with col2:
        # Vehicle counts pie chart
        if sum(vehicle_counts.values()) > 0:
            fig = px.pie(df_vehicles, values='Count', names='Vehicle Type',
                        title="Vehicle Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Emergency Vehicle Alerts
    if results['emergency_alerts']:
        st.subheader("🚨 Emergency Vehicle Alerts")
        emergency_df = pd.DataFrame(results['emergency_alerts'])
        st.dataframe(emergency_df, use_container_width=True)

        if emergency_sound and len(results['emergency_alerts']) > 0:
            st.audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQxQp+PwtmMcBjiR1/LNeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMiBSuBzvLDdiMJGL7t3p9NEQ=")

    # Traffic Density Over Time
    if 'frame_analysis' in results:
        st.subheader("📊 Traffic Density Analysis")
        frame_data = results['frame_analysis']

        # Create time series plot
        df_density = pd.DataFrame(frame_data)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_density['frame_number'],
            y=df_density['vehicle_count'],
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title="Vehicle Count Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Vehicle Count",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Download Reports
    st.subheader("📥 Download Reports")
    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV Report
        csv_data = pd.DataFrame([results['summary']]).to_csv(index=False)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_data,
            file_name=f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # JSON Report
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="📋 Download JSON Report",
            data=json_data,
            file_name=f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        # Excel Report
        if st.button("📈 Generate Excel Report"):
            with st.spinner("Generating Excel report..."):
                try:
                    # Create Excel file with multiple sheets
                    excel_buffer = create_excel_report(results)
                    st.download_button(
                        label="📈 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("Excel report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating Excel report: {str(e)}")

# Information Section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
**Traffic Analyzer v1.0**

Built with:
- YOLOv8 for object detection
- OpenCV for video processing
- Streamlit for web interface
- Deep learning for classification

**Features:**
- Real-time vehicle detection
- Emergency vehicle recognition
- Traffic density analysis
- Junction type detection
- Comprehensive reporting
""")

# Helper function for Excel report
def create_excel_report(results):
    from io import BytesIO
    import pandas as pd

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame([results['summary']])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Vehicle counts sheet
        vehicle_df = pd.DataFrame(list(results['vehicle_counts'].items()),
                                columns=['Vehicle_Type', 'Count'])
        vehicle_df.to_excel(writer, sheet_name='Vehicle_Counts', index=False)

        # Emergency alerts sheet
        if results['emergency_alerts']:
            emergency_df = pd.DataFrame(results['emergency_alerts'])
            emergency_df.to_excel(writer, sheet_name='Emergency_Alerts', index=False)

        # Frame analysis sheet
        if 'frame_analysis' in results:
            frame_df = pd.DataFrame(results['frame_analysis'])
            frame_df.to_excel(writer, sheet_name='Frame_Analysis', index=False)

    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("**© 2024 Traffic Analyzer Project - Built with ❤️ using YOLOv8 & Streamlit**")
