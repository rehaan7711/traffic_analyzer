from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import cv2
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
import threading
import time

# Import our custom modules
from traffic_analyzer import TrafficAnalyzer
from utils import save_video_from_bytesio, create_report, setup_logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['REPORT_FOLDER'] = 'reports'

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['REPORT_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)

# Setup logging
setup_logging()

# Global variables for tracking analysis status
analysis_status = {}
analysis_results = {}

@app.route('/')
def index():
    """Main page with video upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start analysis."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get analysis parameters
        model_choice = request.form.get('model', 'yolov8m.pt')
        confidence = float(request.form.get('confidence', 0.5))
        light_traffic_max = int(request.form.get('light_traffic_max', 40))
        heavy_traffic_min = int(request.form.get('heavy_traffic_min', 65))

        # Generate unique analysis ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save uploaded video
        video_filename = f"{analysis_id}_{file.filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)

        # Initialize analysis status
        analysis_status[analysis_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing analysis...',
            'start_time': datetime.now()
        }

        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=run_analysis,
            args=(analysis_id, video_path, model_choice, confidence, light_traffic_max, heavy_traffic_min)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({
            'analysis_id': analysis_id,
            'message': 'Analysis started successfully',
            'status_url': url_for('get_analysis_status', analysis_id=analysis_id)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_analysis(analysis_id, video_path, model_choice, confidence, light_traffic_max, heavy_traffic_min):
    """Run traffic analysis in background thread."""
    try:
        # Update status
        analysis_status[analysis_id].update({
            'status': 'running',
            'progress': 10,
            'message': 'Loading model...'
        })

        # Initialize analyzer
        analyzer = TrafficAnalyzer(
            model_path=model_choice,
            confidence_threshold=confidence,
            light_traffic_max=light_traffic_max,
            heavy_traffic_min=heavy_traffic_min
        )

        # Update status
        analysis_status[analysis_id].update({
            'progress': 20,
            'message': 'Analyzing video...'
        })

        # Custom progress callback
        def progress_callback(progress):
            analysis_status[analysis_id]['progress'] = progress

        # Run analysis
        results = analyzer.analyze_video(video_path, progress_callback=progress_callback)

        # Save results
        analysis_results[analysis_id] = results

        # Update status
        analysis_status[analysis_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Analysis completed successfully',
            'end_time': datetime.now()
        })

        # Generate report
        report_path = os.path.join(app.config['REPORT_FOLDER'], f"{analysis_id}_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        analysis_status[analysis_id].update({
            'status': 'error',
            'message': str(e),
            'end_time': datetime.now()
        })

@app.route('/status/<analysis_id>')
def get_analysis_status(analysis_id):
    """Get status of analysis."""
    if analysis_id not in analysis_status:
        return jsonify({'error': 'Analysis ID not found'}), 404

    status = analysis_status[analysis_id].copy()

    # Add duration if completed
    if 'end_time' in status and 'start_time' in status:
        duration = (status['end_time'] - status['start_time']).total_seconds()
        status['duration_seconds'] = duration

    return jsonify(status)

@app.route('/results/<analysis_id>')
def get_analysis_results(analysis_id):
    """Get analysis results."""
    if analysis_id not in analysis_results:
        return jsonify({'error': 'Results not found'}), 404

    return jsonify(analysis_results[analysis_id])

@app.route('/dashboard/<analysis_id>')
def analysis_dashboard(analysis_id):
    """Show analysis dashboard."""
    if analysis_id not in analysis_status:
        return "Analysis not found", 404

    return render_template('dashboard.html', analysis_id=analysis_id)

@app.route('/download/report/<analysis_id>')
def download_report(analysis_id):
    """Download analysis report."""
    report_path = os.path.join(app.config['REPORT_FOLDER'], f"{analysis_id}_report.json")

    if not os.path.exists(report_path):
        return "Report not found", 404

    return send_file(report_path, as_attachment=True)

@app.route('/download/csv/<analysis_id>')
def download_csv_report(analysis_id):
    """Download CSV report."""
    if analysis_id not in analysis_results:
        return "Results not found", 404

    results = analysis_results[analysis_id]

    # Create CSV content
    import pandas as pd
    from io import StringIO

    # Summary data
    summary_data = [results['summary']]
    df = pd.DataFrame(summary_data)

    # Create CSV in memory
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Create temporary file
    csv_path = os.path.join(app.config['REPORT_FOLDER'], f"{analysis_id}_summary.csv")
    with open(csv_path, 'w') as f:
        f.write(output.getvalue())

    return send_file(csv_path, as_attachment=True, mimetype='text/csv')

@app.route('/api/upload_and_analyze', methods=['POST'])
def api_upload_and_analyze():
    """API endpoint for programmatic access."""
    try:
        # Check if request has video file
        if 'video' not in request.files:
            return jsonify({'error': 'No video file in request'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Get parameters from JSON or form data
        params = request.get_json() or request.form

        config = {
            'model': params.get('model', 'yolov8m.pt'),
            'confidence': float(params.get('confidence', 0.5)),
            'light_traffic_max': int(params.get('light_traffic_max', 40)),
            'heavy_traffic_min': int(params.get('heavy_traffic_min', 65)),
            'return_video': params.get('return_video', False)
        }

        # Save temporary video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            file.save(tmp_file)
            tmp_path = tmp_file.name

        try:
            # Run analysis synchronously for API
            analyzer = TrafficAnalyzer(
                model_path=config['model'],
                confidence_threshold=config['confidence'],
                light_traffic_max=config['light_traffic_max'],
                heavy_traffic_min=config['heavy_traffic_min']
            )

            results = analyzer.analyze_video(tmp_path)

            # Clean up temporary file
            os.unlink(tmp_path)

            response = {
                'success': True,
                'results': results,
                'analysis_timestamp': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_available_models():
    """Get list of available YOLO models."""
    models = [
        {'name': 'yolov8n.pt', 'description': 'YOLOv8 Nano - Fastest, lowest accuracy'},
        {'name': 'yolov8s.pt', 'description': 'YOLOv8 Small - Fast, good accuracy'},
        {'name': 'yolov8m.pt', 'description': 'YOLOv8 Medium - Balanced speed/accuracy'},
        {'name': 'yolov8l.pt', 'description': 'YOLOv8 Large - Slower, high accuracy'},
        {'name': 'yolov8x.pt', 'description': 'YOLOv8 Extra Large - Slowest, highest accuracy'}
    ]

    return jsonify({'models': models})

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/cleanup')
def cleanup_old_files():
    """Clean up old analysis files."""
    try:
        current_time = datetime.now()
        cleanup_count = 0

        # Clean up files older than 24 hours
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['REPORT_FOLDER']]:
            for file_path in Path(folder).glob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > 86400:  # 24 hours
                        file_path.unlink()
                        cleanup_count += 1

        # Clean up old analysis status and results
        old_analyses = []
        for analysis_id, status in analysis_status.items():
            if 'start_time' in status:
                age = current_time - status['start_time']
                if age.total_seconds() > 86400:
                    old_analyses.append(analysis_id)

        for analysis_id in old_analyses:
            if analysis_id in analysis_status:
                del analysis_status[analysis_id]
            if analysis_id in analysis_results:
                del analysis_results[analysis_id]
            cleanup_count += 1

        return jsonify({
            'message': f'Cleaned up {cleanup_count} old files/analyses',
            'cleanup_count': cleanup_count
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Template creation function
def create_html_templates():
    """Create HTML templates for the Flask app."""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)

    # Index template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Analyzer - Upload Video</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: #2E86C1; margin-bottom: 30px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #2E86C1; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1B4F72; }
        .progress { display: none; margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #2E86C1; width: 0%; transition: width 0.3s ease; }
        .results { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; display: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Traffic Analyzer</h1>
        <p>Upload a traffic video for AI-powered analysis</p>
    </div>

    <form id="uploadForm" enctype="multipart/form-data">
        <div class="form-group">
            <label for="video">Select Traffic Video:</label>
            <input type="file" id="video" name="video" accept="video/*" required>
        </div>

        <div class="form-group">
            <label for="model">YOLO Model:</label>
            <select id="model" name="model">
                <option value="yolov8n.pt">YOLOv8 Nano (Fastest)</option>
                <option value="yolov8s.pt">YOLOv8 Small</option>
                <option value="yolov8m.pt" selected>YOLOv8 Medium (Recommended)</option>
                <option value="yolov8l.pt">YOLOv8 Large</option>
                <option value="yolov8x.pt">YOLOv8 Extra Large (Most Accurate)</option>
            </select>
        </div>

        <div class="form-group">
            <label for="confidence">Confidence Threshold:</label>
            <input type="range" id="confidence" name="confidence" min="0.1" max="1.0" step="0.05" value="0.5">
            <span id="confidenceValue">0.5</span>
        </div>

        <div class="form-group">
            <label for="light_traffic_max">Light Traffic Max (%):</label>
            <input type="number" id="light_traffic_max" name="light_traffic_max" value="40" min="10" max="50">
        </div>

        <div class="form-group">
            <label for="heavy_traffic_min">Heavy Traffic Min (%):</label>
            <input type="number" id="heavy_traffic_min" name="heavy_traffic_min" value="65" min="50" max="90">
        </div>

        <button type="submit">🚀 Analyze Video</button>
    </form>

    <div class="progress" id="progressDiv">
        <h3>Analysis Progress</h3>
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        <p id="progressText">Starting analysis...</p>
    </div>

    <div class="results" id="resultsDiv">
        <h3>Analysis Complete!</h3>
        <div id="resultsContent"></div>
        <button onclick="downloadReport()">📥 Download Report</button>
        <button onclick="viewDashboard()">📊 View Dashboard</button>
    </div>

    <script>
        let currentAnalysisId = null;

        document.getElementById('confidence').addEventListener('input', function() {
            document.getElementById('confidenceValue').textContent = this.value;
        });

        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.analysis_id) {
                    currentAnalysisId = data.analysis_id;
                    document.getElementById('progressDiv').style.display = 'block';
                    monitorProgress(data.analysis_id);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Upload failed: ' + error.message);
            });
        });

        function monitorProgress(analysisId) {
            const interval = setInterval(() => {
                fetch(`/status/${analysisId}`)
                    .then(response => response.json())
                    .then(data => {
                        const progressFill = document.getElementById('progressFill');
                        const progressText = document.getElementById('progressText');

                        progressFill.style.width = data.progress + '%';
                        progressText.textContent = data.message;

                        if (data.status === 'completed') {
                            clearInterval(interval);
                            showResults(analysisId);
                        } else if (data.status === 'error') {
                            clearInterval(interval);
                            alert('Analysis failed: ' + data.message);
                        }
                    });
            }, 2000);
        }

        function showResults(analysisId) {
            fetch(`/results/${analysisId}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('resultsDiv');
                    const resultsContent = document.getElementById('resultsContent');

                    resultsContent.innerHTML = `
                        <p><strong>Total Vehicles:</strong> ${data.summary.total_vehicles}</p>
                        <p><strong>Emergency Vehicles:</strong> ${data.summary.emergency_vehicles}</p>
                        <p><strong>Junction Type:</strong> ${data.summary.junction_type}</p>
                        <p><strong>Traffic Density:</strong> ${data.summary.overall_traffic_density}</p>
                    `;

                    resultsDiv.style.display = 'block';
                });
        }

        function downloadReport() {
            if (currentAnalysisId) {
                window.open(`/download/report/${currentAnalysisId}`, '_blank');
            }
        }

        function viewDashboard() {
            if (currentAnalysisId) {
                window.open(`/dashboard/${currentAnalysisId}`, '_blank');
            }
        }
    </script>
</body>
</html>
    """

    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_html)

    # Dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: #2E86C1; margin-bottom: 30px; }
        .card { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px 20px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2E86C1; }
        .metric-label { color: #666; }
        .emergency { color: #E74C3C !important; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Traffic Analysis Dashboard</h1>
            <p>Analysis ID: {{ analysis_id }}</p>
        </div>

        <div id="dashboardContent">
            <p>Loading dashboard...</p>
        </div>
    </div>

    <script>
        const analysisId = "{{ analysis_id }}";

        function loadDashboard() {
            fetch(`/results/${analysisId}`)
                .then(response => response.json())
                .then(data => {
                    const content = document.getElementById('dashboardContent');

                    let html = `
                        <div class="card">
                            <h2>Summary Metrics</h2>
                            <div class="metric">
                                <div class="metric-value">${data.summary.total_vehicles}</div>
                                <div class="metric-label">Total Vehicles</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value ${data.summary.emergency_vehicles > 0 ? 'emergency' : ''}">${data.summary.emergency_vehicles}</div>
                                <div class="metric-label">Emergency Vehicles</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.summary.junction_type}</div>
                                <div class="metric-label">Junction Type</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.summary.overall_traffic_density}</div>
                                <div class="metric-label">Traffic Density</div>
                            </div>
                        </div>

                        <div class="card">
                            <h2>Vehicle Classification</h2>
                            <table>
                                <tr><th>Vehicle Type</th><th>Count</th></tr>
                    `;

                    for (const [type, count] of Object.entries(data.vehicle_counts)) {
                        html += `<tr><td>${type}</td><td>${count}</td></tr>`;
                    }

                    html += `</table></div>`;

                    if (data.emergency_alerts && data.emergency_alerts.length > 0) {
                        html += `
                            <div class="card">
                                <h2>🚨 Emergency Vehicle Alerts</h2>
                                <table>
                                    <tr><th>Time</th><th>Vehicle Type</th><th>Confidence</th><th>Frame</th></tr>
                        `;

                        data.emergency_alerts.forEach(alert => {
                            html += `<tr><td>${alert.timestamp}</td><td>${alert.vehicle_type}</td><td>${alert.confidence.toFixed(2)}</td><td>${alert.frame_number}</td></tr>`;
                        });

                        html += `</table></div>`;
                    }

                    content.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('dashboardContent').innerHTML = '<p>Error loading dashboard data.</p>';
                });
        }

        // Load dashboard on page load
        loadDashboard();
    </script>
</body>
</html>
    """

    with open(templates_dir / 'dashboard.html', 'w') as f:
        f.write(dashboard_html)

if __name__ == '__main__':
    # Create HTML templates
    create_html_templates()

    print("Flask Traffic Analyzer Application")
    print("="*40)
    print("Available endpoints:")
    print("- / : Main upload page")
    print("- /upload : Upload video for analysis")
    print("- /status/<id> : Check analysis status")
    print("- /results/<id> : Get analysis results")
    print("- /dashboard/<id> : View analysis dashboard")
    print("- /api/upload_and_analyze : API endpoint")
    print("- /api/models : Get available models")
    print("- /api/health : Health check")

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
