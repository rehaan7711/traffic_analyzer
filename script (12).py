# Create demo script
demo_script_code = '''#!/usr/bin/env python3
"""
Traffic Analyzer Project - Demo Script
=====================================

This script demonstrates the basic usage of the Traffic Analyzer system.
Run this script to test the system with sample data or your own videos.

Usage:
    python demo_script.py --video path/to/video.mp4
    python demo_script.py --help
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Import our modules
try:
    from traffic_analyzer import TrafficAnalyzer
    from utils import create_directories, setup_logging, get_system_info
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all project files are in the same directory.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traffic Analyzer Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_script.py --video traffic_sample.mp4
    python demo_script.py --video traffic_sample.mp4 --model yolov8s.pt
    python demo_script.py --video traffic_sample.mp4 --confidence 0.3
    python demo_script.py --system-info
        """
    )
    
    parser.add_argument('--video', '-v', 
                       help='Path to traffic video file')
    
    parser.add_argument('--model', '-m', 
                       default='yolov8m.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLO model to use (default: yolov8m.pt)')
    
    parser.add_argument('--confidence', '-c', 
                       type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    
    parser.add_argument('--light-traffic', 
                       type=int, default=40,
                       help='Light traffic maximum percentage (default: 40)')
    
    parser.add_argument('--heavy-traffic', 
                       type=int, default=65,
                       help='Heavy traffic minimum percentage (default: 65)')
    
    parser.add_argument('--output-dir', 
                       default='outputs',
                       help='Output directory for results (default: outputs)')
    
    parser.add_argument('--save-video', 
                       action='store_true',
                       help='Save processed video with detections')
    
    parser.add_argument('--system-info', 
                       action='store_true',
                       help='Show system information and exit')
    
    parser.add_argument('--test-modules', 
                       action='store_true',
                       help='Test all modules and exit')
    
    parser.add_argument('--verbose', '-V', 
                       action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def show_system_info():
    """Display system information."""
    print("="*60)
    print("🖥️  SYSTEM INFORMATION")
    print("="*60)
    
    info = get_system_info()
    
    print(f"Platform: {info['platform']}")
    print(f"Python Version: {info['python_version']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Devices: {info['cuda_devices']}")
    
    print()
    print("📦 PYTHON PACKAGES")
    print("-" * 30)
    
    required_packages = [
        'ultralytics', 'opencv-python', 'torch', 'streamlit', 
        'flask', 'numpy', 'pandas', 'matplotlib', 'plotly'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")

def test_modules():
    """Test all project modules."""
    print("="*60)
    print("🧪 MODULE TESTING")
    print("="*60)
    
    modules_to_test = [
        ('traffic_analyzer', 'TrafficAnalyzer'),
        ('vehicle_detector', 'VehicleDetector'),
        ('emergency_detector', 'EmergencyDetector'),
        ('centroid_tracker', 'CentroidTracker'),
        ('traffic_density', 'TrafficDensityAnalyzer'),
        ('junction_detector', 'JunctionDetector'),
        ('utils', 'create_directories')
    ]
    
    results = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
                results.append((module_name, True, None))
            else:
                print(f"⚠️  {module_name} (missing {class_name})")
                results.append((module_name, False, f"Missing {class_name}"))
        except Exception as e:
            print(f"❌ {module_name} ({str(e)})")
            results.append((module_name, False, str(e)))
    
    print()
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Module Test Results: {successful}/{total} passed")
    
    if successful == total:
        print("🎉 All modules loaded successfully!")
    else:
        print("⚠️  Some modules failed to load. Check installation.")
        return False
    
    return True

def analyze_video(video_path, args):
    """Analyze a video file."""
    print("="*60)
    print("🎬 VIDEO ANALYSIS")
    print("="*60)
    
    # Validate video file
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"📹 Video: {video_path}")
    print(f"🧠 Model: {args.model}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"📊 Light Traffic Max: {args.light_traffic}%")
    print(f"📊 Heavy Traffic Min: {args.heavy_traffic}%")
    print()
    
    try:
        # Initialize analyzer
        print("Initializing Traffic Analyzer...")
        analyzer = TrafficAnalyzer(
            model_path=args.model,
            confidence_threshold=args.confidence,
            light_traffic_max=args.light_traffic,
            heavy_traffic_min=args.heavy_traffic
        )
        
        # Simple progress tracking
        class ProgressCallback:
            def __init__(self):
                self.last_progress = 0
            
            def __call__(self, progress):
                if progress - self.last_progress >= 10:
                    print(f"Progress: {progress}%")
                    self.last_progress = progress
        
        progress_callback = ProgressCallback()
        
        # Start analysis
        print("\\nStarting video analysis...")
        start_time = time.time()
        
        results = analyzer.analyze_video(
            video_path, 
            progress_callback=progress_callback,
            save_output=args.save_video
        )
        
        end_time = time.time()
        analysis_duration = end_time - start_time
        
        # Display results
        print("\\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        summary = results['summary']
        
        print(f"⏱️  Analysis Duration: {analysis_duration:.2f} seconds")
        print(f"🚗 Total Vehicles: {summary['total_vehicles']}")
        print(f"🚨 Emergency Vehicles: {summary['emergency_vehicles']}")
        print(f"🛣️  Junction Type: {summary['junction_type']}")
        print(f"🚦 Traffic Density: {summary['overall_traffic_density']}")
        print(f"📽️  Frames Processed: {summary['frames_processed']}")
        
        # Vehicle breakdown
        print("\\n📋 VEHICLE CLASSIFICATION")
        print("-" * 30)
        for vehicle_type, count in results['vehicle_counts'].items():
            print(f"{vehicle_type.capitalize()}: {count}")
        
        # Emergency alerts
        if results['emergency_alerts']:
            print("\\n🚨 EMERGENCY ALERTS")
            print("-" * 30)
            for alert in results['emergency_alerts']:
                print(f"Frame {alert['frame_number']}: {alert['vehicle_type']} "
                      f"(confidence: {alert['confidence']:.2f})")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = output_dir / f"analysis_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n💾 Results saved to: {json_path}")
        
        # CSV summary
        import pandas as pd
        summary_df = pd.DataFrame([summary])
        csv_path = output_dir / f"analysis_summary_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        
        print(f"📊 Summary saved to: {csv_path}")
        
        if args.save_video:
            print(f"🎥 Processed video saved in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

def run_interactive_demo():
    """Run interactive demo mode."""
    print("="*60)
    print("🎮 INTERACTIVE DEMO MODE")
    print("="*60)
    
    print("Welcome to the Traffic Analyzer Interactive Demo!")
    print("This mode will guide you through the analysis process.\\n")
    
    # Get video path
    while True:
        video_path = input("Enter path to traffic video (or 'quit' to exit): ").strip()
        
        if video_path.lower() == 'quit':
            print("Goodbye!")
            return
        
        if os.path.exists(video_path):
            break
        else:
            print(f"File not found: {video_path}")
            print("Please check the path and try again.\\n")
    
    # Get model choice
    print("\\nAvailable models:")
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input("\\nSelect model (1-5) [default: 3]: ").strip()
            if not choice:
                choice = '3'
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                break
            else:
                print("Invalid choice. Please select 1-5.")
        except ValueError:
            print("Please enter a number.")
    
    # Get confidence threshold
    while True:
        try:
            confidence_str = input("\\nConfidence threshold (0.1-1.0) [default: 0.5]: ").strip()
            if not confidence_str:
                confidence = 0.5
            else:
                confidence = float(confidence_str)
                
            if 0.1 <= confidence <= 1.0:
                break
            else:
                print("Confidence must be between 0.1 and 1.0")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create args object
    class Args:
        def __init__(self):
            self.model = selected_model
            self.confidence = confidence
            self.light_traffic = 40
            self.heavy_traffic = 65
            self.output_dir = 'outputs'
            self.save_video = True
            self.verbose = True
    
    args = Args()
    
    print(f"\\n🚀 Starting analysis with {selected_model}...")
    success = analyze_video(video_path, args)
    
    if success:
        print("\\n🎉 Analysis completed successfully!")
        print("Check the 'outputs' directory for results.")
    else:
        print("\\n😞 Analysis failed. Please check the error messages above.")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Create necessary directories
    create_directories()
    
    print("🚗 Traffic Analyzer Project - Demo Script")
    print("=" * 50)
    
    # Handle special modes
    if args.system_info:
        show_system_info()
        return
    
    if args.test_modules:
        success = test_modules()
        if not success:
            sys.exit(1)
        return
    
    # Interactive mode if no video specified
    if not args.video:
        run_interactive_demo()
        return
    
    # Video analysis mode
    success = analyze_video(args.video, args)
    
    if success:
        print("\\n🎉 Demo completed successfully!")
    else:
        print("\\n😞 Demo failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
'''

with open('demo_script.py', 'w') as f:
    f.write(demo_script_code)

print("demo_script.py created successfully!")
print(f"File size: {len(demo_script_code)} characters")

# Create a summary of all files created
print("\n" + "="*60)
print("📁 PROJECT FILES SUMMARY")
print("="*60)

files_created = [
    "requirements.txt",
    "app.py", 
    "traffic_analyzer.py",
    "vehicle_detector.py",
    "emergency_detector.py", 
    "centroid_tracker.py",
    "traffic_density.py",
    "junction_detector.py",
    "utils.py",
    "flask_app.py",
    "SETUP.md",
    "README.md",
    "demo_script.py"
]

print("✅ Successfully created all project files:")
for i, filename in enumerate(files_created, 1):
    print(f"{i:2}. {filename}")

print(f"\n🎯 Total files created: {len(files_created)}")
print("\n🚀 Ready to use! Follow these steps:")
print("1. pip install -r requirements.txt")
print("2. streamlit run app.py  (for web interface)")
print("3. python demo_script.py --help  (for command line)")
print("\n📖 See README.md and SETUP.md for detailed instructions!")