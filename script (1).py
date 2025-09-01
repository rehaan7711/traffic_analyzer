# Create requirements.txt content
requirements_content = """# Core Dependencies
ultralytics==8.0.196
opencv-python==4.8.1.78
streamlit==1.27.2
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0

# Machine Learning & Computer Vision
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Web Framework (Alternative to Streamlit)
Flask>=2.3.0
flask-cors>=4.0.0

# Video Processing
imageio>=2.31.0
moviepy>=1.0.3

# Tracking & Detection
supervision>=0.13.0
filterpy>=1.4.5

# Utilities
tqdm>=4.65.0
requests>=2.31.0
python-dotenv>=1.0.0

# Data Export
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Optional GPU Support
# torch-audio  # Uncomment for audio processing
# torchtext    # Uncomment for text processing
"""

print("requirements.txt Content:")
print("="*50)
print(requirements_content)

# Save to file
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("\nrequirements.txt file created successfully!")