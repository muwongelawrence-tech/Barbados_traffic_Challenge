"""
Update predict.py to load from custom model directory
"""
import sys
import os

# Get model directory from environment variable or use default
MODEL_DIR = os.environ.get('MODEL_DIR', 'models/baseline')

# Rest of imports...
