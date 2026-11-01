"""
Entry point for Streamlit Deployment.
Loads the complete 5-Page Predictive Maintenance Operations Center.
"""

import sys
import os

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dashboard.app import *
