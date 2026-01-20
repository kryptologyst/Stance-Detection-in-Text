#!/usr/bin/env python3
"""
Setup script for the Stance Detection project.

This script helps set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🎯 Stance Detection Project Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create necessary directories
    directories = ["data", "models", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Test import
    print("🧪 Testing imports...")
    try:
        sys.path.append("src")
        from stance_detector import StanceDetector
        print("✅ Core module imports successfully!")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the example: python example.py")
    print("2. Launch web interface: streamlit run web_app/app.py")
    print("3. Use CLI: python src/cli.py --help")
    print("4. Read the README.md for detailed usage instructions")


if __name__ == "__main__":
    main()
