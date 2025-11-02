#!/bin/bash

# Setup script for Ambient Drone Synthesizer
# Works on Termux and regular Linux/Mac systems

echo "🎵 Ambient Drone Synthesizer Setup"
echo "=================================="

# Detect if we're on Termux
if [ -d "/data/data/com.termux" ]; then
    echo "📱 Termux detected!"
    
    # Update package list
    echo "Updating package list..."
    pkg update -y
    
    # Install Python if not already installed
    echo "Installing Python..."
    pkg install python -y
    
    # Install audio libraries for Termux
    echo "Installing audio libraries..."
    pkg install portaudio -y
    pkg install libsndfile -y
    
    # Install build dependencies
    echo "Installing build tools..."
    pkg install build-essential -y
    pkg install pkg-config -y
    
    # Install numpy (may take a while on Termux)
    echo "Installing numpy (this may take a few minutes)..."
    pip install numpy
    
    # Install sounddevice
    echo "Installing sounddevice..."
    pip install sounddevice
    
else
    echo "💻 Regular system detected!"
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
        exit 1
    fi
    
    # Install with pip
    echo "Installing Python packages..."
    pip3 install --user numpy sounddevice
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the synthesizer:"
echo "  python ambient_drone_synth.py"
echo ""
echo "Note: On Termux, make sure you have:"
echo "  1. Given Termux microphone/audio permissions"
echo "  2. Plugged in headphones or speakers"
echo ""
echo "Enjoy creating ambient soundscapes! 🎶"
