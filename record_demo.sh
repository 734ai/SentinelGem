#!/bin/bash

# 🎬 SentinelGem Competition Demo Recording Script
# For Google Gemma 3n Impact Challenge 2025

echo "🛡️ SentinelGem Demo Recording Setup"
echo "===================================="
echo ""

# Get screen dimensions
SCREEN_INFO=$(xrandr | grep '*' | head -1 | awk '{print $1}')
echo "📺 Screen Resolution: $SCREEN_INFO"

# Create recordings directory
mkdir -p recordings
cd recordings

echo ""
echo "🎯 RECORDING OPTIONS:"
echo ""
echo "1. FULL SCREEN RECORDING"
echo "   Command: ffmpeg -f x11grab -s $SCREEN_INFO -i :0.0 -r 30 -c:v libx264 -preset fast -crf 23 sentinelgem_demo_full.mp4"
echo ""
echo "2. AREA SELECTION (Recommended for Terminal + Editor)"
echo "   Command: ffmpeg -f x11grab -s 1920x1080 -i :0.0+100,100 -r 30 -c:v libx264 -preset fast -crf 23 sentinelgem_demo_area.mp4"
echo ""
echo "3. SPECIFIC WINDOW RECORDING"
echo "   First run: xwininfo (click on window to get ID)"
echo "   Then: ffmpeg -f x11grab -i :0.0+WINDOW_ID -r 30 -c:v libx264 -preset fast -crf 23 sentinelgem_demo_window.mp4"
echo ""
echo "4. QUICK START - RECORDMYDESKTOP (Alternative)"
echo "   Command: recordmydesktop --width=1920 --height=1080 -x=100 -y=100 --fps=30 -o sentinelgem_demo.ogv"
echo ""

echo "🚀 DEMO RECORDING SEQUENCE:"
echo ""
echo "Step 1: Position terminal and VS Code side by side"
echo "Step 2: Start recording with area selection"
echo "Step 3: Run SentinelGem demo scenarios:"
echo "   • python production_status.py"
echo "   • python main.py --mode agent"
echo "   • Show threat analysis examples"
echo "   • Display auto-generated notebooks"
echo "Step 4: Stop recording (Ctrl+C)"
echo "Step 5: Convert to competition format if needed"
echo ""

echo "📝 RECORDING TIPS:"
echo "• Use 30fps for smooth playback"  
echo "• Keep recording under 3 minutes for competition"
echo "• Focus on terminal output and code editor"
echo "• Show real-time threat detection"
echo "• Highlight Gemma 3n integration"
echo ""

echo "🎬 Ready to record! Choose your preferred method above."
