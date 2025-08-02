#!/bin/bash

# SentinelGem Dashboard Launcher
# Quick setup script for the real-time monitoring dashboard

echo "🛡️ Setting up SentinelGem Real-Time Dashboard..."

# Check if we're in the correct directory
if [ ! -f "dashboard/realtime_monitor.py" ]; then
    echo "❌ Error: Please run this script from the SentinelGem root directory"
    exit 1
fi

# Install dashboard dependencies
echo "📦 Installing dashboard dependencies..."
pip install -r dashboard/requirements.txt

# Launch the dashboard
echo "🚀 Launching SentinelGem Real-Time Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔄 Dashboard updates every 5 seconds with live threat data"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=====================================\n"

# Run Streamlit dashboard
streamlit run dashboard/realtime_monitor.py --server.port 8501 --server.address localhost
