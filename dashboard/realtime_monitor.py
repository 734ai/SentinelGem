"""
Real-Time Threat Monitoring Dashboard
Live visualization of threats detected by SentinelGem
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List
import threading
from queue import Queue

# Mock data generators for demonstration
class ThreatDataGenerator:
    """Generate realistic threat data for dashboard demo"""
    
    def __init__(self):
        self.threat_types = ['phishing', 'malware', 'bec_fraud', 'crypto_scam', 'credential_theft']
        self.severities = ['low', 'medium', 'high', 'critical']
        self.sources = ['email', 'web', 'file', 'network', 'mobile']
        self.regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Africa']
        
    def generate_threat_event(self) -> Dict:
        """Generate a single threat event"""
        import random
        
        return {
            'timestamp': datetime.now(),
            'threat_type': random.choice(self.threat_types),
            'severity': random.choice(self.severities),
            'source': random.choice(self.sources),
            'region': random.choice(self.regions),
            'confidence': round(random.uniform(0.6, 0.99), 2),
            'blocked': random.choice([True, False]),
            'target_type': random.choice(['individual', 'organization', 'government']),
            'attack_vector': random.choice(['link', 'attachment', 'social_engineering', 'exploit']),
            'gemma_model': random.choice(['gemma-2b', 'gemma-9b', 'gemma-27b'])
        }
    
    def generate_batch_events(self, count: int = 10) -> List[Dict]:
        """Generate multiple threat events"""
        return [self.generate_threat_event() for _ in range(count)]

class RealTimeDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.threat_generator = ThreatDataGenerator()
        self.threat_history = []
        self.threat_queue = Queue()
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start real-time threat monitoring"""
        self.is_monitoring = True
        
        def monitor_threats():
            while self.is_monitoring:
                # Generate new threats
                new_threats = self.threat_generator.generate_batch_events(
                    np.random.randint(1, 5)
                )
                
                for threat in new_threats:
                    self.threat_queue.put(threat)
                    self.threat_history.append(threat)
                
                # Keep only last 1000 events
                if len(self.threat_history) > 1000:
                    self.threat_history = self.threat_history[-1000:]
                
                time.sleep(2)  # Update every 2 seconds
        
        monitor_thread = threading.Thread(target=monitor_threats, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
    
    def get_recent_threats(self, minutes: int = 60) -> List[Dict]:
        """Get threats from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            threat for threat in self.threat_history 
            if threat['timestamp'] >= cutoff_time
        ]
    
    def get_threat_statistics(self) -> Dict:
        """Calculate threat statistics"""
        recent_threats = self.get_recent_threats(60)
        
        if not recent_threats:
            return {
                'total_threats': 0,
                'blocked_threats': 0,
                'high_severity': 0,
                'avg_confidence': 0,
                'threats_per_minute': 0
            }
        
        total = len(recent_threats)
        blocked = sum(1 for t in recent_threats if t['blocked'])
        high_severity = sum(1 for t in recent_threats if t['severity'] in ['high', 'critical'])
        avg_confidence = np.mean([t['confidence'] for t in recent_threats])
        
        return {
            'total_threats': total,
            'blocked_threats': blocked,
            'high_severity': high_severity,
            'avg_confidence': round(avg_confidence, 3),
            'threats_per_minute': round(total / 60, 2)
        }

def create_threat_timeline(threats: List[Dict]) -> go.Figure:
    """Create threat timeline visualization"""
    if not threats:
        return go.Figure()
    
    df = pd.DataFrame(threats)
    df['hour'] = df['timestamp'].dt.floor('H')
    
    timeline_data = df.groupby(['hour', 'threat_type']).size().reset_index(name='count')
    
    fig = px.line(timeline_data, x='hour', y='count', color='threat_type',
                  title="Threat Detection Timeline (Last 24 Hours)")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Threats",
        showlegend=True
    )
    
    return fig

def create_threat_distribution(threats: List[Dict]) -> go.Figure:
    """Create threat type distribution chart"""
    if not threats:
        return go.Figure()
    
    df = pd.DataFrame(threats)
    threat_counts = df['threat_type'].value_counts()
    
    fig = px.pie(values=threat_counts.values, names=threat_counts.index,
                 title="Threat Type Distribution")
    
    return fig

def create_severity_heatmap(threats: List[Dict]) -> go.Figure:
    """Create severity heatmap by source"""
    if not threats:
        return go.Figure()
    
    df = pd.DataFrame(threats)
    heatmap_data = pd.crosstab(df['source'], df['severity'])
    
    fig = px.imshow(heatmap_data.values,
                    labels=dict(x="Severity", y="Source", color="Count"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    title="Threat Severity by Source")
    
    return fig

def create_geographic_map(threats: List[Dict]) -> go.Figure:
    """Create geographic threat distribution"""
    if not threats:
        return go.Figure()
    
    df = pd.DataFrame(threats)
    region_counts = df['region'].value_counts()
    
    # Map regions to coordinates (simplified)
    region_coords = {
        'North America': {'lat': 45, 'lon': -100},
        'Europe': {'lat': 50, 'lon': 10},
        'Asia Pacific': {'lat': 25, 'lon': 120},
        'Latin America': {'lat': -15, 'lon': -60},
        'Africa': {'lat': 0, 'lon': 20}
    }
    
    map_data = []
    for region, count in region_counts.items():
        if region in region_coords:
            map_data.append({
                'region': region,
                'count': count,
                'lat': region_coords[region]['lat'],
                'lon': region_coords[region]['lon']
            })
    
    if not map_data:
        return go.Figure()
    
    map_df = pd.DataFrame(map_data)
    
    fig = px.scatter_geo(map_df, lat='lat', lon='lon', size='count',
                         hover_name='region', hover_data=['count'],
                         title="Global Threat Distribution")
    
    return fig

def create_confidence_distribution(threats: List[Dict]) -> go.Figure:
    """Create confidence score distribution"""
    if not threats:
        return go.Figure()
    
    df = pd.DataFrame(threats)
    
    fig = px.histogram(df, x='confidence', nbins=20,
                       title="AI Confidence Score Distribution")
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Number of Detections"
    )
    
    return fig

def main():
    """Main Streamlit dashboard"""
    st.set_page_config(
        page_title="SentinelGem - Real-Time Threat Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = RealTimeDashboard()
        st.session_state.dashboard.start_monitoring()
    
    dashboard = st.session_state.dashboard
    
    # Header
    st.title("🛡️ SentinelGem - Real-Time Threat Intelligence")
    st.markdown("*Powered by Google Gemma 3n - Live Cybersecurity Monitoring*")
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 15 minutes", "Last 30 minutes", "Last hour", "Last 6 hours"]
    )
    
    time_mapping = {
        "Last 15 minutes": 15,
        "Last 30 minutes": 30,
        "Last hour": 60,
        "Last 6 hours": 360
    }
    
    selected_minutes = time_mapping[time_range]
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
    
    if st.sidebar.button("Manual Refresh"):
        st.rerun()
    
    # Get recent threats
    recent_threats = dashboard.get_recent_threats(selected_minutes)
    stats = dashboard.get_threat_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Threats Detected",
            value=stats['total_threats'],
            delta=f"+{np.random.randint(0, 5)} new"
        )
    
    with col2:
        st.metric(
            label="Threats Blocked",
            value=stats['blocked_threats'],
            delta=f"{round(stats['blocked_threats']/max(stats['total_threats'], 1)*100, 1)}% blocked"
        )
    
    with col3:
        st.metric(
            label="High Severity",
            value=stats['high_severity'],
            delta="Critical alerts",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="AI Confidence",
            value=f"{stats['avg_confidence']*100:.1f}%",
            delta="Avg. detection confidence"
        )
    
    # Charts
    if recent_threats:
        # Timeline
        st.subheader("📈 Threat Detection Timeline")
        timeline_fig = create_threat_timeline(recent_threats)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Two-column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Threat Types")
            pie_fig = create_threat_distribution(recent_threats)
            st.plotly_chart(pie_fig, use_container_width=True)
            
            st.subheader("🎚️ Confidence Distribution")
            conf_fig = create_confidence_distribution(recent_threats)
            st.plotly_chart(conf_fig, use_container_width=True)
        
        with col2:
            st.subheader("🔥 Severity Heatmap")
            heatmap_fig = create_severity_heatmap(recent_threats)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.subheader("🌍 Global Distribution")
            map_fig = create_geographic_map(recent_threats)
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Recent alerts table
        st.subheader("🚨 Recent Threat Alerts")
        
        # Format recent threats for display
        display_threats = []
        for threat in recent_threats[-20:]:  # Show last 20
            display_threats.append({
                'Time': threat['timestamp'].strftime('%H:%M:%S'),
                'Type': threat['threat_type'].replace('_', ' ').title(),
                'Severity': threat['severity'].upper(),
                'Source': threat['source'].title(),
                'Region': threat['region'],
                'Confidence': f"{threat['confidence']*100:.1f}%",
                'Status': '🛡️ Blocked' if threat['blocked'] else '⚠️ Detected',
                'Gemma Model': threat['gemma_model']
            })
        
        if display_threats:
            threats_df = pd.DataFrame(display_threats)
            st.dataframe(
                threats_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Live feed simulation
        st.subheader("📡 Live Threat Feed")
        live_container = st.container()
        
        # Show latest threat events
        if not dashboard.threat_queue.empty():
            latest_threats = []
            while not dashboard.threat_queue.empty() and len(latest_threats) < 5:
                latest_threats.append(dashboard.threat_queue.get())
            
            for threat in latest_threats:
                severity_emoji = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }
                
                status_emoji = '🛡️' if threat['blocked'] else '⚠️'
                
                live_container.info(
                    f"{severity_emoji[threat['severity']]} {status_emoji} "
                    f"**{threat['threat_type'].replace('_', ' ').title()}** detected "
                    f"from {threat['source']} in {threat['region']} "
                    f"(Confidence: {threat['confidence']*100:.1f}%) - "
                    f"{threat['timestamp'].strftime('%H:%M:%S')}"
                )
    
    else:
        st.info("🔍 No threats detected in the selected time range. System is monitoring...")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**SentinelGem** | Real-time cybersecurity threat detection powered by Google Gemma 3n | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
