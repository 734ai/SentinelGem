/**
 * SentinelGem UI Components
 * Author: Muzan Sano
 * React-based interactive threat monitoring dashboard
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  Eye, 
  Activity, 
  Users, 
  Settings,
  Download,
  Upload,
  Mic,
  Image,
  Link,
  MessageSquare,
  TrendingUp,
  Calendar,
  Filter,
  Search,
  Bell,
  Lock
} from 'lucide-react';

// Main Dashboard Component
export const ThreatDashboard = () => {
  const [threats, setThreats] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [realTimeData, setRealTimeData] = useState({});
  const [isMonitoring, setIsMonitoring] = useState(true);

  // Real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchRealTimeData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchRealTimeData = async () => {
    try {
      const response = await fetch('/api/v1/dashboard/realtime');
      const data = await response.json();
      setRealTimeData(data);
    } catch (error) {
      console.error('Failed to fetch real-time data:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <DashboardHeader 
          isMonitoring={isMonitoring}
          onToggleMonitoring={() => setIsMonitoring(!isMonitoring)}
        />
        
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Threats Detected"
            value={realTimeData.threats_detected || 0}
            change="+12%"
            icon={<Shield className="h-6 w-6" />}
            color="red"
          />
          <MetricCard
            title="Analysis Complete"
            value={realTimeData.analyses_completed || 0}
            change="+8%"
            icon={<Eye className="h-6 w-6" />}
            color="blue"
          />
          <MetricCard
            title="Active Monitoring"
            value={realTimeData.active_sessions || 0}
            change="+5%"
            icon={<Activity className="h-6 w-6" />}
            color="green"
          />
          <MetricCard
            title="Risk Score"
            value={`${realTimeData.avg_risk_score || 0}%`}
            change="-2%"
            icon={<AlertTriangle className="h-6 w-6" />}
            color="yellow"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            <RealTimeThreatFeed threats={realTimeData.recent_threats || []} />
            <ThreatAnalyticsChart data={analytics} />
          </div>
          
          {/* Right Column */}
          <div className="space-y-8">
            <QuickAnalysisPanel />
            <SystemStatusPanel />
            <RecentActivityPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

// Dashboard Header Component
const DashboardHeader = ({ isMonitoring, onToggleMonitoring }) => {
  return (
    <div className="flex items-center justify-between mb-8">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Shield className="h-8 w-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-900">SentinelGem</h1>
        </div>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${
          isMonitoring 
            ? 'bg-green-100 text-green-800' 
            : 'bg-gray-100 text-gray-800'
        }`}>
          {isMonitoring ? 'Live Monitoring' : 'Paused'}
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <button
          onClick={onToggleMonitoring}
          className={`px-4 py-2 rounded-lg font-medium ${
            isMonitoring
              ? 'bg-red-100 text-red-700 hover:bg-red-200'
              : 'bg-green-100 text-green-700 hover:bg-green-200'
          }`}
        >
          {isMonitoring ? 'Pause Monitoring' : 'Resume Monitoring'}
        </button>
        <button className="p-2 text-gray-600 hover:text-gray-900">
          <Bell className="h-5 w-5" />
        </button>
        <button className="p-2 text-gray-600 hover:text-gray-900">
          <Settings className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ title, value, change, icon, color }) => {
  const colorClasses = {
    red: 'text-red-600 bg-red-50',
    blue: 'text-blue-600 bg-blue-50',
    green: 'text-green-600 bg-green-50',
    yellow: 'text-yellow-600 bg-yellow-50'
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          {icon}
        </div>
        <span className={`text-sm font-medium ${
          change.startsWith('+') ? 'text-green-600' : 'text-red-600'
        }`}>
          {change}
        </span>
      </div>
      <div className="mt-4">
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        <p className="text-sm text-gray-600">{title}</p>
      </div>
    </div>
  );
};

// Real-time Threat Feed Component
const RealTimeThreatFeed = ({ threats }) => {
  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">
            Real-time Threat Feed
          </h2>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Live</span>
          </div>
        </div>
      </div>
      
      <div className="divide-y divide-gray-200">
        {threats.map((threat, index) => (
          <ThreatItem key={index} threat={threat} />
        ))}
      </div>
    </div>
  );
};

// Individual Threat Item Component
const ThreatItem = ({ threat }) => {
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'phishing': return <MessageSquare className="h-4 w-4" />;
      case 'social_engineering': return <Users className="h-4 w-4" />;
      case 'malicious_url': return <Link className="h-4 w-4" />;
      case 'audio_anomaly': return <Mic className="h-4 w-4" />;
      case 'image_analysis': return <Image className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  return (
    <div className="p-4 hover:bg-gray-50">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {getTypeIcon(threat.type)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-900 truncate">
              {threat.title}
            </p>
            <div className="flex items-center space-x-2">
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                getSeverityColor(threat.severity)
              }`}>
                {threat.severity}
              </span>
              <span className="text-xs text-gray-500">
                {threat.timestamp}
              </span>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            {threat.description}
          </p>
          <div className="flex items-center mt-2 space-x-4">
            <span className="text-xs text-gray-500">
              Risk Score: {threat.risk_score}%
            </span>
            <span className="text-xs text-gray-500">
              Source: {threat.source}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Quick Analysis Panel Component
const QuickAnalysisPanel = () => {
  const [analysisType, setAnalysisType] = useState('text');
  const [inputContent, setInputContent] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: inputContent,
          analysis_types: [analysisType],
          priority: 'high'
        })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Quick Analysis
      </h3>
      
      <div className="space-y-4">
        {/* Analysis Type Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Analysis Type
          </label>
          <select
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="text">Text Analysis</option>
            <option value="url">URL Analysis</option>
            <option value="phishing">Phishing Detection</option>
            <option value="social_engineering">Social Engineering</option>
          </select>
        </div>

        {/* Input Area */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Content to Analyze
          </label>
          <textarea
            value={inputContent}
            onChange={(e) => setInputContent(e.target.value)}
            placeholder="Enter text, URL, or content to analyze..."
            className="w-full border border-gray-300 rounded-md px-3 py-2 h-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Analyze Button */}
        <button
          onClick={handleAnalysis}
          disabled={!inputContent || isAnalyzing}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </button>

        {/* Results */}
        {result && (
          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <h4 className="font-medium text-gray-900 mb-2">Analysis Result</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Risk Score:</span>
                <span className={`text-sm font-medium ${
                  result.risk_score > 0.7 ? 'text-red-600' : 
                  result.risk_score > 0.4 ? 'text-yellow-600' : 'text-green-600'
                }`}>
                  {Math.round(result.risk_score * 100)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Threat Type:</span>
                <span className="text-sm font-medium">{result.threat_category}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Confidence:</span>
                <span className="text-sm font-medium">
                  {Math.round(result.confidence * 100)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// System Status Panel Component
const SystemStatusPanel = () => {
  const [systemStatus, setSystemStatus] = useState({});

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/v1/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        System Status
      </h3>
      
      <div className="space-y-3">
        {Object.entries(systemStatus.services || {}).map(([service, status]) => (
          <div key={service} className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-700 capitalize">
                {service.replace('_', ' ')}
              </span>
            </div>
            <span className={`text-sm font-medium ${
              status === 'healthy' ? 'text-green-600' : 'text-red-600'
            }`}>
              {status}
            </span>
          </div>
        ))}
      </div>

      {systemStatus.performance && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Performance</h4>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Response Time</span>
              <span className="font-medium">
                {systemStatus.performance.avg_response_time_ms}ms
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">CPU Usage</span>
              <span className="font-medium">
                {Math.round(systemStatus.performance.cpu_usage * 100)}%
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Memory Usage</span>
              <span className="font-medium">
                {Math.round(systemStatus.performance.memory_usage * 100)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Recent Activity Panel Component
const RecentActivityPanel = () => {
  const [activities, setActivities] = useState([]);

  useEffect(() => {
    fetchRecentActivities();
  }, []);

  const fetchRecentActivities = async () => {
    try {
      const response = await fetch('/api/v1/activities/recent');
      const data = await response.json();
      setActivities(data.activities || []);
    } catch (error) {
      console.error('Failed to fetch activities:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Recent Activity
      </h3>
      
      <div className="space-y-3">
        {activities.map((activity, index) => (
          <div key={index} className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-900">{activity.description}</p>
              <p className="text-xs text-gray-500 mt-1">{activity.timestamp}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Threat Analytics Chart Component (placeholder for actual chart implementation)
const ThreatAnalyticsChart = ({ data }) => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Threat Analytics
        </h3>
        <div className="flex items-center space-x-2">
          <Calendar className="h-4 w-4 text-gray-400" />
          <select className="text-sm border border-gray-300 rounded-md px-2 py-1">
            <option>Last 24 hours</option>
            <option>Last 7 days</option>
            <option>Last 30 days</option>
          </select>
        </div>
      </div>
      
      {/* Chart placeholder - integrate with your preferred charting library */}
      <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <TrendingUp className="h-12 w-12 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">Chart will be rendered here</p>
          <p className="text-sm text-gray-400">
            Integrate with Chart.js, D3, or similar library
          </p>
        </div>
      </div>
    </div>
  );
};

export default ThreatDashboard;
