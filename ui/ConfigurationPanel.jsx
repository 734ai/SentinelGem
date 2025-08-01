/**
 * SentinelGem Configuration Panel
 * Author: Muzan Sano
 * System configuration and settings management
 */

import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Shield, 
  Sliders, 
  Users, 
  Bell, 
  Lock,
  Database,
  Monitor,
  AlertTriangle,
  CheckCircle,
  Save,
  RotateCcw,
  Eye,
  EyeOff
} from 'lucide-react';

export const ConfigurationPanel = () => {
  const [activeTab, setActiveTab] = useState('detection');
  const [config, setConfig] = useState({});
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    try {
      const response = await fetch('/api/v1/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Failed to load configuration:', error);
    }
  };

  const saveConfiguration = async () => {
    setSaving(true);
    try {
      await fetch('/api/v1/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const updateConfig = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setHasChanges(true);
  };

  const tabs = [
    { id: 'detection', label: 'Detection Settings', icon: Shield },
    { id: 'thresholds', label: 'Thresholds', icon: Sliders },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Lock },
    { id: 'performance', label: 'Performance', icon: Monitor },
    { id: 'users', label: 'User Management', icon: Users }
  ];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Settings className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">System Configuration</h1>
                <p className="text-gray-600">Manage SentinelGem settings and preferences</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {hasChanges && (
                <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm rounded-full">
                  Unsaved changes
                </span>
              )}
              <button
                onClick={() => {
                  loadConfiguration();
                  setHasChanges(false);
                }}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 flex items-center space-x-2"
              >
                <RotateCcw className="h-4 w-4" />
                <span>Reset</span>
              </button>
              <button
                onClick={saveConfiguration}
                disabled={!hasChanges || saving}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
              >
                <Save className="h-4 w-4" />
                <span>{saving ? 'Saving...' : 'Save Changes'}</span>
              </button>
            </div>
          </div>
        </div>

        <div className="flex">
          {/* Sidebar Navigation */}
          <div className="w-64 bg-gray-50 border-r border-gray-200">
            <nav className="p-4 space-y-2">
              {tabs.map(tab => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                      activeTab === tab.id
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1 p-6">
            {activeTab === 'detection' && (
              <DetectionSettings config={config} updateConfig={updateConfig} />
            )}
            {activeTab === 'thresholds' && (
              <ThresholdSettings config={config} updateConfig={updateConfig} />
            )}
            {activeTab === 'notifications' && (
              <NotificationSettings config={config} updateConfig={updateConfig} />
            )}
            {activeTab === 'security' && (
              <SecuritySettings config={config} updateConfig={updateConfig} />
            )}
            {activeTab === 'performance' && (
              <PerformanceSettings config={config} updateConfig={updateConfig} />
            )}
            {activeTab === 'users' && (
              <UserManagement config={config} updateConfig={updateConfig} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Detection Settings Component
const DetectionSettings = ({ config, updateConfig }) => {
  const detectionConfig = config.detection || {};

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Detection Settings</h2>
        <p className="text-gray-600 mb-6">Configure threat detection parameters and analysis types.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Analysis Types */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Analysis Types</h3>
          <div className="space-y-3">
            {[
              { key: 'phishing_detection', label: 'Phishing Detection' },
              { key: 'social_engineering', label: 'Social Engineering' },
              { key: 'malware_scanning', label: 'Malware Scanning' },
              { key: 'url_analysis', label: 'URL Analysis' },
              { key: 'ocr_analysis', label: 'OCR Analysis' },
              { key: 'audio_analysis', label: 'Audio Analysis' }
            ].map(item => (
              <label key={item.key} className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={detectionConfig[item.key] || false}
                  onChange={(e) => updateConfig('detection', item.key, e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{item.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Real-time Monitoring */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Real-time Monitoring</h3>
          <div className="space-y-4">
            <label className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Enable Real-time Scanning</span>
              <input
                type="checkbox"
                checked={detectionConfig.realtime_enabled || false}
                onChange={(e) => updateConfig('detection', 'realtime_enabled', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
            </label>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Scan Interval (seconds)
              </label>
              <input
                type="number"
                min="1"
                max="3600"
                value={detectionConfig.scan_interval || 60}
                onChange={(e) => updateConfig('detection', 'scan_interval', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Concurrent Analyses
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={detectionConfig.max_concurrent || 10}
                onChange={(e) => updateConfig('detection', 'max_concurrent', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Threshold Settings Component
const ThresholdSettings = ({ config, updateConfig }) => {
  const thresholds = config.thresholds || {};

  const thresholdItems = [
    { key: 'phishing_threshold', label: 'Phishing Detection', default: 0.75 },
    { key: 'social_engineering_threshold', label: 'Social Engineering', default: 0.70 },
    { key: 'malware_threshold', label: 'Malware Detection', default: 0.80 },
    { key: 'url_reputation_threshold', label: 'URL Reputation', default: 0.60 },
    { key: 'confidence_threshold', label: 'Minimum Confidence', default: 0.65 },
    { key: 'auto_block_threshold', label: 'Auto-block Threshold', default: 0.90 }
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Threshold Settings</h2>
        <p className="text-gray-600 mb-6">Configure sensitivity thresholds for different threat types.</p>
      </div>

      <div className="space-y-6">
        {thresholdItems.map(item => (
          <div key={item.key} className="bg-gray-50 p-6 rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <label className="text-lg font-medium text-gray-900">
                {item.label}
              </label>
              <span className="text-2xl font-bold text-blue-600">
                {Math.round((thresholds[item.key] || item.default) * 100)}%
              </span>
            </div>
            
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={thresholds[item.key] || item.default}
              onChange={(e) => updateConfig('thresholds', item.key, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            />
            
            <div className="flex justify-between text-sm text-gray-500 mt-2">
              <span>Low (0%)</span>
              <span>Medium (50%)</span>
              <span>High (100%)</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Notification Settings Component
const NotificationSettings = ({ config, updateConfig }) => {
  const notifications = config.notifications || {};

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Notification Settings</h2>
        <p className="text-gray-600 mb-6">Configure how and when you receive threat notifications.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Notification Types */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Notification Types</h3>
          <div className="space-y-3">
            {[
              { key: 'email_notifications', label: 'Email Notifications' },
              { key: 'slack_notifications', label: 'Slack Notifications' },
              { key: 'webhook_notifications', label: 'Webhook Notifications' },
              { key: 'desktop_notifications', label: 'Desktop Notifications' },
              { key: 'sms_notifications', label: 'SMS Notifications' }
            ].map(item => (
              <label key={item.key} className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={notifications[item.key] || false}
                  onChange={(e) => updateConfig('notifications', item.key, e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{item.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Notification Triggers */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Notification Triggers</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Minimum Risk Score for Notifications
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={notifications.min_risk_score || 0.65}
                onChange={(e) => updateConfig('notifications', 'min_risk_score', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600 mt-1">
                {Math.round((notifications.min_risk_score || 0.65) * 100)}%
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notification Frequency
              </label>
              <select
                value={notifications.frequency || 'immediate'}
                onChange={(e) => updateConfig('notifications', 'frequency', e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="immediate">Immediate</option>
                <option value="hourly">Hourly Summary</option>
                <option value="daily">Daily Summary</option>
                <option value="weekly">Weekly Summary</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Security Settings Component
const SecuritySettings = ({ config, updateConfig }) => {
  const security = config.security || {};
  const [showApiKey, setShowApiKey] = useState(false);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Security Settings</h2>
        <p className="text-gray-600 mb-6">Configure authentication, access control, and security policies.</p>
      </div>

      <div className="space-y-6">
        {/* API Security */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">API Security</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                API Key
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={security.api_key || ''}
                  onChange={(e) => updateConfig('security', 'api_key', e.target.value)}
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter API key..."
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="p-2 text-gray-500 hover:text-gray-700"
                >
                  {showApiKey ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={security.require_https || false}
                onChange={(e) => updateConfig('security', 'require_https', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Require HTTPS for all API calls</span>
            </label>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={security.rate_limiting || false}
                onChange={(e) => updateConfig('security', 'rate_limiting', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Enable rate limiting</span>
            </label>
          </div>
        </div>

        {/* Access Control */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Access Control</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Allowed IP Addresses (comma-separated)
              </label>
              <textarea
                value={security.allowed_ips || ''}
                onChange={(e) => updateConfig('security', 'allowed_ips', e.target.value)}
                placeholder="192.168.1.0/24, 10.0.0.0/8"
                className="w-full border border-gray-300 rounded-md px-3 py-2 h-20 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Timeout (minutes)
              </label>
              <input
                type="number"
                min="5"
                max="480"
                value={security.session_timeout || 60}
                onChange={(e) => updateConfig('security', 'session_timeout', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Performance Settings Component
const PerformanceSettings = ({ config, updateConfig }) => {
  const performance = config.performance || {};

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance Settings</h2>
        <p className="text-gray-600 mb-6">Optimize system performance and resource usage.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Resource Limits */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Resource Limits</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Memory Usage (GB)
              </label>
              <input
                type="number"
                min="1"
                max="64"
                value={performance.max_memory || 8}
                onChange={(e) => updateConfig('performance', 'max_memory', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max CPU Cores
              </label>
              <input
                type="number"
                min="1"
                max="32"
                value={performance.max_cpu_cores || 4}
                onChange={(e) => updateConfig('performance', 'max_cpu_cores', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Queue Size Limit
              </label>
              <input
                type="number"
                min="10"
                max="10000"
                value={performance.queue_size_limit || 1000}
                onChange={(e) => updateConfig('performance', 'queue_size_limit', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Caching */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Caching</h3>
          <div className="space-y-4">
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={performance.enable_caching || false}
                onChange={(e) => updateConfig('performance', 'enable_caching', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Enable Result Caching</span>
            </label>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cache TTL (hours)
              </label>
              <input
                type="number"
                min="1"
                max="168"
                value={performance.cache_ttl || 24}
                onChange={(e) => updateConfig('performance', 'cache_ttl', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Cache Size (MB)
              </label>
              <input
                type="number"
                min="100"
                max="10000"
                value={performance.max_cache_size || 1000}
                onChange={(e) => updateConfig('performance', 'max_cache_size', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// User Management Component
const UserManagement = ({ config, updateConfig }) => {
  const [users, setUsers] = useState([]);
  const [showAddUser, setShowAddUser] = useState(false);

  useEffect(() => {
    // Load users from API
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      const response = await fetch('/api/v1/users');
      const data = await response.json();
      setUsers(data.users || []);
    } catch (error) {
      console.error('Failed to load users:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">User Management</h2>
          <button
            onClick={() => setShowAddUser(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Add User
          </button>
        </div>
        <p className="text-gray-600 mb-6">Manage user accounts and permissions.</p>
      </div>

      {/* Users Table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                User
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Role
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Last Login
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {users.map((user) => (
              <tr key={user.id}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                      <span className="text-sm font-medium text-gray-700">
                        {user.name?.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">{user.name}</div>
                      <div className="text-sm text-gray-500">{user.email}</div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                    {user.role}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    user.status === 'active' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {user.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {user.last_login || 'Never'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button className="text-blue-600 hover:text-blue-900 mr-3">
                    Edit
                  </button>
                  <button className="text-red-600 hover:text-red-900">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ConfigurationPanel;
