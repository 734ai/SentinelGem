/**
 * SentinelGem Main UI App
 * Author: Muzan Sano
 * Root application component with routing and layout
 */

import React, { useState, useEffect } from 'react';
import { 
  BrowserRouter as Router, 
  Routes, 
  Route, 
  Navigate,
  Link,
  useLocation 
} from 'react-router-dom';
import { 
  Shield, 
  Home, 
  Activity, 
  Upload, 
  Settings, 
  Users, 
  BarChart3, 
  Bell,
  Menu,
  X,
  LogOut
} from 'lucide-react';

// Import components
import ThreatDashboard from './ThreatDashboard';
import FileUploadAnalyzer from './FileUploadAnalyzer';
import ConfigurationPanel from './ConfigurationPanel';

// Main App Component
export const App = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    // Initialize user session and load initial data
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Load user session
      const userResponse = await fetch('/api/v1/auth/me');
      if (userResponse.ok) {
        const userData = await userResponse.json();
        setUser(userData);
      }

      // Load notifications
      const notificationsResponse = await fetch('/api/v1/notifications');
      if (notificationsResponse.ok) {
        const notificationsData = await notificationsResponse.json();
        setNotifications(notificationsData.notifications || []);
      }
    } catch (error) {
      console.error('Failed to initialize app:', error);
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('/api/v1/auth/logout', { method: 'POST' });
      setUser(null);
      // Redirect to login
      window.location.href = '/login';
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Sidebar */}
        <Sidebar 
          isOpen={sidebarOpen} 
          onClose={() => setSidebarOpen(false)}
          user={user}
          onLogout={handleLogout}
        />

        {/* Main Content */}
        <div className={`transition-all duration-300 ${sidebarOpen ? 'lg:ml-64' : 'lg:ml-20'}`}>
          {/* Top Navigation */}
          <TopNavigation 
            onMenuClick={() => setSidebarOpen(!sidebarOpen)}
            notifications={notifications}
            user={user}
          />

          {/* Page Content */}
          <main className="p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<ThreatDashboard />} />
              <Route path="/analyze" element={<FileUploadAnalyzer />} />
              <Route path="/config" element={<ConfigurationPanel />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="/logs" element={<LogsPage />} />
              <Route path="/users" element={<UsersPage />} />
            </Routes>
          </main>
        </div>

        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div 
            className="fixed inset-0 bg-gray-600 bg-opacity-50 lg:hidden z-40"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </div>
    </Router>
  );
};

// Sidebar Component
const Sidebar = ({ isOpen, onClose, user, onLogout }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Analyze Files', href: '/analyze', icon: Upload },
    { name: 'Reports', href: '/reports', icon: BarChart3 },
    { name: 'Activity Logs', href: '/logs', icon: Activity },
    { name: 'Configuration', href: '/config', icon: Settings },
    { name: 'Users', href: '/users', icon: Users }
  ];

  return (
    <div className={`fixed inset-y-0 left-0 z-50 bg-white shadow-lg transform transition-transform duration-300 ease-in-out ${
      isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
    } ${isOpen ? 'w-64' : 'lg:w-20'}`}>
      
      {/* Header */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <Shield className="h-8 w-8 text-blue-600" />
          {isOpen && (
            <span className="text-xl font-bold text-gray-900">SentinelGem</span>
          )}
        </div>
        
        <button
          onClick={onClose}
          className="lg:hidden text-gray-400 hover:text-gray-600"
        >
          <X className="h-6 w-6" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="mt-8 px-4">
        <ul className="space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;
            
            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                  onClick={() => {
                    if (window.innerWidth < 1024) onClose();
                  }}
                >
                  <Icon className="h-5 w-5" />
                  {isOpen && (
                    <span className="ml-3">{item.name}</span>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* User Profile */}
      {user && isOpen && (
        <div className="absolute bottom-0 w-full p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-sm font-medium text-blue-700">
                {user.name?.charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {user.name}
              </p>
              <p className="text-xs text-gray-500 truncate">
                {user.role}
              </p>
            </div>
          </div>
          
          <button
            onClick={onLogout}
            className="mt-3 w-full flex items-center justify-center px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </button>
        </div>
      )}
    </div>
  );
};

// Top Navigation Component
const TopNavigation = ({ onMenuClick, notifications, user }) => {
  const [showNotifications, setShowNotifications] = useState(false);
  
  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between h-16 px-6">
        <div className="flex items-center space-x-4">
          <button
            onClick={onMenuClick}
            className="text-gray-500 hover:text-gray-700 lg:hidden"
          >
            <Menu className="h-6 w-6" />
          </button>
          
          <div className="hidden lg:block">
            <h1 className="text-lg font-semibold text-gray-900">
              Threat Detection Platform
            </h1>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 text-gray-400 hover:text-gray-600"
            >
              <Bell className="h-6 w-6" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            {/* Notifications Dropdown */}
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Notifications
                  </h3>
                </div>
                
                <div className="max-h-96 overflow-y-auto">
                  {notifications.length > 0 ? (
                    notifications.slice(0, 5).map((notification) => (
                      <NotificationItem key={notification.id} notification={notification} />
                    ))
                  ) : (
                    <div className="p-4 text-center text-gray-500">
                      No notifications
                    </div>
                  )}
                </div>
                
                {notifications.length > 5 && (
                  <div className="p-4 border-t border-gray-200">
                    <Link
                      to="/notifications"
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View all notifications
                    </Link>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* User Avatar */}
          {user && (
            <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-sm font-medium text-blue-700">
                {user.name?.charAt(0).toUpperCase()}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

// Notification Item Component
const NotificationItem = ({ notification }) => {
  return (
    <div className={`p-4 border-b border-gray-100 hover:bg-gray-50 ${
      !notification.read ? 'bg-blue-50' : ''
    }`}>
      <div className="flex items-start space-x-3">
        <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
          notification.type === 'threat' ? 'bg-red-500' :
          notification.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
        }`}></div>
        
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900">
            {notification.title}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            {notification.message}
          </p>
          <p className="text-xs text-gray-500 mt-2">
            {notification.timestamp}
          </p>
        </div>
      </div>
    </div>
  );
};

// Placeholder Pages
const ReportsPage = () => (
  <div className="bg-white rounded-lg shadow p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Reports</h1>
    <p className="text-gray-600">Comprehensive threat analysis reports and analytics will be displayed here.</p>
  </div>
);

const LogsPage = () => (
  <div className="bg-white rounded-lg shadow p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Activity Logs</h1>
    <p className="text-gray-600">System activity logs and audit trails will be displayed here.</p>
  </div>
);

const UsersPage = () => (
  <div className="bg-white rounded-lg shadow p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">User Management</h1>
    <p className="text-gray-600">User account management and permissions will be displayed here.</p>
  </div>
);

export default App;
