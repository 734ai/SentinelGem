# SentinelGem UI Application

This is the React-based user interface for SentinelGem, an advanced AI-powered threat detection platform.

## Features

- **Real-time Threat Dashboard**: Live monitoring of security threats and system status
- **Multi-modal File Analysis**: Upload and analyze files using AI-powered detection
- **Configuration Management**: Comprehensive system settings and security controls
- **Responsive Design**: Modern UI that works across desktop and mobile devices
- **Dark/Light Mode**: User preference-based theme switching
- **Progressive Web App**: Installable web application with offline capabilities

## Tech Stack

- **React 18.2.0**: Modern React with hooks and concurrent features
- **React Router 6.8.1**: Client-side routing and navigation
- **Tailwind CSS 3.1.8**: Utility-first CSS framework
- **Lucide React**: Beautiful SVG icons
- **Axios**: HTTP client for API communication
- **Recharts**: Data visualization and charts

## Getting Started

### Prerequisites

- Node.js 16.x or higher
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:3000`.

### Building for Production

```bash
# Create production build
npm run build

# Serve production build locally
npm run serve
```

## Project Structure

```
ui/
├── public/              # Static assets
│   ├── index.html      # Main HTML template
│   ├── manifest.json   # PWA manifest
│   └── favicon.ico     # Application icon
├── src/                # Source code (when using src/ structure)
├── components/         # React components
│   ├── ThreatDashboard.jsx
│   ├── FileUploadAnalyzer.jsx
│   └── ConfigurationPanel.jsx
├── App.jsx            # Main application component
├── package.json       # Project dependencies
├── tailwind.config.js # Tailwind CSS configuration
└── README.md          # This file
```

## Component Overview

### ThreatDashboard.jsx
The main dashboard component providing:
- Real-time threat monitoring
- System status indicators
- Security metrics and analytics
- Quick analysis capabilities

### FileUploadAnalyzer.jsx
Multi-modal file analysis interface featuring:
- Drag-and-drop file upload
- Support for images, audio, text, and documents
- AI-powered threat analysis
- Detailed results visualization

### ConfigurationPanel.jsx
System configuration management with:
- Detection sensitivity settings
- Notification preferences
- Security policy configuration
- User access controls

### App.jsx
Main application shell including:
- Navigation and routing
- Authentication handling
- Layout management
- Global state management

## API Integration

The UI communicates with the SentinelGem backend API for:
- Authentication and user management
- File upload and analysis
- Real-time threat data
- System configuration
- Webhook notifications

## Development Guidelines

### Code Style
- Use modern React patterns (hooks, functional components)
- Follow ESLint and Prettier configurations
- Implement proper error boundaries
- Use TypeScript for type safety (when applicable)

### Testing
```bash
# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Generate coverage report
npm run test:coverage
```

### Performance
- Implement lazy loading for route components
- Use React.memo for expensive components
- Optimize bundle size with code splitting
- Monitor performance with React DevTools

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t sentinelgem-ui .

# Run container
docker run -p 3000:3000 sentinelgem-ui
```

### Production Environment Variables

```bash
REACT_APP_API_URL=https://api.sentinelgem.com
REACT_APP_WS_URL=wss://ws.sentinelgem.com
REACT_APP_VERSION=1.0.0
REACT_APP_ENVIRONMENT=production
```

## Security Considerations

- Implement Content Security Policy (CSP)
- Use HTTPS in production
- Sanitize user inputs
- Implement proper authentication flows
- Regular security audits with `npm audit`

## Performance Optimization

- Bundle size optimization with webpack-bundle-analyzer
- Image optimization and lazy loading
- Service worker for caching strategies
- Code splitting at route level
- Progressive loading for large datasets

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Ensure CI/CD pipeline passes

## License

This project is part of the SentinelGem platform. See the main repository for license information.

## Support

For technical support or questions:
- Documentation: `/docs`
- Issues: GitHub Issues
- Community: Discord Server
- Email: support@sentinelgem.com
