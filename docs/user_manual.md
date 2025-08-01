# SentinelGem User Manual
# Author: Muzan Sano
# Version: 1.0

## Welcome to SentinelGem

SentinelGem is a state-of-the-art AI-powered threat analysis platform designed to detect and analyze various forms of digital threats including phishing emails, social engineering calls, malware infections, and other cybersecurity risks. This comprehensive user manual will guide you through all aspects of using SentinelGem effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [First Steps](#first-steps)
5. [User Interface Guide](#user-interface-guide)
6. [Analysis Features](#analysis-features)
7. [Advanced Usage](#advanced-usage)
8. [Integration Options](#integration-options)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Getting Started

### What is SentinelGem?

SentinelGem is an advanced AI assistant that helps organizations and individuals identify potential cybersecurity threats across multiple communication channels and data types. The system uses cutting-edge AI models including Google's Gemma 3n, OpenAI's Whisper, and advanced OCR technologies to provide comprehensive threat analysis.

### Key Capabilities

- **Email Threat Analysis**: Detect phishing attempts, malicious attachments, and suspicious communications
- **Audio Analysis**: Identify social engineering calls, voice phishing (vishing), and scam attempts
- **Image Analysis**: Scan screenshots, documents, and visual content for threats
- **Log Analysis**: Monitor system logs, browser history, and application logs for indicators of compromise
- **Real-time Analysis**: Process threats as they occur with sub-2-second response times
- **Automated Reporting**: Generate comprehensive analysis reports and recommendations

---

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and data
- **CPU**: Multi-core processor (Intel i5 or AMD Ryzen 5 equivalent)
- **Internet**: Stable connection for model downloads and updates

### Recommended Requirements

- **RAM**: 32GB for optimal performance
- **Storage**: 50GB SSD storage
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for acceleration)

### Software Dependencies

- **Python**: 3.11 or higher
- **Git**: For version control and updates
- **Docker**: For containerized deployment (optional)

---

## Installation

### Quick Start Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/734ai/SentinelGem.git
   cd SentinelGem
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download AI Models**
   ```bash
   python scripts/download_models.py
   ```

4. **Verify Installation**
   ```bash
   python -m pytest tests/ -v
   ```

### Docker Installation

1. **Using Docker Compose**
   ```bash
   git clone https://github.com/734ai/SentinelGem.git
   cd SentinelGem
   docker-compose up -d
   ```

2. **Access the Application**
   - Web UI: http://localhost:8501
   - API: http://localhost:8000

### Model Configuration

SentinelGem supports multiple AI model configurations:

- **Gemma 3n 2B**: Fast analysis, lower memory usage
- **Gemma 3n 9B**: Balanced performance and accuracy
- **Gemma 3n 27B**: Maximum accuracy, higher resource requirements

Configure your preferred model in `config/model_config.yaml`:

```yaml
gemma_model:
  variant: "2b"  # Options: 2b, 9b, 27b
  quantization: true
  device: "auto"  # auto, cpu, cuda

whisper_model:
  variant: "base"  # Options: tiny, base, small, medium, large
  language: "en"
  device: "auto"
```

---

## First Steps

### 1. Launch SentinelGem

#### Web Interface
```bash
streamlit run ui/app.py
```
Access at: http://localhost:8501

#### Command Line Interface
```bash
python -m sentinelgem analyze --input "suspicious_email.txt"
```

#### API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. Your First Analysis

Let's analyze a potentially suspicious email:

1. **Prepare Your Input**: Save suspicious content to a text file
2. **Submit for Analysis**: Upload through web UI or use CLI
3. **Review Results**: Examine threat assessment and recommendations
4. **Take Action**: Follow provided security recommendations

**Example Analysis:**
```bash
python -m sentinelgem analyze --input examples/phishing_email.txt --output-format json
```

### 3. Understanding Results

SentinelGem provides structured analysis results:

```json
{
  "threat_detected": true,
  "confidence_score": 0.92,
  "threat_type": "phishing",
  "description": "High-confidence phishing attempt detected",
  "recommendations": [
    "Do not click any links in this email",
    "Do not provide personal information",
    "Report to your IT security team"
  ],
  "analysis_details": {
    "suspicious_patterns": ["urgent_language", "credential_request"],
    "risk_indicators": ["suspicious_url", "grammar_errors"],
    "confidence_breakdown": {
      "pattern_matching": 0.85,
      "ai_analysis": 0.95
    }
  }
}
```

---

## User Interface Guide

### Web Interface Overview

The SentinelGem web interface provides an intuitive way to analyze threats and review results.

#### Main Dashboard

- **Upload Area**: Drag and drop files or browse to select
- **Analysis Queue**: View pending and completed analyses
- **Recent Results**: Quick access to recent threat assessments
- **System Status**: Model availability and performance metrics

#### Analysis Page

1. **Input Selection**
   - File upload support for text, images, audio
   - Direct text input for quick analysis
   - Batch processing for multiple files

2. **Analysis Options**
   - Model selection (speed vs. accuracy tradeoff)
   - Output format (JSON, HTML, Jupyter notebook)
   - Sensitivity level adjustment

3. **Results Display**
   - Visual threat indicators
   - Detailed analysis breakdown
   - Actionable recommendations
   - Export options

#### Settings Page

- **Model Configuration**: Select AI models and parameters
- **Performance Settings**: Adjust speed/accuracy balance
- **Security Settings**: Configure access controls
- **Notification Settings**: Set up alerts and reports

### Command Line Interface

The CLI provides powerful automation capabilities:

#### Basic Commands

```bash
# Analyze a single file
sentinelgem analyze --input file.txt

# Batch analysis
sentinelgem batch --directory ./suspicious_files/

# Real-time monitoring
sentinelgem monitor --watch ./email_folder/

# Configuration
sentinelgem config --set model.variant=9b
```

#### Advanced Options

```bash
# Custom output format
sentinelgem analyze --input email.txt --format notebook --output report.ipynb

# Specify confidence threshold
sentinelgem analyze --input data.txt --threshold 0.8

# Enable detailed logging
sentinelgem analyze --input file.txt --verbose --log-level debug
```

### API Integration

RESTful API for programmatic access:

#### Authentication
```bash
curl -X POST "http://localhost:8000/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}'
```

#### Analysis Endpoints
```bash
# Text analysis
curl -X POST "http://localhost:8000/analyze/text" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"content": "Suspicious email content..."}'

# File upload analysis
curl -X POST "http://localhost:8000/analyze/file" \
     -H "Authorization: Bearer <token>" \
     -F "file=@suspicious_email.txt"

# Batch analysis
curl -X POST "http://localhost:8000/analyze/batch" \
     -H "Authorization: Bearer <token>" \
     -F "files=@file1.txt" -F "files=@file2.txt"
```

---

## Analysis Features

### Email Threat Detection

SentinelGem excels at identifying various email-based threats:

#### Phishing Detection
- **URL Analysis**: Suspicious links and domains
- **Content Analysis**: Urgency tactics and social engineering
- **Sender Verification**: Email spoofing detection
- **Attachment Scanning**: Malicious file detection

**Example Use Case**: Corporate IT team receives employee reports of suspicious emails. SentinelGem analyzes each email and provides immediate threat assessment, helping prioritize response efforts.

#### Business Email Compromise (BEC)
- **Executive Impersonation**: Detect fake executive emails
- **Financial Fraud**: Wire transfer and payment scams
- **Vendor Impersonation**: Fake supplier communications
- **Account Takeover**: Compromised email indicators

### Audio Analysis Capabilities

Advanced voice and audio threat detection:

#### Social Engineering Calls
- **Voice Stress Analysis**: Detect manipulation tactics
- **Scam Script Recognition**: Identify common fraud patterns
- **Caller Verification**: Voice spoofing detection
- **Emotional Manipulation**: Urgency and fear tactics

#### Voice Phishing (Vishing)
- **Identity Theft**: Personal information requests
- **Financial Scams**: Bank and credit company impersonation
- **Tech Support Scams**: Fake technical assistance calls
- **Government Impersonation**: IRS, Social Security scams

**Example Use Case**: Customer service team records suspicious calls. SentinelGem analyzes call transcripts to identify social engineering attempts and provides training recommendations.

### Image and Document Analysis

Comprehensive visual threat detection:

#### Screenshot Analysis
- **Fake Login Pages**: Phishing website detection
- **Malicious Popups**: Scareware and fake alerts
- **Social Media Scams**: Fake profiles and posts
- **QR Code Analysis**: Malicious QR code detection

#### Document Scanning
- **PDF Malware**: Embedded threats in documents
- **Macro Analysis**: Malicious Office document detection
- **Image Steganography**: Hidden data detection
- **Fake Invoices**: Business fraud attempts

### System Log Analysis

Monitor systems for signs of compromise:

#### Security Event Detection
- **Failed Login Attempts**: Brute force attack indicators
- **Unusual Network Activity**: Data exfiltration attempts
- **Process Anomalies**: Malware execution patterns
- **File System Changes**: Unauthorized modifications

#### Malware Indicators
- **Command & Control**: C2 communication patterns
- **Persistence Mechanisms**: Startup and registry changes
- **Data Collection**: Information gathering activities
- **Lateral Movement**: Network propagation attempts

---

## Advanced Usage

### Custom Model Configuration

#### Fine-tuning for Your Organization

SentinelGem supports custom fine-tuning for organization-specific threats:

1. **Data Collection**: Gather organization-specific threat samples
2. **Model Training**: Fine-tune models on custom data
3. **Validation**: Test performance on validation set
4. **Deployment**: Deploy custom models to production

```yaml
# custom_model_config.yaml
custom_models:
  phishing_detector:
    base_model: "gemma-3n-2b"
    training_data: "./data/custom_phishing_samples.jsonl"
    epochs: 5
    learning_rate: 0.0001
    
  audio_analyzer:
    base_model: "whisper-base"
    training_data: "./data/social_engineering_calls.wav"
    fine_tune_layers: 4
```

#### Multi-Model Ensemble

Combine multiple models for enhanced accuracy:

```python
from sentinelgem import EnsembleAnalyzer

# Create ensemble with multiple models
analyzer = EnsembleAnalyzer([
    "gemma-3n-9b",
    "custom-phishing-detector",
    "domain-specific-model"
])

# Weighted voting for final decision
result = analyzer.analyze(input_data, weights=[0.4, 0.3, 0.3])
```

### Batch Processing

Efficiently analyze large datasets:

#### Command Line Batch Processing
```bash
# Process entire directory
sentinelgem batch --input ./email_archive/ --output ./results/ --parallel 4

# Filter by file type
sentinelgem batch --input ./mixed_files/ --filter "*.txt,*.eml" --output ./results/

# Resume interrupted processing
sentinelgem batch --input ./data/ --resume --checkpoint ./checkpoint.json
```

#### Programmatic Batch Processing
```python
from sentinelgem import BatchProcessor
from pathlib import Path

# Initialize batch processor
processor = BatchProcessor(
    model_config="gemma-3n-9b",
    max_workers=4,
    checkpoint_interval=100
)

# Process files
input_files = Path("./email_archive/").glob("*.txt")
results = processor.process_files(input_files)

# Save results
processor.save_results(results, "./batch_results.json")
```

### Real-time Monitoring

Set up continuous threat monitoring:

#### Email Monitoring
```python
from sentinelgem import EmailMonitor

# Monitor IMAP folder
monitor = EmailMonitor(
    imap_server="imap.company.com",
    username="security@company.com",
    password="secure_password",
    folder="INBOX"
)

# Set up threat detection
monitor.on_threat_detected(lambda result: send_alert(result))
monitor.start_monitoring()
```

#### File System Monitoring
```bash
# Monitor directory for new files
sentinelgem monitor --directory ./downloads/ --action quarantine --alert-email security@company.com
```

### Custom Integration

#### Webhook Integration
```python
from sentinelgem import WebhookHandler

# Set up webhook for external system integration
handler = WebhookHandler(
    endpoint="/analyze",
    auth_token="secure_token_here"
)

@handler.route("/email-analysis", methods=["POST"])
def handle_email_analysis(request):
    email_content = request.json["content"]
    result = analyzer.analyze_text(email_content)
    
    # Send result to SIEM system
    siem.send_event(result)
    
    return {"status": "processed", "threat_level": result.threat_level}
```

#### SIEM Integration
```python
from sentinelgem.integrations import SplunkIntegration, QRadarIntegration

# Splunk integration
splunk = SplunkIntegration(
    host="splunk.company.com",
    token="splunk_hec_token"
)

# Send analysis results to Splunk
splunk.send_threat_event(analysis_result)

# IBM QRadar integration
qradar = QRadarIntegration(
    host="qradar.company.com",
    api_token="qradar_api_token"
)

# Create offense for high-confidence threats
if analysis_result.confidence_score > 0.9:
    qradar.create_offense(analysis_result)
```

---

## Integration Options

### Email Systems

#### Microsoft 365 Integration
```python
from sentinelgem.integrations import O365Integration

# Connect to Microsoft 365
o365 = O365Integration(
    tenant_id="your_tenant_id",
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Analyze emails in specific folder
results = o365.analyze_folder("Phishing Reports")

# Quarantine threats automatically
for result in results:
    if result.threat_detected:
        o365.quarantine_email(result.email_id)
```

#### Gmail Integration
```python
from sentinelgem.integrations import GmailIntegration

# Connect to Gmail API
gmail = GmailIntegration(
    credentials_file="gmail_credentials.json"
)

# Monitor inbox for threats
gmail.monitor_inbox(callback=threat_handler)
```

### Security Tools

#### SOAR Integration
```python
from sentinelgem.integrations import PhantomIntegration

# Phantom SOAR integration
phantom = PhantomIntegration(
    server="phantom.company.com",
    token="phantom_auth_token"
)

# Create playbook trigger
phantom.create_playbook_trigger(
    trigger_condition="threat_confidence > 0.8",
    playbook="phishing_response_playbook"
)
```

#### Threat Intelligence Platforms
```python
from sentinelgem.integrations import MISPIntegration

# MISP integration
misp = MISPIntegration(
    url="https://misp.company.com",
    api_key="misp_api_key"
)

# Enrich analysis with threat intelligence
enriched_result = misp.enrich_analysis(analysis_result)

# Share IOCs with community
if analysis_result.threat_detected:
    misp.share_indicators(analysis_result.indicators)
```

### Custom Workflows

#### Slack Integration
```python
from sentinelgem.integrations import SlackIntegration

# Slack notifications
slack = SlackIntegration(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
)

# Send threat alerts
def send_threat_alert(result):
    if result.threat_detected:
        message = f"🚨 Threat Detected: {result.threat_type}\n"
        message += f"Confidence: {result.confidence_score:.2%}\n"
        message += f"Recommendations: {', '.join(result.recommendations)}"
        
        slack.send_message(message, channel="#security-alerts")
```

#### Custom REST API
```python
from sentinelgem import CustomIntegration

class MySecurityTool(CustomIntegration):
    def __init__(self, api_endpoint, api_key):
        self.endpoint = api_endpoint
        self.api_key = api_key
    
    def send_threat_data(self, analysis_result):
        payload = {
            "threat_type": analysis_result.threat_type,
            "confidence": analysis_result.confidence_score,
            "timestamp": analysis_result.timestamp,
            "indicators": analysis_result.indicators
        }
        
        response = requests.post(
            f"{self.endpoint}/threats",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        return response.json()
```

---

## Troubleshooting

### Common Issues

#### Model Loading Problems

**Issue**: Models fail to load or run out of memory
```
Error: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. **Use smaller model variant**:
   ```bash
   sentinelgem config --set model.variant=2b
   ```

2. **Enable model quantization**:
   ```yaml
   gemma_model:
     quantization: true
     precision: "int8"
   ```

3. **Force CPU usage**:
   ```bash
   sentinelgem analyze --device cpu --input file.txt
   ```

#### Performance Issues

**Issue**: Analysis takes too long or system becomes unresponsive

**Solutions**:
1. **Adjust batch size**:
   ```bash
   sentinelgem batch --batch-size 2 --input ./files/
   ```

2. **Limit concurrent processes**:
   ```bash
   sentinelgem batch --max-workers 2 --input ./files/
   ```

3. **Enable result caching**:
   ```yaml
   performance:
     enable_cache: true
     cache_ttl: 3600  # 1 hour
   ```

#### Network and Connectivity

**Issue**: Model downloads fail or API requests timeout

**Solutions**:
1. **Configure proxy settings**:
   ```bash
   export HTTPS_PROXY=http://proxy.company.com:8080
   sentinelgem download-models
   ```

2. **Use offline mode**:
   ```bash
   sentinelgem analyze --offline --input file.txt
   ```

3. **Increase timeout values**:
   ```yaml
   network:
     timeout: 300  # 5 minutes
     retries: 3
   ```

### Diagnostic Tools

#### System Health Check
```bash
# Run comprehensive system check
sentinelgem diagnose --full

# Check specific components
sentinelgem diagnose --component models
sentinelgem diagnose --component gpu
sentinelgem diagnose --component storage
```

#### Performance Profiling
```bash
# Profile analysis performance
sentinelgem analyze --input file.txt --profile --output profile.json

# Generate performance report
sentinelgem report --profile profile.json --format html
```

#### Log Analysis
```bash
# Enable debug logging
sentinelgem analyze --input file.txt --log-level debug --log-file debug.log

# View real-time logs
tail -f ~/.sentinelgem/logs/sentinelgem.log
```

### Getting Help

#### Documentation Resources
- **User Manual**: Comprehensive usage guide (this document)
- **API Reference**: Detailed API documentation
- **Developer Guide**: Technical implementation details
- **FAQ**: Frequently asked questions

#### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Discord Community**: Real-time chat support
- **Stack Overflow**: Tagged questions and answers
- **Email Support**: technical-support@sentinelgem.ai

#### Professional Support
- **Enterprise Support**: Dedicated technical support
- **Custom Development**: Tailored solutions and integrations
- **Training Services**: On-site training and workshops
- **Consulting**: Security assessment and implementation guidance

---

## Best Practices

### Security Considerations

#### Data Privacy
- **Minimize Data Retention**: Only keep analysis results as long as necessary
- **Encrypt Sensitive Data**: Use AES-256 encryption for stored data
- **Access Controls**: Implement role-based access control
- **Audit Logging**: Maintain comprehensive audit trails

#### Safe Testing
- **Isolated Environment**: Test in sandboxed environments
- **Sample Data**: Use synthetic or sanitized test data
- **Limited Permissions**: Run with minimal required privileges
- **Regular Updates**: Keep models and dependencies updated

### Performance Optimization

#### Model Selection
- **Fast Analysis**: Use Gemma 3n 2B for real-time processing
- **Balanced Performance**: Use Gemma 3n 9B for general use
- **Maximum Accuracy**: Use Gemma 3n 27B for critical analysis

#### Resource Management
```yaml
# Optimal configuration for different scenarios
production:
  model_cache_size: 2  # Keep 2 models in memory
  max_concurrent_analyses: 4
  result_cache_ttl: 1800  # 30 minutes

development:
  model_cache_size: 1
  max_concurrent_analyses: 2
  result_cache_ttl: 300  # 5 minutes
```

#### Monitoring and Alerting
```python
# Set up performance monitoring
from sentinelgem.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(
    metrics_endpoint="http://prometheus:9090",
    alert_thresholds={
        "analysis_time": 5.0,  # seconds
        "memory_usage": 0.8,   # 80% of available memory
        "error_rate": 0.02     # 2% error rate
    }
)

monitor.start()
```

### Operational Excellence

#### Backup and Recovery
```bash
# Backup configuration and models
sentinelgem backup --output ./backup/sentinelgem-backup-$(date +%Y%m%d).tar.gz

# Restore from backup
sentinelgem restore --input ./backup/sentinelgem-backup-20241201.tar.gz
```

#### Version Management
```bash
# Check current version
sentinelgem version

# Update to latest version
sentinelgem update --check
sentinelgem update --install

# Rollback to previous version
sentinelgem rollback --version 1.2.0
```

#### Health Monitoring
```python
# Automated health checks
from sentinelgem.health import HealthChecker

checker = HealthChecker(
    check_interval=300,  # 5 minutes
    alert_email="admin@company.com"
)

# Define custom health checks
@checker.register_check("model_availability")
def check_models():
    return all(model.is_available() for model in loaded_models)

checker.start_monitoring()
```

### Quality Assurance

#### Result Validation
```python
# Implement result validation
from sentinelgem.validation import ResultValidator

validator = ResultValidator(
    confidence_threshold=0.7,
    require_evidence=True,
    max_processing_time=10.0
)

# Validate analysis results
if validator.validate(analysis_result):
    process_result(analysis_result)
else:
    flag_for_manual_review(analysis_result)
```

#### Continuous Improvement
- **Regular Model Updates**: Update models monthly
- **Performance Tracking**: Monitor accuracy metrics
- **Feedback Integration**: Collect user feedback for improvements
- **A/B Testing**: Test new models against current versions

---

## Conclusion

SentinelGem provides a comprehensive platform for AI-powered threat analysis across multiple modalities. This user manual covers the essential aspects of installation, configuration, and usage to help you maximize the platform's effectiveness in your security operations.

For additional support, advanced configurations, or enterprise features, please refer to the developer documentation or contact our support team.

**Remember**: Regular updates, proper configuration, and adherence to security best practices are essential for maintaining optimal performance and security posture.

---

*This manual is regularly updated to reflect new features and improvements. Please check for updates periodically or subscribe to our release notifications.*
