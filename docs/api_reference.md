# SentinelGem API Reference Documentation
# Version: 1.0.0 
# Last Updated: 2025

# Table of Contents
## 1. Overview
## 2. Authentication
## 3. Core API Endpoints
## 4. Analysis Endpoints
## 5. Configuration Endpoints
## 6. Status & Health Endpoints
## 7. Webhook Endpoints
## 8. Data Models & Schemas
## 9. Error Handling
## 10. Rate Limiting
## 11. SDKs & Client Libraries
## 12. Integration Examples

---

## 1. Overview

The SentinelGem API provides comprehensive threat analysis capabilities through a RESTful interface. Our platform combines multiple AI models to detect phishing attempts, social engineering attacks, malicious content, and suspicious patterns across text, images, audio, and URLs.

### Base URL
```
https://api.sentinelgem.com/v1
```

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`
- **File Uploads**: `multipart/form-data`

### API Versioning
All API endpoints are versioned using URL path versioning (e.g., `/v1/`). The current stable version is `v1`.

---

## 2. Authentication

SentinelGem uses API key authentication with support for multiple authentication methods.

### API Key Authentication

Include your API key in the request header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Authentication Endpoints

#### Generate API Key
```http
POST /v1/auth/keys
```

**Request Body:**
```json
{
  "name": "My Application Key",
  "permissions": ["analyze", "configure"],
  "rate_limit": 1000,
  "expires_at": "2024-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "key_id": "key_123456789",
  "api_key": "sk_live_abcdef123456789",
  "name": "My Application Key",
  "permissions": ["analyze", "configure"],
  "rate_limit": 1000,
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2024-12-31T23:59:59Z"
}
```

#### List API Keys
```http
GET /v1/auth/keys
```

#### Revoke API Key
```http
DELETE /v1/auth/keys/{key_id}
```

---

## 3. Core API Endpoints

### Universal Analysis Endpoint

The primary endpoint for analyzing any type of content.

#### Analyze Content
```http
POST /v1/analyze
```

**Parameters:**
- `content` (string): Text content to analyze
- `url` (string): URL to analyze  
- `file` (file): Image, audio, or document file
- `analysis_types` (array): Specific analysis types to run
- `priority` (string): Analysis priority (`low`, `normal`, `high`)
- `callback_url` (string): Webhook URL for async results

**Request Examples:**

**Text Analysis:**
```json
{
  "content": "Urgent! Your account will be suspended. Click here immediately: http://suspicious-link.com",
  "analysis_types": ["phishing", "social_engineering", "url_analysis"],
  "priority": "high"
}
```

**URL Analysis:**
```json
{
  "url": "https://example.com/suspicious-page",
  "analysis_types": ["url_analysis", "content_scraping"],
  "priority": "normal"
}
```

**File Upload:**
```bash
curl -X POST https://api.sentinelgem.com/v1/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@suspicious_image.png" \
  -F "analysis_types=[\"ocr\", \"phishing\"]" \
  -F "priority=high"
```

**Response:**
```json
{
  "analysis_id": "analysis_123456789",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:00:05Z",
  "results": {
    "overall_risk_score": 0.85,
    "threat_category": "phishing",
    "confidence_score": 0.92,
    "analysis_details": {
      "phishing": {
        "detected": true,
        "confidence": 0.89,
        "indicators": [
          "urgent_language",
          "suspicious_url",
          "account_threat"
        ],
        "risk_factors": [
          {
            "type": "urgent_language",
            "description": "Contains urgent action language",
            "severity": "high"
          },
          {
            "type": "suspicious_url", 
            "description": "URL domain mismatch",
            "severity": "critical"
          }
        ]
      },
      "social_engineering": {
        "detected": true,
        "confidence": 0.78,
        "techniques": ["urgency", "authority", "fear"],
        "psychological_indicators": [
          "Creates false sense of urgency",
          "Appeals to authority (account suspension)",
          "Induces fear of consequences"
        ]
      },
      "url_analysis": {
        "url": "http://suspicious-link.com",
        "is_malicious": true,
        "confidence": 0.94,
        "reputation_score": 0.15,
        "categories": ["phishing", "malicious"],
        "domain_age": 2,
        "ssl_valid": false,
        "redirects": [
          {
            "from": "http://suspicious-link.com",
            "to": "http://phishing-site.evil",
            "status": 302
          }
        ]
      }
    },
    "metadata": {
      "processing_time_ms": 1250,
      "models_used": ["gemma-3n", "url-classifier", "domain-reputation"],
      "analysis_version": "1.0.0"
    }
  }
}
```

### Batch Analysis

Analyze multiple items in a single request.

#### Batch Analyze
```http
POST /v1/analyze/batch
```

**Request Body:**
```json
{
  "items": [
    {
      "id": "item_1",
      "content": "First suspicious message",
      "analysis_types": ["phishing"]
    },
    {
      "id": "item_2", 
      "url": "https://suspicious-site.com",
      "analysis_types": ["url_analysis"]
    }
  ],
  "priority": "normal",
  "callback_url": "https://your-app.com/webhook"
}
```

**Response:**
```json
{
  "batch_id": "batch_123456789",
  "status": "processing",
  "total_items": 2,
  "created_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:00:30Z",
  "results_url": "/v1/analyze/batch/batch_123456789/results"
}
```

---

## 4. Analysis Endpoints

### Specialized Analysis Endpoints

#### Text Analysis
```http
POST /v1/analyze/text
```

**Request Body:**
```json
{
  "text": "Your message content here",
  "language": "en",
  "analysis_types": ["phishing", "social_engineering", "sentiment"],
  "context": {
    "source": "email",
    "sender": "unknown@domain.com",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### Image Analysis  
```http
POST /v1/analyze/image
```

**Parameters:**
- `image` (file): Image file to analyze
- `analysis_types` (array): Types of analysis (`ocr`, `phishing`, `brand_impersonation`)
- `extract_text` (boolean): Whether to extract text via OCR

**Response:**
```json
{
  "analysis_id": "img_123456789",
  "results": {
    "ocr": {
      "extracted_text": "Please verify your account by clicking here",
      "confidence": 0.95,
      "text_regions": [
        {
          "text": "Please verify your account",
          "bbox": [10, 20, 300, 50],
          "confidence": 0.98
        }
      ]
    },
    "phishing": {
      "detected": true,
      "confidence": 0.87,
      "visual_indicators": ["fake_login_form", "brand_impersonation"]
    },
    "brand_impersonation": {
      "detected_brand": "PayPal", 
      "confidence": 0.92,
      "legitimate": false,
      "impersonation_indicators": [
        "Logo similarity: 0.89",
        "Color scheme match: 0.94",
        "Typography mismatch: 0.73"
      ]
    }
  }
}
```

#### Audio Analysis
```http
POST /v1/analyze/audio
```

**Parameters:**
- `audio` (file): Audio file (WAV, MP3, M4A)
- `transcribe` (boolean): Enable speech-to-text
- `detect_social_engineering` (boolean): Analyze for social engineering

**Response:**
```json
{
  "analysis_id": "audio_123456789",
  "results": {
    "transcription": {
      "text": "Hello, this is your bank calling about suspicious activity...",
      "confidence": 0.93,
      "language": "en",
      "duration_seconds": 45.2,
      "segments": [
        {
          "text": "Hello, this is your bank",
          "start_time": 0.0,
          "end_time": 2.5,
          "confidence": 0.95
        }
      ]
    },
    "social_engineering": {
      "detected": true,
      "confidence": 0.82,
      "techniques": ["authority", "urgency", "fear"],
      "voice_analysis": {
        "stress_indicators": 0.67,
        "speech_patterns": "deceptive",
        "background_noise": "call_center"
      },
      "language_analysis": {
        "urgency_score": 0.89,
        "authority_claims": ["bank representative", "immediate action required"],
        "suspicious_phrases": [
          "verify your account immediately",
          "suspicious activity detected",
          "account will be frozen"
        ]
      }
    }
  }
}
```

#### URL Analysis
```http
POST /v1/analyze/url
```

**Request Body:**
```json
{
  "url": "https://example.com/page",
  "deep_scan": true,
  "include_screenshot": true,
  "follow_redirects": true,
  "max_redirects": 5
}
```

**Response:**
```json
{
  "analysis_id": "url_123456789", 
  "results": {
    "url_classification": {
      "category": "phishing",
      "confidence": 0.91,
      "risk_score": 0.88
    },
    "domain_analysis": {
      "domain": "example.com",
      "registrar": "GoDaddy",
      "creation_date": "2024-01-01",
      "expiry_date": "2025-01-01",
      "age_days": 30,
      "reputation_score": 0.25,
      "is_newly_registered": true
    },
    "content_analysis": {
      "title": "Verify Your Account - Secure Banking",
      "description": "Urgent account verification required",
      "has_login_form": true,
      "suspicious_elements": [
        "fake_ssl_indicators",
        "urgent_language",
        "credential_harvesting_form"
      ]
    },
    "technical_analysis": {
      "ssl_certificate": {
        "valid": false,
        "issuer": "Unknown CA",
        "expires": "2024-02-01"
      },
      "redirects": [
        {
          "from": "https://example.com/page",
          "to": "http://phishing-site.evil",
          "status_code": 302
        }
      ],
      "response_time_ms": 1250,
      "server_location": "Unknown",
      "hosting_provider": "Suspicious Hosting Ltd"
    },
    "screenshot": {
      "url": "https://api.sentinelgem.com/v1/screenshots/url_123456789.png",
      "analysis": {
        "visual_similarity": 0.92,
        "brand_detected": "PayPal",
        "legitimate_brand": false
      }
    }
  }
}
```

---

## 5. Configuration Endpoints

### Detection Thresholds

#### Get Detection Settings
```http
GET /v1/config/detection
```

#### Update Detection Settings  
```http
PUT /v1/config/detection
```

**Request Body:**
```json
{
  "phishing_threshold": 0.75,
  "social_engineering_threshold": 0.70,
  "url_reputation_threshold": 0.60,
  "confidence_threshold": 0.80,
  "auto_block_threshold": 0.90,
  "notification_threshold": 0.65
}
```

### Custom Models

#### Upload Custom Model
```http
POST /v1/config/models
```

#### List Custom Models
```http
GET /v1/config/models
```

### Whitelist/Blacklist Management

#### Add to Whitelist
```http
POST /v1/config/whitelist
```

**Request Body:**
```json
{
  "type": "domain",
  "value": "trusted-domain.com",
  "reason": "Corporate partner domain",
  "expires_at": "2024-12-31T23:59:59Z"
}
```

#### Add to Blacklist
```http
POST /v1/config/blacklist
```

**Request Body:**
```json
{
  "type": "url",
  "value": "https://known-phishing-site.com",
  "reason": "Confirmed phishing site",
  "severity": "high"
}
```

---

## 6. Status & Health Endpoints

### Analysis Status

#### Get Analysis Status
```http
GET /v1/analysis/{analysis_id}/status
```

**Response:**
```json
{
  "analysis_id": "analysis_123456789",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:01Z", 
  "completed_at": "2024-01-01T12:00:05Z",
  "processing_time_ms": 4250,
  "queue_position": null,
  "estimated_completion": null
}
```

#### Get Analysis Results
```http  
GET /v1/analysis/{analysis_id}/results
```

### System Health

#### Health Check
```http  
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "services": {
    "api": "healthy",
    "database": "healthy", 
    "ai_models": "healthy",
    "queue": "healthy"
  },
  "performance": {
    "avg_response_time_ms": 150,
    "requests_per_second": 25.5,
    "cpu_usage": 0.45,
    "memory_usage": 0.67
  }
}
```

#### System Status
```http
GET /v1/status
```

**Response:**
```json
{
  "system_status": "operational",
  "last_updated": "2024-01-01T12:00:00Z",
  "incidents": [],
  "scheduled_maintenance": [],
  "model_status": {
    "gemma-3n": "online",
    "whisper": "online", 
    "tesseract": "online",
    "url-classifier": "online"
  },
  "api_performance": {
    "success_rate": 99.8,
    "avg_latency_ms": 145,
    "throughput_rpm": 1500
  }
}
```

---

## 7. Webhook Endpoints

### Webhook Configuration

#### Create Webhook
```http
POST /v1/webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook/sentinelgem",
  "events": ["analysis.completed", "analysis.failed", "threshold.exceeded"],
  "secret": "your-webhook-secret",
  "active": true,
  "retry_config": {
    "max_retries": 3,
    "retry_delay_seconds": 60
  }
}
```

#### List Webhooks
```http
GET /v1/webhooks
```

#### Update Webhook
```http
PUT /v1/webhooks/{webhook_id}
```

#### Delete Webhook  
```http
DELETE /v1/webhooks/{webhook_id}
```

### Webhook Events

#### Analysis Completed
```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "analysis_id": "analysis_123456789",
    "status": "completed",
    "results": {
      "overall_risk_score": 0.85,
      "threat_category": "phishing"
    }
  }
}
```

#### High Risk Detection
```json
{
  "event": "threshold.exceeded", 
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "analysis_id": "analysis_123456789",
    "risk_score": 0.95,
    "threshold": 0.90,
    "threat_category": "phishing",
    "urgent": true
  }
}
```

---

## 8. Data Models & Schemas

### Analysis Result Schema

```json
{
  "type": "object",
  "properties": {
    "analysis_id": {"type": "string"},
    "status": {"enum": ["pending", "processing", "completed", "failed"]},
    "created_at": {"type": "string", "format": "date-time"},
    "completed_at": {"type": "string", "format": "date-time"},
    "results": {
      "type": "object",
      "properties": {
        "overall_risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "threat_category": {"enum": ["phishing", "social_engineering", "malware", "spam", "benign"]},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "analysis_details": {"type": "object"}
      }
    }
  }
}
```

### Error Response Schema

```json
{
  "type": "object", 
  "properties": {
    "error": {
      "type": "object",
      "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "details": {"type": "object"},
        "request_id": {"type": "string"}
      },
      "required": ["code", "message"]
    }
  }
}
```

---

## 9. Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request  
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Unprocessable Entity
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request body contains invalid parameters",
    "details": {
      "field": "content",
      "reason": "Content cannot be empty"
    },
    "request_id": "req_123456789",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### Common Error Codes

- `INVALID_API_KEY` - API key is missing or invalid
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INVALID_REQUEST` - Request validation failed
- `UNSUPPORTED_FILE_TYPE` - File format not supported
- `FILE_TOO_LARGE` - File exceeds size limit
- `ANALYSIS_FAILED` - Analysis processing error
- `INSUFFICIENT_PERMISSIONS` - API key lacks required permissions

---

## 10. Rate Limiting

### Rate Limits

- **Free Tier**: 100 requests/hour
- **Basic Plan**: 1,000 requests/hour  
- **Pro Plan**: 10,000 requests/hour
- **Enterprise**: Custom limits

### Rate Limit Headers

All API responses include rate limiting headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
X-RateLimit-Retry-After: 3600
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": "1 hour",
      "retry_after": 3600
    }
  }
}
```

---

## 11. SDKs & Client Libraries

### Official SDKs

#### Python SDK
```bash
pip install sentinelgem-python
```

```python
from sentinelgem import SentinelGem

client = SentinelGem(api_key="your-api-key")

# Analyze text
result = client.analyze_text("Suspicious message content")
print(f"Risk Score: {result.risk_score}")
```

#### JavaScript/Node.js SDK
```bash
npm install sentinelgem-js
```

```javascript
const SentinelGem = require('sentinelgem-js');

const client = new SentinelGem('your-api-key');

// Analyze URL
client.analyzeUrl('https://suspicious-site.com')
  .then(result => {
    console.log(`Risk Score: ${result.riskScore}`);
  });
```

#### Go SDK
```bash
go get github.com/sentinelgem/sentinelgem-go
```

```go
package main

import (
    "github.com/sentinelgem/sentinelgem-go"
)

func main() {
    client := sentinelgem.NewClient("your-api-key")
    
    result, err := client.AnalyzeText("Suspicious content")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Risk Score: %.2f\n", result.RiskScore)
}
```

---

## 12. Integration Examples

### Email Security Integration

```python
import email
from sentinelgem import SentinelGem

def analyze_email(raw_email):
    client = SentinelGem(api_key="your-api-key")
    
    # Parse email
    msg = email.message_from_string(raw_email)
    
    # Analyze email content
    result = client.analyze({
        "content": msg.get_payload(),
        "analysis_types": ["phishing", "social_engineering"],
        "context": {
            "source": "email",
            "sender": msg.get("From"),
            "subject": msg.get("Subject")
        }
    })
    
    # Take action based on risk score
    if result.risk_score > 0.8:
        quarantine_email(msg)
    elif result.risk_score > 0.6:
        flag_for_review(msg)
    
    return result
```

### Web Security Integration

```javascript
// Express.js middleware for URL analysis
const express = require('express');
const SentinelGem = require('sentinelgem-js');

const app = express();
const sentinel = new SentinelGem('your-api-key');

app.use(async (req, res, next) => {
    const urlParams = req.query.url || req.body.url;
    
    if (urlParams) {
        try {
            const result = await sentinel.analyzeUrl(urlParams);
            
            if (result.riskScore > 0.9) {
                return res.status(403).json({
                    error: 'Blocked: High-risk URL detected',
                    riskScore: result.riskScore
                });
            }
            
            req.sentinelResult = result;
        } catch (error) {
            console.error('SentinelGem analysis failed:', error);
        }
    }
    
    next();
});
```

### Slack Bot Integration

```python
from slack_bolt import App
from sentinelgem import SentinelGem

app = App(token="your-slack-token")
sentinel = SentinelGem(api_key="your-api-key")

@app.message()
def analyze_message(message, say, client):
    text = message.get('text', '')
    
    # Analyze message content
    result = sentinel.analyze_text(text)
    
    if result.risk_score > 0.7:
        # Alert security team
        client.chat_postMessage(
            channel="#security-alerts",
            text=f"🚨 Suspicious message detected (Risk: {result.risk_score:.2f})\n"
                 f"User: <@{message['user']}>\n"
                 f"Channel: <#{message['channel']}>\n"
                 f"Message: {text[:100]}..."
        )
        
        # Warn user
        say(f"⚠️ This message may contain suspicious content. Please verify any links before clicking.")
```

### File Upload Security

```python
from flask import Flask, request, jsonify
from sentinelgem import SentinelGem

app = Flask(__name__)
sentinel = SentinelGem(api_key="your-api-key")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Analyze uploaded file
    result = sentinel.analyze_file(
        file=file,
        analysis_types=['ocr', 'phishing', 'malware']
    )
    
    if result.risk_score > 0.8:
        return jsonify({
            'status': 'blocked',
            'reason': 'High-risk content detected',
            'risk_score': result.risk_score,
            'threats': result.threats
        }), 403
    
    # Process file normally
    return jsonify({
        'status': 'success',
        'risk_score': result.risk_score,
        'file_id': save_file(file)
    })
```

---

## Support & Resources

### Documentation
- [API Reference](https://docs.sentinelgem.com/api)
- [Integration Guides](https://docs.sentinelgem.com/guides)
- [Security Best Practices](https://docs.sentinelgem.com/security)

### Support Channels
- **Email**: api-support@sentinelgem.com
- **Discord**: [SentinelGem Community](https://discord.gg/sentinelgem)
- **GitHub**: [Issues & Discussions](https://github.com/sentinelgem/api)

### API Status
- **Status Page**: https://status.sentinelgem.com
- **Uptime**: 99.9% SLA
- **Response Time**: <200ms average

---

*This documentation is version 1.0.0 and was last updated on January 1, 2024. For the most current version, please visit our [documentation portal](https://docs.sentinelgem.com).*
