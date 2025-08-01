# SentinelGem Test Assets Documentation

## Overview
This directory contains carefully crafted test samples designed to validate SentinelGem's threat detection capabilities across multiple modalities. Each asset is documented with expected detection outcomes and threat indicators.

## Audio Test Samples

### 1. social_engineering_call.wav
**Purpose**: Test social engineering detection capabilities
- **Duration**: 15 seconds
- **Content**: Simulated scam call with authority impersonation
- **Expected Result**: HIGH threat confidence (0.75-0.85)
- **Key Indicators**: Urgency tactics, credential requests, stress patterns
- **Use Cases**: Vishing detection, scam call identification

### 2. legitimate_call.wav  
**Purpose**: Baseline sample for false positive testing
- **Duration**: 15 seconds
- **Content**: Professional customer service interaction
- **Expected Result**: LOW threat confidence (0.05-0.15)
- **Key Indicators**: Professional tone, standard procedures
- **Use Cases**: System calibration, accuracy validation

## Text Test Samples

### 3. phishing_email_sample.txt
**Purpose**: Email phishing detection validation
- **Content**: PayPal account suspension scam
- **Expected Result**: HIGH threat confidence (0.80-0.90)
- **Key Indicators**: Urgency language, suspicious URLs, credential harvesting
- **Use Cases**: Email security, phishing awareness training

### 4. example_logs.txt
**Purpose**: System log analysis and malware detection
- **Content**: Mixed legitimate and suspicious system events
- **Expected Result**: Multiple threat classifications
- **Key Indicators**: Failed logins, suspicious processes, network anomalies
- **Use Cases**: SIEM integration, incident response

## Validation Criteria

### Threat Detection Thresholds
- **High Confidence**: 0.70+ (Clear threat detected)
- **Medium Confidence**: 0.40-0.69 (Suspicious activity)
- **Low Confidence**: 0.00-0.39 (Likely benign)

### Performance Benchmarks
- **Processing Time**: < 3 seconds per sample
- **Accuracy Target**: > 90% correct classification
- **False Positive Rate**: < 5% on benign samples

## Testing Usage

### Automated Testing
```bash
# Run full test suite
python -m pytest tests/ -v

# Test specific modality
python main.py --test-audio assets/social_engineering_call.wav
python main.py --test-text assets/phishing_email_sample.txt
```

### Manual Validation
```bash
# Interactive analysis
python main.py --input-file assets/social_engineering_call.wav --verbose
```

## Asset Maintenance

### Quality Assurance
- All samples validated by cybersecurity experts
- Regular updates to reflect current threat landscape
- Comprehensive metadata documentation maintained

### Expansion Plan
- Add visual phishing screenshots
- Include BEC (Business Email Compromise) samples  
- Create multilingual threat samples
- Develop advanced persistent threat scenarios

## Security Note
These samples are synthetic and created solely for testing purposes. They contain no real personal information or actual malicious code.
