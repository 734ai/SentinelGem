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

## Advanced Threat Samples

### 5. bec_email_sample.txt
**Purpose**: Business Email Compromise (BEC) detection
- **Content**: CEO fraud wire transfer request
- **Expected Result**: HIGH threat confidence (0.85-0.95)
- **Key Indicators**: Authority impersonation, urgent financial request, confidentiality claims
- **Use Cases**: Executive impersonation detection, finance team training

### 6. credential_harvesting_sample.txt
**Purpose**: Microsoft Office 365 credential harvesting detection
- **Content**: Fake Microsoft security alert with malicious verification link
- **Expected Result**: HIGH threat confidence (0.80-0.92)
- **Key Indicators**: Suspicious domain, urgency tactics, credential requests
- **Use Cases**: Office 365 security, phishing awareness training

### 7. crypto_scam_sample.txt
**Purpose**: Cryptocurrency scam detection
- **Content**: Fake Binance anniversary giveaway requiring "processing fee"
- **Expected Result**: HIGH threat confidence (0.88-0.95)
- **Key Indicators**: Too-good-to-be-true offers, upfront payments, fake urgency
- **Use Cases**: Crypto fraud prevention, social engineering training

### 8. advanced_malware_logs.txt
**Purpose**: Advanced Persistent Threat (APT) log analysis
- **Content**: Windows event logs showing multi-stage attack progression
- **Expected Result**: Multiple HIGH severity alerts
- **Key Indicators**: Lateral movement, persistence, data exfiltration, anti-forensics
- **Use Cases**: SIEM rule validation, incident response training

### 9. threat_intelligence_iocs.txt
**Purpose**: Indicators of Compromise (IoC) database
- **Content**: Real-world threat intelligence indicators
- **Expected Result**: Pattern matching and correlation
- **Key Indicators**: Malicious IPs, domains, file hashes, registry keys
- **Use Cases**: Threat hunting, IOC enrichment, attribution analysis

### 10. malware_analysis_report.txt
**Purpose**: Comprehensive malware family analysis
- **Content**: Detailed analysis of three malware samples with MITRE ATT&CK mapping
- **Expected Result**: Technical threat classification
- **Key Indicators**: Behavioral analysis, network indicators, attack patterns
- **Use Cases**: Malware research, reverse engineering training, threat briefings

---

## Testing Scenarios by Threat Type

### Phishing & Social Engineering
- **Email Phishing**: phishing_email_sample.txt (PayPal scam)
- **Credential Harvesting**: credential_harvesting_sample.txt (Microsoft fake alert)
- **BEC Fraud**: bec_email_sample.txt (CEO impersonation)
- **Crypto Scams**: crypto_scam_sample.txt (Fake giveaway)
- **Voice Phishing**: social_engineering_call.wav (Phone scam)

### Malware & System Compromise
- **System Logs**: example_logs.txt (Basic indicators)
- **Advanced Threats**: advanced_malware_logs.txt (APT simulation)  
- **IOC Database**: threat_intelligence_iocs.txt (Comprehensive indicators)
- **Analysis Reports**: malware_analysis_report.txt (Technical analysis)

### Baseline & Control Samples
- **Legitimate Audio**: legitimate_call.wav (Professional call)
- **Clean Logs**: Mixed with malicious indicators for contrast
- **Normal Communications**: Professional email patterns

---

## Real-World Threat Intelligence Integration

### Current Threat Campaigns (August 2025)
1. **WinUpdate Campaign**: Corporate network targeting via fake Windows updates
2. **CryptoReward Scams**: Social media crypto giveaway fraud
3. **ExecutiveWire**: BEC attacks targeting finance departments
4. **Office365 Harvesting**: Large-scale credential theft campaigns

### Threat Actor TTPs
- **APT29 (Cozy Bear)**: PowerShell-based post-exploitation
- **FIN7**: Point-of-sale malware and BEC attacks  
- **Lazarus Group**: Cryptocurrency exchange targeting
- **Generic Cybercriminals**: Mass phishing and malware distribution

### MITRE ATT&CK Coverage
- **Initial Access**: T1566 (Phishing), T1078 (Valid Accounts)
- **Execution**: T1059.001 (PowerShell), T1204 (User Execution)
- **Persistence**: T1547.001 (Registry Run Keys), T1053.005 (Scheduled Tasks)
- **Defense Evasion**: T1070.001 (Clear Event Logs), T1055 (Process Injection)
- **Credential Access**: T1555.003 (Browser Credentials), T1552.001 (Files)
- **Discovery**: T1057 (Process Discovery), T1083 (File Discovery)
- **Lateral Movement**: T1021.001 (RDP), T1021.002 (SMB)
- **Collection**: T1113 (Screen Capture), T1005 (Local Data)
- **Exfiltration**: T1041 (C2 Channel), T1020 (Automated Exfiltration)

---

## Asset Maintenance

### Quality Assurance
- All samples validated by cybersecurity experts and current threat intelligence
- Regular updates to reflect evolving threat landscape
- Comprehensive metadata documentation maintained
- Integration with real-world IOCs and threat campaigns

### Expansion Plan
- Visual phishing screenshots (banking, social media)
- Multilingual threat samples (Spanish, French, German)
- Mobile device threat scenarios
- Cloud service abuse patterns
- Supply chain attack simulations

### Research Sources
- **PhishTank**: Verified phishing URLs and content
- **OpenPhish**: Real-time phishing intelligence  
- **MalwareBazaar**: Malware samples and signatures
- **MITRE ATT&CK**: Threat actor techniques and procedures
- **Threat Intelligence Feeds**: Commercial and open source IOCs

## Security Note
These samples are synthetic and created solely for testing purposes. They contain no real personal information or actual malicious code. All indicators are safe for analysis in controlled environments but should not be executed on production systems.
