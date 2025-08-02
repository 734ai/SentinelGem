#!/usr/bin/env python3
"""
SentinelGem Log Parser
Author: Muzan Sano

Advanced log parsing module for detecting malware, intrusion attempts,
and suspicious system activity through pattern analysis and AI reasoning.
"""

import re
import os
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

from rich.console import Console
from rich.progress import track

# Import SentinelGem components
from .inference import ThreatAnalysis, get_inference_engine
from .utils import Timer, validate_input_file, clean_text

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Represents a parsed log entry"""
    timestamp: datetime
    level: str
    source: str
    message: str
    raw_line: str
    metadata: Dict[str, Any]

@dataclass
class LogAnalysisResult:
    """Results from log analysis"""
    total_entries: int
    suspicious_entries: List[LogEntry]
    threat_indicators: Dict[str, List[str]]
    timeline_analysis: Dict[str, Any]
    ai_analysis: ThreatAnalysis
    confidence_score: float

class LogParser:
    """
    Advanced log parser for cybersecurity threat detection
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or "config/rules.yaml"
        self.inference = get_inference_engine()
        
        # Load detection rules
        self.rules = self._load_rules()
        
        # Log format patterns
        self.log_patterns = self._initialize_log_patterns()
        
        # Threat indicators
        self.threat_indicators = self._load_threat_indicators()
        
        logger.info("Log Parser initialized")
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load threat detection rules"""
        try:
            rules_file = Path(self.rules_path)
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Rules file not found: {rules_file}")
                return self._get_default_rules()
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Default threat detection rules"""
        return {
            'malware': {
                'suspicious_processes': [
                    r'powershell.*-enc.*',
                    r'cmd.*\/c.*echo.*',
                    r'regsvr32.*scrobj\.dll',
                    r'rundll32.*javascript:',
                    r'wscript.*\.vbs',
                    r'cscript.*\.js',
                    r'bitsadmin.*\/transfer',
                    r'certutil.*-decode'
                ],
                'file_indicators': [
                    r'\\temp\\.*\.exe',
                    r'\\appdata\\roaming\\.*\.exe',
                    r'.*\.scr$',
                    r'.*\.bat$',
                    r'.*\.vbs$',
                    r'.*\.ps1$'
                ],
                'network_indicators': [
                    r'connection.*refused',
                    r'tcp.*established.*:[0-9]{4,5}',
                    r'dns.*query.*suspicious',
                    r'http.*[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
                ],
                'registry_indicators': [
                    r'hklm\\.*\\run',
                    r'hkcu\\.*\\run',
                    r'hklm\\software\\microsoft\\windows\\currentversion\\run'
                ]
            },
            'intrusion': {
                'failed_logins': r'failed.*login.*attempt',
                'privilege_escalation': r'privilege.*escalat',
                'lateral_movement': r'net.*use.*\$',
                'data_exfiltration': r'copy.*network.*path'
            },
            'system_anomalies': {
                'unusual_processes': r'process.*not.*found',
                'service_failures': r'service.*failed.*start',
                'disk_anomalies': r'disk.*full|corrupt.*file',
                'memory_issues': r'memory.*leak|out.*of.*memory'
            }
        }
    
    def _initialize_log_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize common log format patterns"""
        return {
            'syslog': re.compile(
                r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+)\s+'
                r'(?P<hostname>\S+)\s+'
                r'(?P<process>\S+):\s+'
                r'(?P<message>.*)'
            ),
            'windows_event': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
                r'\[(?P<level>\w+)\]\s+'
                r'(?P<message>.*)'
            ),
            'apache_access': re.compile(
                r'(?P<ip>\S+)\s+\S+\s+\S+\s+'
                r'\[(?P<timestamp>[^\]]+)\]\s+'
                r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>\S+)"\s+'
                r'(?P<status>\d+)\s+'
                r'(?P<size>\S+)'
            ),
            'generic_timestamped': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
                r'(?:\.\d+)?\s+'
                r'\[(?P<level>\w+)\]\s+'
                r'(?P<message>.*)'
            )
        }
    
    def _load_threat_indicators(self) -> Dict[str, List[str]]:
        """Load threat indicators from rules"""
        indicators = defaultdict(list)
        
        malware_rules = self.rules.get('malware', {})
        for category, patterns in malware_rules.items():
            if isinstance(patterns, list):
                indicators[category].extend(patterns)
        
        intrusion_rules = self.rules.get('intrusion', {})
        for category, pattern in intrusion_rules.items():
            if isinstance(pattern, str):
                indicators[category].append(pattern)
        
        return dict(indicators)
    
    def parse_log_file(self, log_file_path: str) -> List[LogEntry]:
        """
        Parse log file and extract structured entries
        
        Args:
            log_file_path: Path to log file
            
        Returns:
            List of parsed log entries
        """
        try:
            # Validate input file
            validation = validate_input_file(log_file_path, max_size_mb=100)
            if not validation["valid"]:
                logger.error(f"Log file validation failed: {validation['error']}")
                return []
            
            entries = []
            
            with Timer(f"Parsing log file {Path(log_file_path).name}"):
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Try to parse with different patterns
                        entry = self._parse_log_line(line, line_num)
                        if entry:
                            entries.append(entry)
                        
                        # Limit processing for performance
                        if line_num > 10000:
                            logger.warning(f"Processing truncated at {line_num} lines")
                            break
            
            console.print(f"[green]✓ Parsed {len(entries)} log entries[/green]")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing log file {log_file_path}: {e}")
            return []
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[LogEntry]:
        """Parse a single log line"""
        for pattern_name, pattern in self.log_patterns.items():
            match = pattern.match(line)
            if match:
                groups = match.groupdict()
                
                # Parse timestamp
                timestamp = self._parse_timestamp(groups.get('timestamp', ''))
                
                return LogEntry(
                    timestamp=timestamp,
                    level=groups.get('level', 'INFO'),
                    source=groups.get('hostname', groups.get('process', 'unknown')),
                    message=groups.get('message', line),
                    raw_line=line,
                    metadata={
                        'line_number': line_num,
                        'pattern_type': pattern_name,
                        'parsed_groups': groups
                    }
                )
        
        # If no pattern matches, create generic entry
        return LogEntry(
            timestamp=datetime.now(),
            level='UNKNOWN',
            source='unknown',
            message=line,
            raw_line=line,
            metadata={
                'line_number': line_num,
                'pattern_type': 'unparsed'
            }
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats"""
        if not timestamp_str:
            return datetime.now()
        
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%b %d %H:%M:%S',
            '%d/%b/%Y:%H:%M:%S',
            '%Y/%m/%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.split()[0] + ' ' + timestamp_str.split()[1], fmt)
            except:
                continue
        
        # If all parsing fails, return current time
        return datetime.now()
    
    def detect_threats(self, entries: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """
        Detect threats in parsed log entries
        
        Args:
            entries: List of log entries to analyze
            
        Returns:
            Dictionary of threat categories and matching entries
        """
        threats = defaultdict(list)
        
        console.print("[blue]Analyzing log entries for threats...[/blue]")
        
        for entry in track(entries, description="Scanning for threats..."):
            message_lower = entry.message.lower()
            
            # Check against threat indicators
            for category, patterns in self.threat_indicators.items():
                for pattern in patterns:
                    try:
                        if re.search(pattern, message_lower, re.IGNORECASE):
                            threats[category].append(entry)
                            # Add threat info to metadata
                            entry.metadata['threat_category'] = category
                            entry.metadata['matched_pattern'] = pattern
                            break
                    except re.error:
                        continue
        
        return dict(threats)
    
    def analyze_timeline(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """
        Analyze timeline patterns in log entries
        
        Args:
            entries: List of log entries
            
        Returns:
            Timeline analysis results
        """
        if not entries:
            return {}
        
        # Sort entries by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        # Calculate time ranges
        start_time = sorted_entries[0].timestamp
        end_time = sorted_entries[-1].timestamp
        duration = end_time - start_time
        
        # Analyze entry frequency
        hourly_counts = defaultdict(int)
        level_counts = Counter()
        source_counts = Counter()
        
        for entry in entries:
            hour_key = entry.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
            level_counts[entry.level] += 1
            source_counts[entry.source] += 1
        
        # Identify suspicious patterns
        suspicious_patterns = []
        
        # High frequency periods (potential DoS or brute force)
        avg_hourly = len(entries) / max(1, duration.total_seconds() / 3600)
        for hour, count in hourly_counts.items():
            if count > avg_hourly * 3:  # 3x average
                suspicious_patterns.append({
                    'type': 'high_frequency',
                    'hour': hour,
                    'count': count,
                    'description': f'Unusually high activity: {count} events'
                })
        
        # Unusual error patterns
        error_levels = ['ERROR', 'CRITICAL', 'FATAL', 'ALERT']
        error_count = sum(level_counts[level] for level in error_levels)
        if error_count > len(entries) * 0.1:  # > 10% errors
            suspicious_patterns.append({
                'type': 'high_error_rate',
                'error_count': error_count,
                'total_count': len(entries),
                'description': f'High error rate: {error_count}/{len(entries)} ({error_count/len(entries)*100:.1f}%)'
            })
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_entries': len(entries),
            'hourly_distribution': dict(hourly_counts),
            'level_distribution': dict(level_counts),
            'source_distribution': dict(source_counts.most_common(10)),
            'suspicious_patterns': suspicious_patterns,
            'avg_entries_per_hour': avg_hourly
        }
    
    def analyze_log_file(self, log_file_path: str) -> LogAnalysisResult:
        """
        Complete log file analysis pipeline
        
        Args:
            log_file_path: Path to log file
            
        Returns:
            Complete analysis results
        """
        console.print(f"[blue]Starting comprehensive log analysis: {Path(log_file_path).name}[/blue]")
        
        try:
            # Parse log entries
            entries = self.parse_log_file(log_file_path)
            
            if not entries:
                return LogAnalysisResult(
                    total_entries=0,
                    suspicious_entries=[],
                    threat_indicators={},
                    timeline_analysis={},
                    ai_analysis=ThreatAnalysis(
                        threat_detected=False,
                        confidence_score=0.0,
                        threat_type="no_data",
                        description="No log entries could be parsed",
                        recommendations=["Check log file format and content"],
                        raw_analysis="",
                        metadata={"error": "no_parseable_entries"}
                    ),
                    confidence_score=0.0
                )
            
            # Detect threats
            threats = self.detect_threats(entries)
            
            # Analyze timeline
            timeline_analysis = self.analyze_timeline(entries)
            
            # Collect suspicious entries
            suspicious_entries = []
            threat_indicators = {}
            
            for category, threat_entries in threats.items():
                suspicious_entries.extend(threat_entries)
                threat_indicators[category] = [entry.message for entry in threat_entries[:5]]  # Top 5 examples
            
            # Prepare text for AI analysis
            threat_summary = self._create_threat_summary(threats, timeline_analysis)
            
            # AI-powered analysis
            ai_analysis = self.inference.analyze_threat(
                threat_summary,
                analysis_type="log_analysis",
                input_type="system_logs"
            )
            
            # Calculate overall confidence
            threat_score = len(suspicious_entries) / len(entries) if entries else 0
            pattern_score = len(threats) / 10 if threats else 0  # Normalize by expected categories
            combined_confidence = min((threat_score + pattern_score + ai_analysis.confidence_score) / 3, 1.0)
            
            console.print(f"[green]✓ Log analysis complete: {len(suspicious_entries)} suspicious entries found[/green]")
            
            return LogAnalysisResult(
                total_entries=len(entries),
                suspicious_entries=suspicious_entries,
                threat_indicators=threat_indicators,
                timeline_analysis=timeline_analysis,
                ai_analysis=ai_analysis,
                confidence_score=combined_confidence
            )
            
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return LogAnalysisResult(
                total_entries=0,
                suspicious_entries=[],
                threat_indicators={},
                timeline_analysis={},
                ai_analysis=ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="analysis_error",
                    description=f"Log analysis failed: {str(e)}",
                    recommendations=["Check log file and try again"],
                    raw_analysis="",
                    metadata={"error": str(e)}
                ),
                confidence_score=0.0
            )
    
    def _create_threat_summary(self, threats: Dict[str, List[LogEntry]], timeline_analysis: Dict[str, Any]) -> str:
        """Create threat summary for AI analysis"""
        summary_parts = []
        
        # Overall statistics
        total_threats = sum(len(entries) for entries in threats.values())
        summary_parts.append(f"Log Analysis Summary: {total_threats} potential threats detected")
        
        # Threat breakdown
        if threats:
            summary_parts.append("\nThreat Categories:")
            for category, entries in threats.items():
                summary_parts.append(f"- {category}: {len(entries)} incidents")
                if entries:
                    summary_parts.append(f"  Example: {entries[0].message[:100]}...")
        
        # Timeline patterns
        if timeline_analysis.get('suspicious_patterns'):
            summary_parts.append("\nSuspicious Patterns:")
            for pattern in timeline_analysis['suspicious_patterns']:
                summary_parts.append(f"- {pattern['description']}")
        
        # System activity
        level_dist = timeline_analysis.get('level_distribution', {})
        if level_dist:
            summary_parts.append(f"\nLog Level Distribution: {level_dist}")
        
        return "\n".join(summary_parts)
    
    def batch_analyze_logs(self, log_directory: str) -> List[LogAnalysisResult]:
        """
        Analyze multiple log files in a directory
        
        Args:
            log_directory: Directory containing log files
            
        Returns:
            List of analysis results
        """
        log_dir = Path(log_directory)
        if not log_dir.exists():
            logger.error(f"Log directory not found: {log_directory}")
            return []
        
        # Find log files
        log_files = []
        for pattern in ['*.log', '*.txt', '*.out']:
            log_files.extend(log_dir.glob(pattern))
        
        console.print(f"[blue]Found {len(log_files)} log files to analyze[/blue]")
        
        results = []
        for log_file in track(log_files, description="Analyzing log files..."):
            result = self.analyze_log_file(str(log_file))
            result.metadata = {'filename': log_file.name}
            results.append(result)
        
        return results
    
    def export_analysis_report(self, result: LogAnalysisResult, output_file: str):
        """
        Export analysis results to JSON report
        
        Args:
            result: Analysis results
            output_file: Output file path
        """
        try:
            # Convert LogEntry objects to dictionaries
            suspicious_entries_data = []
            for entry in result.suspicious_entries:
                suspicious_entries_data.append({
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level,
                    'source': entry.source,
                    'message': entry.message,
                    'metadata': entry.metadata
                })
            
            report_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_entries': result.total_entries,
                'suspicious_entries_count': len(result.suspicious_entries),
                'suspicious_entries': suspicious_entries_data,
                'threat_indicators': result.threat_indicators,
                'timeline_analysis': result.timeline_analysis,
                'ai_analysis': {
                    'threat_detected': result.ai_analysis.threat_detected,
                    'confidence_score': result.ai_analysis.confidence_score,
                    'threat_type': result.ai_analysis.threat_type,
                    'description': result.ai_analysis.description,
                    'recommendations': result.ai_analysis.recommendations
                },
                'overall_confidence': result.confidence_score
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            console.print(f"[green]✓ Analysis report exported: {output_file}[/green]")
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")

# Convenience functions
def analyze_log_file(log_file_path: str, rules_path: Optional[str] = None) -> LogAnalysisResult:
    """Convenience function to analyze a single log file"""
    parser = LogParser(rules_path)
    return parser.analyze_log_file(log_file_path)

def batch_analyze_logs(log_directory: str, rules_path: Optional[str] = None) -> List[LogAnalysisResult]:
    """Convenience function to analyze multiple log files"""
    parser = LogParser(rules_path)
    return parser.batch_analyze_logs(log_directory)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        result = analyze_log_file(log_path)
        
        print(f"\n=== LOG ANALYSIS RESULTS ===")
        print(f"Total entries: {result.total_entries}")
        print(f"Suspicious entries: {len(result.suspicious_entries)}")
        print(f"Confidence score: {result.confidence_score:.2f}")
        print(f"AI Analysis: {result.ai_analysis.description}")
        
        if result.threat_indicators:
            print(f"\nThreat indicators found:")
            for category, indicators in result.threat_indicators.items():
                print(f"  {category}: {len(indicators)} indicators")
    else:
        print("Usage: python log_parser.py <log_file_path>")
