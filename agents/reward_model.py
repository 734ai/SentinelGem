#!/usr/bin/env python3
"""
SentinelGem Reward Model
Author: Muzan Sano

Reward model for evaluating threat detection outputs, providing feedback,
and enabling continuous improvement of detection accuracy.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

from rich.console import Console
from rich.progress import track

# Import SentinelGem components
from ..src.inference import ThreatAnalysis
from ..src.utils import Timer

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class FeedbackRecord:
    """Represents human feedback on an analysis"""
    analysis_id: str
    timestamp: datetime
    ground_truth: bool  # True if threat, False if benign
    predicted: bool     # Model prediction
    confidence: float   # Model confidence
    feedback_score: float  # Human rating (0-1)
    feedback_notes: Optional[str] = None
    corrected_threat_type: Optional[str] = None
    threat_severity: Optional[int] = None  # 1-5 scale

@dataclass 
class ModelPerformance:
    """Performance metrics for the model"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    avg_confidence: float
    total_samples: int

@dataclass
class RewardScore:
    """Comprehensive reward score for an analysis"""
    accuracy_reward: float      # Correct classification
    confidence_reward: float    # Appropriate confidence level
    severity_reward: float      # Correct threat severity assessment
    explanation_reward: float   # Quality of explanation
    total_reward: float         # Combined reward score
    feedback_notes: str

class RewardModel:
    """
    Reward model for evaluating and improving SentinelGem's threat detection
    """
    
    def __init__(self, feedback_storage_path: str = "data/feedback.json"):
        self.feedback_storage = Path(feedback_storage_path)
        self.feedback_storage.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        self.feedback_records: List[FeedbackRecord] = self._load_feedback()
        
        # Performance tracking
        self.performance_history: List[ModelPerformance] = []
        
        # Reward configuration
        self.reward_config = {
            'accuracy_weight': 0.4,      # Correct classification
            'confidence_weight': 0.25,   # Confidence calibration
            'severity_weight': 0.2,      # Threat severity accuracy
            'explanation_weight': 0.15,  # Explanation quality
            'confidence_penalty': 0.5,   # Penalty for overconfidence
            'false_positive_penalty': 0.8,  # Heavy penalty for false positives
            'false_negative_penalty': 1.0   # Maximum penalty for missed threats
        }
        
        logger.info("Reward Model initialized")
    
    def _load_feedback(self) -> List[FeedbackRecord]:
        """Load existing feedback records"""
        if not self.feedback_storage.exists():
            return []
        
        try:
            with open(self.feedback_storage, 'r') as f:
                data = json.load(f)
            
            records = []
            for record_data in data:
                record = FeedbackRecord(
                    analysis_id=record_data['analysis_id'],
                    timestamp=datetime.fromisoformat(record_data['timestamp']),
                    ground_truth=record_data['ground_truth'],
                    predicted=record_data['predicted'],
                    confidence=record_data['confidence'],
                    feedback_score=record_data['feedback_score'],
                    feedback_notes=record_data.get('feedback_notes'),
                    corrected_threat_type=record_data.get('corrected_threat_type'),
                    threat_severity=record_data.get('threat_severity')
                )
                records.append(record)
            
            console.print(f"[blue]Loaded {len(records)} feedback records[/blue]")
            return records
            
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return []
    
    def _save_feedback(self):
        """Save feedback records to storage"""
        try:
            data = []
            for record in self.feedback_records:
                record_dict = asdict(record)
                record_dict['timestamp'] = record.timestamp.isoformat()
                data.append(record_dict)
            
            with open(self.feedback_storage, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def add_feedback(self, 
                    analysis_id: str,
                    analysis_result: ThreatAnalysis,
                    ground_truth: bool,
                    feedback_score: float,
                    feedback_notes: Optional[str] = None,
                    corrected_threat_type: Optional[str] = None,
                    threat_severity: Optional[int] = None) -> FeedbackRecord:
        """
        Add human feedback for an analysis result
        
        Args:
            analysis_id: Unique identifier for the analysis
            analysis_result: The AI analysis result
            ground_truth: True if actually a threat, False if benign
            feedback_score: Human rating (0.0 to 1.0)
            feedback_notes: Optional human feedback notes
            corrected_threat_type: Corrected threat type if misclassified
            threat_severity: Threat severity rating (1-5)
            
        Returns:
            Created feedback record
        """
        record = FeedbackRecord(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            ground_truth=ground_truth,
            predicted=analysis_result.threat_detected,
            confidence=analysis_result.confidence_score,
            feedback_score=feedback_score,
            feedback_notes=feedback_notes,
            corrected_threat_type=corrected_threat_type,
            threat_severity=threat_severity
        )
        
        self.feedback_records.append(record)
        self._save_feedback()
        
        console.print(f"[green]✅ Feedback added for analysis {analysis_id}[/green]")
        logger.info(f"Feedback added: {analysis_id} - GT: {ground_truth}, Pred: {analysis_result.threat_detected}")
        
        return record
    
    def calculate_reward_score(self, 
                              analysis_result: ThreatAnalysis,
                              ground_truth: bool,
                              human_severity: Optional[int] = None,
                              explanation_quality: Optional[float] = None) -> RewardScore:
        """
        Calculate comprehensive reward score for an analysis
        
        Args:
            analysis_result: AI analysis result
            ground_truth: True if actually a threat
            human_severity: Human-assessed threat severity (1-5)
            explanation_quality: Human rating of explanation quality (0-1)
            
        Returns:
            Detailed reward score
        """
        # 1. Accuracy Reward
        correct_classification = (analysis_result.threat_detected == ground_truth)
        accuracy_reward = 1.0 if correct_classification else 0.0
        
        # Apply penalties for specific error types
        if not correct_classification:
            if analysis_result.threat_detected and not ground_truth:
                # False positive - less severe but still penalized
                accuracy_reward = -self.reward_config['false_positive_penalty']
            elif not analysis_result.threat_detected and ground_truth:
                # False negative - most severe penalty
                accuracy_reward = -self.reward_config['false_negative_penalty']
        
        # 2. Confidence Reward (calibration)
        confidence = analysis_result.confidence_score
        
        if correct_classification:
            # Reward high confidence for correct predictions
            confidence_reward = confidence
        else:
            # Penalize high confidence for incorrect predictions
            confidence_reward = -(confidence * self.reward_config['confidence_penalty'])
        
        # 3. Severity Reward
        severity_reward = 0.0
        if human_severity is not None and ground_truth:
            # Map AI threat type to expected severity
            ai_severity = self._map_threat_type_to_severity(analysis_result.threat_type)
            severity_diff = abs(ai_severity - human_severity)
            severity_reward = max(0, 1.0 - (severity_diff / 4.0))  # Normalize by max diff
        
        # 4. Explanation Reward
        explanation_reward = explanation_quality if explanation_quality is not None else 0.5
        
        # If we don't have human rating, use heuristic
        if explanation_quality is None:
            explanation_reward = self._evaluate_explanation_quality(analysis_result)
        
        # Calculate weighted total reward
        total_reward = (
            accuracy_reward * self.reward_config['accuracy_weight'] +
            confidence_reward * self.reward_config['confidence_weight'] +
            severity_reward * self.reward_config['severity_weight'] +
            explanation_reward * self.reward_config['explanation_weight']
        )
        
        # Generate feedback notes
        feedback_notes = self._generate_feedback_notes(
            analysis_result, ground_truth, accuracy_reward, confidence_reward
        )
        
        return RewardScore(
            accuracy_reward=accuracy_reward,
            confidence_reward=confidence_reward,
            severity_reward=severity_reward,
            explanation_reward=explanation_reward,
            total_reward=total_reward,
            feedback_notes=feedback_notes
        )
    
    def _map_threat_type_to_severity(self, threat_type: str) -> int:
        """Map threat type to severity level (1-5)"""
        severity_mapping = {
            'phishing': 4,
            'malware': 5,
            'social_engineering': 3,
            'ransomware': 5,
            'data_breach': 5,
            'intrusion': 4,
            'spam': 2,
            'suspicious': 3,
            'safe': 1,
            'unknown': 2
        }
        
        return severity_mapping.get(threat_type.lower(), 3)  # Default to medium
    
    def _evaluate_explanation_quality(self, analysis_result: ThreatAnalysis) -> float:
        """Heuristic evaluation of explanation quality"""
        description = analysis_result.description.lower()
        
        quality_score = 0.5  # Base score
        
        # Positive indicators
        positive_indicators = [
            'detected', 'pattern', 'analysis', 'indicates', 'suggests',
            'evidence', 'confidence', 'probability', 'risk', 'threat'
        ]
        
        positive_matches = sum(1 for indicator in positive_indicators if indicator in description)
        quality_score += min(0.3, positive_matches * 0.05)
        
        # Negative indicators (vague language)
        negative_indicators = [
            'might', 'could', 'possibly', 'unclear', 'unknown', 'unsure'
        ]
        
        negative_matches = sum(1 for indicator in negative_indicators if indicator in description)
        quality_score -= min(0.2, negative_matches * 0.05)
        
        # Length bonus (detailed explanations are better)
        if len(description) > 50:
            quality_score += 0.1
        if len(description) > 100:
            quality_score += 0.1
        
        # Recommendations bonus
        if len(analysis_result.recommendations) > 2:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_feedback_notes(self, 
                                analysis_result: ThreatAnalysis,
                                ground_truth: bool,
                                accuracy_reward: float,
                                confidence_reward: float) -> str:
        """Generate feedback notes for the analysis"""
        notes = []
        
        if accuracy_reward > 0:
            notes.append("✅ Correct classification")
        elif accuracy_reward < 0:
            if analysis_result.threat_detected and not ground_truth:
                notes.append("❌ False positive - classified safe content as threat")
            else:
                notes.append("❌ False negative - missed actual threat")
        
        if confidence_reward > 0.7:
            notes.append("✅ Well-calibrated confidence")
        elif confidence_reward < 0:
            notes.append("⚠️ Overconfident in incorrect prediction")
        
        # Specific improvement suggestions
        if not ground_truth and analysis_result.threat_detected:
            notes.append("💡 Consider more conservative threat detection")
        elif ground_truth and not analysis_result.threat_detected:
            notes.append("💡 Consider more sensitive threat detection")
        
        return " | ".join(notes)
    
    def calculate_model_performance(self, recent_only: bool = False) -> ModelPerformance:
        """
        Calculate overall model performance metrics
        
        Args:
            recent_only: If True, only consider recent feedback (last 100 records)
            
        Returns:
            Performance metrics
        """
        records = self.feedback_records
        if recent_only and len(records) > 100:
            records = records[-100:]
        
        if not records:
            return ModelPerformance(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate confusion matrix
        true_positives = sum(1 for r in records if r.ground_truth and r.predicted)
        true_negatives = sum(1 for r in records if not r.ground_truth and not r.predicted)
        false_positives = sum(1 for r in records if not r.ground_truth and r.predicted)
        false_negatives = sum(1 for r in records if r.ground_truth and not r.predicted)
        
        total = len(records)
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        avg_confidence = sum(r.confidence for r in records) / total if total > 0 else 0
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            avg_confidence=avg_confidence,
            total_samples=total
        )
        
        # Track performance history
        self.performance_history.append(performance)
        
        return performance
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving model performance"""
        if len(self.feedback_records) < 10:
            return ["Need more feedback data for meaningful suggestions (minimum 10 samples)"]
        
        performance = self.calculate_model_performance()
        suggestions = []
        
        # Accuracy suggestions
        if performance.accuracy < 0.8:
            suggestions.append("🎯 Overall accuracy is low - consider retraining with more diverse data")
        
        # Precision/Recall balance
        if performance.precision < 0.7:
            suggestions.append("⚠️ High false positive rate - consider stricter detection thresholds")
        
        if performance.recall < 0.7:
            suggestions.append("🚨 High false negative rate - consider more sensitive detection")
        
        # Confidence calibration
        if performance.avg_confidence > 0.9 and performance.accuracy < 0.9:
            suggestions.append("📊 Model appears overconfident - consider confidence calibration")
        
        # Pattern analysis
        recent_records = self.feedback_records[-50:] if len(self.feedback_records) > 50 else self.feedback_records
        
        # Threat type analysis
        threat_type_errors = defaultdict(int)
        for record in recent_records:
            if record.predicted != record.ground_truth and record.corrected_threat_type:
                threat_type_errors[record.corrected_threat_type] += 1
        
        if threat_type_errors:
            most_confused = max(threat_type_errors.items(), key=lambda x: x[1])
            suggestions.append(f"🔍 Most confused threat type: {most_confused[0]} ({most_confused[1]} errors)")
        
        # Confidence distribution analysis
        overconfident_errors = [r for r in recent_records 
                              if r.confidence > 0.8 and r.predicted != r.ground_truth]
        if len(overconfident_errors) > len(recent_records) * 0.1:
            suggestions.append("🎚️ Consider lowering confidence for ambiguous cases")
        
        return suggestions or ["✅ Model performance looks good! Keep monitoring."]
    
    def generate_performance_report(self, output_file: str):
        """Generate comprehensive performance report"""
        try:
            performance = self.calculate_model_performance()
            suggestions = self.get_improvement_suggestions()
            
            # Analyze feedback trends
            feedback_by_month = defaultdict(list)
            for record in self.feedback_records:
                month_key = record.timestamp.strftime('%Y-%m')
                feedback_by_month[month_key].append(record)
            
            monthly_performance = {}
            for month, records in feedback_by_month.items():
                if len(records) >= 5:  # Minimum for meaningful metrics
                    tp = sum(1 for r in records if r.ground_truth and r.predicted)
                    tn = sum(1 for r in records if not r.ground_truth and not r.predicted)
                    total = len(records)
                    monthly_performance[month] = (tp + tn) / total
            
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'overall_performance': asdict(performance),
                'improvement_suggestions': suggestions,
                'monthly_performance_trends': monthly_performance,
                'feedback_summary': {
                    'total_feedback_records': len(self.feedback_records),
                    'avg_human_feedback_score': sum(r.feedback_score for r in self.feedback_records) / len(self.feedback_records) if self.feedback_records else 0,
                    'recent_performance_trend': 'improving' if len(self.performance_history) > 1 and self.performance_history[-1].accuracy > self.performance_history[-2].accuracy else 'stable'
                },
                'threat_type_breakdown': self._analyze_threat_type_performance()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            console.print(f"[green]✅ Performance report generated: {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Report generation failed: {e}[/red]")
            logger.error(f"Report generation failed: {e}")
    
    def _analyze_threat_type_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by threat type"""
        threat_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for record in self.feedback_records:
            if record.corrected_threat_type:
                threat_type = record.corrected_threat_type
            else:
                # Try to infer from feedback
                threat_type = 'unknown'
            
            threat_type_stats[threat_type]['total'] += 1
            if record.predicted == record.ground_truth:
                threat_type_stats[threat_type]['correct'] += 1
        
        # Calculate accuracy by threat type
        performance_by_type = {}
        for threat_type, stats in threat_type_stats.items():
            if stats['total'] > 0:
                performance_by_type[threat_type] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'sample_count': stats['total']
                }
        
        return performance_by_type

    def interactive_feedback_session(self, analysis_results: List[Tuple[str, ThreatAnalysis]]):
        """Interactive session for collecting human feedback"""
        console.print("[blue]🎯 Starting interactive feedback session[/blue]")
        console.print("Rate each analysis on a scale of 0-1 (0=completely wrong, 1=perfect)")
        
        for analysis_id, result in analysis_results:
            console.print(f"\n" + "="*60)
            console.print(f"Analysis ID: {analysis_id}")
            console.print(f"Prediction: {'THREAT' if result.threat_detected else 'SAFE'}")
            console.print(f"Confidence: {result.confidence_score:.2f}")
            console.print(f"Type: {result.threat_type}")
            console.print(f"Description: {result.description}")
            console.print(f"Recommendations: {', '.join(result.recommendations)}")
            
            try:
                # Get ground truth
                while True:
                    gt_input = input("Is this actually a threat? (y/n): ").lower().strip()
                    if gt_input in ['y', 'yes', '1', 'true']:
                        ground_truth = True
                        break
                    elif gt_input in ['n', 'no', '0', 'false']:
                        ground_truth = False
                        break
                    else:
                        print("Please enter y/n")
                
                # Get feedback score
                while True:
                    try:
                        score_input = input("Feedback score (0.0-1.0): ").strip()
                        feedback_score = float(score_input)
                        if 0.0 <= feedback_score <= 1.0:
                            break
                        else:
                            print("Score must be between 0.0 and 1.0")
                    except ValueError:
                        print("Please enter a valid number")
                
                # Optional feedback
                feedback_notes = input("Additional notes (optional): ").strip()
                corrected_type = None
                if ground_truth and result.threat_type == 'unknown':
                    corrected_type = input("Correct threat type (optional): ").strip()
                
                # Add feedback
                self.add_feedback(
                    analysis_id=analysis_id,
                    analysis_result=result,
                    ground_truth=ground_truth,
                    feedback_score=feedback_score,
                    feedback_notes=feedback_notes if feedback_notes else None,
                    corrected_threat_type=corrected_type if corrected_type else None
                )
                
                # Calculate and show reward
                reward = self.calculate_reward_score(result, ground_truth)
                console.print(f"[yellow]Reward Score: {reward.total_reward:.3f}[/yellow]")
                console.print(f"[yellow]Notes: {reward.feedback_notes}[/yellow]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Feedback session interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error during feedback: {e}[/red]")
                continue
        
        # Show session summary
        performance = self.calculate_model_performance()
        console.print(f"\n[green]📊 Current Performance Summary:[/green]")
        console.print(f"   Accuracy: {performance.accuracy:.3f}")
        console.print(f"   Precision: {performance.precision:.3f}")
        console.print(f"   Recall: {performance.recall:.3f}")
        console.print(f"   F1 Score: {performance.f1_score:.3f}")

# Convenience functions
def evaluate_analysis_batch(analysis_results: List[Tuple[str, ThreatAnalysis]], 
                           ground_truths: List[bool],
                           reward_model: Optional[RewardModel] = None) -> List[RewardScore]:
    """Evaluate a batch of analysis results"""
    if reward_model is None:
        reward_model = RewardModel()
    
    rewards = []
    for (analysis_id, result), ground_truth in zip(analysis_results, ground_truths):
        reward = reward_model.calculate_reward_score(result, ground_truth)
        rewards.append(reward)
        
        # Add to feedback automatically
        reward_model.add_feedback(
            analysis_id=analysis_id,
            analysis_result=result,
            ground_truth=ground_truth,
            feedback_score=reward.total_reward
        )
    
    return rewards

if __name__ == "__main__":
    # Example usage
    reward_model = RewardModel()
    
    # Generate performance report
    report_file = "reports/model_performance.json"
    Path(report_file).parent.mkdir(exist_ok=True)
    reward_model.generate_performance_report(report_file)
    
    print("Reward model initialized. Use interactive methods for feedback collection.")
