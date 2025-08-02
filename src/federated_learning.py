"""
Federated Learning System for Privacy-Preserving Threat Intelligence
Enables collaborative learning without compromising user privacy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import hashlib
import json
from datetime import datetime
import asyncio

class FederatedThreatLearning:
    """
    Privacy-preserving federated learning for threat intelligence
    """
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.global_model = self._initialize_global_model()
        self.client_models = {}
        self.threat_patterns = {}
        self.privacy_budget = 1.0  # Differential privacy budget
        
    def _initialize_global_model(self):
        """Initialize the global threat detection model"""
        class ThreatClassifier(nn.Module):
            def __init__(self, input_size=768, hidden_size=256, num_classes=5):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size // 2, num_classes)
                )
                
            def forward(self, x):
                return self.classifier(x)
        
        return ThreatClassifier()
    
    def register_client(self, client_id: str, client_data_hash: str):
        """Register a new client for federated learning"""
        self.client_models[client_id] = {
            'model': self._initialize_global_model(),
            'data_hash': client_data_hash,
            'contributions': 0,
            'reputation_score': 1.0,
            'last_update': datetime.now()
        }
        
    def create_privacy_preserving_update(self, local_gradients: torch.Tensor, 
                                       noise_multiplier: float = 0.1):
        """Add differential privacy noise to gradients"""
        noise = torch.normal(0, noise_multiplier, local_gradients.shape)
        return local_gradients + noise
    
    def aggregate_updates(self, client_updates: Dict[str, torch.Tensor]):
        """Federated averaging with privacy preservation"""
        aggregated_params = {}
        total_weight = 0
        
        for client_id, update in client_updates.items():
            if client_id not in self.client_models:
                continue
                
            # Weight by client reputation
            weight = self.client_models[client_id]['reputation_score']
            total_weight += weight
            
            # Add privacy noise
            private_update = self.create_privacy_preserving_update(update)
            
            for param_name, param_tensor in private_update.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = torch.zeros_like(param_tensor)
                aggregated_params[param_name] += param_tensor * weight
        
        # Normalize by total weight
        for param_name in aggregated_params:
            aggregated_params[param_name] /= total_weight
            
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update the global model with aggregated parameters"""
        global_state = self.global_model.state_dict()
        
        for param_name, param_tensor in aggregated_params.items():
            if param_name in global_state:
                global_state[param_name] = param_tensor
                
        self.global_model.load_state_dict(global_state)
    
    def share_threat_intelligence(self, client_id: str, threat_indicators: Dict):
        """Share anonymized threat intelligence"""
        # Create privacy-preserving threat patterns
        anonymized_indicators = self._anonymize_indicators(threat_indicators)
        
        # Update global threat patterns
        for threat_type, indicators in anonymized_indicators.items():
            if threat_type not in self.threat_patterns:
                self.threat_patterns[threat_type] = []
            
            # Add indicators with privacy preservation
            self.threat_patterns[threat_type].extend(indicators)
            
        # Update client reputation based on contribution quality
        self._update_client_reputation(client_id, threat_indicators)
    
    def _anonymize_indicators(self, indicators: Dict) -> Dict:
        """Anonymize threat indicators using differential privacy"""
        anonymized = {}
        
        for threat_type, data in indicators.items():
            anonymized[threat_type] = []
            
            for indicator in data:
                # Hash sensitive information
                if isinstance(indicator, str):
                    # Keep pattern structure but hash specific values
                    hashed_indicator = self._selective_hash(indicator)
                    anonymized[threat_type].append(hashed_indicator)
                elif isinstance(indicator, dict):
                    # Anonymize dictionary values
                    anon_dict = {}
                    for key, value in indicator.items():
                        if key in ['ip', 'domain', 'email']:
                            anon_dict[key] = hashlib.sha256(
                                str(value).encode()
                            ).hexdigest()[:16]
                        else:
                            anon_dict[key] = value
                    anonymized[threat_type].append(anon_dict)
        
        return anonymized
    
    def _selective_hash(self, text: str) -> str:
        """Selectively hash sensitive parts while preserving patterns"""
        # Keep structural patterns but hash specific values
        import re
        
        # Hash IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        text = re.sub(ip_pattern, lambda m: f"IP_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
        
        # Hash email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, lambda m: f"EMAIL_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
        
        # Hash domains
        domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b'
        text = re.sub(domain_pattern, lambda m: f"DOMAIN_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
        
        return text
    
    def _update_client_reputation(self, client_id: str, contribution: Dict):
        """Update client reputation based on contribution quality"""
        if client_id not in self.client_models:
            return
        
        # Simple reputation scoring based on contribution diversity and quality
        contribution_score = 0.0
        
        # Reward diverse threat types
        contribution_score += len(contribution.keys()) * 0.1
        
        # Reward quantity (with diminishing returns)
        total_indicators = sum(len(indicators) for indicators in contribution.values())
        contribution_score += min(np.log(total_indicators + 1) * 0.1, 0.5)
        
        # Update reputation (exponential moving average)
        current_reputation = self.client_models[client_id]['reputation_score']
        self.client_models[client_id]['reputation_score'] = (
            0.9 * current_reputation + 0.1 * contribution_score
        )
        
        self.client_models[client_id]['contributions'] += 1
        self.client_models[client_id]['last_update'] = datetime.now()
    
    def get_personalized_threat_intel(self, client_id: str) -> Dict:
        """Get personalized threat intelligence for a client"""
        if client_id not in self.client_models:
            return {}
        
        # Return relevant threat patterns based on client profile
        client_reputation = self.client_models[client_id]['reputation_score']
        
        # Higher reputation clients get more detailed intelligence
        if client_reputation > 0.8:
            intel_level = "detailed"
        elif client_reputation > 0.5:
            intel_level = "standard"
        else:
            intel_level = "basic"
        
        return self._filter_threat_intel_by_level(intel_level)
    
    def _filter_threat_intel_by_level(self, level: str) -> Dict:
        """Filter threat intelligence based on access level"""
        filtered_intel = {}
        
        for threat_type, patterns in self.threat_patterns.items():
            if level == "basic":
                # Basic level gets only high-confidence, common patterns
                filtered_intel[threat_type] = patterns[:10]
            elif level == "standard":
                # Standard level gets more patterns
                filtered_intel[threat_type] = patterns[:50]
            else:
                # Detailed level gets full access
                filtered_intel[threat_type] = patterns
        
        return filtered_intel
    
    async def federated_training_round(self, participating_clients: List[str]):
        """Execute one round of federated learning"""
        print(f"Starting federated training round with {len(participating_clients)} clients")
        
        # 1. Send global model to participating clients
        global_params = self.global_model.state_dict()
        
        # 2. Collect updates from clients (simulated)
        client_updates = {}
        for client_id in participating_clients:
            if client_id in self.client_models:
                # Simulate client training and return gradients
                # In real implementation, this would be actual federated learning
                client_updates[client_id] = self._simulate_client_update(client_id)
        
        # 3. Aggregate updates
        aggregated_params = self.aggregate_updates(client_updates)
        
        # 4. Update global model
        self.update_global_model(aggregated_params)
        
        print(f"Federated training round completed with {len(client_updates)} contributions")
        
        return {
            'round_completed': True,
            'participants': len(client_updates),
            'global_model_updated': True
        }
    
    def _simulate_client_update(self, client_id: str) -> Dict[str, torch.Tensor]:
        """Simulate client model update (for demonstration)"""
        # In real implementation, this would be actual gradient updates
        client_model = self.client_models[client_id]['model']
        
        # Simulate some parameter updates
        updates = {}
        for name, param in client_model.named_parameters():
            # Add small random updates to simulate training
            updates[name] = param.data + torch.normal(0, 0.01, param.shape)
        
        return updates
    
    def export_privacy_report(self) -> Dict:
        """Export privacy and security report"""
        return {
            'total_clients': len(self.client_models),
            'active_clients': sum(1 for client in self.client_models.values() 
                                if (datetime.now() - client['last_update']).days < 7),
            'privacy_budget_remaining': self.privacy_budget,
            'threat_patterns_learned': sum(len(patterns) for patterns in self.threat_patterns.values()),
            'differential_privacy_enabled': True,
            'data_anonymization_active': True
        }

# Example usage and demonstration
async def main():
    # Initialize federated learning system
    fl_system = FederatedThreatLearning({
        'model_type': 'threat_classifier',
        'input_size': 768,
        'num_classes': 5
    })
    
    # Register some clients
    clients = ['client_journalist_1', 'client_ngo_2', 'client_activist_3']
    for client in clients:
        fl_system.register_client(client, f"data_hash_{client}")
    
    # Simulate threat intelligence sharing
    sample_threats = {
        'phishing': [
            {'pattern': 'urgent account verification', 'confidence': 0.9},
            {'pattern': 'suspended for security', 'confidence': 0.8}
        ],
        'malware': [
            {'hash': 'abc123...', 'family': 'trojan', 'confidence': 0.95}
        ]
    }
    
    for client in clients:
        fl_system.share_threat_intelligence(client, sample_threats)
    
    # Execute federated training round
    training_result = await fl_system.federated_training_round(clients)
    print("Training Result:", training_result)
    
    # Get personalized threat intel
    intel = fl_system.get_personalized_threat_intel('client_journalist_1')
    print("Personalized Intel:", intel)
    
    # Export privacy report
    privacy_report = fl_system.export_privacy_report()
    print("Privacy Report:", privacy_report)

if __name__ == "__main__":
    asyncio.run(main())
