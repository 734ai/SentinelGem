"""
Blockchain-based Threat Intelligence Sharing System
Immutable, verifiable threat intelligence with reputation scoring
"""

import hashlib
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    threat_type: str
    indicators: List[str]
    confidence_score: float
    source_reputation: float
    timestamp: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'threat_type': self.threat_type,
            'indicators': self.indicators,
            'confidence_score': self.confidence_score,
            'source_reputation': self.source_reputation,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def calculate_hash(self) -> str:
        """Calculate hash of threat intelligence"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class Block:
    """Blockchain block for threat intelligence"""
    index: int
    timestamp: str
    threat_intel: List[ThreatIntelligence]
    previous_hash: str
    nonce: int = 0
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_content = {
            'index': self.index,
            'timestamp': self.timestamp,
            'threat_intel': [ti.to_dict() for ti in self.threat_intel],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        content = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine block with proof-of-work"""
        target = "0" * difficulty
        while not self.calculate_hash().startswith(target):
            self.nonce += 1
        print(f"Block mined with nonce: {self.nonce}")

class ThreatIntelBlockchain:
    """Blockchain for threat intelligence sharing"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_intel: List[ThreatIntelligence] = []
        self.mining_reward = 10
        self.difficulty = 4
        self.node_reputation: Dict[str, float] = {}
        self.consensus_nodes: List[str] = []
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_intel = ThreatIntelligence(
            threat_type="genesis",
            indicators=["blockchain_initialized"],
            confidence_score=1.0,
            source_reputation=1.0,
            timestamp=datetime.now().isoformat(),
            metadata={"block_type": "genesis"}
        )
        
        genesis_block = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            threat_intel=[genesis_intel],
            previous_hash="0"
        )
        
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_threat_intelligence(self, node_id: str, threat_intel: ThreatIntelligence):
        """Add threat intelligence to pending pool"""
        # Validate source reputation
        if node_id not in self.node_reputation:
            self.node_reputation[node_id] = 0.5  # Default reputation
        
        # Adjust threat intel based on source reputation
        threat_intel.source_reputation = self.node_reputation[node_id]
        
        # Validate threat intelligence
        if self._validate_threat_intel(threat_intel):
            self.pending_intel.append(threat_intel)
            print(f"Threat intelligence added from {node_id}")
            return True
        else:
            print(f"Invalid threat intelligence from {node_id}")
            return False
    
    def _validate_threat_intel(self, threat_intel: ThreatIntelligence) -> bool:
        """Validate threat intelligence data"""
        # Basic validation rules
        if not threat_intel.threat_type:
            return False
        
        if not threat_intel.indicators:
            return False
        
        if not (0.0 <= threat_intel.confidence_score <= 1.0):
            return False
        
        # Check for duplicates
        intel_hash = threat_intel.calculate_hash()
        for block in self.chain:
            for existing_intel in block.threat_intel:
                if existing_intel.calculate_hash() == intel_hash:
                    return False  # Duplicate found
        
        return True
    
    def mine_pending_intel(self, mining_node: str) -> bool:
        """Mine pending threat intelligence into a new block"""
        if not self.pending_intel:
            print("No pending threat intelligence to mine")
            return False
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            threat_intel=self.pending_intel.copy(),
            previous_hash=self.get_latest_block().calculate_hash()
        )
        
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(new_block)
        
        # Reward the mining node
        self._reward_mining_node(mining_node)
        
        # Update reputation scores based on validated intelligence
        self._update_reputation_scores()
        
        # Clear pending intel
        self.pending_intel = []
        
        print(f"Block {new_block.index} mined by {mining_node}")
        return True
    
    def _reward_mining_node(self, node_id: str):
        """Reward node for mining"""
        if node_id not in self.node_reputation:
            self.node_reputation[node_id] = 0.5
        
        # Increase reputation for successful mining
        self.node_reputation[node_id] = min(1.0, self.node_reputation[node_id] + 0.05)
    
    def _update_reputation_scores(self):
        """Update node reputation based on intelligence quality"""
        if not self.chain:
            return
        
        latest_block = self.get_latest_block()
        
        for threat_intel in latest_block.threat_intel:
            # Find the source node (simplified - in real implementation would track this)
            # For now, update all nodes that contributed to this block
            for node_id in self.node_reputation:
                # Positive reputation for high-confidence intelligence
                if threat_intel.confidence_score > 0.8:
                    self.node_reputation[node_id] = min(1.0, 
                        self.node_reputation[node_id] + 0.01)
                # Negative reputation for low-confidence intelligence
                elif threat_intel.confidence_score < 0.3:
                    self.node_reputation[node_id] = max(0.0, 
                        self.node_reputation[node_id] - 0.02)
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Validate current block hash
            if current_block.calculate_hash() != current_block.calculate_hash():
                print(f"Invalid hash for block {i}")
                return False
            
            # Validate link to previous block
            if current_block.previous_hash != previous_block.calculate_hash():
                print(f"Invalid previous hash for block {i}")
                return False
            
            # Validate proof of work
            if not current_block.calculate_hash().startswith("0" * self.difficulty):
                print(f"Invalid proof of work for block {i}")
                return False
        
        return True
    
    def get_threat_intelligence(self, threat_type: Optional[str] = None, 
                              min_confidence: float = 0.0) -> List[ThreatIntelligence]:
        """Retrieve threat intelligence from blockchain"""
        all_intel = []
        
        for block in self.chain[1:]:  # Skip genesis block
            for threat_intel in block.threat_intel:
                if threat_type and threat_intel.threat_type != threat_type:
                    continue
                
                if threat_intel.confidence_score < min_confidence:
                    continue
                
                all_intel.append(threat_intel)
        
        # Sort by confidence score and recency
        all_intel.sort(key=lambda x: (x.confidence_score, x.timestamp), reverse=True)
        return all_intel
    
    def get_node_reputation(self, node_id: str) -> float:
        """Get reputation score for a node"""
        return self.node_reputation.get(node_id, 0.0)
    
    def register_consensus_node(self, node_id: str):
        """Register a node for consensus participation"""
        if node_id not in self.consensus_nodes:
            self.consensus_nodes.append(node_id)
            if node_id not in self.node_reputation:
                self.node_reputation[node_id] = 0.5
    
    def consensus_validation(self, new_block: Block) -> bool:
        """Validate new block through consensus"""
        if len(self.consensus_nodes) < 3:
            return True  # Skip consensus if not enough nodes
        
        # Simple majority consensus
        votes = 0
        for node_id in self.consensus_nodes:
            # Simulate node validation (in real implementation, 
            # this would involve network communication)
            if self._simulate_node_validation(node_id, new_block):
                votes += 1
        
        # Require majority approval
        return votes > len(self.consensus_nodes) / 2
    
    def _simulate_node_validation(self, node_id: str, block: Block) -> bool:
        """Simulate consensus node validation"""
        # Nodes with higher reputation are more likely to validate correctly
        reputation = self.node_reputation.get(node_id, 0.5)
        
        # Basic validation checks
        if not block.threat_intel:
            return False
        
        # Check proof of work
        if not block.calculate_hash().startswith("0" * self.difficulty):
            return False
        
        # Reputation-based validation probability
        import random
        return random.random() < (0.7 + 0.3 * reputation)
    
    def export_chain_summary(self) -> Dict:
        """Export blockchain summary"""
        total_intel = sum(len(block.threat_intel) for block in self.chain[1:])
        
        threat_types = {}
        for block in self.chain[1:]:
            for intel in block.threat_intel:
                threat_types[intel.threat_type] = threat_types.get(intel.threat_type, 0) + 1
        
        return {
            'total_blocks': len(self.chain),
            'total_threat_intelligence': total_intel,
            'threat_types': threat_types,
            'registered_nodes': len(self.node_reputation),
            'consensus_nodes': len(self.consensus_nodes),
            'chain_valid': self.validate_chain(),
            'node_reputations': self.node_reputation.copy()
        }

# Integration with Gemma 3n
class GemmaThreatIntelIntegration:
    """Integration between Gemma 3n and threat intelligence blockchain"""
    
    def __init__(self, blockchain: ThreatIntelBlockchain, gemma_model):
        self.blockchain = blockchain
        self.gemma_model = gemma_model
        self.node_id = f"gemma_node_{int(time.time())}"
        
        # Register as consensus node
        self.blockchain.register_consensus_node(self.node_id)
    
    async def analyze_and_share_threats(self, input_data: Dict) -> Dict:
        """Analyze threats with Gemma and share findings"""
        # Use Gemma 3n to analyze input
        analysis_result = await self._gemma_threat_analysis(input_data)
        
        if analysis_result['is_threat'] and analysis_result['confidence'] > 0.7:
            # Create threat intelligence
            threat_intel = ThreatIntelligence(
                threat_type=analysis_result['threat_type'],
                indicators=analysis_result['indicators'],
                confidence_score=analysis_result['confidence'],
                source_reputation=0.8,  # Gemma has high reputation
                timestamp=datetime.now().isoformat(),
                metadata={
                    'source': 'gemma_3n',
                    'analysis_method': 'multimodal_ai',
                    'input_type': input_data.get('type', 'unknown')
                }
            )
            
            # Add to blockchain
            success = self.blockchain.add_threat_intelligence(self.node_id, threat_intel)
            
            if success:
                # Attempt to mine if enough pending intel
                if len(self.blockchain.pending_intel) >= 5:
                    self.blockchain.mine_pending_intel(self.node_id)
            
            return {
                'threat_detected': True,
                'added_to_blockchain': success,
                'threat_intel': threat_intel.to_dict()
            }
        
        return {
            'threat_detected': False,
            'confidence_too_low': analysis_result['confidence'] < 0.7
        }
    
    async def _gemma_threat_analysis(self, input_data: Dict) -> Dict:
        """Simulate Gemma 3n threat analysis"""
        # In real implementation, this would use actual Gemma 3n model
        
        # Simulate analysis based on input type
        if input_data.get('type') == 'email':
            return {
                'is_threat': True,
                'threat_type': 'phishing',
                'confidence': 0.85,
                'indicators': ['urgent_action_required', 'credential_request', 'suspicious_link']
            }
        elif input_data.get('type') == 'url':
            return {
                'is_threat': True,
                'threat_type': 'malicious_url',
                'confidence': 0.92,
                'indicators': ['suspicious_domain', 'url_shortener', 'typosquatting']
            }
        else:
            return {
                'is_threat': False,
                'threat_type': 'benign',
                'confidence': 0.1,
                'indicators': []
            }
    
    def query_threat_intelligence(self, query_params: Dict) -> List[Dict]:
        """Query blockchain for relevant threat intelligence"""
        threat_intel = self.blockchain.get_threat_intelligence(
            threat_type=query_params.get('threat_type'),
            min_confidence=query_params.get('min_confidence', 0.7)
        )
        
        return [intel.to_dict() for intel in threat_intel[:20]]  # Return top 20

# Example usage
async def main():
    # Initialize blockchain
    blockchain = ThreatIntelBlockchain()
    
    # Create Gemma integration (placeholder for actual model)
    gemma_integration = GemmaThreatIntelIntegration(blockchain, None)
    
    # Simulate threat analysis and sharing
    sample_inputs = [
        {'type': 'email', 'content': 'Urgent: Verify your account immediately'},
        {'type': 'url', 'content': 'http://suspicious-bank-site.com/login'},
        {'type': 'file', 'content': 'suspicious_file.exe'}
    ]
    
    for input_data in sample_inputs:
        result = await gemma_integration.analyze_and_share_threats(input_data)
        print(f"Analysis result: {result}")
    
    # Mine remaining pending intelligence
    blockchain.mine_pending_intel("manual_miner")
    
    # Query threat intelligence
    phishing_intel = gemma_integration.query_threat_intelligence({
        'threat_type': 'phishing',
        'min_confidence': 0.8
    })
    print(f"Phishing intelligence: {len(phishing_intel)} entries")
    
    # Export summary
    summary = blockchain.export_chain_summary()
    print("Blockchain Summary:", summary)

if __name__ == "__main__":
    asyncio.run(main())
