"""
Real-Time Visual Threat Detection Pipeline
Enhanced screenshot analysis with computer vision
"""

import cv2
import numpy as np
import torch
from PIL import Image
import easyocr
from transformers import pipeline
import time

class VisualThreatDetector:
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en'])
        self.vision_model = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )
        self.phishing_patterns = self._load_visual_patterns()
        
    def _load_visual_patterns(self):
        """Load visual patterns for phishing detection"""
        return {
            'login_forms': {
                'color_schemes': [(255, 255, 255), (0, 100, 200)],  # Common bank colors
                'layout_patterns': ['two_column', 'centered_form'],
                'suspicious_elements': ['urgency_banners', 'fake_security_badges']
            },
            'fake_popups': {
                'warning_colors': [(255, 0, 0), (255, 165, 0)],  # Red/Orange warnings
                'text_patterns': ['virus detected', 'call microsoft', 'update required']
            },
            'social_media_scams': {
                'profile_indicators': ['low_followers', 'recent_creation', 'stock_photos'],
                'post_patterns': ['too_good_to_be_true', 'urgent_offers', 'fake_testimonials']
            }
        }
    
    def analyze_screenshot(self, image_path):
        """Comprehensive screenshot analysis"""
        start_time = time.time()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        results = {
            'timestamp': time.time(),
            'processing_time': 0,
            'threats_detected': [],
            'confidence_scores': {},
            'visual_analysis': {},
            'text_analysis': {},
            'recommendations': []
        }
        
        # 1. OCR Text Extraction
        ocr_results = self.ocr_reader.readtext(image)
        extracted_text = ' '.join([text[1] for text in ocr_results])
        results['text_analysis']['extracted_text'] = extracted_text
        
        # 2. Visual Pattern Detection
        visual_threats = self._detect_visual_patterns(image, pil_image)
        results['visual_analysis'] = visual_threats
        
        # 3. Phishing Site Detection
        phishing_score = self._detect_phishing_site(image, extracted_text)
        results['confidence_scores']['phishing'] = phishing_score
        
        # 4. Fake Popup Detection
        popup_score = self._detect_fake_popups(image, extracted_text)
        results['confidence_scores']['fake_popup'] = popup_score
        
        # 5. Social Media Scam Detection
        social_score = self._detect_social_scams(image, extracted_text)
        results['confidence_scores']['social_scam'] = social_score
        
        # 6. QR Code Analysis
        qr_threats = self._analyze_qr_codes(image)
        results['visual_analysis']['qr_codes'] = qr_threats
        
        # 7. Generate Final Assessment
        overall_threat = self._calculate_overall_threat(results)
        results['overall_threat_level'] = overall_threat
        
        # 8. Generate Recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        results['processing_time'] = time.time() - start_time
        return results
    
    def _detect_visual_patterns(self, image, pil_image):
        """Detect suspicious visual patterns"""
        patterns = {}
        
        # Color analysis
        dominant_colors = self._extract_dominant_colors(image)
        patterns['dominant_colors'] = dominant_colors
        
        # Layout analysis
        layout_type = self._analyze_layout(image)
        patterns['layout_type'] = layout_type
        
        # UI element detection
        ui_elements = self._detect_ui_elements(image)
        patterns['ui_elements'] = ui_elements
        
        return patterns
    
    def _detect_phishing_site(self, image, text):
        """Detect phishing websites"""
        score = 0.0
        
        # Check for common phishing indicators
        phishing_keywords = [
            'verify account', 'suspended', 'urgent action', 'click here',
            'expires today', 'security alert', 'update payment'
        ]
        
        text_lower = text.lower()
        for keyword in phishing_keywords:
            if keyword in text_lower:
                score += 0.15
        
        # Visual indicators
        if self._detect_fake_ssl_indicators(image):
            score += 0.25
            
        if self._detect_urgency_banners(image):
            score += 0.20
            
        return min(score, 1.0)
    
    def _detect_fake_popups(self, image, text):
        """Detect fake security popups"""
        score = 0.0
        
        popup_indicators = [
            'virus detected', 'malware found', 'call microsoft',
            'tech support', 'computer locked', 'windows defender'
        ]
        
        text_lower = text.lower()
        for indicator in popup_indicators:
            if indicator in text_lower:
                score += 0.30
        
        # Visual popup characteristics
        if self._has_popup_appearance(image):
            score += 0.25
            
        return min(score, 1.0)
    
    def _detect_social_scams(self, image, text):
        """Detect social media scams"""
        score = 0.0
        
        scam_patterns = [
            'congratulations', 'you won', 'free money', 'limited time',
            'click to claim', 'exclusive offer', 'act now'
        ]
        
        text_lower = text.lower()
        for pattern in scam_patterns:
            if pattern in text_lower:
                score += 0.12
                
        return min(score, 1.0)
    
    def _analyze_qr_codes(self, image):
        """Analyze QR codes for malicious content"""
        # QR code detection and analysis would go here
        # This is a placeholder for QR code threat analysis
        return {'qr_codes_found': 0, 'malicious_qr_detected': False}
    
    def _extract_dominant_colors(self, image):
        """Extract dominant colors from image"""
        pixels = image.reshape(-1, 3)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dominant = colors[np.argmax(counts)]
        return dominant.tolist()
    
    def _analyze_layout(self, image):
        """Analyze webpage layout patterns"""
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            return "wide_layout"
        elif aspect_ratio < 0.8:
            return "mobile_layout"
        else:
            return "standard_layout"
    
    def _detect_ui_elements(self, image):
        """Detect UI elements like buttons, forms, etc."""
        # This would use computer vision to detect UI elements
        # Placeholder implementation
        return {
            'buttons_detected': 2,
            'forms_detected': 1,
            'links_detected': 5
        }
    
    def _detect_fake_ssl_indicators(self, image):
        """Detect fake SSL/security indicators"""
        # Look for fake padlock icons, security badges, etc.
        return False
    
    def _detect_urgency_banners(self, image):
        """Detect urgency-inducing visual elements"""
        # Look for red banners, flashing elements, etc.
        return False
    
    def _has_popup_appearance(self, image):
        """Check if image has popup window characteristics"""
        # Analyze for popup window visual characteristics
        return False
    
    def _calculate_overall_threat(self, results):
        """Calculate overall threat level"""
        scores = results['confidence_scores']
        max_score = max(scores.values()) if scores else 0.0
        
        if max_score >= 0.8:
            return "CRITICAL"
        elif max_score >= 0.6:
            return "HIGH"
        elif max_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, results):
        """Generate security recommendations"""
        recommendations = []
        scores = results['confidence_scores']
        
        if scores.get('phishing', 0) > 0.6:
            recommendations.extend([
                "DO NOT enter any login credentials on this page",
                "Verify the website URL carefully",
                "Contact the organization directly through official channels",
                "Close this browser tab immediately"
            ])
        
        if scores.get('fake_popup', 0) > 0.6:
            recommendations.extend([
                "This appears to be a fake security warning",
                "DO NOT call any phone numbers shown",
                "DO NOT download any suggested software",
                "Close the popup and run a legitimate antivirus scan"
            ])
        
        if scores.get('social_scam', 0) > 0.4:
            recommendations.extend([
                "This appears to be a social media scam",
                "Do not click any links or provide personal information",
                "Report this content to the platform",
                "Verify any claims through independent sources"
            ])
        
        return recommendations

# Example usage
if __name__ == "__main__":
    detector = VisualThreatDetector()
    
    # Analyze a screenshot
    results = detector.analyze_screenshot("test_screenshot.png")
    
    print(f"Overall Threat Level: {results['overall_threat_level']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    print(f"Threats Detected: {results['threats_detected']}")
    print(f"Recommendations: {results['recommendations']}")
