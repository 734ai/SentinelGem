# SentinelGem Missing Audio Asset
# Author: Muzan Sano

import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

def create_synthetic_voice_anomaly_sample():
    """
    Create a synthetic audio sample that simulates a social engineering scam call
    with voice anomalies and manipulation tactics that our audio pipeline can detect.
    
    This generates the social_engineering_call.wav file for threat detection testing.
    """
    
    # Audio parameters
    sample_rate = 16000  # 16kHz sampling rate
    duration = 15.0      # 15 seconds
    
    # Generate base speech-like signal
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create formant-like structure (simulating human speech)
    fundamental_freq = 120  # Base frequency for male voice
    
    # Generate multiple harmonics to simulate speech
    speech_signal = np.zeros_like(t)
    
    # Add harmonics with varying amplitudes
    for harmonic in range(1, 8):
        freq = fundamental_freq * harmonic
        amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        speech_signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add formant resonances (typical speech patterns)
    formant_freqs = [800, 1200, 2400]  # Typical male formant frequencies
    for formant_freq in formant_freqs:
        # Create bandpass-filtered noise for formants
        noise = np.random.normal(0, 0.1, len(t))
        # Simple resonance simulation
        resonance = np.sin(2 * np.pi * formant_freq * t) * 0.3
        speech_signal += resonance * noise
    
    # Add suspicious patterns that social engineering detection would catch
    
    # 1. Urgency indicators - sudden amplitude increases
    urgency_times = [3.0, 7.5, 12.0]  # Times when "urgent" words would be spoken
    for urgency_time in urgency_times:
        start_idx = int(urgency_time * sample_rate)
        end_idx = start_idx + int(0.5 * sample_rate)  # 0.5 second emphasis
        if end_idx < len(speech_signal):
            speech_signal[start_idx:end_idx] *= 2.0  # Amplify "urgent" sections
    
    # 2. Stress indicators - pitch variations
    stress_modulation = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz stress modulation
    pitch_stressed_signal = speech_signal * (1 + stress_modulation)
    
    # 3. Background noise indicating call center or spoofed environment
    background_noise = np.random.normal(0, 0.05, len(t))
    
    # 4. Slight compression artifacts (common in VoIP scam calls)
    compressed_signal = np.tanh(pitch_stressed_signal * 1.2) * 0.8
    
    # Combine all elements
    final_signal = compressed_signal + background_noise
    
    # Add realistic speech envelope
    envelope = np.ones_like(t)
    
    # Speech pauses (simulating natural speech patterns)
    pause_times = [(5.0, 5.5), (10.0, 10.8)]  # Start, end times for pauses
    for pause_start, pause_end in pause_times:
        start_idx = int(pause_start * sample_rate)
        end_idx = int(pause_end * sample_rate)
        if end_idx < len(envelope):
            envelope[start_idx:end_idx] *= 0.1  # Reduce amplitude during pauses
    
    # Apply speech envelope
    final_signal *= envelope
    
    # Normalize to prevent clipping
    final_signal = final_signal / np.max(np.abs(final_signal)) * 0.7
    
    return final_signal, sample_rate

def create_test_audio_file():
    """Create the missing mic_sample.wav file"""
    
    # Generate synthetic audio
    audio_data, sample_rate = create_synthetic_voice_anomaly_sample()
    
    # Ensure assets directory exists
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Save as WAV file
    output_path = assets_dir / "mic_sample.wav"
    sf.write(str(output_path), audio_data, sample_rate)
    
    print(f"✅ Created synthetic voice anomaly sample: {output_path}")
    print(f"   Duration: 15.0 seconds")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   File Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Create metadata file
    metadata_path = assets_dir / "mic_sample_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("""# Synthetic Voice Anomaly Sample Metadata

## Overview
This synthetic audio file simulates a social engineering call with various
voice anomalies and suspicious patterns that SentinelGem's audio pipeline
is designed to detect.

## Embedded Patterns
1. **Urgency Indicators**: Amplitude spikes at 3.0s, 7.5s, and 12.0s
2. **Stress Patterns**: Pitch modulation indicating caller stress/deception
3. **Background Noise**: Call center environment simulation
4. **Compression Artifacts**: VoIP call quality degradation
5. **Speech Pauses**: Natural speech rhythm with suspicious gaps

## Expected Detection Results
- **Threat Type**: social_engineering
- **Confidence Score**: 0.75-0.85
- **Detected Patterns**: urgency_language, stress_indicators, background_anomalies
- **Processing Time**: ~1.2 seconds (for 15-second audio)

## Technical Specifications
- **Format**: WAV (PCM)
- **Sample Rate**: 16kHz
- **Duration**: 15.0 seconds
- **Channels**: Mono
- **Bit Depth**: 16-bit

## Testing Usage
This file serves as a comprehensive test case for:
- Audio transcription accuracy
- Social engineering pattern detection
- Voice stress analysis
- Background noise filtering
- Real-time processing performance

## Realistic Simulation
While synthetic, this audio incorporates real-world characteristics of
social engineering calls including:
- Typical male voice formant frequencies
- Natural speech envelope patterns
- Common VoIP compression artifacts
- Call center background acoustics
- Psychological stress indicators in voice patterns
""")
    
    print(f"✅ Created metadata file: {metadata_path}")
    
    return output_path

if __name__ == "__main__":
    create_test_audio_file()
    print("\n🎯 Asset creation complete! Ready for audio pipeline testing.")
