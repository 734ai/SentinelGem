# 📦 SentinelGem: Technical Requirements Specification

## 🧠 Core Dependencies

### ✅ Machine Learning / LLMs

* `transformers>=4.40.0`  # For Gemma 3n and model utilities
* `sentencepiece`          # Tokenizer support for Gemma
* `torch>=2.1.0`           # PyTorch backend

### 📊 Data Processing & NLP

* `scikit-learn`
* `pandas`
* `langdetect`
* `unstructured`           # Optional: for parsing raw documents/logs

### 🖼️ Image Processing

* `pytesseract`
* `Pillow`
* `opencv-python`

### 🔊 Audio & Voice

* `openai-whisper`
* `ffmpeg-python`

### 📓 Notebook Generation

* `nbformat`
* `nbconvert`

### ⚙️ Utility

* `rich`                   # Logging output
* `psutil`                 # Process/memory info (for system inspection)
* `tqdm`                   # Progress bars
* `python-dotenv`          # Optional config loading

---

## 📁 Assets and Models

* Pretrained **Gemma 3n 2B or 4B** quantized version in GGUF or HF format
* Whisper base/medium model weights
* Sample screenshots/logs/voice in `/assets`

---

## 🖥️ Platform Compatibility

* ✅ Colab
* ✅ Kaggle
* ✅ Jetson Nano (via MLC or quantized Gemma)
* ✅ Local desktop (Ollama-compatible with GGUF)

---

## 🧪 Test Dataset (for simulation)

* Fake phishing UI screenshot
* SSH brute-force logs
* Voice command samples
* Simulated clipboard dumps or browsing history

---

## 📦 External Tools (Optional)

* `Sigma rules`: threat\_rules.yaml loader
* `langchain`: for agent-style orchestration (optional)
* `llama.cpp` bindings (if using local C++ inference engine)

---

> This file ensures full reproducibility and cross-platform execution.
