# SentinelGem: Advanced Project Structure

## Root Directory

```
sentinelgem/
├── .github/                  # GitHub workflows, actions
├── .vscode/                  # Debug configs for devs
├── assets/                   # Threat test samples (audio, logs, emails)
├── notebooks/                # AI-generated security analysis reports
├── src/                      # Core threat detection engine
├── models/                   # Quantized Gemma 3n + Whisper models
├── agents/                   # AI agent orchestration
├── config/                   # MITRE ATT&CK rules and detection configs
├── ui/                       # React-based threat analysis dashboard
├── tests/                    # Comprehensive security testing suite
├── docs/                     # Professional documentation (77k+ words)
├── data/                     # Threat intelligence datasets
├── .env                      # Environment configuration
├── requirements.txt
├── LICENSE
├── README.md
└── main.py                   # SentinelGem launch interface
```

---

## `assets/` - Threat Detection Test Samples

```
assets/
├── phishing_email_sample.txt          # Realistic phishing email
├── social_engineering_call.wav        # Synthetic scam call audio
├── social_engineering_call_metadata.txt
├── legitimate_call.wav                 # Baseline benign audio
├── legitimate_call_metadata.txt
├── example_logs.txt                    # System logs with malware indicators
├── malicious_screenshot.png           # Fake login page (planned)
└── README.md                          # Testing asset documentation
```

---

## `src/` - Threat Detection Engine

```
src/
├── inference.py              # Gemma 3n AI threat analysis
├── ocr_pipeline.py           # Screenshot phishing detection
├── audio_pipeline.py         # Social engineering call analysis
├── log_parser.py             # Malware and intrusion detection
├── threat_rules.py           # MITRE ATT&CK pattern matching
├── autogen_notebook.py       # Automated security reports
├── utils.py                  # Security utilities and helpers
├── file_watcher.py           # Real-time threat monitoring
└── finetune_interface.py     # Adaptive threat learning
```

---

## `agents/` - AI Security Orchestration

```
agents/
├── orchestrator.py           # Multi-modal threat coordinator
├── planner.py                # Threat response planning
├── memory.py                 # Vector store or file-based memory
├── prompts/                  # Static and dynamic prompts
│   ├── log_review.txt
│   ├── ocr_screen_eval.txt
│   └── mic_transcript_eval.txt
├── reward_model.py           # Agent self-eval reward scores
└── agent_loop.py             # Main step-by-step logic
```

---

## `notebooks/` - Auto-Generated + Demos

```
notebooks/
├── 00_bootstrap.ipynb        # Entry point, triggers agent
├── 01_data_demo.ipynb        # Sample inputs
├── 02_inference_demo.ipynb   # Interactive LLM session
├── autogen/
│   ├── logs_YYYYMMDD.ipynb
│   ├── phishing_screen_*.ipynb
│   └── voice_alerts_*.ipynb
```

---

## `config/` - Custom Rules, Parameters, Threat Profiles

```
config/
├── rules.yaml                # MITRE/RedTeam detection rules
├── prompt_config.json        # Prompt tuning settings
├── system_flags.yaml         # Flags for suspicious states
├── thresholds.json           # Detection thresholds
└── feedback_tuning.json      # Human/Ai-in-the-loop settings
```

---

## `ui/` - Optional GUI

```
ui/
├── app.py                    # Streamlit interface
├── dashboard.py              # Visual threat timeline
└── components/               # Custom widgets, visualizers
```

---

## `tests/` - Evaluation Suite

```
tests/
├── test_inference.py         # LLM inference testing
├── test_rules.py             # Rule matching tests
├── test_audio_pipeline.py    # Whisper transcription unit tests
├── test_ocr.py               # OCR accuracy tests
└── integration_test.py       # End-to-end agent test
```

---

## `docs/` - Publishable Technical + User Docs

```
docs/
├── architecture.md           # System diagrams, flowcharts
├── user_manual.md            # Non-technical intro for field users
├── developer_guide.md        # Instructions for contributors
└── model_card.md             # Gemma + Whisper usage profiles
```

---

## Version Control Strategy

* `dev/` branch for cutting-edge
* `main/` stable for Kaggle/public
* GitHub Actions for: linting, unit tests, build notebook report

---

## Starter Files To Generate

* [ ] `src/inference.py`
* [ ] `agents/agent_loop.py`
* [ ] `notebooks/00_bootstrap.ipynb`
* [ ] `src/autogen_notebook.py`
* [ ] `tests/test_inference.py`

> A well-structured repo doesn’t just impress judges — it enables scalable autonomy.
