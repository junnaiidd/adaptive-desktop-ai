# Adaptive Desktop AI

Adaptive Desktop AI is a Windows-first, local-first desktop application intended to help people maintain continuity in recurring computer work.

Its long-term operating loop is:

**OBSERVE → UNDERSTAND → LEARN → PREDICT → ACT**

The project exists to turn privacy-conscious, local desktop activity signals into meaningful context and, eventually, user-approved assistance with routines and workspaces.

## Status

This repository is at **Phase 1D**. It provides a native PySide6 Activity dashboard for starting/stopping local monitoring and viewing current activity, activity summaries, recent segments, and session information. Foreground observations are grouped into local SQLite segments and sessions. It does not perform ML, routine discovery, prediction, or automation.

## Intended technology stack

- Python
- PySide6 for the desktop application shell
- Windows APIs via `pywin32` and process metadata via `psutil`
- SQLite for future local activity storage
- pandas, NumPy, and scikit-learn for future data processing and meaningful ML inference
- pytest for testing

## Privacy-first and Windows-first

The application is designed for Windows and keeps future activity data local. It will never record keystrokes, collect passwords or private-message contents, capture screenshots, or transmit activity data externally. Any future desktop automation must require explicit user approval.

## Launch the Phase 1D desktop interface (Windows)

From the repository root, launch the desktop application:

```powershell
.\.venv\Scripts\python.exe main.py
```

Use **Start monitoring** to begin local observation and **Stop monitoring** (or close the app) to persist the final segment. Data remains in `data/activity.db`; window titles shown in the interface are the existing privacy-sanitized values.
"# adaptive-desktop-ai" 
