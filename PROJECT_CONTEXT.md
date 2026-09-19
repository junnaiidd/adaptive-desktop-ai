# Adaptive Desktop AI: Project Context

## Purpose

Adaptive Desktop AI is a Windows-first, local-first machine learning desktop application for understanding and supporting recurring desktop work.

## Product vision

The product will eventually observe privacy-respecting desktop signals, infer the user's work context, learn recurring routines, predict likely next steps, and offer user-approved help with those routines and workspaces.

## Core loop

**OBSERVE → UNDERSTAND → LEARN → PREDICT → ACT**

## Intended architecture

```text
Desktop Layer
    ↓
Activity Engine
    ↓
Context Engine (ML)
    ↓
Routine Engine       Memory Engine
    ↓                 ↓
Prediction Layer
    ↓
Action Engine
    ↓
Windows Desktop
```

## Development phases

### Phase 1

- Desktop observation
- Active application/window detection
- Activity logging (Phase 1B implemented with local SQLite)
- Session detection (Phase 1C implemented with a configurable idle gap)
- Activity segmentation (Phase 1C implemented)
- Native Activity dashboard (Phase 1D implemented)
- Feature extraction (not yet implemented)

### Phase 2

- ML context detection

### Phase 3

- Routine discovery

### Phase 4

- Approved automation
- Workspace restoration

### Phase 5

- Prediction
- Personalization
- Memory

### Future

- Work-stage detection
- Interruption/abandonment detection
- Semantic memory
- Intelligent rediscovery
- Personalized scheduling
- Mobile companion

## Privacy principles

- Never record keystrokes.
- Never collect passwords.
- Never collect private message contents.
- Do not capture screenshots.
- Do not transmit activity data externally.
- Keep activity data local.
- Require explicit user approval for automation.

## Important rule

ML must perform meaningful inference. It must not be added merely for presentation.
