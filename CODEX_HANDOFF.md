# Adaptive Desktop AI — authoritative Codex handoff

**Use before any code change.** This document records verified repository facts
and durable product intent at the `85d3c39` checkpoint. The current working
tree, executable tests, and Git history are the implementation source of truth
if they conflict with this document.

## Verified repository checkpoint

- Repository: `adaptive-desktop-ai`; branch: `main`.
- At handoff, `main` is synchronized with `origin/main` and the working tree
  is clean.
- Current HEAD: `85d3c39 Add unfinished work dashboard`.
- Relevant history (oldest to newest):
  - `807f903` — Complete Phase 2A–2D ML foundation
  - `939e6f4` — Complete Phase 2E durable training pipeline
  - `ea97b6f` — Complete Phase 2F and 2G context inference integration
  - `6bdf4b7` — Complete Phase 2H durable context observations
  - `1a9f4fe` — Complete Milestone A context integration
  - `b7b418f` — Complete Milestone B work threads
  - `37987af`, `85d3c39` — Add unfinished work dashboard
- A previous checkpoint reported 507 passing tests. Do **not** repeat that as
  a current result until the suite runs successfully in this environment.
  At handoff, a bare `pytest` is blocked during collection by an inaccessible
  `pytest-cache-files-3a3lpn03` directory; preserve it and exclude/repair it
  only with explicit approval. The repository contains 495 explicit `test_`
  functions; parametrization accounts for the historical difference.

## Purpose, privacy, and product direction

Adaptive Desktop AI is a Windows-first, local-first PySide6 application that
helps people maintain continuity in recurring desktop work. Its intended
architecture is:

`Desktop → Activity Engine → Context Engine (ML) → Routine/Memory Engines →
Prediction → Action → Windows Desktop`

The active product path currently is:

`Activity → sessions → ML context inference → durable context observations →
user-created Work Threads → user-created Tasks → Unfinished Work`

Long-term loop: **Observe → Understand → Learn → Predict → Act**. The next
approved planning target is **Resume Work / Workspace Restoration**.

Privacy is non-negotiable:

- never record keystrokes, passwords, or private-message contents;
- never capture screenshots or transmit activity data externally;
- keep activity data local; and
- require explicit user approval for automation.

ML must perform meaningful inference, not exist as presentation decoration.

## Domain boundaries (preserve these)

| Concept | Definition and boundary |
| --- | --- |
| Activity | A privacy-sanitized desktop observation from `WindowsActivityMonitor`, persisted as an activity segment. It is evidence, not a user commitment. |
| Session | A time-bounded grouping managed by `SessionManager` and stored by `ActivityRepository`. It is not a context or Work Thread. |
| ML context | An inference result generated for a completed session through existing Phase 2 components. |
| Context Observation | An append-only durable record of a predicted label, full class probabilities, model version, timestamp, and session ID. |
| Work Thread | A persistent, strictly user-created semantic work entity. It may be explicitly associated with historical Context Observations; it is not inferred or auto-created. |
| Task | A persistent, strictly user-created action belonging to exactly one Work Thread. |
| Unfinished Work | Tasks with `Task.is_done == False`, shown across Work Threads; do not redefine it as activity, an incomplete session, or an inferred intent. |
| Workspace Snapshot | A possible next-milestone record associated with a Work Thread. It must support only verified, best-effort restoration; it is not yet implemented. |

Never merge Context, Work Thread, and Task into one model or make a Work
Thread an ML-generated abstraction.

## Current implementation map

### Activity and sessions

- `app/core/activity_monitor.py` — `ActivityEvent`, title sanitization, and
  Windows foreground-window/process observation.
- `app/core/session_manager.py` — `Session`, `SessionUpdate`, and idle-gap
  session lifecycle.
- `app/core/monitoring_service.py` — polling, segment lifecycle, and
  coordination of monitor/repository/session manager.
- `app/database/activity_repository.py` — local SQLite persistence for
  `ActivitySegment`, `StoredActivity`, and `StoredSession`.

### ML/context pipeline — implemented, protected, and not to be redesigned

- `app/ml/feature_engineering.py` — `FeatureVector`, `FeatureExtractor`, and
  `BatchFeatureExtractor`.
- `app/ml/context_clustering.py` and `app/ml/context_labeling.py` — clustering
  and user-facing labeling support.
- `app/ml/training_examples.py` and `app/ml/training_pipeline.py` — durable
  labeled examples and retraining.
- `app/ml/context_classifier.py` — classifier contracts including
  `TrainingSummary`, `PredictionResult`, and `EvaluationResult`.
- `app/ml/context_inference.py` — inference engine.
- `app/ml/session_context_integration.py` — Phase 2G's narrow orchestration:
  completed session lookup → feature extraction → existing inference engine.
  It deliberately does not persist predictions, train models, create contexts,
  or introduce Work Thread/Task/Workspace concepts.
- `app/ml/context_observation_store.py` — Phase 2H durable,
  append-only `ContextObservation` persistence in `context_observations`.
- `app/ui/context_observation_coordinator.py` — Milestone A coordinator that
  finds completed, unobserved sessions and delegates to the existing Phase
  2G/2F/2D code. It loads the model lazily and does not redesign inference.
- `app/ui/context_inference_worker.py` — `QThread` wrapper for coordinator
  execution.

### Work Threads, Tasks, and dashboard

- `app/ml/work_thread_store.py` — `WorkThreadStore`, `WorkThread`, and
  `WorkThreadObservation`. It owns SQLite tables `work_threads` and
  `work_thread_observations`, with foreign keys enabled per connection.
  Associations are explicit and preserve Context Observations untouched.
  Work Thread deletion is blocked if associated observations or tasks exist.
- `app/ml/task_store.py` — `TaskStore` and `Task`; owns `tasks`. Each task has
  `id`, `work_thread_id`, `title`, UTC `created_at`, and boolean `is_done`.
  Its foreign key enforces one existing Work Thread per Task.
- `app/ui/dashboard_controller.py` — read-only UI composition plus narrow
  Work Thread/Task methods. `DashboardSnapshot.open_tasks` groups incomplete
  tasks by Work Thread, in Work Thread creation order.
- `app/ui/main_window.py` — PySide6 presentation and event wiring: Activity,
  latest context, Work Threads, Tasks, and Unfinished Work. Workspaces is a
  navigation label, not evidence of a workspace-restoration implementation.
- `app/ui/application.py` and `main.py` — composition and application launch.

All durable application data shares the local SQLite database, normally
`data/activity.db`. Store APIs use explicit timezone-aware UTC timestamps and
deterministic ordering. Preserve those contracts.

## UI and data-flow contract

1. Monitoring observes a Windows foreground application/window with sanitized
   title data and creates activity segments.
2. Sessions group these segments.
3. After a session is complete, the existing ML pipeline extracts features and
   makes an inference.
4. `ContextObservationCoordinator` writes one durable observation for each
   completed session that has not yet been observed. The observation preserves
   the full probability distribution and deterministic model version.
5. The user creates a Work Thread and can explicitly associate the displayed
   latest Context Observation with it.
6. The user creates and completes Tasks inside a selected Work Thread.
7. The dashboard derives Unfinished Work only by filtering `is_done == False`.

## Deliberate constraints / do-not-touch rules

- Do not redesign the existing ML pipeline, duplicate classifier/inference
  logic, or put training/retraining into session prediction.
- Do not modify unrelated `app/core` or ML modules for a Work Thread, Task, or
  workspace feature. Reuse existing public contracts.
- Do not auto-create Work Threads, Tasks, associations, contexts, or labels.
- Context Observations are durable and append-only; associations must never
  rewrite them.
- Do not add embeddings, similarity matching, LLMs, automatic task generation,
  deadlines, priorities, subtasks, or task/observation associations unless a
  separately approved milestone requests them.
- Preserve SQLite foreign-key enforcement, UTC behavior, deterministic sort
  order, public exception types, and architecture guard tests.
- Do not treat stale `README.md` status text (it says Phase 1D) as a current
  implementation description; Git and the current source clearly contain the
  later Phase 2 and milestone work.

## Known limitations / intentionally deferred work

- No workspace snapshot or restoration model/store/service/UI action exists.
- Routines, prediction, memory, work-stage detection, interruption detection,
  semantic memory, rediscovery, personalized scheduling, and mobile support
  remain future direction rather than current behavior.
- Browser tabs/authenticated browser state, arbitrary file/editor state,
  unsaved buffers, and arbitrary application automation have no approved
  restoration guarantee.
- Windows window/process control is inherently best-effort: permissions,
  elevation, missing executables, app-specific recovery, and timing may fail.

## Next milestone: Resume Work / Workspace Restoration

Do reconnaissance before implementation. Determine the smallest Windows-first
v1 from this existing codebase:

1. What reliable workspace facts can be captured from the present activity and
   Work Thread data without broadening privacy collection.
2. Which Windows APIs/dependencies already available (`pywin32`, `psutil`) can
   identify supported windows/processes and support a safe best-effort restore.
3. Exactly which applications/window state can be supported in v1, and a clear
   opt-in/capture model tied to a Work Thread.
4. Explicit deferrals: browser tabs/sessions, arbitrary files, unsaved state,
   and unsupported applications by default.
5. A small, isolated design: persistence schema/store, restoration-planning
   adapter, UI affordance, and error/status reporting. Avoid touching the ML
   pipeline.
6. Unit-test seams that mock OS/process launch behavior—tests must never launch
   real apps, alter windows, or change the user's desktop.
7. User-visible failure handling that describes best-effort results and never
   claims an exact restoration.

Do not implement this milestone, add dependencies, migrate data, or change
application behavior until the user gives a separate implementation request.

## Required first Codex response

After reading this file, `PROJECT_CONTEXT.md`, source, tests, and Git history,
return a concise checkpoint report with:

- current branch/working-tree status and relevant commits;
- the resolved discrepancy between `README.md` and the actual current source;
- verified module locations and flow for Activity, Session, ML Context,
  Context Observation, Work Thread, Task, and Unfinished Work;
- test command/result, including any environment blockage;
- preserved constraints and do-not-touch areas; and
- a no-code, small-scope Resume Work reconnaissance proposal.

Then stop and wait for approval.
