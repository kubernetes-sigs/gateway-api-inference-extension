# SLO Aware Routing Refactor Implementation Plan

## 1. Introduction

The objective of this refactor is to decouple the SLO-aware routing logic from the core request handling pipeline. We will move the existing hardcoded logic into a dedicated, plugin-based scheduling profile. This will improve modularity, testability, and maintainability, while isolating SLO-aware functionality to prevent disruption of other features.

This plan outlines the steps to transition from the current implementation to the desired plugin-based architecture, as described in `slo_aware_refactor.md`.

---

## 2. Phase 1: Creating New SLO-Aware Components

This phase focuses on creating the new, self-contained components for the SLO-aware scheduling profile.

### Task 2.1: Create the SLO Scorer Plugin

This plugin will encapsulate the core logic of predicting latency and scoring pods based on SLOs.

- **Create New File**: `pkg/epp/scheduler/plugins/sloscorer/slo_scorer.go`
- **Define `SLOScorer` struct**: This struct will implement the `ScorePlugin` and `PreFilterPlugin` interfaces from the scheduling framework. It will require access to the `LatencyPredictor` and `Datastore`.
- **Implement `Name()`**: Return `"SLOScorer"`.
- **Implement `PreFilter()`**: This method will run before any scoring. It will perform an initial check to ensure that the request has the necessary SLOs (`ttft_slo`, `avg_tpot_slo`) defined in its headers. If not, it can return a status that skips this plugin for the request.
- **Implement `Score()`**:
    - Move the logic from `ScoreAndFilterPods` in `pkg/epp/requestcontrol/prediction_based_scorer.go` into this method.
    - The method will iterate through candidate pods.
    - For each pod, it will:
        1. Get the `prefix_cache_score` (this assumes the prefix cache scorer has already run).
        2. Call the latency predictor.
        3. Validate the prediction against the request's SLOs (`validatePrediction` logic).
        4. Calculate a score based on the headroom (`Headroom-weighted draw` logic). The score should be normalized (e.g., 1-100). Pods that don't meet the SLO should receive a minimal score.
- **Dependency Injection**: The `SLOScorer` will need the `LatencyPredictor` and `Datastore`. These dependencies should be provided during its instantiation in the main application setup.

### Task 2.2: Create the Request Lifecycle Plugin

This plugin will manage adding and removing requests from a pod's running request queue, a task currently split between the `director` and `response handler`.

- **Create New File**: `pkg/epp/requestcontrol/plugins/slorequest/slo_request_tracker.go`
- **Define `SLORequestTracker` struct**: This struct will implement the `PreRequest` and `PostResponse` plugin interfaces. It will need access to the `Datastore`.
- **Implement `Name()`**: Return `"SLORequestTracker"`.
- **Implement `PreRequest()`**:
    - This method will be called after a pod has been selected.
    - It will contain the logic currently in `director.go`'s `prepareRequest` function to add the request to the pod's queue: `d.datastore.PodAddRequest(...)`.
- **Implement `PostResponse()`**:
    - This method will be called when the response is complete.
    - It will contain the logic currently in `handlers/response.go`'s `HandleResponseBodyModelStreaming` to remove the request from the pod's queue: `s.director.GetDatastore().PodRemoveRequest(...)`.
- **Dependency Injection**: The `SLORequestTracker` will need the `Datastore`, which will be provided during its instantiation.

### Task 2.3: Define the `slo-aware` Scheduling Profile

A new scheduling profile will be defined in the application's configuration. This profile will orchestrate the execution of the new plugins.

- **Configuration**: In the scheduler configuration (likely initialized in `cmd/epp/main.go`), define a new profile named `slo-aware`.
- **Plugin-Set**: The `slo-aware` profile will be configured with the following plugins in order:
    1. **Filters**: Default filters.
    2. **Scorers**:
        - `PrefixCacheScorer` (existing)
        - `SLOScorer` (new)
    3. **Picker**:
        - A `WeightedRandom` picker that respects the scores from the scorers. Invalid pods should be given a very low weight as per the existing logic.

---

## 3. Phase 2: Integrating New Components and Refactoring

This phase involves modifying the existing codebase to remove the old logic and integrate the new plugin-based flow.

### Task 3.1: Modify `pkg/epp/requestcontrol/director.go`

- **Remove `applyPredictionScoring`**: Delete the `applyPredictionScoring` method and its call within `HandleRequest`. The `SLOScorer` now handles this.
- **Remove `PodAddRequest` call**: In the `prepareRequest` method, remove the direct call to `d.datastore.PodAddRequest`. The `SLORequestTracker` `PreRequest` plugin now handles this.
- **Implement Profile Selection**:
    - In `HandleRequest`, before calling `d.scheduler.Schedule`, add logic to select the scheduling profile.
    - If the latency predictor is enabled (`d.latencyPredictor != nil` and SLOs are provided), instruct the scheduler to use the `slo-aware` profile for this request. Otherwise, it should use the default profile. This can be done by passing a profile name or context to the scheduler.

### Task 3.2: Modify `pkg/epp/handlers/response.go`

- **Remove `PodRemoveRequest` call**: In the `HandleResponseBodyModelStreaming` method, remove the call to `s.director.GetDatastore().PodRemoveRequest`. The `SLORequestTracker` `PostResponse` plugin now handles this.

### Task 3.3: Update Scheduler and Director Configuration

- **Location**: `cmd/epp/main.go` or a similar setup file.
- **Register New Plugins**: Instantiate and register the `SLOScorer` and `SLORequestTracker` plugins with the scheduler and director respectively.
- **Configure `slo-aware` Profile**: Add the `slo-aware` profile to the scheduler's configuration, associating it with the correct plugins as defined in Task 2.3.
- **Pass Dependencies**: Ensure the `LatencyPredictor` and `Datastore` are correctly passed to the new plugins during their creation.

---

## 4. Phase 3: Cleanup

### Task 4.1: Delete Obsolete File

- **Remove File**: Once all logic has been migrated and the refactor is verified, delete the now-redundant file: `pkg/epp/requestcontrol/prediction_based_scorer.go`.

---

## 5. Summary of File Changes

| Action    | File Path                                                              | Reason                                                                          |
| :-------- | :--------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **Create**  | `pkg/epp/scheduler/plugins/sloscorer/slo_scorer.go`                    | New plugin to house the SLO-based scoring logic.                                |
| **Create**  | `pkg/epp/requestcontrol/plugins/slorequest/slo_request_tracker.go`     | New plugin to manage adding/removing requests from the pod queue.               |
| **Modify**  | `pkg/epp/requestcontrol/director.go`                                   | Remove old hardcoded logic, add profile selection logic.                        |
| **Modify**  | `pkg/epp/handlers/response.go`                                         | Remove request removal logic, now handled by a plugin.                          |
| **Modify**  | `cmd/epp/main.go` (or equivalent config file)                          | Register new plugins and configure the `slo-aware` scheduling profile.          |
| **Delete**  | `pkg/epp/requestcontrol/prediction_based_scorer.go`                    | This file's logic is moved to the new `SLOScorer` plugin.                       |
