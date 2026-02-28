# MARL Implementation Roadmap

This document tracks the progress of the Multi-Agent Reinforcement Learning (MARL) implementation for MARLGame.

## Status Overview

| Phase | Description                   | Status   | Target Completion |
| :---- | :---------------------------- | :------- | :---------------- |
| **1** | **Foundation & Architecture** | **DONE** | Week 2            |
| **2** | **Twin Stick MARL**           | **DONE** | Week 5            |
| **3** | **Strategy MARL**             | **DONE** | Week 8            |
| 4     | C++ LibTorch Integration      | PLANNED  | Week 11           |
| 5     | Training Infrastructure       | PLANNED  | Week 13           |
| 6     | Advanced Features             | PLANNED  | Week 16           |
| 7     | Deployment & Evaluation       | PLANNED  | Week 18           |

---

## Detailed Progress

### Phase 1: Foundation & Architecture (Weeks 1-2) [DONE]

- [x] **Week 1: Plugin Setup & Core Infrastructure**
  - [x] Create MARL Plugin structure (`Plugins/MARLPlugin`)
  - [x] Implement `UMARLAgentComponent` (Observation/Action interfaces)
  - [x] Implement `AMARLEnvironmentManager` (Registration, Reset, Step)
  - [x] Implement `UMARLLibTorchInterface` (IPC/Shared Memory skeleton)
  - [x] Define `MARLTypes.h` (Observation/Action spaces)
- [x] **Week 2: Sensor System & Configuration**
  - [x] `UMARLRaycastSensor`: 360° spatial awareness
  - [x] `UMARLCameraSensor`: Visual observation interface
  - [x] `UMARLGameStateSensor`: Internal state access
  - [x] `UMARLConfig`: JSON-loadable DataAsset for MARL parameters
  - [x] `ROADMAP.md`: Created to track implementation progress

### Phase 2: Twin Stick MARL (Weeks 3-5) [PLANNED]

- [x] **Week 3: Single Agent Integration**
  - [x] Implement `AMARLCharacter` base class
  - [x] Integrate `UMARLAgentComponent` with Twin Stick pawn
  - [x] Map movement/shooting to `FMARLAction`
- [x] **Week 4: Multi-Agent Support**
  - [x] Coordinate multiple agents in Twin Stick arena
  - [x] Implement team-based reward functions
- [x] **Week 5: NPCs & Adversaries**
  - [x] Scripted vs RL-controlled enemy behaviors
  - [x] Environment randomization (spawns, obstacles)

### Phase 3: Strategy MARL (Weeks 6-8) [IN PROGRESS]

- [x] **Week 6: Strategy Unit Integration**
  - [x] Implement `IMARLAgentInterface` for unified control
  - [x] Integrate `AStrategyUnit` with `AMARLCharacter`
- [x] **Week 7: Hierarchical Control**
  - [x] Implement `AStrategyCommanderPawn`
  - [x] Commander RL policy (macro) vs Unit policies (micro)
- [x] **Week 8: Team Coordination**
  - [x] Formations and tactical behaviors
  - [x] Shared team observations structure

### Phase 4: C++ LibTorch Integration (Weeks 9-11) [PLANNED]

- [ ] **Week 9: Environment Wrapper**
  - [ ] High-performance C++ wrapper for UE environment
- [ ] **Week 10: Training Implementation**
  - [ ] MAPPO/PPO algorithm implementation in LibTorch
- [ ] **Week 11: Logging & Optimization**
  - [ ] TensorBoard C++ integration

### Phase 5: Training Infrastructure (Weeks 12-13) [PLANNED]

- [ ] **Week 12: Multi-instance Training**
- [ ] **Week 13: Curriculum Learning**

### Phase 6: Advanced Features (Weeks 14-16) [PLANNED]

- [ ] **Week 14: Agent Communication** (CommNet/TarMAC)
- [ ] **Week 15: Self-Play & League Training**
- [ ] **Week 16: Transfer Learning**

### Phase 7: Deployment & Evaluation (Weeks 17-18) [PLANNED]

- [ ] **Week 17: Production Inference** (TorchScript)
- [ ] **Week 18: Evaluation & Human Playtesting**
