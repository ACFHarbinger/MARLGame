# MARLGame

> **A Comprehensive Multi-Agent Reinforcement Learning Framework for Unreal Engine 5.7**

[![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5.7-blue.svg)](https://www.unrealengine.com/)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.0+-orange.svg)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C++-17/20-00599C.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MARLGame is a cutting-edge research and development framework that implements state-of-the-art AI and optimization techniques for game development. Built on Unreal Engine 5.7 with LibTorch C++ integration, it provides a complete ecosystem for Multi-Agent Reinforcement Learning (MARL), Procedural Content Generation (PCG), Dynamic Difficulty Adjustment (DDA), and real-time rendering optimization.

---

## 🎯 Project Vision

Create a production-ready game framework that demonstrates:

- **Dynamic NPC Behaviors** through Multi-Agent Reinforcement Learning
- **Adaptive Game Content** via procedural generation and diffusion models
- **Player-Centric Difficulty** using flow theory and automated playtesting
- **Visual Quality Assurance** leveraging Vision-Language Models
- **Optimized Rendering** with quasi-Newton methods for real-time performance

---

## ✨ Key Features

### 🤖 Multi-Agent Reinforcement Learning (MARL)

- **CTDE Architecture**: Centralized Training with Decentralized Execution
- **6 Neural Network Architectures**: MLP, LSTM, GRU, Transformer, CNN, ResNet
- **Flexible Model Factory**: Dynamic model selection with configuration system
- **Advanced Algorithms**: PPO (Proximal Policy Optimization) and MAPPO (Multi-Agent PPO)
- **GAE Support**: Generalized Advantage Estimation for improved credit assignment
- **Cross-Platform**: Windows, Linux, macOS support

### 🧠 Neural Network Models

| Model           | Use Case               | Features                                                    |
| :-------------- | :--------------------- | :---------------------------------------------------------- |
| **MLP**         | Vector observations    | Fast, simple feedforward networks with layer norm & dropout |
| **LSTM**        | Temporal sequences     | Long Short-Term Memory for time-dependent policies          |
| **GRU**         | Efficient temporal     | Faster alternative to LSTM with similar capabilities        |
| **Transformer** | Relational reasoning   | Self-attention for multi-agent coordination                 |
| **CNN**         | Visual observations    | Convolutional networks for image-based input                |
| **ResNet**      | Deep visual processing | Residual networks for complex visual patterns               |

### 🎮 Game Modes

- **Twin Stick Shooter**: Fast-paced multi-agent combat
- **Strategy Game**: Turn-based tactical decision-making
- Both modes support MARL-controlled agents and human players

### 🔧 LibTorch Integration

- Full PyTorch C++ API support
- CUDA and CPU device management
- Automatic library detection and linking
- Memory-efficient tensor operations
- Model save/load with optimizer states

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [Research & Publications](#research--publications)
- [License](#license)

---

## 🚀 Installation

### Prerequisites

- **Unreal Engine**: 5.7 or higher
- **CMake**: 3.18+
- **Compiler**:
  - Windows: Visual Studio 2022
  - Linux: GCC 9+ or Clang 10+
  - macOS: Xcode 12+
- **LibTorch**: 2.0+ (CPU or CUDA version)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/MARLGame.git
cd MARLGame
git lfs pull
```

### Step 2: Install LibTorch

Download LibTorch from [PyTorch official site](https://pytorch.org/get-started/locally/):

**Option A: Set Environment Variable**

```bash
# Linux/macOS
export LIBTORCH_PATH=/path/to/libtorch

# Windows (PowerShell)
$env:LIBTORCH_PATH = "C:\path\to\libtorch"
```

**Option B: Place in Project Directory**

```
MARLGame/
├── Plugins/
│   └── MARLPlugin/
│       └── ThirdParty/
│           └── LibTorch/    # Place LibTorch here
```

See [LIBTORCH_SETUP.md](Plugins/MARLPlugin/LIBTORCH_SETUP.md) for detailed instructions.

### Step 3: Generate Project Files

```bash
# Right-click MARLGame.uproject → Generate Visual Studio project files (Windows)
# Or use command line:
UnrealBuildTool -projectfiles -project="/path/to/MARLGame.uproject"
```

### Step 4: Build the Project

**Windows:**

```bash
"C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" \
    MARLGameEditor Win64 Development \
    -Project="C:\path\to\MARLGame.uproject"
```

**Linux:**

```bash
/path/to/UE_5.7/Engine/Build/BatchFiles/Linux/Build.sh \
    MARLGameEditor Linux Development \
    -Project="/path/to/MARLGame.uproject"
```

### Step 5: Verify Installation

Launch Unreal Editor:

```bash
# Windows
MARLGame.uproject

# Linux
/path/to/UE_5.7/Engine/Binaries/Linux/UnrealEditor MARLGame.uproject
```

Check that MARLPlugin is loaded in **Edit → Plugins → Project → MARL Plugin**.

---

## 🎯 Quick Start

### 1. Basic Training Setup

```cpp
#if WITH_LIBTORCH

#include "MARLPPO.h"
#include "MARLTrainingEnvironment.h"
#include "Models/MARLModelFactory.h"

// Create training environment
auto TrainingEnv = NewObject<UMARLTrainingEnvironment>();
TrainingEnv->Initialize(EnvironmentManager, NumAgents, ObsSize, ActionSize);

// Create PPO agent
PPOAlgorithm Agent(
    ObsSize,              // Observation dimension
    ActionSize,           // Action dimension
    0.0003f,              // Learning rate
    0.99f,                // Gamma (discount factor)
    0.2f,                 // PPO clip epsilon
    0.95f,                // GAE lambda
    4,                    // Epochs
    64                    // Batch size
);

// Training loop
for (int Episode = 0; Episode < MaxEpisodes; Episode++)
{
    auto Obs = TrainingEnv->ResetTorch();
    float EpisodeReward = 0.0f;

    for (int Step = 0; Step < MaxSteps; Step++)
    {
        // Select action
        auto Action = Agent.SelectAction(Obs, false);

        // Step environment
        torch::Tensor NextObs, Reward;
        bool Done;
        TrainingEnv->StepTorch(Action, NextObs, Reward, Done);

        // Store transition
        Agent.StoreTransition(Obs, Action, Reward, Done);

        EpisodeReward += Reward.item<float>();
        Obs = NextObs;

        if (Done) break;
    }

    // Update policy
    auto [PolicyLoss, ValueLoss, Entropy] = Agent.Update();

    UE_LOG(LogTemp, Log, TEXT("Episode %d: Reward=%.2f"), Episode, EpisodeReward);
}

// Save trained model
Agent.SaveModel("/path/to/model.pt");

#endif
```

### 2. Using Different Neural Network Architectures

```cpp
#if WITH_LIBTORCH

// MLP (default)
auto Actor1 = std::make_shared<ActorNetwork>(ObsSize, ActionSize);

// LSTM for temporal sequences
NetworkConfig LSTMConfig;
LSTMConfig.BackboneType = EMARLModelType::LSTM;
LSTMConfig.HiddenSizes = {256, 256};
LSTMConfig.NumLayers = 3;
LSTMConfig.Dropout = 0.1f;
auto Actor2 = std::make_shared<ActorNetwork>(ObsSize, ActionSize, LSTMConfig);

// Transformer for multi-agent coordination
NetworkConfig TransformerConfig;
TransformerConfig.BackboneType = EMARLModelType::Transformer;
TransformerConfig.EmbedDim = 256;
TransformerConfig.NumHeads = 8;
TransformerConfig.NumLayers = 4;
auto Actor3 = std::make_shared<ActorNetwork>(ObsSize, ActionSize, TransformerConfig);

// CNN for visual observations
NetworkConfig CNNConfig;
CNNConfig.BackboneType = EMARLModelType::CNN;
CNNConfig.InputChannels = 3;
CNNConfig.InputHeight = 84;
CNNConfig.InputWidth = 84;
CNNConfig.HiddenSizes = {512};
auto Actor4 = std::make_shared<ActorNetwork>(ObsSize, ActionSize, CNNConfig);

#endif
```

### 3. Multi-Agent Setup (MAPPO)

```cpp
#if WITH_LIBTORCH

#include "MARLPPO.h"

// Create MAPPO algorithm for 4 agents
MAPPOAlgorithm MAPPO(
    NumAgents,           // 4 agents
    ObsSize,             // Local observation size
    GlobalStateSize,     // Global state size (concatenated obs)
    ActionSize,
    0.0003f,             // Learning rate
    0.99f,               // Gamma
    0.2f,                // Clip epsilon
    0.95f,               // Lambda
    4,                   // Epochs
    64                   // Batch size
);

// Training loop
for (int Episode = 0; Episode < MaxEpisodes; Episode++)
{
    auto Observations = Env->ResetMultiAgent();

    for (int Step = 0; Step < MaxSteps; Step++)
    {
        // Select actions for all agents
        auto Actions = MAPPO.SelectActions(Observations, false);

        // Step environment
        torch::Tensor NextObs, Rewards, GlobalState;
        bool Done;
        Env->StepMultiAgent(Actions, NextObs, Rewards, GlobalState, Done);

        // Store transition with global state
        MAPPO.StoreTransition(Observations, GlobalState, Actions, Rewards, Done);

        Observations = NextObs;
        if (Done) break;
    }

    // Update all agents
    MAPPO.Update();
}

#endif
```

---

## 🏗️ Architecture

### Project Structure

```
MARLGame/
├── Content/                          # Unreal Engine assets
│   ├── TwinStickShooter/            # Twin stick game mode assets
│   └── Strategy/                     # Strategy game mode assets
├── Plugins/
│   └── MARLPlugin/                   # Main MARL plugin
│       ├── Source/
│       │   └── MARLPlugin/
│       │       ├── Public/
│       │       │   ├── Models/       # Neural network architectures
│       │       │   │   ├── MARLModelBase.h
│       │       │   │   ├── MLPModel.h
│       │       │   │   ├── LSTMModel.h
│       │       │   │   ├── GRUModel.h
│       │       │   │   ├── TransformerModel.h
│       │       │   │   ├── CNNModel.h
│       │       │   │   ├── ResNetModel.h
│       │       │   │   └── MARLModelFactory.h
│       │       │   ├── MCTS/         # Monte Carlo Tree Search
│       │       │   ├── MARLNeuralNetwork.h
│       │       │   ├── MARLPPO.h
│       │       │   ├── MARLTrainingEnvironment.h
│       │       │   └── MARLTypes.h
│       │       └── Private/
│       │           ├── Models/       # Model implementations
│       │           ├── MCTS/
│       │           ├── MARLNeuralNetwork.cpp
│       │           ├── MARLPPO.cpp
│       │           └── MARLTrainingEnvironment.cpp
│       ├── ThirdParty/
│       │   └── LibTorch/             # LibTorch library (optional location)
│       ├── MARLPlugin.uplugin
│       └── MARLPlugin.Build.cs       # Build configuration
├── Source/                           # Game source code
│   └── MARLGame/
├── MARLGame.uproject                 # Unreal project file
├── README.md                         # This file
├── ROADMAP.md                        # Development roadmap
└── Documentation/                    # Additional documentation
    ├── LIBTORCH_SETUP.md
    ├── MODELS_README.md
    ├── MODELS_QUICK_REF.md
    └── MLP_REFACTORING_SUMMARY.md
```

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Unreal Engine 5.7                        │
│                   (Game Environment)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   MARLPlugin Core                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          MARLTrainingEnvironment                     │   │
│  │  • Reset/Step interface                             │   │
│  │  • Tensor conversion (UE ↔ LibTorch)                │   │
│  │  • Reward computation                               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Neural Networks (LibTorch)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MLP        │  │   LSTM       │  │   GRU        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Transformer  │  │   CNN        │  │   ResNet     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Model Factory                             │   │
│  │  • Dynamic model creation                           │   │
│  │  • Configuration management                         │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Actor-Critic Architecture                       │
│  ┌──────────────┐                    ┌──────────────┐       │
│  │ ActorNetwork │                    │CriticNetwork │       │
│  │   (Policy)   │                    │   (Value)    │       │
│  └──────────────┘                    └──────────────┘       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      CentralizedCriticNetwork (MAPPO)               │    │
│  │      • Global state evaluation                      │    │
│  │      • Multi-agent coordination                     │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 RL Algorithms                                │
│  ┌──────────────┐              ┌──────────────┐             │
│  │     PPO      │              │    MAPPO     │             │
│  │  • GAE       │              │ • Centralized│             │
│  │  • Clipping  │              │   Critic     │             │
│  │  • Entropy   │              │ • Multi-Agent│             │
│  └──────────────┘              └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Documentation

### Core Documentation

- **[ROADMAP.md](ROADMAP.md)** - Complete development roadmap with 7 phases
- **[LIBTORCH_SETUP.md](Plugins/MARLPlugin/LIBTORCH_SETUP.md)** - LibTorch installation and configuration
- **[MODELS_README.md](Plugins/MARLPlugin/MODELS_README.md)** - Comprehensive guide to all neural network models
- **[MODELS_QUICK_REF.md](Plugins/MARLPlugin/MODELS_QUICK_REF.md)** - Quick reference for model usage
- **[MLP_REFACTORING_SUMMARY.md](MLP_REFACTORING_SUMMARY.md)** - MLP architecture refactoring details

### Neural Network Models

Each model has detailed documentation including:

- Architecture diagrams
- Mathematical formulations
- Use cases and recommendations
- Code examples
- Performance characteristics

See [MODELS_README.md](Plugins/MARLPlugin/MODELS_README.md) for complete details.

### API Reference

Generate API documentation using Doxygen:

```bash
doxygen Doxyfile
```

---

## 🗺️ Roadmap

MARLGame follows a comprehensive **7-phase development plan** spanning 28 weeks:

| Phase       | Focus                               | Status           |
| :---------- | :---------------------------------- | :--------------- |
| **Phase 1** | MARL Foundation & CTDE Architecture | ✅ **COMPLETED** |
| **Phase 2** | MCTS & Advanced Search              | 📋 Planned       |
| **Phase 3** | Procedural Content Generation       | 📋 Planned       |
| **Phase 4** | Dynamic Difficulty Adjustment       | 📋 Planned       |
| **Phase 5** | Visual QA with VLMs                 | 📋 Planned       |
| **Phase 6** | L-BFGS Rendering Optimization       | 📋 Planned       |
| **Phase 7** | Integration & Deployment            | 📋 Planned       |

### Phase 1 Completed ✅

- ✅ LibTorch integration with cross-platform support
- ✅ 6 neural network architectures (MLP, LSTM, GRU, Transformer, CNN, ResNet)
- ✅ Model factory with flexible configuration
- ✅ Actor-Critic networks with multiple backbone options
- ✅ PPO and MAPPO algorithms with GAE
- ✅ Rollout buffer and mini-batch training
- ✅ Model save/load functionality
- ✅ Comprehensive documentation

See [ROADMAP.md](ROADMAP.md) for complete details on all phases.

---

## ⚡ Performance Benchmarks

### Current Status (Phase 1)

| Metric                           | Target              | Status           |
| :------------------------------- | :------------------ | :--------------- |
| **Neural Network Architectures** | 6 models            | ✅ 6/6           |
| **Cross-Platform Support**       | Win/Linux/Mac       | ✅ All platforms |
| **CUDA Support**                 | GPU acceleration    | ✅ Implemented   |
| **Model Factory**                | Dynamic creation    | ✅ Functional    |
| **PPO/MAPPO**                    | Full implementation | ✅ Complete      |

### Future Targets (Phases 2-6)

| System     | Metric                     | Target        |
| :--------- | :------------------------- | :------------ |
| **MCTS**   | Decision time (1000 sims)  | <100ms        |
| **PCG**    | Level generation           | <5s per level |
| **DDA**    | Player engagement increase | +20%          |
| **VLM QA** | Glitch detection accuracy  | >85%          |
| **L-BFGS** | 4K rendering @ 60 FPS      | ✅ Target     |

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow C++ coding standards
   - Add unit tests for new features
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Code Standards

- **C++ Style**: Follow [Unreal Engine Coding Standard](https://docs.unrealengine.com/en-US/Programming/Development/CodingStandard/)
- **Comments**: Use Doxygen-style documentation
- **Testing**: All new features must include unit tests
- **Performance**: Profile code changes for performance impact

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix issues
- 📝 **Documentation**: Improve guides and examples
- 🧪 **Testing**: Add test coverage
- ✨ **Features**: Implement roadmap phases
- 🎨 **Examples**: Create demo projects
- 🌍 **Localization**: Translate documentation

---

## 📖 Research & Publications

MARLGame is designed with research-grade rigor and targets top-tier AI conferences:

### Target Conferences

- **NeurIPS**: MARL with CTDE architecture, MCTS frameworks
- **ICML**: L-BFGS rendering optimization
- **ICLR**: Novelty Search + Transformer PCG
- **CVPR**: DI-PCG diffusion-based inverse generation

### Mathematical Rigor

All algorithms include:

- ✅ Formal mathematical definitions
- ✅ Theoretical derivations and proofs
- ✅ Rigorous baseline comparisons
- ✅ Ablation studies
- ✅ Statistical significance testing

### Experimental Standards

- Baseline comparisons against SOTA (QMIX, MAPPO)
- Sparse reward regime testing
- Generalization to unseen environments
- Precision-recall analysis
- Class imbalance handling

See [ROADMAP.md](ROADMAP.md) for detailed research objectives.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research Foundations

This project builds upon cutting-edge research in:

- Multi-Agent Reinforcement Learning (MARL)
- Monte Carlo Tree Search (MCTS)
- Procedural Content Generation (PCG)
- Dynamic Difficulty Adjustment (DDA)
- Vision-Language Models (VLMs)
- Quasi-Newton Optimization Methods

### Key References

- **MARL**: Centralized Training with Decentralized Execution (CTDE)
- **MCTS**: Upper Confidence Bound for Trees (UCT)
- **PCG**: MarioGPT, DI-PCG diffusion models
- **DDA**: Flow theory, expressive range analysis
- **VLM**: VideoGameQA-Bench framework
- **L-BFGS**: Limited-memory quasi-Newton methods

See [AI and Optimization in Game Development.md](AI%20and%20Optimization%20in%20Game%20Development.md) for comprehensive references.

### Technologies

- [Unreal Engine 5.7](https://www.unrealengine.com/)
- [PyTorch LibTorch](https://pytorch.org/cppdocs/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/MARLGame/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MARLGame/discussions)
- **Email**: your.email@example.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

<div align="center">

**Built with ❤️ for AI Research and Game Development**

[Website](https://yourwebsite.com) • [Documentation](https://docs.yourwebsite.com) • [Blog](https://blog.yourwebsite.com)

</div>
