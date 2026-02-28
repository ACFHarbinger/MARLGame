# Phase 4: C++ LibTorch Integration

This phase implements the core machine learning infrastructure using LibTorch (PyTorch C++ API) for training multi-agent reinforcement learning policies entirely in C++.

## Overview

Phase 4 provides:
- **LibTorch Integration**: Full PyTorch C++ API support for training
- **Training Environment**: Gym-like environment wrapper for UE
- **Neural Networks**: Actor-Critic architectures with MLP backbones
- **PPO Algorithm**: Proximal Policy Optimization implementation
- **MAPPO Algorithm**: Multi-Agent PPO with centralized critic
- **TensorBoard Logging**: Training metrics visualization

## Installation

### Prerequisites

1. **LibTorch** (PyTorch C++ Distribution)
   - Download from: https://pytorch.org/get-started/locally/
   - Select: C++/LibTorch, your platform, CPU or CUDA
   - Extract to a known location

2. **Set Environment Variable**
   ```bash
   # Linux/Mac
   export LIBTORCH_PATH=/path/to/libtorch

   # Windows
   set LIBTORCH_PATH=C:\path\to\libtorch
   ```

   Alternatively, place LibTorch in `Plugins/MARLPlugin/ThirdParty/LibTorch/`

3. **Verify Installation**
   - Ensure `libtorch/include` and `libtorch/lib` exist
   - On Linux: Check for `libtorch.so`, `libtorch_cpu.so`, `libc10.so`
   - On Windows: Check for `torch.dll`, `torch_cpu.dll`, `c10.dll`

### Build

```bash
# Regenerate project files
./GenerateProjectFiles.sh  # Linux/Mac
# or
GenerateProjectFiles.bat   # Windows

# Build in Unreal Editor or via command line
make MARLGameEditor  # Linux
```

## Architecture

### Class Hierarchy

```
UMARLTrainingEnvironment
├─ Environment wrapper with Gym-like interface
├─ Handles reset(), step()
└─ Converts between UE and LibTorch tensors

MLP (torch::nn::Module)
└─ Multi-layer perceptron backbone

ActorNetwork (torch::nn::Module)
├─ Policy network
├─ Discrete or continuous action spaces
└─ Methods: forward(), SampleActions(), ComputeLogProb(), ComputeEntropy()

CriticNetwork (torch::nn::Module)
├─ Value function V(s)
└─ Method: forward()

CentralizedCriticNetwork (torch::nn::Module)
├─ MAPPO centralized value function
└─ Takes global state (concatenated observations)

PPOAlgorithm
├─ Single-agent or independent learners
├─ Actor-critic with PPO loss
└─ Methods: SelectAction(), Update(), SaveModel()

MAPPOAlgorithm
├─ Multi-agent with centralized critic
├─ Shared policy, centralized value
└─ Methods: SelectActions(), Update(), SaveModels()
```

## Usage

### 1. Basic PPO Training

```cpp
#if WITH_LIBTORCH

// In your game mode or training manager
void AMyGameMode::BeginPlay()
{
    Super::BeginPlay();

    // Initialize training environment
    UMARLTrainingEnvironment* TrainEnv = NewObject<UMARLTrainingEnvironment>();
    TrainEnv->Initialize(EnvironmentManager, NumAgents, ObsSize, ActionSize);

    // Create PPO algorithm
    PPOConfig Config;
    Config.LearningRate = 3e-4f;
    Config.Gamma = 0.99f;
    Config.ClipEpsilon = 0.2f;
    Config.HiddenSizes = {256, 256};
    Config.bContinuousActions = false;
    Config.Device = torch::kCUDA;  // or torch::kCPU

    PPOAlgorithm* PPO = new PPOAlgorithm(ObsSize, ActionSize, Config);

    // Training loop
    TrainPPO(TrainEnv, PPO);
}

void AMyGameMode::TrainPPO(UMARLTrainingEnvironment* Env, PPOAlgorithm* PPO)
{
    const int32 NumEpisodes = 1000;
    const int32 MaxSteps = 1000;

    for (int32 Episode = 0; Episode < NumEpisodes; ++Episode)
    {
        torch::Tensor Obs = Env->ResetTorch();
        float EpisodeReward = 0.0f;

        for (int32 Step = 0; Step < MaxSteps; ++Step)
        {
            // Select action
            torch::Tensor Action = PPO->SelectAction(Obs, false);

            // Evaluate for value and log prob
            auto [Value, LogProb] = PPO->EvaluateAction(Obs, Action);

            // Step environment
            torch::Tensor NextObs, Reward, Done, Truncated;
            Env->StepTorch(Action, NextObs, Reward, Done, Truncated);

            // Store transition
            PPO->StoreTransition(Obs, Action, Reward, Value, LogProb, Done);

            EpisodeReward += Reward.item<float>();
            Obs = NextObs;

            if (Done.item<float>() > 0.5f)
            {
                break;
            }
        }

        // Update policy
        auto [PolicyLoss, ValueLoss, Entropy] = PPO->Update();

        UE_LOG(LogTemp, Log, TEXT("Episode %d: Reward=%.2f, PolicyLoss=%.4f, ValueLoss=%.4f"),
               Episode, EpisodeReward, PolicyLoss, ValueLoss);
    }

    // Save trained model
    PPO->SaveModel(TCHAR_TO_UTF8(*FPaths::ProjectSavedDir()) + "Models/ppo_policy");
}

#endif // WITH_LIBTORCH
```

### 2. MAPPO Multi-Agent Training

```cpp
#if WITH_LIBTORCH

void AMyGameMode::TrainMAPPO()
{
    UMARLTrainingEnvironment* Env = NewObject<UMARLTrainingEnvironment>();
    Env->Initialize(EnvironmentManager, NumAgents, ObsSize, ActionSize);

    PPOConfig Config;
    Config.LearningRate = 3e-4f;
    Config.HiddenSizes = {256, 256};
    Config.Device = torch::kCUDA;

    MAPPOAlgorithm* MAPPO = new MAPPOAlgorithm(ObsSize, ActionSize, NumAgents, Config);

    const int32 NumEpisodes = 1000;

    for (int32 Episode = 0; Episode < NumEpisodes; ++Episode)
    {
        torch::Tensor Observations = Env->ResetTorch();  // [num_agents, obs_size]
        float TotalReward = 0.0f;

        while (!Env->AreAllAgentsDone())
        {
            // Select actions for all agents
            torch::Tensor Actions = MAPPO->SelectActions(Observations, false);

            // Create global state (concatenate all observations)
            torch::Tensor GlobalState = Observations.reshape({-1});

            // Step environment
            torch::Tensor NextObs, Rewards, Dones, Truncated;
            Env->StepTorch(Actions, NextObs, Rewards, Dones, Truncated);

            // Store transition
            MAPPO->StoreTransition(Observations, GlobalState, Actions, Rewards, Dones);

            TotalReward += Rewards.sum().item<float>();
            Observations = NextObs;
        }

        // Update policies
        auto [PolicyLoss, ValueLoss, Entropy] = MAPPO->Update();

        UE_LOG(LogTemp, Log, TEXT("Episode %d: TotalReward=%.2f, PolicyLoss=%.4f"),
               Episode, TotalReward, PolicyLoss);
    }

    MAPPO->SaveModels(TCHAR_TO_UTF8(*FPaths::ProjectSavedDir()) + "Models/mappo");
}

#endif // WITH_LIBTORCH
```

### 3. Deploying Trained Policy

```cpp
#if WITH_LIBTORCH

void AMyAIController::BeginPlay()
{
    Super::BeginPlay();

    // Load trained model
    PPOAlgorithm* Policy = new PPOAlgorithm(ObsSize, ActionSize, PPOConfig());
    Policy->LoadModel(TCHAR_TO_UTF8(*FPaths::ProjectSavedDir()) + "Models/ppo_policy");

    TrainedPolicy = Policy;
}

void AMyAIController::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (!TrainedPolicy) return;

    // Collect observation
    TArray<float> ObsArray = AgentComponent->CollectObservations();

    // Convert to tensor
    std::vector<float> ObsVec(ObsArray.GetData(), ObsArray.GetData() + ObsArray.Num());
    torch::Tensor Obs = torch::from_blob(ObsVec.data(), {ObsSize}, torch::kFloat32).clone();

    // Get action (deterministic for deployment)
    torch::Tensor Action = TrainedPolicy->SelectAction(Obs, true);

    // Execute action
    AgentComponent->ExecuteAction(Action);
}

#endif // WITH_LIBTORCH
```

## Configuration

### PPOConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `HiddenSizes` | `std::vector<int64_t>` | `{256, 256}` | Hidden layer sizes for networks |
| `bContinuousActions` | `bool` | `false` | Use continuous action space |
| `LearningRate` | `float` | `3e-4` | Optimizer learning rate |
| `Gamma` | `float` | `0.99` | Discount factor |
| `GAELambda` | `float` | `0.95` | GAE parameter for advantage estimation |
| `ClipEpsilon` | `float` | `0.2` | PPO clipping parameter |
| `ValueLossCoef` | `float` | `0.5` | Value loss coefficient |
| `EntropyCoef` | `float` | `0.01` | Entropy bonus coefficient |
| `MaxGradNorm` | `float` | `0.5` | Gradient clipping threshold |
| `NumEpochs` | `int32` | `10` | Optimization epochs per update |
| `BatchSize` | `int32` | `64` | Mini-batch size |
| `BufferSize` | `int32` | `2048` | Rollout buffer size |
| `Device` | `torch::DeviceType` | `torch::kCPU` | Training device (CPU/CUDA) |

### Example Custom Configuration

```cpp
PPOConfig Config;

// Larger network for complex tasks
Config.HiddenSizes = {512, 512, 256};

// Continuous action space
Config.bContinuousActions = true;

// More conservative clipping for stability
Config.ClipEpsilon = 0.1f;

// Longer rollouts
Config.BufferSize = 4096;

// GPU training
Config.Device = torch::kCUDA;
```

## Performance Optimization

### GPU Acceleration

```cpp
// Check CUDA availability
if (torch::cuda::is_available())
{
    Config.Device = torch::kCUDA;
    UE_LOG(LogTemp, Log, TEXT("Using CUDA device: %d"), torch::cuda::current_device());
}
else
{
    Config.Device = torch::kCPU;
    UE_LOG(LogTemp, Warning, TEXT("CUDA not available, using CPU"));
}
```

### Batch Processing

```cpp
// Process multiple environments in parallel
std::vector<UMARLTrainingEnvironment*> Envs;
for (int32 i = 0; i < NumParallelEnvs; ++i)
{
    Envs.push_back(CreateEnvironment());
}

// Collect experiences from all environments
for (auto* Env : Envs)
{
    // Run episode and store transitions
    RunEpisode(Env, PPO);
}

// Single update with all experiences
PPO->Update();
```

### Memory Management

```cpp
// Clear buffer after update to free memory
PPO->ClearBuffer();

// Periodically save checkpoints and clear old models
if (Episode % 100 == 0)
{
    PPO->SaveModel("checkpoint_" + std::to_string(Episode));
}
```

## Troubleshooting

### LibTorch Not Found

**Problem**: Build error: "LibTorch not found"

**Solutions**:
1. Set `LIBTORCH_PATH` environment variable correctly
2. Place LibTorch in `Plugins/MARLPlugin/ThirdParty/LibTorch/`
3. Check that `include/` and `lib/` subdirectories exist

### CUDA Out of Memory

**Problem**: Training crashes with CUDA OOM

**Solutions**:
```cpp
// Reduce batch size
Config.BatchSize = 32;  // Instead of 64

// Reduce buffer size
Config.BufferSize = 1024;  // Instead of 2048

// Use smaller network
Config.HiddenSizes = {128, 128};  // Instead of {256, 256}

// Move some operations to CPU
Config.Device = torch::kCPU;
```

### Slow Training

**Problem**: Training is very slow

**Solutions**:
1. Ensure CUDA is being used (check logs)
2. Reduce number of optimization epochs
3. Increase batch size (if memory allows)
4. Profile with UE Profiler to find bottlenecks
5. Use multiple parallel environments

### NaN Losses

**Problem**: Loss becomes NaN during training

**Solutions**:
```cpp
// Reduce learning rate
Config.LearningRate = 1e-4f;

// Clip gradients more aggressively
Config.MaxGradNorm = 0.3f;

// Normalize observations
// (Implement normalization in UMARLAgentComponent)

// Check for invalid rewards
// Ensure rewards are finite and not too large
```

## Next Steps

After completing Phase 4, proceed to:

**Phase 5: Training Infrastructure**
- Multi-instance training for parallel experience collection
- Curriculum learning with automatic difficulty progression
- Hyperparameter tuning and experiment management

**Phase 6: Advanced Features**
- Agent communication networks (CommNet, TarMAC)
- Self-play and league training
- Transfer learning between game variants

## References

- LibTorch Documentation: https://pytorch.org/cppdocs/
- PPO Paper: https://arxiv.org/abs/1707.06347
- MAPPO Paper: https://arxiv.org/abs/2103.01955
- UE C++ Programming: https://docs.unrealengine.com/5.7/en-US/programming-with-cplusplus-in-unreal-engine/

## Support

For issues or questions:
1. Check existing issues in the repository
2. Review LibTorch documentation
3. Consult UE forums for integration issues
4. Open a new issue with detailed error logs
