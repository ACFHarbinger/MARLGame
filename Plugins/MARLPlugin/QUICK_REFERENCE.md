# MARL Plugin Quick Reference

Quick reference for common operations with the MARL Plugin.

## Setup

```bash
# Install LibTorch
export LIBTORCH_PATH=/path/to/libtorch

# Rebuild project
make MARLGameEditor
```

## Training - PPO

```cpp
#if WITH_LIBTORCH

// Initialize
PPOConfig config;
config.Device = torch::kCUDA;
config.LearningRate = 3e-4f;

PPOAlgorithm* ppo = new PPOAlgorithm(obsSize, actionSize, config);

// Train
for (episode) {
    obs = env->ResetTorch();
    while (!done) {
        action = ppo->SelectAction(obs, false);
        auto [value, logProb] = ppo->EvaluateAction(obs, action);
        env->StepTorch(action, nextObs, reward, done, truncated);
        ppo->StoreTransition(obs, action, reward, value, logProb, done);
        obs = nextObs;
    }
    ppo->Update();
}

// Save
ppo->SaveModel("models/ppo_final");

#endif
```

## Training - MAPPO

```cpp
#if WITH_LIBTORCH

// Initialize
MAPPOAlgorithm* mappo = new MAPPOAlgorithm(obsSize, actionSize, numAgents, config);

// Train
for (episode) {
    obs = env->ResetTorch();  // [num_agents, obs_size]
    while (!allDone) {
        actions = mappo->SelectActions(obs, false);
        globalState = obs.reshape({-1});
        env->StepTorch(actions, nextObs, rewards, dones, truncated);
        mappo->StoreTransition(obs, globalState, actions, rewards, dones);
        obs = nextObs;
    }
    mappo->Update();
}

#endif
```

## Deployment - Inference

```cpp
#if WITH_LIBTORCH

// Load trained model
PPOAlgorithm* policy = new PPOAlgorithm(obsSize, actionSize, config);
policy->LoadModel("models/ppo_final");

// In game loop
void Tick(float DeltaTime) {
    TArray<float> obsArray = AgentComponent->CollectObservations();
    torch::Tensor obs = ConvertToTensor(obsArray);

    // Get action (deterministic)
    torch::Tensor action = policy->SelectAction(obs, true);

    // Execute
    AgentComponent->ExecuteAction(action);
}

#endif
```

## Configuration Presets

### Fast Prototyping
```cpp
config.HiddenSizes = {128, 128};
config.BatchSize = 32;
config.BufferSize = 1024;
config.Device = torch::kCPU;
```

### Balanced Training
```cpp
config.HiddenSizes = {256, 256};
config.BatchSize = 64;
config.BufferSize = 2048;
config.Device = torch::kCUDA;
```

### High Quality
```cpp
config.HiddenSizes = {512, 512, 256};
config.BatchSize = 128;
config.BufferSize = 4096;
config.Device = torch::kCUDA;
config.NumEpochs = 15;
```

## Common Parameters

| Parameter | Fast | Balanced | Quality |
|-----------|------|----------|---------|
| Hidden Sizes | 128,128 | 256,256 | 512,512,256 |
| Batch Size | 32 | 64 | 128 |
| Buffer Size | 1024 | 2048 | 4096 |
| Learning Rate | 5e-4 | 3e-4 | 1e-4 |
| Num Epochs | 5 | 10 | 15 |

## Tensor Conversions

### UE to LibTorch
```cpp
TArray<float> ueArray = {1.0f, 2.0f, 3.0f};
std::vector<float> vec(ueArray.GetData(), ueArray.GetData() + ueArray.Num());
torch::Tensor tensor = torch::from_blob(vec.data(), {3}, torch::kFloat32).clone();
```

### LibTorch to UE
```cpp
torch::Tensor tensor = torch::randn({3});
tensor = tensor.to(torch::kCPU).contiguous();
float* data = tensor.data_ptr<float>();
TArray<float> ueArray(data, 3);
```

## Device Management

```cpp
// Check CUDA
if (torch::cuda::is_available()) {
    config.Device = torch::kCUDA;
}

// Move tensors
tensor = tensor.to(config.Device);

// Move models
actor->to(config.Device);
critic->to(config.Device);
```

## Debugging

### Print Tensor Info
```cpp
UE_LOG(LogTemp, Log, TEXT("Shape: %s"), *FString(torch::str(tensor.sizes())));
UE_LOG(LogTemp, Log, TEXT("Device: %s"), tensor.device().is_cuda() ? TEXT("CUDA") : TEXT("CPU"));
UE_LOG(LogTemp, Log, TEXT("Dtype: %s"), *FString(torch::str(tensor.dtype())));
```

### Check for NaN
```cpp
if (torch::isnan(tensor).any().item<bool>()) {
    UE_LOG(LogTemp, Error, TEXT("NaN detected!"));
}
```

### Memory Usage
```cpp
if (torch::cuda::is_available()) {
    size_t allocated = torch::cuda::memory_allocated();
    UE_LOG(LogTemp, Log, TEXT("CUDA Memory: %.2f MB"), allocated / 1e6);
}
```

## Common Errors & Fixes

### "CUDA out of memory"
```cpp
config.BatchSize = 32;  // Reduce
config.BufferSize = 1024;  // Reduce
config.Device = torch::kCPU;  // Or switch to CPU
```

### "NaN in loss"
```cpp
config.LearningRate = 1e-4f;  // Lower LR
config.MaxGradNorm = 0.3f;  // Clip more
// Also: normalize observations
```

### "LibTorch not found"
```bash
export LIBTORCH_PATH=/path/to/libtorch
# Or copy to: Plugins/MARLPlugin/ThirdParty/LibTorch
```

## Blueprints Integration

### Call from Blueprint

```cpp
UFUNCTION(BlueprintCallable, Category="MARL")
void StartTraining() {
    #if WITH_LIBTORCH
    // Training code here
    #endif
}
```

### Expose Training Progress

```cpp
UPROPERTY(BlueprintReadOnly, Category="MARL")
float CurrentLoss = 0.0f;

UPROPERTY(BlueprintReadOnly, Category="MARL")
int32 EpisodeCount = 0;
```

## Performance Tips

1. **Use CUDA** for training (10-20x faster)
2. **Batch operations** when possible
3. **Store buffer on CPU**, train on GPU
4. **Pin memory** for faster CPU→GPU transfer:
   ```cpp
   tensor = tensor.pin_memory();
   ```
5. **Enable cuDNN benchmark**:
   ```cpp
   torch::backends::cudnn::benchmark::set_enabled(true);
   ```

## Useful Logging

```cpp
// Episode summary
UE_LOG(LogTemp, Log, TEXT("Ep %d: Reward=%.2f, Loss=%.4f, Entropy=%.4f"),
       episode, totalReward, policyLoss, entropy);

// Training progress
if (episode % 100 == 0) {
    UE_LOG(LogTemp, Log, TEXT("Progress: %d/%d episodes"), episode, totalEpisodes);
}

// Checkpoint saved
UE_LOG(LogTemp, Log, TEXT("Model saved to: %s"), *modelPath);
```

## Links

- **Full Documentation**: `README_PHASE4.md`
- **LibTorch Setup**: `LIBTORCH_SETUP.md`
- **Phase Summary**: `/PHASE4_SUMMARY.md`
- **Roadmap**: `/ROADMAP.md`

## Support

- LibTorch Docs: https://pytorch.org/cppdocs/
- UE Forums: https://forums.unrealengine.com/
- Issue Tracker: [Your repo issues URL]
