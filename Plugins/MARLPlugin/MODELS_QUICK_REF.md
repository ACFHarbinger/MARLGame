# Neural Network Models - Quick Reference

## Model Selection

```cpp
Vector obs     → MLP
Temporal data  → LSTM or GRU
Visual input   → CNN or ResNet
Multi-agent    → Transformer
```

## Quick Creation

```cpp
#if WITH_LIBTORCH

// MLP
auto mlp = FMARLModelFactory::CreateMLPModel(128, 64, {256, 256});

// LSTM
auto lstm = FMARLModelFactory::CreateLSTMModel(128, 256, 64, 2);

// GRU
auto gru = FMARLModelFactory::CreateGRUModel(128, 256, 64, 2);

// Transformer
auto transformer = FMARLModelFactory::CreateTransformerModel(128, 256, 64, 8, 4);

// CNN
auto cnn = FMARLModelFactory::CreateCNNModel(3, 84, 84, 512);

// ResNet
auto resnet = FMARLModelFactory::CreateResNetModel(3, 84, 84, 512);

#endif
```

## Using with PPO

```cpp
#if WITH_LIBTORCH

// Replace actor backbone
auto backbone = FMARLModelFactory::CreateLSTMModel(obs_size, 256, 256, 2);

// In actor network
struct CustomActor : torch::nn::Module {
    std::shared_ptr<MARLModelBase> backbone;
    torch::nn::Linear action_head{nullptr};

    CustomActor(int64_t obs_size, int64_t action_size) {
        backbone = register_module("backbone",
            FMARLModelFactory::CreateLSTMModel(obs_size, 256, 256, 2));
        action_head = register_module("action_head",
            torch::nn::Linear(256, action_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        return action_head->forward(backbone->forward(x));
    }
};

#endif
```

## Model Sizes

| Model | Params | Speed | Memory |
|-------|--------|-------|--------|
| MLP | 130K | ⚡⚡⚡ | 🔵 |
| LSTM | 520K | ⚡⚡ | 🔵🔵 |
| GRU | 390K | ⚡⚡ | 🔵🔵 |
| Transformer | 2.1M | ⚡ | 🔵🔵🔵 |
| CNN | 1.2M | ⚡⚡ | 🔵🔵 |
| ResNet | 11M | ⚡ | 🔵🔵🔵 |

## Debugging

```cpp
// Check output shape
auto output = model->forward(input);
UE_LOG(LogTemp, Log, TEXT("Shape: %s"), *FString(torch::str(output.sizes())));

// Check for NaN
if (torch::isnan(output).any().item<bool>()) {
    UE_LOG(LogTemp, Error, TEXT("NaN detected!"));
}

// Count parameters
int64_t params = 0;
for (auto& p : model->parameters()) params += p.numel();
UE_LOG(LogTemp, Log, TEXT("Parameters: %lld"), params);
```

## Common Configurations

### Twin Stick (Temporal)
```cpp
auto actor = CreateLSTMModel(128, 256, 24, 2, false, 0.0f);
auto critic = CreateLSTMModel(128, 256, 1, 2, false, 0.0f);
```

### Strategy (Relational)
```cpp
auto actor = CreateTransformerModel(128, 256, 64, 8, 4, 1024, 0.1f, 100);
auto critic = CreateMLPModel(128, 1, {512, 512});
```

### Visual (CNN)
```cpp
auto actor = CreateCNNModel(3, 84, 84, 512, {32,64,128}, {8,4,3}, {4,2,1});
auto critic = CreateCNNModel(3, 84, 84, 512, {32,64,128}, {8,4,3}, {4,2,1});
```

## Learning Rates

```cpp
MLP:         3e-4
LSTM/GRU:    1e-4
Transformer: 1e-4 (with warmup)
CNN/ResNet:  3e-4 (with decay)
```

## Links

- Full docs: `MODELS_README.md`
- Summary: `MODELS_SUMMARY.md`
- Phase 4: `README_PHASE4.md`
