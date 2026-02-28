# MARL Neural Network Models

This directory contains various neural network architectures that can be used as actor or critic networks in MARL training.

## Available Models

### 1. MLP (Multi-Layer Perceptron)
**File:** `MARLNeuralNetwork.h`

**Use Case:** Vector-based observations (positions, velocities, game state)

**Architecture:**
```
Input → Linear → LayerNorm → ReLU → ... → Linear → Output
```

**Example:**
```cpp
auto mlp = FMARLModelFactory::CreateMLPModel(128, 64, {256, 256});
```

**Pros:**
- Simple and fast
- Works well with low-dimensional inputs
- Easy to train

**Cons:**
- No temporal or spatial structure
- Limited expressiveness

---

### 2. LSTM (Long Short-Term Memory)
**Files:** `Models/LSTMModel.h/cpp`

**Use Case:** Sequential/temporal observations (agent trajectories, time series)

**Architecture:**
```
Input → LSTM(layers) → LayerNorm → Linear → Output
```

**Example:**
```cpp
#if WITH_LIBTORCH

auto lstm = FMARLModelFactory::CreateLSTMModel(
    128,    // input_dim
    256,    // hidden_dim
    64,     // output_dim
    2,      // n_layers
    false,  // bidirectional
    0.0f    // dropout
);

// Forward pass
torch::Tensor obs = torch::randn({batch_size, seq_len, 128});
torch::Tensor output = lstm->forward(obs);

// With external hidden state
auto [h0, c0] = lstm->InitHidden(batch_size, torch::kCUDA);
auto [output, new_hidden] = lstm->ForwardWithState(obs, std::make_tuple(h0, c0));

#endif
```

**Pros:**
- Handles temporal dependencies
- Good for credit assignment over time
- Remembers past observations

**Cons:**
- Slower than feedforward networks
- Requires more memory
- Can suffer from vanishing gradients

---

### 3. GRU (Gated Recurrent Unit)
**Files:** `Models/GRUModel.h/cpp`

**Use Case:** Similar to LSTM but faster (temporal data with less memory requirements)

**Architecture:**
```
Input → GRU(layers) → LayerNorm → Linear → Output
```

**Example:**
```cpp
#if WITH_LIBTORCH

auto gru = FMARLModelFactory::CreateGRUModel(
    128,    // input_dim
    256,    // hidden_dim
    64,     // output_dim
    2,      // n_layers
    false,  // bidirectional
    0.0f    // dropout
);

torch::Tensor output = gru->forward(obs);

#endif
```

**Pros:**
- Faster than LSTM
- Less memory than LSTM
- Similar performance to LSTM

**Cons:**
- Slightly less expressive than LSTM
- Still slower than feedforward

---

### 4. Transformer
**Files:** `Models/TransformerModel.h/cpp`

**Use Case:** Multi-agent relational reasoning, attention-based policies

**Architecture:**
```
Input → Embedding → Positional Encoding →
TransformerEncoder(layers) → MeanPooling →
LayerNorm → Linear → Output
```

**Example:**
```cpp
#if WITH_LIBTORCH

auto transformer = FMARLModelFactory::CreateTransformerModel(
    128,   // input_dim
    256,   // embed_dim
    64,    // output_dim
    8,     // n_heads
    4,     // n_layers
    1024,  // ff_dim
    0.1f,  // dropout
    100    // max_seq_len
);

// Forward pass
torch::Tensor obs = torch::randn({batch_size, seq_len, 128});
torch::Tensor output = transformer->forward(obs);

// With attention mask
torch::Tensor mask = torch::ones({seq_len, seq_len});
output = transformer->ForwardWithMask(obs, mask);

#endif
```

**Pros:**
- Excellent for modeling relationships
- Parallelizable (unlike RNNs)
- State-of-the-art for many tasks
- Can visualize attention weights

**Cons:**
- Computationally expensive
- Requires more data to train
- Memory intensive

---

### 5. CNN (Convolutional Neural Network)
**Files:** `Models/CNNModel.h/cpp`

**Use Case:** Visual observations (top-down game view, camera images)

**Architecture:**
```
Input[H,W,C] → Conv2d → BatchNorm → ReLU → ... →
Flatten → Linear → ReLU → ... → Output
```

**Example:**
```cpp
#if WITH_LIBTORCH

auto cnn = FMARLModelFactory::CreateCNNModel(
    3,              // input_channels (RGB)
    84,             // input_height
    84,             // input_width
    512,            // output_dim
    {32, 64, 128},  // conv_channels
    {8, 4, 3},      // kernel_sizes
    {4, 2, 1},      // strides
    {512}           // fc_hidden_sizes
);

// Forward pass
torch::Tensor image = torch::randn({batch_size, 3, 84, 84});
torch::Tensor output = cnn->forward(image);

// Get feature maps for visualization
auto feature_maps = cnn->GetFeatureMaps(image);

#endif
```

**Pros:**
- Excellent for visual input
- Spatial invariance
- Hierarchical features

**Cons:**
- Only for image-like inputs
- Many hyperparameters
- Requires more compute

**Recommended Settings:**
```cpp
// Atari-style (84x84 grayscale)
CreateCNNModel(1, 84, 84, 512, {32, 64, 64}, {8, 4, 3}, {4, 2, 1});

// High-res (128x128 RGB)
CreateCNNModel(3, 128, 128, 512, {32, 64, 128, 256}, {5, 3, 3, 3}, {2, 2, 2, 1});
```

---

### 6. ResNet (Residual Network)
**Files:** `Models/ResNetModel.h/cpp`

**Use Case:** Deep visual feature extraction with residual connections

**Architecture:**
```
Input[H,W,C] → Conv7x7 → BatchNorm → MaxPool →
ResidualBlocks(layers) → GlobalAvgPool →
Linear → ReLU → ... → Output
```

**Example:**
```cpp
#if WITH_LIBTORCH

auto resnet = FMARLModelFactory::CreateResNetModel(
    3,                      // input_channels
    84,                     // input_height
    84,                     // input_width
    512,                    // output_dim
    {2, 2, 2, 2},          // num_blocks (ResNet-18)
    {64, 128, 256, 512},   // channels
    {512}                   // fc_hidden_sizes
);

torch::Tensor image = torch::randn({batch_size, 3, 84, 84});
torch::Tensor output = resnet->forward(image);

#endif
```

**Pros:**
- Very deep networks possible
- Better gradient flow than CNN
- Stronger feature extraction

**Cons:**
- More parameters than CNN
- Slower than basic CNN
- Overkill for simple tasks

**Variants:**
```cpp
// ResNet-18
{2, 2, 2, 2}

// ResNet-34
{3, 4, 6, 3}

// ResNet-50 (bottleneck blocks)
{3, 4, 6, 3}
```

---

## Model Selection Guide

### By Observation Type

| Observation Type | Recommended Model | Alternative |
|------------------|-------------------|-------------|
| Vector (positions, velocities) | MLP | - |
| Sequences (trajectories) | LSTM, GRU | Transformer |
| Images (camera, top-down) | CNN | ResNet |
| Multi-agent relations | Transformer | MLP |
| Mixed (vector + image) | CNN + MLP | ResNet + MLP |

### By Task Complexity

| Complexity | Model | Hidden Size | Layers |
|------------|-------|-------------|--------|
| Simple | MLP | 128-256 | 2-3 |
| Medium | LSTM/GRU | 256-512 | 2-4 |
| Complex | Transformer/ResNet | 512-1024 | 4-8 |

### By Computational Budget

| Budget | Training | Inference | Recommended Model |
|--------|----------|-----------|-------------------|
| Low | CPU only | <10ms | MLP (128x2) |
| Medium | GPU | <20ms | LSTM/GRU (256x2) |
| High | Multi-GPU | <50ms | Transformer/ResNet |

---

## Usage with PPO/MAPPO

### Replacing Actor Network

```cpp
#if WITH_LIBTORCH

// Instead of using default MLP in ActorNetwork...
// Create custom model
auto backbone = FMARLModelFactory::CreateLSTMModel(obs_size, 256, 256, 2);

// Use in custom actor
struct CustomActor : torch::nn::Module {
    CustomActor(std::shared_ptr<MARLModelBase> backbone, int64_t action_size) {
        this->backbone = register_module("backbone", backbone);
        this->action_head = register_module("action_head",
            torch::nn::Linear(backbone->GetOutputDim(), action_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto features = backbone->forward(x);
        return action_head->forward(features);
    }

    std::shared_ptr<MARLModelBase> backbone;
    torch::nn::Linear action_head{nullptr};
};

#endif
```

### Hybrid Models (CNN + RNN)

```cpp
#if WITH_LIBTORCH

// Visual features from CNN
auto cnn = FMARLModelFactory::CreateCNNModel(3, 84, 84, 512);

// Temporal processing with LSTM
auto lstm = FMARLModelFactory::CreateLSTMModel(512, 256, action_size, 2);

// Forward pass
auto visual_features = cnn->forward(image);  // [batch, 512]
auto visual_seq = visual_features.unsqueeze(1);  // [batch, 1, 512]
auto action_logits = lstm->forward(visual_seq);  // [batch, action_size]

#endif
```

---

## Model Factory

### Quick Creation

```cpp
#if WITH_LIBTORCH

// Using factory with config
FMARLModelConfig config;
config.ModelType = EMARLModelType::LSTM;
config.InputDim = 128;
config.OutputDim = 64;
config.HiddenSizes = {256};
config.NumLayers = 2;

auto model = FMARLModelFactory::CreateModel(config);

#endif
```

### Auto-select by Observation Type

```cpp
#if WITH_LIBTORCH

FString obs_type = TEXT("visual");  // or "sequence", "vector", "relational"
EMARLModelType model_type = FMARLModelFactory::GetRecommendedModelType(obs_type);

FMARLModelConfig config = FMARLModelFactory::GetDefaultConfig(model_type);
config.InputDim = 128;
config.OutputDim = 64;

auto model = FMARLModelFactory::CreateModel(config);

#endif
```

---

## Performance Tips

### 1. Start Simple
```cpp
// Begin with MLP
auto model = FMARLModelFactory::CreateMLPModel(obs_size, action_size, {256, 256});

// If performance inadequate, try LSTM
auto model = FMARLModelFactory::CreateLSTMModel(obs_size, 256, action_size, 2);
```

### 2. Batch Normalization vs Layer Normalization
- **BatchNorm:** Better for CNNs, requires larger batches
- **LayerNorm:** Better for RNNs/Transformers, works with small batches

### 3. Gradient Clipping
```cpp
// In PPO/MAPPO
torch::nn::utils::clip_grad_norm_(model->parameters(), max_grad_norm);
```

### 4. Learning Rate by Model Type
```cpp
// MLP: 3e-4
// LSTM/GRU: 1e-4 (slower learning)
// Transformer: 1e-4 with warmup
// CNN/ResNet: 3e-4 with decay
```

---

## Debugging Models

### Check Output Shape
```cpp
torch::Tensor input = torch::randn({batch_size, input_dim});
torch::Tensor output = model->forward(input);

UE_LOG(LogTemp, Log, TEXT("Output shape: %s"),
    *FString(torch::str(output.sizes())));
```

### Visualize Parameters
```cpp
int64_t total_params = 0;
for (const auto& param : model->parameters()) {
    total_params += param.numel();
}
UE_LOG(LogTemp, Log, TEXT("Total parameters: %lld"), total_params);
```

### Test Forward Pass
```cpp
// Ensure no NaN
auto output = model->forward(input);
if (torch::isnan(output).any().item<bool>()) {
    UE_LOG(LogTemp, Error, TEXT("NaN detected in output!"));
}
```

---

## Examples

### Example 1: Twin Stick with LSTM Memory
```cpp
// Agents remember past positions
auto actor_lstm = FMARLModelFactory::CreateLSTMModel(128, 256, 24, 2);
auto critic_mlp = FMARLModelFactory::CreateMLPModel(128, 1, {256, 256});
```

### Example 2: Strategy Game with Transformer
```cpp
// Model agent interactions
auto transformer = FMARLModelFactory::CreateTransformerModel(
    128, 256, 64, 8, 4, 1024, 0.1f, 100
);
```

### Example 3: Visual Observations with CNN
```cpp
// Top-down camera view
auto cnn = FMARLModelFactory::CreateCNNModel(
    3, 84, 84, 512,
    {32, 64, 128},
    {8, 4, 3},
    {4, 2, 1}
);
```

---

## References

- **LSTM:** Hochreiter & Schmidhuber (1997)
- **GRU:** Cho et al. (2014)
- **Transformer:** Vaswani et al. (2017) "Attention is All You Need"
- **ResNet:** He et al. (2015) "Deep Residual Learning"
- **CNNs for RL:** Mnih et al. (2015) "Human-level control through deep RL"

---

## Next Steps

1. Experiment with different models for your task
2. Use TensorBoard to visualize training curves
3. Compare model performance on held-out test episodes
4. Consider ensemble methods (multiple models voting)

For integration with PPO/MAPPO, see `README_PHASE4.md`.
