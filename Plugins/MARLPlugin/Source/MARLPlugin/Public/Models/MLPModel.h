#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Multi-Layer Perceptron (MLP) neural network
 * Simple feedforward network for processing non-sequential data
 * Supports optional layer normalization for training stability
 */
struct MLPModelImpl : MARLModelBase {
    MLPModelImpl(
        int64_t input_dim,
        int64_t output_dim,
        std::vector<int64_t> hidden_sizes = {256, 256},
        bool use_layer_norm = false,
        float dropout = 0.0f
    );

    torch::Tensor forward(torch::Tensor x) override;

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_dim; }
    const char* GetModelType() const override { return "MLP"; }

private:
    int64_t input_dim;
    int64_t output_dim;
    bool use_layer_norm;
    float dropout_rate;

    std::vector<torch::nn::Linear> layers;
    std::vector<torch::nn::LayerNorm> layer_norms;
    torch::nn::Dropout dropout{nullptr};
};

TORCH_MODULE(MLPModel);

#endif // WITH_LIBTORCH
