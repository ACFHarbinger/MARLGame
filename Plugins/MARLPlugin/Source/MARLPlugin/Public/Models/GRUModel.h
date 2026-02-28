#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * GRU-based neural network for sequential data
 * Similar to LSTM but with simpler gating mechanism (faster, less memory)
 */
struct GRUModelImpl : MARLModelBase {
    GRUModelImpl(
        int64_t input_dim,
        int64_t hidden_dim,
        int64_t output_dim,
        int64_t n_layers = 2,
        bool bidirectional = false,
        float dropout = 0.0f
    );

    torch::Tensor forward(torch::Tensor x) override;

    /**
     * Forward with external hidden state
     */
    std::tuple<torch::Tensor, torch::Tensor>
    ForwardWithState(torch::Tensor x, torch::Tensor hidden);

    /**
     * Initialize hidden state
     */
    torch::Tensor InitHidden(int64_t batch_size, torch::Device device);

    void ResetState() override;

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_dim; }
    const char* GetModelType() const override { return "GRU"; }

private:
    int64_t input_dim;
    int64_t hidden_dim;
    int64_t output_dim;
    int64_t n_layers;
    int64_t num_directions;

    torch::nn::GRU gru{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};

    // Stateful hidden state
    torch::Tensor h_state;
    bool use_stateful = false;
};

TORCH_MODULE(GRUModel);

#endif // WITH_LIBTORCH
