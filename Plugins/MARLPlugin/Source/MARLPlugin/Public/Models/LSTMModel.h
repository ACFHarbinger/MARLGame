#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * LSTM-based neural network for sequential data
 * Useful for temporal credit assignment and memory-based policies
 */
struct LSTMModelImpl : MARLModelBase {
    LSTMModelImpl(
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
     * @param x Input tensor [batch_size, seq_len, input_dim]
     * @param hidden Tuple of (h0, c0)
     * @return Tuple of (output, (h_n, c_n))
     */
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
    ForwardWithState(torch::Tensor x, std::tuple<torch::Tensor, torch::Tensor> hidden);

    /**
     * Initialize hidden state
     * @param batch_size Batch size
     * @param device Device to create tensors on
     * @return Tuple of (h0, c0)
     */
    std::tuple<torch::Tensor, torch::Tensor> InitHidden(int64_t batch_size, torch::Device device);

    void ResetState() override;

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_dim; }
    const char* GetModelType() const override { return "LSTM"; }

private:
    int64_t input_dim;
    int64_t hidden_dim;
    int64_t output_dim;
    int64_t n_layers;
    int64_t num_directions;

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};

    // Stateful hidden state (optional)
    torch::Tensor h_state;
    torch::Tensor c_state;
    bool use_stateful = false;
};

TORCH_MODULE(LSTMModel);

#endif // WITH_LIBTORCH
