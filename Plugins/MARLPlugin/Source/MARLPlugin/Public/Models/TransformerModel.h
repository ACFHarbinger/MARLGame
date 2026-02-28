#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Transformer-based neural network with self-attention
 * Excellent for modeling agent interactions and relational reasoning
 */
struct TransformerModelImpl : MARLModelBase {
    TransformerModelImpl(
        int64_t input_dim,
        int64_t embed_dim,
        int64_t output_dim,
        int64_t n_heads = 8,
        int64_t n_layers = 4,
        int64_t ff_dim = 0,  // If 0, defaults to 4 * embed_dim
        float dropout = 0.1f,
        int64_t max_seq_len = 100
    );

    torch::Tensor forward(torch::Tensor x) override;

    /**
     * Forward with attention mask
     * @param x Input tensor [batch, seq_len, input_dim]
     * @param mask Attention mask [seq_len, seq_len]
     */
    torch::Tensor ForwardWithMask(torch::Tensor x, torch::Tensor mask);

    /**
     * Get attention weights from last forward pass (for visualization)
     */
    torch::Tensor GetAttentionWeights();

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_dim; }
    const char* GetModelType() const override { return "Transformer"; }

private:
    int64_t input_dim;
    int64_t embed_dim;
    int64_t output_dim;
    int64_t n_heads;
    int64_t n_layers;
    int64_t max_seq_len;

    // Input embedding
    torch::nn::Linear input_projection{nullptr};

    // Positional encoding
    torch::Tensor positional_encoding;

    // Transformer encoder
    torch::nn::TransformerEncoder transformer_encoder{nullptr};

    // Output projection
    torch::nn::Linear output_projection{nullptr};

    // Layer normalization
    torch::nn::LayerNorm layer_norm{nullptr};

    // Store attention weights for visualization
    torch::Tensor last_attention_weights;
};

TORCH_MODULE(TransformerModel);

#endif // WITH_LIBTORCH
