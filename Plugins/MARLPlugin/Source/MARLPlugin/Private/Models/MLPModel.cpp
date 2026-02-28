#include "Models/MLPModel.h"

#if WITH_LIBTORCH

MLPModelImpl::MLPModelImpl(
    int64_t input_dim_,
    int64_t output_dim_,
    std::vector<int64_t> hidden_sizes,
    bool use_layer_norm_,
    float dropout_rate_
)
    : input_dim(input_dim_)
    , output_dim(output_dim_)
    , use_layer_norm(use_layer_norm_)
    , dropout_rate(dropout_rate_)
{
    int64_t prev_size = input_dim;

    // Create hidden layers
    for (size_t i = 0; i < hidden_sizes.size(); i++)
    {
        auto linear = register_module(
            "fc" + std::to_string(i),
            torch::nn::Linear(prev_size, hidden_sizes[i])
        );
        layers.push_back(linear);

        if (use_layer_norm)
        {
            auto layer_norm = register_module(
                "ln" + std::to_string(i),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[i]}))
            );
            layer_norms.push_back(layer_norm);
        }

        prev_size = hidden_sizes[i];
    }

    // Create output layer
    auto output_layer = register_module(
        "fc_out",
        torch::nn::Linear(prev_size, output_dim)
    );
    layers.push_back(output_layer);

    // Create dropout if needed
    if (dropout_rate > 0.0f)
    {
        dropout = register_module(
            "dropout",
            torch::nn::Dropout(dropout_rate)
        );
    }
}

torch::Tensor MLPModelImpl::forward(torch::Tensor x)
{
    // Process through all hidden layers
    for (size_t i = 0; i < layers.size() - 1; i++)
    {
        x = layers[i]->forward(x);

        // Apply layer normalization if enabled
        if (use_layer_norm && i < layer_norms.size())
        {
            x = layer_norms[i]->forward(x);
        }

        // Apply ReLU activation
        x = torch::relu(x);

        // Apply dropout if enabled (only during training)
        if (dropout && dropout_rate > 0.0f && is_training())
        {
            x = dropout->forward(x);
        }
    }

    // Output layer (no activation)
    x = layers.back()->forward(x);

    return x;
}

#endif // WITH_LIBTORCH
