#include "Models/GRUModel.h"

#if WITH_LIBTORCH

GRUModelImpl::GRUModelImpl(
    int64_t input_dim,
    int64_t hidden_dim,
    int64_t output_dim,
    int64_t n_layers,
    bool bidirectional,
    float dropout
) : input_dim(input_dim),
    hidden_dim(hidden_dim),
    output_dim(output_dim),
    n_layers(n_layers),
    num_directions(bidirectional ? 2 : 1)
{
    // Initialize GRU
    auto gru_options = torch::nn::GRUOptions(input_dim, hidden_dim)
        .num_layers(n_layers)
        .batch_first(true)
        .bidirectional(bidirectional);

    if (n_layers > 1 && dropout > 0.0f) {
        gru_options.dropout(dropout);
    }

    gru = register_module("gru", torch::nn::GRU(gru_options));

    // Layer normalization
    layer_norm = register_module("layer_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim * num_directions})));

    // Output projection
    fc1 = register_module("fc1", torch::nn::Linear(hidden_dim * num_directions, output_dim));
}

torch::Tensor GRUModelImpl::forward(torch::Tensor x) {
    auto device = x.device();
    auto batch_size = x.size(0);

    // Add sequence dimension if not present
    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    // Initialize hidden state
    auto h0 = torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                           torch::TensorOptions().device(device));

    // Use stateful hidden if available
    if (use_stateful && h_state.defined()) {
        h0 = h_state;
    }

    // Forward pass through GRU
    auto gru_out = gru->forward(x, h0);
    auto out = std::get<0>(gru_out);  // GRU output
    auto h_n = std::get<1>(gru_out);  // Hidden state

    // Store for stateful mode
    if (use_stateful) {
        h_state = h_n.detach();
    }

    // Get last time step
    auto last_time_step = out.select(1, -1);

    // Layer normalization
    last_time_step = layer_norm->forward(last_time_step);

    // Output projection
    auto final_out = fc1->forward(last_time_step);

    return final_out;
}

std::tuple<torch::Tensor, torch::Tensor>
GRUModelImpl::ForwardWithState(torch::Tensor x, torch::Tensor hidden) {
    // Add sequence dimension if needed
    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    // Forward pass with provided hidden state
    auto gru_out = gru->forward(x, hidden);
    auto out = std::get<0>(gru_out);
    auto new_hidden = std::get<1>(gru_out);

    // Get last time step
    auto last_time_step = out.select(1, -1);

    // Layer normalization
    last_time_step = layer_norm->forward(last_time_step);

    // Output projection
    auto final_out = fc1->forward(last_time_step);

    return std::make_tuple(final_out, new_hidden);
}

torch::Tensor GRUModelImpl::InitHidden(int64_t batch_size, torch::Device device) {
    return torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                       torch::TensorOptions().device(device));
}

void GRUModelImpl::ResetState() {
    h_state = torch::Tensor();
}

#endif // WITH_LIBTORCH
