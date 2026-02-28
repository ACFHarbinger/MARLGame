#include "Models/LSTMModel.h"

#if WITH_LIBTORCH

LSTMModelImpl::LSTMModelImpl(
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
    // Initialize LSTM
    auto lstm_options = torch::nn::LSTMOptions(input_dim, hidden_dim)
        .num_layers(n_layers)
        .batch_first(true)
        .bidirectional(bidirectional);

    if (n_layers > 1 && dropout > 0.0f) {
        lstm_options.dropout(dropout);
    }

    lstm = register_module("lstm", torch::nn::LSTM(lstm_options));

    // Layer normalization
    layer_norm = register_module("layer_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim * num_directions})));

    // Output projection
    fc1 = register_module("fc1", torch::nn::Linear(hidden_dim * num_directions, output_dim));
}

torch::Tensor LSTMModelImpl::forward(torch::Tensor x) {
    auto device = x.device();
    auto batch_size = x.size(0);

    // Add sequence dimension if not present [batch, input] -> [batch, 1, input]
    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    // Initialize hidden and cell states
    auto h0 = torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                           torch::TensorOptions().device(device));
    auto c0 = torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                           torch::TensorOptions().device(device));

    // Use stateful hidden state if available
    if (use_stateful && h_state.defined() && c_state.defined()) {
        h0 = h_state;
        c0 = c_state;
    }

    // Forward pass through LSTM
    auto lstm_out = lstm->forward(x, std::make_tuple(h0, c0));
    auto out = std::get<0>(lstm_out);  // LSTM output
    auto hidden = std::get<1>(lstm_out);  // (h_n, c_n)

    // Store hidden state for stateful mode
    if (use_stateful) {
        h_state = std::get<0>(hidden).detach();
        c_state = std::get<1>(hidden).detach();
    }

    // Get last time step output: out[:, -1, :]
    auto last_time_step = out.select(1, -1);

    // Layer normalization
    last_time_step = layer_norm->forward(last_time_step);

    // Pass through output layer
    auto final_out = fc1->forward(last_time_step);

    return final_out;
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
LSTMModelImpl::ForwardWithState(torch::Tensor x, std::tuple<torch::Tensor, torch::Tensor> hidden) {
    // Add sequence dimension if not present
    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    // Forward pass with provided hidden state
    auto lstm_out = lstm->forward(x, hidden);
    auto out = std::get<0>(lstm_out);
    auto new_hidden = std::get<1>(lstm_out);

    // Get last time step
    auto last_time_step = out.select(1, -1);

    // Layer normalization
    last_time_step = layer_norm->forward(last_time_step);

    // Output projection
    auto final_out = fc1->forward(last_time_step);

    return std::make_tuple(final_out, new_hidden);
}

std::tuple<torch::Tensor, torch::Tensor> LSTMModelImpl::InitHidden(int64_t batch_size, torch::Device device) {
    auto h0 = torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                           torch::TensorOptions().device(device));
    auto c0 = torch::zeros({n_layers * num_directions, batch_size, hidden_dim},
                           torch::TensorOptions().device(device));
    return std::make_tuple(h0, c0);
}

void LSTMModelImpl::ResetState() {
    h_state = torch::Tensor();
    c_state = torch::Tensor();
}

#endif // WITH_LIBTORCH
