#include "Models/TransformerModel.h"

#if WITH_LIBTORCH

TransformerModelImpl::TransformerModelImpl(
    int64_t input_dim,
    int64_t embed_dim,
    int64_t output_dim,
    int64_t n_heads,
    int64_t n_layers,
    int64_t ff_dim,
    float dropout,
    int64_t max_seq_len
) : input_dim(input_dim),
    embed_dim(embed_dim),
    output_dim(output_dim),
    n_heads(n_heads),
    n_layers(n_layers),
    max_seq_len(max_seq_len)
{
    // Default feedforward dimension
    if (ff_dim == 0) {
        ff_dim = 4 * embed_dim;
    }

    // Input projection to embedding dimension
    input_projection = register_module("input_proj",
        torch::nn::Linear(input_dim, embed_dim));

    // Create positional encoding
    // PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
    auto pe = torch::zeros({max_seq_len, embed_dim});
    auto position = torch::arange(0, max_seq_len, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(
        torch::arange(0, embed_dim, 2, torch::kFloat32) * (-std::log(10000.0) / embed_dim)
    );

    pe.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)},
                  torch::sin(position * div_term));
    pe.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)},
                  torch::cos(position * div_term));

    positional_encoding = pe.unsqueeze(0);  // [1, max_seq_len, embed_dim]
    register_buffer("pos_encoding", positional_encoding);

    // Transformer encoder layer
    auto encoder_layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(embed_dim, n_heads)
            .dim_feedforward(ff_dim)
            .dropout(dropout)
            .activation(torch::kGELU)
    );

    // Stack encoder layers
    transformer_encoder = register_module("transformer_encoder",
        torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, n_layers)
        ));

    // Layer normalization
    layer_norm = register_module("layer_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));

    // Output projection
    output_projection = register_module("output_proj",
        torch::nn::Linear(embed_dim, output_dim));
}

torch::Tensor TransformerModelImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);

    // Add sequence dimension if not present [batch, input] -> [batch, 1, input]
    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    auto seq_len = x.size(1);

    // Input projection
    x = input_projection->forward(x);  // [batch, seq_len, embed_dim]

    // Add positional encoding
    auto pos_enc = positional_encoding.index({torch::indexing::Slice(), torch::indexing::Slice(0, seq_len), torch::indexing::Slice()});
    x = x + pos_enc.to(x.device());

    // Transformer expects [seq_len, batch, embed_dim]
    x = x.permute({1, 0, 2});

    // Pass through transformer encoder
    x = transformer_encoder->forward(x);

    // Back to [batch, seq_len, embed_dim]
    x = x.permute({1, 0, 2});

    // Use last time step or pool across sequence
    // Here we use mean pooling
    x = x.mean(1);  // [batch, embed_dim]

    // Layer normalization
    x = layer_norm->forward(x);

    // Output projection
    auto output = output_projection->forward(x);

    return output;
}

torch::Tensor TransformerModelImpl::ForwardWithMask(torch::Tensor x, torch::Tensor mask) {
    auto batch_size = x.size(0);

    if (x.dim() == 2) {
        x = x.unsqueeze(1);
    }

    auto seq_len = x.size(1);

    // Input projection
    x = input_projection->forward(x);

    // Add positional encoding
    auto pos_enc = positional_encoding.index({torch::indexing::Slice(), torch::indexing::Slice(0, seq_len), torch::indexing::Slice()});
    x = x + pos_enc.to(x.device());

    // Transformer expects [seq_len, batch, embed_dim]
    x = x.permute({1, 0, 2});

    // Pass through transformer with mask
    x = transformer_encoder->forward(x, mask);

    // Back to [batch, seq_len, embed_dim]
    x = x.permute({1, 0, 2});

    // Mean pooling
    x = x.mean(1);

    // Layer normalization
    x = layer_norm->forward(x);

    // Output projection
    auto output = output_projection->forward(x);

    return output;
}

torch::Tensor TransformerModelImpl::GetAttentionWeights() {
    return last_attention_weights;
}

#endif // WITH_LIBTORCH
