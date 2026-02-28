#include "Models/ResNetModel.h"

#if WITH_LIBTORCH

// ============================================================================
// Residual Block Implementation
// ============================================================================

ResidualBlock::ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t stride, bool use_1x1conv)
    : use_1x1conv(use_1x1conv)
{
    // First convolution
    conv1 = register_module("conv1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                .stride(stride)
                .padding(1)
                .bias(false)
        ));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

    // Second convolution
    conv2 = register_module("conv2",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                .stride(1)
                .padding(1)
                .bias(false)
        ));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

    // 1x1 convolution for dimension matching (if needed)
    if (use_1x1conv) {
        conv3 = register_module("conv3",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                    .stride(stride)
                    .bias(false)
            ));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));
    }
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    // First conv block
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);

    // Second conv block
    out = conv2->forward(out);
    out = bn2->forward(out);

    // Adjust identity if dimensions don't match
    if (use_1x1conv) {
        identity = conv3->forward(x);
        identity = bn3->forward(identity);
    }

    // Residual connection
    out = out + identity;
    out = torch::relu(out);

    return out;
}

// ============================================================================
// ResNet Model Implementation
// ============================================================================

ResNetModelImpl::ResNetModelImpl(
    int64_t input_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_dim,
    std::vector<int64_t> num_blocks,
    std::vector<int64_t> channels,
    std::vector<int64_t> fc_hidden_sizes
) : input_channels(input_channels),
    input_height(input_height),
    input_width(input_width),
    output_dim(output_dim)
{
    // Initial convolution layer
    conv1 = register_module("conv1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(input_channels, 64, 7)
                .stride(2)
                .padding(3)
                .bias(false)
        ));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

    // Max pooling
    max_pool = register_module("max_pool",
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

    // Build residual layers
    int64_t in_channels = 64;
    for (size_t i = 0; i < num_blocks.size() && i < channels.size(); ++i) {
        int64_t out_channels = channels[i];
        int64_t stride = (i == 0) ? 1 : 2;  // First layer no stride, others downsample

        auto layer = MakeLayer(in_channels, out_channels, num_blocks[i], stride, i);
        layers.push_back(layer);

        in_channels = out_channels;
    }

    // Global average pooling
    global_pool = register_module("global_pool",
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

    // Fully connected layers
    int64_t fc_in_size = channels.back();

    for (size_t i = 0; i < fc_hidden_sizes.size(); ++i) {
        auto fc = register_module("fc" + std::to_string(i),
            torch::nn::Linear(fc_in_size, fc_hidden_sizes[i]));
        fc_layers.push_back(fc);
        fc_in_size = fc_hidden_sizes[i];
    }

    // Final output layer
    auto fc_out = register_module("fc_out",
        torch::nn::Linear(fc_in_size, output_dim));
    fc_layers.push_back(fc_out);
}

torch::Tensor ResNetModelImpl::forward(torch::Tensor x) {
    // Ensure correct shape [batch, channels, height, width]
    if (x.dim() == 3) {
        x = x.unsqueeze(0);
    }

    // Initial convolution
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = max_pool->forward(x);

    // Pass through residual layers
    for (auto& layer : layers) {
        x = layer->forward(x);
    }

    // Global average pooling
    x = global_pool->forward(x);

    // Flatten
    x = x.view({x.size(0), -1});

    // Fully connected layers
    for (size_t i = 0; i < fc_layers.size() - 1; ++i) {
        x = fc_layers[i]->forward(x);
        x = torch::relu(x);
    }

    // Final layer (no activation)
    x = fc_layers.back()->forward(x);

    return x;
}

torch::nn::Sequential ResNetModelImpl::MakeLayer(int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t stride, int layer_idx) {
    torch::nn::Sequential layer;

    // First block (may downsample)
    bool use_1x1conv = (stride != 1 || in_channels != out_channels);
    auto block0 = std::make_shared<ResidualBlock>(in_channels, out_channels, stride, use_1x1conv);
    layer->push_back(register_module("layer" + std::to_string(layer_idx) + "_block0", block0));

    // Remaining blocks
    for (int64_t i = 1; i < num_blocks; ++i) {
        auto block = std::make_shared<ResidualBlock>(out_channels, out_channels, 1, false);
        layer->push_back(register_module("layer" + std::to_string(layer_idx) + "_block" + std::to_string(i), block));
    }

    return layer;
}

#endif // WITH_LIBTORCH
