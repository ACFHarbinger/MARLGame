#include "Models/CNNModel.h"

#if WITH_LIBTORCH

CNNModelImpl::CNNModelImpl(
    int64_t input_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_dim,
    std::vector<int64_t> conv_channels,
    std::vector<int64_t> kernel_sizes,
    std::vector<int64_t> strides,
    std::vector<int64_t> fc_hidden_sizes
) : input_channels(input_channels),
    input_height(input_height),
    input_width(input_width),
    output_dim(output_dim)
{
    // Build convolutional layers
    int64_t in_channels = input_channels;
    int64_t current_height = input_height;
    int64_t current_width = input_width;

    for (size_t i = 0; i < conv_channels.size(); ++i) {
        int64_t out_channels = conv_channels[i];
        int64_t kernel_size = (i < kernel_sizes.size()) ? kernel_sizes[i] : 3;
        int64_t stride = (i < strides.size()) ? strides[i] : 1;

        // Convolutional layer
        auto conv = register_module("conv" + std::to_string(i),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                    .stride(stride)
                    .padding(0)
            ));
        conv_layers.push_back(conv);

        // Batch normalization
        auto bn = register_module("bn" + std::to_string(i),
            torch::nn::BatchNorm2d(out_channels));
        batch_norms.push_back(bn);

        // Update dimensions
        current_height = CalculateConvOutputSize(current_height, kernel_size, stride);
        current_width = CalculateConvOutputSize(current_width, kernel_size, stride);

        in_channels = out_channels;
    }

    // Calculate flattened size
    flattened_size = in_channels * current_height * current_width;

    // Build fully connected layers
    int64_t fc_in_size = flattened_size;

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

torch::Tensor CNNModelImpl::forward(torch::Tensor x) {
    // Ensure input has correct shape [batch, channels, height, width]
    if (x.dim() == 3) {
        x = x.unsqueeze(0);  // Add batch dimension
    }

    // Pass through convolutional layers
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        x = conv_layers[i]->forward(x);
        x = batch_norms[i]->forward(x);
        x = torch::relu(x);
    }

    // Flatten
    x = x.view({x.size(0), -1});

    // Pass through fully connected layers
    for (size_t i = 0; i < fc_layers.size() - 1; ++i) {
        x = fc_layers[i]->forward(x);
        x = torch::relu(x);
    }

    // Final layer (no activation)
    x = fc_layers.back()->forward(x);

    return x;
}

std::vector<torch::Tensor> CNNModelImpl::GetFeatureMaps(torch::Tensor x) {
    std::vector<torch::Tensor> feature_maps;

    if (x.dim() == 3) {
        x = x.unsqueeze(0);
    }

    // Collect feature maps from each conv layer
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        x = conv_layers[i]->forward(x);
        x = batch_norms[i]->forward(x);
        x = torch::relu(x);
        feature_maps.push_back(x.clone());
    }

    return feature_maps;
}

int64_t CNNModelImpl::CalculateConvOutputSize(int64_t input_size, int64_t kernel_size, int64_t stride, int64_t padding) {
    return ((input_size + 2 * padding - kernel_size) / stride) + 1;
}

#endif // WITH_LIBTORCH
