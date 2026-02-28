#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Residual Block for ResNet
 */
struct ResidualBlock : torch::nn::Module {
    ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1, bool use_1x1conv = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};  // 1x1 conv for dimension matching
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::BatchNorm2d bn3{nullptr};
    bool use_1x1conv;
};

/**
 * ResNet-based model for deep visual feature extraction
 * Uses residual connections for better gradient flow
 */
struct ResNetModelImpl : MARLModelBase {
    ResNetModelImpl(
        int64_t input_channels,
        int64_t input_height,
        int64_t input_width,
        int64_t output_dim,
        std::vector<int64_t> num_blocks = {2, 2, 2, 2},  // ResNet-18 style
        std::vector<int64_t> channels = {64, 128, 256, 512},
        std::vector<int64_t> fc_hidden_sizes = {512}
    );

    torch::Tensor forward(torch::Tensor x) override;

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_channels * input_height * input_width; }
    const char* GetModelType() const override { return "ResNet"; }

private:
    int64_t input_channels;
    int64_t input_height;
    int64_t input_width;
    int64_t output_dim;

    // Initial convolution
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::MaxPool2d max_pool{nullptr};

    // Residual blocks
    std::vector<torch::nn::Sequential> layers;

    // Global average pooling
    torch::nn::AdaptiveAvgPool2d global_pool{nullptr};

    // Fully connected layers
    std::vector<torch::nn::Linear> fc_layers;

    /**
     * Build a layer of residual blocks
     */
    torch::nn::Sequential MakeLayer(int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t stride, int layer_idx);
};

TORCH_MODULE(ResNetModel);

#endif // WITH_LIBTORCH
