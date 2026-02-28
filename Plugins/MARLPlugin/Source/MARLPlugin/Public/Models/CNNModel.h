#pragma once

#include "CoreMinimal.h"
#include "Models/MARLModelBase.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Convolutional Neural Network for visual observations
 * Processes image/grid-based inputs (e.g., top-down game views)
 */
struct CNNModelImpl : MARLModelBase {
    CNNModelImpl(
        int64_t input_channels,
        int64_t input_height,
        int64_t input_width,
        int64_t output_dim,
        std::vector<int64_t> conv_channels = {32, 64, 128},
        std::vector<int64_t> kernel_sizes = {8, 4, 3},
        std::vector<int64_t> strides = {4, 2, 1},
        std::vector<int64_t> fc_hidden_sizes = {512}
    );

    torch::Tensor forward(torch::Tensor x) override;

    /**
     * Get feature maps from convolutional layers (for visualization)
     */
    std::vector<torch::Tensor> GetFeatureMaps(torch::Tensor x);

    int64_t GetOutputDim() const override { return output_dim; }
    int64_t GetInputDim() const override { return input_channels * input_height * input_width; }
    const char* GetModelType() const override { return "CNN"; }

private:
    int64_t input_channels;
    int64_t input_height;
    int64_t input_width;
    int64_t output_dim;
    int64_t flattened_size;

    std::vector<torch::nn::Conv2d> conv_layers;
    std::vector<torch::nn::BatchNorm2d> batch_norms;
    std::vector<torch::nn::Linear> fc_layers;

    /**
     * Calculate output size after convolutions
     */
    int64_t CalculateConvOutputSize(int64_t input_size, int64_t kernel_size, int64_t stride, int64_t padding = 0);
};

TORCH_MODULE(CNNModel);

#endif // WITH_LIBTORCH
