#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Base interface for all MARL neural network models
 * Defines the common interface that all models must implement
 */
struct MARLModelBase : torch::nn::Module {
    virtual ~MARLModelBase() = default;

    /**
     * Forward pass through the network
     * @param x Input tensor [batch_size, input_dim]
     * @return Output tensor [batch_size, output_dim]
     */
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /**
     * Get the output dimension of the model
     */
    virtual int64_t GetOutputDim() const = 0;

    /**
     * Get the input dimension of the model
     */
    virtual int64_t GetInputDim() const = 0;

    /**
     * Reset internal state (for recurrent models)
     */
    virtual void ResetState() {}

    /**
     * Get model type name
     */
    virtual const char* GetModelType() const = 0;
};

#endif // WITH_LIBTORCH
