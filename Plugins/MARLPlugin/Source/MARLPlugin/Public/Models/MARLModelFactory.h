#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>
#include "Models/MARLModelBase.h"
#include "Models/MLPModel.h"
#include "Models/LSTMModel.h"
#include "Models/GRUModel.h"
#include "Models/TransformerModel.h"
#include "Models/CNNModel.h"
#include "Models/ResNetModel.h"

/**
 * Model type enumeration
 */
enum class EMARLModelType {
    MLP,         // Multi-Layer Perceptron
    LSTM,        // Long Short-Term Memory
    GRU,         // Gated Recurrent Unit
    Transformer, // Self-Attention Transformer
    CNN,         // Convolutional Neural Network
    ResNet       // Residual Network
};

/**
 * Configuration for model creation
 */
struct FMARLModelConfig {
    EMARLModelType ModelType = EMARLModelType::MLP;

    // Common parameters
    int64_t InputDim = 128;
    int64_t OutputDim = 64;

    // MLP/RNN parameters
    std::vector<int64_t> HiddenSizes = {256, 256};
    int64_t NumLayers = 2;
    float Dropout = 0.0f;

    // RNN specific
    bool Bidirectional = false;

    // Transformer specific
    int64_t EmbedDim = 256;
    int64_t NumHeads = 8;
    int64_t FFDim = 1024;
    int64_t MaxSeqLen = 100;

    // CNN/ResNet specific
    int64_t InputChannels = 3;
    int64_t InputHeight = 84;
    int64_t InputWidth = 84;
    std::vector<int64_t> ConvChannels = {32, 64, 128};
    std::vector<int64_t> KernelSizes = {8, 4, 3};
    std::vector<int64_t> Strides = {4, 2, 1};

    // ResNet specific
    std::vector<int64_t> NumBlocks = {2, 2, 2, 2};
    std::vector<int64_t> ResNetChannels = {64, 128, 256, 512};

    FMARLModelConfig() = default;
};

/**
 * Factory for creating MARL neural network models
 */
class MARLPLUGIN_API FMARLModelFactory {
public:
    /**
     * Create a model based on configuration
     * @param Config Model configuration
     * @return Shared pointer to the created model
     */
    static std::shared_ptr<MARLModelBase> CreateModel(const FMARLModelConfig& Config);

    /**
     * Create MLP model
     */
    static std::shared_ptr<MLPModel> CreateMLPModel(
        int64_t InputDim,
        int64_t OutputDim,
        std::vector<int64_t> HiddenSizes = {256, 256},
        bool UseLayerNorm = true,
        float Dropout = 0.0f
    );

    /**
     * Create LSTM model
     */
    static std::shared_ptr<LSTMModel> CreateLSTMModel(
        int64_t InputDim,
        int64_t HiddenDim,
        int64_t OutputDim,
        int64_t NumLayers = 2,
        bool Bidirectional = false,
        float Dropout = 0.0f
    );

    /**
     * Create GRU model
     */
    static std::shared_ptr<GRUModel> CreateGRUModel(
        int64_t InputDim,
        int64_t HiddenDim,
        int64_t OutputDim,
        int64_t NumLayers = 2,
        bool Bidirectional = false,
        float Dropout = 0.0f
    );

    /**
     * Create Transformer model
     */
    static std::shared_ptr<TransformerModel> CreateTransformerModel(
        int64_t InputDim,
        int64_t EmbedDim,
        int64_t OutputDim,
        int64_t NumHeads = 8,
        int64_t NumLayers = 4,
        int64_t FFDim = 0,
        float Dropout = 0.1f,
        int64_t MaxSeqLen = 100
    );

    /**
     * Create CNN model
     */
    static std::shared_ptr<CNNModel> CreateCNNModel(
        int64_t InputChannels,
        int64_t InputHeight,
        int64_t InputWidth,
        int64_t OutputDim,
        std::vector<int64_t> ConvChannels = {32, 64, 128},
        std::vector<int64_t> KernelSizes = {8, 4, 3},
        std::vector<int64_t> Strides = {4, 2, 1},
        std::vector<int64_t> FCHiddenSizes = {512}
    );

    /**
     * Create ResNet model
     */
    static std::shared_ptr<ResNetModel> CreateResNetModel(
        int64_t InputChannels,
        int64_t InputHeight,
        int64_t InputWidth,
        int64_t OutputDim,
        std::vector<int64_t> NumBlocks = {2, 2, 2, 2},
        std::vector<int64_t> Channels = {64, 128, 256, 512},
        std::vector<int64_t> FCHiddenSizes = {512}
    );

    /**
     * Get recommended model for observation type
     */
    static EMARLModelType GetRecommendedModelType(const FString& ObservationType);

    /**
     * Get default configuration for model type
     */
    static FMARLModelConfig GetDefaultConfig(EMARLModelType ModelType);
};

#endif // WITH_LIBTORCH
