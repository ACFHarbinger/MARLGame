#include "Models/MARLModelFactory.h"

#if WITH_LIBTORCH

std::shared_ptr<MARLModelBase> FMARLModelFactory::CreateModel(const FMARLModelConfig& Config) {
    switch (Config.ModelType) {
        case EMARLModelType::LSTM:
            return CreateLSTMModel(
                Config.InputDim,
                Config.HiddenSizes[0],
                Config.OutputDim,
                Config.NumLayers,
                Config.Bidirectional,
                Config.Dropout
            );

        case EMARLModelType::GRU:
            return CreateGRUModel(
                Config.InputDim,
                Config.HiddenSizes[0],
                Config.OutputDim,
                Config.NumLayers,
                Config.Bidirectional,
                Config.Dropout
            );

        case EMARLModelType::Transformer:
            return CreateTransformerModel(
                Config.InputDim,
                Config.EmbedDim,
                Config.OutputDim,
                Config.NumHeads,
                Config.NumLayers,
                Config.FFDim,
                Config.Dropout,
                Config.MaxSeqLen
            );

        case EMARLModelType::CNN:
            return CreateCNNModel(
                Config.InputChannels,
                Config.InputHeight,
                Config.InputWidth,
                Config.OutputDim,
                Config.ConvChannels,
                Config.KernelSizes,
                Config.Strides,
                Config.HiddenSizes
            );

        case EMARLModelType::ResNet:
            return CreateResNetModel(
                Config.InputChannels,
                Config.InputHeight,
                Config.InputWidth,
                Config.OutputDim,
                Config.NumBlocks,
                Config.ResNetChannels,
                Config.HiddenSizes
            );

        case EMARLModelType::MLP:
        default:
            return CreateMLPModel(
                Config.InputDim,
                Config.OutputDim,
                Config.HiddenSizes,
                true,
                Config.Dropout
            );
    }
}

std::shared_ptr<MLPModel> FMARLModelFactory::CreateMLPModel(
    int64_t InputDim,
    int64_t OutputDim,
    std::vector<int64_t> HiddenSizes,
    bool UseLayerNorm,
    float Dropout
) {
    return std::make_shared<MLPModel>(InputDim, OutputDim, HiddenSizes, UseLayerNorm, Dropout);
}

std::shared_ptr<LSTMModel> FMARLModelFactory::CreateLSTMModel(
    int64_t InputDim,
    int64_t HiddenDim,
    int64_t OutputDim,
    int64_t NumLayers,
    bool Bidirectional,
    float Dropout
) {
    return std::make_shared<LSTMModel>(InputDim, HiddenDim, OutputDim, NumLayers, Bidirectional, Dropout);
}

std::shared_ptr<GRUModel> FMARLModelFactory::CreateGRUModel(
    int64_t InputDim,
    int64_t HiddenDim,
    int64_t OutputDim,
    int64_t NumLayers,
    bool Bidirectional,
    float Dropout
) {
    return std::make_shared<GRUModel>(InputDim, HiddenDim, OutputDim, NumLayers, Bidirectional, Dropout);
}

std::shared_ptr<TransformerModel> FMARLModelFactory::CreateTransformerModel(
    int64_t InputDim,
    int64_t EmbedDim,
    int64_t OutputDim,
    int64_t NumHeads,
    int64_t NumLayers,
    int64_t FFDim,
    float Dropout,
    int64_t MaxSeqLen
) {
    return std::make_shared<TransformerModel>(
        InputDim, EmbedDim, OutputDim, NumHeads, NumLayers, FFDim, Dropout, MaxSeqLen
    );
}

std::shared_ptr<CNNModel> FMARLModelFactory::CreateCNNModel(
    int64_t InputChannels,
    int64_t InputHeight,
    int64_t InputWidth,
    int64_t OutputDim,
    std::vector<int64_t> ConvChannels,
    std::vector<int64_t> KernelSizes,
    std::vector<int64_t> Strides,
    std::vector<int64_t> FCHiddenSizes
) {
    return std::make_shared<CNNModel>(
        InputChannels, InputHeight, InputWidth, OutputDim,
        ConvChannels, KernelSizes, Strides, FCHiddenSizes
    );
}

std::shared_ptr<ResNetModel> FMARLModelFactory::CreateResNetModel(
    int64_t InputChannels,
    int64_t InputHeight,
    int64_t InputWidth,
    int64_t OutputDim,
    std::vector<int64_t> NumBlocks,
    std::vector<int64_t> Channels,
    std::vector<int64_t> FCHiddenSizes
) {
    return std::make_shared<ResNetModel>(
        InputChannels, InputHeight, InputWidth, OutputDim,
        NumBlocks, Channels, FCHiddenSizes
    );
}

EMARLModelType FMARLModelFactory::GetRecommendedModelType(const FString& ObservationType) {
    if (ObservationType.Contains(TEXT("image")) || ObservationType.Contains(TEXT("visual"))) {
        return EMARLModelType::CNN;
    }
    else if (ObservationType.Contains(TEXT("sequence")) || ObservationType.Contains(TEXT("temporal"))) {
        return EMARLModelType::LSTM;
    }
    else if (ObservationType.Contains(TEXT("attention")) || ObservationType.Contains(TEXT("relational"))) {
        return EMARLModelType::Transformer;
    }
    else {
        return EMARLModelType::MLP;
    }
}

FMARLModelConfig FMARLModelFactory::GetDefaultConfig(EMARLModelType ModelType) {
    FMARLModelConfig Config;
    Config.ModelType = ModelType;

    switch (ModelType) {
        case EMARLModelType::LSTM:
            Config.HiddenSizes = {256};
            Config.NumLayers = 2;
            Config.Dropout = 0.0f;
            Config.Bidirectional = false;
            break;

        case EMARLModelType::GRU:
            Config.HiddenSizes = {256};
            Config.NumLayers = 2;
            Config.Dropout = 0.0f;
            Config.Bidirectional = false;
            break;

        case EMARLModelType::Transformer:
            Config.EmbedDim = 256;
            Config.NumHeads = 8;
            Config.NumLayers = 4;
            Config.FFDim = 1024;
            Config.Dropout = 0.1f;
            Config.MaxSeqLen = 100;
            break;

        case EMARLModelType::CNN:
            Config.InputChannels = 3;
            Config.InputHeight = 84;
            Config.InputWidth = 84;
            Config.ConvChannels = {32, 64, 128};
            Config.KernelSizes = {8, 4, 3};
            Config.Strides = {4, 2, 1};
            Config.HiddenSizes = {512};
            break;

        case EMARLModelType::ResNet:
            Config.InputChannels = 3;
            Config.InputHeight = 84;
            Config.InputWidth = 84;
            Config.NumBlocks = {2, 2, 2, 2};
            Config.ResNetChannels = {64, 128, 256, 512};
            Config.HiddenSizes = {512};
            break;

        case EMARLModelType::MLP:
        default:
            Config.HiddenSizes = {256, 256};
            break;
    }

    return Config;
}

#endif // WITH_LIBTORCH
