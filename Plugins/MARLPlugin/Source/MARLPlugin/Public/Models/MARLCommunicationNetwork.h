#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * CommNet implementation in LibTorch for MARL Agent communication.
 * Averages messages from teammates and uses MLP to produce next hidden state.
 */
struct MARLPLUGIN_API CommNetImpl : torch::nn::Module
{
	CommNetImpl(int64_t ObsSize, int64_t HiddenSize, int64_t ActionSize, int64_t MsgSize);

	/**
	 * Forward pass for the communication network.
	 * @param Obs Local observation tensor [Batch, ObsSize]
	 * @param Messages Tensor of teammate messages [Batch, NumTeammates, MsgSize]
	 * @return A tuple containing (ActionLogits [Batch, ActionSize], NextMessage [Batch, MsgSize])
	 */
	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor Obs, torch::Tensor Messages);

	torch::nn::Linear Encoder{nullptr};
	torch::nn::Linear CommLayer{nullptr};
	torch::nn::Linear DecoderAction{nullptr};
	torch::nn::Linear DecoderMessage{nullptr};
};

TORCH_MODULE(CommNet);

#endif // WITH_LIBTORCH
