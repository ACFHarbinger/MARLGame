#include "Models/MARLCommunicationNetwork.h"

#if WITH_LIBTORCH

CommNetImpl::CommNetImpl(int64_t ObsSize, int64_t HiddenSize, int64_t ActionSize, int64_t MsgSize)
{
	Encoder = register_module("encoder", torch::nn::Linear(ObsSize, HiddenSize));
	CommLayer = register_module("comm_layer", torch::nn::Linear(MsgSize, HiddenSize));
	
	// Two separate heads: one for the environment action, one to broadcast the next message
	DecoderAction = register_module("decoder_action", torch::nn::Linear(HiddenSize, ActionSize));
	DecoderMessage = register_module("decoder_message", torch::nn::Linear(HiddenSize, MsgSize));
}

std::tuple<torch::Tensor, torch::Tensor> CommNetImpl::forward(torch::Tensor Obs, torch::Tensor Messages)
{
	// Encode local observation
	auto h_obs = torch::relu(Encoder->forward(Obs));
	
	// Process incoming messages (mean pooling across teammates)
	// Messages expected shape: [Batch, NumTeammates, MsgSize]
	// Mean along dimension 1 (NumTeammates) -> [Batch, MsgSize]
	torch::Tensor PooledMessages = Messages.mean(1);
	auto h_comm = torch::relu(CommLayer->forward(PooledMessages));
	
	// Combine local encoding with communication context (Skip connection / Addition)
	auto h_combined = h_obs + h_comm;
	
	// Decode outputs
	auto ActionLogits = DecoderAction->forward(h_combined);
	auto NextMessage = torch::tanh(DecoderMessage->forward(h_combined)); // Constrain message magnitude
	
	return std::make_tuple(ActionLogits, NextMessage);
}

#endif // WITH_LIBTORCH
