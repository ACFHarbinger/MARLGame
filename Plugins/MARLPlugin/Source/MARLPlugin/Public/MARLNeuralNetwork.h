#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>
#include "Models/MARLModelBase.h"
#include "Models/MARLModelFactory.h"

/**
 * Network configuration for Actor/Critic networks
 */
struct NetworkConfig
{
	EMARLModelType BackboneType = EMARLModelType::MLP;
	std::vector<int64_t> HiddenSizes = {256, 256};

	// RNN specific
	int64_t NumLayers = 2;
	bool Bidirectional = false;
	float Dropout = 0.0f;

	// Transformer specific
	int64_t EmbedDim = 256;
	int64_t NumHeads = 8;
	int64_t FFDim = 1024;

	// CNN specific
	int64_t InputChannels = 3;
	int64_t InputHeight = 84;
	int64_t InputWidth = 84;

	NetworkConfig() = default;
};

/**
 * Actor Network (Policy)
 * Outputs action logits for discrete actions or mean/std for continuous actions
 * Supports multiple backbone architectures via MARLModelFactory
 */
struct ActorNetwork : torch::nn::Module
{
	/**
	 * Constructor with MLP backbone (default)
	 */
	ActorNetwork(int64_t ObsSize, int64_t ActionSize, std::vector<int64_t> HiddenSizes = {256, 256}, bool bContinuous = false);

	/**
	 * Constructor with configurable backbone
	 */
	ActorNetwork(int64_t ObsSize, int64_t ActionSize, const NetworkConfig& Config, bool bContinuous = false);

	/**
	 * Constructor with custom backbone model
	 */
	ActorNetwork(int64_t ObsSize, int64_t ActionSize, std::shared_ptr<MARLModelBase> CustomBackbone, bool bContinuous = false);

	/**
	 * Forward pass
	 * @param Observations Input observations [batch_size, obs_size]
	 * @return For discrete: logits [batch_size, action_size]
	 *         For continuous: tuple of (mean, log_std) [batch_size, action_size]
	 */
	torch::Tensor forward(torch::Tensor Observations);

	/**
	 * Get action distribution
	 */
	torch::Tensor GetActionLogits(torch::Tensor Observations);

	/**
	 * Sample actions from the policy
	 * @param Observations Input observations
	 * @param Deterministic If true, use argmax/mean instead of sampling
	 * @return Sampled actions [batch_size, action_size]
	 */
	torch::Tensor SampleActions(torch::Tensor Observations, bool bDeterministic = false);

	/**
	 * Compute log probabilities of actions
	 * @param Observations Input observations
	 * @param Actions Actions to evaluate
	 * @return Log probabilities [batch_size]
	 */
	torch::Tensor ComputeLogProb(torch::Tensor Observations, torch::Tensor Actions);

	/**
	 * Compute entropy of the policy
	 */
	torch::Tensor ComputeEntropy(torch::Tensor Observations);

	/**
	 * Get the backbone model type
	 */
	const char* GetBackboneType() const { return Backbone ? Backbone->GetModelType() : "None"; }

	/**
	 * Reset internal state (for recurrent models)
	 */
	void ResetState() { if (Backbone) Backbone->ResetState(); }

	std::shared_ptr<MARLModelBase> Backbone{nullptr};
	torch::nn::Linear ActionHead{nullptr};
	torch::nn::Linear LogStdHead{nullptr};  // For continuous actions
	bool bContinuousActions;
	int64_t BackboneOutputDim;
};

/**
 * Critic Network (Value Function)
 * Outputs state value V(s)
 * Supports multiple backbone architectures via MARLModelFactory
 */
struct CriticNetwork : torch::nn::Module
{
	/**
	 * Constructor with MLP backbone (default)
	 */
	CriticNetwork(int64_t ObsSize, std::vector<int64_t> HiddenSizes = {256, 256});

	/**
	 * Constructor with configurable backbone
	 */
	CriticNetwork(int64_t ObsSize, const NetworkConfig& Config);

	/**
	 * Constructor with custom backbone model
	 */
	CriticNetwork(int64_t ObsSize, std::shared_ptr<MARLModelBase> CustomBackbone);

	/**
	 * Forward pass
	 * @param Observations Input observations [batch_size, obs_size]
	 * @return State values [batch_size, 1]
	 */
	torch::Tensor forward(torch::Tensor Observations);

	/**
	 * Get the backbone model type
	 */
	const char* GetBackboneType() const { return Backbone ? Backbone->GetModelType() : "None"; }

	/**
	 * Reset internal state (for recurrent models)
	 */
	void ResetState() { if (Backbone) Backbone->ResetState(); }

	std::shared_ptr<MARLModelBase> Backbone{nullptr};
	torch::nn::Linear ValueHead{nullptr};
	int64_t BackboneOutputDim;
};

/**
 * Centralized Critic Network (for MAPPO)
 * Takes global state (concatenated observations) as input
 * Supports multiple backbone architectures via MARLModelFactory
 */
struct CentralizedCriticNetwork : torch::nn::Module
{
	/**
	 * Constructor with MLP backbone (default)
	 */
	CentralizedCriticNetwork(int64_t GlobalStateSize, std::vector<int64_t> HiddenSizes = {512, 512, 256});

	/**
	 * Constructor with configurable backbone
	 */
	CentralizedCriticNetwork(int64_t GlobalStateSize, const NetworkConfig& Config);

	/**
	 * Constructor with custom backbone model
	 */
	CentralizedCriticNetwork(int64_t GlobalStateSize, std::shared_ptr<MARLModelBase> CustomBackbone);

	/**
	 * Forward pass
	 * @param GlobalState Concatenated observations from all agents [batch_size, global_state_size]
	 * @return State values [batch_size, 1]
	 */
	torch::Tensor forward(torch::Tensor GlobalState);

	/**
	 * Get the backbone model type
	 */
	const char* GetBackboneType() const { return Backbone ? Backbone->GetModelType() : "None"; }

	/**
	 * Reset internal state (for recurrent models)
	 */
	void ResetState() { if (Backbone) Backbone->ResetState(); }

	std::shared_ptr<MARLModelBase> Backbone{nullptr};
	torch::nn::Linear ValueHead{nullptr};
	int64_t BackboneOutputDim;
};

#endif // WITH_LIBTORCH
