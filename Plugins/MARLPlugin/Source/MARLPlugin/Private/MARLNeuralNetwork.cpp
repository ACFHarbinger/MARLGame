#include "MARLNeuralNetwork.h"

#if WITH_LIBTORCH

// ============================================================================
// Actor Network Implementation
// ============================================================================

ActorNetwork::ActorNetwork(int64_t ObsSize, int64_t ActionSize, std::vector<int64_t> HiddenSizes, bool bContinuous)
	: bContinuousActions(bContinuous)
{
	// Create MLP backbone (default)
	Backbone = FMARLModelFactory::CreateMLPModel(ObsSize, HiddenSizes.back(), HiddenSizes, true);
	register_module("backbone", Backbone);
	BackboneOutputDim = HiddenSizes.back();

	// Action head
	ActionHead = register_module("action_head", torch::nn::Linear(BackboneOutputDim, ActionSize));

	// For continuous actions, separate log_std head
	if (bContinuousActions)
	{
		LogStdHead = register_module("log_std_head", torch::nn::Linear(BackboneOutputDim, ActionSize));
	}
}

ActorNetwork::ActorNetwork(int64_t ObsSize, int64_t ActionSize, const NetworkConfig& Config, bool bContinuous)
	: bContinuousActions(bContinuous)
{
	// Create model config from network config
	FMARLModelConfig ModelConfig;
	ModelConfig.ModelType = Config.BackboneType;
	ModelConfig.InputDim = ObsSize;
	ModelConfig.HiddenSizes = Config.HiddenSizes;
	ModelConfig.NumLayers = Config.NumLayers;
	ModelConfig.Bidirectional = Config.Bidirectional;
	ModelConfig.Dropout = Config.Dropout;
	ModelConfig.EmbedDim = Config.EmbedDim;
	ModelConfig.NumHeads = Config.NumHeads;
	ModelConfig.FFDim = Config.FFDim;
	ModelConfig.InputChannels = Config.InputChannels;
	ModelConfig.InputHeight = Config.InputHeight;
	ModelConfig.InputWidth = Config.InputWidth;

	// Determine output dimension based on model type
	int64_t BackboneOut = Config.HiddenSizes.empty() ? 256 : Config.HiddenSizes[0];
	if (Config.BackboneType == EMARLModelType::Transformer)
	{
		BackboneOut = Config.EmbedDim;
	}
	ModelConfig.OutputDim = BackboneOut;

	// Create backbone using factory
	Backbone = FMARLModelFactory::CreateModel(ModelConfig);
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	// Action head
	ActionHead = register_module("action_head", torch::nn::Linear(BackboneOutputDim, ActionSize));

	// For continuous actions, separate log_std head
	if (bContinuousActions)
	{
		LogStdHead = register_module("log_std_head", torch::nn::Linear(BackboneOutputDim, ActionSize));
	}
}

ActorNetwork::ActorNetwork(int64_t ObsSize, int64_t ActionSize, std::shared_ptr<MARLModelBase> CustomBackbone, bool bContinuous)
	: bContinuousActions(bContinuous)
{
	// Use custom backbone
	Backbone = CustomBackbone;
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	// Action head
	ActionHead = register_module("action_head", torch::nn::Linear(BackboneOutputDim, ActionSize));

	// For continuous actions, separate log_std head
	if (bContinuousActions)
	{
		LogStdHead = register_module("log_std_head", torch::nn::Linear(BackboneOutputDim, ActionSize));
	}
}

torch::Tensor ActorNetwork::forward(torch::Tensor Observations)
{
	torch::Tensor Features = Backbone->forward(Observations);
	torch::Tensor ActionOutput = ActionHead->forward(Features);

	if (bContinuousActions)
	{
		// Return mean (ActionOutput will be used as mean, log_std computed separately)
		return ActionOutput;
	}
	else
	{
		// Return logits for discrete actions
		return ActionOutput;
	}
}

torch::Tensor ActorNetwork::GetActionLogits(torch::Tensor Observations)
{
	return forward(Observations);
}

torch::Tensor ActorNetwork::SampleActions(torch::Tensor Observations, bool bDeterministic)
{
	torch::Tensor Features = Backbone->forward(Observations);

	if (bContinuousActions)
	{
		torch::Tensor Mean = ActionHead->forward(Features);

		if (bDeterministic)
		{
			return Mean;
		}
		else
		{
			torch::Tensor LogStd = LogStdHead->forward(Features);
			torch::Tensor Std = torch::exp(LogStd);
			torch::Tensor Noise = torch::randn_like(Mean);
			return Mean + Std * Noise;
		}
	}
	else
	{
		// Discrete actions
		torch::Tensor Logits = ActionHead->forward(Features);

		if (bDeterministic)
		{
			return torch::argmax(Logits, /*dim=*/1);
		}
		else
		{
			torch::Tensor Probs = torch::softmax(Logits, /*dim=*/1);
			return torch::multinomial(Probs, /*num_samples=*/1).squeeze(1);
		}
	}
}

torch::Tensor ActorNetwork::ComputeLogProb(torch::Tensor Observations, torch::Tensor Actions)
{
	torch::Tensor Features = Backbone->forward(Observations);

	if (bContinuousActions)
	{
		torch::Tensor Mean = ActionHead->forward(Features);
		torch::Tensor LogStd = LogStdHead->forward(Features);
		torch::Tensor Std = torch::exp(LogStd);

		// Gaussian log probability
		torch::Tensor Variance = Std * Std;
		torch::Tensor LogProb = -0.5 * (torch::pow(Actions - Mean, 2) / Variance + torch::log(2.0 * M_PI * Variance));

		// Sum over action dimensions
		return LogProb.sum(/*dim=*/1);
	}
	else
	{
		// Discrete actions
		torch::Tensor Logits = ActionHead->forward(Features);
		torch::Tensor LogProbs = torch::log_softmax(Logits, /*dim=*/1);

		// Gather log probs for taken actions
		return LogProbs.gather(1, Actions.unsqueeze(1).to(torch::kLong)).squeeze(1);
	}
}

torch::Tensor ActorNetwork::ComputeEntropy(torch::Tensor Observations)
{
	torch::Tensor Features = Backbone->forward(Observations);

	if (bContinuousActions)
	{
		torch::Tensor LogStd = LogStdHead->forward(Features);
		torch::Tensor Std = torch::exp(LogStd);

		// Gaussian entropy: 0.5 * log(2 * pi * e * std^2)
		return (0.5 * torch::log(2.0 * M_PI * M_E * Std * Std)).sum(/*dim=*/1);
	}
	else
	{
		// Discrete entropy
		torch::Tensor Logits = ActionHead->forward(Features);
		torch::Tensor Probs = torch::softmax(Logits, /*dim=*/1);
		torch::Tensor LogProbs = torch::log_softmax(Logits, /*dim=*/1);

		return -(Probs * LogProbs).sum(/*dim=*/1);
	}
}

// ============================================================================
// Critic Network Implementation
// ============================================================================

CriticNetwork::CriticNetwork(int64_t ObsSize, std::vector<int64_t> HiddenSizes)
{
	// Create MLP backbone (default)
	Backbone = FMARLModelFactory::CreateMLPModel(ObsSize, HiddenSizes.back(), HiddenSizes, true);
	register_module("backbone", Backbone);
	BackboneOutputDim = HiddenSizes.back();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

CriticNetwork::CriticNetwork(int64_t ObsSize, const NetworkConfig& Config)
{
	// Create model config from network config
	FMARLModelConfig ModelConfig;
	ModelConfig.ModelType = Config.BackboneType;
	ModelConfig.InputDim = ObsSize;
	ModelConfig.HiddenSizes = Config.HiddenSizes;
	ModelConfig.NumLayers = Config.NumLayers;
	ModelConfig.Bidirectional = Config.Bidirectional;
	ModelConfig.Dropout = Config.Dropout;
	ModelConfig.EmbedDim = Config.EmbedDim;
	ModelConfig.NumHeads = Config.NumHeads;
	ModelConfig.FFDim = Config.FFDim;
	ModelConfig.InputChannels = Config.InputChannels;
	ModelConfig.InputHeight = Config.InputHeight;
	ModelConfig.InputWidth = Config.InputWidth;

	// Determine output dimension
	int64_t BackboneOut = Config.HiddenSizes.empty() ? 256 : Config.HiddenSizes[0];
	if (Config.BackboneType == EMARLModelType::Transformer)
	{
		BackboneOut = Config.EmbedDim;
	}
	ModelConfig.OutputDim = BackboneOut;

	// Create backbone using factory
	Backbone = FMARLModelFactory::CreateModel(ModelConfig);
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

CriticNetwork::CriticNetwork(int64_t ObsSize, std::shared_ptr<MARLModelBase> CustomBackbone)
{
	// Use custom backbone
	Backbone = CustomBackbone;
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

torch::Tensor CriticNetwork::forward(torch::Tensor Observations)
{
	torch::Tensor Features = Backbone->forward(Observations);
	torch::Tensor Value = ValueHead->forward(Features);
	return Value;
}

// ============================================================================
// Centralized Critic Network Implementation
// ============================================================================

CentralizedCriticNetwork::CentralizedCriticNetwork(int64_t GlobalStateSize, std::vector<int64_t> HiddenSizes)
{
	// Create MLP backbone (default)
	Backbone = FMARLModelFactory::CreateMLPModel(GlobalStateSize, HiddenSizes.back(), HiddenSizes, true);
	register_module("backbone", Backbone);
	BackboneOutputDim = HiddenSizes.back();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

CentralizedCriticNetwork::CentralizedCriticNetwork(int64_t GlobalStateSize, const NetworkConfig& Config)
{
	// Create model config from network config
	FMARLModelConfig ModelConfig;
	ModelConfig.ModelType = Config.BackboneType;
	ModelConfig.InputDim = GlobalStateSize;
	ModelConfig.HiddenSizes = Config.HiddenSizes;
	ModelConfig.NumLayers = Config.NumLayers;
	ModelConfig.Bidirectional = Config.Bidirectional;
	ModelConfig.Dropout = Config.Dropout;
	ModelConfig.EmbedDim = Config.EmbedDim;
	ModelConfig.NumHeads = Config.NumHeads;
	ModelConfig.FFDim = Config.FFDim;
	ModelConfig.InputChannels = Config.InputChannels;
	ModelConfig.InputHeight = Config.InputHeight;
	ModelConfig.InputWidth = Config.InputWidth;

	// Determine output dimension
	int64_t BackboneOut = Config.HiddenSizes.empty() ? 512 : Config.HiddenSizes[0];
	if (Config.BackboneType == EMARLModelType::Transformer)
	{
		BackboneOut = Config.EmbedDim;
	}
	ModelConfig.OutputDim = BackboneOut;

	// Create backbone using factory
	Backbone = FMARLModelFactory::CreateModel(ModelConfig);
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

CentralizedCriticNetwork::CentralizedCriticNetwork(int64_t GlobalStateSize, std::shared_ptr<MARLModelBase> CustomBackbone)
{
	// Use custom backbone
	Backbone = CustomBackbone;
	register_module("backbone", Backbone);
	BackboneOutputDim = Backbone->GetOutputDim();

	ValueHead = register_module("value_head", torch::nn::Linear(BackboneOutputDim, 1));
}

torch::Tensor CentralizedCriticNetwork::forward(torch::Tensor GlobalState)
{
	torch::Tensor Features = Backbone->forward(GlobalState);
	torch::Tensor Value = ValueHead->forward(Features);
	return Value;
}

#endif // WITH_LIBTORCH
