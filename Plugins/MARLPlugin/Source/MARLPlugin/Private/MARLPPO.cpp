#include "MARLPPO.h"

#if WITH_LIBTORCH

// ============================================================================
// Rollout Buffer Implementation
// ============================================================================

void RolloutBuffer::Clear()
{
	Observations.clear();
	Actions.clear();
	Rewards.clear();
	Values.clear();
	LogProbs.clear();
	Dones.clear();
}

void RolloutBuffer::Add(torch::Tensor Obs, torch::Tensor Action, torch::Tensor Reward,
                         torch::Tensor Value, torch::Tensor LogProb, torch::Tensor Done)
{
	Observations.push_back(Obs);
	Actions.push_back(Action);
	Rewards.push_back(Reward);
	Values.push_back(Value);
	LogProbs.push_back(LogProb);
	Dones.push_back(Done);
}

// ============================================================================
// PPO Algorithm Implementation
// ============================================================================

PPOAlgorithm::PPOAlgorithm(int64_t ObsSize, int64_t ActionSize, const PPOConfig& InConfig)
	: Config(InConfig)
	, Device(Config.Device)
{
	// Create networks
	Actor = std::make_shared<ActorNetwork>(ObsSize, ActionSize, Config.HiddenSizes, Config.bContinuousActions);
	Critic = std::make_shared<CriticNetwork>(ObsSize, Config.HiddenSizes);

	// Move to device
	Actor->to(Device);
	Critic->to(Device);

	// Create optimizers
	ActorOptimizer = std::make_shared<torch::optim::Adam>(
		Actor->parameters(),
		torch::optim::AdamOptions(Config.LearningRate)
	);

	CriticOptimizer = std::make_shared<torch::optim::Adam>(
		Critic->parameters(),
		torch::optim::AdamOptions(Config.LearningRate)
	);
}

PPOAlgorithm::~PPOAlgorithm()
{
}

torch::Tensor PPOAlgorithm::SelectAction(torch::Tensor Observation, bool bDeterministic)
{
	torch::NoGradGuard NoGrad;

	Observation = Observation.to(Device);

	// Add batch dimension if needed
	if (Observation.dim() == 1)
	{
		Observation = Observation.unsqueeze(0);
	}

	torch::Tensor Action = Actor->SampleActions(Observation, bDeterministic);

	return Action;
}

std::tuple<torch::Tensor, torch::Tensor> PPOAlgorithm::EvaluateAction(torch::Tensor Observation, torch::Tensor Action)
{
	Observation = Observation.to(Device);
	Action = Action.to(Device);

	// Add batch dimension if needed
	if (Observation.dim() == 1)
	{
		Observation = Observation.unsqueeze(0);
	}

	torch::Tensor Value = Critic->forward(Observation);
	torch::Tensor LogProb = Actor->ComputeLogProb(Observation, Action);

	return std::make_tuple(Value, LogProb);
}

void PPOAlgorithm::StoreTransition(torch::Tensor Obs, torch::Tensor Action, torch::Tensor Reward,
                                    torch::Tensor Value, torch::Tensor LogProb, torch::Tensor Done)
{
	Buffer.Add(Obs.cpu(), Action.cpu(), Reward.cpu(), Value.cpu(), LogProb.cpu(), Done.cpu());
}

torch::Tensor PPOAlgorithm::ComputeAdvantages(const std::vector<torch::Tensor>& Rewards,
                                               const std::vector<torch::Tensor>& Values,
                                               const std::vector<torch::Tensor>& Dones)
{
	int64_t NumSteps = Rewards.size();
	std::vector<float> Advantages(NumSteps);

	float LastGAE = 0.0f;

	// GAE computation (backward pass)
	for (int64_t t = NumSteps - 1; t >= 0; --t)
	{
		float Reward = Rewards[t].item<float>();
		float Value = Values[t].item<float>();
		float Done = Dones[t].item<float>();
		float NextValue = (t < NumSteps - 1) ? Values[t + 1].item<float>() : 0.0f;

		// TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
		float Delta = Reward + Config.Gamma * NextValue * (1.0f - Done) - Value;

		// GAE: A = delta + gamma * lambda * (1 - done) * A_next
		LastGAE = Delta + Config.Gamma * Config.GAELambda * (1.0f - Done) * LastGAE;
		Advantages[t] = LastGAE;
	}

	return torch::from_blob(Advantages.data(), {NumSteps}, torch::kFloat32).clone();
}

torch::Tensor PPOAlgorithm::ComputeReturns(const std::vector<torch::Tensor>& Rewards,
                                            const std::vector<torch::Tensor>& Dones)
{
	int64_t NumSteps = Rewards.size();
	std::vector<float> Returns(NumSteps);

	float LastReturn = 0.0f;

	for (int64_t t = NumSteps - 1; t >= 0; --t)
	{
		float Reward = Rewards[t].item<float>();
		float Done = Dones[t].item<float>();

		LastReturn = Reward + Config.Gamma * LastReturn * (1.0f - Done);
		Returns[t] = LastReturn;
	}

	return torch::from_blob(Returns.data(), {NumSteps}, torch::kFloat32).clone();
}

std::tuple<float, float, float> PPOAlgorithm::Update()
{
	if (Buffer.Size() == 0)
	{
		return std::make_tuple(0.0f, 0.0f, 0.0f);
	}

	// Stack all tensors
	torch::Tensor ObsTensor = torch::stack(Buffer.Observations).to(Device);
	torch::Tensor ActionTensor = torch::stack(Buffer.Actions).to(Device);
	torch::Tensor OldLogProbTensor = torch::stack(Buffer.LogProbs).to(Device);

	// Compute advantages and returns
	torch::Tensor Advantages = ComputeAdvantages(Buffer.Rewards, Buffer.Values, Buffer.Dones).to(Device);
	torch::Tensor Returns = ComputeReturns(Buffer.Rewards, Buffer.Dones).to(Device);

	// Normalize advantages
	Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-8);

	float TotalPolicyLoss = 0.0f;
	float TotalValueLoss = 0.0f;
	float TotalEntropy = 0.0f;
	int32 NumUpdates = 0;

	// Multiple epochs of optimization
	for (int32 Epoch = 0; Epoch < Config.NumEpochs; ++Epoch)
	{
		// Create random mini-batches
		int64_t NumSamples = ObsTensor.size(0);
		std::vector<int64_t> Indices(NumSamples);
		for (int64_t i = 0; i < NumSamples; ++i)
		{
			Indices[i] = i;
		}
		std::random_shuffle(Indices.begin(), Indices.end());

		// Process mini-batches
		for (int64_t Start = 0; Start < NumSamples; Start += Config.BatchSize)
		{
			int64_t End = std::min(Start + Config.BatchSize, NumSamples);

			std::vector<int64_t> BatchIndices(Indices.begin() + Start, Indices.begin() + End);
			torch::Tensor IndexTensor = torch::from_blob(BatchIndices.data(), {(int64_t)BatchIndices.size()}, torch::kLong).to(Device);

			torch::Tensor ObsBatch = ObsTensor.index_select(0, IndexTensor);
			torch::Tensor ActionBatch = ActionTensor.index_select(0, IndexTensor);
			torch::Tensor OldLogProbBatch = OldLogProbTensor.index_select(0, IndexTensor);
			torch::Tensor AdvantageBatch = Advantages.index_select(0, IndexTensor);
			torch::Tensor ReturnBatch = Returns.index_select(0, IndexTensor);

			auto [PolicyLoss, ValueLoss, Entropy] = TrainStep(ObsBatch, ActionBatch, OldLogProbBatch, AdvantageBatch, ReturnBatch);

			TotalPolicyLoss += PolicyLoss;
			TotalValueLoss += ValueLoss;
			TotalEntropy += Entropy;
			NumUpdates++;
		}
	}

	// Clear buffer
	ClearBuffer();

	return std::make_tuple(
		TotalPolicyLoss / NumUpdates,
		TotalValueLoss / NumUpdates,
		TotalEntropy / NumUpdates
	);
}

std::tuple<float, float, float> PPOAlgorithm::TrainStep(torch::Tensor ObsBatch,
                                                         torch::Tensor ActionBatch,
                                                         torch::Tensor OldLogProbBatch,
                                                         torch::Tensor AdvantageBatch,
                                                         torch::Tensor ReturnBatch)
{
	// Evaluate current policy
	torch::Tensor NewLogProbs = Actor->ComputeLogProb(ObsBatch, ActionBatch);
	torch::Tensor Entropy = Actor->ComputeEntropy(ObsBatch);
	torch::Tensor Values = Critic->forward(ObsBatch).squeeze();

	// PPO policy loss
	torch::Tensor Ratio = torch::exp(NewLogProbs - OldLogProbBatch);
	torch::Tensor SurrogateLoss1 = Ratio * AdvantageBatch;
	torch::Tensor SurrogateLoss2 = torch::clamp(Ratio, 1.0f - Config.ClipEpsilon, 1.0f + Config.ClipEpsilon) * AdvantageBatch;
	torch::Tensor PolicyLoss = -torch::min(SurrogateLoss1, SurrogateLoss2).mean();

	// Value loss (MSE)
	torch::Tensor ValueLoss = torch::mse_loss(Values, ReturnBatch);

	// Total loss
	torch::Tensor TotalLoss = PolicyLoss + Config.ValueLossCoef * ValueLoss - Config.EntropyCoef * Entropy.mean();

	// Update actor
	ActorOptimizer->zero_grad();
	PolicyLoss.backward({}, /*retain_graph=*/true);
	torch::nn::utils::clip_grad_norm_(Actor->parameters(), Config.MaxGradNorm);
	ActorOptimizer->step();

	// Update critic
	CriticOptimizer->zero_grad();
	ValueLoss.backward();
	torch::nn::utils::clip_grad_norm_(Critic->parameters(), Config.MaxGradNorm);
	CriticOptimizer->step();

	return std::make_tuple(
		PolicyLoss.item<float>(),
		ValueLoss.item<float>(),
		Entropy.mean().item<float>()
	);
}

void PPOAlgorithm::ClearBuffer()
{
	Buffer.Clear();
}

void PPOAlgorithm::SaveModel(const std::string& Path)
{
	torch::save(Actor, Path + "_actor.pt");
	torch::save(Critic, Path + "_critic.pt");
}

void PPOAlgorithm::LoadModel(const std::string& Path)
{
	torch::load(Actor, Path + "_actor.pt");
	torch::load(Critic, Path + "_critic.pt");
}

float PPOAlgorithm::GetLearningRate() const
{
	return Config.LearningRate;
}

void PPOAlgorithm::SetLearningRate(float NewLR)
{
	Config.LearningRate = NewLR;

	for (auto& Group : ActorOptimizer->param_groups())
	{
		static_cast<torch::optim::AdamOptions&>(Group.options()).lr(NewLR);
	}

	for (auto& Group : CriticOptimizer->param_groups())
	{
		static_cast<torch::optim::AdamOptions&>(Group.options()).lr(NewLR);
	}
}

// ============================================================================
// MAPPO Algorithm Implementation
// ============================================================================

MAPPOAlgorithm::MAPPOAlgorithm(int64_t ObsSize, int64_t ActionSize, int64_t InNumAgents, const PPOConfig& InConfig)
	: Config(InConfig)
	, NumAgents(InNumAgents)
	, GlobalStateSize(ObsSize * NumAgents)
	, Device(Config.Device)
{
	// Create shared actor for all agents
	Actor = std::make_shared<ActorNetwork>(ObsSize, ActionSize, Config.HiddenSizes, Config.bContinuousActions);

	// Create centralized critic
	Critic = std::make_shared<CentralizedCriticNetwork>(GlobalStateSize, Config.HiddenSizes);

	// Move to device
	Actor->to(Device);
	Critic->to(Device);

	// Create optimizers
	ActorOptimizer = std::make_shared<torch::optim::Adam>(
		Actor->parameters(),
		torch::optim::AdamOptions(Config.LearningRate)
	);

	CriticOptimizer = std::make_shared<torch::optim::Adam>(
		Critic->parameters(),
		torch::optim::AdamOptions(Config.LearningRate)
	);
}

MAPPOAlgorithm::~MAPPOAlgorithm()
{
}

torch::Tensor MAPPOAlgorithm::SelectActions(torch::Tensor Observations, bool bDeterministic)
{
	torch::NoGradGuard NoGrad;

	Observations = Observations.to(Device);

	// Observations should be [num_agents, obs_size]
	torch::Tensor Actions = Actor->SampleActions(Observations, bDeterministic);

	return Actions;
}

void MAPPOAlgorithm::StoreTransition(torch::Tensor Observations, torch::Tensor GlobalState,
                                      torch::Tensor Actions, torch::Tensor Rewards, torch::Tensor Dones)
{
	// Store to buffer (on CPU to save GPU memory)
	ObservationsBuffer.push_back(Observations.cpu());
	GlobalStatesBuffer.push_back(GlobalState.cpu());
	ActionsBuffer.push_back(Actions.cpu());
	RewardsBuffer.push_back(Rewards.cpu());
	DonesBuffer.push_back(Dones.cpu());

	// Compute values and log probs
	torch::NoGradGuard NoGrad;
	torch::Tensor Values = Critic->forward(GlobalState.to(Device));
	torch::Tensor LogProbs = Actor->ComputeLogProb(Observations.to(Device), Actions.to(Device));

	ValuesBuffer.push_back(Values.cpu());
	LogProbsBuffer.push_back(LogProbs.cpu());
}

torch::Tensor MAPPOAlgorithm::ComputeAdvantages(const std::vector<torch::Tensor>& Rewards,
                                                 const std::vector<torch::Tensor>& Values,
                                                 const std::vector<torch::Tensor>& Dones)
{
	int64_t NumSteps = Rewards.size();
	std::vector<torch::Tensor> Advantages(NumSteps);

	torch::Tensor LastGAE = torch::zeros({NumAgents});

	for (int64_t t = NumSteps - 1; t >= 0; --t)
	{
		torch::Tensor Reward = Rewards[t];
		torch::Tensor Value = Values[t].squeeze();
		torch::Tensor Done = Dones[t];
		torch::Tensor NextValue = (t < NumSteps - 1) ? Values[t + 1].squeeze() : torch::zeros({NumAgents});

		torch::Tensor Delta = Reward + Config.Gamma * NextValue * (1.0f - Done) - Value;
		LastGAE = Delta + Config.Gamma * Config.GAELambda * (1.0f - Done) * LastGAE;
		Advantages[t] = LastGAE;
	}

	return torch::stack(Advantages);
}

std::tuple<float, float, float> MAPPOAlgorithm::Update()
{
	if (ObservationsBuffer.size() == 0)
	{
		return std::make_tuple(0.0f, 0.0f, 0.0f);
	}

	// Stack all tensors [num_steps, num_agents, ...]
	torch::Tensor ObsTensor = torch::stack(ObservationsBuffer).to(Device);
	torch::Tensor GlobalStateTensor = torch::stack(GlobalStatesBuffer).to(Device);
	torch::Tensor ActionTensor = torch::stack(ActionsBuffer).to(Device);
	torch::Tensor OldLogProbTensor = torch::stack(LogProbsBuffer).to(Device);

	// Compute advantages
	torch::Tensor Advantages = ComputeAdvantages(RewardsBuffer, ValuesBuffer, DonesBuffer).to(Device);

	// Compute returns
	torch::Tensor Returns = Advantages + torch::stack(ValuesBuffer).to(Device).squeeze();

	// Normalize advantages
	Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-8);

	// Reshape to [num_steps * num_agents, ...]
	int64_t NumSteps = ObsTensor.size(0);
	ObsTensor = ObsTensor.reshape({NumSteps * NumAgents, -1});
	GlobalStateTensor = GlobalStateTensor.reshape({NumSteps * NumAgents, -1});
	ActionTensor = ActionTensor.reshape({NumSteps * NumAgents, -1});
	OldLogProbTensor = OldLogProbTensor.reshape({NumSteps * NumAgents});
	Advantages = Advantages.reshape({NumSteps * NumAgents});
	Returns = Returns.reshape({NumSteps * NumAgents});

	float TotalPolicyLoss = 0.0f;
	float TotalValueLoss = 0.0f;
	float TotalEntropy = 0.0f;
	int32 NumUpdates = 0;

	// Multiple epochs
	for (int32 Epoch = 0; Epoch < Config.NumEpochs; ++Epoch)
	{
		int64_t NumSamples = ObsTensor.size(0);
		std::vector<int64_t> Indices(NumSamples);
		for (int64_t i = 0; i < NumSamples; ++i)
		{
			Indices[i] = i;
		}
		std::random_shuffle(Indices.begin(), Indices.end());

		for (int64_t Start = 0; Start < NumSamples; Start += Config.BatchSize)
		{
			int64_t End = std::min(Start + Config.BatchSize, NumSamples);

			std::vector<int64_t> BatchIndices(Indices.begin() + Start, Indices.begin() + End);
			torch::Tensor IndexTensor = torch::from_blob(BatchIndices.data(), {(int64_t)BatchIndices.size()}, torch::kLong).to(Device);

			torch::Tensor ObsBatch = ObsTensor.index_select(0, IndexTensor);
			torch::Tensor GlobalStateBatch = GlobalStateTensor.index_select(0, IndexTensor);
			torch::Tensor ActionBatch = ActionTensor.index_select(0, IndexTensor);
			torch::Tensor OldLogProbBatch = OldLogProbTensor.index_select(0, IndexTensor);
			torch::Tensor AdvantageBatch = Advantages.index_select(0, IndexTensor);
			torch::Tensor ReturnBatch = Returns.index_select(0, IndexTensor);

			auto [PolicyLoss, ValueLoss, Entropy] = TrainStep(ObsBatch, GlobalStateBatch, ActionBatch, OldLogProbBatch, AdvantageBatch, ReturnBatch);

			TotalPolicyLoss += PolicyLoss;
			TotalValueLoss += ValueLoss;
			TotalEntropy += Entropy;
			NumUpdates++;
		}
	}

	ClearBuffer();

	return std::make_tuple(
		TotalPolicyLoss / NumUpdates,
		TotalValueLoss / NumUpdates,
		TotalEntropy / NumUpdates
	);
}

std::tuple<float, float, float> MAPPOAlgorithm::TrainStep(torch::Tensor ObsBatch,
                                                           torch::Tensor GlobalStateBatch,
                                                           torch::Tensor ActionBatch,
                                                           torch::Tensor OldLogProbBatch,
                                                           torch::Tensor AdvantageBatch,
                                                           torch::Tensor ReturnBatch)
{
	// Evaluate current policy
	torch::Tensor NewLogProbs = Actor->ComputeLogProb(ObsBatch, ActionBatch);
	torch::Tensor Entropy = Actor->ComputeEntropy(ObsBatch);
	torch::Tensor Values = Critic->forward(GlobalStateBatch).squeeze();

	// PPO policy loss
	torch::Tensor Ratio = torch::exp(NewLogProbs - OldLogProbBatch);
	torch::Tensor SurrogateLoss1 = Ratio * AdvantageBatch;
	torch::Tensor SurrogateLoss2 = torch::clamp(Ratio, 1.0f - Config.ClipEpsilon, 1.0f + Config.ClipEpsilon) * AdvantageBatch;
	torch::Tensor PolicyLoss = -torch::min(SurrogateLoss1, SurrogateLoss2).mean();

	// Value loss
	torch::Tensor ValueLoss = torch::mse_loss(Values, ReturnBatch);

	// Update actor
	ActorOptimizer->zero_grad();
	PolicyLoss.backward({}, /*retain_graph=*/true);
	torch::nn::utils::clip_grad_norm_(Actor->parameters(), Config.MaxGradNorm);
	ActorOptimizer->step();

	// Update critic
	CriticOptimizer->zero_grad();
	ValueLoss.backward();
	torch::nn::utils::clip_grad_norm_(Critic->parameters(), Config.MaxGradNorm);
	CriticOptimizer->step();

	return std::make_tuple(
		PolicyLoss.item<float>(),
		ValueLoss.item<float>(),
		Entropy.mean().item<float>()
	);
}

void MAPPOAlgorithm::ClearBuffer()
{
	ObservationsBuffer.clear();
	GlobalStatesBuffer.clear();
	ActionsBuffer.clear();
	RewardsBuffer.clear();
	ValuesBuffer.clear();
	LogProbsBuffer.clear();
	DonesBuffer.clear();
}

void MAPPOAlgorithm::SaveModels(const std::string& PathPrefix)
{
	torch::save(Actor, PathPrefix + "_actor.pt");
	torch::save(Critic, PathPrefix + "_critic.pt");
}

void MAPPOAlgorithm::LoadModels(const std::string& PathPrefix)
{
	torch::load(Actor, PathPrefix + "_actor.pt");
	torch::load(Critic, PathPrefix + "_critic.pt");
}

#endif // WITH_LIBTORCH
