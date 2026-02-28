#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>
#include "MARLNeuralNetwork.h"
#include <vector>
#include <memory>

/**
 * Rollout Buffer for storing experience
 */
struct RolloutBuffer
{
	std::vector<torch::Tensor> Observations;
	std::vector<torch::Tensor> Actions;
	std::vector<torch::Tensor> Rewards;
	std::vector<torch::Tensor> Values;
	std::vector<torch::Tensor> LogProbs;
	std::vector<torch::Tensor> Dones;

	void Clear();
	void Add(torch::Tensor Obs, torch::Tensor Action, torch::Tensor Reward,
	         torch::Tensor Value, torch::Tensor LogProb, torch::Tensor Done);
	int64_t Size() const { return Observations.size(); }
};

/**
 * PPO Algorithm Configuration
 */
struct PPOConfig
{
	// Network architecture
	std::vector<int64_t> HiddenSizes = {256, 256};
	bool bContinuousActions = false;

	// Training hyperparameters
	float LearningRate = 3e-4f;
	float Gamma = 0.99f;          // Discount factor
	float GAELambda = 0.95f;      // GAE parameter
	float ClipEpsilon = 0.2f;     // PPO clipping parameter
	float ValueLossCoef = 0.5f;   // Value loss coefficient
	float EntropyCoef = 0.01f;    // Entropy bonus coefficient
	float MaxGradNorm = 0.5f;     // Gradient clipping

	// Training schedule
	int32 NumEpochs = 10;         // Number of optimization epochs per update
	int32 BatchSize = 64;         // Mini-batch size
	int32 BufferSize = 2048;      // Rollout buffer size

	// Device
	torch::DeviceType Device = torch::kCPU;  // Use torch::kCUDA for GPU

	PPOConfig() = default;
};

/**
 * Proximal Policy Optimization (PPO) Algorithm
 * Supports both single-agent and multi-agent (independent learners) scenarios
 */
class MARLPLUGIN_API PPOAlgorithm
{
public:
	PPOAlgorithm(int64_t ObsSize, int64_t ActionSize, const PPOConfig& Config);
	~PPOAlgorithm();

	/**
	 * Select action for given observation
	 * @param Observation Input observation tensor [obs_size]
	 * @param bDeterministic Use deterministic policy (no exploration)
	 * @return Action tensor
	 */
	torch::Tensor SelectAction(torch::Tensor Observation, bool bDeterministic = false);

	/**
	 * Evaluate action (get value and log prob)
	 * @param Observation Input observation
	 * @param Action Action to evaluate
	 * @return Tuple of (Value, LogProb)
	 */
	std::tuple<torch::Tensor, torch::Tensor> EvaluateAction(torch::Tensor Observation, torch::Tensor Action);

	/**
	 * Add experience to rollout buffer
	 */
	void StoreTransition(torch::Tensor Obs, torch::Tensor Action, torch::Tensor Reward,
	                     torch::Tensor Value, torch::Tensor LogProb, torch::Tensor Done);

	/**
	 * Update policy using collected rollouts
	 * @return Training statistics (policy loss, value loss, entropy)
	 */
	std::tuple<float, float, float> Update();

	/**
	 * Clear rollout buffer
	 */
	void ClearBuffer();

	/**
	 * Save model to file
	 */
	void SaveModel(const std::string& Path);

	/**
	 * Load model from file
	 */
	void LoadModel(const std::string& Path);

	/**
	 * Get current learning rate
	 */
	float GetLearningRate() const;

	/**
	 * Set learning rate (for lr scheduling)
	 */
	void SetLearningRate(float NewLR);

protected:
	/**
	 * Compute advantages using GAE
	 */
	torch::Tensor ComputeAdvantages(const std::vector<torch::Tensor>& Rewards,
	                                 const std::vector<torch::Tensor>& Values,
	                                 const std::vector<torch::Tensor>& Dones);

	/**
	 * Compute returns from rewards
	 */
	torch::Tensor ComputeReturns(const std::vector<torch::Tensor>& Rewards,
	                              const std::vector<torch::Tensor>& Dones);

	/**
	 * Single update step
	 */
	std::tuple<float, float, float> TrainStep(torch::Tensor ObsBatch,
	                                            torch::Tensor ActionBatch,
	                                            torch::Tensor OldLogProbBatch,
	                                            torch::Tensor AdvantageBatch,
	                                            torch::Tensor ReturnBatch);

private:
	PPOConfig Config;

	std::shared_ptr<ActorNetwork> Actor;
	std::shared_ptr<CriticNetwork> Critic;

	std::shared_ptr<torch::optim::Adam> ActorOptimizer;
	std::shared_ptr<torch::optim::Adam> CriticOptimizer;

	RolloutBuffer Buffer;

	torch::Device Device;
};

/**
 * Multi-Agent PPO (MAPPO) with Centralized Critic
 * Uses centralized value function for better coordination
 */
class MARLPLUGIN_API MAPPOAlgorithm
{
public:
	MAPPOAlgorithm(int64_t ObsSize, int64_t ActionSize, int64_t NumAgents, const PPOConfig& Config);
	~MAPPOAlgorithm();

	/**
	 * Select actions for all agents
	 * @param Observations Batched observations [num_agents, obs_size]
	 * @param bDeterministic Use deterministic policy
	 * @return Actions tensor [num_agents, action_size]
	 */
	torch::Tensor SelectActions(torch::Tensor Observations, bool bDeterministic = false);

	/**
	 * Store transition for all agents
	 * @param Observations Individual observations [num_agents, obs_size]
	 * @param GlobalState Concatenated global state [num_agents * obs_size]
	 * @param Actions Actions [num_agents, action_size]
	 * @param Rewards Rewards [num_agents]
	 * @param Dones Done flags [num_agents]
	 */
	void StoreTransition(torch::Tensor Observations, torch::Tensor GlobalState,
	                     torch::Tensor Actions, torch::Tensor Rewards, torch::Tensor Dones);

	/**
	 * Update all policies using collected rollouts
	 * @return Training statistics (avg_policy_loss, avg_value_loss, avg_entropy)
	 */
	std::tuple<float, float, float> Update();

	/**
	 * Clear rollout buffer
	 */
	void ClearBuffer();

	/**
	 * Save models
	 */
	void SaveModels(const std::string& PathPrefix);

	/**
	 * Load models
	 */
	void LoadModels(const std::string& PathPrefix);

protected:
	torch::Tensor ComputeAdvantages(const std::vector<torch::Tensor>& Rewards,
	                                 const std::vector<torch::Tensor>& Values,
	                                 const std::vector<torch::Tensor>& Dones);

	std::tuple<float, float, float> TrainStep(torch::Tensor ObsBatch,
	                                            torch::Tensor GlobalStateBatch,
	                                            torch::Tensor ActionBatch,
	                                            torch::Tensor OldLogProbBatch,
	                                            torch::Tensor AdvantageBatch,
	                                            torch::Tensor ReturnBatch);

private:
	PPOConfig Config;
	int64_t NumAgents;
	int64_t GlobalStateSize;

	std::shared_ptr<ActorNetwork> Actor;  // Shared policy for all agents
	std::shared_ptr<CentralizedCriticNetwork> Critic;  // Centralized critic

	std::shared_ptr<torch::optim::Adam> ActorOptimizer;
	std::shared_ptr<torch::optim::Adam> CriticOptimizer;

	// Rollout buffer
	std::vector<torch::Tensor> ObservationsBuffer;
	std::vector<torch::Tensor> GlobalStatesBuffer;
	std::vector<torch::Tensor> ActionsBuffer;
	std::vector<torch::Tensor> RewardsBuffer;
	std::vector<torch::Tensor> ValuesBuffer;
	std::vector<torch::Tensor> LogProbsBuffer;
	std::vector<torch::Tensor> DonesBuffer;

	torch::Device Device;
};

#endif // WITH_LIBTORCH
