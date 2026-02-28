#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#if WITH_LIBTORCH
#include <torch/torch.h>
#endif

#include "MARLTrainingEnvironment.generated.h"

class AMARLEnvironmentManager;

/**
 * C++ Training Environment Wrapper for MARL
 * Provides a Gym-like interface for training multi-agent RL policies using LibTorch
 */
UCLASS()
class MARLPLUGIN_API UMARLTrainingEnvironment : public UObject
{
	GENERATED_BODY()

public:
	UMARLTrainingEnvironment();

	/**
	 * Initialize the training environment
	 * @param InEnvironmentManager Reference to the UE environment manager
	 * @param InNumAgents Number of agents in the environment
	 * @param InObservationSize Size of observation vector per agent
	 * @param InActionSize Size of action space per agent
	 * @return True if initialization succeeded
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	bool Initialize(AMARLEnvironmentManager* InEnvironmentManager, int32 InNumAgents, int32 InObservationSize, int32 InActionSize);

	/**
	 * Reset the environment to initial state
	 * @return Observations for all agents as TMap
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	TMap<FString, FMARLFloatArray> Reset();

	/**
	 * Execute one environment step with the given actions
	 * @param Actions Map of agent ID to action vectors
	 * @param OutObservations Next observations for all agents
	 * @param OutRewards Rewards for all agents
	 * @param OutDones Terminal flags for all agents
	 * @param OutTruncated Truncation flags for all agents
	 * @return True if step succeeded
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	bool Step(const TMap<FString, FMARLFloatArray>& Actions,
	          TMap<FString, FMARLFloatArray>& OutObservations,
	          TMap<FString, float>& OutRewards,
	          TMap<FString, bool>& OutDones,
	          TMap<FString, bool>& OutTruncated);

#if WITH_LIBTORCH
	/**
	 * LibTorch-compatible reset (returns torch tensors)
	 * @return Batch tensor of observations [num_agents, obs_size]
	 */
	torch::Tensor ResetTorch();

	/**
	 * LibTorch-compatible step
	 * @param ActionTensor Batch tensor of actions [num_agents, action_size]
	 * @param OutObservationTensor Next observations [num_agents, obs_size]
	 * @param OutRewardTensor Rewards [num_agents]
	 * @param OutDoneTensor Done flags [num_agents]
	 * @param OutTruncatedTensor Truncation flags [num_agents]
	 * @return True if step succeeded
	 */
	bool StepTorch(const torch::Tensor& ActionTensor,
	               torch::Tensor& OutObservationTensor,
	               torch::Tensor& OutRewardTensor,
	               torch::Tensor& OutDoneTensor,
	               torch::Tensor& OutTruncatedTensor);

	/**
	 * Get observation space dimensions
	 */
	std::vector<int64_t> GetObservationShape() const { return {NumAgents, ObservationSize}; }

	/**
	 * Get action space dimensions
	 */
	std::vector<int64_t> GetActionShape() const { return {NumAgents, ActionSize}; }
#endif

	/**
	 * Check if all agents are done
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	bool AreAllAgentsDone() const;

	/**
	 * Get current episode step count
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	int32 GetEpisodeStepCount() const { return EpisodeSteps; }

	/**
	 * Get total episode count
	 */
	UFUNCTION(BlueprintCallable, Category="MARL|Training")
	int32 GetTotalEpisodes() const { return TotalEpisodes; }

protected:
	UPROPERTY()
	TObjectPtr<AMARLEnvironmentManager> EnvironmentManager;

	UPROPERTY()
	int32 NumAgents;

	UPROPERTY()
	int32 ObservationSize;

	UPROPERTY()
	int32 ActionSize;

	UPROPERTY()
	int32 EpisodeSteps;

	UPROPERTY()
	int32 TotalEpisodes;

	UPROPERTY()
	int32 MaxEpisodeSteps;

	// Agent state tracking
	TMap<FString, bool> AgentDoneFlags;
	TMap<FString, bool> AgentTruncatedFlags;

private:
#if WITH_LIBTORCH
	/**
	 * Convert UE TMap observations to torch tensor
	 */
	torch::Tensor ConvertObservationsToTensor(const TMap<FString, FMARLFloatArray>& Observations);

	/**
	 * Convert UE TMap actions to torch tensor
	 */
	torch::Tensor ConvertActionsToTensor(const TMap<FString, FMARLFloatArray>& Actions);

	/**
	 * Convert torch tensor to UE TMap observations
	 */
	TMap<FString, FMARLFloatArray> ConvertTensorToObservations(const torch::Tensor& Tensor);

	/**
	 * Convert torch tensor to UE TMap actions
	 */
	TMap<FString, FMARLFloatArray> ConvertTensorToActions(const torch::Tensor& Tensor);
#endif
};
