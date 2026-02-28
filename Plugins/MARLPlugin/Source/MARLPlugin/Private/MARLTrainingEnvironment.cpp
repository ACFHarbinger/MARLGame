#include "MARLTrainingEnvironment.h"
#include "MARLEnvironmentManager.h"
#include "MARLAgentComponent.h"

UMARLTrainingEnvironment::UMARLTrainingEnvironment()
	: NumAgents(0)
	, ObservationSize(0)
	, ActionSize(0)
	, EpisodeSteps(0)
	, TotalEpisodes(0)
	, MaxEpisodeSteps(1000)
{
}

bool UMARLTrainingEnvironment::Initialize(AMARLEnvironmentManager* InEnvironmentManager, int32 InNumAgents, int32 InObservationSize, int32 InActionSize)
{
	if (!InEnvironmentManager)
	{
		UE_LOG(LogTemp, Error, TEXT("MARLTrainingEnvironment: EnvironmentManager is null"));
		return false;
	}

	EnvironmentManager = InEnvironmentManager;
	NumAgents = InNumAgents;
	ObservationSize = InObservationSize;
	ActionSize = InActionSize;

	UE_LOG(LogTemp, Log, TEXT("MARLTrainingEnvironment initialized: %d agents, obs_size=%d, action_size=%d"),
	       NumAgents, ObservationSize, ActionSize);

	return true;
}

TMap<FString, FMARLFloatArray> UMARLTrainingEnvironment::Reset()
{
	if (!EnvironmentManager)
	{
		return TMap<FString, FMARLFloatArray>();
	}

	// Reset environment manager
	EnvironmentManager->ResetEnvironment();

	// Reset episode counters
	EpisodeSteps = 0;
	TotalEpisodes++;

	// Clear done flags
	AgentDoneFlags.Empty();
	AgentTruncatedFlags.Empty();

	// Collect observations from all agents
	TMap<FString, FMARLFloatArray> Observations;
	const TArray<UMARLAgentComponent*>& Agents = EnvironmentManager->GetAgents();

	for (UMARLAgentComponent* Agent : Agents)
	{
		if (Agent)
		{
			FString AgentID = Agent->GetAgentID();
			TArray<float> Obs = Agent->CollectObservations();
			Observations.Add(AgentID, FMARLFloatArray(Obs));

			// Initialize done flags
			AgentDoneFlags.Add(AgentID, false);
			AgentTruncatedFlags.Add(AgentID, false);
		}
	}

	UE_LOG(LogTemp, Log, TEXT("Environment reset. Episode: %d, Agents: %d"), TotalEpisodes, Observations.Num());

	return Observations;
}

bool UMARLTrainingEnvironment::Step(const TMap<FString, FMARLFloatArray>& Actions,
                                     TMap<FString, FMARLFloatArray>& OutObservations,
                                     TMap<FString, float>& OutRewards,
                                     TMap<FString, bool>& OutDones,
                                     TMap<FString, bool>& OutTruncated)
{
	if (!EnvironmentManager)
	{
		return false;
	}

	EpisodeSteps++;

	// Execute actions for all agents
	const TArray<UMARLAgentComponent*>& Agents = EnvironmentManager->GetAgents();

	for (UMARLAgentComponent* Agent : Agents)
	{
		if (Agent)
		{
			FString AgentID = Agent->GetAgentID();

			if (const FMARLFloatArray* ActionPtr = Actions.Find(AgentID))
			{
				// Execute action
				FMARLAction ActionRecord;
				ActionRecord.ContinuousActions = ActionPtr->Values;
				Agent->ExecuteAction(ActionRecord);
			}
		}
	}

	// Step the environment (physics, AI, etc.)
	EnvironmentManager->StepEnvironment(GetWorld()->GetDeltaSeconds());

	// Collect new observations and rewards
	OutObservations.Empty();
	OutRewards.Empty();
	OutDones.Empty();
	OutTruncated.Empty();

	for (UMARLAgentComponent* Agent : Agents)
	{
		if (Agent)
		{
			FString AgentID = Agent->GetAgentID();

			// Collect observation
			TArray<float> Obs = Agent->CollectObservations();
			OutObservations.Add(AgentID, FMARLFloatArray(Obs));

			// Get reward
			float Reward = Agent->GetReward();
			OutRewards.Add(AgentID, Reward);

			// Check if agent is done
			bool bDone = Agent->IsTerminal();
			OutDones.Add(AgentID, bDone);
			AgentDoneFlags.FindOrAdd(AgentID) = bDone;

			// Check truncation (episode timeout)
			bool bTruncated = (EpisodeSteps >= MaxEpisodeSteps);
			OutTruncated.Add(AgentID, bTruncated);
			AgentTruncatedFlags.FindOrAdd(AgentID) = bTruncated;
		}
	}

	// Check if episode is done (all agents done or truncated)
	bool bAllDone = AreAllAgentsDone();

	return true;
}

bool UMARLTrainingEnvironment::AreAllAgentsDone() const
{
	if (EpisodeSteps >= MaxEpisodeSteps)
	{
		return true;
	}

	for (const auto& Pair : AgentDoneFlags)
	{
		if (!Pair.Value)
		{
			return false;
		}
	}

	return AgentDoneFlags.Num() > 0;
}

#if WITH_LIBTORCH

torch::Tensor UMARLTrainingEnvironment::ResetTorch()
{
	TMap<FString, FMARLFloatArray> Observations = Reset();
	return ConvertObservationsToTensor(Observations);
}

bool UMARLTrainingEnvironment::StepTorch(const torch::Tensor& ActionTensor,
                                          torch::Tensor& OutObservationTensor,
                                          torch::Tensor& OutRewardTensor,
                                          torch::Tensor& OutDoneTensor,
                                          torch::Tensor& OutTruncatedTensor)
{
	// Convert tensor to UE actions
	TMap<FString, FMARLFloatArray> Actions = ConvertTensorToActions(ActionTensor);

	// Execute step
	TMap<FString, FMARLFloatArray> Observations;
	TMap<FString, float> Rewards;
	TMap<FString, bool> Dones;
	TMap<FString, bool> Truncated;

	bool bSuccess = Step(Actions, Observations, Rewards, Dones, Truncated);

	if (!bSuccess)
	{
		return false;
	}

	// Convert results back to tensors
	OutObservationTensor = ConvertObservationsToTensor(Observations);

	// Convert rewards to tensor
	std::vector<float> RewardVec;
	RewardVec.reserve(NumAgents);
	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		float Reward = Rewards.Contains(AgentID) ? Rewards[AgentID] : 0.0f;
		RewardVec.push_back(Reward);
	}
	OutRewardTensor = torch::from_blob(RewardVec.data(), {NumAgents}, torch::kFloat32).clone();

	// Convert dones to tensor
	std::vector<float> DoneVec;
	DoneVec.reserve(NumAgents);
	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		bool bDone = Dones.Contains(AgentID) ? Dones[AgentID] : false;
		DoneVec.push_back(bDone ? 1.0f : 0.0f);
	}
	OutDoneTensor = torch::from_blob(DoneVec.data(), {NumAgents}, torch::kFloat32).clone();

	// Convert truncated to tensor
	std::vector<float> TruncatedVec;
	TruncatedVec.reserve(NumAgents);
	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		bool bTruncated = Truncated.Contains(AgentID) ? Truncated[AgentID] : false;
		TruncatedVec.push_back(bTruncated ? 1.0f : 0.0f);
	}
	OutTruncatedTensor = torch::from_blob(TruncatedVec.data(), {NumAgents}, torch::kFloat32).clone();

	return true;
}

torch::Tensor UMARLTrainingEnvironment::ConvertObservationsToTensor(const TMap<FString, FMARLFloatArray>& Observations)
{
	// Create tensor of shape [num_agents, obs_size]
	std::vector<float> FlatObs;
	FlatObs.reserve(NumAgents * ObservationSize);

	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		if (const FMARLFloatArray* ObsPtr = Observations.Find(AgentID))
		{
			for (float Val : ObsPtr->Values)
			{
				FlatObs.push_back(Val);
			}
		}
		else
		{
			// Fill with zeros if agent not found
			for (int32 j = 0; j < ObservationSize; j++)
			{
				FlatObs.push_back(0.0f);
			}
		}
	}

	return torch::from_blob(FlatObs.data(), {NumAgents, ObservationSize}, torch::kFloat32).clone();
}

torch::Tensor UMARLTrainingEnvironment::ConvertActionsToTensor(const TMap<FString, FMARLFloatArray>& Actions)
{
	std::vector<float> FlatActions;
	FlatActions.reserve(NumAgents * ActionSize);

	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		if (const FMARLFloatArray* ActionPtr = Actions.Find(AgentID))
		{
			for (float Val : ActionPtr->Values)
			{
				FlatActions.push_back(Val);
			}
		}
		else
		{
			for (int32 j = 0; j < ActionSize; j++)
			{
				FlatActions.push_back(0.0f);
			}
		}
	}

	return torch::from_blob(FlatActions.data(), {NumAgents, ActionSize}, torch::kFloat32).clone();
}

TMap<FString, FMARLFloatArray> UMARLTrainingEnvironment::ConvertTensorToObservations(const torch::Tensor& Tensor)
{
	TMap<FString, FMARLFloatArray> Observations;

	// Ensure tensor is on CPU and contiguous
	torch::Tensor CpuTensor = Tensor.to(torch::kCPU).contiguous();
	float* Data = CpuTensor.data_ptr<float>();

	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		TArray<float> Obs;
		Obs.Reserve(ObservationSize);

		for (int32 j = 0; j < ObservationSize; j++)
		{
			Obs.Add(Data[i * ObservationSize + j]);
		}

		Observations.Add(AgentID, FMARLFloatArray(Obs));
	}

	return Observations;
}

TMap<FString, FMARLFloatArray> UMARLTrainingEnvironment::ConvertTensorToActions(const torch::Tensor& Tensor)
{
	TMap<FString, FMARLFloatArray> Actions;

	// Ensure tensor is on CPU and contiguous
	torch::Tensor CpuTensor = Tensor.to(torch::kCPU).contiguous();
	float* Data = CpuTensor.data_ptr<float>();

	for (int32 i = 0; i < NumAgents; i++)
	{
		FString AgentID = FString::Printf(TEXT("agent_%d"), i);
		TArray<float> Action;
		Action.Reserve(ActionSize);

		for (int32 j = 0; j < ActionSize; j++)
		{
			Action.Add(Data[i * ActionSize + j]);
		}

		Actions.Add(AgentID, FMARLFloatArray(Action));
	}

	return Actions;
}

#endif // WITH_LIBTORCH
