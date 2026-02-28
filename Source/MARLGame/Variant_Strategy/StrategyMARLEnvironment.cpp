#include "StrategyMARLEnvironment.h"

AStrategyMARLEnvironment::AStrategyMARLEnvironment()
{
}

void AStrategyMARLEnvironment::BeginPlay()
{
	Super::BeginPlay();
}

void AStrategyMARLEnvironment::ResetEpisode()
{
	Super::ResetEpisode();
	CurrentStep = 0;
}

void AStrategyMARLEnvironment::Step(const TMap<FString, FMARLAction>& Actions)
{
	Super::Step(Actions);
	CurrentStep++;

	CalculateHierarchicalRewards();
}

void AStrategyMARLEnvironment::CalculateHierarchicalRewards()
{
	// Placeholder for hierarchical reward logic
	// e.g., Commander gets team-wide win/loss, Units get proximity/efficiency
	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent)
		{
			Agent->AddReward(0.1f); // Global step reward
		}
	}
}

bool AStrategyMARLEnvironment::IsTerminal() const
{
	return CurrentStep >= MaxSteps;
}
