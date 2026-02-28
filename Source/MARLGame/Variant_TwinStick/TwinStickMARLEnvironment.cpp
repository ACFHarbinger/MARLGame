#include "TwinStickMARLEnvironment.h"
#include "MARLAgentComponent.h"
#include "TwinStickGameMode.h"

ATwinStickMARLEnvironment::ATwinStickMARLEnvironment()
{
}

void ATwinStickMARLEnvironment::BeginPlay()
{
	Super::BeginPlay();
}

void ATwinStickMARLEnvironment::ResetEpisode()
{
	Super::ResetEpisode();
	CurrentStep = 0;

	if (ResetConfig.bRandomizeSpawn)
	{
		RandomizeEnvironment();
	}

	// Call GameMode reset if necessary
	if (ATwinStickGameMode* GM = Cast<ATwinStickGameMode>(GetWorld()->GetAuthGameMode()))
	{
		// GM->RestartGame();
	}
}

void ATwinStickMARLEnvironment::RandomizeEnvironment()
{
	UWorld* World = GetWorld();
	if (!World) return;

	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent && Agent->GetOwner())
		{
			FVector RandomOffset = FMath::VRand();
			RandomOffset.Z = 0.0f;
			FVector NewLoc = RandomOffset * ResetConfig.SpawnRadius;
			Agent->GetOwner()->SetActorLocation(NewLoc);
		}
	}
}

void ATwinStickMARLEnvironment::HandleNPCDeath(AActor* DeadNPC)
{
	// Award kill reward to team
	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent)
		{
			Agent->AddReward(50.0f); // Kill reward
		}
	}
}

void ATwinStickMARLEnvironment::Step(const TMap<FString, FMARLAction>& Actions)
{
	Super::Step(Actions);
	CurrentStep++;

	// Calculate and distribute rewards
	CalculateAndDistributeRewards();
}

void ATwinStickMARLEnvironment::CalculateAndDistributeRewards()
{
	float TotalTeamReward = 0.0f;

	// Example: Reward for survival, penalty for time
	TotalTeamReward += 0.1f; // Survival bonus per step

	// Check GameMode for kills or objectives
	if (ATwinStickGameMode* GM = Cast<ATwinStickGameMode>(GetWorld()->GetAuthGameMode()))
	{
		// Hypothetical: if (GM->RecentKills > 0) TotalTeamReward += 50.0f;
	}

	// Distribute to all agents
	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent)
		{
			// Shared reward + individual incentive could be added here
			Agent->AddReward(TotalTeamReward);
		}
	}
}

bool ATwinStickMARLEnvironment::IsTerminal() const
{
	if (CurrentStep >= MaxSteps)
	{
		return true;
	}

	// Check if all players are dead or objective met via GameMode
	return false;
}
