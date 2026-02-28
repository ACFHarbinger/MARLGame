#include "MARLEnvironmentManager.h"
#include "MARLAgentComponent.h"

AMARLEnvironmentManager::AMARLEnvironmentManager()
{
	PrimaryActorTick.bCanEverTick = true;
}

void AMARLEnvironmentManager::BeginPlay()
{
	Super::BeginPlay();
}

void AMARLEnvironmentManager::RegisterAgent(UMARLAgentComponent* Agent)
{
	if (Agent && !RegisteredAgents.Contains(Agent))
	{
		RegisteredAgents.Add(Agent);
	}
}

void AMARLEnvironmentManager::UnregisterAgent(UMARLAgentComponent* Agent)
{
	RegisteredAgents.Remove(Agent);
}

void AMARLEnvironmentManager::ResetEpisode()
{
	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent)
		{
			Agent->ResetAgent();
		}
	}
}

void AMARLEnvironmentManager::Step(const TMap<FString, FMARLAction>& Actions)
{
	for (UMARLAgentComponent* Agent : RegisteredAgents)
	{
		if (Agent && Actions.Contains(Agent->AgentID))
		{
			Agent->ExecuteAction(Actions[Agent->AgentID]);
		}
	}
}

bool AMARLEnvironmentManager::IsTerminal() const
{
	// Base implementation, override in derived classes
	return false;
}
