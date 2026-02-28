#include "MARLAgentComponent.h"
#include "MARLSensorComponent.h"
#include "GameFramework/Actor.h"

UMARLAgentComponent::UMARLAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UMARLAgentComponent::BeginPlay()
{
	Super::BeginPlay();

	// Find all child sensor components
	if (AActor* Owner = GetOwner())
	{
		Owner->GetComponents(Sensors);
	}
}

TArray<float> UMARLAgentComponent::CollectObservations()
{
	TArray<float> CombinedObservations;

	// E.g., combine observations from multiple sensors
	for (UMARLSensorComponent* Sensor : Sensors)
	{
		if (Sensor)
		{
			CombinedObservations.Append(Sensor->GetObservations());
		}
	}

	return CombinedObservations;
}

void UMARLAgentComponent::ExecuteAction(const FMARLAction& Action)
{
	OnActionReceived.Broadcast(Action);
}

void UMARLAgentComponent::AddReward(float Reward)
{
	CurrentReward += Reward;
}

float UMARLAgentComponent::ConsumeReward()
{
	float R = CurrentReward;
	CurrentReward = 0.0f;
	return R;
}

void UMARLAgentComponent::SetIncomingMessages(const TArray<float>& Messages)
{
	IncomingMessages = Messages;
}

TArray<float> UMARLAgentComponent::GetIncomingMessages() const
{
	return IncomingMessages;
}

void UMARLAgentComponent::ResetAgent()
{
	CurrentReward = 0.0f;
	bIsTerminal = false;
}
