#include "StrategyCommanderPawn.h"
#include "MARLAgentComponent.h"

AStrategyCommanderPawn::AStrategyCommanderPawn()
{
	AgentComponent = CreateDefaultSubobject<UMARLAgentComponent>(TEXT("MARLAgentComponent"));
}

void AStrategyCommanderPawn::BeginPlay()
{
	Super::BeginPlay();

	if (AgentComponent)
	{
		AgentComponent->OnActionReceived.AddDynamic(this, &AStrategyCommanderPawn::HandleMARLAction);
	}
}

void AStrategyCommanderPawn::HandleMARLAction(const FMARLAction& Action)
{
	ApplyAction(Action);
}

void AStrategyCommanderPawn::ApplyAction_Implementation(const FMARLAction& Action)
{
	// Commander high-level logic (e.g., assigning goals to sub-agents)
}
