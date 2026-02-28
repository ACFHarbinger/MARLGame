#include "MARLCharacter.h"
#include "MARLAgentComponent.h"

AMARLCharacter::AMARLCharacter()
{
	AgentComponent = CreateDefaultSubobject<UMARLAgentComponent>(TEXT("MARLAgentComponent"));
}

void AMARLCharacter::BeginPlay()
{
	Super::BeginPlay();

	if (AgentComponent)
	{
		AgentComponent->OnActionReceived.AddDynamic(this, &AMARLCharacter::HandleMARLAction);
	}
}

void AMARLCharacter::HandleMARLAction(const FMARLAction& Action)
{
	ApplyAction(Action);
}

void AMARLCharacter::ApplyAction_Implementation(const FMARLAction& Action)
{
	// Default implementation (can be overridden in BP or C++)
}
