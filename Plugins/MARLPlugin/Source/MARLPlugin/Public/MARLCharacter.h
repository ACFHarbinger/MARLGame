#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "MARLTypes.h"
#include "MARLAgentInterface.h"
#include "MARLCharacter.generated.h"

class UMARLAgentComponent;

/**
 * Base class for characters controlled by Multi-Agent Reinforcement Learning.
 */
UCLASS()
class MARLPLUGIN_API AMARLCharacter : public ACharacter, public IMARLAgentInterface
{
	GENERATED_BODY()

public:
	AMARLCharacter();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MARL", meta = (AllowPrivateAccess = "true"))
	UMARLAgentComponent* AgentComponent;

	/** IMARLAgentInterface Implementation */
	virtual void ApplyAction_Implementation(const FMARLAction& Action) override;

protected:
	virtual void BeginPlay() override;

	UFUNCTION()
	virtual void HandleMARLAction(const FMARLAction& Action);
};
