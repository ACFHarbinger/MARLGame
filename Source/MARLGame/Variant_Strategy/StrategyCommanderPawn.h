#pragma once

#include "CoreMinimal.h"
#include "StrategyPawn.h"
#include "MARLAgentInterface.h"
#include "StrategyCommanderPawn.generated.h"

class UMARLAgentComponent;

/**
 * Hierarchical commander agent for Strategy matches.
 */
UCLASS()
class MARLGAME_API AStrategyCommanderPawn : public AStrategyPawn, public IMARLAgentInterface
{
	GENERATED_BODY()

public:
	AStrategyCommanderPawn();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MARL", meta = (AllowPrivateAccess = "true"))
	UMARLAgentComponent* AgentComponent;

	/** IMARLAgentInterface Implementation */
	virtual void ApplyAction_Implementation(const FMARLAction& Action) override;

protected:
	virtual void BeginPlay() override;

	UFUNCTION()
	void HandleMARLAction(const FMARLAction& Action);
};
