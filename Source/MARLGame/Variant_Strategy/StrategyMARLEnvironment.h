#pragma once

#include "CoreMinimal.h"
#include "MARLEnvironmentManager.h"
#include "StrategyMARLEnvironment.generated.h"

/**
 * Environment manager for hierarchical strategy matches.
 */
UCLASS()
class MARLGAME_API AStrategyMARLEnvironment : public AMARLEnvironmentManager
{
	GENERATED_BODY()

public:
	AStrategyMARLEnvironment();

	virtual void ResetEpisode() override;
	virtual void Step(const TMap<FString, FMARLAction>& Actions) override;
	virtual bool IsTerminal() const override;

protected:
	virtual void BeginPlay() override;

private:
	void CalculateHierarchicalRewards();

	UPROPERTY(EditAnywhere, Category = "MARL")
	int32 MaxSteps = 10000;

	int32 CurrentStep = 0;
};
