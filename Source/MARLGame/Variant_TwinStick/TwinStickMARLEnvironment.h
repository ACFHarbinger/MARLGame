#pragma once

#include "CoreMinimal.h"
#include "MARLEnvironmentManager.h"
#include "TwinStickMARLEnvironment.generated.h"

/**
 * Specialized environment manager for the Twin Stick game variant.
 */
UCLASS()
class MARLGAME_API ATwinStickMARLEnvironment : public AMARLEnvironmentManager
{
	GENERATED_BODY()

public:
	ATwinStickMARLEnvironment();

	virtual void ResetEpisode() override;
	virtual void Step(const TMap<FString, FMARLAction>& Actions) override;
	virtual bool IsTerminal() const override;

	UFUNCTION()
	void HandleNPCDeath(AActor* DeadNPC);

protected:
	virtual void BeginPlay() override;

private:
	void CalculateAndDistributeRewards();
	void RandomizeEnvironment();

	UPROPERTY(EditAnywhere, Category = "MARL")
	FMARLResetConfig ResetConfig;

	UPROPERTY(EditAnywhere, Category = "MARL")
	int32 MaxSteps = 3000;

	int32 CurrentStep = 0;
};
