#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MARLTypes.h"
#include "MARLEnvironmentManager.generated.h"

class UMARLAgentComponent;

UCLASS()
class MARLPLUGIN_API AMARLEnvironmentManager : public AActor
{
	GENERATED_BODY()

public:
	AMARLEnvironmentManager();

	UFUNCTION(BlueprintCallable, Category="MARL")
	void RegisterAgent(UMARLAgentComponent* Agent);

	UFUNCTION(BlueprintCallable, Category="MARL")
	void UnregisterAgent(UMARLAgentComponent* Agent);

	UFUNCTION(BlueprintCallable, Category="MARL")
	virtual void ResetEpisode();

	UFUNCTION(BlueprintCallable, Category="MARL")
	virtual void ResetEnvironment() { ResetEpisode(); }

	UFUNCTION(BlueprintCallable, Category="MARL")
	virtual void Step(const TMap<FString, FMARLAction>& Actions);

	UFUNCTION(BlueprintCallable, Category="MARL")
	virtual void StepEnvironment(float DeltaSeconds) {}

	UFUNCTION(BlueprintCallable, Category="MARL")
	virtual bool IsTerminal() const;

	UFUNCTION(BlueprintPure, Category="MARL")
	const TArray<UMARLAgentComponent*>& GetAgents() const { return RegisteredAgents; }

protected:
	virtual void BeginPlay() override;

	UPROPERTY(Transient)
	TArray<UMARLAgentComponent*> RegisteredAgents;
};
