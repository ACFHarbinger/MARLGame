#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MARLTypes.h"
#include "MARLAgentComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActionReceived, const FMARLAction&, Action);

class UMARLSensorComponent;

UCLASS(ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UMARLAgentComponent();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	FString AgentID;

	UPROPERTY(EditAnywhere, Category="MARL")
	int32 ObservationSize = 128;

	UPROPERTY(EditAnywhere, Category="MARL")
	int32 ActionSize = 24;

	UPROPERTY(BlueprintAssignable, Category="MARL")
	FOnActionReceived OnActionReceived;

	UFUNCTION(BlueprintCallable, Category="MARL")
	TArray<float> CollectObservations();

	UFUNCTION(BlueprintCallable, Category="MARL")
	void ExecuteAction(const FMARLAction& Action);

	UFUNCTION(BlueprintCallable, Category="MARL")
	float GetAccumulatedReward() const { return AccumulatedReward; }

	UFUNCTION(BlueprintCallable, Category="MARL")
	void AddReward(float Reward);

	UFUNCTION(BlueprintCallable, Category="MARL")
	void ResetAgent();

protected:
	virtual void BeginPlay() override;

private:
	float AccumulatedReward = 0.0f;

	UPROPERTY(Transient)
	TArray<UMARLSensorComponent*> Sensors;
};
