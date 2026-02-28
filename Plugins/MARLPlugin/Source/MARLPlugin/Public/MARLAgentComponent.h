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

	/** Returns the accumulated reward and resets it if requested */
	UFUNCTION(BlueprintCallable, Category="MARL")
	float ConsumeReward();

	UFUNCTION(BlueprintCallable, Category="MARL")
	void AddReward(float Reward);

	/** Gets the last message this agent broadcasted to its team */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="MARL|Communication")
	TArray<float> OutboundMessage;

	/** Sets the incoming messages from teammates for the current step */
	UFUNCTION(BlueprintCallable, Category="MARL|Communication")
	void SetIncomingMessages(const TArray<float>& Messages);

	/** Gets the concatenated incoming messages from teammates */
	UFUNCTION(BlueprintPure, Category="MARL|Communication")
	TArray<float> GetIncomingMessages() const;

	UFUNCTION(BlueprintCallable, Category="MARL")
	void ResetAgent();

protected:
	virtual void BeginPlay() override;

private:
	// Current episodic reward
	float CurrentReward = 0.0f;

	// Incoming messages from teammates
	TArray<float> IncomingMessages;

	UPROPERTY(Transient)
	TArray<UMARLSensorComponent*> Sensors;
};
