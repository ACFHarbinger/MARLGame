#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MARLLibTorchInterface.generated.h"

UCLASS(ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLLibTorchInterface : public UActorComponent
{
	GENERATED_BODY()

public:
	UMARLLibTorchInterface();

	UFUNCTION(BlueprintCallable, Category="MARL|LibTorch")
	bool InitializeIPCServer(int32 Port);

	UFUNCTION(BlueprintCallable, Category="MARL|LibTorch")
	void ShutdownIPCServer();

	// Function to send observations via IPC
	UFUNCTION(BlueprintCallable, Category="MARL|LibTorch")
	void SendObservations(const TMap<FString, TArray<float>>& Observations);

	// Function to receive actions via IPC
	UFUNCTION(BlueprintCallable, Category="MARL|LibTorch")
	bool ReceiveActions(TMap<FString, TArray<float>>& OutActions);

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	bool bIsServerRunning = false;
	int32 CurrentPort = 9876;
};
