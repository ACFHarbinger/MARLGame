#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MARLMultiInstanceManager.generated.h"

/**
 * Manages multiple headless UE server instances for parallel MARL training.
 */
UCLASS(BlueprintType, Blueprintable)
class MARLPLUGIN_API UMARLMultiInstanceManager : public UObject
{
	GENERATED_BODY()

public:
	UMARLMultiInstanceManager();

	/** Initializes the manager and spawns target number of instances */
	UFUNCTION(BlueprintCallable, Category = "MARL|MultiInstance")
	bool InitializeInstances(int32 NumInstances, const FString& MapName, int32 BasePort = 7000);

	/** Stops all managed instances */
	UFUNCTION(BlueprintCallable, Category = "MARL|MultiInstance")
	void StopAllInstances();

	/** Returns the number of currently active instances */
	UFUNCTION(BlueprintPure, Category = "MARL|MultiInstance")
	int32 GetActiveInstanceCount() const;

	virtual void BeginDestroy() override;

private:
	/** Path to the UE executable */
	FString EditorExecutablePath;

	/** Path to the project file */
	FString ProjectFilePath;

	/** List of active process handles */
	TArray<FProcHandle> ActiveProcessHandles;

	/** Flag to prevent multiple initialization attempts */
	bool bIsInitialized = false;

	/** Internal helper to spawn a single instance */
	bool SpawnInstance(const FString& MapName, int32 Port);
};
