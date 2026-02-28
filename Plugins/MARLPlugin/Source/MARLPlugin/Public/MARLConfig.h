#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "MARLTypes.h"
#include "MARLConfig.generated.h"

UCLASS(BlueprintType)
class MARLPLUGIN_API UMARLConfig : public UPrimaryDataAsset
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="MARL|Config")
	int32 NumAgents = 4;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="MARL|Config")
	int32 MaxEpisodeSteps = 3000;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="MARL|Config")
	FMARLObservationSpace ObservationSpace;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="MARL|Config")
	FMARLActionSpace ActionSpace;

	UFUNCTION(BlueprintCallable, Category="MARL|Config")
	void LoadFromJson(const FString& JsonFilePath);
};
