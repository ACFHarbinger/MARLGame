#pragma once

#include "CoreMinimal.h"
#include "MARLTypes.generated.h"

UENUM(BlueprintType)
enum class EMARLObservationType : uint8
{
	Vector    UMETA(DisplayName = "Vector"),
	Image     UMETA(DisplayName = "Image"),
	Hybrid    UMETA(DisplayName = "Hybrid")
};

UENUM(BlueprintType)
enum class EMARLActionSpace : uint8
{
	Discrete       UMETA(DisplayName = "Discrete"),
	Continuous     UMETA(DisplayName = "Continuous"),
	MultiDiscrete  UMETA(DisplayName = "MultiDiscrete")
};

USTRUCT(BlueprintType)
struct FMARLAction
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	TArray<int32> DiscreteActions;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	TArray<float> ContinuousActions;

	FMARLAction() {}
};

USTRUCT(BlueprintType)
struct FMARLObservationSpace
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	EMARLObservationType Type = EMARLObservationType::Vector;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	int32 Size = 0;
};
