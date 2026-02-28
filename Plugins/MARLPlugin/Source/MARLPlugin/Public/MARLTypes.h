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

/** Wrapper for float arrays to bypass UnrealHeaderTool limitations with nested containers in TMaps */
USTRUCT(BlueprintType)
struct FMARLFloatArray
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	TArray<float> Values;

	FMARLFloatArray() {}
	FMARLFloatArray(const TArray<float>& InValues) : Values(InValues) {}
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

USTRUCT(BlueprintType)
struct FMARLActionSpace
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	EMARLActionSpace Type = EMARLActionSpace::Discrete;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	int32 Size = 0;
};

USTRUCT(BlueprintType)
struct FMARLResetConfig
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	bool bRandomizeSpawn = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	float SpawnRadius = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	int32 InitialEnemies = 5;
};

USTRUCT(BlueprintType)
struct FMARLTeamObservation
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	TArray<float> TeamMeanPosition;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL")
	int32 ActiveUnits = 0;
};
