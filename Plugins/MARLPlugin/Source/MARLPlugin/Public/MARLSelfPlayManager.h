#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MARLSelfPlayManager.generated.h"

/**
 * Tracks performance and ratings for historical policies to facilitate league training.
 */
USTRUCT(BlueprintType)
struct FMARLHistoricalPolicy
{
	GENERATED_BODY()

	/** Absolute path to the TorchScript (.pt) model file */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|SelfPlay")
	FString ModelFilePath;

	/** Elo rating of this historical policy */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|SelfPlay")
	float EloRating = 1200.0f;

	/** Number of matches played */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|SelfPlay")
	int32 MatchesPlayed = 0;
};

/**
 * Manages the self-play pool, loading historical models for adversary agents.
 */
UCLASS(BlueprintType, Blueprintable)
class MARLPLUGIN_API UMARLSelfPlayManager : public UObject
{
	GENERATED_BODY()

public:
	UMARLSelfPlayManager();

	/** Add a newly saved model to the pool */
	UFUNCTION(BlueprintCallable, Category = "MARL|SelfPlay")
	void AddHistoricalPolicy(const FString& ModelFilePath, float InitialElo = 1200.0f);

	/** Samples an opponent profile, optionally prioritizing similar skill (matchmaking) */
	UFUNCTION(BlueprintCallable, Category = "MARL|SelfPlay")
	FMARLHistoricalPolicy SampleOpponent(float CurrentAgentElo) const;

	/** Update ELO ratings after a match */
	UFUNCTION(BlueprintCallable, Category = "MARL|SelfPlay")
	void UpdateEloRatings(const FString& OpponentFilePath, bool bMainAgentWon, float& InOutMainAgentElo);

private:
	UPROPERTY()
	TArray<FMARLHistoricalPolicy> PolicyPool;
};
