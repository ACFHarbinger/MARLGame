#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MARLEvaluationManager.generated.h"

/**
 * Statistics recorded for a single evaluation episode.
 */
USTRUCT(BlueprintType)
struct FMARLEpisodeStats
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="MARL|Evaluation")
	float TotalReward = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="MARL|Evaluation")
	float DurationSeconds = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="MARL|Evaluation")
	bool bWonMatch = false;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="MARL|Evaluation")
	int32 EnemiesDefeated = 0;
};

/**
 * Subsystem/Manager for collecting metrics during evaluation phases or human playtests.
 */
UCLASS(BlueprintType, Blueprintable)
class MARLPLUGIN_API UMARLEvaluationManager : public UObject
{
	GENERATED_BODY()

public:
	UMARLEvaluationManager();

	/** Start a new recording session */
	UFUNCTION(BlueprintCallable, Category = "MARL|Evaluation")
	void StartEvaluationSession(const FString& SessionName);

	/** Log an episode's results to the current session */
	UFUNCTION(BlueprintCallable, Category = "MARL|Evaluation")
	void RecordEpisodeResult(const FMARLEpisodeStats& Stats);

	/** Save the current session data to a CSV or JSON file */
	UFUNCTION(BlueprintCallable, Category = "MARL|Evaluation")
	void ExportSessionData() const;

	/** Get aggregate win rate */
	UFUNCTION(BlueprintPure, Category = "MARL|Evaluation")
	float GetWinRate() const;

	/** Get average reward across the session */
	UFUNCTION(BlueprintPure, Category = "MARL|Evaluation")
	float GetAverageReward() const;

private:
	FString CurrentSessionName;
	TArray<FMARLEpisodeStats> SessionHistory;
};
