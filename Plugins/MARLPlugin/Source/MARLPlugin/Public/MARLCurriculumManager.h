#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MARLCurriculumManager.generated.h"

/**
 * Defines a single stage in the learning curriculum, adjusting environment difficulty.
 */
USTRUCT(BlueprintType)
struct FMARLCurriculumStage
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Curriculum")
	int32 StageIndex = 0;

	/** Number of agents in this stage */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Curriculum")
	int32 NumAgents = 1;

	/** Number of opposing NPCs or complexity level */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Curriculum")
	int32 NumAdversaries = 0;

	/** Should the spawns be randomized? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Curriculum")
	bool bRandomizeSpawns = false;

	/** Success rate required to pass to the next stage (0.0 to 1.0) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Curriculum")
	float RequiredSuccessRate = 0.8f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnCurriculumStageUpdatedDelegate, const FMARLCurriculumStage&, NewStage);

/**
 * Manages the difficulty progression of the training environment.
 */
UCLASS(BlueprintType, Blueprintable)
class MARLPLUGIN_API UMARLCurriculumManager : public UObject
{
	GENERATED_BODY()

public:
	UMARLCurriculumManager();

	/** Initializes the curriculum with a set of stages */
	UFUNCTION(BlueprintCallable, Category = "MARL|Curriculum")
	void InitializeCurriculum(const TArray<FMARLCurriculumStage>& InStages);

	/** Needs to be called periodically (e.g., end of episode) to track progress */
	UFUNCTION(BlueprintCallable, Category = "MARL|Curriculum")
	void UpdateProgress(bool bEpisodeSuccess, float EpisodeReward);

	/** Gets the current curriculum stage */
	UFUNCTION(BlueprintPure, Category = "MARL|Curriculum")
	FMARLCurriculumStage GetCurrentStage() const;

	/** Event fired when the curriculum stage advances */
	UPROPERTY(BlueprintAssignable, Category = "MARL|Curriculum")
	FOnCurriculumStageUpdatedDelegate OnStageUpdated;

private:
	UPROPERTY()
	TArray<FMARLCurriculumStage> Stages;

	int32 CurrentStageIndex = 0;

	// Progress tracking
	TArray<bool> RecentSuccesses;
	int32 SuccessWindowSize = 100;

	bool CheckProgression();
};
