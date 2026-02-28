#include "MARLCurriculumManager.h"

UMARLCurriculumManager::UMARLCurriculumManager()
{
}

void UMARLCurriculumManager::InitializeCurriculum(const TArray<FMARLCurriculumStage>& InStages)
{
	Stages = InStages;
	// Sort by stage index just in case
	Stages.Sort([](const FMARLCurriculumStage& A, const FMARLCurriculumStage& B) {
		return A.StageIndex < B.StageIndex;
	});

	CurrentStageIndex = 0;
	RecentSuccesses.Empty();

	if (Stages.Num() > 0)
	{
		UE_LOG(LogTemp, Log, TEXT("Initialized MARL Curriculum with %d stages. Starting at stage %d."), Stages.Num(), Stages[0].StageIndex);
		OnStageUpdated.Broadcast(Stages[0]);
	}
}

void UMARLCurriculumManager::UpdateProgress(bool bEpisodeSuccess, float EpisodeReward)
{
	if (Stages.Num() == 0 || CurrentStageIndex >= Stages.Num() - 1)
	{
		return; // We have finished all stages
	}

	RecentSuccesses.Add(bEpisodeSuccess);
	if (RecentSuccesses.Num() > SuccessWindowSize)
	{
		RecentSuccesses.RemoveAt(0); // Pop oldest
	}

	if (RecentSuccesses.Num() >= SuccessWindowSize)
	{
		if (CheckProgression())
		{
			CurrentStageIndex++;
			RecentSuccesses.Empty();
			
			UE_LOG(LogTemp, Log, TEXT("MARL Curriculum Advanced to Stage %d"), Stages[CurrentStageIndex].StageIndex);
			OnStageUpdated.Broadcast(Stages[CurrentStageIndex]);
		}
	}
}

bool UMARLCurriculumManager::CheckProgression()
{
	if (Stages.IsValidIndex(CurrentStageIndex))
	{
		int32 SuccessCount = 0;
		for (bool bMatchWon : RecentSuccesses)
		{
			if (bMatchWon) SuccessCount++;
		}

		float CurrentSuccessRate = (float)SuccessCount / RecentSuccesses.Num();
		float ReqdRate = Stages[CurrentStageIndex].RequiredSuccessRate;

		return CurrentSuccessRate >= ReqdRate;
	}

	return false;
}

FMARLCurriculumStage UMARLCurriculumManager::GetCurrentStage() const
{
	if (Stages.IsValidIndex(CurrentStageIndex))
	{
		return Stages[CurrentStageIndex];
	}

	return FMARLCurriculumStage();
}
