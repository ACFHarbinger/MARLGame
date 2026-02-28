#include "MARLEvaluationManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformTime.h"

UMARLEvaluationManager::UMARLEvaluationManager()
{
}

void UMARLEvaluationManager::StartEvaluationSession(const FString& SessionName)
{
	CurrentSessionName = SessionName;
	SessionHistory.Empty();
	UE_LOG(LogTemp, Log, TEXT("Started MARL Evaluation Session: %s"), *SessionName);
}

void UMARLEvaluationManager::RecordEpisodeResult(const FMARLEpisodeStats& Stats)
{
	SessionHistory.Add(Stats);
}

void UMARLEvaluationManager::ExportSessionData() const
{
	if (SessionHistory.Num() == 0 || CurrentSessionName.IsEmpty()) return;

	FString CsvData = TEXT("Episode,TotalReward,DurationSeconds,WonMatch,EnemiesDefeated\n");

	for (int32 i = 0; i < SessionHistory.Num(); ++i)
	{
		const FMARLEpisodeStats& S = SessionHistory[i];
		CsvData += FString::Printf(TEXT("%d,%f,%f,%d,%d\n"),
			i + 1, S.TotalReward, S.DurationSeconds, S.bWonMatch ? 1 : 0, S.EnemiesDefeated);
	}

	FString FilePath = FPaths::ProjectSavedDir() / FString::Printf(TEXT("MARLEval_%s_%f.csv"), *CurrentSessionName, FPlatformTime::Seconds());

	if (FFileHelper::SaveStringToFile(CsvData, *FilePath))
	{
		UE_LOG(LogTemp, Log, TEXT("Successfully exported MARL evaluation data to %s"), *FilePath);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to export MARL evaluation data to %s"), *FilePath);
	}
}

float UMARLEvaluationManager::GetWinRate() const
{
	if (SessionHistory.Num() == 0) return 0.0f;

	int32 Wins = 0;
	for (const FMARLEpisodeStats& S : SessionHistory)
	{
		if (S.bWonMatch) Wins++;
	}

	return (float)Wins / SessionHistory.Num();
}

float UMARLEvaluationManager::GetAverageReward() const
{
	if (SessionHistory.Num() == 0) return 0.0f;

	float Total = 0.0f;
	for (const FMARLEpisodeStats& S : SessionHistory)
	{
		Total += S.TotalReward;
	}

	return Total / SessionHistory.Num();
}
