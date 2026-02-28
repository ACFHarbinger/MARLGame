#include "MARLSelfPlayManager.h"
#include "Math/UnrealMathUtility.h"

UMARLSelfPlayManager::UMARLSelfPlayManager()
{
}

void UMARLSelfPlayManager::AddHistoricalPolicy(const FString& ModelFilePath, float InitialElo)
{
	FMARLHistoricalPolicy NewPolicy;
	NewPolicy.ModelFilePath = ModelFilePath;
	NewPolicy.EloRating = InitialElo;
	NewPolicy.MatchesPlayed = 0;

	PolicyPool.Add(NewPolicy);
	UE_LOG(LogTemp, Log, TEXT("Added new historical policy to pool: %s (Initial ELO: %f)"), *ModelFilePath, InitialElo);
}

FMARLHistoricalPolicy UMARLSelfPlayManager::SampleOpponent(float CurrentAgentElo) const
{
	if (PolicyPool.Num() == 0)
	{
		return FMARLHistoricalPolicy();
	}

	// Simple random sampling for now. Could be enhanced to prioritize opponents with similar ELO.
	int32 RandomIndex = FMath::RandRange(0, PolicyPool.Num() - 1);
	return PolicyPool[RandomIndex];
}

void UMARLSelfPlayManager::UpdateEloRatings(const FString& OpponentFilePath, bool bMainAgentWon, float& InOutMainAgentElo)
{
	float KFactor = 32.0f; // Standard ELO K-factor

	for (FMARLHistoricalPolicy& Opponent : PolicyPool)
	{
		if (Opponent.ModelFilePath == OpponentFilePath)
		{
			// Calculate expected score for Main Agent
			float ExpectedScoreMain = 1.0f / (1.0f + FMath::Pow(10.0f, (Opponent.EloRating - InOutMainAgentElo) / 400.0f));
			float ExpectedScoreOpp = 1.0f - ExpectedScoreMain;

			float ActualScoreMain = bMainAgentWon ? 1.0f : 0.0f;
			float ActualScoreOpp = 1.0f - ActualScoreMain;

			// Update Elos
			InOutMainAgentElo = InOutMainAgentElo + KFactor * (ActualScoreMain - ExpectedScoreMain);
			Opponent.EloRating = Opponent.EloRating + KFactor * (ActualScoreOpp - ExpectedScoreOpp);
			Opponent.MatchesPlayed++;

			UE_LOG(LogTemp, Log, TEXT("Updated ELOs: Main Agent = %.1f, Opponent = %.1f"), InOutMainAgentElo, Opponent.EloRating);
			break;
		}
	}
}
