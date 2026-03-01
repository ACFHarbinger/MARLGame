#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MCTSNode.h"
#include "MCTSState.h"
#include "MCTSManager.generated.h"

/**
 * Manager class for running MCTS simulations.
 */
UCLASS(BlueprintType)
class MARLPLUGIN_API UMCTSManager : public UObject
{
	GENERATED_BODY()

public:
	UMCTSManager();

	/** Runs MCTS search from the given state for a fixed number of iterations */
	UFUNCTION(BlueprintCallable, Category="MARL|MCTS")
	FMARLAction RunSearch(TScriptInterface<IMCTSState> InitialState, int32 Iterations, float ExplorationWeight = 1.414f);

	/** Runs Information Set MCTS (ISMCTS) for imperfect information scenarios */
	UFUNCTION(BlueprintCallable, Category="MARL|MCTS")
	FMARLAction RunISMCTS(TScriptInterface<IMCTSState> InitialState, const FString& ObserverID, int32 Iterations, float ExplorationWeight = 1.414f);

private:
	/** Helper to convert raw interface to SharedPtr for the tree */
	TSharedPtr<IMCTSState> WrapState(TScriptInterface<IMCTSState> StateInterface);

	/** Default simulation (random rollout) */
	float Simulate(TSharedPtr<IMCTSState> State);
};
