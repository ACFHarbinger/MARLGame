#pragma once

#include "CoreMinimal.h"
#include "MARLTypes.h"
#include "UObject/Interface.h"
#include "MCTSState.generated.h"

UINTERFACE(MinimalAPI)
class UMCTSState : public UInterface
{
	GENERATED_BODY()
};

/**
 * Interface for game states that can be used with MCTS.
 * Must support cloning and stepping.
 */
class MARLPLUGIN_API IMCTSState
{
	GENERATED_BODY()

public:
	/** Returns a deep copy of the current state */
	virtual TSharedPtr<IMCTSState> Clone() const = 0;

	/** Executes an action and updates the state. Returns the reward for this step. */
	virtual float Step(const TMap<FString, FMARLAction>& Actions) = 0;

	/** Returns true if the state is terminal */
	virtual bool IsTerminal() const = 0;

	/** Returns the list of possible actions for the current player/agent */
	virtual TArray<FMARLAction> GetLegalActions(const FString& AgentID) const = 0;

	/** Returns the list of agents active in this state */
	virtual TArray<FString> GetAgentIDs() const = 0;

	/** ISMCTS: Returns a version of the state sampled from the observer's belief set */
	virtual TSharedPtr<IMCTSState> GetRandomInterpretation(const FString& ObserverID) const { return Clone(); }

	// UObject Mapping (for Blueprint support)
	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	UObject* CloneUObject() const;

	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	float StepEnvironment(const TMap<FString, FMARLAction>& Actions);

	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	bool IsTerminalState() const;

	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	TArray<FMARLAction> GetLegalActionsForAgent(const FString& AgentID) const;

	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	TArray<FString> GetActiveAgentIDs() const;

	UFUNCTION(BlueprintNativeEvent, Category="MARL|MCTS")
	UObject* SampleInterpretation(const FString& ObserverID) const;
};
