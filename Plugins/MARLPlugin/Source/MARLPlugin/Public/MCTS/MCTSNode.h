#pragma once

#include "CoreMinimal.h"
#include "MCTSState.h"

/**
 * A single node in the MCTS tree.
 */
class MARLPLUGIN_API FMCTSNode
{
public:
	FMCTSNode(TSharedPtr<IMCTSState> InState, FMCTSNode* InParent = nullptr, FMARLAction InAction = FMARLAction());
	~FMCTSNode();

	/** Expansion: Create child nodes for all legal actions */
	void Expand();

	/** Selection: Choose the best child using UCT */
	FMCTSNode* SelectChild(float ExplorationWeight);

	/** Backpropagation: Update node statistics */
	void Update(float Reward);

	/** Returns true if the node is fully expanded */
	bool IsFullyExpanded() const;

	/** Returns true if the node is a leaf (no children) */
	bool IsLeaf() const { return Children.Num() == 0; }

public:
	TSharedPtr<IMCTSState> State;
	FMCTSNode* Parent;
	FMARLAction Action; // Action that led to this node

	TArray<FMCTSNode*> Children;
	
	int32 Visits = 0;
	float TotalReward = 0.0f;

	TArray<FString> UntriedAgents;
	TMap<FString, TArray<FMARLAction>> UntriedActions;
};
