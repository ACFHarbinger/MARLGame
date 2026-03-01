#include "MCTS/MCTSManager.h"
#include "MARLTypes.h"
#include <algorithm>
#include <random>

UMCTSManager::UMCTSManager()
{
}

FMARLAction UMCTSManager::RunSearch(TScriptInterface<IMCTSState> InitialState, int32 Iterations, float ExplorationWeight)
{
	if (!InitialState) return FMARLAction();

	TSharedPtr<IMCTSState> RootState = WrapState(InitialState);
	if (!RootState.IsValid()) return FMARLAction();

	FMCTSNode* RootNode = new FMCTSNode(RootState);

	for (int32 i = 0; i < Iterations; ++i)
	{
		FMCTSNode* Node = RootNode;

		// 1. Selection
		while (!Node->IsLeaf() && Node->IsFullyExpanded())
		{
			Node = Node->SelectChild(ExplorationWeight);
		}

		// 2. Expansion
		if (!Node->IsFullyExpanded() && !Node->State->IsTerminal())
		{
			Node->Expand();
			Node = Node->Children.Last();
		}

		// 3. Simulation (Rollout)
		float Reward = Simulate(Node->State->Clone());

		// 4. Backpropagation
		while (Node != nullptr)
		{
			Node->Update(Reward);
			Node = Node->Parent;
		}
	}

	// Choose final action (most visited child)
	FMCTSNode* BestChild = nullptr;
	int32 MaxVisits = -1;

	for (FMCTSNode* Child : RootNode->Children)
	{
		if (Child->Visits > MaxVisits)
		{
			MaxVisits = Child->Visits;
			BestChild = Child;
		}
	}

	FMARLAction FinalAction = BestChild ? BestChild->Action : FMARLAction();

	delete RootNode;
	return FinalAction;
}

FMARLAction UMCTSManager::RunISMCTS(TScriptInterface<IMCTSState> InitialState, const FString& ObserverID, int32 Iterations, float ExplorationWeight)
{
	if (!InitialState) return FMARLAction();

	TSharedPtr<IMCTSState> RootState = WrapState(InitialState);
	if (!RootState.IsValid()) return FMARLAction();

	FMCTSNode* RootNode = new FMCTSNode(RootState);

	for (int32 i = 0; i < Iterations; ++i)
	{
		// 1. Sample an interpretation (Belief state sampling)
		TSharedPtr<IMCTSState> SampleState = RootState->GetRandomInterpretation(ObserverID);
		FMCTSNode* Node = RootNode;

		// 2. Selection (Tree traversal with sampled state)
		while (!Node->IsLeaf() && Node->IsFullyExpanded())
		{
			Node = Node->SelectChild(ExplorationWeight);
			// Ideally we would update SampleState here too, or have the node hold 
			// the path and we re-apply it to the SampleState.
		}

		// 3. Expansion & Simulation \& Backprop as before...
		// (Simplified for standard ISMCTS implementation)
		if (!Node->IsFullyExpanded() && !SampleState->IsTerminal())
		{
			Node->Expand();
			Node = Node->Children.Last();
		}

		float Reward = Simulate(SampleState->Clone());

		while (Node != nullptr)
		{
			Node->Update(Reward);
			Node = Node->Parent;
		}
	}

	FMCTSNode* BestChild = nullptr;
	int32 MaxVisits = -1;
	for (FMCTSNode* Child : RootNode->Children)
	{
		if (Child->Visits > MaxVisits)
		{
			MaxVisits = Child->Visits;
			BestChild = Child;
		}
	}

	FMARLAction FinalAction = BestChild ? BestChild->Action : FMARLAction();
	delete RootNode;
	return FinalAction;
}

TSharedPtr<IMCTSState> UMCTSManager::WrapState(TScriptInterface<IMCTSState> StateInterface)
{
	if (!StateInterface) return nullptr;

	UMCTSStateWrapper* Wrapper = NewObject<UMCTSStateWrapper>(this);
	Wrapper->Initialize(StateInterface.GetObject());

	// We use a custom deleter that does nothing because 'this' (the Manager)
    // owns the Wrapper, or we can just hope it lives long enough for the search.
    // For safer implementation, we could use FGCObject or TStrongObjectPtr.
	return TSharedPtr<IMCTSState>(Wrapper, [](IMCTSState*){});
}

float UMCTSManager::Simulate(TSharedPtr<IMCTSState> State)
{
	float TotalReward = 0.0f;
	int32 Depth = 0;
	const int32 MaxDepth = 50;

	while (!State->IsTerminal() && Depth < MaxDepth)
	{
		TArray<FString> AgentIDs = State->GetAgentIDs();
		TMap<FString, FMARLAction> RandomActions;

		for (const FString& ID : AgentIDs)
		{
			TArray<FMARLAction> Legal = State->GetLegalActions(ID);
			if (Legal.Num() > 0)
			{
				RandomActions.Add(ID, Legal[FMath::RandRange(0, Legal.Num() - 1)]);
			}
		}

		TotalReward += State->Step(RandomActions);
		Depth++;
	}

	return TotalReward;
}
