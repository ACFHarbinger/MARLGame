#include "MCTS/MCTSNode.h"
#include "MCTS/MCTSState.h"
#include "Math/UnrealMathUtility.h"

FMCTSNode::FMCTSNode(TSharedPtr<IMCTSState> InState, FMCTSNode* InParent, FMARLAction InAction)
	: State(InState), Parent(InParent), Action(InAction)
{
	if (State.IsValid())
	{
		TArray<FString> AgentIDs = State->GetAgentIDs();
		if (AgentIDs.Num() > 0)
		{
			UntriedAgents = AgentIDs;
			for (const FString& ID : AgentIDs)
			{
				UntriedActions.Add(ID, State->GetLegalActions(ID));
			}
		}
	}
}

FMCTSNode::~FMCTSNode()
{
	for (FMCTSNode* Child : Children)
	{
		delete Child;
	}
	Children.Empty();
}

void FMCTSNode::Expand()
{
	if (!State.IsValid() || State->IsTerminal()) return;

	// Pick an agent and an action that hasn't been tried yet
	if (UntriedAgents.Num() > 0)
	{
		FString AgentID = UntriedAgents[0];
		TArray<FMARLAction>& Actions = UntriedActions[AgentID];

		if (Actions.Num() > 0)
		{
			FMARLAction NewAction = Actions.Pop();
			
			// For simplicity in standard MCTS, we evaluate one agent at a time or joint actions
			// Here we clone and step
			TSharedPtr<IMCTSState> NewState = State->Clone();
			
			TMap<FString, FMARLAction> ActionMap;
			ActionMap.Add(AgentID, NewAction);
			
			NewState->Step(ActionMap);

			FMCTSNode* Child = new FMCTSNode(NewState, this, NewAction);
			Children.Add(Child);
		}

		if (Actions.Num() == 0)
		{
			UntriedAgents.RemoveAt(0);
		}
	}
}

FMCTSNode* FMCTSNode::SelectChild(float ExplorationWeight)
{
	FMCTSNode* BestChild = nullptr;
	float BestValue = -MAX_FLT;

	for (FMCTSNode* Child : Children)
	{
		// UCT Formula: (TotalReward / Visits) + C * sqrt(log(ParentVisits) / Visits)
		float Exploitation = Child->TotalReward / (float)Child->Visits;
		float Exploration = ExplorationWeight * FMath::Sqrt(FMath::Loge((float)Visits) / (float)Child->Visits);
		float Value = Exploitation + Exploration;

		if (Value > BestValue)
		{
			BestValue = Value;
			BestChild = Child;
		}
	}

	return BestChild;
}

void FMCTSNode::Update(float Reward)
{
	Visits++;
	TotalReward += Reward;
}

bool FMCTSNode::IsFullyExpanded() const
{
	return UntriedAgents.Num() == 0;
}
