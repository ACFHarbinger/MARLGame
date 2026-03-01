#include "MCTS/MCTSStateWrapper.h"

void UMCTSStateWrapper::Initialize(UObject* InSourceObject)
{
	SourceObject = InSourceObject;
}

TSharedPtr<IMCTSState> UMCTSStateWrapper::Clone() const
{
	if (!SourceObject || !SourceObject->GetClass()->ImplementsInterface(UMCTSState::StaticClass()))
	{
		return nullptr;
	}

	UObject* ClonedSource = IMCTSState::Execute_CloneUObject(SourceObject);
	if (!ClonedSource) return nullptr;

	UMCTSStateWrapper* ClonedWrapper = NewObject<UMCTSStateWrapper>();
	ClonedWrapper->Initialize(ClonedSource);
	
	// Note: ClonedWrapper is a UObject, and we're returning it as a SharedPtr.
	// This is dangerous if the SharedPtr is the only thing keeping it alive.
    // In a production MARL system, we'd either use a pure C++ state or 
    // root the UObjects in the MCTSManager.
	return TSharedPtr<IMCTSState>(ClonedWrapper, [](IMCTSState*){ /* UObject handled by GC */ });
}

float UMCTSStateWrapper::Step(const TMap<FString, FMARLAction>& Actions)
{
	if (SourceObject)
	{
		return IMCTSState::Execute_StepEnvironment(SourceObject, Actions);
	}
	return 0.0f;
}

bool UMCTSStateWrapper::IsTerminal() const
{
	if (SourceObject)
	{
		return IMCTSState::Execute_IsTerminalState(SourceObject);
	}
	return true;
}

TArray<FMARLAction> UMCTSStateWrapper::GetLegalActions(const FString& AgentID) const
{
	if (SourceObject)
	{
		return IMCTSState::Execute_GetLegalActionsForAgent(SourceObject, AgentID);
	}
	return TArray<FMARLAction>();
}

TArray<FString> UMCTSStateWrapper::GetAgentIDs() const
{
	if (SourceObject)
	{
		return IMCTSState::Execute_GetActiveAgentIDs(SourceObject);
	}
	return TArray<FString>();
}

TSharedPtr<IMCTSState> UMCTSStateWrapper::GetRandomInterpretation(const FString& ObserverID) const
{
	if (!SourceObject) return nullptr;

	UObject* Interpretation = IMCTSState::Execute_SampleInterpretation(SourceObject, ObserverID);
	if (!Interpretation) return Clone();

	UMCTSStateWrapper* Wrapper = NewObject<UMCTSStateWrapper>();
	Wrapper->Initialize(Interpretation);
	return TSharedPtr<IMCTSState>(Wrapper, [](IMCTSState*){});
}
