#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MCTSState.h"
#include "MCTSStateWrapper.generated.h"

/**
 * A concrete UObject that wraps another UObject implementing IMCTSState.
 * This allows us to handle the interface in a way that's safe for the SharedPtr-based tree.
 */
UCLASS(BlueprintType)
class MARLPLUGIN_API UMCTSStateWrapper : public UObject, public IMCTSState
{
	GENERATED_BODY()

public:
	void Initialize(UObject* InSourceObject);

	// IMCTSState Interface
	virtual TSharedPtr<IMCTSState> Clone() const override;
	virtual float Step(const TMap<FString, FMARLAction>& Actions) override;
	virtual bool IsTerminal() const override;
	virtual TArray<FMARLAction> GetLegalActions(const FString& AgentID) const override;
	virtual TArray<FString> GetAgentIDs() const override;
	virtual TSharedPtr<IMCTSState> GetRandomInterpretation(const FString& ObserverID) const override;

private:
	UPROPERTY()
	UObject* SourceObject;
};
