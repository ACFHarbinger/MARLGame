#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "MARLTypes.h"
#include "MARLAgentInterface.generated.h"

UINTERFACE(MinimalAPI, Blueprintable)
class UMARLAgentInterface : public UInterface
{
	GENERATED_BODY()
};

/**
 * Interface for any actor that can be controlled by a MARL policy.
 */
class MARLPLUGIN_API IMARLAgentInterface
{
	GENERATED_BODY()

public:
	/** Called when an action is received from the RL policy */
	UFUNCTION(BlueprintNativeEvent, Category = "MARL")
	void ApplyAction(const FMARLAction& Action);
};
