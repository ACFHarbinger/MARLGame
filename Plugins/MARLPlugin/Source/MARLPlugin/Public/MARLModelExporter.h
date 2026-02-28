#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "MARLModelExporter.generated.h"

/**
 * Utility functions for exporting LibTorch models to TorchScript.
 */
UCLASS()
class MARLPLUGIN_API UMARLModelExporter : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Traces a PyTorch model and saves it to a TorchScript .pt file.
	 * Requires the model to be valid and a sample observation size.
	 */
	UFUNCTION(BlueprintCallable, Category = "MARL|Export")
	static bool ExportModelToTorchScript(const FString& SourceName, const FString& OutputFilePath, int32 ObsSize);

};
