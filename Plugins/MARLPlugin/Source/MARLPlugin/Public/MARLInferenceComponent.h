#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MARLTypes.h"
#include "MARLInferenceComponent.generated.h"

#if WITH_LIBTORCH
#include <torch/script.h> // One-stop header for TorchScript
#endif

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActionSelectedDelegate, const FMARLAction&, SelectedAction);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnExplainabilityDataDelegate, const TArray<float>&, ActionProbabilities);

/**
 * Lightweight component for running inference on TorchScript models (.pt files) in production.
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLInferenceComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	UMARLInferenceComponent();

	/** Path to the compiled .pt TorchScript model */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Inference")
	FString ModelPath;

	/** If true, the component will attempt to load the model on BeginPlay */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MARL|Inference")
	bool bAutoLoad = true;

	/** Delegate fired when a forward pass completes */
	UPROPERTY(BlueprintAssignable, Category = "MARL|Inference")
	FOnActionSelectedDelegate OnActionSelected;

	/** Delegate fired exposing raw probabilities/values for UI Explainability */
	UPROPERTY(BlueprintAssignable, Category = "MARL|Explainability")
	FOnExplainabilityDataDelegate OnExplainabilityData;

	/** Loads the TorchScript model from the specified path */
	UFUNCTION(BlueprintCallable, Category = "MARL|Inference")
	bool LoadModel(const FString& Path);

	/** Runs the model forward pass with the given flattened observation array */
	UFUNCTION(BlueprintCallable, Category = "MARL|Inference")
	FMARLAction RunInference(const TArray<float>& Observation);

protected:
	virtual void BeginPlay() override;

private:

#if WITH_LIBTORCH
	/** The JIT compiled TorchScript module */
	torch::jit::script::Module ScriptModule;
#endif

	bool bIsModelLoaded = false;
};
