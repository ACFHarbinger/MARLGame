#include "MARLInferenceComponent.h"
#include "Misc/Paths.h"

UMARLInferenceComponent::UMARLInferenceComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UMARLInferenceComponent::BeginPlay()
{
	Super::BeginPlay();

	if (bAutoLoad)
	{
		LoadModel(ModelPath);
	}
}

bool UMARLInferenceComponent::LoadModel(const FString& Path)
{
	if (Path.IsEmpty()) return false;
	
	FString FullPath = FPaths::Combine(FPaths::ProjectContentDir(), Path);
	
#if WITH_LIBTORCH
	try {
		ScriptModule = torch::jit::load(TCHAR_TO_UTF8(*FullPath));
		bIsModelLoaded = true;
		UE_LOG(LogTemp, Log, TEXT("Successfully loaded TorchScript model from: %s"), *FullPath);
		return true;
	}
	catch (const c10::Error& e) {
		UE_LOG(LogTemp, Error, TEXT("Error loading TorchScript model %s: %s"), *FullPath, UTF8_TO_TCHAR(e.what()));
		return false;
	}
#else
	UE_LOG(LogTemp, Warning, TEXT("LibTorch is not enabled. Cannot load model %s"), *FullPath);
	return false;
#endif
}

FMARLAction UMARLInferenceComponent::RunInference(const TArray<float>& Observation)
{
	FMARLAction ResultAction;

	if (!bIsModelLoaded)
	{
		UE_LOG(LogTemp, Warning, TEXT("UMARLInferenceComponent: Cannot run inference, model is not loaded."));
		return ResultAction;
	}

#if WITH_LIBTORCH
	// 1. Convert UE Array to LibTorch Tensor [1, ObsSize]
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor ObsTensor = torch::from_blob((void*)Observation.GetData(), {1, (long long)Observation.Num()}, options).clone();

	// 2. Wrap inputs
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(ObsTensor);

	// 3. Forward Pass
	try {
		torch::Tensor OutputTensor = ScriptModule.forward(inputs).toTensor();
		
		// 4. Extract action
		// Assuming OutputTensor shape is [1, ActionSize]. We can parse it into the FMARLAction struct
		if (OutputTensor.numel() > 0)
		{
			// Example extraction logic
			ResultAction.DiscreteActions.Add(OutputTensor[0][0].argmax().item<int>());

			// Broadcast explainability data (e.g. raw logits/probs) for the UI
			TArray<float> RawProbs;
			for (int i = 0; i < OutputTensor.size(1); ++i)
			{
				RawProbs.Add(OutputTensor[0][i].item<float>());
			}
			OnExplainabilityData.Broadcast(RawProbs);

			// Extract continuous (if applicable)
			for (int i = 1; i < OutputTensor.size(1); ++i)
			{
				ResultAction.ContinuousActions.Add(OutputTensor[0][i].item<float>());
			}
		}

		OnActionSelected.Broadcast(ResultAction);

	} catch (const c10::Error& e) {
		UE_LOG(LogTemp, Error, TEXT("Error during TorchScript forward pass: %s"), UTF8_TO_TCHAR(e.what()));
	}
#endif

	return ResultAction;
}
