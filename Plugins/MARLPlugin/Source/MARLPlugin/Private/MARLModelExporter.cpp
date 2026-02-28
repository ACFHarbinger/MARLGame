#include "MARLModelExporter.h"

#if WITH_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include "Models/MARLModelFactory.h"
#endif

bool UMARLModelExporter::ExportModelToTorchScript(const FString& SourceName, const FString& OutputFilePath, int32 ObsSize)
{
#if WITH_LIBTORCH
	// 1. We would retrieve the trained model from the factory or trainer
	// For example, an ActorNetwork with specific config:
	FMARLActionSpace DummyActionSpace;
	DummyActionSpace.DiscreteSizes.Add(2);
	DummyActionSpace.ContinuousSize = 2;

	auto PolicyNet = std::make_shared<MARLActorNetwork>(ObsSize, DummyActionSpace, 128);
	
	// Create dummy input for tracing
	auto DummyInput = torch::randn({1, ObsSize});

	try {
		// 2. Trace the model
		auto TracedModule = torch::jit::trace(PolicyNet, DummyInput);
		
		// 3. Save to file
		TracedModule.save(TCHAR_TO_UTF8(*OutputFilePath));
		
		UE_LOG(LogTemp, Log, TEXT("Successfully exported TorchScript model to: %s"), *OutputFilePath);
		return true;
		
	} catch (const c10::Error& e) {
		UE_LOG(LogTemp, Error, TEXT("Failed to export TorchScript model: %s"), UTF8_TO_TCHAR(e.what()));
		return false;
	}
#else
	UE_LOG(LogTemp, Warning, TEXT("LibTorch is not enabled. Cannot export model."));
	return false;
#endif
}
