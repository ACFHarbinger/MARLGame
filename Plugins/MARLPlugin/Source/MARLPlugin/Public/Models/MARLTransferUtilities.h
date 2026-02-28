#pragma once

#include "CoreMinimal.h"

#if WITH_LIBTORCH
#include <torch/torch.h>

/**
 * Utility functions for Transfer Learning and Meta-Learning in LibTorch.
 */
class MARLPLUGIN_API FMARLTransferUtilities
{
public:
	/**
	 * Freezes or unfreezes specific layers of a module.
	 * Useful for fine-tuning task-specific heads while keeping core encoders frozen.
	 * 
	 * @param Module The PyTorch module to modify.
	 * @param LayerPrefix Any parameter name starting with this prefix will be affected.
	 * @param bFreeze True sets requires_grad to false, False sets requires_grad to true.
	 */
	static void SetLayerGradRequirement(std::shared_ptr<torch::nn::Module> Module, const std::string& LayerPrefix, bool bFreeze = true)
	{
		if (!Module) return;

		for (auto& param : Module->named_parameters())
		{
			// Check if the parameter name starts with the given prefix
			if (param.key().find(LayerPrefix) == 0)
			{
				param.value().set_requires_grad(!bFreeze);
				
				FString UEKey(param.key().c_str());
				FString StateStr = bFreeze ? TEXT("Frozen") : TEXT("Unfrozen");
				UE_LOG(LogTemp, Log, TEXT("Transfer Learning: Parameter %s is now %s."), *UEKey, *StateStr);
			}
		}
	}

	/**
	 * Copies weights from a source module to a destination module for matching parameter names.
	 * Used to transfer learned representation encoders between completely different algorithm networks.
	 */
	static void TransferMatchingWeights(std::shared_ptr<torch::nn::Module> SourceModule, std::shared_ptr<torch::nn::Module> DestModule)
	{
		if (!SourceModule || !DestModule) return;

		torch::NoGradGuard no_grad; // Disable gradient tracking during copy

		auto SourceParams = SourceModule->named_parameters();
		auto DestParams = DestModule->named_parameters();

		for (auto& dest_param : DestParams)
		{
			if (SourceParams.contains(dest_param.key()))
			{
				auto& source_tensor = SourceParams[dest_param.key()];
				if (source_tensor.sizes() == dest_param.value().sizes())
				{
					dest_param.value().copy_(source_tensor);
					FString UEKey(dest_param.key().c_str());
					UE_LOG(LogTemp, Log, TEXT("Transferred weights for parameter: %s"), *UEKey);
				}
			}
		}
	}
};

#endif // WITH_LIBTORCH
