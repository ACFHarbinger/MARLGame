#include "MARLLibTorchInterface.h"

UMARLLibTorchInterface::UMARLLibTorchInterface()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UMARLLibTorchInterface::BeginPlay()
{
	Super::BeginPlay();
}

void UMARLLibTorchInterface::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	ShutdownIPCServer();
	Super::EndPlay(EndPlayReason);
}

bool UMARLLibTorchInterface::InitializeIPCServer(int32 Port)
{
	CurrentPort = Port;
	bIsServerRunning = true;
	// Initialize IPC / Shared memory here (LibTorch specific code)
	return true;
}

void UMARLLibTorchInterface::ShutdownIPCServer()
{
	if (bIsServerRunning)
	{
		// Cleanup IPC resources
		bIsServerRunning = false;
	}
}

void UMARLLibTorchInterface::SendObservations(const TMap<FString, TArray<float>>& Observations)
{
	if (!bIsServerRunning) return;
	
	// Convert Unreal arrays to torch::Tensor and send memory
}

bool UMARLLibTorchInterface::ReceiveActions(TMap<FString, TArray<float>>& OutActions)
{
	if (!bIsServerRunning) return false;
	
	// Read torch::Tensor from memory and convert to Unreal arrays
	return true;
}
