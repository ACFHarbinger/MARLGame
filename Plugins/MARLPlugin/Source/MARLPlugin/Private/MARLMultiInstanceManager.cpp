#include "MARLMultiInstanceManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Misc/CommandLine.h"

UMARLMultiInstanceManager::UMARLMultiInstanceManager()
{
	// Resolve paths assuming we are running from within the project or editor
	EditorExecutablePath = FPlatformProcess::ExecutablePath();
	ProjectFilePath = FPaths::GetProjectFilePath();
}

bool UMARLMultiInstanceManager::InitializeInstances(int32 NumInstances, const FString& MapName, int32 BasePort)
{
	if (bIsInitialized)
	{
		UE_LOG(LogTemp, Warning, TEXT("Multi-Instance Manager already initialized."));
		return false;
	}

	if (NumInstances <= 0)
	{
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("Starting %d MARL headless instances starting from port %d"), NumInstances, BasePort);

	for (int32 i = 0; i < NumInstances; ++i)
	{
		if (!SpawnInstance(MapName, BasePort + i))
		{
			UE_LOG(LogTemp, Error, TEXT("Failed to spawn instance on port %d"), BasePort + i);
			// Optionally we could stop already spawned instances here
		}
	}

	bIsInitialized = true;
	return ActiveProcessHandles.Num() > 0;
}

bool UMARLMultiInstanceManager::SpawnInstance(const FString& MapName, int32 Port)
{
	FString Arguments = FString::Printf(TEXT("\"%s\" %s -server -log -NoUI -Port=%d -ResX=800 -ResY=600 -windowed"), 
		*ProjectFilePath, *MapName, Port);

	uint32 ProcessID = 0;
	FProcHandle Handle = FPlatformProcess::CreateProc(
		*EditorExecutablePath,
		*Arguments,
		true, // bLaunchDetached
		false, // bLaunchHidden
		false, // bLaunchReallyHidden
		&ProcessID,
		0, // PriorityModifier
		nullptr, // OptionalWorkingDirectory
		nullptr // PipeWriteChild
	);

	if (Handle.IsValid())
	{
		ActiveProcessHandles.Add(Handle);
		UE_LOG(LogTemp, Log, TEXT("Successfully spawned UE instance PID: %u on port %d"), ProcessID, Port);
		return true;
	}

	return false;
}

void UMARLMultiInstanceManager::StopAllInstances()
{
	for (FProcHandle& Handle : ActiveProcessHandles)
	{
		if (Handle.IsValid() && FPlatformProcess::IsProcRunning(Handle))
		{
			FPlatformProcess::TerminateProc(Handle, true);
		}
	}
	ActiveProcessHandles.Empty();
	bIsInitialized = false;

	UE_LOG(LogTemp, Log, TEXT("All MARL headless instances stopped."));
}

int32 UMARLMultiInstanceManager::GetActiveInstanceCount() const
{
	int32 Count = 0;
	for (const FProcHandle& Handle : ActiveProcessHandles)
	{
		if (Handle.IsValid() && FPlatformProcess::IsProcRunning(Handle))
		{
			Count++;
		}
	}
	return Count;
}

void UMARLMultiInstanceManager::BeginDestroy()
{
	StopAllInstances();
	Super::BeginDestroy();
}
