// Copyright Epic Games, Inc. All Rights Reserved.


#include "TwinStickSpawner.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "NavigationSystem.h"
#include "NavMesh/RecastNavMesh.h"
#include "Kismet/GameplayStatics.h"
#include "TwinStickNPC.h"
#include "TwinStickGameMode.h"

ATwinStickSpawner::ATwinStickSpawner()
{
 	PrimaryActorTick.bCanEverTick = true;

}

void ATwinStickSpawner::BeginPlay()
{
	Super::BeginPlay();
	
	// find the recast navmesh actor on the level
	TArray<AActor*> ActorList;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), ARecastNavMesh::StaticClass(), ActorList);

	if (ActorList.Num() > 0)
	{
		NavData = Cast<ARecastNavMesh>(ActorList[0]);
	} else {

		UE_LOG(LogTemp, Log, TEXT("Could not find recast navmesh"));

	}

	SpawnObstacles();

	// set up the spawn timer
	GetWorld()->GetTimerManager().SetTimer(SpawnGroupTimer, this, &ATwinStickSpawner::SpawnNPCGroup, SpawnGroupDelay, true);

	// spawn the first group of NPCs
	SpawnNPCGroup();
}

void ATwinStickSpawner::EndPlay(EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	// clear the spawn timers
	GetWorld()->GetTimerManager().ClearTimer(SpawnGroupTimer);
	GetWorld()->GetTimerManager().ClearTimer(SpawnNPCTimer);
}

void ATwinStickSpawner::SpawnNPCGroup()
{
	// reset the group spawn counter
	SpawnCount = 0;

	// check if we're still under the max NPC cap
	if (ATwinStickGameMode* GM = Cast<ATwinStickGameMode>(GetWorld()->GetAuthGameMode()))
	{
		if (GM->CanSpawnNPCs())
		{
			SpawnNPC();
		}
	}
}

void ATwinStickSpawner::SpawnNPC()
{
	FTransform SpawnTransform;

	// find a random point around the spawner
	FVector SpawnLoc;
	if (UNavigationSystemV1::K2_GetRandomReachablePointInRadius(GetWorld(), GetActorLocation(), SpawnLoc, SpawnRadius, NavData))
	{
		SpawnTransform.SetLocation(SpawnLoc);

		// spawn the NPC
		ATwinStickNPC* NPC = GetWorld()->SpawnActor<ATwinStickNPC>(NPCClass, SpawnTransform);
	}

	// increase the spawn counter
	++SpawnCount;

	// do we still have enemies left to spawn?
	if (SpawnCount < SpawnGroupSize)
	{
		GetWorld()->GetTimerManager().SetTimer(SpawnNPCTimer, this, &ATwinStickSpawner::SpawnNPC, FMath::RandRange(MinSpawnDelay, MaxSpawnDelay), false);
	}

}

void ATwinStickSpawner::SpawnObstacles()
{
	if (ObstacleClasses.Num() == 0 || NumObstacles <= 0) return;

	for (int32 i = 0; i < NumObstacles; ++i)
	{
		FVector SpawnLoc;
		if (UNavigationSystemV1::K2_GetRandomReachablePointInRadius(GetWorld(), GetActorLocation(), SpawnLoc, SpawnRadius, NavData))
		{
			int32 ClassIndex = FMath::RandRange(0, ObstacleClasses.Num() - 1);
			if (ObstacleClasses[ClassIndex])
			{
				FTransform SpawnTransform(FRotator(0, FMath::RandRange(0.0f, 360.0f), 0), SpawnLoc);
				GetWorld()->SpawnActor<AActor>(ObstacleClasses[ClassIndex], SpawnTransform);
			}
		}
	}
}
