#include "MARLRaycastSensor.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"
#include "GameFramework/Actor.h"

UMARLRaycastSensor::UMARLRaycastSensor()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UMARLRaycastSensor::BeginPlay()
{
	Super::BeginPlay();
}

int32 UMARLRaycastSensor::GetObservationSize() const
{
	return NumRays * FeaturesPerRay;
}

TArray<float> UMARLRaycastSensor::GetObservations() const
{
	TArray<float> Observations;
	Observations.Init(0.0f, GetObservationSize());

	if (AActor* Owner = GetOwner())
	{
		UWorld* World = GetWorld();
		if (!World) return Observations;

		FVector StartLocation = Owner->GetActorLocation();
		FRotator Rotation = Owner->GetActorRotation();

		float AngleStep = 360.0f / (float)NumRays;

		for (int32 i = 0; i < NumRays; ++i)
		{
			float CurrentAngle = AngleStep * i;
			FRotator RayDirection = Rotation + FRotator(0.0f, CurrentAngle, 0.0f);
			FVector EndLocation = StartLocation + (RayDirection.Vector() * RayLength);

			FHitResult HitResult;
			FCollisionQueryParams QueryParams;
			QueryParams.AddIgnoredActor(Owner);

			bool bHit = World->LineTraceSingleByChannel(HitResult, StartLocation, EndLocation, ECC_Visibility, QueryParams);

			int32 BaseIdx = i * FeaturesPerRay;
			
			if (bHit)
			{
				float Distance = HitResult.Distance / RayLength;
				Observations[BaseIdx] = Distance;
				Observations[BaseIdx + 1] = 1.0f; // Placeholder for object type
				Observations[BaseIdx + 2] = 0.0f; // Placeholder for is_enemy
			}
			else
			{
				Observations[BaseIdx] = 1.0f;
				Observations[BaseIdx + 1] = 0.0f;
				Observations[BaseIdx + 2] = 0.0f;
			}
		}
	}

	return Observations;
}
