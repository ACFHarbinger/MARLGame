#include "MARLGameStateSensor.h"
#include "Engine/World.h"
#include "GameFramework/GameStateBase.h"

UMARLGameStateSensor::UMARLGameStateSensor()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UMARLGameStateSensor::BeginPlay()
{
	Super::BeginPlay();
}

int32 UMARLGameStateSensor::GetObservationSize() const
{
	return ExpectedStateSize;
}

TArray<float> UMARLGameStateSensor::GetObservations() const
{
	TArray<float> Observations;
	Observations.Init(0.0f, ExpectedStateSize);
	return Observations;
}
