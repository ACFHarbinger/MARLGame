#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MARLSensorComponent.generated.h"

UCLASS(Abstract, Blueprintable, ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLSensorComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UMARLSensorComponent();

	UFUNCTION(BlueprintCallable, Category = "MARL|Sensor")
	virtual TArray<float> GetObservations() const PURE_VIRTUAL(UMARLSensorComponent::GetObservations, return TArray<float>(););

	UFUNCTION(BlueprintCallable, Category = "MARL|Sensor")
	virtual int32 GetObservationSize() const PURE_VIRTUAL(UMARLSensorComponent::GetObservationSize, return 0;);
};
