#pragma once

#include "CoreMinimal.h"
#include "MARLSensorComponent.h"
#include "MARLRaycastSensor.generated.h"

UCLASS(ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLRaycastSensor : public UMARLSensorComponent
{
	GENERATED_BODY()

public:
	UMARLRaycastSensor();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	int32 NumRays = 32;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	float RayLength = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	int32 FeaturesPerRay = 3; // e.g., distance, object_type, is_enemy

	virtual TArray<float> GetObservations() const override;
	virtual int32 GetObservationSize() const override;

protected:
	virtual void BeginPlay() override;
};
