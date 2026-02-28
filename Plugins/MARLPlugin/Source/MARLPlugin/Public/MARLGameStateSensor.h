#pragma once

#include "CoreMinimal.h"
#include "MARLSensorComponent.h"
#include "MARLGameStateSensor.generated.h"

UCLASS(ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLGameStateSensor : public UMARLSensorComponent
{
	GENERATED_BODY()

public:
	UMARLGameStateSensor();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	int32 ExpectedStateSize = 5;

	virtual TArray<float> GetObservations() const override;
	virtual int32 GetObservationSize() const override;

protected:
	virtual void BeginPlay() override;
};
