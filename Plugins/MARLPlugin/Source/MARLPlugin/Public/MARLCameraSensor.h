#pragma once

#include "CoreMinimal.h"
#include "MARLSensorComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "MARLCameraSensor.generated.h"

UCLASS(ClassGroup=(MARL), meta=(BlueprintSpawnableComponent))
class MARLPLUGIN_API UMARLCameraSensor : public UMARLSensorComponent
{
	GENERATED_BODY()

public:
	UMARLCameraSensor();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	int32 Width = 84;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	int32 Height = 84;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="MARL|Sensor")
	bool bGrayscale = false;

	virtual TArray<float> GetObservations() const override;
	virtual int32 GetObservationSize() const override;

protected:
	virtual void BeginPlay() override;

private:
	UPROPERTY()
	USceneCaptureComponent2D* SceneCapture;
};
