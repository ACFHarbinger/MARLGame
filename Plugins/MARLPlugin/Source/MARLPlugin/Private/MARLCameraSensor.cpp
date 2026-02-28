#include "MARLCameraSensor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"

UMARLCameraSensor::UMARLCameraSensor()
{
	PrimaryComponentTick.bCanEverTick = true;
	SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
}

void UMARLCameraSensor::BeginPlay()
{
	Super::BeginPlay();

	if (SceneCapture && !SceneCapture->TextureTarget)
	{
		UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>(this);
		RenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, true);
		RenderTarget->UpdateResourceImmediate();
		SceneCapture->TextureTarget = RenderTarget;
	}
}

int32 UMARLCameraSensor::GetObservationSize() const
{
	return Width * Height * (bGrayscale ? 1 : 3);
}

TArray<float> UMARLCameraSensor::GetObservations() const
{
	TArray<float> Observations;
	Observations.Init(0.0f, GetObservationSize());

	if (SceneCapture && SceneCapture->TextureTarget)
	{
		// Render target read logic
	}

	return Observations;
}
