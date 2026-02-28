#include "MARLConfig.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

void UMARLConfig::LoadFromJson(const FString& JsonFilePath)
{
	FString JsonString;
	if (FFileHelper::LoadFileToString(JsonString, *JsonFilePath))
	{
		TSharedPtr<FJsonObject> JsonObject;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

		if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
		{
			if (JsonObject->HasField("num_agents"))
			{
				NumAgents = JsonObject->GetIntegerField("num_agents");
			}
			if (JsonObject->HasField("max_episode_steps"))
			{
				MaxEpisodeSteps = JsonObject->GetIntegerField("max_episode_steps");
			}
			// Parse Observation/Action spaces, nested maps...
		}
	}
}
