// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class MARLGame : ModuleRules
{
	public MARLGame(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",
			"AIModule",
			"NavigationSystem",
			"StateTreeModule",
			"GameplayStateTreeModule",
			"Niagara",
			"UMG",
			"Slate",
			"MARLPlugin"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });

		PublicIncludePaths.AddRange(new string[] {
			"MARLGame",
			"MARLGame/Variant_Strategy",
			"MARLGame/Variant_Strategy/UI",
			"MARLGame/Variant_TwinStick",
			"MARLGame/Variant_TwinStick/AI",
			"MARLGame/Variant_TwinStick/Gameplay",
			"MARLGame/Variant_TwinStick/UI"
		});

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}
}
