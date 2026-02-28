using UnrealBuildTool;

public class MARLPlugin : ModuleRules
{
	public MARLPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Sockets",
				"Networking",
				"HTTP",
				"Json",
				"JsonUtilities",
				"AIModule",
				"NavigationSystem"
			}
		);
			
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
			}
		);
	}
}
