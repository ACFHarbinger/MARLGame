using UnrealBuildTool;
using System.IO;

public class MARLPlugin : ModuleRules
{
	public MARLPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		// Enable C++20 for UE5.7 compatibility
		CppStandard = CppStandardVersion.Cpp20;
		bEnableExceptions = true;  // LibTorch uses exceptions

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

		// LibTorch Integration
		SetupLibTorch(Target);
	}

	private void SetupLibTorch(ReadOnlyTargetRules Target)
	{
		// Path to LibTorch installation
		// Users should set LIBTORCH_PATH environment variable or download LibTorch to ThirdParty folder
		string LibTorchPath = System.Environment.GetEnvironmentVariable("LIBTORCH_PATH");

		if (string.IsNullOrEmpty(LibTorchPath))
		{
			// Fallback to ThirdParty directory
			LibTorchPath = Path.Combine(ModuleDirectory, "..", "..", "..", "ThirdParty", "LibTorch");
		}

		if (Directory.Exists(LibTorchPath))
		{
			// Include directories
			PublicIncludePaths.Add(Path.Combine(LibTorchPath, "include"));
			PublicIncludePaths.Add(Path.Combine(LibTorchPath, "include", "torch", "csrc", "api", "include"));

			// Library directories and files
			if (Target.Platform == UnrealTargetPlatform.Linux)
			{
				string LibPath = Path.Combine(LibTorchPath, "lib");
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libtorch.so"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libtorch_cpu.so"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libc10.so"));

				// Runtime dependencies
				RuntimeDependencies.Add(Path.Combine(LibPath, "libtorch.so"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "libtorch_cpu.so"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "libc10.so"));

				// Add CUDA libraries if available
				string CudaLibPath = Path.Combine(LibPath, "libtorch_cuda.so");
				if (File.Exists(CudaLibPath))
				{
					PublicAdditionalLibraries.Add(CudaLibPath);
					RuntimeDependencies.Add(CudaLibPath);
				}
			}
			else if (Target.Platform == UnrealTargetPlatform.Win64)
			{
				string LibPath = Path.Combine(LibTorchPath, "lib");
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "torch.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "torch_cpu.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "c10.lib"));

				// Runtime dependencies (DLLs)
				RuntimeDependencies.Add(Path.Combine(LibPath, "torch.dll"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "torch_cpu.dll"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "c10.dll"));

				// Add CUDA libraries if available
				string CudaLibPath = Path.Combine(LibPath, "torch_cuda.dll");
				if (File.Exists(CudaLibPath))
				{
					PublicAdditionalLibraries.Add(Path.Combine(LibPath, "torch_cuda.lib"));
					RuntimeDependencies.Add(CudaLibPath);
				}
			}
			else if (Target.Platform == UnrealTargetPlatform.Mac)
			{
				string LibPath = Path.Combine(LibTorchPath, "lib");
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libtorch.dylib"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libtorch_cpu.dylib"));
				PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libc10.dylib"));

				RuntimeDependencies.Add(Path.Combine(LibPath, "libtorch.dylib"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "libtorch_cpu.dylib"));
				RuntimeDependencies.Add(Path.Combine(LibPath, "libc10.dylib"));
			}

			// Define preprocessor macro to enable LibTorch features
			PublicDefinitions.Add("WITH_LIBTORCH=1");
		}
		else
		{
			// LibTorch not found - disable features
			PublicDefinitions.Add("WITH_LIBTORCH=0");
			System.Console.WriteLine("WARNING: LibTorch not found at: " + LibTorchPath);
			System.Console.WriteLine("LibTorch training features will be disabled. Set LIBTORCH_PATH environment variable or install to ThirdParty/LibTorch");
		}
	}
}
