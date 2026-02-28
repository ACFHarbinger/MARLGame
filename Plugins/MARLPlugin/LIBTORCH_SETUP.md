# LibTorch Setup Guide

This guide explains how to install and configure LibTorch (PyTorch C++ API) for the MARL Plugin.

## Quick Start

### Linux

```bash
# Download LibTorch (CPU version)
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Move to project
mv libtorch /home/pkhunter/Repositories/MARLGame/Plugins/MARLPlugin/ThirdParty/LibTorch

# Or set environment variable
export LIBTORCH_PATH=/tmp/libtorch
echo 'export LIBTORCH_PATH=/tmp/libtorch' >> ~/.bashrc
```

### Linux with CUDA

```bash
# Download LibTorch (CUDA 11.8 version)
cd /tmp
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Set environment variable
export LIBTORCH_PATH=/tmp/libtorch
```

### Windows

```powershell
# Download LibTorch (CPU version)
# Visit: https://pytorch.org/get-started/locally/
# Select: LibTorch, Windows, CPU

# Extract to:
C:\LibTorch

# Set environment variable
setx LIBTORCH_PATH "C:\LibTorch"
```

### macOS

```bash
# Download LibTorch (CPU version)
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip

# Set environment variable
export LIBTORCH_PATH=/tmp/libtorch
echo 'export LIBTORCH_PATH=/tmp/libtorch' >> ~/.zshrc
```

## Detailed Installation

### Step 1: Choose Your Version

Visit: https://pytorch.org/get-started/locally/

**Select:**
- PyTorch Build: Stable
- Your OS: Linux / Mac / Windows
- Package: LibTorch
- Language: C++ / Java
- Compute Platform: CPU or CUDA

**Recommendations:**
- **Development**: CPU version (easier setup, portable)
- **Training**: CUDA version (much faster)
- **Version**: 2.1.0 or later

### Step 2: Download

**Option A: Direct Download**
```bash
# Example for Linux + CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

**Option B: Using pip (then extract)**
```bash
pip install torch==2.1.0
# LibTorch is in: python_site_packages/torch/lib/
```

### Step 3: Extract and Verify

```bash
unzip libtorch-*.zip

# Verify structure
ls libtorch/
# Expected: bin/ include/ lib/ share/

# Verify libraries (Linux)
ls libtorch/lib/
# Expected: libtorch.so, libtorch_cpu.so, libc10.so, etc.

# Verify libraries (Windows)
ls libtorch/lib/
# Expected: torch.lib, torch_cpu.lib, c10.lib, etc.
```

### Step 4: Configure MARLGame

**Option A: Environment Variable (Recommended)**
```bash
# Linux/Mac
export LIBTORCH_PATH=/path/to/libtorch
echo 'export LIBTORCH_PATH=/path/to/libtorch' >> ~/.bashrc

# Windows (PowerShell as Administrator)
setx LIBTORCH_PATH "C:\path\to\libtorch" /M
```

**Option B: Copy to Plugin Directory**
```bash
mkdir -p Plugins/MARLPlugin/ThirdParty
cp -r /path/to/libtorch Plugins/MARLPlugin/ThirdParty/LibTorch
```

### Step 5: Build MARLGame

```bash
# Regenerate project files
./GenerateProjectFiles.sh  # Linux/Mac
# or
GenerateProjectFiles.bat   # Windows

# Build
make MARLGameEditor  # Linux
# or open in Unreal Editor and build
```

### Step 6: Verify Installation

Check build output for:
```
LibTorch found at: /path/to/libtorch
WITH_LIBTORCH=1
```

If you see:
```
WARNING: LibTorch not found at: ...
LibTorch training features will be disabled.
```

Then LibTorch was not found. Check your `LIBTORCH_PATH`.

## Version Compatibility

| LibTorch Version | UE Version | Platform | Status |
|------------------|------------|----------|--------|
| 2.1.0 | 5.3+ | Linux, Windows, Mac | ✅ Tested |
| 2.0.0 | 5.3+ | Linux, Windows, Mac | ✅ Compatible |
| 1.13.0 | 5.3+ | Linux, Windows | ⚠️ Older, works |
| 2.2.0+ | 5.3+ | All | ✅ Should work |

## ABI Compatibility (Linux)

LibTorch comes in two ABI versions for Linux:

1. **cxx11-abi** (Recommended for UE 5.3+)
   - Modern C++11 ABI
   - File: `libtorch-cxx11-abi-shared-with-deps-*.zip`

2. **Pre-cxx11-abi** (For older systems)
   - Legacy ABI
   - File: `libtorch-shared-with-deps-*.zip`

**UE 5.3+ on Linux uses cxx11 ABI by default**, so download the `cxx11-abi` version.

## CUDA Setup

If using CUDA version of LibTorch:

### Linux

```bash
# Install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-530
sudo apt install nvidia-cuda-toolkit

# Verify
nvidia-smi
nvcc --version

# Download CUDA LibTorch (match your CUDA version)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

### Windows

1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Download matching LibTorch CUDA version
3. Ensure `PATH` includes CUDA `bin` directory

### Verify CUDA in Code

```cpp
#if WITH_LIBTORCH
if (torch::cuda::is_available())
{
    UE_LOG(LogTemp, Log, TEXT("CUDA is available!"));
    UE_LOG(LogTemp, Log, TEXT("CUDA device count: %d"), torch::cuda::device_count());
}
else
{
    UE_LOG(LogTemp, Warning, TEXT("CUDA not available, using CPU"));
}
#endif
```

## Troubleshooting

### "LibTorch not found"

**Check environment variable:**
```bash
echo $LIBTORCH_PATH
```

**Check directory structure:**
```bash
ls $LIBTORCH_PATH/include
ls $LIBTORCH_PATH/lib
```

**Verify in Build.cs output:**
- Look for: `WITH_LIBTORCH=1` in build logs
- If you see `WITH_LIBTORCH=0`, LibTorch wasn't found

### "Undefined symbol" errors (Linux)

This usually means ABI mismatch.

**Solution 1: Use cxx11-abi version**
```bash
# Re-download with cxx11-abi
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
```

**Solution 2: Check UE ABI settings**
In `MARLPlugin.Build.cs`, ensure:
```csharp
CppStandard = CppStandardVersion.Cpp17;
```

### "Library not loaded" errors (macOS)

**Fix library paths:**
```bash
cd libtorch/lib
for lib in *.dylib; do
    install_name_tool -add_rpath @loader_path $lib
done
```

### DLL not found (Windows)

**Add to PATH:**
```powershell
$env:PATH += ";C:\path\to\libtorch\lib"
```

**Or copy DLLs:**
```powershell
cp libtorch/lib/*.dll Binaries/Win64/
```

### CUDA out of memory

**Reduce batch size in code:**
```cpp
Config.BatchSize = 32;  // Instead of 64
```

**Or switch to CPU:**
```cpp
Config.Device = torch::kCPU;
```

## Performance Tips

### CPU Performance

```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

In code:
```cpp
torch::set_num_threads(8);
```

### GPU Performance

```cpp
// Use TensorFloat-32 for faster training (Ampere GPUs)
torch::set_float32_matmul_precision("medium");

// Enable cuDNN benchmarking
torch::backends::cudnn::benchmark::set_enabled(true);
```

## Upgrading LibTorch

```bash
# 1. Download new version
wget https://download.pytorch.org/libtorch/...

# 2. Remove old version
rm -rf $LIBTORCH_PATH

# 3. Extract new version
unzip libtorch-*.zip

# 4. Update environment variable if path changed
export LIBTORCH_PATH=/new/path

# 5. Rebuild project
make clean
make MARLGameEditor
```

## Alternative: Build LibTorch from Source

For advanced users who need custom builds:

```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Build LibTorch
python setup.py install
# LibTorch is in: build/lib/
```

See: https://github.com/pytorch/pytorch#from-source

## Support

If you encounter issues:

1. **Check LibTorch version compatibility**
2. **Verify environment variable**: `echo $LIBTORCH_PATH`
3. **Check build logs** for `WITH_LIBTORCH=1`
4. **Consult LibTorch docs**: https://pytorch.org/cppdocs/installing.html
5. **Open an issue** with your setup details

## References

- LibTorch Download: https://pytorch.org/get-started/locally/
- LibTorch C++ API: https://pytorch.org/cppdocs/
- Installation Guide: https://pytorch.org/cppdocs/installing.html
- PyTorch Forums: https://discuss.pytorch.org/
