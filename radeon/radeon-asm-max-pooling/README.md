# Debugger Example

## Prerequisites

### Remote Machine

The following software needs to be installed on the machine where the kernels will run:

* [ROCm Platform](https://rocm.github.io/ROCmInstall.html)
* [.NET Core 2.2 Runtime](https://dotnet.microsoft.com/download/dotnet-core/2.2)
* CMake and additional script dependencies:
```sh
sudo apt-get -y install cmake libboost-program-options-dev liblist-moreutils-perl
```

### Host Machine

* Visual Studio 2017/2019
* [RadeonAsmDebugger Extension for Visual Studio](https://github.com/vsrad/radeon-asm-tools)

## Building and Running

### Remote Machine

1. Transfer the `RadeonPooling` directory to the remote machine
2. Create a build directory and run `cmake` followed by `make` there:
```sh
cd RadeonPooling
mkdir build
cd build
cmake ..
make
```
3. Download latest release, copy [DebugServerLinux64](https://github.com/vsrad/radeon-asm-tools/releases) and navigate to this directory
4. Launch the debug server with `chmod +x RadeonAsmDebugServer && ./RadeonAsmDebugServer`

### Host machine

1. Open `RadeonPoolingExample.sln` in Visual Studio
2. Go to *Tools* -> *RAD Debug* -> *Options*
3. Click the *Edit* button in the opened window to edit the active debug profile
4. In the *Debugger* tab, set *Working Directory* to the absolute path
to `RadeonPooling` on the remote machine
5. Press *Apply* to save the changes and *OK* to close the profile editor
6. Open one kernel in *Solution Explorer*
7. Start debugging by pressing F5 (alternatively, clicking on the *RAD Debugger* button with a green arrow)

## Notes and Remarks

* If you see *Permissions denied* messages in the output, make sure that the following files have the executable bit set (`chmod +x`):
  - `RadeonPooling/common/debugger/dbg_clang_wrapper.sh`
  - `RadeonPooling/common/debugger/vadd_debug.sh`
  - `RadeonPooling/common/debugger/breakpoint_gcnasm.pl`

* Wave size is hardcoded in this example (see [breakpoint_gcnasm.pl](RadeonPooling/common/debugger/breakpoint_gcnasm.pl#L170)):
    ```
    -   s_mul_i32 s[dbg_soff], s[dbg_soff], wave_size * (1 + $n_var) * 4
    +   s_mul_i32 s[dbg_soff], s[dbg_soff], 64 * (1 + $n_var) * 4
    ```
