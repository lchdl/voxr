#pragma once
// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nvrtc.h>
// standard C
#include <assert.h>
#include <memory.h>
#include <unistd.h>
#include <libgen.h>
// C++
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
//
using namespace std;


#if defined(_DEBUG) || defined(_DEBUG_) || defined(__DEBUG__)
#define DEBUG_MODE
#define ASSERT assert
#else
#define ASSERT(x)
#endif

#define DISABLE_COPY_CONSTRUCTOR(classname)    \
	classname(const classname &that) = delete; \
	void operator=(const classname &that) = delete

#define DISABLE_COPY DISABLE_COPY_CONSTRUCTOR

#define cuSafeCall(x)                                         \
	do                                                        \
	{                                                         \
		CUresult result = x;                                  \
		if (result != CUDA_SUCCESS)                           \
		{                                                     \
			const char *msg;                                  \
			cuGetErrorName(result, &msg);                     \
			std::cerr << "\nerror: " #x " failed with error " \
					  << msg << '\n';                         \
			exit(1);                                          \
		}                                                     \
	} while (0)

#define cudaSafeCall(a)                                           \
	do                                                                \
	{                                                                 \
		if (cudaSuccess != (a))                                       \
		{                                                             \
			cudaError_t err = cudaGetLastError();                     \
			printf("Cuda runtime error %d in line %d of file %s \
: %s \n",                             \
				   err, __LINE__, __FILE__, cudaGetErrorString(err)); \
			exit(1);                                                  \
		}                                                             \
	} while (0)

#define nvrtcSafeCall(x)                                      \
    do                                                        \
    {                                                         \
        nvrtcResult result = x;                               \
        if (result != NVRTC_SUCCESS)                          \
        {                                                     \
            std::cerr << "\nerror: " #x " failed with error " \
                      << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                          \
        }                                                     \
    } while (0)


#define cudaAbortIfError cudaSafeCall
#define cuAbortIfError cuSafeCall
#define nvrtcAbortIfError nvrtcSafeCall

// utility functions
cudaError_t cudaCheckDevice()
{
	int cuDeviceCount;
	cudaSafeCall(cudaGetDeviceCount(&cuDeviceCount));
	if (cuDeviceCount == 0) return cudaErrorNoDevice;
	return cudaSuccess;
}

cudaError_t cudaPrintDevicesInfo()
{
	cudaSafeCall(cudaCheckDevice());
	int cuDeviceCount;
	cudaSafeCall(cudaGetDeviceCount(&cuDeviceCount));
	cout << "------------------ CUDA device capabilities log -----------------" << endl;
	cout << "Find " << cuDeviceCount << " device(s) in total." << endl;
	cout << "-----------------------------------------------------------------" << endl;

	for (int cuDevID = 0; cuDevID < cuDeviceCount; cuDevID++)
	{
		cudaDeviceProp cuDeviceProp;
		cudaSafeCall(cudaGetDeviceProperties(&cuDeviceProp, cuDevID));
		cout << cuDeviceProp.name << endl;
		cout << "  Avaliable                        : " <<
			(cuDeviceProp.computeMode == cudaComputeMode::cudaComputeModeProhibited ? "False ( GPU is in prohibited mode)" : "True") << endl;
		cout << "  Brief :" << endl;
		cout << "    Compute capabilities           : " << cuDeviceProp.major << "." << cuDeviceProp.minor << endl;
		cout << "    Is intergrated ?               : " << (cuDeviceProp.integrated == 0 ? "False" : "True") << endl;
		cout << "    Multiprocessor count           : " << cuDeviceProp.multiProcessorCount << endl;
		cout << "    Warp size                      : " << cuDeviceProp.warpSize << endl;
		cout << "    Clock rate              (MHz)  : " << cuDeviceProp.clockRate / 1024 << endl;
		cout << "    Memory clock rate       (MHz)  : " << cuDeviceProp.memoryClockRate / 1024 << endl;
		cout << "    Memory bus width        (bits) : " << cuDeviceProp.memoryBusWidth << endl;
		cout << "    Total global memory     (MB)   : " << cuDeviceProp.totalGlobalMem / 1024 / 1024 << endl;
		cout << "    Total constant memory   (MB)   : " << cuDeviceProp.totalConstMem / 1024 / 1024 << endl;
		cout << "  Limits :" << endl;
		cout << "    GridDim limits                 : "
			 << "(" << cuDeviceProp.maxGridSize[0] << "," << cuDeviceProp.maxGridSize[1] << "," << cuDeviceProp.maxGridSize[2] << ")" << endl;
		cout << "    BlockDim limits                : "
			 << "(" << cuDeviceProp.maxThreadsDim[0] << "," << cuDeviceProp.maxThreadsDim[1] << "," << cuDeviceProp.maxThreadsDim[2] << ")" << endl;
		cout << "    Max threads per block          : " << cuDeviceProp.maxThreadsPerBlock << endl;
		cout << "    Max threads per multiprocessor : " << cuDeviceProp.maxThreadsPerMultiProcessor << endl;
		cout << "    Max texture1D size             : " << cuDeviceProp.maxTexture1D << endl;
		cout << "    Max texture2D size             : "
			 << "(" << cuDeviceProp.maxTexture2D[0] << "," << cuDeviceProp.maxTexture2D[1] << ")" << endl;
		cout << "    Max texture3D size             : "
			 << "(" << cuDeviceProp.maxTexture3D[0] << "," << cuDeviceProp.maxTexture3D[1] << "," << cuDeviceProp.maxTexture3D[2] << ")" << endl;
		cout << "  Details :" << endl;
		cout << "    Can map host memory ?          : " << (cuDeviceProp.canMapHostMemory == 0 ? "False" : "True") << endl;
		cout << "    Concurrent managed access ?    : " << (cuDeviceProp.concurrentManagedAccess == 0 ? "False" : "True") << endl;
		cout << "    Is in multi GPU board ?        : " << (cuDeviceProp.isMultiGpuBoard == 0 ? "False" : "True") << endl;
		cout << "    Async engine count             : " << cuDeviceProp.asyncEngineCount << endl;
		cout << "    Concurrent kernels             : " << cuDeviceProp.concurrentKernels << endl;
		cout << "    Shared memory per block (Bytes): " << cuDeviceProp.sharedMemPerBlock << " (" << cuDeviceProp.sharedMemPerBlock / 1024 << "KB)" << endl;
		cout << "    32bit registers per block      : " << cuDeviceProp.regsPerBlock << endl;
		cout << "    32bit regs per multiProcessor  : " << cuDeviceProp.regsPerMultiprocessor << endl;
		cout << "    Texture pitch alignment (Bytes): " << cuDeviceProp.texturePitchAlignment << endl;
		cout << "-----------------------------------------------------------------" << endl;
	}
	return cudaSuccess;
}
