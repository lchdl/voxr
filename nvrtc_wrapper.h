#pragma once
#include <string>
#include <fstream>
#include <streambuf>
#include <libgen.h>
#include <vector>
#include <iostream>
#include "cuda_env.h"
using namespace std;

std::string read_file(const char *path)
{
	std::ifstream t(path);
	std::string str((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());
	return str;
}

class cuSource
{
  protected:
	string source;
	string name;

  public:
	cuSource(){}
	int LoadSource(const char *path)
	{
		source = read_file(path);
		if(source.size()==0) return -1;
		char *p = (char *)malloc(sizeof(char) * 4096);
		memset(p, 0, sizeof(char) * 4096);
		memcpy(p, path, strlen(path));
		name = basename(p);
		free(p);
		return 0;
	}
	void ClearSource(){
		source = "";
		name = "";
	}
	const char *GetSource() const { return source.c_str(); }
	const char *GetName() const { return name.c_str(); }
};

class cuModule
{
  protected:
	CUmodule module;
	CUfunction kernel; // entry point
	vector<void *> launch_param;
  public:
	cuModule() : module(nullptr), kernel(nullptr) {}
	~cuModule()
	{
		Unload();
	}
	
	void LoadPTX(const char* ptx,const char* entry_func_name){
		cuSafeCall(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
		cuSafeCall(cuModuleGetFunction(&kernel, module, entry_func_name));
	}

	int GetRegisterUsage(){
		int nreg=-1;
		cuSafeCall(cuFuncGetAttribute(&nreg,CU_FUNC_ATTRIBUTE_NUM_REGS,kernel));
		return nreg;
	}

	int GetSharedMemUsage(){
		int shmem_usg=-1;
		cuSafeCall(cuFuncGetAttribute(&shmem_usg,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,kernel));
		return shmem_usg;
	}

	int GetConstMemUsage(){
		int n=-1;
		cuSafeCall(cuFuncGetAttribute(&n,CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,kernel));
		return n;
	}

	int GetLocalMemUsage(){
		int n=-1;
		cuSafeCall(cuFuncGetAttribute(&n,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,kernel));
		return n;
	}
	
	void Unload()
	{
		if (module)
		{
			cuSafeCall(cuModuleUnload(module));
			module = 0;
		}
	}
	void AddParamPtr(void *param_ptr)
	{
		launch_param.push_back(param_ptr);
	}
	void ClearParams()
	{
		launch_param.clear();
	}
	void Launch(int gridDimX, int gridDimY, int gridDimZ,
				int blockDimX, int blockDimY, int blockDimZ)
	{
		cuSafeCall(cuLaunchKernel(kernel,
								  gridDimX, gridDimY, gridDimZ,
								  blockDimX, blockDimY, blockDimZ,
								  0, NULL,                         // shared mem and stream
								  launch_param.data(), 0));        // arguments
		ClearParams();
	}
};

class cuDevice
{
	friend class cuContext;

  protected:
	CUdevice device;
	string name;
	size_t totalMemBytes;

  public:
	cuDevice() : device(-1), totalMemBytes(0) {}
	void GetDevice(int index)
	{
		char buf[1024] = {0};
		cuSafeCall(cuDeviceGet(&device, index));
		cuSafeCall(cuDeviceGetName(buf, 1024, device));
		cuSafeCall(cuDeviceTotalMem(&totalMemBytes, device));
		name = buf;
	}
	cuDevice(const cuDevice &dev)
	{
		this->device = dev.device;
		this->name = dev.name;
		this->totalMemBytes = dev.totalMemBytes;
	}
	int GetHandle() { return device; }
	size_t GetTotalMemBytes() { return totalMemBytes; }

	static vector<cuDevice> GetAllDevices()
	{
		vector<cuDevice> devices;
		int count = 0;
		cuSafeCall(cuDeviceGetCount(&count));
		for (int i = 0; i < count; i++){
			cuDevice dev;
			dev.GetDevice(i);
			devices.push_back(dev);
		}
		return devices;
	}
};

class cuContext
{
  protected:
	CUcontext context;

  public:
	cuContext() : context(0) {}
	~cuContext()
	{
		Destroy();
	}

	void CreateInDevice(cuDevice *device)
	{
		cuSafeCall(cuCtxCreate(&context, 0, device->device));
	}
	void Destroy()
	{
		if (context)
		{
			cuSafeCall(cuCtxDestroy(context));
			context = 0;
		}
	}

	static void GlobalSync()
	{
		cuSafeCall(cuCtxSynchronize());
	}
};

typedef CUdeviceptr cuDevicePtr;



class cuModuleCompiler
{
  protected:
	cuSource src;
	string ptx;
	string log;
	vector<string> options;

	vector<void *> launch_param;

  public:
	cuModuleCompiler() {}
	~cuModuleCompiler()
	{
	}

	void ClearStatus()
	{
		ptx = "";
		log = "";
		options.clear();
		src.ClearSource();
	}

	// return 0 if success
	// return -1 if error
	int LoadSource(const char *path)
	{
		int b = src.LoadSource(path);
		if(b < 0){
			return -1;
		}
		return 0;
	}

	// return 0 if success
	// return -1 if error
	int Compile()
	{
		// create program
		nvrtcProgram prog = 0;
		if (strlen(src.GetSource()) == 0)
		{
			// no source files added.
			log = "== Error : Please add one source file before compiling... ==\n";
			log += "== Compilation terminated. ==";
			return -1;
		}

		string str = src.GetSource();
		string nam = src.GetName();
		const char *buffer = str.c_str();
		const char *name = nam.c_str();
		nvrtcSafeCall(
			nvrtcCreateProgram(&prog,  // prog
							   buffer, // buffer
							   name,   // name
							   0,      // numHeaders
							   NULL,   // headers
							   NULL)   // includeNames
		);
		vector<const char *> options_arr;
		for (size_t i = 0; i < options.size(); i++)
		{
			options_arr.push_back(options[i].c_str());
		}
		nvrtcResult compileResult = nvrtcCompileProgram(prog, options.size(), options_arr.data());

		// Obtain compilation log from the program.
		size_t logSize;
		nvrtcSafeCall(nvrtcGetProgramLogSize(prog, &logSize));
		char *szlog = new char[logSize];
		nvrtcSafeCall(nvrtcGetProgramLog(prog, szlog));
		log = szlog;
		delete[] szlog;
		if (compileResult != NVRTC_SUCCESS)
		{
			return -1;
		}

		// Obtain PTX from the program.
		size_t ptxSize;
		nvrtcSafeCall(nvrtcGetPTXSize(prog, &ptxSize));
		char *szptx = new char[ptxSize];
		nvrtcSafeCall(nvrtcGetPTX(prog, szptx));
		ptx = szptx;
		delete[] szptx;

		// Destroy the program.
		nvrtcSafeCall(nvrtcDestroyProgram(&prog));
		return 0;
	}
	const char *GetCompileLog()
	{
		return log.c_str();
	}
	const char *GetPTX() { return ptx.c_str(); }

	void AddCompileOption(const char *option)
	{
		string opt = option;
		options.push_back(opt);
	}
};
