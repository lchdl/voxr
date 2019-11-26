#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

#include "nvrtc_wrapper.h"
#include "cuda_env.h"
#include "timer.h"

using namespace std;

struct param{
	float camera_position[3];
	float camera_look[3];
	float camera_up[3];
	int iteration;
	int w,h;
	float fov;
	string input_path;
	int X,Y,Z;
	string output_dir;
	int gpuid;
	string shader_path;
	param(){
		camera_position[0]=nanf("");
		camera_position[1]=nanf("");
		camera_position[2]=nanf("");
		camera_look[0]=nanf("");
		camera_look[1]=nanf("");
		camera_look[2]=nanf("");
		camera_up[0]=nanf("");
		camera_up[1]=nanf("");
		camera_up[2]=nanf("");
		iteration = -1;
		w=-1;h=-1;fov=-1;
		X=Y=Z=-1;
		input_path="";
		output_dir="";
		gpuid=-1;
	}
	int check_valid(){
		if(isnan(camera_position[0])||isnan(camera_position[1])
			||isnan(camera_position[2])) return 0;
		if(isnan(camera_look[0])||isnan(camera_look[1])
			||isnan(camera_look[2])) return 0;
		if(isnan(camera_up[0])||isnan(camera_up[1])
			||isnan(camera_up[2])) return 0;
		if (iteration<=0 || w<=0 || h<=0 || fov<=0) return 0;
		if (input_path.size()==0) return 0;
		if(X<0||Y<0||Z<0) return 0;
		if (output_dir.size()==0) return 0;
		if(gpuid<0) return 0;
		if(shader_path.size()==0) return 0;
		return 1;
	}
};

struct vox_decl
{
	unsigned char x,y,z,w;
};

vector<string> splitstr(string s,string delimiter)
{
	std::vector<string> words;
	size_t pos = 0;
	std::string token;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		words.push_back(token);
		s.erase(0, pos + delimiter.length());
	}
	words.push_back(s);
	return words;
}

param parse_argv(int argc,char** argv)
{
	param p;
	printf("render args:\n");
	for(int i=1; i<argc; i++){
		string arg = argv[i];
		printf("%s\n",arg.c_str());
		if(arg.size() > 0 && arg[0] == '-'){
			if(arg.size()>=2){
				string t;
				std::vector<string> words;
				switch(arg[1]){
					case 'p': // camera position
						t = arg.substr(2);
						words = splitstr(t,",");
						if(words.size()<3){
							printf("error parameter: -p.\n");
							exit(1);
						}
						p.camera_position[0] = std::stof(words[0]);
						p.camera_position[1] = std::stof(words[1]);
						p.camera_position[2] = std::stof(words[2]);
					break;
					case 'l':// camera lookpos
						t = arg.substr(2);
						words = splitstr(t,",");
						if(words.size()<3){
							printf("error parameter: -l.\n");
							exit(1);
						}
						p.camera_look[0] = std::stof(words[0]);
						p.camera_look[1] = std::stof(words[1]);
						p.camera_look[2] = std::stof(words[2]);
					break;
					case 'u':// camera upvec
						t = arg.substr(2);
						words = splitstr(t,",");
						if(words.size()<3){
							printf("error parameter: -u.\n");
							exit(1);
						}
						p.camera_up[0] = std::stof(words[0]);
						p.camera_up[1] = std::stof(words[1]);
						p.camera_up[2] = std::stof(words[2]);
					break;
					case 'i':// iterations
						t = arg.substr(2);
						p.iteration=std::stoi(t);
					break;
					case 'w':// width
						t = arg.substr(2);
						p.w=std::stoi(t);
					break;
					case 'h':// height
						t = arg.substr(2);
						p.h=std::stoi(t);
					break;
					case 'f':// fov (measured in degree)
						t = arg.substr(2);
						p.fov=std::stoi(t);
					break;
					case 'v':// voxel
						t = arg.substr(2);
						p.input_path=t;
					break;
					case 'o':// voxel
						t = arg.substr(2);
						p.output_dir=t;
					break;
					case 'X':// x dim
						t = arg.substr(2); p.X=std::stoi(t);
					break;
					case 'Y':// y dim
						t = arg.substr(2); p.Y=std::stoi(t);
					break;
					case 'Z':// z dim
						t = arg.substr(2); p.Z=std::stoi(t);
					break;
					case 'g':// gpuid
						t = arg.substr(2); p.gpuid=std::stoi(t);
					break;
					case 's':// shader path
						t = arg.substr(2); p.shader_path=t;
					break;
					default:
						printf("warning: unknown parameter '%s'\n", arg.c_str());
				}
			}
		}
	}
	return p;
}


float* alloc_render_buffer(int w,int h)
{
	int buffer_size = sizeof(float)*w*h;
	float* ptr = (float*)malloc(buffer_size);
	if(ptr) memset(ptr,0,buffer_size);
	return ptr;
}

vox_decl* alloc_voxel_buffer(int n){
	int total_voxels = n*n*n;
	int buffer_size = total_voxels*sizeof(vox_decl);
	vox_decl* ptr = (vox_decl*)malloc(buffer_size);
	if (ptr) memset(ptr,0,buffer_size);
	return ptr;
}

void save_raw(const char* file, float* data, int w,int h)
{
	FILE *fp;
	if ((fp=fopen(file, "w"))!=NULL){
		for(int y=0;y<h;y++){
			for(int x=0;x<w;x++){
				float value=data[y*w+x];
				fwrite(&value,sizeof(float),1,fp); 
			}
		}
	}
	else{
		printf("error, cannot open file '%s' for writing...\n", file);
	}
}

void print_usage()
{
	printf("=============================================================\n");
	printf("usage: ./voxr -s<shader_path> -p<camera_position> \\ \n");
	printf("            -l<camera_look_target> \\ \n");
	printf("            -u<camera_up_direction> -i<total_iterations> \\ \n");
	printf("            -w<canvas_width> -h<canvas_height> -f<fov> \\ \n");
	printf("            -v<voxel_data> \\ \n");
	printf("            -X<vox_X_dim> -Y<vox_X_dim> -Z<vox_Z_dim>\n");
	printf("            -o<output_dir> -g<gpuid>\n");
	printf("<shader_path>: a path indicating where compiled *.ptx file\n");
	printf("              are located.\n");
	printf("<camera_position>,<camera_look_target>,<camera_up_direction> \n");
	printf("          are triplet pairs x,y,z indicating a 3D position.\n");
	printf("<total_iterations>: an integer specify how many iterations \n"
		   "                will be rendered in this image.\n");
	printf("<w>,<h>: specify width and height.\n");
	printf("<fov>: specify field of view (measured in degrees).\n");
	printf("<v>: input data path.\n");
	printf("<X>,<Y>,<Z>: voxel dimensions.\n");	
	printf("<o>: output directory path.\n");
	printf("<gpuid>: GPU id.\n");
	printf("for example:\n");
	printf("./voxr -p100,100,100 -l0,0,0 -u0,0,1 -i1000 -w800 -h800 -f60 -v./input.dat -X256 -Y256 -Z256 -o./render -g0\n");
	printf("=============================================================\n");
}

int max3(int a,int b,int c)
{
	int m=a;
	if(m<b) m=b;
	if(m<c) m=c;
	return m;
}

int main(int argc,char** argv)
{
	param render_params = parse_argv(argc,argv);
	if(render_params.check_valid()==0){
		printf("some parameters are not properly set, I will stop here.\n");
		print_usage();
		exit(1);
	}
	printf("========\n");
	printf("shader path is: %s\n", render_params.shader_path.c_str());
	printf("setting camera position as (%f,%f,%f) in world coordinate.\n",
		render_params.camera_position[0],
		render_params.camera_position[1],
		render_params.camera_position[2]);
	printf("setting camera look target as (%f,%f,%f) in world coordinate.\n",
		render_params.camera_look[0],
		render_params.camera_look[1],
		render_params.camera_look[2]);
	printf("setting camera upvec as (%f,%f,%f) in world coordinate.\n",
		render_params.camera_up[0],
		render_params.camera_up[1],
		render_params.camera_up[2]);
	printf("I will render for %d iterations.\n", render_params.iteration);
	printf("Render resolution is %dx%d pixels.\n", render_params.w,render_params.h);
	printf("Field of view is %f degrees.\n", render_params.fov);
	printf("Input file is %s\n", render_params.input_path.c_str());
	printf("Voxel dim is (%d,%d,%d).\n", render_params.X,render_params.Y,render_params.Z);
	printf("Output directory is %s.\n",render_params.output_dir.c_str());
	printf("Using GPU id %d.\n",render_params.gpuid);
	/////////////////////////////////////////////////////////////////////////////
	int nvox = max3(render_params.X,render_params.Y,render_params.Z);
	int ndimz = render_params.Z, ndimy = render_params.Y, ndimx = render_params.X;
	int width = render_params.w, height = render_params.h;
	//int ndimz = 156, ndimy = 220, ndimx = 172;
	//
	cuSafeCall(cuInit(0));
	cudaSafeCall(cudaCheckDevice());
	vector<cuDevice> dev = cuDevice::GetAllDevices();
	cuContext ctx;
	ctx.CreateInDevice(&dev[render_params.gpuid]);
	cuModule module;

	string ptxcode = read_file(render_params.shader_path.c_str());
	if(ptxcode.size()==0){
		printf("error, shader %s not found.\n",render_params.shader_path.c_str());
		exit(1);
	}
	else{
		printf("shader is successfully loaded.. (%d bytes)\n", int(ptxcode.size()));
	}
	module.LoadPTX(ptxcode.c_str(),"render");
/*	char* cubin_ptr = load_raw_bytes("./dda.cubin");
	if(cubin_ptr==0) {
		printf("cannot load CUDA binaries! (*.cubin)\n");
		exit(1);
	}
	else{
		int l = get_file_size("./dda.cubin");
		for(int i=0;i<l;i++){
			printf("%02X ", cubin_ptr[i]);
		}
		module.LoadCubin(cubin_ptr,"render");
		free(cubin_ptr);
	}*/

	// print kernel info
	int num_reg_per_thread = module.GetRegisterUsage();
	printf("========\n");
	printf("Each thread uses %d register(s).\n", num_reg_per_thread);
	printf("Memory usage (bytes/thread): local|shared|const=%d|%d|%d\n",
		module.GetLocalMemUsage(),module.GetSharedMemUsage(),module.GetConstMemUsage());


	//
	int render_buffer_comp_size = sizeof(float)*width*height;
	int voxdata_size = sizeof(vox_decl)*nvox*nvox*nvox;
	printf("========\n");
	printf("allocating buffers for rendering...\n");
	float* rcomp_host = alloc_render_buffer(width,height);
	float* gcomp_host = alloc_render_buffer(width,height);
	float* bcomp_host = alloc_render_buffer(width,height);
	vox_decl* voxdata_host = alloc_voxel_buffer(nvox);

	FILE* fp = fopen(render_params.input_path.c_str(),"rb");
	if (fp==0) {
		printf("error, cannot open file.\n");
		exit(1);
	}
	unsigned char read_buf[4];
	for(int k=0;k<nvox;k++){
		for(int j=0;j<nvox;j++){
			for(int i=0;i<nvox;i++){
				int voxid=k*nvox*nvox+j*nvox+i;
				if (k<ndimz&&j<ndimy&&i<ndimx){
					if(fread(&read_buf,1,4,fp)!=4){
						printf("error occurred when loading file...\n");
						exit(1);
					}
					voxdata_host[voxid].x=read_buf[0];
					voxdata_host[voxid].y=read_buf[1];
					voxdata_host[voxid].z=read_buf[2];
					voxdata_host[voxid].w=read_buf[3];
				}
			}
		}
	}

	cuDevicePtr rcomp_gpu,gcomp_gpu,bcomp_gpu;
	cuDevicePtr voxdata_gpu;
	cuSafeCall(cuMemAlloc(&rcomp_gpu, render_buffer_comp_size));
	cuSafeCall(cuMemAlloc(&gcomp_gpu, render_buffer_comp_size));
	cuSafeCall(cuMemAlloc(&bcomp_gpu, render_buffer_comp_size));
	cuSafeCall(cuMemAlloc(&voxdata_gpu, voxdata_size));
	cuSafeCall(cuMemcpyHtoD(rcomp_gpu,rcomp_host,render_buffer_comp_size));
	cuSafeCall(cuMemcpyHtoD(gcomp_gpu,gcomp_host,render_buffer_comp_size));
	cuSafeCall(cuMemcpyHtoD(bcomp_gpu,bcomp_host,render_buffer_comp_size));
	cuSafeCall(cuMemcpyHtoD(voxdata_gpu,voxdata_host,voxdata_size));
	
	// setting camera parameters
	float camera_position[3] = {
		render_params.camera_position[0],
		render_params.camera_position[1],
		render_params.camera_position[2]};
	float camera_heading[3] = {
		render_params.camera_look[0]-render_params.camera_position[0],
		render_params.camera_look[1]-render_params.camera_position[1],
		render_params.camera_look[2]-render_params.camera_position[2]};
	float camera_up[3] = {
		render_params.camera_up[0],
		render_params.camera_up[1],
		render_params.camera_up[2]};
	cuDevicePtr camera_position_gpu,camera_heading_gpu,camera_up_gpu;
	cuSafeCall(cuMemAlloc(&camera_position_gpu, 3*sizeof(float)));
	cuSafeCall(cuMemAlloc(&camera_heading_gpu, 3*sizeof(float)));
	cuSafeCall(cuMemAlloc(&camera_up_gpu, 3*sizeof(float)));
	cuSafeCall(cuMemcpyHtoD(camera_position_gpu,camera_position,3*sizeof(float)));
	cuSafeCall(cuMemcpyHtoD(camera_heading_gpu,camera_heading,3*sizeof(float)));
	cuSafeCall(cuMemcpyHtoD(camera_up_gpu,camera_up,3*sizeof(float)));

	printf("Begin rendering...\n");
	int grid_dim_x = width/16+1,grid_dim_y=height/16+1;
	timer global_tm;
	global_tm.tick(false);
	int total_pass = render_params.iteration;
	for(int pass = 1;pass<=total_pass;pass++){
		cuContext::GlobalSync();
		module.AddParamPtr(&pass);
		module.AddParamPtr(&width);
		module.AddParamPtr(&height);
		module.AddParamPtr(&rcomp_gpu);
		module.AddParamPtr(&gcomp_gpu);
		module.AddParamPtr(&bcomp_gpu);
		module.AddParamPtr(&nvox);
		module.AddParamPtr(&voxdata_gpu);
		module.AddParamPtr(&camera_position_gpu);
		module.AddParamPtr(&camera_heading_gpu);
		module.AddParamPtr(&camera_up_gpu);
		module.AddParamPtr(&(render_params.fov));		
		cuContext::GlobalSync();
		timer local_tm;
		local_tm.tick();
		module.Launch(grid_dim_x, grid_dim_y, 1, 16, 16, 1);
		cuContext::GlobalSync();
		double dt = local_tm.tick();
		double t = global_tm.tick(false);
		double eta = float(total_pass-pass)/float(pass)*t;
		printf("\rPASS | ITER | ELAPSED | ETA : %d/%d | %.6lfs | %ds | %ds.",
			pass,total_pass,dt,int(t),int(eta));
	}
	printf("\nRendering takes %d second(s).\n",int(global_tm.tick(false)));

	cuSafeCall(cuMemcpyDtoH(rcomp_host, rcomp_gpu, render_buffer_comp_size));
	cuSafeCall(cuMemcpyDtoH(gcomp_host, gcomp_gpu, render_buffer_comp_size));
	cuSafeCall(cuMemcpyDtoH(bcomp_host, bcomp_gpu, render_buffer_comp_size));

	module.Unload();
	cuSafeCall(cuMemFree(rcomp_gpu));
	cuSafeCall(cuMemFree(gcomp_gpu));
	cuSafeCall(cuMemFree(bcomp_gpu));
	cuSafeCall(cuMemFree(voxdata_gpu));

	cuSafeCall(cuMemFree(camera_position_gpu));
	cuSafeCall(cuMemFree(camera_heading_gpu));
	cuSafeCall(cuMemFree(camera_up_gpu));

	printf("saving render buffers...\n");
	int status = mkdir(render_params.output_dir.c_str(), 
		S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH); // 755
	if(status!=0 && errno!=EEXIST){
		printf("error, cannot create directory %s.\n", 
			render_params.output_dir.c_str());
		printf("errno:%d\n", errno);
	}
	save_raw((render_params.output_dir+"/fb_r.raw").c_str(),rcomp_host,width,height);
	save_raw((render_params.output_dir+"/fb_g.raw").c_str(),gcomp_host,width,height);
	save_raw((render_params.output_dir+"/fb_b.raw").c_str(),bcomp_host,width,height);

	free(rcomp_host);
	free(gcomp_host);
	free(bcomp_host);
	free(voxdata_host);

	return 0;
}

