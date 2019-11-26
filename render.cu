extern "C"{
// must include this header, because we need some built-in math functions and structure definitions.
//#include <__nv_nvrtc_builtin_header.h> 

struct vox_decl
{
	unsigned char x,y,z,w;
};

struct dda_trav 
{
	// constant during ray-traversal
	float oxyz[3];
	float dxyz[3];
	float invdxyz[3];
	int signxyz[3];
	// variables during ray-traversal
	float pxyz[3];
	float nxyz[3];
	int voxijk[3];
	vox_decl vd;
};

__device__ int argmax3(float* arr)
{
	float max = arr[0];
	int arg=0;
	if (arr[1]>max){max=arr[1];arg=1;}
	if (arr[2]>max){max=arr[2];arg=2;}
	return arg;
}

__device__ void norm3(float *in, float *out)
{
	float invl = 1.0f / sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
	out[0] = in[0] * invl;out[1] = in[1] * invl;out[2] = in[2] * invl;
}

__device__ float lensq3(float* in)
{
	return in[0]*in[0]+in[1]*in[1]+in[2]*in[2];
}

__device__ void cross3(float* out,float* a, float* b)
{
	out[0] = a[1]*b[2]-b[1]*a[2];
	out[1] = a[2]*b[0]-b[2]*a[0];
	out[2] = a[0]*b[1]-b[0]*a[1];
}

// state used for PRNG
struct xorwow_state {
	unsigned int a, b, c, d;
	unsigned int counter;
};

__device__ unsigned int hash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}
__device__ void xorwow_init(xorwow_state* state, int thread_id, int pass)
{
	state->a = hash((thread_id^0x7123bbcc)+(pass^0x0baabfcb));
	state->b = hash((thread_id^0xfabbcddc)+7+(pass^0xa30fb67a));
	state->c = hash((thread_id^0x0078ddcc)-23+(pass^0xffaabccb));
	state->d = hash((thread_id^0x78633ff0)+1001+(pass^0x98ab47f1));
	state->counter = hash((thread_id^0x0893ff87)-(pass^0x19d86b2d));
	// state->a = hash(thread_id+pass);
	// state->b = hash(thread_id+7+pass);
	// state->c = hash(thread_id-23+pass);
	// state->d = hash(thread_id+1001+pass);
	// state->counter = hash(thread_id+11+pass);

}
// From wikipedia
// https://en.wikipedia.org/wiki/Xorshift
// xorwow is used as default PRNG in CUDA Toolkit.
/* The state array must be initialized to not be all zero in the first four words */
__device__ unsigned int xorwow(xorwow_state* state)
{
	/* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
	unsigned int t = state->d;
	unsigned int s = state->a;
	state->d = state->c;
	state->c = state->b;
	state->b = s;
	t ^= t >> 2;
	t ^= t << 1;
	t ^= s ^ (s << 4);
	state->a = t;
	state->counter += 362437;
	return t + state->counter;
}
__device__ float xorwow_f(xorwow_state* state)
{
	return float(xorwow(state)&0x7fffffff)/float(0x80000000);
}

// generate random points on a hemisphere
__device__ void xorwow_hs(xorwow_state* state,float* result)
{
	float l;
	do{
		result[0] = (xorwow_f(state)-0.5f)*2;
		result[1] = (xorwow_f(state)-0.5f)*2;
		result[2] = (xorwow_f(state)-0.5f)*2;
		l=lensq3(result);
	}while(l>1.0f||l<0.0001f); // reject sampling
	result[2] = fabsf(result[2]);
	norm3(result,result);
}

__device__ void xorwow_hsn(xorwow_state* state,float* normal,float* result)
{
	// 3D coordinate axis
	float u[3] = {0.0f,0.0f,1.0f};
	float v[3];
	float t[3];
	cross3(t,normal,u);
	if(lensq3(t)<0.001f) { u[0]=0.0f;u[1]=1.0f;u[2]=0.0f;}
	cross3(v,normal,u);
	norm3(v,v);
	cross3(u,v,normal);
	norm3(u,u);
	// sample from hemisphere
	xorwow_hs(state,t);
	// transform t to new coordinate
	result[0] = u[0]*t[0]+v[0]*t[1]+normal[0]*t[2];
	result[1] = u[1]*t[0]+v[1]*t[1]+normal[1]*t[2];
	result[2] = u[2]*t[0]+v[2]*t[1]+normal[2]*t[2];
}

// ray-box intersect
__device__ int rbi(
	float *oxyz, float *dxyz, float *dinvxyz,
	int n, float *t0, float *t1)
{
	*t0 = 0, *t1 = 1e20f;
	float near_t, far_t;
	near_t = -oxyz[0] * dinvxyz[0];
	far_t = (n - oxyz[0]) * dinvxyz[0];
	if (near_t > far_t){float t = near_t;near_t = far_t;far_t = t;}
	*t0 = *t0 < near_t ? near_t : *t0;
	*t1 = *t1 > far_t ? far_t : *t1;
	if (*t0 > *t1) return 0;
	near_t = -oxyz[1] * dinvxyz[1];
	far_t = (n - oxyz[1]) * dinvxyz[1];
	if (near_t > far_t){float t = near_t;near_t = far_t;far_t = t;}
	*t0 = *t0 < near_t ? near_t : *t0;
	*t1 = *t1 > far_t ? far_t : *t1;
	if (*t0 > *t1) return 0;
	near_t = -oxyz[2] * dinvxyz[2];
	far_t = (n - oxyz[2]) * dinvxyz[2];
	if (near_t > far_t){float t = near_t;near_t = far_t;far_t = t;}
	*t0 = *t0 < near_t ? near_t : *t0;
	*t1 = *t1 > far_t ? far_t : *t1;
	if (*t0 > *t1) return 0;
	return 1;
}

__device__ void minxyz(
	float *x, float *y, float *z,
	float *f, int *idx)
{
	*f = *x; *idx = 0;
	if (*f > *y){*f = *y;*idx = 1;}
	if (*f > *z){*f = *z;*idx = 2;}
}

__device__ void maxxyz(
	float *x, float *y, float *z,
	float *f, int *idx)
{
	*f = *x;*idx = 0;
	if (*f < *y){*f = *y;*idx = 1;}
	if (*f < *z){*f = *z;*idx = 2;}
}

__device__ void dda_trav_init(dda_trav *trav)
{
	norm3(trav->dxyz, trav->dxyz);
	trav->invdxyz[0] = 1.0f / trav->dxyz[0];
	trav->invdxyz[1] = 1.0f / trav->dxyz[1];
	trav->invdxyz[2] = 1.0f / trav->dxyz[2];
	trav->signxyz[0] = trav->dxyz[0] > 0 ? 1 : -1;
	trav->signxyz[1] = trav->dxyz[1] > 0 ? 1 : -1;
	trav->signxyz[2] = trav->dxyz[2] > 0 ? 1 : -1;
}

__device__ int dda_trav_begin(int n, dda_trav *trav)
{
	float t0, t1;
	if (!rbi(trav->oxyz, trav->dxyz, trav->invdxyz, n, &t0, &t1)) return 0;
	trav->pxyz[0] = trav->oxyz[0] + t0 * trav->dxyz[0];
	trav->pxyz[1] = trav->oxyz[1] + t0 * trav->dxyz[1];
	trav->pxyz[2] = trav->oxyz[2] + t0 * trav->dxyz[2];
	trav->voxijk[0] = int(trav->pxyz[0]);
	trav->voxijk[1] = int(trav->pxyz[1]);
	trav->voxijk[2] = int(trav->pxyz[2]);
	if (trav->voxijk[0]<0) trav->voxijk[0] = 0;
	if (trav->voxijk[0]>n-1) trav->voxijk[0] = n-1;
	if (trav->voxijk[1]<0) trav->voxijk[1] = 0;
	if (trav->voxijk[1]>n-1) trav->voxijk[1] = n-1;
	if (trav->voxijk[2]<0) trav->voxijk[2] = 0;
	if (trav->voxijk[2]>n-1) trav->voxijk[2] = n-1;
	return 1;
}

__device__ int dda_trav_next(int n, dda_trav *trav)
{
	float voxcx = trav->voxijk[0] + 0.5f;
	float voxcy = trav->voxijk[1] + 0.5f;
	float voxcz = trav->voxijk[2] + 0.5f;
	float tx = fabsf((voxcx + 0.5f * trav->signxyz[0] - trav->pxyz[0]) * trav->invdxyz[0]);
	float ty = fabsf((voxcy + 0.5f * trav->signxyz[1] - trav->pxyz[1]) * trav->invdxyz[1]);
	float tz = fabsf((voxcz + 0.5f * trav->signxyz[2] - trav->pxyz[2]) * trav->invdxyz[2]);
	float tmin;
	int id;
	minxyz(&tx, &ty, &tz, &tmin, &id);
	trav->voxijk[id] += trav->signxyz[id];
	if (trav->voxijk[0] < 0 || trav->voxijk[0] > n-1) return 0;
	if (trav->voxijk[1] < 0 || trav->voxijk[1] > n-1) return 0;
	if (trav->voxijk[2] < 0 || trav->voxijk[2] > n-1) return 0;
	trav->pxyz[0] += tmin * trav->dxyz[0];
	trav->pxyz[1] += tmin * trav->dxyz[1];
	trav->pxyz[2] += tmin * trav->dxyz[2];
	return 1;
}

__device__ void emit_ray(
	float* camera_position, float* camera_heading, float* camera_up,
//	float camera_depth, float canvas_width, float canvas_height, 
	float fov, float px, float py, int w, int h,
	float* ray_origin, float* ray_direction) 
{
	float canvas_width = 1.0f;
	float canvas_height = float(h)/float(w);
	float theta = fov/360.0f*3.1415926f;
	float camera_depth = canvas_width*0.5f / tan(theta);
	//
	ray_origin[0] = camera_position[0];
	ray_origin[1] = camera_position[1];
	ray_origin[2] = camera_position[2];
	float canvas_center[3];
	canvas_center[0] = camera_position[0] + camera_heading[0]*camera_depth;
	canvas_center[1] = camera_position[1] + camera_heading[1]*camera_depth;
	canvas_center[2] = camera_position[2] + camera_heading[2]*camera_depth;
	float camera_right[3];
	cross3(camera_right,camera_heading,camera_up);
	float dx = (px/w-0.5f)*canvas_width;
	float dy = (0.5f-py/h)*canvas_height;
	float target[3];
	target[0] = canvas_center[0] + camera_right[0]*dx + camera_up[0]*dy;
	target[1] = canvas_center[1] + camera_right[1]*dx + camera_up[1]*dy;
	target[2] = canvas_center[2] + camera_right[2]*dx + camera_up[2]*dy;
	ray_direction[0] = target[0] - ray_origin[0];
	ray_direction[1] = target[1] - ray_origin[1];
	ray_direction[2] = target[2] - ray_origin[2];
	norm3(ray_direction,ray_direction);
}

__device__ void regularize_camera_vectors(float* front,float* up)
{
	float right[3];
	norm3(front,front);	
	cross3(right,front,up);
	norm3(right,right);
	cross3(up,right,front);
	norm3(up,up);
}

__device__ void calculate_normal(int* signxyz,float* pxyz,int* voxijk,float* normal)
{
	float voxel_center[3];
	voxel_center[0] = float(voxijk[0])+0.5f;
	voxel_center[1] = float(voxijk[1])+0.5f;
	voxel_center[2] = float(voxijk[2])+0.5f;
	float delta[3];
	delta[0] = fabsf(pxyz[0]-voxel_center[0]);
	delta[1] = fabsf(pxyz[1]-voxel_center[1]);
	delta[2] = fabsf(pxyz[2]-voxel_center[2]);
	int axis = argmax3(delta);
	normal[0] = 0.0f;
	normal[1] = 0.0f;
	normal[2] = 0.0f;
	normal[axis] = -signxyz[axis];
}

__device__ int trace(int n, dda_trav* trav, vox_decl* voxdata,xorwow_state* state)
{
	if(dda_trav_begin(n,trav)==0) return 0;
	int voxid = trav->voxijk[2]*n*n+trav->voxijk[1]*n+trav->voxijk[0];
	trav->vd = voxdata[voxid];
	// trav->vd.x resembles alpha value
	if((xorwow(state)%255)<trav->vd.x){
		calculate_normal(
			trav->signxyz,
			trav->pxyz,
			trav->voxijk,
			trav->nxyz);
		return 1;
	}
	while(dda_trav_next(n,trav)){
		int voxid = trav->voxijk[2]*n*n+trav->voxijk[1]*n+trav->voxijk[0];
		trav->vd = voxdata[voxid];
		if((xorwow(state)%255)<trav->vd.x){
			calculate_normal(
				trav->signxyz,
				trav->pxyz,
				trav->voxijk,
				trav->nxyz);
			return 1;
		}
	}
	return 0;
}

__global__ void render(
	int pass,
	int w,int h, // render buffer size
	float* rcomp, float* gcomp, float* bcomp, // target render buffer
	int n, vox_decl* voxdata, // voxel data
	float* position, float* heading, float* up, // camera settings
	float fov) // field of view
{
	int ipx = blockIdx.x*blockDim.x + threadIdx.x;
	int ipy = blockIdx.y*blockDim.y + threadIdx.y;
	float px = float(ipx);
	float py = float(ipy);
	if (px>=w||py>=h) return;
	// global thread_id, see richiesams blogspot
	int thread_id = (blockIdx.x+blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y)
		+(threadIdx.y*blockDim.x)+threadIdx.x;
	// initializing random seed, we use "xorwow" algorithm.
	xorwow_state state;
	xorwow_init(&state,thread_id,pass);
	// jitter pixel location for anti-aliasing
	px = px+xorwow_f(&state)-0.5f;
	py = py+xorwow_f(&state)-0.5f;
	// copy camera configurations
	float camera_position[3] = {position[0],position[1],position[2]};
	float camera_heading[3] = {heading[0],heading[1],heading[2]};
	float camera_up[3] = {up[0],up[1],up[2]};
	regularize_camera_vectors(camera_heading,camera_up);
	// emit initial ray from camera
	dda_trav trav;
	emit_ray(camera_position, camera_heading, camera_up, 
		fov, px, py, w, h, trav.oxyz, trav.dxyz); 
	// initialize traversal structure
	dda_trav_init(&trav);
	// initialize rendering context
	float energy[3]={1.0f,1.0f,1.0f};
	int max_depth = 12;
	int hit_anything = 0;
	// start raytrace
	for (int depth=0; depth<max_depth;depth++){
		int trace_result = trace(n,&trav,voxdata,&state);
		if(trace_result==0){
			if(hit_anything==0) {energy[0]=0;energy[1]=0;energy[2]=0;}
			else if(trav.dxyz[2]<0) {energy[0]=0;energy[1]=0;energy[2]=0;}
			else {/*...*/}
			break;
		}
		else{
			hit_anything=1;
			energy[0]=energy[0]*trav.vd.y/255.0f;
			energy[1]=energy[1]*trav.vd.z/255.0f;
			energy[2]=energy[2]*trav.vd.w/255.0f;
			// slightly move intersection point outward to avoid self-intersection
			trav.pxyz[0] = trav.pxyz[0]+trav.nxyz[0]*0.001f;
			trav.pxyz[1] = trav.pxyz[1]+trav.nxyz[1]*0.001f;
			trav.pxyz[2] = trav.pxyz[2]+trav.nxyz[2]*0.001f;
			// fill in info to prepare for next raytrace
			trav.oxyz[0] = trav.pxyz[0];trav.oxyz[1] = trav.pxyz[1];trav.oxyz[2] = trav.pxyz[2];
			xorwow_hsn(&state,trav.nxyz,trav.dxyz);
			dda_trav_init(&trav);
		}
	}
	rcomp[ipy*w+ipx]+=energy[0];
	gcomp[ipy*w+ipx]+=energy[1];
	bcomp[ipy*w+ipx]+=energy[2];
}









}