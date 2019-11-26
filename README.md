# voxr
voxr: a very light-weight, simple but effective NVIDIA GPU voxel renderer used for visualizing 3D voxels, such as visualizing 3D biomedical images. Implemented with CUDA C++ and NVRTC.

https://github.com/lchdl/voxr.git

<img src="https://github.com/lchdl/voxr/blob/master/render_output_solid.png" width="300"> <img src="https://github.com/lchdl/voxr/blob/master/render_output_volumetric.png" width="300">

## The aim of voxr
1. 3D voxel data visualization;
2. Volumetric rendering.

## How to install
1. Install NVIDIA CUDA Toolkit.
2. Make sure you have set up CUDA_PATH environment variable.
   (Use export CUDA_PATH="/usr/local/cuda-...")
3. If CUDA_PATH is correctly set, then you can compile voxr by typing:
   ./install.sh
4. If no error occured during compilation, you can see a new directory has been created (./bin). It contains 4 files:
   
   ptxc : a binary executable, used for compiling CUDA source file (*.cu) into PTX codes (*.ptx).
   
   voxr : a binary executable, used for rendering 3D voxel data.
   
   render.ptx : a compiled PTX code, used for rendering.
   
   save_png.py : a Python script used for converting raw framebuffer data into a PNG file, this is the final render result.

## How to run example
1. Install this toolkit;
2. ./run_example.sh
3. After running the bash script, you will find some rendered images named as "./examples/render_output_solid.png" for demonstrating basic voxel rendering and "./examples/render_output_volumetric.png" for demonstrating volumetric rendering.

## voxr file format
Each voxel is represented by 4 bytes. The format of each voxel is : A | R | G | B

"A" is a value used for represent the transparency of the voxel. 0 means the voxel is invisible, 255 means the corresponding voxel is solid, while a value between 0~255 means the translucent voxel. Rendering a translucent voxel will have a "fog" effect.
"R,G,B" is RGB color component of the voxel. Voxel data is stored in an array structure. If you want to render a volume with 100x100x100 voxels, you should provide a binary file with size of (100x100x100)x4 bytes. The storage order of each voxel is from low dimension to high dimension. 

For example, if position (0,0,0) has a voxel with color (100,100,100) and transparency value 255, if position (0,0,1) has a voxel with color (128,255,255) and transparency value 200,... then you should write 255,100,100,100,200,128,255,255,... into your disk.

## How do these examples work?
In these examples I uploaded a full brain parcellation result, the data shape is 156x220x172 with 156 slices, 220 rows, 172 columns. Each voxel is stored from low dimension to high dimension, so voxel at position (0,0,0) is stored into disk first, then (0,0,1),(0,0,2),...,(0,0,171),(0,1,0),(0,1,1),(0,1,2),...,(0,1,171),..., until (155,219,171).
The renderer is implemented with CUDA C++ and NVRTC, using a 3D-DDA ray traversal algorithm, the rendering method is brute force path tracing, so you can still see some noises in the final rendered image even after 1000 iterations...

