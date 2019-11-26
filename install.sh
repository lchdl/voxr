#!/bin/bash
# I will first check if CUDA_PATH environment variable is set
if [[ -z "${CUDA_PATH}" ]]; then
	echo "CUDA_PATH is not set. Use : export CUDA_PATH=\"...\""
	exit 1	
else
	if [ ! -d "${CUDA_PATH}" ]; then
		echo "directory ${CUDA_PATH} not exist!"
		exit 1
	elif [ \( ! -d "${CUDA_PATH}/include" \) -o \( ! -d "${CUDA_PATH}/lib64" \) ]; then
		echo "CUDA_PATH is incorrect, some directories are missing..."
		exit 1
	fi
fi
#
echo "-----------------------"
echo "| Compiling & Linking |"
echo "-----------------------"
mkdir ./bin
make all
chmod 755 ./ptxc
chmod 755 ./voxr
cp ./ptxc ./bin/ptxc
cp ./voxr ./bin/voxr
cp ./save_png.py ./bin/save_png.py
echo "-----------------------"
echo "|   Compiling shader  |"
echo "-----------------------"
./ptxc -i./render.cu -o./render.ptx --use_fast_math --gpu-architecture=compute_61 #--std=c++14
mv ./render.ptx ./bin/render.ptx
echo "-----------------------"
echo "|       Cleaning      |"
echo "-----------------------"
make clean
