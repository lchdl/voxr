NVRTC_HEADER_DIR = ${CUDA_PATH}/include/
NVRTC_SO_DIR = ${CUDA_PATH}/lib64/

COMPILE_OPTIONS = -O3 -std=c++14 -Wall -fPIC -I${NVRTC_HEADER_DIR}
LINK_OPTIONS = -L${NVRTC_SO_DIR}

.PHONY : all clean

all : ptxc voxr

ptxc : ptxc.o
	g++ ${LINK_OPTIONS} ptxc.o -lnvrtc -lcudart -o ptxc
ptxc.o : ptxc.cpp cuda_env.h nvrtc_wrapper.h
	g++ ${COMPILE_OPTIONS} -c ptxc.cpp 

voxr : voxr.o
	g++ ${LINK_OPTIONS} voxr.o -lnvrtc -lcuda -lcudart -o voxr
voxr.o : voxr.cpp cuda_env.h nvrtc_wrapper.h timer.h
	g++ ${COMPILE_OPTIONS} -c voxr.cpp 

clean :
	rm *.o
	rm voxr
	rm ptxc

