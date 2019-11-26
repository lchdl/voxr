import numpy as np
import argparse
import os
import csv
import struct
from matplotlib.image import imsave

def file_exist(path):
	if os.path.exists(path) and os.path.isfile(path): return True
	else: return False

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-W','--width',help = 'Specify render buffer width.',type = int,required=True)
	parser.add_argument('-H','--height',help = 'Specify render buffer height.',type = int,required=True)
	parser.add_argument('-P','--iter',help = 'Specify render pass.',type = int,required=True)
	parser.add_argument('-D','--directory',help = 'Specify framebuffer directory.',type = str,required=True)
	parser.add_argument('-O','--output',help = 'Specify output file location.',type = str,required=True)
	args = parser.parse_args()
	return args

def read_component_raw(raw_file,width,height):
	fb = np.zeros([height,width])
	with open(raw_file,'rb') as f:
		for y in range(height):
			for x in range(width):
				fb[y,x] = struct.unpack('f', f.read(4))[0]
	return fb

if __name__ == '__main__':
	args = parse_args()
	buf = np.zeros([args.height,args.width,3])
	rcomp = read_component_raw(os.path.join(args.directory,'fb_r.raw'),args.width,args.height)
	gcomp = read_component_raw(os.path.join(args.directory,'fb_g.raw'),args.width,args.height)
	bcomp = read_component_raw(os.path.join(args.directory,'fb_b.raw'),args.width,args.height)
	buf[:,:,0]=rcomp
	buf[:,:,1]=gcomp
	buf[:,:,2]=bcomp
	buf = buf / args.iter
	print('saving image as png...')
	imsave(args.output,buf)
