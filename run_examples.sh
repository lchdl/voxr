unzip examples/full_brain_solid.dat.zip -d examples/
./bin/voxr -s./bin/render.ptx -p250,250,220 -l86,110,80 -u0,0,1 -i1000 -w800 -h800 -f60 -v./examples/full_brain_solid.dat -X172 -Y220 -Z156 -o./examples/raw_framebuffer -g0
python save_png.py -O examples/render_output_solid.png -W 800 -H 800 -D ./examples/raw_framebuffer/ -P 1000
unzip examples/full_brain_volumetric.dat.zip -d examples/
./bin/voxr -s./bin/render.ptx -p250,250,220 -l86,110,80 -u0,0,1 -i1000 -w800 -h800 -f60 -v./examples/full_brain_volumetric.dat -X172 -Y220 -Z156 -o./examples/raw_framebuffer -g0
python save_png.py -O examples/render_output_volumetric.png -W 800 -H 800 -D ./examples/raw_framebuffer/ -P 1000
