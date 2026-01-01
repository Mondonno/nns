gen:
	mkdir -p build && magick -density 300 main.pdf -append -crop 1x2@ +repage \
       -sampling-factor 4:2:0 -strip -quality 85 build/output_%03d.jpg