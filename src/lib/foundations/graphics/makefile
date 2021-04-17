DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	animate.o \
	draw_bitmap_control.o \
	draw_mod_n_description.o \
	drawable_set_of_objects.o \
	EasyBMP.o \
	graphical_output.o \
	mp_graphics.o \
	parametric_curve_point.o \
	parametric_curve.o \
	plot_tools.o \
	povray_interface.o \
	scene.o \
	scene2.o \
	tree.o \
	tree_node.o \
	video_draw_options.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
