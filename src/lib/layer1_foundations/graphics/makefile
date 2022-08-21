DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	animate.o \
	draw_bitmap_control.o \
	draw_incidence_structure_description.o \
	draw_mod_n_description.o \
	draw_projective_curve_description.o \
	drawable_set_of_objects.o \
	EasyBMP.o \
	graphical_output.o \
	layered_graph_draw_options.o \
	mp_graphics.o \
	parametric_curve_point.o \
	parametric_curve.o \
	plot_tools.o \
	povray_interface.o \
	povray_job_description.o \
	scene_element_of_type_edge.o \
	scene_element_of_type_face.o \
	scene_element_of_type_line.o \
	scene_element_of_type_plane.o \
	scene_element_of_type_point.o \
	scene_element_of_type_surface.o \
	scene_init.o \
	scene.o \
	scene2.o \
	tree_draw_options.o \
	tree_node.o \
	tree.o \
	video_draw_options.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
