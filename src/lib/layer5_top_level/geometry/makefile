DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_generator_description.o \
	arc_generator.o \
	arc_lifting_simeon.o \
	choose_points_or_lines.o \
	classify_cubic_curves.o \
	cubic_curve_with_action.o \
	hermitian_spreads_classify.o \
	linear_set_classify.o \
	mapping_description.o \
	mapping.o \
	ovoid_classify_description.o \
	ovoid_classify.o \
	polar.o \
	search_blocking_set.o \
	singer_cycle.o \
	tensor_classify.o \
	top_level_geometry_global.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

