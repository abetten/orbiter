DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	draw.o \
	mp_graphics.o \
	plot.o \
	scene.o \
	tree.o \
	tree_node.o


all : $(MODULES)

clean:
	- rm *.o *.a *~


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



