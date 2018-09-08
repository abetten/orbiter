DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	classification.o \
	flag_orbit_node.o \
	flag_orbits.o \
	orbit_node.o
	

all : $(MODULES)

clean:
	- rm *.o *.a *~
