DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	compute_stabilizer.o \
	set_stabilizer_compute.o



all : $(MODULES)

clean:
	- rm *.o *.a *~

