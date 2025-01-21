DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	flag_orbit_folding.o \
	isomorph_arguments.o \
	isomorph_context.o \
	isomorph_global.o \
	isomorph_worker.o \
	isomorph.o \
	representatives.o \
	substructure_classification.o \
	substructure_lifting_data.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~


