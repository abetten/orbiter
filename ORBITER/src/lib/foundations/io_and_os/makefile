DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	file_output.o \
	latex_interface.o \
	memory_object.o \
	memory.o \
	orbiter_data_file.o \
	util.o


all : $(MODULES)

clean:
	- rm *.o *.a *~




