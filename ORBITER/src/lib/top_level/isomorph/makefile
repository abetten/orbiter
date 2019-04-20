DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	isomorph_arguments.o \
	isomorph_database.o \
	isomorph_files.o \
	isomorph_global.o \
	isomorph_testing.o \
	isomorph_trace.o \
	isomorph.o \
	representatives.o


all : $(MODULES)

clean:
	- rm *.o *.a *~


