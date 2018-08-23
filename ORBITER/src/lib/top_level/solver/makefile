DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	exact_cover.o \
	exact_cover_arguments.o


all : $(MODULES)

clean:
	- rm *.o *.a *~
