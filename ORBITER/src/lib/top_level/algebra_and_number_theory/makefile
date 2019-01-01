DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	analyze_group.o \
	extra.o \
	factor_group.o \
	young.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~

