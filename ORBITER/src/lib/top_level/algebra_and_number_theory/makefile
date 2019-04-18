DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	analyze_group.o \
	factor_group.o \
	semifield_classify.o \
	semifield_downstep_node.o \
	semifield_level_two.o \
	semifield_lifting.o \
	semifield_middle_layer_node.o \
	semifield_trace.o \
	young.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~

