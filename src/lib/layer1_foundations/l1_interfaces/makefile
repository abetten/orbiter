DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++17
# -frename-registers
#-march=native
LIB = 
LFLAGS = -lm

MODULES = \
	interface_gap_low.o \
	interface_magma_low.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



