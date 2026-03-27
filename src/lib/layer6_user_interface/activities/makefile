DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	interface_algebra.o \
	interface_coding_theory.o \
	interface_combinatorics.o \
	interface_cryptography.o \
	interface_povray.o \
	interface_projective.o \
	interface_symbol_table.o \
	interface_toolkit.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~




