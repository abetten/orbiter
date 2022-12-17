DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	linear_algebra_global.o \
	linear_algebra_RREF.o \
	linear_algebra.o \
	linear_algebra2.o \
	linear_algebra3.o \
	representation_theory_domain.o 


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



