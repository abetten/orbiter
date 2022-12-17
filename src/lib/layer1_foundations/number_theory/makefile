DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	cyclotomic_sets.o \
	elliptic_curve.o \
	number_theoretic_transform.o \
	number_theory_domain.o 


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



