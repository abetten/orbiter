DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
#-march=native
LIB = 
LFLAGS = -lm

MODULES = \
		brick_domain.o \
		domino_assignment.o \
		domino_change.o \
		pentomino_puzzle.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~

