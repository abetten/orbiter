DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
#-march=native
LIB = 
LFLAGS = -lm

MODULES = \
		combinatorics_domain.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~

