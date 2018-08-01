DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	naugraph.o \
	naurng.o \
	nautil.o \
	nauty.o \
	nauty_interface.o \
	schreier.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



