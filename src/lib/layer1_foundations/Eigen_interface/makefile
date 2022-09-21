DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. \
			-I. -I../../ \
			-std=c++14 

# -frename-registers
LIB = 
LFLAGS = -lm 

MODULES = \
	eigen_interface.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
