DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	core_system


core_system : core_system2




core_system2 :  ; cd core_system; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd core_system; $(MAKE) clean
	- rm *.o *.a 

