DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	arcs \
	double_sixes \
	general \
	curves





arcs :  ; cd cubic_surfaces_and_arcs; $(MAKE)
double_sixes :  ; cd cubic_surfaces_and_double_sixes; $(MAKE)
general :  ; cd cubic_surfaces_in_general; $(MAKE)
curves :  ; cd quartic_curves; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd cubic_surfaces_and_arcs; $(MAKE) clean
	cd cubic_surfaces_and_double_sixes; $(MAKE) clean
	cd cubic_surfaces_in_general; $(MAKE) clean
	cd quartic_curves; $(MAKE) clean
	- rm *.o *.a 




