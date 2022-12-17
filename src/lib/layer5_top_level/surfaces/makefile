DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	curves \
	arcs \
	double_sixes \
	general 





curves :  ; cd quartic_curves; $(MAKE)
arcs :  ; cd surfaces_and_arcs; $(MAKE)
double_sixes :  ; cd surfaces_and_double_sixes; $(MAKE)
general :  ; cd surfaces_general; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd quartic_curves; $(MAKE) clean
	cd surfaces_and_arcs; $(MAKE) clean
	cd surfaces_and_double_sixes; $(MAKE) clean
	cd surfaces_general; $(MAKE) clean
	- rm *.o *.a 




