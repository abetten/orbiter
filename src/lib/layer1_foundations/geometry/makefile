DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	finite_geometries1 \
	other_geometry1 \
	projective_geometry1





finite_geometries1 :  ; cd finite_geometries; $(MAKE)
other_geometry1 :  ; cd other_geometry; $(MAKE)
projective_geometry1 :  ; cd projective_geometry; $(MAKE)






%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd finite_geometries; $(MAKE) clean
	cd other_geometry; $(MAKE) clean
	cd projective_geometry; $(MAKE) clean
	- rm *.o *.a 



