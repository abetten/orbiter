DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	activities \
	activities_layer1 \
	activities_layer5 \
	core_system


activities :  activities2 
activities_layer1 :  activities_layer12 
activities_layer5 :  activities_layer52 
core_system : core_system2




activities2 :  ; cd activities; $(MAKE)
activities_layer12 :  ; cd activities_layer1; $(MAKE)
activities_layer52 :  ; cd activities_layer5; $(MAKE)
core_system2 :  ; cd core_system; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd activities; $(MAKE) clean
	cd activities_layer1; $(MAKE) clean
	cd activities_layer5; $(MAKE) clean
	cd core_system; $(MAKE) clean
	- rm *.o *.a 

