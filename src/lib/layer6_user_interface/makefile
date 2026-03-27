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
	control_everything


activities :  activities2 
activities_layer1 :  activities_layer12 
activities_layer5 :  activities_layer52 
control_everything : control_everything2




activities2 :  ; cd activities; $(MAKE)
activities_layer12 :  ; cd activities_layer1; $(MAKE)
activities_layer52 :  ; cd activities_layer5; $(MAKE)
control_everything2 :  ; cd control_everything; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd activities; $(MAKE) clean
	cd activities_layer1; $(MAKE) clean
	cd activities_layer5; $(MAKE) clean
	cd control_everything; $(MAKE) clean
	- rm *.o *.a 

