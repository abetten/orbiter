DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../../.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	classification_of_cubic_surfaces_with_double_sixes_activity_description.o \
	classification_of_cubic_surfaces_with_double_sixes_activity.o \
	classify_double_sixes.o \
	surface_classify_wedge.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

