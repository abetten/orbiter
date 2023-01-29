DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	classification_of_cubic_surfaces_with_double_sixes_activity_description.o \
	classification_of_cubic_surfaces_with_double_sixes_activity.o \
	classify_double_sixes.o \
	classify_five_plus_one.o \
	surface_classify_wedge.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

