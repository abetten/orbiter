DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	diophant_activity_description.o \
	diophant_activity.o \
	finite_field_activity_description.o \
	finite_field_activity.o \
	polynomial_ring_activity_description.o \
	symbolic_object_activity_description.o \
	symbolic_object_activity.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~




