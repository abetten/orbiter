DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	projective_space_activity_description.o \
	projective_space_activity.o \
	projective_space_global.o \
	projective_space_with_action_description.o \
	projective_space_with_action.o \
	ring_with_action.o \
	summary_of_properties_of_objects.o

all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
