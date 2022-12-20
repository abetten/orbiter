DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	cubic_surface_activity_description.o \
	cubic_surface_activity.o \
	surface_clebsch_map.o \
	surface_create_description.o \
	surface_create.o \
	surface_domain_high_level.o \
	surface_object_with_action.o \
	surface_study.o \
	surface_with_action.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

