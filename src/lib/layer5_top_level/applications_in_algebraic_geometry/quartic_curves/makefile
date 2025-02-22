DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	quartic_curve_activity_description.o \
	quartic_curve_activity.o \
	quartic_curve_create_description.o \
	quartic_curve_create.o \
	quartic_curve_domain_with_action.o \
	quartic_curve_from_surface.o \
	quartic_curve_object_with_group.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
