DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	algebra_global_with_action.o \
	character_table_burnside.o \
	factor_group.o \
	finite_field_activity_description.o \
	finite_field_activity.o \
	group_theoretic_activity_description.o \
	group_theoretic_activity.o \
	group_theoretic_activity_for_surfaces.o \
	orbits_on_polynomials.o \
	projective_space_activity_description.o \
	projective_space_activity.o \
	young.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

