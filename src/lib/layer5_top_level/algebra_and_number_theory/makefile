DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action_on_forms_activity_description.o \
	action_on_forms_activity.o \
	action_on_forms_description.o \
	action_on_forms.o \
	algebra_global_with_action.o \
	any_group_linear.o \
	any_group_orbits.o \
	any_group.o \
	character_table_burnside.o \
	group_modification_description.o \
	group_theoretic_activity_description.o \
	group_theoretic_activity.o \
	modified_group_create.o \
	orbit_cascade.o \
	orbits_on_polynomials.o \
	orbits_on_subspaces.o \
	polynomial_ring_activity.o \
	vector_ge_builder.o \
	young.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

