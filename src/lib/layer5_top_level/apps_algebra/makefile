DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action_on_forms_activity_description.o \
	action_on_forms_activity.o \
	action_on_forms_description.o \
	action_on_forms.o \
	algebra_global_with_action.o \
	character_table_burnside.o \
	classes_of_elements_expanded.o \
	classes_of_subgroups_expanded.o \
	element_processing_description.o \
	group_theoretic_activity_description.o \
	group_theoretic_activity.o \
	orbit_of_elements.o \
	orbit_of_subgroups.o \
	polynomial_ring_activity.o \
	vector_ge_activity_description.o \
	vector_ge_activity.o \
	vector_ge_builder.o \
	young.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

