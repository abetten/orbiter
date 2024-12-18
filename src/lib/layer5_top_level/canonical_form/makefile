DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	automorphism_group_of_variety.o \
	canonical_form_classifier_description.o \
	canonical_form_classifier.o \
	canonical_form_global.o \
	classification_of_varieties_nauty.o \
	combinatorial_object_with_properties.o \
	input_objects_of_type_variety.o \
	objects_after_classification.o \
	variety_activity_description.o \
	variety_activity.o \
	variety_compute_canonical_form.o \
	variety_object_with_action.o \
	variety_stabilizer_compute.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

