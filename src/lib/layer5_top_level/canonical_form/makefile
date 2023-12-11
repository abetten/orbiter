DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	canonical_form_classifier_description.o \
	canonical_form_classifier.o \
	canonical_form_nauty.o \
	canonical_form_of_variety.o \
	canonical_form_substructure.o \
	classification_of_combinatorial_objects.o \
	classification_of_varieties.o \
	input_objects_of_type_variety.o \
	object_in_projective_space_with_action.o \
	object_with_properties.o \
	quartic_curve_object_with_action.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

