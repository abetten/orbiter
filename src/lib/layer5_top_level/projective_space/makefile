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
	classification_of_varieties.o \
	input_objects_of_type_variety.o \
	object_in_projective_space_with_action.o \
	projective_space_activity_description.o \
	projective_space_activity.o \
	projective_space_global.o \
	projective_space_with_action_description.o \
	projective_space_with_action.o \
	summary_of_properties_of_objects.o

all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
