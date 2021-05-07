DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	canonical_form_classifier_description.o \
	canonical_form_classifier.o \
	canonical_form.o \
	compute_stabilizer.o \
	object_in_projective_space_with_action.o \
	projective_space_activity_description.o \
	projective_space_activity.o \
	projective_space_job_description.o \
	projective_space_job.o \
	projective_space_object_classifier_description.o \
	projective_space_object_classifier.o \
	projective_space_with_action_description.o \
	projective_space_with_action.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
