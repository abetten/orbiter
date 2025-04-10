DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
#-march=native 
LIB = 
LFLAGS = -lm

MODULES = \
		any_combinatorial_object.o \
		classification_of_objects_description.o \
		classification_of_objects.o \
		classify_bitvectors.o \
		classify_using_canonical_forms.o \
		data_input_stream_description_element.o \
		data_input_stream_description.o \
		data_input_stream_output.o \
		data_input_stream.o \
		encoded_combinatorial_object.o \
		objects_report_options.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

