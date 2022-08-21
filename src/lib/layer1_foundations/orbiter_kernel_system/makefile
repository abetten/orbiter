DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	create_file_description.o \
	file_io.o \
	file_output.o \
	latex_interface.o \
	magma_interface.o \
	mem_object_registry_entry.o \
	mem_object_registry.o \
	memory_object.o \
	orbiter_data_file.o \
	orbiter_session.o \
	orbiter_symbol_table_entry.o \
	orbiter_symbol_table.o \
	os_interface.o \
	override_double.o \
	prepare_frames.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~




