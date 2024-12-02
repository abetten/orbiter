DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	activity_output.o \
	create_file_description.o \
	csv_file_support.o \
	file_io.o \
	file_output.o \
	mem_object_registry_entry.o \
	mem_object_registry.o \
	memory_object.o \
	numerics.o \
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




