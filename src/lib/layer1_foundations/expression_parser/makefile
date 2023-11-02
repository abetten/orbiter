DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -std=c++17 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	formula_vector.o \
	formula.o \
	symbolic_object_activity_description.o \
	symbolic_object_activity.o \
	symbolic_object_builder_description.o \
	symbolic_object_builder.o \
	syntax_tree_latex.o \
	syntax_tree_node_io.o \
	syntax_tree_node_terminal.o \
	syntax_tree_node.o \
	syntax_tree.o 



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

