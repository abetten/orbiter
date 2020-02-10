DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	bent_function_classify.o \
	cayley_graph_search.o \
	create_graph_description.o \
	create_graph.o \
	delandtsheer_doyen.o \
	design_create_description.o \
	design_create.o \
	difference_set_in_heisenberg_group.o \
	graph_classify.o \
	hadamard_classify.o \
	hall_system_classify.o \
	large_set_classify.o \
	pentomino_puzzle.o \
	regular_ls_classify.o \
	tactical_decomposition.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

