DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. \
			-I./Clique -I../BitSet -I../../ \
			-std=c++14 

# -frename-registers
LIB = 
LFLAGS = -lm -lginac -lcln 

MODULES = \
	clique_finder.o \
	clique_finder_control.o \
	colored_graph.o \
	graph_layer.o \
	graph_node.o \
	graph_theory_domain.o \
	layered_graph.o \
	layered_graph_draw_options.o \
	rainbow_cliques.o 


all : $(MODULES)
	cd ./Clique; $(MAKE)

%.o: %.cpp %.c %.h ./Clique/*.h; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $< -lpthread

clean:
	- rm *.o *.a *~
	cd ./Clique; $(MAKE) clean
