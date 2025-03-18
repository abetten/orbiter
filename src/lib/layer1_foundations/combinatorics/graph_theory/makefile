DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. \
			-I./Clique -I../BitSet -I../../ \
			-std=c++17 

# -frename-registers
LIB = 
LFLAGS = -lm -lginac -lcln 

MODULES = \
	clique_finder.o \
	clique_finder_control.o \
	colored_graph_cliques.o \
	colored_graph.o \
	graph_layer.o \
	graph_node.o \
	graph_theory_domain.o \
	graph_theory_subgraph_search.o \
	layered_graph.o \
	rainbow_cliques.o 



 
all : $(MODULES)
	cd ./Clique; $(MAKE) KClique.o

%.o: %.cpp %.c %.h ./Clique/*.h; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $< -lpthread

clean:
	- rm *.o *.a *~
	cd ./Clique; $(MAKE) clean
