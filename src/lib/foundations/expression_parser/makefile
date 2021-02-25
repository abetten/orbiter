DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	expression_parser.o \
	formula.o \
	lexer.o \
	syntax_tree_node_terminal.o \
	syntax_tree_node.o \
	syntax_tree.o 



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

