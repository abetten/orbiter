DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	cayley_graph_search.o \
	create_graph_description.o \
	create_graph.o \
	graph_classification_activity_description.o \
	graph_classification_activity.o \
	graph_classify_description.o \
	graph_classify.o \
	graph_modification_description.o \
	graph_theoretic_activity_description.o \
	graph_theoretic_activity.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

