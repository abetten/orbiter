SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = $(SRC)/LIB/liborbiter.a
LFLAGS = -lm



all: graph.out

graph.out:  graph.o graph_generator.o ;
	$(DISCRETA_CPP)  graph.o graph_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<




