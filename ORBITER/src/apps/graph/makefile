SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




all: graph.out

graph.out:  graph.o graph_generator.o ;
	$(DISCRETA_CPP)  graph.o graph_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<




