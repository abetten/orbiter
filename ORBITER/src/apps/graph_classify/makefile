SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm






TARGETS=\
	graph.out

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin


graph.out:  graph.o ;
	$(DISCRETA_CPP)  graph.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

clean:
	- rm *.o *.a *.out *~

