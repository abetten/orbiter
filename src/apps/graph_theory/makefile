SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -lpthread -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64

#.PRECIOUS: %.o



TARGETS=\
		all_cliques.out \
		all_cycles.out \
		all_rainbow_cliques.out \
		cayley_sym_n.out \
		cayley.out \
		draw_colored_graph.out \
		draw_graph.out \
		graph_classify_main.out \
		layered_graph_main.out \
		nauty.out \
		treedraw.out 

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin




#%.o: %.C %.cpp; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
#
#%.out: %.o; 
#	@echo "------------------------------------------------------------------"
#	@echo -n "$<"
#	@echo -n ": "
#	$(DISCRETA_CPP) $(CPPFLAGS)  $(LIB) $(LFLAGS)  $<
#	@echo "------------------------------------------------------------------"




all_cliques.out:  all_cliques.o 
	$(DISCRETA_CPP)  all_cliques.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

all_cycles.out:  all_cycles.o 
	$(DISCRETA_CPP)  all_cycles.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

all_rainbow_cliques.out:  all_rainbow_cliques.o 
	$(DISCRETA_CPP)  all_rainbow_cliques.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley_sym_n.out:  cayley_sym_n.o;
	$(DISCRETA_CPP)  cayley_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley.out:  cayley.o;
	$(DISCRETA_CPP)  cayley.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

draw_colored_graph.out:  draw_colored_graph.o 
	$(DISCRETA_CPP)  draw_colored_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

draw_graph.out:  draw_graph.o 
	$(DISCRETA_CPP)  draw_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

graph_classify_main.out:  graph_classify_main.o 
	$(DISCRETA_CPP)  graph_classify_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

layered_graph_main.out:  layered_graph_main.o 
	$(DISCRETA_CPP)  layered_graph_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

nauty.out:  nauty.o 
	$(DISCRETA_CPP)  nauty.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

treedraw.out:  treedraw.o 
	$(DISCRETA_CPP)  treedraw.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




clean:
	@echo "------------------------------------------------------------------"
	@echo -n "cleaning"
	@echo -n ": "
	- rm *.o *.a *.out *~

