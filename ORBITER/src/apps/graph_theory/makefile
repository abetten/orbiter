SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm

#.PRECIOUS: %.o



TARGETS=\
		all_cliques.out \
		all_cycles.out \
		all_rainbow_cliques.out \
		cayley_sym_n.out \
		cayley.out \
		colored_graph.out \
		create_graph.out \
		create_layered_graph_file.out \
		draw_colored_graph.out \
		draw_graph.out \
		graph.out \
		grassmann_graph.out \
		johnson_graph.out \
		layered_graph_main.out \
		nauty.out \
		paley.out \
		rainbow_cliques.out \
		sarnak.out \
		schlaefli.out \
		shrikhande.out  \
		srg.out \
		treedraw.out \
		winnie_li.out

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin




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

colored_graph.out:  colored_graph.o 
	$(DISCRETA_CPP)  colored_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_graph.out:  create_graph.o 
	$(DISCRETA_CPP)  create_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_layered_graph_file.out:  create_layered_graph_file.o ;
	$(DISCRETA_CPP)  create_layered_graph_file.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

draw_colored_graph.out:  draw_colored_graph.o 
	$(DISCRETA_CPP)  draw_colored_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

draw_graph.out:  draw_graph.o 
	$(DISCRETA_CPP)  draw_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

graph.out:  graph.o 
	$(DISCRETA_CPP)  graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

grassmann_graph.out:  grassmann_graph.o;
	$(DISCRETA_CPP)  grassmann_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

johnson_graph.out:  johnson_graph.o;
	$(DISCRETA_CPP)  johnson_graph.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

layered_graph_main.out:  layered_graph_main.o 
	$(DISCRETA_CPP)  layered_graph_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

nauty.out:  nauty.o 
	$(DISCRETA_CPP)  nauty.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

paley.out:  paley.o;
	$(DISCRETA_CPP)  paley.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rainbow_cliques.out:  rainbow_cliques.o 
	$(DISCRETA_CPP)  rainbow_cliques.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

sarnak.out:  sarnak.o;
	$(DISCRETA_CPP)  sarnak.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

schlaefli.out:  schlaefli.o;
	$(DISCRETA_CPP)  schlaefli.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

shrikhande.out:  shrikhande.o;
	$(DISCRETA_CPP)  shrikhande.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

srg.out:  srg.o;
	$(DISCRETA_CPP)  srg.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

treedraw.out:  treedraw.o 
	$(DISCRETA_CPP)  treedraw.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

winnie_li.out:  winnie_li.o;
	$(DISCRETA_CPP)  winnie_li.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




clean:
	@echo "------------------------------------------------------------------"
	@echo -n "cleaning"
	@echo -n ": "
	- rm *.o *.a *.out *~

