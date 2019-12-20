SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm






TARGETS=\
		all_k_subsets.out \
		bent.out \
		conjugacy_classes_sym_n.out \
		costas.out \
		create_object.out \
		delandtsheer_doyen_main.out \
		design_create_main.out \
		design.out \
		ferdinand.out \
		group_ring.out \
		hadamard.out \
		hall_system_main.out \
		kramer_mesner.out \
		poset_of_subsets.out \
		random_permutation.out \
		rank_anything.out \
		rank_subsets_lex.out \
		tao.out \
		unrank.out 

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin


all_k_subsets.out:  all_k_subsets.o;
	$(DISCRETA_CPP)  all_k_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

bent.out:  bent.o;
	$(DISCRETA_CPP)  bent.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

conjugacy_classes_sym_n.out:  conjugacy_classes_sym_n.o;
	$(DISCRETA_CPP)  conjugacy_classes_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

costas.out:  costas.o;
	$(DISCRETA_CPP)  costas.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_object.out:  create_object.o;
	$(DISCRETA_CPP)  create_object.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design_create_main.out:  design_create_main.o;
	$(DISCRETA_CPP)  design_create_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

delandtsheer_doyen_main.out:  delandtsheer_doyen_main.o;
	$(DISCRETA_CPP)  delandtsheer_doyen_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design.out:  design.o;
	$(DISCRETA_CPP)  design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ferdinand.out:  ferdinand.o;
	$(DISCRETA_CPP)  ferdinand.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

group_ring.out:  group_ring.o;
	$(DISCRETA_CPP)  group_ring.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hadamard.out:  hadamard.o;
	$(DISCRETA_CPP)  hadamard.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hall_system_main.out:  hall_system_main.o;
	$(DISCRETA_CPP)  hall_system_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

kramer_mesner.out:  kramer_mesner.o;
	$(DISCRETA_CPP)  kramer_mesner.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

poset_of_subsets.out:  poset_of_subsets.o;
	$(DISCRETA_CPP)  poset_of_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

random_permutation.out:  random_permutation.o;
	$(DISCRETA_CPP)  random_permutation.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_anything.out:  rank_anything.o;
	$(DISCRETA_CPP)  rank_anything.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_subsets_lex.out:  rank_subsets_lex.o;
	$(DISCRETA_CPP)  rank_subsets_lex.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tao.out:  tao.o;
	$(DISCRETA_CPP)  tao.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_group.out:  test_group.o;
	$(DISCRETA_CPP)  test_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

unrank.out:  unrank.o;
	$(DISCRETA_CPP)  unrank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~



	
