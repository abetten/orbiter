SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




all: all_k_subsets.out \
		burnside.out \
		cayley_sym_n.out \
		cayley.out \
		change_1_dollar.out \
		conjugacy_classes_sym_n.out \
		costas.out \
		design.out \
		ferdinand.out \
		fibonacci.out \
		group_ring.out \
		had.out \
		hadamard.out \
		hamming.out \
		johnson_table.out \
		kramer_mesner.out \
		lehmer.out \
		matrix_rank.out \
		nondecreasing.out \
		orthogonal.out \
		partitions.out \
		poset_of_subsets.out \
		random_permutation.out \
		random.out \
		rank_anything.out \
		rank_subsets_lex.out \
		sequences.out \
		subsets.out \
		tao.out \
		test.out \
		test2.out \
		unrank.out 


all_k_subsets.out:  all_k_subsets.o;
	$(DISCRETA_CPP)  all_k_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

burnside.out:  burnside.o;
	$(DISCRETA_CPP)  burnside.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley_sym_n.out:  cayley_sym_n.o;
	$(DISCRETA_CPP)  cayley_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley.out:  cayley.o;
	$(DISCRETA_CPP)  cayley.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

change_1_dollar.out:  change_1_dollar.o;
	$(DISCRETA_CPP)  change_1_dollar.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

conjugacy_classes_sym_n.out:  conjugacy_classes_sym_n.o;
	$(DISCRETA_CPP)  conjugacy_classes_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

costas.out:  costas.o;
	$(DISCRETA_CPP)  costas.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design.out:  design.o;
	$(DISCRETA_CPP)  design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ferdinand.out:  ferdinand.o;
	$(DISCRETA_CPP)  ferdinand.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

fibonacci.out:  fibonacci.o;
	$(DISCRETA_CPP)  fibonacci.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

group_ring.out:  group_ring.o;
	$(DISCRETA_CPP)  group_ring.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

had.out:  had.o;
	$(DISCRETA_CPP)  had.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hadamard.out:  hadamard.o;
	$(DISCRETA_CPP)  hadamard.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hamming.out:  hamming.o;
	$(DISCRETA_CPP)  hamming.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

johnson_table.out:  johnson_table.o;
	$(DISCRETA_CPP)  johnson_table.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

kramer_mesner.out:  kramer_mesner.o;
	$(DISCRETA_CPP)  kramer_mesner.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

lehmer.out:  lehmer.o;
	$(DISCRETA_CC)  lehmer.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

matrix_rank.out:  matrix_rank.o;
	$(DISCRETA_CPP)  matrix_rank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

nondecreasing.out:  nondecreasing.o;
	$(DISCRETA_CPP)  nondecreasing.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

orthogonal.out:  orthogonal.o;
	$(DISCRETA_CPP)  orthogonal.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

partitions.out: partitions.o
	$(DISCRETA_CPP) partitions.o -o partitions.out

poset_of_subsets.out:  poset_of_subsets.o;
	$(DISCRETA_CPP)  poset_of_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

random_permutation.out:  random_permutation.o;
	$(DISCRETA_CPP)  random_permutation.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

random.out:  random.o;
	$(DISCRETA_CPP)  random.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_anything.out:  rank_anything.o;
	$(DISCRETA_CPP)  rank_anything.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_subsets_lex.out:  rank_subsets_lex.o;
	$(DISCRETA_CPP)  rank_subsets_lex.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

sequences.out:  sequences.o;
	$(DISCRETA_CPP)  sequences.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subsets.out:  subsets.o;
	$(DISCRETA_CPP)  subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tao.out:  tao.o;
	$(DISCRETA_CPP)  tao.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test.out:  test.o;
	$(DISCRETA_CPP)  test.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test2.out:  test2.o;
	$(DISCRETA_CPP)  test2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_group.out:  test_group.o;
	$(DISCRETA_CPP)  test_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

unrank.out:  unrank.o;
	$(DISCRETA_CPP)  unrank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~



	
