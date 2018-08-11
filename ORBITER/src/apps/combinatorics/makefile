SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




all: test.out test2.out \
		partitions.out \
		conic.out hamming.out lehmer.out sarnak.out winnie_li.out \
		field.out design.out change_1_dollar.out \
		puzzle.out \
		pentomino_5x5.out \
		poset_of_subsets.out \
		hadamard.out \
		had.out \
		grassmann.out \
		paley.out \
		fibonacci.out \
		johnson.out \
		sequences.out \
		johnson_table.out \
		schlaefli.out \
		all_k_subsets.out \
		cayley_sym_n.out \
		cayley.out \
		shrikhande.out  \
		burnside.out \
		group_ring.out \
		matrix_rank.out \
		alphabet.out \
		alphabet2.out \
		subsets.out \
		orthogonal.out \
		srg.out \
		conjugacy_classes_sym_n.out \
		unrank.out \
		random.out \
		random_permutation.out \
		rank_subsets_lex.out \
		rank_anything.out \
		nondecreasing.out \
		nauty.out \
		tao.out \
		ferdinand.out \
		costas.out
 
#test_group.out 

test.out:  test.o;
	$(DISCRETA_CPP)  test.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test2.out:  test2.o;
	$(DISCRETA_CPP)  test2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_group.out:  test_group.o;
	$(DISCRETA_CPP)  test_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

partitions.out: partitions.C
	g++ partitions.C -o partitions.out

conic.out:  conic.o;
	$(DISCRETA_CPP)  conic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hamming.out:  hamming.o;
	$(DISCRETA_CPP)  hamming.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

lehmer.out:  lehmer.o;
	$(DISCRETA_CC)  lehmer.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

sarnak.out:  sarnak.o;
	$(DISCRETA_CPP)  sarnak.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

winnie_li.out:  winnie_li.o;
	$(DISCRETA_CPP)  winnie_li.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

field.out:  field.o;
	$(DISCRETA_CPP)  field.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design.out:  design.o;
	$(DISCRETA_CPP)  design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

change_1_dollar.out:  change_1_dollar.o;
	$(DISCRETA_CPP)  change_1_dollar.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

puzzle.out:  puzzle.o;
	$(DISCRETA_CPP)  puzzle.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

pentomino_5x5.out:  pentomino_5x5.o;
	$(DISCRETA_CPP)  pentomino_5x5.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

poset_of_subsets.out:  poset_of_subsets.o;
	$(DISCRETA_CPP)  poset_of_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hadamard.out:  hadamard.o;
	$(DISCRETA_CPP)  hadamard.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

had.out:  had.o;
	$(DISCRETA_CPP)  had.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

grassmann.out:  grassmann.o;
	$(DISCRETA_CPP)  grassmann.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

paley.out:  paley.o;
	$(DISCRETA_CPP)  paley.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

fibonacci.out:  fibonacci.o;
	$(DISCRETA_CPP)  fibonacci.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

johnson.out:  johnson.o;
	$(DISCRETA_CPP)  johnson.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

sequences.out:  sequences.o;
	$(DISCRETA_CPP)  sequences.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

johnson_table.out:  johnson_table.o;
	$(DISCRETA_CPP)  johnson_table.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

schlaefli.out:  schlaefli.o;
	$(DISCRETA_CPP)  schlaefli.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

all_k_subsets.out:  all_k_subsets.o;
	$(DISCRETA_CPP)  all_k_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley_sym_n.out:  cayley_sym_n.o;
	$(DISCRETA_CPP)  cayley_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cayley.out:  cayley.o;
	$(DISCRETA_CPP)  cayley.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

shrikhande.out:  shrikhande.o;
	$(DISCRETA_CPP)  shrikhande.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

burnside.out:  burnside.o;
	$(DISCRETA_CPP)  burnside.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

group_ring.out:  group_ring.o;
	$(DISCRETA_CPP)  group_ring.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

matrix_rank.out:  matrix_rank.o;
	$(DISCRETA_CPP)  matrix_rank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

alphabet.out:  alphabet.o;
	$(DISCRETA_CPP)  alphabet.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

alphabet2.out:  alphabet2.o;
	$(DISCRETA_CPP)  alphabet2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subsets.out:  subsets.o;
	$(DISCRETA_CPP)  subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

orthogonal.out:  orthogonal.o;
	$(DISCRETA_CPP)  orthogonal.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

srg.out:  srg.o;
	$(DISCRETA_CPP)  srg.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

conjugacy_classes_sym_n.out:  conjugacy_classes_sym_n.o;
	$(DISCRETA_CPP)  conjugacy_classes_sym_n.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

unrank.out:  unrank.o;
	$(DISCRETA_CPP)  unrank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

random.out:  random.o;
	$(DISCRETA_CPP)  random.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

random_permutation.out:  random_permutation.o;
	$(DISCRETA_CPP)  random_permutation.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_subsets_lex.out:  rank_subsets_lex.o;
	$(DISCRETA_CPP)  rank_subsets_lex.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_anything.out:  rank_anything.o;
	$(DISCRETA_CPP)  rank_anything.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

nondecreasing.out:  nondecreasing.o;
	$(DISCRETA_CPP)  nondecreasing.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

nauty.out:  nauty.o;
	$(DISCRETA_CPP)  nauty.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tao.out:  tao.o;
	$(DISCRETA_CPP)  tao.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ferdinand.out:  ferdinand.o;
	$(DISCRETA_CPP)  ferdinand.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

costas.out:  costas.o;
	$(DISCRETA_CPP)  costas.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



	
