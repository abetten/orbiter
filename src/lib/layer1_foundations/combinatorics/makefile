DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	canonical_form_classification1 \
	coding_theory1 \
	cryptography1 \
	design_theory1 \
	geometry_builder1 \
	graph_theory1 \
	graph_theory_nauty1 \
	knowledge_base1 \
	other_combinatorics1 \
	puzzles1 \
	solvers1 \
	special_functions1 \
	tactical_decompositions1





canonical_form_classification1 :  ; cd canonical_form_classification; $(MAKE)
coding_theory1 :  ; cd coding_theory; $(MAKE)
cryptography1 :  ; cd cryptography; $(MAKE)
design_theory1 :  ; cd design_theory; $(MAKE)
geometry_builder1 :  ; cd geometry_builder; $(MAKE)
graph_theory1 :  ; cd graph_theory; $(MAKE)
graph_theory_nauty1 :  ; cd graph_theory_nauty; $(MAKE)
knowledge_base1 :  ; cd knowledge_base; $(MAKE)
other_combinatorics1 :  ; cd other_combinatorics; $(MAKE)
puzzles1 :  ; cd puzzles; $(MAKE)
solvers1 :  ; cd solvers; $(MAKE)
special_functions1 :  ; cd special_functions; $(MAKE)
tactical_decompositions1 :  ; cd tactical_decompositions; $(MAKE)
number_theory1 :  ; cd number_theory; $(MAKE)
ring_theory1 :  ; cd ring_theory; $(MAKE)






%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd canonical_form_classification; $(MAKE) clean
	cd coding_theory; $(MAKE) clean
	cd cryptography; $(MAKE) clean
	cd design_theory; $(MAKE) clean
	cd geometry_builder; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd graph_theory_nauty; $(MAKE) clean
	cd knowledge_base; $(MAKE) clean
	cd other_combinatorics; $(MAKE) clean
	cd puzzles; $(MAKE) clean
	cd solvers; $(MAKE) clean
	cd special_functions; $(MAKE) clean
	cd tactical_decompositions; $(MAKE) clean
	- rm *.o *.a 



