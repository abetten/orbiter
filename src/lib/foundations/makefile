DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm


all: algebra_and_number_theory \
	coding_theory \
	combinatorics \
	cryptography \
	data_structures \
	expression_parser \
	finite_fields \
	geometry \
	globals \
	graph_theory \
	graph_theory_nauty \
	graphics \
	io_and_os \
	knowledge_base \
	orthogonal \
	solvers \
	statistics \
	surfaces \
	BitSet




algebra_and_number_theory : algebra_and_number_theory2 
coding_theory : coding_theory2 
combinatorics : combinatorics2  
cryptography : cryptography2
data_structures : data_structures2  
expression_parser : expression_parser2  
finite_fields : finite_fields2  
geometry : geometry2 
globals : globals2 
graph_theory : graph_theory2 
graph_theory_nauty : graph_theory_nauty2 
graphics : graphics2 
io_and_os : io_and_os2  
knowledge_base : knowledge_base2  
orthogonal : orthogonal2  
solvers : solvers2 
statistics : statistics2
surfaces : surfaces2
BitSet	:	BitSet2

algebra_and_number_theory2 : ; cd algebra_and_number_theory; $(MAKE)
coding_theory2 : ; cd coding_theory; $(MAKE)
combinatorics2 : ; cd combinatorics; $(MAKE)
cryptography2 : ; cd cryptography; $(MAKE)
data_structures2 : ; cd data_structures; $(MAKE)
expression_parser2 : ; cd expression_parser; $(MAKE)
finite_fields2 : ; cd finite_fields; $(MAKE)
geometry2 : ; cd geometry; $(MAKE)
globals2 : ; cd globals; $(MAKE)
graph_theory2 : ; cd graph_theory; $(MAKE)
graph_theory_nauty2 : ; cd graph_theory_nauty; $(MAKE)
graphics2 : ; cd graphics; $(MAKE)
io_and_os2 : ; cd io_and_os; $(MAKE)
knowledge_base2 : ; cd knowledge_base; $(MAKE)
orthogonal2 : ; cd orthogonal; $(MAKE)
solvers2 : ; cd solvers; $(MAKE)
statistics2 : ; cd statistics; $(MAKE)
surfaces2 : ; cd surfaces; $(MAKE)
BitSet2: ; cd BitSet; $(MAKE)

clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd coding_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd cryptography; $(MAKE) clean
	cd data_structures; $(MAKE) clean
	cd expression_parser; $(MAKE) clean
	cd finite_fields; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd globals; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd graph_theory_nauty; $(MAKE) clean
	cd graphics; $(MAKE) clean
	cd io_and_os; $(MAKE) clean
	cd knowledge_base; $(MAKE) clean
	cd orthogonal; $(MAKE) clean
	cd solvers; $(MAKE) clean
	cd statistics; $(MAKE) clean
	cd surfaces; $(MAKE) clean
	cd BitSet; $(MAKE) clean
	- rm *.o *.a 




