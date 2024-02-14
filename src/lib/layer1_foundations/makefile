DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm


# expression_parser_sajeeb must come before expression_parser
# because the file expression_parser_sajeeb/parse_tab.h 
# must be generated
# before it is used in expression_parser


all: \
  expression_parser_sajeeb \
	algebra \
	algebraic_geometry \
	canonical_form_classification \
	coding_theory \
	combinatorics \
	cryptography \
	data_structures \
	expression_parser \
	finite_fields \
	geometry \
	geometry_builder \
	graph_theory \
	graph_theory_nauty \
	graphics \
	Jeffrey_Leon_partition_backtrack \
	knowledge_base \
	l1_interfaces \
	linear_algebra \
	number_theory \
	orbiter_kernel_system \
	orthogonal \
	polish \
	ring_theory \
	solvers \
	BitSet




algebra : algebra2 
algebraic_geometry : algebraic_geometry2
canonical_form_classification : canonical_form_classification2
coding_theory : coding_theory2 
combinatorics : combinatorics2  
cryptography : cryptography2
data_structures : data_structures2  
expression_parser : expression_parser2  
finite_fields : finite_fields2  
geometry : geometry2 
geometry_builder : geometry_builder2 
graph_theory : graph_theory2 
graph_theory_nauty : graph_theory_nauty2 
graphics : graphics2 
Jeffrey_Leon_partition_backtrack : Jeffrey_Leon_partition_backtrack2 
knowledge_base : knowledge_base2  
l1_interfaces : l1_interfaces2  
linear_algebra : linear_algebra2  
number_theory : number_theory2  
orbiter_kernel_system : orbiter_kernel_system2  
orthogonal : orthogonal2  
polish : polish2 
ring_theory : ring_theory2  
solvers : solvers2 
BitSet	:	BitSet2
expression_parser_sajeeb : expression_parser_sajeeb2

algebra2 : ; cd algebra; $(MAKE)
algebraic_geometry2 : ; cd algebraic_geometry; $(MAKE)
canonical_form_classification2 : ; cd canonical_form_classification; $(MAKE)
coding_theory2 : ; cd coding_theory; $(MAKE)
combinatorics2 : ; cd combinatorics; $(MAKE)
cryptography2 : ; cd cryptography; $(MAKE)
data_structures2 : ; cd data_structures; $(MAKE)
expression_parser2 : ; cd expression_parser; $(MAKE)
finite_fields2 : ; cd finite_fields; $(MAKE)
geometry2 : ; cd geometry; $(MAKE)
geometry_builder2 : ; cd geometry_builder; $(MAKE)
graph_theory2 : ; cd graph_theory; $(MAKE)
graph_theory_nauty2 : ; cd graph_theory_nauty; $(MAKE)
graphics2 : ; cd graphics; $(MAKE)
Jeffrey_Leon_partition_backtrack2 : ; cd Jeffrey_Leon_partition_backtrack; $(MAKE)
knowledge_base2 : ; cd knowledge_base; $(MAKE)
l1_interfaces2 : ; cd l1_interfaces; $(MAKE)
linear_algebra2 : ; cd linear_algebra; $(MAKE)
number_theory2 : ; cd number_theory; $(MAKE)
orbiter_kernel_system2 : ; cd orbiter_kernel_system; $(MAKE)
orthogonal2 : ; cd orthogonal; $(MAKE)
polish2 : ; cd polish; $(MAKE)
ring_theory2 : ; cd ring_theory; $(MAKE)
solvers2 : ; cd solvers; $(MAKE)
BitSet2: ; cd BitSet; $(MAKE)
expression_parser_sajeeb2:
	$(MAKE) -C expression_parser_sajeeb all

clean:
	cd algebra; $(MAKE) clean
	cd algebraic_geometry; $(MAKE) clean
	cd canonical_form_classification; $(MAKE) clean
	cd coding_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd cryptography; $(MAKE) clean
	cd data_structures; $(MAKE) clean
	cd Eigen_interface; $(MAKE) clean
	cd expression_parser; $(MAKE) clean
	cd expression_parser_sajeeb; $(MAKE) clean
	cd finite_fields; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd geometry_builder; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd graph_theory_nauty; $(MAKE) clean
	cd graphics; $(MAKE) clean
	cd Jeffrey_Leon_partition_backtrack; $(MAKE) clean
	cd knowledge_base; $(MAKE) clean
	cd l1_interfaces; $(MAKE) clean
	cd linear_algebra; $(MAKE) clean
	cd number_theory; $(MAKE) clean
	cd orbiter_kernel_system; $(MAKE) clean
	cd orthogonal; $(MAKE) clean
	cd polish; $(MAKE) clean
	cd ring_theory; $(MAKE) clean
	cd solvers; $(MAKE) clean
	cd BitSet; $(MAKE) clean
	- rm *.o *.a 




