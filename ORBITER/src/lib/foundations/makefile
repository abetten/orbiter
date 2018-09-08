DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm


all: algebra_and_number_theory \
	coding_theory \
	combinatorics \
	data_structures \
	geometry \
	globals \
	graph_theory \
	graph_theory_nauty \
	graphics \
	io_and_os \
	solvers \
	statistics 




algebra_and_number_theory : algebra_and_number_theory2 
coding_theory : coding_theory2 
combinatorics : combinatorics2  
data_structures : data_structures2  
geometry : geometry2 
globals : globals2 
graph_theory : graph_theory2 
graph_theory_nauty : graph_theory_nauty2 
graphics : graphics2 
io_and_os : io_and_os2  
solvers : solvers2 
statistics : statistics2

algebra_and_number_theory2 : ; cd algebra_and_number_theory; $(MAKE)
coding_theory2 : ; cd coding_theory; $(MAKE)
combinatorics2 : ; cd combinatorics; $(MAKE)
data_structures2 : ; cd data_structures; $(MAKE)
geometry2 : ; cd geometry; $(MAKE)
globals2 : ; cd globals; $(MAKE)
graph_theory2 : ; cd graph_theory; $(MAKE)
graph_theory_nauty2 : ; cd graph_theory_nauty; $(MAKE)
graphics2 : ; cd graphics; $(MAKE)
io_and_os2 : ; cd io_and_os; $(MAKE)
solvers2 : ; cd solvers; $(MAKE)
statistics2 : ; cd statistics; $(MAKE)

clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd coding_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd data_structures; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd globals; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd graph_theory_nauty; $(MAKE) clean
	cd graphics; $(MAKE) clean
	cd io_and_os; $(MAKE) clean
	cd solvers; $(MAKE) clean
	cd statistics; $(MAKE) clean
	- rm *.o *.a 




