DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

all: algebra_and_number_theory \
	canonical_form \
	coding_theory_apps \
	combinatorics \
	geometry \
	graph_theory \
	interfaces \
	orbits \
	orthogonal \
	packings \
	projective_space \
	semifields \
	spreads \
	surfaces




algebra_and_number_theory : algebra_and_number_theory2 ; 
canonical_form : canonical_form2 ;
coding_theory_apps : coding_theory_apps2 ;
combinatorics : combinatorics2 ; 
geometry : geometry2 ; 
graph_theory : graph_theory2 ; 
interfaces : interfaces2 ; 
orbits : orbits2 ; 
orthogonal : orthogonal2 ; 
packings : packings2 ; 
projective_space : projective_space2 ; 
semifields : semifields2 ; 
spreads : spreads2 ; 
surfaces : surfaces2 ; 

algebra_and_number_theory2 :  ; cd algebra_and_number_theory; $(MAKE)
canonical_form2 : ; cd canonical_form; $(MAKE)
coding_theory_apps2 : ; cd coding_theory_apps; $(MAKE)
combinatorics2 :  ; cd combinatorics; $(MAKE)
geometry2 :  ; cd geometry; $(MAKE)
graph_theory2 :  ; cd graph_theory; $(MAKE)
interfaces2 :  ; cd interfaces; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
orthogonal2 :  ; cd orthogonal; $(MAKE)
packings2 :  ; cd packings; $(MAKE)
projective_space2 :  ; cd projective_space; $(MAKE)
semifields2 :  ; cd semifields; $(MAKE)
spreads2 :  ; cd spreads; $(MAKE)
surfaces2 :  ; cd surfaces; $(MAKE)



	



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd canonical_form; $(MAKE) clean
	cd coding_theory_apps; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd interfaces; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd orthogonal; $(MAKE) clean
	cd projective_space; $(MAKE) clean
	cd semifields; $(MAKE) clean
	cd spreads; $(MAKE) clean
	cd packings; $(MAKE) clean
	cd surfaces; $(MAKE) clean
	- rm *.o *.a 




