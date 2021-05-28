DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

all: algebra_and_number_theory \
	combinatorics \
	geometry \
	graph_theory \
	interfaces \
	isomorph \
	orbits \
	projective_space \
	quartic_curves \
	semifields \
	solver \
	spreads \
	packings \
	surfaces




algebra_and_number_theory : algebra_and_number_theory2 ; 
combinatorics : combinatorics2 ; 
geometry : geometry2 ; 
graph_theory : graph_theory2 ; 
interfaces : interfaces2 ; 
isomorph : isomorph2 ; 
orbits : orbits2 ; 
projective_space : projective_space2 ; 
quartic_curves : quartic_curves2 ; 
semifields : semifields2 ; 
solver : solver2 ; 
spreads : spreads2 ; 
packings : packings2 ; 
surfaces : surfaces2 ; 

algebra_and_number_theory2 :  ; cd algebra_and_number_theory; $(MAKE)
combinatorics2 :  ; cd combinatorics; $(MAKE)
geometry2 :  ; cd geometry; $(MAKE)
graph_theory2 :  ; cd graph_theory; $(MAKE)
interfaces2 :  ; cd interfaces; $(MAKE)
isomorph2 :  ; cd isomorph; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
projective_space2 :  ; cd projective_space; $(MAKE)
quartic_curves2 :  ; cd quartic_curves; $(MAKE)
semifields2 :  ; cd semifields; $(MAKE)
solver2 :  ; cd solver; $(MAKE)
spreads2 :  ; cd spreads; $(MAKE)
packings2 :  ; cd packings; $(MAKE)
surfaces2 :  ; cd surfaces; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd graph_theory; $(MAKE) clean
	cd interfaces; $(MAKE) clean
	cd isomorph; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd projective_space; $(MAKE) clean
	cd quartic_curves; $(MAKE) clean
	cd semifields; $(MAKE) clean
	cd solver; $(MAKE) clean
	cd spreads; $(MAKE) clean
	cd packings; $(MAKE) clean
	cd surfaces; $(MAKE) clean
	- rm *.o *.a 




