DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

all: algebra_and_number_theory \
	combinatorics \
	geometry \
	isomorph \
	orbits \
	semifields \
	solver \
	spreads_and_packings \
	surfaces




algebra_and_number_theory : algebra_and_number_theory2 ; 
combinatorics : combinatorics2 ; 
geometry : geometry2 ; 
isomorph : isomorph2 ; 
orbits : orbits2 ; 
semifields : semifields2 ; 
solver : solver2 ; 
spreads_and_packings : spreads_and_packings2 ; 
surfaces : surfaces2 ; 
interfaces : interfaces2 ; 

algebra_and_number_theory2 :  ; cd algebra_and_number_theory; $(MAKE)
combinatorics2 :  ; cd combinatorics; $(MAKE)
geometry2 :  ; cd geometry; $(MAKE)
isomorph2 :  ; cd isomorph; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
semifields2 :  ; cd semifields; $(MAKE)
solver2 :  ; cd solver; $(MAKE)
spreads_and_packings2 :  ; cd spreads_and_packings; $(MAKE)
surfaces2 :  ; cd surfaces; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd isomorph; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd semifields; $(MAKE) clean
	cd solver; $(MAKE) clean
	cd spreads_and_packings; $(MAKE) clean
	cd surfaces; $(MAKE) clean
	- rm *.o *.a 




