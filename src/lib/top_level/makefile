DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

all: algebra_and_number_theory \
	combinatorics \
	geometry \
	isomorph \
	orbits \
	solver \
	surfaces




algebra_and_number_theory : algebra_and_number_theory2 ; 
combinatorics : combinatorics2 ; 
geometry : geometry2 ; 
isomorph : isomorph2 ; 
orbits : orbits2 ; 
solver : solver2 ; 
surfaces : surfaces2 ; 
interfaces : interfaces2 ; 

algebra_and_number_theory2 :  ; cd algebra_and_number_theory; $(MAKE)
combinatorics2 :  ; cd combinatorics; $(MAKE)
geometry2 :  ; cd geometry; $(MAKE)
isomorph2 :  ; cd isomorph; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
solver2 :  ; cd solver; $(MAKE)
surfaces2 :  ; cd surfaces; $(MAKE)







%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd algebra_and_number_theory; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd isomorph; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd solver; $(MAKE) clean
	cd surfaces; $(MAKE) clean
	- rm *.o *.a 




