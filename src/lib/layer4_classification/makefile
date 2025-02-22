DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: invariant_relations \
	isomorph \
	orbits_schreier \
	poset_classification \
	set_stabilizer \
	solvers_package
	
	

invariant_relations : invariant_relations2 
isomorph : isomorph2 
orbits_schreier : orbits_schreier2 
poset_classification : poset_classification2 
set_stabilizer : set_stabilizer2 
solvers_package : solvers_package2 

invariant_relations2 :  ; cd invariant_relations; $(MAKE)
isomorph2 :  ; cd isomorph; $(MAKE)
orbits_schreier2 :  ; cd orbits_schreier; $(MAKE)
poset_classification2 :  ; cd poset_classification; $(MAKE)
set_stabilizer2 :  ; cd set_stabilizer; $(MAKE)
solvers_package2 :  ; cd solvers_package; $(MAKE)


clean:
	cd invariant_relations; $(MAKE) clean
	cd isomorph; $(MAKE) clean
	cd orbits_schreier; $(MAKE) clean
	cd poset_classification; $(MAKE) clean
	cd set_stabilizer; $(MAKE) clean
	cd solvers_package; $(MAKE) clean
	- rm *.o *.a 




