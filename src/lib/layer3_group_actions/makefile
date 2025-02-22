DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	actions \
	combinatorics_with_groups \
	data_structures_groups \
	group_constructions \
	groups \
	induced_actions \
	interfaces




actions : actions2 
combinatorics_with_groups : combinatorics_with_groups2 
data_structures_groups : data_structures_groups2 
group_constructions : group_constructions2 
groups : groups2 
induced_actions : induced_actions2 
interfaces : interfaces2 

actions2 :  ; cd actions; $(MAKE)
combinatorics_with_groups2 :  ; cd combinatorics_with_groups; $(MAKE)
data_structures_groups2 :  ; cd data_structures_groups; $(MAKE)
group_constructions2 :  ; cd group_constructions; $(MAKE)
groups2 :  ; cd groups; $(MAKE)
induced_actions2 :  ; cd induced_actions; $(MAKE)
interfaces2 :  ; cd interfaces; $(MAKE)


clean:
	cd actions; $(MAKE) clean
	cd combinatorics_with_groups; $(MAKE) clean
	cd data_structures_groups; $(MAKE) clean
	cd group_constructions; $(MAKE) clean
	cd groups; $(MAKE) clean
	cd induced_actions; $(MAKE) clean
	cd interfaces; $(MAKE) clean
	- rm *.o *.a 




