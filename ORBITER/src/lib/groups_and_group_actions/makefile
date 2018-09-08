DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: data_structures \
	group_actions \
	group_theory \
	induced_actions




data_structures : data_structures2 
group_actions : group_actions2 
group_theory : group_theory2 
induced_actions : induced_actions2 

data_structures2 :  ; cd data_structures; $(MAKE)
group_actions2 :  ; cd group_actions; $(MAKE)
group_theory2 :  ; cd group_theory; $(MAKE)
induced_actions2 :  ; cd induced_actions; $(MAKE)


clean:
	cd data_structures; $(MAKE) clean
	cd group_actions; $(MAKE) clean
	cd group_theory; $(MAKE) clean
	cd induced_actions; $(MAKE) clean
	- rm *.o *.a 




