DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	bitset1 \
	data_structures1 \
	graphics1 \
	Jeffrey_Leon_partition_backtrack1 \
	l1_interfaces1 \
	orbiter_kernel_system1 \
	polish1


#	bitset1 \
#	Eigen_interface1 \




bitset1 :  ; cd BitSet; $(MAKE) bitset.o
data_structures1 :  ; cd data_structures; $(MAKE)
Eigen_interface1 :  ; cd Eigen_interface; $(MAKE)
graphics1 :  ; cd graphics; $(MAKE)
Jeffrey_Leon_partition_backtrack1 :  ; cd Jeffrey_Leon_partition_backtrack; $(MAKE)
l1_interfaces1 :  ; cd l1_interfaces; $(MAKE)
orbiter_kernel_system1 :  ; cd orbiter_kernel_system; $(MAKE)
polish1 :  ; cd polish; $(MAKE)






%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd BitSet; $(MAKE) clean
	cd data_structures; $(MAKE) clean
	cd Eigen_interface; $(MAKE) clean
	cd graphics; $(MAKE) clean
	cd Jeffrey_Leon_partition_backtrack; $(MAKE) clean
	cd l1_interfaces; $(MAKE) clean
	cd orbiter_kernel_system; $(MAKE) clean
	cd polish; $(MAKE) clean
	- rm *.o *.a 


