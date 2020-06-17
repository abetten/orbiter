CC=g++
CFLAGS=-Wall -pthread
CSTD=-std=c++14
OPTFLAGS=-O3 -march=native -mtune=native -funroll-loops
INCLUDES=-I../../BitSet -I../../../ -I../../../group_actions/groups/


all: main main_k_clique


main: main.cpp Graph.h RainbowClique.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) $< -o $@
	

main_k_clique: main_k_clique.cpp Graph.h KClique.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) $< -o $@


clean:
	- rm -rf *.o main main_k_clique
