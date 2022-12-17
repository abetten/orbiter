CC=g++
CFLAGS=-Wall -pthread
CSTD=-std=c++17
OPTFLAGS=-O3 -mtune=native -funroll-loops
INCLUDES=-I../../BitSet  -I../../../ -I../../../group_actions/groups/


all: main.out main_k_clique.out


main.out: main.cpp Graph.h RainbowClique.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) $< -o $@
	

main_k_clique.out: KClique.o main_k_clique.o
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) $^ -o $@

KClique.o: KClique.cpp KClique.h Graph.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) -c $< -o $@

main_k_clique.o: main_k_clique.cpp Graph.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) -c $< -o $@

clean:
	- rm -rf *.o *.out
