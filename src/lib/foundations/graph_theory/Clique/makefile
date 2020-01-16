CC=g++
CFLAGS=-Wall -pthread
CSTD=-std=c++14
OPTFLAGS=-O3 -march=native -mtune=native -funroll-loops
INCLUDES=-I../../BitSet -I../../../


all: Graph.o RainbowClique.o main.o
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) RainbowClique.o Graph.o main.o -o main


main.o: main.cpp Graph.h RainbowClique.h
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) -c main.cpp -o main.o


 RainbowClique.o:  RainbowClique.h  RainbowClique.cpp
	 $(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) -c RainbowClique.cpp -o RainbowClique.o


Graph.o: Graph.h Graph.cpp
	$(CC) $(CFLAGS) $(CSTD) $(OPTFLAGS) $(INCLUDES) -c Graph.cpp -o Graph.o
	

clean:
	- rm -rf *.o main
