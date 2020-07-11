CC=g++
CFLAGS=-std=c++11 -Wall -O3
INCLUDES=-I ../../ -I .

all: element vector main
	$(CC) $(CFLAGS) $(INCLUDES) *.o -o ./main

element: element.cpp element.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

vector: vector.cpp vector.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

main: main.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

clean:
	rm -rf ./*.o
	rm -rf ./main