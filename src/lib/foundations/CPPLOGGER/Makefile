CC=g++
CFLAGS=-std=c++17 -Wall
INCLUDE_DIRS=-I.
LIBS=-lpthread

all: benchmark async_benchmark

benchmark: benchmark.cpp *.h
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $< -o $@ $(LIBS)
	

async_benchmark: async_benchmark.cpp *.h
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $< -o $@ $(LIBS)

	
clean:
	rm -rf ./*.o ./benchmark ./async_benchmark