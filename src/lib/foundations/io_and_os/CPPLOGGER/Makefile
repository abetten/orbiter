CC=g++
CFLAGS=-std=c++17 -Wall -g3 -Wformat -O3
INCLUDE_DIRS=-I.
LIBS=-lpthread

all: benchmark \
		async_benchmark \
		typographic_benchmark

benchmark: benchmark.cpp *.h
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $< -o $@ $(LIBS)
	

async_benchmark: async_benchmark.cpp *.h
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $< -o $@ $(LIBS)
	
	
typographic_benchmark: typographic_benchmark.cpp *.h
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $< -o $@ $(LIBS)
	
	
clean:
	rm -vrf ./*.o ./benchmark ./async_benchmark ./typographic_benchmark