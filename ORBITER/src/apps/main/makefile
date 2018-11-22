SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I/usr/local/opt/readline/include 
LIB = $(SRC)/lib/liborbiter.a -L/usr/local/opt/readline/lib -lreadline -lm
LFLAGS = -lm


all:
	g++ $(CPPFLAGS) $(LIB) readline.cpp -o orbiter

	
clean:
	- rm orbiter
	- rm *.o
		
	