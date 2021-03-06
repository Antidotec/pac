CC	= icpc
CFLAGS	= -std=c++11 -ffast-math -O3 -fopenmp -march=native 

all: logVS

logVS: main.o
	$(CC) -o $@ $^ $(CFLAGS)

main.o: main.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean: 
	rm -f *.o
	rm -f logVS
