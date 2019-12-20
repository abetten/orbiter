// subprimitive.cpp
// 
// Anton Betten
//
// 5/2/2007
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;


int main(int argc, char **argv)
{
	algebra_global AG;
	int Q_max, H_max;
	
	Q_max = atoi(argv[1]);
	H_max = atoi(argv[2]);
	AG.count_subprimitive(Q_max, H_max);
}

