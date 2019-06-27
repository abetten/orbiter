// random.cpp
// 
// Anton Betten
// January 22, 2016

#include "orbiter.h"

using namespace std;


using namespace orbiter;


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}

	
	cout << "RAND_MAX=" << RAND_MAX << endl;

}

