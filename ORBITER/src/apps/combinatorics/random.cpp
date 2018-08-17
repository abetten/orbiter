// random.C
// 
// Anton Betten
// January 22, 2016

#include "orbiter.h"


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}

	
	cout << "RAND_MAX=" << RAND_MAX << endl;

}

