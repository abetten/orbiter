// pentomino_5x5.cpp
//
// Anton Betten
// June 17, 2015
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;






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

	pentomino_puzzle *P;

	P = NEW_OBJECT(pentomino_puzzle);

	P->main(verbose_level);

	FREE_OBJECT(P);
}

