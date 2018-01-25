// colored_graph.C
// 
// Anton Betten
// December 1, 2017
//
// 
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started



int main(int argc, char **argv)
{
	INT i;
	t0 = os_ticks();
	INT verbose_level = 0;
	INT f_file = FALSE;	
	const BYTE *fname = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		}


	colored_graph CG;


	cout << "loading graph from file " << fname << endl;
	CG.load(fname, verbose_level - 1);
	cout << "found a graph with " << CG.nb_points << " points"  << endl;

	the_end(t0);
	//the_end_quietly(t0);
}

