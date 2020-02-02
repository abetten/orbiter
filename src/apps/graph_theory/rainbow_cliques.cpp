// rainbow_cliques.cpp
//
// Anton Betten
// October 11, 2018
//
//
// based on all_rainbow_cliques.cpp
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv)
{
	int i;
	os_interface Os;
	t0 = Os.os_ticks();
	int verbose_level = 0;
	int f_clique_finder_control = FALSE;
	clique_finder_control CFC;


	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-clique_finder") == 0) {
			f_clique_finder_control = TRUE;
			CFC.parse_arguments(argc - i, argv + i);
		}
	}
	if (!f_clique_finder_control) {
		cout << "please use option -clique_finder" << endl;
		exit(1);
	}

	cout << "before CFC.all_cliques" << endl;
	CFC.all_cliques(verbose_level);
	cout << "after CFC.all_cliques" << endl;

	cout << "nb_sol = " << CFC.nb_sol << endl;

	cout << "rainbow_cliques.out is done" << endl;
	the_end(t0);

}
