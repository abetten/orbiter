// colored_graph.C
// 
// Anton Betten
// July 20, 2018
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
	INT f_save = FALSE;	
	const BYTE *save_fname = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			save_fname = argv[++i];
			cout << "-save " << save_fname << endl;
			}
		}


	colored_graph CG;


	INT nb_points = 6;
	INT nb_colors = 3;
	INT point_color[] = {0,1,2,0,1,2};
	INT Adj[] = {1,1,0,1,1, 1,1,0,1, 1,1,0, 1,1, 1};

	CG.init_adjacency_upper_triangle(nb_points, nb_colors, 
		point_color, Adj, verbose_level);

	cout << "saving graph from file " << save_fname << endl;
	CG.save(save_fname, verbose_level - 1);
	cout << "created a graph with " << CG.nb_points << " points"  << endl;


	the_end(t0);
	//the_end_quietly(t0);
}

