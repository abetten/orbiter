// poset_of_subsets.cpp
// 
// Anton Betten
// July 2, 2016

#include "orbiter.h"

using namespace std;
using namespace orbiter;



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n;
	int f_tree = FALSE;
	int f_depth_first = FALSE;
	int f_breadth_first = FALSE;
	int f_depth = FALSE;
	int depth = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-depth_first") == 0) {
			f_depth_first = TRUE;
			cout << "-depth_first " << endl;
			}
		else if (strcmp(argv[i], "-breadth_first") == 0) {
			f_breadth_first = TRUE;
			cout << "-breadth_first " << endl;
			}
		else if (strcmp(argv[i], "-tree") == 0) {
			f_tree = TRUE;
			cout << "-tree " << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		}

	if (!f_n) {
		cout << "Please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_depth) {
		cout << "Please use option -depth <depth>" << endl;
		exit(1);
		}

	layered_graph *LG;
	char fname[1000];

	sprintf(fname, "poset_of_subsets_%d", n);
	if (f_tree) {
		sprintf(fname + strlen(fname), "_tree");
		}
	sprintf(fname + strlen(fname), ".layered_graph");
	
	LG = NEW_OBJECT(layered_graph);

	LG->make_subset_lattice(n, depth, f_tree,
		f_depth_first, f_breadth_first, verbose_level);
	
	LG->write_file(fname, 0 /*verbose_level*/);

	delete LG;
}


