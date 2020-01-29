// graph.cpp
// 
// Anton Betten
// April 16, 2018
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started

int main(int argc, char **argv)
{
	int i;
	os_interface Os;
	t0 = Os.os_ticks();
	int verbose_level = 0;
	int f_file = FALSE;	
	const char *fname = NULL;
	int f_sort_by_colors = FALSE;
	int f_split = FALSE;
	const char *split_file = NULL;


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
		else if (strcmp(argv[i], "-sort_by_colors") == 0) {
			f_sort_by_colors = TRUE;
			cout << "-sort_by_colors " << endl;
		}
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_file = argv[++i];
			cout << "-split " << endl;
		}
	}

	if (!f_file) {
		cout << "Please specify the file name using -file <fname>" << endl;
		exit(1);
		}
	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->load(fname, verbose_level);


	if (f_sort_by_colors) {
		colored_graph *CG2;
		char fname2[1000];
		
		strcpy(fname2, fname);
		replace_extension_with(fname2, "_sorted.bin");
		CG2 = CG->sort_by_color_classes(verbose_level);
		CG2->save(fname2, verbose_level);
		delete CG2;
		}
	
	if (f_split) {
		cout << "splitting by file " << split_file << endl;
		file_io Fio;
		long int *Split;
		char fname_out[1000];
		char extension[1000];
		int m, n;
		int a, c;

		Fio.lint_matrix_read_csv(split_file, Split, m, n, verbose_level - 2);
		cout << "We found " << m << " cases for splitting" << endl;
		for (c = 0; c < m; c++) {

			cout << "splitting case " << c << " / " << m << ":" << endl;
			a = Split[2 * c + 0];

			colored_graph *Subgraph;
			fancy_set *color_subset;
			fancy_set *vertex_subset;

			Subgraph = CG->compute_neighborhood_subgraph(a,
					vertex_subset, color_subset, verbose_level);

			sprintf(fname_out, "%s", fname);
			sprintf(extension, "_case_%03d.bin", c);
			replace_extension_with(fname_out, extension);


			Subgraph->save(fname_out, verbose_level - 2);
		}
	}

	delete CG;
}


