// graph.C
// 
// Anton Betten
// April 16, 2018
//
// 
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, char **argv)
{
	INT i;
	t0 = os_ticks();
	INT verbose_level = 0;
	INT f_file = FALSE;	
	const char *fname = NULL;
	INT f_sort_by_colors = FALSE;


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
	
	delete CG;
}


