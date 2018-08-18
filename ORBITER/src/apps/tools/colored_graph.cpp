// colored_graph.C
// 
// Anton Betten
// December 1, 2017
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
	const BYTE *fname = NULL;
	INT f_export_magma = FALSE;
	INT f_export_maple = FALSE;
	INT f_print = FALSE;

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
		else if (strcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma " << endl;
			}
		else if (strcmp(argv[i], "-export_maple") == 0) {
			f_export_maple = TRUE;
			cout << "-export_maple " << endl;
			}
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print " << endl;
			}
		}


	colored_graph CG;


	cout << "loading graph from file " << fname << endl;
	CG.load(fname, verbose_level - 1);
	cout << "found a graph with " << CG.nb_points << " points"  << endl;


	if (f_export_magma) {

		cout << "export_magma" << endl;

		BYTE fname_magma[1000];
		BYTE fname_text[1000];

		strcpy(fname_magma, fname);

		strcpy(fname_text, fname);

	
		replace_extension_with(fname_magma, ".magma");
		replace_extension_with(fname_text, ".txt");

		cout << "exporting to magma as " << fname_magma << endl;


		CG.export_to_magma(fname_magma, verbose_level);

		CG.export_to_text(fname_text, verbose_level);
		
		cout << "export_magma done" << endl;
		}

	if (f_export_maple) {

		cout << "export_maple" << endl;

		BYTE fname_maple[1000];

		strcpy(fname_maple, fname);

	
		replace_extension_with(fname_maple, ".maple");

		cout << "exporting to maple as " << fname_maple << endl;


		CG.export_to_maple(fname_maple, verbose_level);

		cout << "export_maple done" << endl;
		}

	if (f_print) {
		CG.print();
		}



	the_end(t0);
	//the_end_quietly(t0);
}

