// geo_main.C


/* f_ignore_square gibt Eindeutigkeit der Verbindungsgeraden auf - JS 120100 */
/* s_flag[v - 1] kann als Option �bergeben werden - JS 120100 */
/* f_simple eingebaut JS 180100 */

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {

long int gl_t0;

void geo_main(int argc, char **argv,
	int &nb_GEN, int &nb_GEO, int &ticks, int &tps)
{
	int i, no = 0;
	char *p, *control_file = 0;
	int f_no_inc_files = FALSE;
	geometry_builder *B;
	int verbose_level;
	int f_no = FALSE;
	int flag_numeric = 4; /* JS 120100 */
	
	if (argc < 2) {
		cout << "usage: a.out [options] control_file" << endl;
		cout << "available options are" << endl;
		cout << "-no n         : start with input case n" << endl;
		cout << "              : this searches for a line" << endl;
		cout << "              : STARTINGPOint n in the control_file." << endl;
		cout << "-no_inc_files : do not write inc-files" << endl;
		cout << "              : (overrides the commands in the control-file)" << endl;
		cout << "-f ABCD       : with ABCD \\in \\{ T, F \\}^4" << endl;
		cout << "              : sets value for s_flag[v-1]" << endl;
		exit(1);
		}
	//gl_f_maxtime = FALSE;
	for (i = 0; i < argc; i++) {
		p = argv[i];
		if (strcmp(p, "-no") == 0) {
			f_no = TRUE;
			p = argv[i + 1];
			sscanf(p, "%d", &no);
			i++;
		}
		else if (strcmp(p, "-v") == 0) {
			sscanf(argv[i + 1], "%d", &verbose_level);
			i++;
		}
		else if (strcmp(p, "-no_inc_files") == 0) {
			f_no_inc_files = TRUE;
		}
		/* Routine zum Erkennen der Option "-f ABCD" */
		/* JS 120100 */
		else if (strcmp(p, "-f") == 0) {
			int zweipot = 8, j;
			flag_numeric = 0;
			// Die Schleife wandelt den True-False-String in eine Zahl JS 120100 */
			for (j = 0; j < strlen(argv[i + 1]); j++) {
				if (argv[i + 1][j] == 'T') {
					flag_numeric += zweipot;
				}
				zweipot /= 2;
			}
			i++;
		}
	}
	control_file = argv[argc - 1];
	cout << "control_file = " << control_file << endl;


	
	if (!f_no) {
		cout << "please use option -no <case> to select case from control file" << endl;
		exit(1);
	}
	gl_t0 = os_ticks();


	cout << "sizeof(cperm) = " << sizeof(cperm) << endl;
	cout << "sizeof(grid) = " << sizeof(grid) << endl;
	cout << "sizeof(tactical_decomposition) = " << sizeof(tactical_decomposition) << endl;
	cout << "sizeof(tdo_scheme) = " << sizeof(tdo_scheme) << endl;
	cout << "sizeof(iso_type) = " << sizeof(iso_type) << endl;
	cout << "sizeof(gen_geo) = " << sizeof(gen_geo) << endl;
	cout << "sizeof(gen_geo_conf) = " << sizeof(gen_geo_conf) << endl;
	cout << "sizeof(incidence) = " << sizeof(incidence) << endl;

	
	B = new geometry_builder;
	B->init(control_file, no, flag_numeric, f_no_inc_files, verbose_level);
	

	cout << "B->gg.main2" << endl;

	B->gg->main2(nb_GEN, nb_GEO, ticks, tps, verbose_level);

	delete B;
}

}}


