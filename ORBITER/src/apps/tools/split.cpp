// split.C
//
// Anton Betten
// January 2, 2018

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, char **argv);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT f_file = FALSE;
	const char *fname = NULL;
	INT f_split = FALSE;
	INT split_m;
	INT i;

	t0 = os_ticks();
	cout << argv[0] << endl;
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
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_m = atoi(argv[++i]);
			cout << "-split " << split_m << endl;
			}
		}
	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}

	INT f_v = (verbose_level >= 1);

	char fname_base[1000];
	char fname_out[1000];

	get_fname_base(fname, fname_base);

	INT *Set_sizes;
	INT **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	INT *Casenumbers;
	INT nb_cases;
	INT j, h;

	if (f_v) {
		cout << "before read_and_parse_data_file_fancy" << endl;
		}
	read_and_parse_data_file_fancy(fname, 
		FALSE /*f_casenumbers */, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);
		// GALOIS/util.C

	cout << "found " << nb_cases << " cases" << endl;
	
	ofstream *Fp;

	cout << "opening output files:" << endl;
	Fp = new ofstream[split_m];
	for (i = 0; i < split_m; i++) {
		sprintf(fname_out, "%s_split%ldm%ld", fname_base, i, split_m);
		Fp[i].open(fname_out);
		}
	cout << "opening output files done" << endl;

	for (i = 0; i < nb_cases; i++) {
		j = i % split_m;
		Fp[j] << Set_sizes[i];
		for (h = 0; h < Set_sizes[i]; h++) {
			Fp[j] << " " << Sets[i][h];
			}
		Fp[j] << " " << Ago_ascii[i];
		Fp[j] << " " << Aut_ascii[i];
		Fp[j] << endl;
		}


	cout << "closing output files:" << endl;
	for (i = 0; i < split_m; i++) {
		Fp[i] << "-1" << endl;
		Fp[i].close();
		}
	delete [] Fp;
	cout << "closing output files done" << endl;



	the_end(t0);

}




