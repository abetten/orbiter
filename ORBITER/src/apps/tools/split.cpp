// split.cpp
//
// Anton Betten
// January 2, 2018

#include "orbiter.h"


using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int f_file = FALSE;
	const char *fname = NULL;
	int f_split = FALSE;
	int split_m;
	int i;
	os_interface Os;

	t0 = Os.os_ticks();
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

	int f_v = (verbose_level >= 1);

	char fname_base[1000];
	char fname_out[1000];

	get_fname_base(fname, fname_base);

	int *Set_sizes;
	int **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	int *Casenumbers;
	int nb_cases;
	int j, h;
	file_io Fio;

	if (f_v) {
		cout << "before read_and_parse_data_file_fancy" << endl;
		}
	Fio.read_and_parse_data_file_fancy(fname,
		FALSE /*f_casenumbers */, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);

	cout << "found " << nb_cases << " cases" << endl;
	
	ofstream *Fp;

	cout << "opening output files:" << endl;
	Fp = new ofstream[split_m];
	for (i = 0; i < split_m; i++) {
		sprintf(fname_out, "%s_split%dm%d", fname_base, i, split_m);
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




