// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::interfaces;


int main(int argc, const char **argv)
{
	int t0;
	int i;
	int verbose_level = 0;
	int f_seed = FALSE;
	int the_seed = TRUE;
	os_interface Os;

	t0 = Os.os_ticks();
	
	cout << "Welcome to Orbiter!" << endl;
	//return 0;

#if 0
	if (argc <= 1) {
		print_usage();
		exit(1);
	}
#endif
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-seed") == 0) {
			f_seed = TRUE;
			the_seed = atoi(argv[++i]);
			cout << "-seed " << the_seed << endl;
		}
		else {
			break;
		}
	}

	int f_v = (verbose_level > 1);

	if (f_seed) {
		os_interface Os;

		cout << "seeding random number generator with " << the_seed << endl;
		srand(the_seed);
		Os.random_integer(1000);
	}

	if (f_v) {
		cout << "before Interface_cryptography" << endl;
	}
	{
		interface_cryptography Interface_cryptography;

		if (f_v) {
			cout << "before Interface_cryptography.recognize_keyword" << endl;
		}
		if (Interface_cryptography.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_cryptography.read_arguments(argc, argv, i, verbose_level);
			Interface_cryptography.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_combinatorics" << endl;
	}
	{
		interface_combinatorics Interface_combinatorics;

		if (Interface_combinatorics.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_combinatorics.read_arguments(argc, argv, i, verbose_level);
			Interface_combinatorics.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_coding_theory" << endl;
	}
	{
		interface_coding_theory Interface_coding_theory;

		if (Interface_coding_theory.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_coding_theory.read_arguments(argc, argv, i, verbose_level);
			Interface_coding_theory.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_povray" << endl;
	}
	{
		interface_povray Interface_povray;

		if (Interface_povray.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_povray.read_arguments(argc, argv, i, verbose_level);
			Interface_povray.worker(verbose_level);
		}
	}

	if (f_v) {
		cout << "before Interface_projective" << endl;
	}
	{
		interface_projective Interface_projective;

		if (Interface_projective.recognize_keyword(argc, argv, i, verbose_level)) {
			Interface_projective.read_arguments(argc, argv, i, verbose_level);
			Interface_projective.worker(verbose_level);
		}
	}



	the_end(t0);

}


