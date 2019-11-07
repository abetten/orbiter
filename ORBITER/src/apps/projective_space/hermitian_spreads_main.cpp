// hermitian_spreads_main.cpp
//
// Anton Betten
//
// started:  March 19, 2010




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;



int main(int argc, const char **argv)
{
	os_interface Os;

	int t0 = Os.os_ticks();
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_Q = FALSE;
	int Q = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n" << n << endl;
			}
		else if (strcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = atoi(argv[++i]);
			cout << "-Q" << Q << endl;
			}

		}
	if (!f_n) {
		cout << "Please specify the projective dimension using -n <n>" << endl;
		exit(1);
		}
	if (!f_Q) {
		cout << "Please specify the order of the field using -Q <Q>" << endl;
		exit(1);
		}

	hermitian_spreads_classify *HS;
	int depth;


	HS = NEW_OBJECT(hermitian_spreads_classify);

	HS->init(n, Q, verbose_level);
	HS->read_arguments(argc, argv);
	HS->init2(verbose_level);

	depth = HS->nb_pts / HS->sz;

	cout << "depth=" << depth << endl;

	HS->compute(depth, verbose_level);

	cout << "Before delete HS" << endl;
	delete HS;


	Os.time_check(cout, t0);
	cout << endl;
}


