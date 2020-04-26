// delandtsheer_doyen_main.cpp
//
// Anton Betten
//
// August 12, 2018
//
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

void usage(int argc, const char **argv);
int main(int argc, const char **argv);





void usage(int argc, const char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <v>                   : verbose level v" << endl;
	cout << "-d1 <d1>                 : set dimension d1" << endl;
	cout << "-q1 <q1>                 : set field size q1" << endl;
	cout << "-d2 <d2>                 : set dimension d2" << endl;
	cout << "-q2 <q2>                 : set field size q2" << endl;
}



int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_d1 = FALSE;
	int d1 = 0;
	int f_d2 = FALSE;
	int d2 = 0;
	int f_q1 = FALSE;
	int q1 = 0;
	int f_q2 = FALSE;
	int q2 = 0;
	int f_depth = FALSE;
	int depth = 0;
	int f_subgroup = FALSE;
	const char *subgroup_gens_text = NULL;
	const char *subgroup_order_text = NULL;
	const char *group_label = NULL;
	os_interface Os;

	t0 = Os.os_ticks();

	//f_memory_debug = TRUE;
	//f_memory_debug_verbose = TRUE;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-h") == 0) {
			usage(argc, argv);
			exit(1);
		}
		else if (strcmp(argv[i], "-help") == 0) {
			usage(argc, argv);
			exit(1);
		}
		else if (strcmp(argv[i], "-d1") == 0) {
			f_d1 = TRUE;
			d1 = atoi(argv[++i]);
			cout << "-d1 " << d1 << endl;
		}
		else if (strcmp(argv[i], "-d2") == 0) {
			f_d2 = TRUE;
			d2 = atoi(argv[++i]);
			cout << "-d2 " << d2 << endl;
		}
		else if (strcmp(argv[i], "-q1") == 0) {
			f_q1 = TRUE;
			q1 = atoi(argv[++i]);
			cout << "-q1 " << q1 << endl;
		}
		else if (strcmp(argv[i], "-q2") == 0) {
			f_q2 = TRUE;
			q2 = atoi(argv[++i]);
			cout << "-q2 " << q2 << endl;
		}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
		}
		else if (strcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = TRUE;
			subgroup_gens_text = argv[++i];
			subgroup_order_text = argv[++i];
			group_label = argv[++i];
			cout << "-subgroup " << subgroup_gens_text
					<< " " << subgroup_order_text
					<< " " << group_label << endl;
		}
	}
	if (!f_d1) {
		cout << "please use -d1 <d1>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_d2) {
		cout << "please use -d2 <d2>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_q1) {
		cout << "please use -q1 <q1>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_q2) {
		cout << "please use -q2 <q2>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_depth) {
		cout << "please use -depth <depth>" << endl;
		usage(argc, argv);
		exit(1);
		}


	//do_it(argc, argv, nb_factors, d, q, verbose_level);


	delandtsheer_doyen *T;

	T = NEW_OBJECT(delandtsheer_doyen);

	T->init(argc, argv, d1, q1, d2, q2,
			f_subgroup, subgroup_gens_text, subgroup_order_text, group_label,
			depth, verbose_level);


	the_end(t0);
	//the_end_quietly(t0);

}

