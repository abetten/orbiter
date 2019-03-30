/*
 * action_on_set_partitions.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started

void do_it(int universal_set_size, int partition_class_size,
		int verbose_level);


int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_n = FALSE;
	int n = 0;
	int f_k = FALSE;
	int k = 0;

	t0 = os_ticks();


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
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		}



	if (!f_n) {
		cout << "please specify -n <n>" << endl;
		exit(1);
		}

	do_it(n, k, verbose_level);

}

void do_it(int universal_set_size, int partition_class_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_it" << endl;
		}

	action *A;
	action *A2;
	longinteger_object go;
	int goi;



	A = NEW_OBJECT(action);
	A->init_symmetric_group(universal_set_size, 0 /*verbose_level*/);
	A->group_order(go);

	goi = go.as_int();
	cout << "Created group Sym(" << universal_set_size
			<< ") of order " << goi << endl;

	A2 = A->induced_action_on_set_partitions(
			partition_class_size,
			verbose_level);

	A->Strong_gens->print_with_given_action(cout, A2);
}
