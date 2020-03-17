
// wreath_product_main.cpp
//
// Anton Betten, Sajeeb Roy Chowdhury
//
// August 4, 2018
//
//
//

#include "orbiter.h"

//#include <cstdint>

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
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-nb_factors <nb_factors> : set number of factors" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}


int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_nb_factors = FALSE;
	int nb_factors = 0;
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;
	int f_depth = FALSE;
	int depth = 0;
	int f_permutations = FALSE;
	int f_orbits = FALSE;
	int f_orbits_restricted = FALSE;
	const char *orbits_restricted_fname = NULL;
	int f_tensor_ranks = FALSE;
	int f_orbits_restricted_compute = FALSE;
	int f_report = FALSE;
	int f_poset_classify = FALSE;
	int poset_classify_depth = 0;
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
		else if (strcmp(argv[i], "-nb_factors") == 0) {
			f_nb_factors = TRUE;
			nb_factors = atoi(argv[++i]);
			cout << "-nb_factors " << nb_factors << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-permutations") == 0) {
			f_permutations = TRUE;
			cout << "-permutations " << endl;
			}
		else if (strcmp(argv[i], "-orbits") == 0) {
			f_orbits = TRUE;
			cout << "-orbits " << endl;
			}
		else if (strcmp(argv[i], "-orbits_restricted") == 0) {
			f_orbits_restricted = TRUE;
			orbits_restricted_fname = argv[++i];
			cout << "-orbits_restricted " << endl;
			}
		else if (strcmp(argv[i], "-tensor_ranks") == 0) {
			f_tensor_ranks = TRUE;
			cout << "-tensor_ranks " << endl;
			}
		else if (strcmp(argv[i], "-orbits_restricted_compute") == 0) {
			f_orbits_restricted_compute = TRUE;
			orbits_restricted_fname = argv[++i];
			cout << "-orbits_restricted_compute " << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
			}
		else if (strcmp(argv[i], "-poset_classify") == 0) {
			f_poset_classify = TRUE;
			poset_classify_depth = atoi(argv[++i]);
			cout << "-poset_classify " << poset_classify_depth << endl;
			}
		}
	if (!f_nb_factors) {
		cout << "please use -nb_factors <nb_factors>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_d) {
		cout << "please use -d <d>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_depth) {
		cout << "please use -depth <depth>" << endl;
		usage(argc, argv);
		exit(1);
		}




	tensor_classify *T;

	T = NEW_OBJECT(tensor_classify);

	T->init(argc, argv, nb_factors, d, q, depth,
			//f_permutations, f_orbits, f_tensor_ranks,
			//f_orbits_restricted, orbits_restricted_fname,
			//f_orbits_restricted_compute,
			0/*verbose_level*/);

	if (f_tensor_ranks) {
		cout << "before T->W->compute_tensor_ranks" << endl;
		T->W->compute_tensor_ranks(verbose_level);
		cout << "after T->W->compute_tensor_ranks" << endl;
	}

	{
		int *result = NULL;

		cout << "time check: ";
		Os.time_check(cout, t0);
		cout << endl;

		cout << "tensor_classify::init " << __FILE__ << ":" << __LINE__ << endl;

		int nb_gens, degree;

		if (f_permutations) {
			cout << "before T->W->compute_permutations_and_write_to_file" << endl;
			T->W->compute_permutations_and_write_to_file(T->SG, T->A, result,
					nb_gens, degree, nb_factors,
					verbose_level);
			cout << "after T->W->compute_permutations_and_write_to_file" << endl;
		}
		//wreath_product_orbits_CUDA(W, SG, A,
		// result, nb_gens, degree, nb_factors, verbose_level);

		if (f_orbits) {
			cout << "before T->W->orbits_using_files_and_union_find" << endl;
			T->W->orbits_using_files_and_union_find(T->SG, T->A, result, nb_gens, degree, nb_factors,
					verbose_level);
			cout << "after T->W->orbits_using_files_and_union_find" << endl;
		}
		if (f_orbits_restricted) {
			cout << "before T->W->orbits_restricted" << endl;
			T->W->orbits_restricted(T->SG, T->A, result,
					nb_gens, degree, nb_factors, orbits_restricted_fname,
					verbose_level);
			cout << "after T->W->orbits_restricted" << endl;
		}
		if (f_orbits_restricted_compute) {
			cout << "before T->W->orbits_restricted_compute" << endl;
			T->W->orbits_restricted_compute(T->SG, T->A, result,
					nb_gens, degree, nb_factors, orbits_restricted_fname,
					verbose_level);
			cout << "after T->W->orbits_restricted_compute" << endl;
		}
	}

	cout << "time check: ";
	Os.time_check(cout, t0);
	cout << endl;


	if (f_poset_classify) {
		cout << "before T->classify_poset" << endl;
		T->classify_poset(poset_classify_depth, verbose_level);
		cout << "after T->classify_poset" << endl;
	}

	if (f_report) {
		cout << "before T->report" << endl;
		T->report(f_poset_classify, poset_classify_depth,
				verbose_level);
		cout << "after T->report" << endl;
	}

	the_end_quietly(t0);

}





