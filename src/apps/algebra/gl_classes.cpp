// gl_classes.cpp
//
// Anton Betten
// October 21, 2013

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


int main(int argc, char **argv)
{
	os_interface Os;
	int t0 = Os.os_ticks();
	int verbose_level = 0;
	int i;
	int f_GL = FALSE;
	int q, d;
	int f_no_eigenvalue_one = FALSE;
	int f_random = FALSE;
	int f_identify_all = FALSE;
	int f_identify_one = FALSE;
	int f_group_table = FALSE;
	int f_centralizer_brute_force = FALSE;
	int f_centralizer = FALSE;
	int elt_idx = -1;
	int f_centralizer_all = FALSE;
	int f_normal_form = FALSE;
	const char *normal_form_data = NULL;
	int *data = NULL;
	int data_sz = 0;
	int f_poly = FALSE;
	const char *poly = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = TRUE;
			memory_debug_verbose_level = atoi(argv[++i]);
			cout << "-memory_debug " << memory_debug_verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
		}
		else if (strcmp(argv[i], "-GL") == 0) {
			f_GL = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-GL " << d << " " << q << endl;
		}
		else if (strcmp(argv[i], "-no_eigenvalue_one") == 0) {
			f_no_eigenvalue_one = TRUE;
			cout << "-no_eigenvalue_one" << endl;
		}
		else if (strcmp(argv[i], "-random") == 0) {
			f_random = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-random " << d << " " << q << endl;
		}
		else if (strcmp(argv[i], "-identify_all") == 0) {
			f_identify_all = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-identify_all " << d << " " << q << endl;
		}
		else if (strcmp(argv[i], "-identify_one") == 0) {
			f_identify_one = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-identify_one " << d << " " << q
					<< " " << elt_idx << endl;
		}
		else if (strcmp(argv[i], "-normal_form") == 0) {
			f_normal_form = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			normal_form_data = argv[++i];
			cout << "-normal_form " << d << " " << q
					<< " " << normal_form_data << endl;
		}
		else if (strcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-group_table " << d << " " << q << endl;
		}
		else if (strcmp(argv[i], "-centralizer_brute_force") == 0) {
			f_centralizer = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-centralizer_brute_force " << d << " "
					<< q << " " << elt_idx << endl;
		}
		else if (strcmp(argv[i], "-centralizer") == 0) {
			f_centralizer = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-centralizer " << d << " " << q << " "
					<< elt_idx << endl;
		}
		else if (strcmp(argv[i], "-centralizer_all") == 0) {
			f_centralizer_all = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-centralizer_all " << d << " " << q << endl;
		}
	}
	

	if (f_GL) {

		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->classes_GL(q, d, f_no_eigenvalue_one, verbose_level);

		FREE_OBJECT(AGA);

	}

	else if (f_random) {

		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->do_random(q, d, f_no_eigenvalue_one, verbose_level);

		FREE_OBJECT(AGA);

	}
	else if (f_identify_all) {


		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);


		AGA->do_identify_all(q, d, f_no_eigenvalue_one, verbose_level);

		FREE_OBJECT(AGA);

	}

	else if (f_identify_one) {

		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->do_identify_one(q, d, f_no_eigenvalue_one,
				elt_idx, verbose_level);

		FREE_OBJECT(AGA);


	}

	else if (f_normal_form) {

		int_vec_scan(normal_form_data, data, data_sz);
		if (data_sz != d * d) {
			cout << "data_sz != d * d" << endl;
			exit(1);
			}
		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->do_normal_form(q, d, f_no_eigenvalue_one,
				data, data_sz, verbose_level);

		FREE_OBJECT(AGA);

	}

	else if (f_group_table) {
		
		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->group_table(q, d, f_poly, poly,
					f_no_eigenvalue_one, verbose_level);
		
		FREE_OBJECT(AGA);

	}
	
	else if (f_centralizer_brute_force) {
		
		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->centralizer_brute_force(q, d, elt_idx, verbose_level);

		FREE_OBJECT(AGA);

	}
	else if (f_centralizer) {

		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->centralizer(q, d, elt_idx, verbose_level);

		FREE_OBJECT(AGA);

	}

	else if (f_centralizer_all) {


		algebra_global_with_action *AGA;

		AGA = NEW_OBJECT(algebra_global_with_action);

		AGA->centralizer(q, d, verbose_level);

		FREE_OBJECT(AGA);

	}

	global_mem_object_registry.dump();

	
	Os.time_check(cout, t0);
	cout << endl;
}








