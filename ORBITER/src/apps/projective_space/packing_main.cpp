// packing_main.C
// 
// Anton Betten
// Nov 8, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

#define MAX_FILES 1000


int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_order = FALSE;
	int order = 0;
	int f_dim_over_kernel = FALSE;
	int dim_over_kernel = 0;
	int f_recoordinatize = FALSE;
	int f_select_spread = FALSE;
	int select_spread[1000];
	int select_spread_nb = 0;

	int f_starter = FALSE;

	int f_klein = FALSE;
	int f_split_klein = FALSE;
	int split_klein_r = 0;
	int split_klein_m = 1;
	int f_draw_poset = FALSE;
	int f_draw_poset_full = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;

	int f_compute_spread_table = FALSE;
	int f_fname_spread_table = FALSE;
	const char *fname_spread_table = NULL;
	int f_fname_spread_table_iso = FALSE;
	const char *fname_spread_table_iso = NULL;

#if 0
	int f_type_of_packing = FALSE;
	const char *type_of_packing_fname = NULL;
#endif
	int f_conjugacy_classes = FALSE;
	int f_conjugacy_classes_and_normalizers = FALSE;
	int f_make_element = FALSE;
	int make_element_idx = 0;
	int f_centralizer = FALSE;
	int centralizer_idx = 0;

	int f_centralizer_of_element = FALSE;
	const char *element_description = NULL;
	int f_label = FALSE;
	const char *label = NULL;

	exact_cover_arguments *ECA = NULL;
	isomorph_arguments *IA = NULL;

	ECA = NEW_OBJECT(exact_cover_arguments);
	IA = NEW_OBJECT(isomorph_arguments);

	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-label") == 0) {
			f_label = TRUE;
			label = argv[++i];
			cout << "-label " << label << endl;
			}
		else if (strcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = atoi(argv[++i]);
			cout << "-order " << order << endl;
			}
		else if (strcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = atoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
			}
		else if (strcmp(argv[i], "-recoordinatize") == 0) {
			f_recoordinatize = TRUE;
			cout << "-recoordinatize " << endl;
			}
		else if (strcmp(argv[i], "-select_spread") == 0) {
			int a;
			
			f_select_spread = TRUE;
			select_spread_nb = 0;
			while (TRUE) {
				a = atoi(argv[++i]);
				if (a == -1) {
					break;
					}
				select_spread[select_spread_nb++] = a;
				}
			cout << "-select_spread ";
			int_vec_print(cout, select_spread, select_spread_nb);
			cout << endl;
			}
		else if (strcmp(argv[i], "-starter") == 0) {
			f_starter = TRUE;
			cout << "-starter " << endl;
			}

		else if (strcmp(argv[i], "-compute_spread_table") == 0) {
			f_compute_spread_table = TRUE;
			cout << "-compute_spread_table " << endl;
			}



		else if (strcmp(argv[i], "-klein") == 0) {
			f_klein = TRUE;
			cout << "-klein " << endl;
			}
		else if (strcmp(argv[i], "-split_klein") == 0) {
			f_split_klein = TRUE;
			split_klein_r = atoi(argv[++i]);
			split_klein_m = atoi(argv[++i]);
			cout << "-split_klein " << split_klein_r
				<< " " << split_klein_m << endl;
			}


		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-draw_poset_full") == 0) {
			f_draw_poset_full = TRUE;
			cout << "-draw_poset_full " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}


		else if (strcmp(argv[i], "-fname_spread_table") == 0) {
			f_fname_spread_table = TRUE;
			fname_spread_table = argv[++i];
			cout << "-fname_spread_table " << fname_spread_table << endl;
			}
		else if (strcmp(argv[i], "-fname_spread_table_iso") == 0) {
			f_fname_spread_table_iso = TRUE;
			fname_spread_table_iso = argv[++i];
			cout << "-fname_spread_table_iso " << fname_spread_table_iso << endl;
			}
#if 0
		else if (strcmp(argv[i], "-type_of_packing") == 0) {
			f_type_of_packing = TRUE;
			type_of_packing_fname = argv[++i];
			cout << "-type_of_packing " << type_of_packing_fname << endl;
			}
#endif
		else if (strcmp(argv[i], "-conjugacy_classes") == 0) {
			f_conjugacy_classes = TRUE;
			cout << "-conjugacy_classes " << endl;
			}
		else if (strcmp(argv[i], "-conjugacy_classes_and_normalizers") == 0) {
			f_conjugacy_classes_and_normalizers = TRUE;
			cout << "-conjugacy_classes_and_normalizers " << endl;
			}
		else if (strcmp(argv[i], "-make_element") == 0) {
			f_make_element = TRUE;
			make_element_idx = atoi(argv[++i]);
			cout << "-make_element " << make_element_idx << endl;
			}
		else if (strcmp(argv[i], "-centralizer") == 0) {
			f_centralizer = TRUE;
			centralizer_idx = atoi(argv[++i]);
			cout << "-centralizer " << centralizer_idx << endl;
			}
		else if (strcmp(argv[i], "-centralizer_of_element") == 0) {
			f_centralizer_of_element = TRUE;
			element_description = argv[++i];
			cout << "-centralizer_of_element " << element_description << endl;
			}
		}


	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);


	if (!ECA->f_starter_size) {
		cout << "please use option -starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "please use option -input_prefix <input_prefix>" << endl;
		exit(1);
		}

	if (!f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}

	int p, e, e1, n, k, q;
	
	factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "dim_over_kernel does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = i_power_j(p, e1);
		cout << "order=" << order << " n=" << n
				<< " k=" << k << " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "order=" << order << " n=" << n
				<< " k=" << k << " q=" << q << endl;
		}


	
	finite_field *F;
	spread *T;
	packing *P;


	F = NEW_OBJECT(finite_field);
	
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);



	T = NEW_OBJECT(spread);

	T->read_arguments(argc, argv);
	
	int max_depth = order + 1;
	
	cout << "before T->init" << endl;
	T->init(order, n, k, max_depth, 
		F, f_recoordinatize, 
		"TP_STARTER", "TP", order + 1, 
		argc, argv, 
		MINIMUM(verbose_level - 1, 2));
	cout << "after T->init" << endl;
	
	cout << "before T->init2" << endl;
	T->init2(verbose_level);
	cout << "after T->init2" << endl;


	P = NEW_OBJECT(packing);


	cout << "before P->init" << endl;
	P->init(T, 
		f_select_spread, 
		select_spread, 
		select_spread_nb, 
		ECA->input_prefix, ECA->base_fname, 
		ECA->starter_size, 
		ECA->f_lex, 
		verbose_level);
	cout << "after P->init" << endl;

	cout << "before IA->init" << endl;
	IA->init(T->A, P->A_on_spreads, P->gen, 
		P->size_of_packing, P->prefix_with_directory, ECA,
		callback_packing_report,
		NULL /*callback_subset_orbits*/,
		P,
		verbose_level);
	cout << "after IA->init" << endl;



	if (f_compute_spread_table) {
		P->compute_spread_table(verbose_level);
	}
#if 0
	else if (f_type_of_packing) {
		cout << "before P->type_of_packing" << endl;
		if (!f_fname_spread_table) {
			cout << "please use option f_fname_spread_table <fname>" << endl;
			exit(1);
		}
		if (!f_fname_spread_table_iso) {
			cout << "please use option f_fname_spread_table_iso <fname>" << endl;
			exit(1);
		}
		P->type_of_packing(
				fname_spread_table,
				fname_spread_table_iso,
				type_of_packing_fname,
				verbose_level);
		cout << "after P->type_of_packing" << endl;
		//exit(1);
		}
#endif
	else if (f_conjugacy_classes) {
		cout << "before P->conjugacy_classes" << endl;
		P->conjugacy_classes(verbose_level);
		cout << "after P->conjugacy_classes" << endl;
		//exit(1);
		}
	else if (f_conjugacy_classes_and_normalizers) {
		cout << "before P->conjugacy_classes_and_normalizers" << endl;
		P->conjugacy_classes_and_normalizers(verbose_level);
		cout << "after P->conjugacy_classes_and_normalizers" << endl;
		//exit(1);
		}
	else if (f_make_element) {
		cout << "before P->make_element" << endl;
		P->make_element(make_element_idx, verbose_level);
		cout << "after P->make_element" << endl;
		//exit(1);
		}
	else if (f_centralizer) {
		cout << "before P->centralizer" << endl;
		P->centralizer(centralizer_idx, verbose_level);
		cout << "after P->centralizer" << endl;
		//exit(1);
		}
	else if (f_centralizer_of_element) {
		cout << "before P->centralizer_of_element" << endl;
		P->centralizer_of_element(element_description, label, verbose_level);
		cout << "after P->centralizer_of_element" << endl;
		//exit(1);
		}
	else {
		cout << "before P->init2" << endl;
		P->init2(verbose_level);
			// this computes the spread table from scratch
		cout << "after P->init2" << endl;
		}


	if (f_starter) {

		cout << "f_starter" << endl;
		int t1, t2, dt;

		t1 = os_ticks();

		cout << "before P->compute" << endl;

		P->compute(verbose_level);

		cout << "after P->compute" << endl;

		t2 = os_ticks();
		if (f_draw_poset) {
			P->gen->draw_poset(P->prefix_with_directory,
					ECA->starter_size, ECA->starter_size,
					f_embedded, f_sideways, 0 /*P->gen->verbose_level*/);
			}
		if (f_draw_poset_full) {
			double x_stretch = 0.4;
			cout << "before draw_poset_full" << endl;
			P->gen->draw_poset_full(P->prefix_with_directory,
					ECA->starter_size, ECA->starter_size,
					f_embedded, f_sideways, x_stretch,
					0 /*P->gen->verbose_level*/);
			cout << "after draw_poset_full" << endl;
			}
		dt = t2 - t1;
		cout << "time in compute(): " << dt << endl;
		}


	if (ECA->f_lift) {
	
		ECA->target_size = P->size_of_packing;
		ECA->user_data = (void *) P;
		ECA->A = P->T->A;
		ECA->A2 = P->A_on_spreads;
		ECA->prepare_function_new = packing_lifting_prepare_function_new;
		ECA->early_test_function = packing_early_test_function;
		ECA->early_test_function_data = (void *) P;
		
		cout << "before compute_lifts" << endl;
		ECA->compute_lifts(verbose_level);
		cout << "after compute_lifts" << endl;

		}


	IA->execute(verbose_level);


	if (f_klein) {
		P->f_split_klein = f_split_klein;
		P->split_klein_r = split_klein_r;
		P->split_klein_m = split_klein_m;


		cout << "before isomorph_worker(callback_packing_"
				"compute_klein_invariants)" << endl;

		isomorph_worker(P->T->A, P->A_on_spreads, P->gen, 
			P->size_of_packing, 
			P->prefix_with_directory, IA->prefix_iso, 
			callback_packing_compute_klein_invariants, P, 
			ECA->starter_size, verbose_level);
		cout << "after isomorph_worker(callback_packing_"
				"compute_klein_invariants)" << endl;
		}


	FREE_OBJECT(P);
	FREE_OBJECT(T);
	FREE_OBJECT(F);
	
	the_end(t0);
	//the_end_quietly(t0);

}


