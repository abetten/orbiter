// spread_classify.cpp
// 
// Anton Betten
// July 9, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;



// global data:

int t0; // the system time when the program started

void print_spread(ostream &ost, int len, long int *S, void *data);


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


	//int f_make_spread = FALSE;
	//int type_of_spread = 0;
	int f_recoordinatize = FALSE;
	int f_starter = FALSE;


	int nb_identify = 0;
	const char *identify_data[1000];


	int f_list = FALSE;
	int f_make_quotients = FALSE;
	int f_print_spread = FALSE;
	const char *fname_print_spread;
	int f_HMO = FALSE;
	const char *fname_HMO;
	int f_print_representatives = FALSE;
	int representatives_size = 0;
	const char *representatives_fname = NULL;
	int f_test_identify = FALSE;
	int identify_level = 0;
	int identify_nb_times = 0;
	int f_draw_poset = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_print_data_structure = FALSE;
	os_interface Os;


	exact_cover_arguments *ECA = NULL;
	isomorph_arguments *IA = NULL;
	poset_classification_control *Control;

	ECA = NEW_OBJECT(exact_cover_arguments);
	IA = NEW_OBJECT(isomorph_arguments);
	Control = NEW_OBJECT(poset_classification_control);



	t0 = Os.os_ticks();
	cout << argv[0] << endl;
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


#if 0
		else if (strcmp(argv[i], "-FTWKB") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_FTWKB;
			cout << "-FTWKB" << endl;
			}
		else if (strcmp(argv[i], "-Kantor") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_KANTOR;
			cout << "-Kantor" << endl;
			}
		else if (strcmp(argv[i], "-DicksonKantor") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_DICKSON_KANTOR;
			cout << "-DicksonKantor" << endl;
			}
		else if (strcmp(argv[i], "-Hudson") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_HUDSON;
			cout << "-Hudson" << endl;
			}
		else if (strcmp(argv[i], "-Kantor2") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_KANTOR2;
			cout << "-Kantor2" << endl;
			}
		else if (strcmp(argv[i], "-Ganley") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_GANLEY;
			cout << "-Ganley" << endl;
			}
		else if (strcmp(argv[i], "-Law_Penttila") == 0) {
			f_make_spread = TRUE;
			type_of_spread = SPREAD_OF_TYPE_LAW_PENTTILA;
			cout << "-Law_Penttila" << endl;
			}
#endif


		else if (strcmp(argv[i], "-starter") == 0) {
			f_starter = TRUE;
			cout << "-starter " << endl;
			}

		else if (strcmp(argv[i], "-identify") == 0) {
			
			identify_data[nb_identify] = argv[++i];
			cout << "-identify " << identify_data[nb_identify] << endl;
			nb_identify++;
			}
		else if (strcmp(argv[i], "-test_identify") == 0) {
			f_test_identify = TRUE;
			identify_level = atoi(argv[++i]);
			identify_nb_times = atoi(argv[++i]);
			cout << "-test_identify " << identify_level << " "
					<< identify_nb_times << endl;
			}



		else if (strcmp(argv[i], "-make_quotients") == 0) {
			f_make_quotients = TRUE;
			cout << "-make_quotients " << endl;
			}

		else if (strcmp(argv[i], "-print_spread") == 0) {
			f_print_spread = TRUE;
			fname_print_spread = argv[++i];
			cout << "-print_spread " << fname_print_spread << endl;
			}
		else if (strcmp(argv[i], "-HMO") == 0) {
			f_HMO = TRUE;
			fname_HMO = argv[++i];
			cout << "-HMO " << fname_HMO << endl;
			}


		else if (strcmp(argv[i], "-print_representatives") == 0) {
			f_print_representatives = TRUE;
			representatives_size = atoi(argv[++i]);
			representatives_fname = argv[++i];
			cout << "-print_representatives" << representatives_size
					<< " " << representatives_fname << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = TRUE;
			cout << "-print_data_structure " << endl;
			}


		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list " << endl;
			}
		}

	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);

	int f_v = (verbose_level >= 1);


	if (!f_order) {
		cout << "spread_classify.cpp please use option "
				"-order <order>" << endl;
		exit(1);
		}
	if (!ECA->f_starter_size) {
		cout << "spread_classify.cpp please use option "
				"-starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "spread_classify.cpp please use option "
				"-input_prefix <input_prefix>" << endl;
		exit(1);
		}

	int p, e, e1, n, k, q;
	number_theory_domain NT;
	
	NT.factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "spread_classify.cpp dim_over_kernel "
					"does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = NT.i_power_j(p, e1);
		cout << "spread_classify.cpp order=" << order
				<< " n=" << n << " k=" << k << " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "spread_classify.cpp order=" << order
				<< " n=" << n << " k=" << k << " q=" << q << endl;
		}

	finite_field *F;
	spread_classify T;

	F = NEW_OBJECT(finite_field);

	cout << "spread_classify.cpp before F->init_override_polynomial" << endl;
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	//cout << "spread_classify.cpp before T.read_arguments" << endl;
	//T.read_arguments(argc, argv);
	
	int max_depth = NT.i_power_j(F->q, k) + 1;

	cout << "spread_classify.cpp before T.init" << endl;
	T.init(order, n, k, max_depth, 
		F, f_recoordinatize, 
		ECA->input_prefix, ECA->base_fname, ECA->starter_size, 
		argc, argv, 
		verbose_level - 1);
	cout << "spread_classify.cpp after T.init" << endl;
	
	cout << "spread_classify.cpp before T.init2" << endl;
	T.init2(Control, verbose_level - 1);
	cout << "spread_classify.cpp after T.init2" << endl;


	char prefix_with_directory[1000];

	sprintf(prefix_with_directory, "%s%s",
			T.starter_directory_name, T.prefix);

	cout << "spread_classify.cpp before IA.init" << endl;
	IA->init(T.A, T.A2, T.gen, 
		T.spread_size, prefix_with_directory, ECA,
		spread_callback_report,
		NULL /* callback_subset_orbits */,
		&T,
		verbose_level - 1);
	cout << "spread_classify.cpp after IA.init" << endl;

#if 0
	if (f_make_spread) {
		cout << "spread_classify.cpp f_make_spread" << endl;
		cout << "spread_classify.cpp before T.write_spread_to_file" << endl;
		T.write_spread_to_file(type_of_spread, verbose_level);
		cout << "spread_classify.cpp after T.write_spread_to_file" << endl;
		}
#endif

	if (f_starter) {


		cout << "spread_classify.cpp f_starter" << endl;
		
		cout << "spread_classify.cpp before T.compute" << endl;

		T.compute(verbose_level);

		cout << "spread_classify.cpp after T.compute" << endl;

		cout << "spread_classify.cpp "
				"starter_size = " << ECA->starter_size << endl;
		cout << "spread_classify.cpp "
				"spread_size = " << T.spread_size << endl;
	

		if (f_list) {
			int f_show_orbit_decomposition = TRUE;
			int f_show_stab = TRUE;
			int f_save_stab = TRUE;
			int f_show_whole_orbit = FALSE;
		
			T.gen->list_all_orbits_at_level(ECA->starter_size, 
				TRUE, 
				print_spread, 
				&T, 
				f_show_orbit_decomposition,
				f_show_stab, f_save_stab, f_show_whole_orbit);

#if 0
			int d;
			for (d = 0; d < 3; d++) {
				T.gen->print_schreier_vectors_at_depth(d, verbose_level);
				}
#endif
			}
		
		if (f_draw_poset) {
			if (f_v) {
				cout << "spread_classify.cpp "
						"before gen->draw_poset" << endl;
				}
			T.gen->draw_poset(T.gen->fname_base,
					ECA->starter_size, 0 /* data1 */,
					f_embedded, f_sideways, verbose_level);
			}


		if (f_print_data_structure) {
			if (f_v) {
				cout << "spread_classify.cpp "
						"before gen->print_data_structure_tex" << endl;
				}
			T.gen->print_data_structure_tex(
					ECA->starter_size, 0 /*gen->verbose_level*/);
			}


		}
	else if (nb_identify) {

		cout << "spread_classify.cpp f_identify" << endl;

		
		cout << "spread_classify.cpp "
				"classifying spreads" << endl;
		T.compute(0 /* verbose_level */);
		cout << "spread_classify.cpp "
				"classifying spreads done" << endl;

		//T.gen->print_node(5);
		int *transporter;
		int orbit_at_level;
		
		transporter = NEW_int(T.gen->Poset->A->elt_size_in_int);
		
		for (i = 0; i < nb_identify; i++) {

			long int *data;
			int sz;

			lint_vec_scan(identify_data[i], data, sz);
			cout << "spread_classify.cpp identifying set "
					<< i << " / " << nb_identify << " : ";
			lint_vec_print(cout, data, sz);
			cout << endl;
			T.gen->identify(data, sz,
					transporter, orbit_at_level, verbose_level);

			cout << "The set " << i << " / " << nb_identify
					<< " is identified to belong to orbit "
					<< orbit_at_level << endl;
			cout << "A transporter is " << endl;
			T.gen->Poset->A->element_print_quick(transporter, cout);

			FREE_lint(data);
		}

		FREE_int(transporter);
	}
	else if (f_test_identify) {
		cout << "spread_classify.cpp f_test_identify" << endl;
		
		cout << "spread_classify.cpp "
				"classifying spreads" << endl;
		T.compute(0 /* verbose_level */);
		cout << "spread_classify.cpp "
				"classifying spreads done" << endl;

		T.gen->test_identify(identify_level,
				identify_nb_times, verbose_level);
	}


	if (ECA->f_lift) {
	
		cout << "spread_classify.cpp f_lift" << endl;
		
		ECA->target_size = T.spread_size;
		ECA->user_data = (void *) &T;
		ECA->A = T.A;
		ECA->A2 = T.A2;
		ECA->prepare_function_new = spread_lifting_prepare_function_new;
		ECA->early_test_function = spread_lifting_early_test_function;
		ECA->early_test_function_data = (void *) &T;
		
		//compute_lifts(ECA, verbose_level);

		if (f_v) {
			cout << "spread_classify.cpp before ECA->compute_lifts" << endl;
		}


		ECA->compute_lifts(verbose_level);

	
		if (f_v) {
			cout << "spread_classify.cpp after ECA->compute_lifts" << endl;
		}



	}

	cout << "spread_classify.cpp before IA->execute" << endl;
	IA->execute(verbose_level);
	cout << "spread_classify.cpp after IA->execute" << endl;


	if (f_print_spread) {
		T.read_and_print_spread(fname_print_spread, verbose_level);
	}
	else if (f_HMO) {
		T.HMO(fname_HMO, verbose_level);
	}

	file_io Fio;

	if (f_print_representatives) {
		orbit_rep *R;
		int *M;
		int no, nb;
		char fname[1000];
		
		cout << "spread_classify.cpp printing orbit "
				"representatives at level "
				<< representatives_size << endl;
		R = NEW_OBJECT(orbit_rep);
		M = NEW_int(T.k * T.n);

		sprintf(fname, "%s_lvl_%d",
				representatives_fname, representatives_size);

		nb = Fio.count_number_of_orbits_in_file(fname, verbose_level);

		cout << "there are " << nb << " orbit representatives "
				"in the file " << fname << endl;
		for (no = 0; no < nb; no++) {
			R->init_from_file(T.A /*A_base*/,
				(char *) representatives_fname,
				representatives_size, no,
				representatives_size - 1/*level_of_candidates_file*/,
				spread_lifting_early_test_function, 
				&T, 
				verbose_level - 1
				);
			// R has: int *candidates; int nb_candidates;
	
			for (i = 0; i < representatives_size; i++) {
				cout << R->rep[i] << " ";
			}
			cout << endl;
			for (i = 0; i < representatives_size; i++) {
				cout << R->rep[i] << " = " << endl;
				T.Grass->unrank_lint_here(M,
						R->rep[i], 0/*verbose_level - 4*/);
				int_matrix_print(M, T.k, T.n);
			}
		}
		FREE_OBJECT(R);
	}




//end:
	the_end(t0);
	//the_end_quietly(t0);
}


void print_spread(ostream &ost, int len, long int *S, void *data)
{
	spread_classify *Spread = (spread_classify *) data;
	
	Spread->print(ost, len, S);
}





