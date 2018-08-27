// spread_classify.C
// 
// Anton Betten
// July 9, 2013
//
//
// 
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

void print_spread(INT len, INT *S, void *data);


#define MAX_FILES 1000

int main(int argc, const char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_poly = FALSE;
	const BYTE *poly = NULL;
	INT f_order = FALSE;
	INT order = 0;
	INT f_dim_over_kernel = FALSE;
	INT dim_over_kernel = 0;
	INT f_make_spread = FALSE;
	INT type_of_spread = 0;
	INT f_recoordinatize = FALSE;
	INT f_starter = FALSE;


	INT nb_identify = 0;
	const BYTE *identify_data[1000];


	INT f_list = FALSE;
	INT f_make_quotients = FALSE;
	INT f_print_spread = FALSE;
	const BYTE *fname_print_spread;
	INT f_HMO = FALSE;
	const BYTE *fname_HMO;
	INT f_print_representatives = FALSE;
	INT representatives_size = 0;
	const BYTE *representatives_fname = NULL;
	INT f_test_identify = FALSE;
	INT identify_level = 0;
	INT identify_nb_times = 0;
	INT f_draw_poset = FALSE;
	INT f_embedded = FALSE;
	INT f_sideways = FALSE;
	INT f_print_data_structure = FALSE;


	exact_cover_arguments *ECA = NULL;
	isomorph_arguments *IA = NULL;

	ECA = NEW_OBJECT(exact_cover_arguments);
	IA = NEW_OBJECT(isomorph_arguments);



	t0 = os_ticks();
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
			cout << "-test_identify " << identify_level << " " << identify_nb_times << endl;
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
			cout << "-print_representatives" << representatives_size << " " << representatives_fname << endl;
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

	INT f_v = (verbose_level >= 1);


	if (!f_order) {
		cout << "spread_classify.C please use option -order <order>" << endl;
		exit(1);
		}
	if (!ECA->f_starter_size) {
		cout << "spread_classify.C please use option -starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "spread_classify.C please use option -input_prefix <input_prefix>" << endl;
		exit(1);
		}

	INT p, e, e1, n, k, q;
	
	factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "spread_classify.C dim_over_kernel does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = i_power_j(p, e1);
		cout << "spread_classify.C order=" << order << " n=" << n << " k=" << k << " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "spread_classify.C order=" << order << " n=" << n << " k=" << k << " q=" << q << endl;
		}

	finite_field *F;
	spread T;

	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	T.read_arguments(argc, argv);
	
	INT max_depth = i_power_j(F->q, k) + 1;

	cout << "spread_classify.C before T.init" << endl;
	T.init(order, n, k, max_depth, 
		F, f_recoordinatize, 
		ECA->input_prefix, ECA->base_fname, ECA->starter_size, 
		argc, argv, 
		verbose_level - 1);
	cout << "spread_classify.C after T.init" << endl;
	
	T.init2(verbose_level - 1);

	cout << "spread_classify.C before IA.init" << endl;

	BYTE prefix_with_directory[1000];

	sprintf(prefix_with_directory, "%s%s", T.starter_directory_name, T.prefix);

	IA->init(T.A, T.A2, T.gen, 
		T.spread_size, prefix_with_directory, ECA,
		spread_callback_report,
		NULL /* callback_subset_orbits */,
		&T,
		verbose_level - 1);


	if (f_make_spread) {
		cout << "spread_classify.C f_make_spread" << endl;
		T.write_spread_to_file(type_of_spread, verbose_level);
		}
	else if (f_starter) {


		cout << "spread_classify.C f_starter" << endl;
		
		T.compute(verbose_level);

		cout << "spread_classify.C starter_size = " << ECA->starter_size << endl;
		cout << "spread_classify.C spread_size = " << T.spread_size << endl;
	

		if (f_list) {
			INT f_show_orbit_decomposition = TRUE, f_show_stab = TRUE, f_save_stab = TRUE, f_show_whole_orbit = FALSE;
		
			T.gen->list_all_orbits_at_level(ECA->starter_size, 
				TRUE, 
				print_spread, 
				&T, 
				f_show_orbit_decomposition, f_show_stab, f_save_stab, f_show_whole_orbit);

#if 0
			INT d;
			for (d = 0; d < 3; d++) {
				T.gen->print_schreier_vectors_at_depth(d, verbose_level);
				}
#endif
			}
		
		if (f_draw_poset) {
			if (f_v) {
				cout << "spread_classify.C before gen->draw_poset" << endl;
				}
			T.gen->draw_poset(T.gen->fname_base, ECA->starter_size, 0 /* data1 */, f_embedded, f_sideways, verbose_level);
			}


		if (f_print_data_structure) {
			if (f_v) {
				cout << "spread_classify.C before gen->print_data_structure_tex" << endl;
				}
			T.gen->print_data_structure_tex(ECA->starter_size, 0 /*gen->verbose_level*/);
			}


		}
	else if (nb_identify) {

		cout << "spread_classify.C f_identify" << endl;

		
		cout << "spread_classify.C classifying translation planes" << endl;
		T.compute(0 /* verbose_level */);
		cout << "spread_classify.C classifying translation planes done" << endl;

		//T.gen->print_node(5);
		INT *transporter;
		INT orbit_at_level;
		
		transporter = NEW_INT(T.gen->A->elt_size_in_INT);
		
		for (i = 0; i < nb_identify; i++) {

			INT *data;
			INT sz;

			INT_vec_scan(identify_data[i], data, sz);
			cout << "spread_classify.C identifying set " << i << " / " << nb_identify << " : ";
			INT_vec_print(cout, data, sz);
			cout << endl;
			T.gen->identify(data, sz, transporter, orbit_at_level, verbose_level);

			cout << "The set " << i << " / " << nb_identify << " is identified to belong to orbit " << orbit_at_level << endl;
			cout << "A transporter is " << endl;
			T.gen->A->element_print_quick(transporter, cout);

			FREE_INT(data);
			}

		FREE_INT(transporter);
		}
	else if (f_test_identify) {
		cout << "spread_classify.C f_test_identify" << endl;
		
		cout << "spread_classify.C classifying translation planes" << endl;
		T.compute(0 /* verbose_level */);
		cout << "spread_classify.C classifying translation planes done" << endl;

		T.gen->test_identify(identify_level, identify_nb_times, verbose_level);
		}


	if (ECA->f_lift) {
	
		cout << "spread_classify.C f_lift" << endl;
		
		ECA->target_size = T.spread_size;
		ECA->user_data = (void *) &T;
		ECA->A = T.A;
		ECA->A2 = T.A2;
		ECA->prepare_function_new = spread_lifting_prepare_function_new;
		ECA->early_test_function = spread_lifting_early_test_function;
		ECA->early_test_function_data = (void *) &T;
		
		//compute_lifts(ECA, verbose_level);
			// in TOP_LEVEL/extra.C

		if (f_v) {
			cout << "spread_classify.C before ECA->compute_lifts" << endl;
			}


		ECA->compute_lifts(verbose_level);

	
		if (f_v) {
			cout << "spread_classify.C after ECA->compute_lifts" << endl;
			}



		}

	cout << "spread_classify.C before IA->execute" << endl;
	IA->execute(verbose_level);
	cout << "spread_classify.C after IA->execute" << endl;


	if (f_print_spread) {
		T.read_and_print_spread(fname_print_spread, verbose_level);
		}
	else if (f_HMO) {
		T.HMO(fname_HMO, verbose_level);
		}


	if (f_print_representatives) {
		orbit_rep *R;
		INT *M;
		INT no, nb;
		BYTE fname[1000];
		
		cout << "spread_classify.C printing orbit representatives at level " << representatives_size << endl;
		R = new orbit_rep;
		M = NEW_INT(T.k * T.n);

		sprintf(fname, "%s_lvl_%ld", representatives_fname, representatives_size);

		nb = count_number_of_orbits_in_file(fname, verbose_level);

		cout << "there are " << nb << " orbit representatives in the file " << fname << endl;
		for (no = 0; no < nb; no++) {
			R->init_from_file(T.A /*A_base*/, (BYTE *) representatives_fname, 
				representatives_size, no, representatives_size - 1/*level_of_candidates_file*/, 
				spread_lifting_early_test_function, 
				&T, 
				verbose_level - 1
				);
			// R has: INT *candidates; INT nb_candidates;
	
			for (i = 0; i < representatives_size; i++) {
				cout << R->rep[i] << " ";
				}
			cout << endl;
			for (i = 0; i < representatives_size; i++) {
				cout << R->rep[i] << " = " << endl;
				T.Grass->unrank_INT_here(M, R->rep[i], 0/*verbose_level - 4*/);
				INT_matrix_print(M, T.k, T.n);
				}
			}
		}




//end:
	the_end(t0);
	//the_end_quietly(t0);
}


void print_spread(INT len, INT *S, void *data)
{
	spread *T = (spread *) data;
	
	T->print(len, S);
}





