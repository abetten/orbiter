// blt_main.C
// 
// Anton Betten
// started 8/13/2006
//
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;

// global data:

int t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_starter = FALSE;
	int f_list = FALSE;
	int f_draw_poset = FALSE;


	int i;
	//int f_Law71 = FALSE;


	int f_create_graphs = FALSE;
	int create_graphs_r, create_graphs_m, create_graphs_level;
	int f_eliminate_graphs_if_possible = FALSE;
	int f_create_graphs_list_of_cases = FALSE;
	const char *create_graphs_list_of_cases = NULL;
	const char *create_graphs_list_of_cases_prefix = NULL;


	exact_cover_arguments *ECA = NULL;
	isomorph_arguments *IA = NULL;

	ECA = NEW_OBJECT(exact_cover_arguments);
	IA = NEW_OBJECT(isomorph_arguments);



	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-starter") == 0) {
			f_starter = TRUE;
			cout << "-starter " << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list " << endl;
			}
		else if (strcmp(argv[i], "-create_graphs") == 0) {
			f_create_graphs = TRUE;
			create_graphs_r = atoi(argv[++i]);
			create_graphs_m = atoi(argv[++i]);
			create_graphs_level = atoi(argv[++i]);
			cout << "-create_graphs " << " " << create_graphs_r
					<< " " << create_graphs_m << " "
					<< create_graphs_level << endl;
			}
		else if (strcmp(argv[i], "-create_graphs_list_of_cases") == 0) {
			f_create_graphs_list_of_cases = TRUE;
			create_graphs_level = atoi(argv[++i]);
			create_graphs_list_of_cases_prefix = argv[++i];
			create_graphs_list_of_cases = argv[++i];
			cout << "-create_graphs_list_of_cases " << " "
					<< create_graphs_level << " "
					<< create_graphs_list_of_cases_prefix
					<< " " << create_graphs_list_of_cases << endl;
			}
#if 0
		else if (strcmp(argv[i], "-Law71") == 0) {
			f_Law71 = TRUE;
			cout << "-Law71" << endl;
			}
#endif
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}


		}
	ECA->read_arguments(argc, argv, verbose_level);
	IA->read_arguments(argc, argv, verbose_level);


	int f_v = (verbose_level >= 1);

	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	if (!ECA->f_starter_size) {
		cout << "please use option -starter_size <starter_size>" << endl;
		exit(1);
		}
	if (!ECA->f_has_input_prefix) {
		cout << "please use option -input_prefix <input_prefix>" << endl;
		exit(1);
		}

	{
	blt_set *Blt_set;
	int schreier_depth = ECA->starter_size;
	int f_debug = FALSE;
	int f_semilinear = FALSE;
	number_theory_domain NT;
	
	finite_field *F;
	orthogonal *O;

	F = NEW_OBJECT(finite_field);
	O = NEW_OBJECT(orthogonal);
	Blt_set = NEW_OBJECT(blt_set);

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}
	if (f_poly) {
		F->init_override_polynomial(q, poly, 0 /* verbose_level */);
		}
	else {
		F->init(q, 0 /* verbose_level */);
		}
	O->init(0 /*epsilon*/, 5 /* n */, F, 0 /*verbose_level*/);

	Blt_set->init_basic(O,
		f_semilinear,
		ECA->input_prefix, ECA->base_fname, ECA->starter_size, 
		argc, argv, verbose_level);
	
	
	Blt_set->init_group(f_semilinear, verbose_level);
	
	Blt_set->init2(verbose_level);
	
	int f_use_invariant_subset_if_available = TRUE;
	
	if (f_v) {
		cout << "init finished, calling main, "
				"schreier_depth = " << schreier_depth << endl;
		}


	IA->init(Blt_set->A, Blt_set->A, Blt_set->gen,
			Blt_set->target_size, Blt_set->prefix_with_directory, ECA,
			blt_set_callback_report,
			NULL /*blt_set_callback_subset_orbits*/,
			Blt_set,
			verbose_level);

	if (f_starter) {

		int depth;
		int f_embedded = TRUE;
		int f_sideways = FALSE;

		depth = Blt_set->gen->main(t0, schreier_depth,
			f_use_invariant_subset_if_available, 
			f_debug, 
			Blt_set->gen->verbose_level);
		cout << "Blt_set->gen->main returns depth=" << depth << endl;
		//Gen.gen->print_data_structure_tex(depth,
		//Gen.gen->verbose_level);
		if (f_draw_poset) {
			Blt_set->gen->draw_poset(Blt_set->prefix_with_directory,
					ECA->starter_size, 0 /* data1 */,
					f_embedded, f_sideways, Blt_set->gen->verbose_level);
			}
		if (f_list) {
				{
				spreadsheet *Sp;
				Blt_set->gen->make_spreadsheet_of_orbit_reps(Sp, depth);
				char fname_csv[1000];
				sprintf(fname_csv, "partial_BLT_sets_%d_%d.csv",
						q, depth);
				Sp->save(fname_csv, verbose_level);
				delete Sp;
				}
			}
		}

	if (ECA->f_lift) {
	
		cout << "lift" << endl;
		
		ECA->target_size = Blt_set->target_size;
		ECA->user_data = (void *) Blt_set;
		ECA->A = Blt_set->A;
		ECA->A2 = Blt_set->A;
		ECA->prepare_function_new = blt_set_lifting_prepare_function_new;
		ECA->early_test_function = blt_set_early_test_func_callback;
		ECA->early_test_function_data = (void *) Blt_set;
		
		ECA->compute_lifts(verbose_level);

		}

	if (f_create_graphs) {

		if (!ECA->f_has_output_prefix) {
			cout << "please use -output_prefix <output_prefix>" << endl;
			exit(1);
			}
		Blt_set->create_graphs(
			create_graphs_r, create_graphs_m, 
			create_graphs_level, 
			ECA->output_prefix, 
			ECA->f_lex, f_eliminate_graphs_if_possible, 
			verbose_level);
		}
	else if (f_create_graphs_list_of_cases) {

		if (!ECA->f_has_output_prefix) {
			cout << "please use -output_prefix <output_prefix>" << endl;
			exit(1);
			}
		Blt_set->create_graphs_list_of_cases(
			create_graphs_list_of_cases_prefix,
			create_graphs_list_of_cases,
			create_graphs_level, 
			ECA->output_prefix, 
			ECA->f_lex, f_eliminate_graphs_if_possible, 
			verbose_level);
		}

#if 0
	else if (f_Law71) {
		Blt_set->Law_71(verbose_level);
		}
#endif

	IA->execute(verbose_level);



	cout << "cleaning up Gen" << endl;
	}


	the_end(t0);
	//the_end_quietly(t0);
}





