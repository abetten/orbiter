// regular_ls.C
// 
// Anton Betten
// 1/1/13
//
// 
//
//

#include "orbiter.h"
#include "regular_ls.h"


// global data:

INT t0; // the system time when the program started



int main(int argc, const char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_starter = FALSE;

	INT f_draw_poset = FALSE;
	INT f_embedded = FALSE;
	INT f_sideways = FALSE;

	exact_cover_arguments *ECA = NULL;
	isomorph_arguments *IA = NULL;

	ECA = new exact_cover_arguments;
	IA = new isomorph_arguments;
	

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-starter") == 0) {
			f_starter = TRUE;
			cout << "-starter" << endl;
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

	INT f_v = (verbose_level >= 1);

	if (f_memory_debug) {
		start_memory_debug();
		}
	{
	regular_ls_generator Gen;

	Gen.init_basic(argc, argv, 
		ECA->input_prefix, ECA->base_fname, ECA->starter_size, 
		verbose_level);
	
	Gen.init_group(verbose_level);


	Gen.init_action_on_k_subsets(Gen.k, 0 /*verbose_level*/);

	Gen.init_generator(FALSE, NULL, 
		Gen.A->Strong_gens, verbose_level);




	IA->init(Gen.A, Gen.A2, Gen.gen, 
		Gen.target_size, Gen.prefix_with_directory, ECA,
		NULL /*callback_report*/,
		NULL /*callback_subset_orbits*/,
		&Gen,
		verbose_level);

	if (f_v) {
		cout << "init finished" << endl;
		}


	if (f_starter) {

		INT f_write_candidate_file = TRUE;
		
		Gen.compute_starter(
			//ECA->f_lex, 
			f_write_candidate_file, 
			f_draw_poset, f_embedded, f_sideways, verbose_level);

		}


	if (ECA->f_lift) {
	
		ECA->target_size = Gen.target_size;
		ECA->user_data = (void *) &Gen;
		ECA->A = Gen.A;
		ECA->A2 = Gen.A2;
		ECA->prepare_function_new = rls_generator_lifting_prepare_function_new;
		ECA->early_test_function = rls_generator_early_test_function;
		ECA->early_test_function_data = (void *) &Gen;
		
		compute_lifts(ECA, verbose_level);
			// in TOP_LEVEL/extra.C

		}

	IA->execute(verbose_level);



	}
	if (f_memory_debug) {
		registry_dump_sorted();
		}

	the_end(t0);
	//the_end_quietly(t0);
}



