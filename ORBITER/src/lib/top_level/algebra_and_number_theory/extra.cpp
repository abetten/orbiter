// extra.C
// 
// Anton Betten
//
// started 9/23/2010
//
//
// 
//
//

#include "orbiter.h"

namespace orbiter {

void isomorph_print_set(ostream &ost, int len, int *S, void *data)
{
	//isomorph *G = (isomorph *) data;
	
	print_vector(ost, S, (int) len);
	//G->print(ost, S, len);
}


sims *create_sims_for_stabilizer(action *A, 
	int *set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	sims *Stab;
	int nb_backtrack_nodes;
	int t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer" << endl;
		}
	strong_generators *Aut_gens;

	poset *Poset;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A, A->Strong_gens, verbose_level);
	STAB.init(Poset, set, set_size, verbose_level);
	STAB.compute_set_stabilizer(t0,
			nb_backtrack_nodes, Aut_gens,
			verbose_level - 1);
	
	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	delete Aut_gens;
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer, "
				"found a group of order " << go << endl;
		}
	FREE_OBJECT(Poset);
	return Stab;
}

sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	int *set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	strong_generators *Aut_gens;
	sims *Stab;
	int nb_backtrack_nodes;
	int t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group" << endl;
		}

	poset *Poset;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A0, A, A0->Strong_gens, verbose_level);
	STAB.init_with_strong_generators(Poset,
			set, set_size, verbose_level);
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group "
				"after STAB.init_with_strong_generators" << endl;
		}

	
	STAB.compute_set_stabilizer(t0, nb_backtrack_nodes,
			Aut_gens, verbose_level - 1);

	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	FREE_OBJECT(Poset);
	delete Aut_gens;

	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group "
				"after STAB.compute_set_stabilizer" << endl;
		}
	
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer_with_input_group, "
				"found a group of order " << go << endl;
		}
	return Stab;
}



void compute_lifts(exact_cover_arguments *ECA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_lifts" << endl;
		}


	ECA->compute_lifts(verbose_level);

	
	if (f_v) {
		cout << "compute_lifts done" << endl;
		}
}



void compute_lifts_new(
	action *A, action *A2, 
	void *user_data, 
	const char *base_fname, 
	const char *input_prefix, const char *output_prefix, 
	const char *solution_prefix, 
	int starter_size, int target_size, 
	int f_lex, int f_split, int split_r, int split_m, 
	int f_solve, int f_save, int f_read_instead, 
	int f_draw_system, const char *fname_system, 
	int f_write_tree, const char *fname_tree,
	void (*prepare_function_new)(exact_cover *E, int starter_case, 
		int *candidates, int nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, int *&col_label, 
		int &f_ruled_out, 
		int verbose_level), 
	void (*early_test_function)(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		void *data, int verbose_level), 
	void *early_test_function_data,
	int f_has_solution_test_function, 
	int (*solution_test_func)(exact_cover *EC,
			int *S, int len, void *data, int verbose_level),
	void *solution_test_func_data,
	int f_has_late_cleanup_function, 
	void (*late_cleanup_function)(exact_cover *EC,
			int starter_case, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_lifts_new" << endl;
		cout << "compute_lifts_new verbose_level=" << verbose_level << endl;
		cout << "compute_lifts_new base_fname=" << base_fname << endl;
		cout << "compute_lifts_new input_prefix=" << input_prefix << endl;
		cout << "compute_lifts_new output_prefix=" << output_prefix << endl;
		cout << "compute_lifts_new solution_prefix="
				<< solution_prefix << endl;
		}



	exact_cover *E;

	E = NEW_OBJECT(exact_cover);

 
	E->init_basic(user_data, 
		A, A2, 
		target_size, starter_size, 
		input_prefix, output_prefix, solution_prefix, base_fname, 
		f_lex, 
		verbose_level - 1);

	E->init_early_test_func(
		early_test_function, early_test_function_data,
		verbose_level);

	E->init_prepare_function_new(
		prepare_function_new, 
		verbose_level);

	if (f_split) {
		E->set_split(split_r, split_m, verbose_level - 1);
		}

	if (f_has_solution_test_function) {
		E->add_solution_test_function(
			solution_test_func, 
			(void *) solution_test_func_data,
			verbose_level - 1);
		}

	if (f_has_late_cleanup_function) {
		E->add_late_cleanup_function(late_cleanup_function);
		}
	
	if (f_v) {
		cout << "compute_lifts_new before compute_liftings_new" << endl;
		}

	E->compute_liftings_new(f_solve, f_save, f_read_instead, 
		f_draw_system, fname_system, 
		f_write_tree, fname_tree,
		verbose_level - 1);

	FREE_OBJECT(E);
	
	if (f_v) {
		cout << "compute_lifts_new done" << endl;
		}
}

}


