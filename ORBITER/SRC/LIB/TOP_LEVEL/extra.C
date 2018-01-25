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

void isomorph_print_set(ostream &ost, INT len, INT *S, void *data)
{
	//isomorph *G = (isomorph *) data;
	
	print_vector(ost, S, (int) len);
	//G->print(ost, S, len);
}


sims *create_sims_for_stabilizer(action *A, 
	INT *set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	sims *Stab;
	INT nb_backtrack_nodes;
	INT t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer" << endl;
		}
	strong_generators *Aut_gens;

	STAB.init(A, set, set_size, verbose_level);
	STAB.compute_set_stabilizer(t0, nb_backtrack_nodes, Aut_gens, verbose_level - 1);
	
	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	delete Aut_gens;
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer, found a group of order " << go << endl;
		}
	return Stab;
}

sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	INT *set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	strong_generators *Aut_gens;
	sims *Stab;
	INT nb_backtrack_nodes;
	INT t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group" << endl;
		}

	STAB.init_with_strong_generators(A, A0, Strong_gens, set, set_size, verbose_level);
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group after STAB.init_with_strong_generators" << endl;
		}

	
	STAB.compute_set_stabilizer(t0, nb_backtrack_nodes, Aut_gens, verbose_level - 1);

	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	delete Aut_gens;

	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group after STAB.compute_set_stabilizer" << endl;
		}
	
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer_with_input_group, found a group of order " << go << endl;
		}
	return Stab;
}

void solve_diophant(INT *Inc, INT nb_rows, INT nb_cols, INT nb_needed, 
	INT f_has_Rhs, INT *Rhs, 
	INT *&Solutions, INT &nb_sol, INT &nb_backtrack, INT &dt, 
	INT f_DLX, 
	INT f_draw_system, const BYTE *fname_system, 
	INT f_write_tree, const BYTE *fname_tree, 
	INT verbose_level)
// allocates Solutions[nb_sol * nb_needed]
{
	INT f_v = (verbose_level >= 1);
	//INT f_v4 = FALSE; //(verbose_level >= 2);
	//INT i, j;
	diophant *Dio;
	INT t0 = os_ticks();

	if (f_v) {
		cout << "solve_diophant nb_rows=" << nb_rows << " nb_cols=" << nb_cols << " f_has_Rhs=" << f_has_Rhs << " verbose_level=" << verbose_level << endl;
		cout << "f_write_tree=" << f_write_tree << endl;
		cout << "f_DLX=" << f_DLX << endl;
		//INT_matrix_print(Inc, nb_rows, nb_cols);
		}
	Dio = new diophant;

	if (f_has_Rhs) {
		Dio->init_problem_of_Steiner_type_with_RHS(nb_rows, 
			nb_cols, Inc, nb_needed, 
			Rhs, 
			0 /* verbose_level */);
		}
	else {
		Dio->init_problem_of_Steiner_type(nb_rows, 
			nb_cols, Inc, nb_needed, 
			0 /* verbose_level */);
		}

	if (f_draw_system) {
		INT xmax_in = 1000000;
		INT ymax_in = 1000000;
		INT xmax_out = 1000000;
		INT ymax_out = 1000000;
		
		if (f_v) {
			cout << "solve_diophant drawing the system" << endl;
			}
		Dio->draw_it(fname_system, xmax_in, ymax_in, xmax_out, ymax_out);
		if (f_v) {
			cout << "solve_diophant drawing the system done" << endl;
			}
		}

	if (FALSE /*f_v4*/) {
		Dio->print();
		}

	if (f_DLX && !f_has_Rhs) {
		Dio->solve_all_DLX(f_write_tree, fname_tree, 0 /* verbose_level*/);
		nb_backtrack = Dio->nb_steps_betten;
		}
	else {
		diophant_solve_all_mckay(Dio, nb_backtrack, verbose_level - 2);
		}

	nb_sol = Dio->_resultanz;
	if (nb_sol) {
		Dio->get_solutions(Solutions, nb_sol, 1 /* verbose_level */);
		if (FALSE /*f_v4*/) {
			cout << "Solutions:" << endl;
			INT_matrix_print(Solutions, nb_sol, nb_needed);
			}
		}
	else {
		Solutions = NULL;
		}
	delete Dio;
	INT t1 = os_ticks();
	dt = t1 - t0;
	if (f_v) {
		cout << "solve_diophant done nb_sol=" << nb_sol << " nb_backtrack=" << nb_backtrack << " dt=" << dt << endl;
		}
}


void compute_lifts(exact_cover_arguments *ECA, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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
	const BYTE *base_fname, 
	const BYTE *input_prefix, const BYTE *output_prefix, 
	const BYTE *solution_prefix, 
	INT starter_size, INT target_size, 
	INT f_lex, INT f_split, INT split_r, INT split_m, 
	INT f_solve, INT f_save, INT f_read_instead, 
	INT f_draw_system, const BYTE *fname_system, 
	INT f_write_tree, const BYTE *fname_tree,
	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level), 
	void (*early_test_function)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_function_data,
	INT f_has_solution_test_function, 
	INT (*solution_test_func)(exact_cover *EC, INT *S, INT len, void *data, INT verbose_level), 
	void *solution_test_func_data,
	INT f_has_late_cleanup_function, 
	void (*late_cleanup_function)(exact_cover *EC, INT starter_case, INT verbose_level), 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_lifts_new" << endl;
		cout << "compute_lifts_new verbose_level=" << verbose_level << endl;
		cout << "compute_lifts_new base_fname=" << base_fname << endl;
		cout << "compute_lifts_new input_prefix=" << input_prefix << endl;
		cout << "compute_lifts_new output_prefix=" << output_prefix << endl;
		cout << "compute_lifts_new solution_prefix=" << solution_prefix << endl;
		}



	exact_cover *E;

	E = new exact_cover;

 
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

	delete E;
	
	if (f_v) {
		cout << "compute_lifts_new done" << endl;
		}
}


