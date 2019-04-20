// packing.C
// 
// Anton Betten
// Feb 6, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




packing::packing()
{
	T = NULL;
	F = NULL;
	spread_size = 0;
	nb_lines = 0;
	search_depth = 0;

	starter_directory_name[0] = 0;
	prefix[0] = 0;
	prefix_with_directory[0] = 0;


	f_lexorder_test = TRUE;
	q = 0;
	size_of_packing = 0;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	P3 = NULL;


	nb_spreads_up_to_isomorphism = 0;
		// the number of spreads
		// from the classification
	input_spreads = NULL;
	input_spread_label = NULL;
	nb_input_spreads = 0;

	Spread_tables = NULL;
	tmp_isomorphism_type_of_spread = NULL;

	A_on_spreads = NULL;


	bitvector_adjacency = NULL;
	bitvector_length = 0;
	degree = NULL;

	Poset = NULL;
	gen = NULL;

	nb_needed = 0;


	f_split_klein = FALSE;
	split_klein_r = 0;
	split_klein_m = 1;
	//null();
}

packing::~packing()
{
	freeself();
}

void packing::null()
{
}

void packing::freeself()
{
	if (bitvector_adjacency) {
		FREE_uchar(bitvector_adjacency);
		}
	if (input_spreads) {
		FREE_int(input_spreads);
		}
	if (input_spread_label) {
		FREE_int(input_spread_label);
		}
	if (Spread_tables) {
		FREE_OBJECT(Spread_tables);
	}
	if (P3) {
		delete P3;
		}
	null();
}

void packing::init(spread *T, 
	int f_packing_select_spread,
	int *packing_select_spread, int packing_select_spread_nb,
	const char *input_prefix, const char *base_fname, 
	int search_depth, 
	int f_lexorder_test,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::init" << endl;
		}

	packing::T = T;
	F = T->F;
	packing::f_lexorder_test = f_lexorder_test;
	q = T->q;
	spread_size = T->spread_size;
	size_of_packing = q * q + q + 1;
	nb_lines = T->A2->degree;

	packing::search_depth = search_depth;
	
	if (f_v) {
		cout << "packing::init q=" << q << endl;
		cout << "packing::init nb_lines=" << nb_lines << endl;
		cout << "packing::init spread_size=" << spread_size << endl;
		cout << "packing::init size_of_packing=" << size_of_packing << endl;
		cout << "packing::init input_prefix=" << input_prefix << endl;
		cout << "packing::init base_fname=" << base_fname << endl;
		cout << "packing::init search_depth=" << search_depth << endl;
		}

	init_P3(verbose_level - 1);

	strcpy(starter_directory_name, input_prefix);
	strcpy(prefix, base_fname);
	sprintf(prefix_with_directory, "%s%s",
			starter_directory_name, base_fname);



	if (f_packing_select_spread) {
		cout << "packing::init selected spreads are "
				"from the following orbits: ";
		int_vec_print(cout,
				packing_select_spread,
				packing_select_spread_nb);
		cout << endl;
		}
	

	Spread_tables = NEW_OBJECT(spread_tables);

	load_input_spreads(Spread_tables->nb_spreads,
			f_packing_select_spread,
			packing_select_spread, packing_select_spread_nb,
			verbose_level - 1);

	if (f_v) {
		cout << "We have " << nb_input_spreads << " input spreads, "
				"nb_spreads = " << Spread_tables->nb_spreads << endl;
		}


	if (f_v) {
		cout << "packing::init done" << endl;
		}
}

void packing::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::init2" << endl;
		}

#if 0
	if (f_v) {
		cout << "packing::init2 "
				"before compute_spread_table" << endl;
		}
	compute_spread_table(verbose_level - 1);
	if (f_v) {
		cout << "packing::init2 "
				"after compute_spread_table" << endl;
		}
#endif

	if (f_v) {
		cout << "packing::init2 "
				"before create_action_on_spreads" << endl;
		}
	create_action_on_spreads(verbose_level - 1);
	if (f_v) {
		cout << "packing::init2 "
				"after create_action_on_spreads" << endl;
		}


	
	if (f_v) {
		cout << "packing::init "
				"before prepare_generator" << endl;
		}
	prepare_generator(search_depth, verbose_level - 1);
	if (f_v) {
		cout << "packing::init "
				"after prepare_generator" << endl;
		}

	if (f_v) {
		cout << "packing::init done" << endl;
		}
}

void packing::compute_spread_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int **Sets;
	int nb_spreads;
	int *isomorphism_type_of_spread;
	int *Spread_table;

	if (f_v) {
		cout << "packing::compute_spread_table" << endl;
		}

	nb_spreads = Spread_tables->nb_spreads;
	Sets = NEW_pint(nb_spreads);

	isomorphism_type_of_spread = NEW_int(nb_spreads);



	if (f_v) {
		cout << "packing::compute_spread_table "
				"before make_spread_table" << endl;
		}
	make_spread_table(nb_spreads,
		input_spreads, nb_input_spreads, input_spread_label,
		Sets, isomorphism_type_of_spread,
		verbose_level - 1);
	if (f_v) {
		cout << "packing::compute_spread_table "
				"after make_spread_table" << endl;
		}


	Spread_table = NEW_int(nb_spreads * spread_size);
	for (i = 0; i < nb_spreads; i++) {
		int_vec_copy(Sets[i], Spread_table + i * spread_size, spread_size);
		}


	Spread_tables->init(F, FALSE, nb_spreads_up_to_isomorphism,
			verbose_level);


	Spread_tables->init_spread_table(nb_spreads,
			Spread_table, isomorphism_type_of_spread,
			verbose_level);

	int *Dual_spread_idx;
	int *self_dual_spread_idx;
	int nb_self_dual_spreads;

	compute_dual_spreads(Sets,
				Dual_spread_idx,
				self_dual_spread_idx,
				nb_self_dual_spreads,
				verbose_level);



	Spread_tables->init_tables(nb_spreads,
			Spread_table, isomorphism_type_of_spread,
			Dual_spread_idx,
			self_dual_spread_idx, nb_self_dual_spreads,
			verbose_level);

	Spread_tables->save(verbose_level);


	if (nb_spreads < 10000) {
		cout << "packing::compute_spread_table "
				"We are computing the adjacency matrix" << endl;
		compute_adjacency_matrix(verbose_level - 1);
		cout << "packing::compute_spread_table "
				"The adjacency matrix has been computed" << endl;
		}
	else {
		cout << "packing::compute_spread_table "
				"We are NOT computing the adjacency matrix" << endl;
		}


	for (i = 0; i < nb_spreads; i++) {
		FREE_int(Sets[i]);
		}
	FREE_pint(Sets);


	if (f_v) {
		cout << "packing::compute_spread_table done" << endl;
		}
}

void packing::init_P3(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::init_P3" << endl;
		}
	P3 = NEW_OBJECT(projective_space);
	
	P3->init(3, T->F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);
	if (f_v) {
		cout << "packing::init_P3 done" << endl;
		cout << "N_points=" << P3->N_points << endl;
		cout << "N_lines=" << P3->N_lines << endl;
		}
}


void packing::load_input_spreads(int &N, 
	int f_packing_select_spread,
	int *packing_select_spread,
	int packing_select_spread_nb,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_do_it, idx, no, orbit_length;
	longinteger_object go, stab_go;
	longinteger_domain D;
	knowledge_base K;

	if (f_v) {
		cout << "packing::load_input_spreads" << endl;
		}


	N = 0;
	
	T->A->Sims->group_order(go);


	nb_spreads_up_to_isomorphism = K.Spread_nb_reps(q, T->k);

	input_spreads = NEW_int(nb_spreads_up_to_isomorphism * spread_size);
	input_spread_label = NEW_int(nb_spreads_up_to_isomorphism);
	nb_input_spreads = 0;


	for (no = 0; no < nb_spreads_up_to_isomorphism; no++) {

		vector_ge *gens;
		const char *stab_order;

		T->A->stabilizer_of_spread_representative(q,
				T->k, no, gens, stab_order, 0 /*verbose_level*/);
			// ACTION/action.C


		f_do_it = FALSE;	
		if (f_packing_select_spread) {
			if (int_vec_search_linear(packing_select_spread,
					packing_select_spread_nb, no, idx)) {
				f_do_it = TRUE;
				}
			}
		else {
			f_do_it = TRUE;
			}
		if (f_do_it) {
			int *rep;
			int sz;

			rep = K.Spread_representative(
					q, T->k, no, sz);
			int_vec_copy(rep,
					input_spreads + nb_input_spreads * spread_size,
					spread_size);


			input_spread_label[nb_input_spreads] = no;
			nb_input_spreads++;


			stab_go.create_from_base_10_string(
					stab_order,
					0 /* verbose_level */);
			//Stab->group_order(stab_go);
		
			orbit_length = D.quotient_as_int(go, stab_go);
			if (f_v) {
				cout << "spread orbit " << no
						<< " has group order "
						<< stab_go << " orbit_length = "
						<< orbit_length << endl;
				}


			N += orbit_length;


			}


		} // next no



	if (f_v) {
		cout << "packing::load_input_spreads done, N = " << N << endl;
		}
}


void packing::make_spread_table(
	int nb_spreads, int *input_spreads,
	int nb_input_spreads, int *input_spread_label,
	int **&Sets, int *&isomorphism_type_of_spread,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int nb_spreads1;

	if (f_v) {
		cout << "packing::make_spread_table" << endl;
		}
	Sets = NEW_pint(nb_spreads);
	isomorphism_type_of_spread = NEW_int(nb_spreads);

	orbit_of_sets *SetOrb;

	SetOrb = NEW_OBJECTS(orbit_of_sets, nb_input_spreads);

	for (i = 0; i < nb_input_spreads; i++) {

		if (f_v) {
			cout << "packing::make_spread_table "
				"Spread " << i << " / "
				<< nb_input_spreads << " computing orbits" << endl;
			}


		SetOrb[i].init(T->A, T->A2,
				input_spreads + i * spread_size,
				spread_size, T->A->Strong_gens->gens,
				verbose_level);


		if (f_v) {
			cout << "packing::make_spread_table Spread "
				<< input_spread_label[i] << " = " << i << " / "
				<< nb_input_spreads << " has orbit length "
				<< SetOrb[i].used_length << endl;
			}


		} // next i

	nb_spreads1 = 0;

	for (i = 0; i < nb_input_spreads; i++) {

		for (j = 0; j < SetOrb[i].used_length; j++) {

			Sets[nb_spreads1] = NEW_int(spread_size);

			int_vec_copy(SetOrb[i].Sets[j], Sets[nb_spreads1], spread_size);

			isomorphism_type_of_spread[nb_spreads1] = i;


			nb_spreads1++;

		} // next j
	} // next i

	if (f_v) {
		cout << "packing::make_spread_table We found "
				<< nb_spreads1 << " labeled spreads" << endl;
		}

	if (nb_spreads1 != nb_spreads) {
		cout << "packing::make_spread_table "
				"nb_spreads1 != nb_spreads" << endl;
		exit(1);
	}

	FREE_OBJECTS(SetOrb);

	if (f_v) {
		cout << "packing::make_spread_table before "
				"sorting spread table of size " << nb_spreads << endl;
	}
	tmp_isomorphism_type_of_spread = isomorphism_type_of_spread;
		// for packing_swap_func
	Heapsort_general(Sets, nb_spreads,
			packing_spread_compare_func,
			packing_swap_func,
			this);
	if (f_v) {
		cout << "packing::make_spread_table after "
				"sorting spread table of size " << nb_spreads << endl;
	}


	if (FALSE) {
		cout << "packing::make_spread_table "
				"The labeled spreads are:" << endl;
		for (i = 0; i < nb_spreads; i++) {
			cout << i << " : ";
			int_vec_print(cout, Sets[i], spread_size /* + 1*/);
			cout << endl;
			}
		}

	if (f_v) {
		cout << "packing::make_spread_table done" << endl;
		}
}


void packing::compute_dual_spreads(int **Sets,
		int *&Dual_spread_idx,
		int *&self_dual_spread_idx,
		int &nb_self_dual_spreads,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *dual_spread;
	int i, j, a, b, idx;
	int nb_spreads;

	if (f_v) {
		cout << "packing::compute_dual_spreads" << endl;
		}

	nb_spreads = Spread_tables->nb_spreads;

	dual_spread = NEW_int(spread_size);
	Dual_spread_idx = NEW_int(nb_spreads);
	self_dual_spread_idx = NEW_int(nb_spreads);

	nb_self_dual_spreads = 0;

	for (i = 0; i < nb_spreads; i++) {

#if 0
		T->compute_dual_spread(
				Spread_table + i * spread_size,
				dual_spread /*+ 1*/, verbose_level - 4);
#else
		for (j = 0; j < spread_size; j++) {
			a = Spread_tables->spread_table[i * spread_size + j];
			b = Spread_tables->dual_line_idx[a];
			dual_spread[j] = b;
		}
#endif
		if (f_v) {
			cout << "packing::compute_dual_spreads spread "
					<< i << " / " << nb_spreads << endl;
			int_vec_print(cout,
					Spread_tables->spread_table +
					i * spread_size, spread_size);
			cout << endl;
			int_vec_print(cout, dual_spread, spread_size);
			cout << endl;
			}
		int_vec_heapsort(dual_spread, spread_size);
		//dual_spread[0] = int_vec_hash(dual_spread + 1, spread_size);
		if (f_v) {
			int_vec_print(cout, dual_spread, spread_size);
			cout << endl;
		}

		int v[1];

		v[0] = spread_size /*+ 1*/;

		if (vec_search((void **)Sets,
			orbit_of_sets_compare_func, (void *) v,
			nb_spreads, dual_spread, idx,
			0 /* verbose_level */)) {
			if (f_vv) {
				cout << "packing::compute_dual_spreads Dual "
						"spread of spread " << i
						<< " is spread no " << idx << endl;
				}
			Dual_spread_idx[i] = idx;
			if (idx == i) {
				self_dual_spread_idx[nb_self_dual_spreads++] = i;
			}
		}
		else {
			cout << "The dual spread is not in the list, error!" << endl;
			cout << "dual_spread: ";
			int_vec_print(cout, dual_spread, spread_size);
			cout << endl;
			exit(1);
		}
	}

	FREE_int(dual_spread);
	if (f_v) {
		cout << "packing::compute_dual_spreads we found "
				<< nb_self_dual_spreads << " self dual spreads" << endl;
		cout << "They are: ";
		int_vec_print(cout, self_dual_spread_idx, nb_self_dual_spreads);
		cout << endl;
		}


	if (f_v) {
		cout << "packing::compute_dual_spreads done" << endl;
		}

}

int packing::test_if_packing_is_self_dual(int *packing, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = FALSE;
	int *sorted_packing;
	int *dual_packing;
	int i, a, b;

	if (f_v) {
		cout << "packing::test_if_packing_is_self_dual" << endl;
	}
	sorted_packing = NEW_int(size_of_packing);
	dual_packing = NEW_int(size_of_packing);
	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		sorted_packing[i] = a;
	}
	int_vec_heapsort(sorted_packing, size_of_packing);

	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		b = Spread_tables->dual_spread_idx[a];
		dual_packing[i] = b;
	}
	int_vec_heapsort(dual_packing, size_of_packing);
	if (int_vec_compare(sorted_packing, dual_packing, size_of_packing) == 0) {
		ret = TRUE;
	}

	if (f_v) {
		cout << "packing::test_if_packing_is_self_dual done" << endl;
	}
	return ret;
}


void packing::compute_adjacency_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::compute_adjacency_matrix" << endl;
		}

	Spread_tables->compute_adjacency_matrix(
			bitvector_adjacency,
			bitvector_length,
			verbose_level);

	
	if (f_v) {
		cout << "packing::compute_adjacency_matrix done" << endl;
		}
}



void packing::prepare_generator(
		int search_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing::prepare_generator" << endl;
		cout << "search_depth=" << search_depth << endl;
		}
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(T->A, A_on_spreads,
			T->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "packing::prepare_generator before "
				"Poset->add_testing_without_group" << endl;
		}
	Poset->add_testing_without_group(
			packing_early_test_function,
				this /* void *data */,
				verbose_level);


	gen = NEW_OBJECT(poset_classification);
	
	gen->f_T = TRUE;
	gen->f_W = TRUE;

	if (f_v) {
		cout << "packing::prepare_generator "
				"calling gen->initialize" << endl;
		}

	gen->initialize(Poset,
		search_depth, 
		"", prefix_with_directory,
		verbose_level - 1);



#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif	



}


void packing::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = gen->depth;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int t0;

	t0 = os_ticks();

	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level - 1);
	
	int length;
	
	if (f_v) {
		cout << "packing::compute done with generator_main" << endl;
		}
	length = gen->nb_orbits_at_level(gen->depth);
	if (f_v) {
		cout << "packing::compute We found "
			<< length << " orbits on "
			<< gen->depth << "-sets" << endl;
		}
}

int packing::spreads_are_disjoint(int i, int j)
{
	int *p1, *p2;

	p1 = Spread_tables->spread_table + i * spread_size;
	p2 = Spread_tables->spread_table + j * spread_size;
	if (test_if_sets_are_disjoint(p1, p2,
			spread_size, spread_size)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void packing::lifting_prepare_function_new(
	exact_cover *E, int starter_case,
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level)
{
	verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v3 = (verbose_level >= 3);
	int *points_covered_by_starter;
	int nb_points_covered_by_starter;
	int *free_points2;
	int nb_free_points2;
	int *free_point_idx;
	int *live_blocks2;
	int nb_live_blocks2;
	int i, j, a;
	int nb_needed, nb_rows, nb_cols;


	if (f_v) {
		cout << "packing::lifting_prepare_function "
				"nb_candidates=" << nb_candidates << endl;
		}

	nb_needed = size_of_packing - E->starter_size;


	if (f_v) {
		cout << "packing::lifting_prepare_function "
				"before compute_covered_points" << endl;
		}

	compute_covered_points(points_covered_by_starter, 
		nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing::lifting_prepare_function "
				"before compute_free_points2" << endl;
		}

	compute_free_points2(
		free_points2, nb_free_points2, free_point_idx,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);

	if (f_v) {
		cout << "packing::lifting_prepare_function "
				"before compute_live_blocks2" << endl;
		}

	compute_live_blocks2(
		E, starter_case, live_blocks2, nb_live_blocks2,
		points_covered_by_starter, nb_points_covered_by_starter, 
		E->starter, E->starter_size, 
		verbose_level - 1);


	if (f_v) {
		cout << "packing::lifting_prepare_function "
				"after compute_live_blocks2" << endl;
		}

	nb_rows = nb_free_points2;
	nb_cols = nb_live_blocks2;
	col_labels = NEW_int(nb_cols);


	int_vec_copy(live_blocks2, col_labels, nb_cols);


	if (f_vv) {
		cout << "packing::lifting_prepare_function_new candidates: ";
		int_vec_print(cout, col_labels, nb_cols);
		cout << " (nb_candidates=" << nb_cols << ")" << endl;
		}



	if (E->f_lex) {
		int nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "packing::lifting_prepare_function_new after "
					"lexorder test nb_candidates before: " << nb_cols_before
					<< " reduced to  " << nb_cols << " (deleted "
					<< nb_cols_before - nb_cols << ")" << endl;
			}
		}

	if (f_vv) {
		cout << "packing::lifting_prepare_function_new "
				"after lexorder test" << endl;
		cout << "packing::lifting_prepare_function_new "
				"nb_cols=" << nb_cols << endl;
		}


	int s, u;
	
	Dio = NEW_OBJECT(diophant);
	Dio->open(nb_rows, nb_cols);
	Dio->f_has_sum = TRUE;
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);


	for (j = 0; j < nb_cols; j++) {
		s = live_blocks2[j];
		for (a = 0; a < spread_size; a++) {
			i = Spread_tables->spread_table[s * spread_size + a];
			u = free_point_idx[i];
			if (u == -1) {
				cout << "packing::lifting_prepare_function "
						"free_point_idx[i] == -1" << endl;
				exit(1);
				}
			Dio->Aij(u, j) = 1;
			}
		}


	FREE_int(points_covered_by_starter);
	FREE_int(free_points2);
	FREE_int(free_point_idx);
	FREE_int(live_blocks2);
}


void packing::compute_covered_points(
	int *&points_covered_by_starter,
	int &nb_points_covered_by_starter,
	int *starter, int starter_size,
	int verbose_level)
// points_covered_by_starter are the lines that
// are contained in the spreads chosen for the starter
{
	int f_v = (verbose_level >= 1);
	int i, j, a, s;
	
	if (f_v) {
		cout << "packing::compute_covered_points" << endl;
		}
	points_covered_by_starter = NEW_int(starter_size * spread_size);
	for (i = 0; i < starter_size; i++) {
		s = starter[i];
		for (j = 0; j < spread_size; j++) {
			a = Spread_tables->spread_table[s * spread_size + j];
			points_covered_by_starter[i * spread_size + j] = a;
			}
		}
#if 0
	cout << "covered lines:" << endl;
	int_vec_print(cout, covered_lines, starter_size * spread_size);
	cout << endl;
#endif
	if (f_v) {
		cout << "packing::compute_covered_points done" << endl;
		}
}

void packing::compute_free_points2(
	int *&free_points2, int &nb_free_points2, int *&free_point_idx,
	int *points_covered_by_starter,
	int nb_points_covered_by_starter,
	int *starter, int starter_size, 
	int verbose_level)
// free_points2 are actually the free lines,
// i.e., the lines that are not
// yet part of the partial packing
{
	int f_v = (verbose_level >= 1);
	int i, a;
	
	if (f_v) {
		cout << "packing::compute_free_points2" << endl;
		}
	free_point_idx = NEW_int(nb_lines);
	free_points2 = NEW_int(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		free_point_idx[i] = 0;
		}
	for (i = 0; i < starter_size * spread_size; i++) {
		a = points_covered_by_starter[i];
		free_point_idx[a] = -1;
		}
	nb_free_points2 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (free_point_idx[i] == 0) {
			free_points2[nb_free_points2] = i;
			free_point_idx[i] = nb_free_points2;
			nb_free_points2++;
			}
		}
#if 0
	cout << "free points2:" << endl;
	int_vec_print(cout, free_points2, nb_free_points2);
	cout << endl;
#endif
	if (f_v) {
		cout << "packing::compute_free_points2 done" << endl;
		}
}

void packing::compute_live_blocks2(
	exact_cover *EC, int starter_case,
	int *&live_blocks2, int &nb_live_blocks2, 
	int *points_covered_by_starter, int nb_points_covered_by_starter, 
	int *starter, int starter_size, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "packing::compute_live_blocks2" << endl;
		}
	live_blocks2 = NEW_int(Spread_tables->nb_spreads);
	nb_live_blocks2 = 0;
	for (i = 0; i < Spread_tables->nb_spreads; i++) {
		for (j = 0; j < starter_size; j++) {
			if (!is_adjacent(starter[j], i)) {
				break;
				}
			}
		if (j == starter_size) {
			live_blocks2[nb_live_blocks2++] = i;
			}
		}
	if (f_v) {
		cout << "packing::compute_live_blocks2 done" << endl;
		}

	if (f_v) {
		cout << "packing::compute_live_blocks2 STARTER_CASE "
			<< starter_case << " / " << EC->starter_nb_cases
			<< " : Found " << nb_live_blocks2 << " live spreads" << endl;
		}
}

int packing::is_adjacent(int i, int j)
{
	int k;
	combinatorics_domain Combi;
	
	if (i == j) {
		return FALSE;
		}
	if (bitvector_adjacency) {
		k = Combi.ij2k(i, j, Spread_tables->nb_spreads);
		if (bitvector_s_i(bitvector_adjacency, k)) {
			return TRUE;
			}
		else {
			return FALSE;
			}
		}
	else {
		if (spreads_are_disjoint(i, j)) {
			return TRUE;
			}
		else {
			return FALSE;
			}
		}
}

void packing::read_spread_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "packing::read_spread_table" << endl;
		}

	Spread_tables = NEW_OBJECT(spread_tables);

	if (f_v) {
		cout << "packing::read_spread_table "
				"before Spread_tables->init" << endl;
		}

	Spread_tables->init(F,
			TRUE /* f_load */, nb_spreads_up_to_isomorphism,
			verbose_level);

	{
		int *type;
		set_of_sets *SoS;
		int a, b;

		Spread_tables->classify_self_dual_spreads(type,
				SoS,
				verbose_level);
		cout << "the self-dual spreads belong to the "
				"following isomorphism types:" << endl;
		for (i = 0; i < nb_spreads_up_to_isomorphism; i++) {
			cout << i << " : " << type[i] << endl;
		}
		SoS->print();
		for (a = 0; a < SoS->nb_sets; a++) {
			if (SoS->Set_size[a] < 10) {
				cout << "iso type " << a << endl;
				int_vec_print(cout, SoS->Sets[a], SoS->Set_size[a]);
				cout << endl;
				for (i = 0; i < SoS->Set_size[a]; i++) {
					b = SoS->Sets[a][i];
					cout << i << " : " << b << " : ";
					int_vec_print(cout, Spread_tables->spread_table +
							b * spread_size, spread_size);
					cout << endl;
				}
			}
		}
		FREE_int(type);
	}

	if (f_v) {
		cout << "packing::read_spread_table "
				"after Spread_tables->init" << endl;
		}



	if (f_v) {
		cout << "packing::read_spread_table done" << endl;
		}
}

void packing::create_action_on_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::create_action_on_spreads" << endl;
		}
	//int f_induce = FALSE;

	if (f_v) {
		cout << "packing::create_action_on_spreads "
				"creating action A_on_spreads" << endl;
		}
	A_on_spreads = T->A2->create_induced_action_on_sets(
			//T->A->Sims,
			Spread_tables->nb_spreads, spread_size,
			Spread_tables->spread_table,
			//f_induce,
			0 /* verbose_level */);

	cout << "created action on spreads" << endl;

	if (f_v) {
		cout << "packing::create_action_on_spreads "
				"creating action A_on_spreads done" << endl;
		}
}


#if 0
void packing::type_of_packing(
		const char *fname_spread_table,
		const char *fname_spread_table_iso,
		const char *fname_packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Packings;
	int nb_packings;
	int *Type;
	char fname[1000];
	int i, j, a, b, n;
	
	if (f_v) {
		cout << "packing::type_of_packing" << endl;
		}
	

	read_spread_table(fname_spread_table,
			fname_spread_table_iso, verbose_level);
	
	int_matrix_read_csv(fname_packings, Packings,
			nb_packings, n, 0 /* verbose_level */);
	
	cout << "Read file " << fname_packings << " with "
			<< nb_packings << " packings of size " << n << endl;

	if (n != size_of_packing) {
		cout << "n != size_of_packing" << endl;
		exit(1);
		}

	Type = NEW_int(nb_packings * nb_spreads_up_to_isomorphism);
	int_vec_zero(Type, nb_packings * nb_spreads_up_to_isomorphism);
	for (i = 0; i < nb_packings; i++) {
		for (j = 0; j < size_of_packing; j++) {
			a = Packings[i * size_of_packing + j];
			b = isomorphism_type_of_spread[a];
			Type[i * nb_spreads_up_to_isomorphism + b]++;
			}
		}
	strcpy(fname, fname_packings);
	replace_extension_with(fname, "_type.csv");
	int_matrix_write_csv(fname, Type, nb_packings,
			nb_spreads_up_to_isomorphism);
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	
	if (f_v) {
		cout << "packing::type_of_packing done" << endl;
		}
}
#endif

void packing::conjugacy_classes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char prefix[1000];
	char fname1[1000];
	char fname2[1000];


	if (f_v) {
		cout << "packing::conjugacy_classes" << endl;
		}

	sprintf(prefix, "PGGL_4_%d", q);
	sprintf(fname1, "%sconjugacy_classes.magma", prefix);
	sprintf(fname2, "%sconjugacy_classes.txt", prefix);


	if (file_size(fname2) > 0) {
		read_conjugacy_classes(fname2, verbose_level);
		}
	else {
		T->A->conjugacy_classes_using_MAGMA(prefix,
				T->A->Sims, verbose_level);
		}

	if (f_v) {
		cout << "packing::conjugacy_classes done" << endl;
		}
}

void packing::read_conjugacy_classes(
		char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	int *class_size;
	int *class_order_of_element;
	
	if (f_v) {
		cout << "packing::read_conjugacy_classes" << endl;
		}

	T->A->read_conjugacy_classes_from_MAGMA(
			fname,
			nb_classes,
			perms,
			class_size,
			class_order_of_element,
			verbose_level - 1);
#if 0
	{
		ifstream fp(fname);

		fp >> nb_classes;
		cout << "We found " << nb_classes
				<< " conjugacy classes" << endl;

		perms = NEW_int(nb_classes * T->A->degree);
		class_size = NEW_int(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			fp >> class_size[i];
			for (j = 0; j < T->A->degree; j++) {
				fp >> perms[i * T->A->degree + j];
				}
			}
		cout << "we read all class representatives "
				"from file " << fname << endl;
	}
	for (i = 0; i < nb_classes * T->A->degree; i++) {
		perms[i]--;
		}
#endif



	longinteger_object go;
	longinteger_domain D;
	
	T->A->group_order(go);
	cout << "The group has order " << go << endl;

	char fname_latex[1000];
	strcpy(fname_latex, fname);
	
	replace_extension_with(fname_latex, ".tex");

	{
	ofstream fp(fname_latex);
	char title[1000];
	latex_interface L;

	sprintf(title, "Conjugacy classes of $%s$",
			T->A->label_tex);
	
	L.head(fp,
		FALSE /* f_book */, TRUE /* f_title */,
		title, "computed by MAGMA" /* const char *author */, 
		FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */, 
		TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */, 
		NULL /* extra_praeamble */);
	//latex_head_easy(fp);
	
	fp << "\\section{Conjugacy classes in $"
			<< T->A->label_tex << "$}" << endl;


	fp << "The group order is " << endl;
	fp << "$$" << endl;
	go.print_not_scientific(fp);
	fp << endl;
	fp << "$$" << endl;
	

	cout << "The conjugacy classes are:" << endl;
	for (i = 0; i < nb_classes; i++) {
		strong_generators *gens;
		longinteger_object go1, Class_size, centralizer_order;
		int goi;
		vector_ge *nice_gens;

		goi = class_order_of_element[i];
		gens = NEW_OBJECT(strong_generators);
		gens->init_from_permutation_representation(T->A, 
			perms + i * T->A->degree, 
			1, goi, nice_gens,
			verbose_level);

		FREE_OBJECT(nice_gens);

		Class_size.create(class_size[i]);
		
		D.integral_division_exact(go, Class_size, centralizer_order);

		fp << "\\bigskip" << endl;
		fp << "\\subsection*{Class " << i << " / "
				<< nb_classes << "}" << endl;
		fp << "Order of element = " << class_order_of_element[i]
				<< "\\\\" << endl;
		fp << "Class size = " << class_size[i] << "\\\\" << endl;
		fp << "Centralizer order = " << centralizer_order
				<< "\\\\" << endl;

		if (class_order_of_element[i] > 1) {
			fp << "Representing element is" << endl;
			fp << "$$" << endl;
			T->A->element_print_latex(gens->gens->ith(0), fp);
			fp << "$$" << endl;
			fp << "$";
			T->A->element_print_for_make_element(gens->gens->ith(0), fp);
			fp << "$\\\\" << endl;
			}

		cout << "class " << i << " / " << nb_classes 
			<< " size = " << class_size[i] 
			<< " order of element = " << class_order_of_element[i] 
			<< " centralizer order = " << centralizer_order 
			<< " : " << endl;
		cout << "packing::make_element created "
				"generators for a group" << endl;
		gens->print_generators();
		gens->group_order(go1);
		cout << "prime_at_a_time::make_element "
				"The group has order " << go1 << endl;

		FREE_OBJECT(gens);
		}
	L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< file_size(fname_latex) << endl;
	
	if (f_v) {
		cout << "packing::read_conjugacy_classes done" << endl;
		}
}


void packing::conjugacy_classes_and_normalizers(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char prefix[1000];
	char fname1[1000];
	char fname2[1000];


	if (f_v) {
		cout << "packing::conjugacy_classes_and_normalizers" << endl;
		}

	sprintf(prefix, "PGGL_4_%d", q);
	sprintf(fname1, "%sconjugacy_classes_and_normalizers.magma", prefix);
	sprintf(fname2, "%sconjugacy_classes_and_normalizers.txt", prefix);


	if (file_size(fname2) > 0) {
		read_conjugacy_classes_and_normalizers(fname2, verbose_level);
		}
	else {
		T->A->conjugacy_classes_and_normalizers_using_MAGMA(prefix,
				T->A->Sims, verbose_level);
		}

	if (f_v) {
		cout << "packing::conjugacy_classes_and_normalizers done" << endl;
		}
}


void packing::read_conjugacy_classes_and_normalizers(
		char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	int *class_size;
	int *class_order_of_element;
	int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;
	projective_space_with_action *PA;
	number_theory_domain NT;

	if (f_v) {
		cout << "packing::read_conjugacy_classes_and_normalizers" << endl;
		}

	T->A->read_conjugacy_classes_and_normalizers_from_MAGMA(
			fname,
			nb_classes,
			perms,
			class_size,
			class_order_of_element,
			class_normalizer_order,
			class_normalizer_number_of_generators,
			normalizer_generators_perms,
			verbose_level - 1);


	PA = NEW_OBJECT(projective_space_with_action);

	int f_semilinear;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}
	PA->init(
		F, 3 /* n */, f_semilinear,
		FALSE /* f_init_incidence_structure */,
		verbose_level);



	longinteger_object go;
	longinteger_domain D;

	T->A->group_order(go);
	cout << "The group has order " << go << endl;

	char fname_latex[1000];
	strcpy(fname_latex, fname);

	replace_extension_with(fname_latex, ".tex");

	{
	ofstream fp(fname_latex);
	char title[1000];
	latex_interface L;

	sprintf(title, "Conjugacy classes of $%s$",
			T->A->label_tex);

	L.head(fp,
		FALSE /* f_book */, TRUE /* f_title */,
		title, "computed by MAGMA" /* const char *author */,
		FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
		TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
		NULL /* extra_praeamble */);
	//latex_head_easy(fp);

	fp << "\\section{Conjugacy classes in $"
			<< T->A->label_tex << "$}" << endl;


	fp << "The group order is " << endl;
	fp << "$$" << endl;
	go.print_not_scientific(fp);
	fp << endl;
	fp << "$$" << endl;


	cout << "The conjugacy classes are:" << endl;
	for (i = 0; i < nb_classes; i++) {
		strong_generators *gens;
		longinteger_object go1, Class_size, centralizer_order;
		int goi;
		vector_ge *nice_gens;


		goi = class_order_of_element[i];
		gens = NEW_OBJECT(strong_generators);

		gens->init_from_permutation_representation(T->A,
			perms + i * T->A->degree,
			1, goi, nice_gens,
			verbose_level);

		if (f_v) {
			cout << "action::normalizer_using_MAGMA "
				"after gens->init_from_permutation_"
				"representation" << endl;
		}

		Class_size.create(class_size[i]);

		D.integral_division_exact(go, Class_size, centralizer_order);



		int ngo;
		int nb_perms;
		strong_generators *N_gens;
		vector_ge *nice_gens_N;

		ngo = class_normalizer_order[i];
		nb_perms = class_normalizer_number_of_generators[i];

		//int *class_normalizer_order;
		//int *class_normalizer_number_of_generators;
		//int **normalizer_generators_perms;

		N_gens = NEW_OBJECT(strong_generators);
		N_gens->init_from_permutation_representation(T->A,
				normalizer_generators_perms[i],
				nb_perms, ngo, nice_gens_N,
				verbose_level - 1);

		cout << "class " << i << " / " << nb_classes
			<< " size = " << class_size[i]
			<< " order of element = " << class_order_of_element[i]
			<< " centralizer order = " << centralizer_order
			<< " normalizer order = " << ngo
			<< " : " << endl;
		cout << "packing::read_conjugacy_classes_and_normalizers created "
				"generators for a group" << endl;
		gens->print_generators();
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "packing::read_conjugacy_classes_and_normalizers "
				"The group has order " << go1 << endl;

		fp << "\\bigskip" << endl;
		fp << "\\subsection*{Class " << i << " / "
				<< nb_classes << "}" << endl;
		fp << "Order of element = " << class_order_of_element[i]
				<< "\\\\" << endl;
		fp << "Class size = " << class_size[i] << "\\\\" << endl;
		fp << "Centralizer order = " << centralizer_order
				<< "\\\\" << endl;
		fp << "Normalizer order = " << ngo
				<< "\\\\" << endl;

		int *Elt = NULL;


		if (class_order_of_element[i] > 1) {
			Elt = nice_gens->ith(0);
			fp << "Representing element is" << endl;
			fp << "$$" << endl;
			T->A->element_print_latex(Elt, fp);
			fp << "$$" << endl;
			fp << "$";
			T->A->element_print_for_make_element(Elt, fp);
			fp << "$\\\\" << endl;



		}
		fp << "The normalizer is generated by:\\\\" << endl;
		N_gens->print_generators_tex(fp);


		if (class_order_of_element[i] > 1) {
			fp << "The fix structure is:\\\\" << endl;
			PA->report_fixed_objects_in_PG_3_tex(
					Elt, fp,
					verbose_level);

			fp << "The orbit structure is:\\\\" << endl;
			PA->report_orbits_in_PG_3_tex(
				Elt, fp,
				verbose_level);
		}
		if (class_order_of_element[i] > 1) {

			PA->report_decomposition_by_single_automorphism(
					Elt, fp,
					verbose_level);
			// PA->
			//action *A; // linear group PGGL(d,q)
			//action *A_on_lines; // linear group PGGL(d,q) acting on lines


		}

		FREE_int(normalizer_generators_perms[i]);

		FREE_OBJECT(nice_gens_N);
		FREE_OBJECT(nice_gens);
		FREE_OBJECT(N_gens);
		FREE_OBJECT(gens);
		}
	L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< file_size(fname_latex) << endl;

	FREE_int(perms);
	FREE_int(class_size);
	FREE_int(class_order_of_element);
	FREE_int(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	FREE_OBJECT(PA);

	if (f_v) {
		cout << "packing::read_conjugacy_classes_and_normalizers done" << endl;
		}
}


void packing::report_fixed_objects(int *Elt,
		char *fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, cnt;
	int v[4];
	
	if (f_v) {
		cout << "packing::report_fixed_objects" << endl;
		}


	{
	ofstream fp(fname_latex);
	char title[1000];
	latex_interface L;

	sprintf(title, "Fixed Objects");
	
	L.head(fp,
		FALSE /* f_book */, TRUE /* f_title */,
		title, "" /* const char *author */, 
		FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */, 
		TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */, 
		NULL /* extra_praeamble */);
	//latex_head_easy(fp);
	
	fp << "\\section{Fixed Objects}" << endl;



	fp << "The element" << endl;
	fp << "$$" << endl;
	T->A->element_print_latex(Elt, fp);
	fp << "$$" << endl;
	fp << "has the following fixed objects:" << endl;


	fp << "\\subsection{Fixed Points}" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = T->A->element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			fp << i << " : ";
			int_vec_print(fp, v, 4);
			fp << "\\\\" << endl;
			cnt++;
			}
		}

	fp << "\\subsection{Fixed Lines}" << endl;

	{
	action *A2;
	
	A2 = T->A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			fp << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
			fp << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}

	fp << "\\subsection{Fixed Planes}" << endl;

	{
	action *A2;
	
	A2 = T->A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			fp << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
			fp << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}


	L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< file_size(fname_latex) << endl;

	
	if (f_v) {
		cout << "packing::report_fixed_objects done" << endl;
		}
}

void packing::make_element(int idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "packing::make_element" << endl;
		}
	int goi = 2; // was 5 before
	int nb_perms = 3;
	int perms[] = {
			// the order of PGGL(4,4) is 1974067200
			// three elements of order 2:
			1, 2, 9, 26, 45, 6, 7, 8, 3, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 4, 30, 29, 28, 27, 34, 33, 32, 31, 38, 37, 36, 35, 41, 42, 39, 40, 44, 43, 5, 48, 49, 46, 47, 52, 53, 50, 51, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62, 65, 64, 67, 66, 69, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 
			1, 2, 3, 4, 64, 6, 8, 7, 9, 11, 10, 12, 13, 15, 14, 20, 21, 23, 22, 16, 17, 19, 18, 25, 24, 26, 31, 33, 32, 34, 27, 29, 28, 30, 35, 37, 36, 38, 54, 56, 55, 57, 62, 63, 65, 58, 60, 59, 61, 66, 68, 67, 69, 39, 41, 40, 42, 46, 48, 47, 49, 43, 44, 5, 45, 50, 52, 51, 53, 70, 72, 71, 73, 78, 80, 79, 81, 74, 76, 75, 77, 82, 84, 83, 85, 
			1, 2, 13, 35, 50, 6, 7, 8, 12, 15, 14, 9, 3, 11, 10, 21, 20, 23, 22, 17, 16, 19, 18, 36, 37, 38, 31, 32, 33, 34, 27, 28, 29, 30, 4, 24, 25, 26, 47, 46, 49, 48, 51, 53, 52, 40, 39, 42, 41, 5, 43, 45, 44, 60, 61, 58, 59, 56, 57, 54, 55, 68, 69, 66, 67, 64, 65, 62, 63, 73, 72, 71, 70, 77, 76, 75, 74, 81, 80, 79, 78, 85, 84, 83, 82, 
			// class orders : order of centralizer : class rep
			// 5355 : 368640 matrix(1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1)
			// 48960 : 40320 identity matrix, frobenius automorphism
			// 64260 : 30720 ('problem group')  matrix(1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1)
			
			// three elements of order 5:
			//22, 38, 39, 54, 81, 76, 40, 62, 37, 77, 63, 64, 36, 42, 74, 75, 35, 65, 41, 23, 1, 20, 21, 47, 84, 30, 18, 14, 10, 2, 80, 58, 51, 26, 5, 72, 66, 34, 70, 45, 32, 68, 59, 25, 50, 13, 16, 7, 11, 48, 57, 27, 83, 3, 8, 17, 15, 85, 29, 55, 46, 69, 31, 44, 71, 53, 24, 78, 60, 4, 52, 61, 79, 33, 67, 73, 43, 28, 82, 49, 56, 6, 9, 12, 19, 
			//22, 14, 16, 24, 44, 10, 18, 2, 13, 11, 7, 8, 15, 17, 3, 9, 12, 6, 19, 23, 1, 20, 21, 53, 60, 78, 82, 56, 49, 28, 45, 32, 70, 68, 64, 74, 36, 42, 75, 65, 41, 35, 31, 69, 71, 57, 83, 27, 48, 52, 4, 79, 61, 47, 30, 84, 54, 58, 80, 26, 51, 38, 40, 62, 76, 72, 66, 5, 34, 67, 73, 33, 43, 39, 37, 77, 63, 81, 59, 50, 25, 29, 46, 55, 85, 
			//20, 84, 46, 72, 38, 48, 29, 56, 54, 27, 82, 28, 47, 55, 83, 57, 30, 49, 85, 21, 22, 23, 1, 33, 68, 44, 80, 60, 52, 25, 76, 41, 37, 64, 2, 16, 3, 12, 13, 10, 19, 8, 65, 42, 77, 24, 51, 59, 79, 43, 66, 31, 70, 61, 81, 26, 53, 32, 71, 5, 67, 17, 6, 14, 11, 39, 74, 62, 35, 36, 63, 75, 40, 9, 15, 7, 18, 69, 45, 73, 34, 50, 4, 78, 58, 
};
	for (i = 0; i < nb_perms * T->A->degree; i++) {
		perms[i]--;
		}

	strong_generators *gens;
	longinteger_object go;
	vector_ge *nice_gens;

	gens = NEW_OBJECT(strong_generators);
	gens->init_from_permutation_representation(T->A,
		perms + idx * T->A->degree,
		1, goi, nice_gens,
		verbose_level);

	if (f_v) {
		cout << "packing::make_element "
				"created generators for a group" << endl;
		gens->print_generators();
		}
	gens->group_order(go);
	if (f_v) {
		cout << "prime_at_a_time::make_element "
				"The group has order " << go << endl;
		}

	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "packing::make_element done" << endl;
		}
}

void packing::centralizer(int idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt;
	int *Data;
	int *Poly2;
	int *Poly4;
	const char *poly;
	char prefix[1000];
	
	if (f_v) {
		cout << "packing::centralizer idx=" << idx << endl;
		}
	sprintf(prefix, "element_%d", idx);
	
	Elt = NEW_int(T->A->elt_size_in_int);
	Data = NEW_int(17);
	Poly2 = NEW_int(3);
	Poly4 = NEW_int(5);

	poly = get_primitive_polynomial(q, 2, 0 /*verbose_level */);
	unipoly_object m;
	unipoly_domain D(F);
		
	D.create_object_by_rank_string(m, poly, verbose_level - 2);
	for (i = 0; i <= 2; i++) {
		Poly2[i] = D.s_i(m, i);
		}
	if (f_v) {
		cout << "packing::centralizer The coefficients "
				"of the polynomial are:" << endl;
		int_vec_print(cout, Poly2, 3);
		cout << endl;
		}
	poly = get_primitive_polynomial(q, 4, 0 /*verbose_level */);
		
	D.create_object_by_rank_string(m, poly, verbose_level - 2);
	for (i = 0; i <= 4; i++) {
		Poly4[i] = D.s_i(m, i);
		}
	if (f_v) {
		cout << "packing::centralizer The coefficients "
				"of the polynomial are:" << endl;
		int_vec_print(cout, Poly4, 5);
		cout << endl;
		}

	int_vec_zero(Data, 17);

	if (idx == 0) {
		Data[1 * 4 + 0] = 1;
		for (i = 0; i < 2; i++) {
			Data[i * 4 + 1] = F->negate(Poly2[i]);
			}
		Data[2 * 4 + 2] = 3;
		Data[3 * 4 + 3] = 3;
		}
	else if (idx == 1) {
		Data[1 * 4 + 0] = 1;
		for (i = 0; i < 2; i++) {
			Data[i * 4 + 1] = F->negate(Poly2[i]);
			}
		Data[3 * 4 + 2] = 1;
		for (i = 0; i < 2; i++) {
			Data[(2 + i) * 4 + 3] = F->negate(Poly2[i]);
			}
		}
	else if (idx == 2) {
		int d[16] = {0,1,0,0,  3,3,0,0,  0,0,1,2,  0,0,2,0}; // AB
		for (i = 0; i < 16; i++) {
			Data[i] = d[i];
			}
		}
	else if (idx == 3) {
		for (i = 0; i < 3; i++) {
			Data[(i + 1) * 4 + i] = 1;
			}
		for (i = 0; i < 4; i++) {
			Data[i * 4 + 3] = F->negate(Poly4[i]);
			}
		}
	

	if (f_v) {
		cout << "packing::centralizer Matrix:" << endl;
		int_matrix_print(Data, 4, 4);
		}

	T->A->make_element(Elt, Data, 0 /* verbose_level */);

	int o;

	o = T->A->element_order(Elt);
	if (f_v) {
		cout << "packing::centralizer Elt:" << endl;
		T->A->element_print_quick(Elt, cout);
		cout << "packing::centralizer on points:" << endl;
		T->A->element_print_as_permutation(Elt, cout);
		cout << "packing::centralizer on lines:" << endl;
		T->A2->element_print_as_permutation(Elt, cout);
		}

	cout << "packing::centralizer the element has order " << o << endl;
	

	if (idx == 3) {
		T->A->element_power_int_in_place(Elt, 17, 0 /* verbose_level */);
		if (f_v) {
			cout << "packing::centralizer "
					"after power(17), Elt:" << endl;
			T->A->element_print_quick(Elt, cout);
			cout << "packing::centralizer on points:" << endl;
			T->A->element_print_as_permutation(Elt, cout);
			cout << "packing::centralizer on lines:" << endl;
			T->A2->element_print_as_permutation(Elt, cout);
			}
		}
	
	if (f_v) {
		cout << "packing::centralizer "
				"before centralizer_using_MAGMA" << endl;
		}

	T->A->centralizer_using_MAGMA(prefix,
			T->A->Sims, Elt, verbose_level);
	
	if (f_v) {
		cout << "packing::centralizer "
				"after centralizer_using_MAGMA" << endl;
		}
	

	if (f_v) {
		cout << "packing::centralizer done" << endl;
		}
}

void packing::centralizer_of_element(
	const char *element_description,
	const char *label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	char prefix[1000];
	
	if (f_v) {
		cout << "packing::centralizer_of_element label=" << label
				<< " element_description="
				<< element_description << endl;
		}
	sprintf(prefix, "element_%s", label);
	
	Elt = NEW_int(T->A->elt_size_in_int);

	int *data;
	int data_len;


	int_vec_scan(element_description, data, data_len);

	if (data_len < 16) {
		cout << "data_len < 16" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "packing::centralizer_of_element Matrix:" << endl;
		int_matrix_print(data, 4, 4);
		}

	T->A->make_element(Elt, data, 0 /* verbose_level */);

	int o;

	o = T->A->element_order(Elt);
	if (f_v) {
		cout << "packing::centralizer_of_element Elt:" << endl;
		T->A->element_print_quick(Elt, cout);
		cout << "packing::centralizer_of_element on points:" << endl;
		T->A->element_print_as_permutation(Elt, cout);
		cout << "packing::centralizer_of_element on lines:" << endl;
		T->A2->element_print_as_permutation(Elt, cout);
		}

	cout << "packing::centralizer_of_element "
			"the element has order " << o << endl;
	

	
	if (f_v) {
		cout << "packing::centralizer_of_element "
				"before centralizer_using_MAGMA" << endl;
		}

	T->A->centralizer_using_MAGMA(prefix,
			T->A->Sims, Elt, verbose_level);
	
	if (f_v) {
		cout << "packing::centralizer_of_element "
				"after centralizer_using_MAGMA" << endl;
		}
	

	FREE_int(data);
	
	if (f_v) {
		cout << "packing::centralizer_of_element done" << endl;
		}
}

int packing::test_if_orbit_is_partial_packing(
	schreier *Orbits, int orbit_idx,
	int *orbit1, int verbose_level)
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len;
	int a, b;
	int i, j, ret;

	if (f_v) {
		cout << "packing::test_if_orbit_is_partial_packing "
				"orbit_idx = " << orbit_idx << endl;
		}
	Orbits->get_orbit(orbit_idx,
			orbit1, len, 0 /* verbose_level*/);
	for (i = 0; i < len; i++) {
		a = orbit1[i];
		for (j = i + 1; j < len; j++) {
			b = orbit1[j];
			if (!test_if_spreads_are_disjoint_based_on_table(
					a, b)) {
				break;
				}
			}
		if (j < len) {
			break;
			}
		}
	if (i < len) {
		//cout << "is NOT a partial packing" << endl;
		ret = FALSE;
		}
	else {
		ret = TRUE;
		//cout << "IS a partial packing" << endl;
		}
	return ret;
}

int packing::test_if_pair_of_orbits_are_adjacent(
	schreier *Orbits, int a, int b,
	int *orbit1, int *orbit2,
	int verbose_level)
// tests if every spread from orbit a
// is line-disjoint from every spread from orbit b
{
	int f_v = FALSE; // (verbose_level >= 1);
	int len1, len2;
	int s1, s2;
	int i, j;

	if (f_v) {
		cout << "packing::test_if_pair_of_orbits_"
				"are_adjacent a=" << a << " b=" << b << endl;
		}
	if (a == b) {
		return FALSE;
		}
	Orbits->get_orbit(a, orbit1, len1, 0 /* verbose_level*/);
	Orbits->get_orbit(b, orbit2, len2, 0 /* verbose_level*/);
	for (i = 0; i < len1; i++) {
		s1 = orbit1[i];
		for (j = 0; j < len2; j++) {
			s2 = orbit2[j];
			if (!test_if_spreads_are_disjoint_based_on_table(
					s1, s2)) {
				break;
				}
			}
		if (j < len2) {
			break;
			}
		}
	if (i < len1) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

int packing::test_if_pair_of_sets_are_adjacent(
		int *set1, int sz1,
		int *set2, int sz2,
		int verbose_level)
{
	int f_v = FALSE; // (verbose_level >= 1);
	int s1, s2;
	int i, j;

	if (f_v) {
		cout << "packing::test_if_"
				"pair_of_sets_are_adjacent" << endl;
		}
	for (i = 0; i < sz1; i++) {
		s1 = set1[i];
		for (j = 0; j < sz2; j++) {
			s2 = set2[j];
			if (!test_if_spreads_are_disjoint_based_on_table(
					s1, s2)) {
				break;
				}
			}
		if (j < sz2) {
			break;
			}
		}
	if (i < sz1) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

int packing::test_if_spreads_are_disjoint_based_on_table(int a, int b)
{
		return Spread_tables->test_if_spreads_are_disjoint(a, b);
}


// #############################################################################
// global functions:
// #############################################################################

void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level)
{
	packing *P = (packing *) data;
	
	P->compute_klein_invariants(Iso, verbose_level);
}


void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level)
{
	packing *P = (packing *) data;
	
	P->report(Iso, verbose_level);
}


void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	packing *P = (packing *) EC->user_data;

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_candidates=" << nb_candidates << endl;
		}

	P->lifting_prepare_function_new(
		EC, starter_case,
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"after lifting_prepare_function_new" << endl;
		}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new "
				"nb_rows=" << Dio->m
				<< " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "packing_lifting_prepare_function_new done" << endl;
		}
}



void packing_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	packing *P = (packing *) data;
	int f_v = (verbose_level >= 1);
	int i, k, a, b;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "packing_early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	a = S[len - 1];
	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		b = candidates[i];

		if (b == a) {
			continue;
			}
		if (P->bitvector_adjacency) {
			k = Combi.ij2k(a, b, P->Spread_tables->nb_spreads);
			if (bitvector_s_i(P->bitvector_adjacency, k)) {
				good_candidates[nb_good_candidates++] = b;
				}
			}
		else {
			if (P->spreads_are_disjoint(a, b)) {
				good_candidates[nb_good_candidates++] = b;
				}
			}
		}
	if (f_v) {
		cout << "packing_early_test_function done" << endl;
		}
}




int count(int *Inc, int n, int m, int *set, int t)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
				}
			}
		if (h == t) {
			nb++;
			}
		}
	return nb;
}

int count_and_record(int *Inc,
		int n, int m, int *set, int t, int *occurances)
{
	int i, j;
	int nb, h;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < t; h++) {
			i = set[h];
			if (Inc[i * m + j] == 0) {
				break;
				}
			}
		if (h == t) {
			occurances[nb++] = j;
			}
		}
	return nb;
}

int packing_spread_compare_func(void *data, int i, int j, void *extra_data)
{
	packing *P = (packing *) extra_data;
	int **Sets = (int **) data;
	int ret;

	ret = int_vec_compare(Sets[i], Sets[j], P->spread_size);
	return ret;
}

void packing_swap_func(void *data, int i, int j, void *extra_data)
{
	packing *P = (packing *) extra_data;
	int *d = P->tmp_isomorphism_type_of_spread;
	int **Sets = (int **) data;
	int *p;
	int a;

	p = Sets[i];
	Sets[i] = Sets[j];
	Sets[j] = p;

	a = d[i];
	d[i] = d[j];
	d[j] = a;
}


}}

