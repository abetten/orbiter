/*
 * spread_table_with_selection.cpp
 *
 *  Created on: Jun 27, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


static int spread_table_with_selection_compare_func(
		void *data, int i, int j, void *extra_data);
static void spread_table_with_selection_swap_func(
		void *data, int i, int j, void *extra_data);


spread_table_with_selection::spread_table_with_selection()
{
	Record_birth();

	Spread_table_description = NULL;

	PA = NULL;

	SD = NULL;

	dimension_of_spread_elements = 0;

	Spread_classify = NULL;
	F = NULL;
	n = 0;
	q = 0;
	spread_size = 0;
	size_of_packing = 0;
	nb_lines = 0;
#if 0
	f_select_spread = false;
	//select_spread_text = NULL;
#endif
	select_spread = NULL;
	select_spread_nb = 0;
	//path_to_spread_tables = NULL;

	spread_reps = NULL;
	spread_reps_idx = NULL;
	spread_orbit_length = NULL;
	nb_spread_reps = 0;
	total_nb_of_spreads = 0;
	nb_iso_types_of_spreads = 0;
	sorted_packing = NULL;
	dual_packing = NULL;

	Spread_table = NULL;
	tmp_isomorphism_type_of_spread = NULL;

	Bitvec = NULL;
	//bitvector_adjacency = NULL;
	//bitvector_length = 0;

	A_on_spreads = NULL;


}

spread_table_with_selection::~spread_table_with_selection()
{
	Record_death();

	if (SD) {
		FREE_OBJECT(SD);
	}
	if (select_spread) {
		FREE_int(select_spread);
	}
	if (spread_reps) {
		FREE_lint(spread_reps);
	}
	if (spread_reps_idx) {
		FREE_int(spread_reps_idx);
	}
	if (spread_orbit_length) {
		FREE_lint(spread_orbit_length);
	}
	if (sorted_packing) {
		FREE_int(sorted_packing);
	}
	if (dual_packing) {
		FREE_int(dual_packing);
	}
	if (Spread_table) {
		FREE_OBJECT(Spread_table);
	}
	if (Bitvec) {
		FREE_OBJECT(Bitvec);
	}
}


void spread_table_with_selection::do_spread_table_init(
		geometry::finite_geometries::spread_table_description *Spread_table_description,
		int verbose_level)
// creates geometry::spread_domain and spreads::spread_classify objects
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::do_spread_table_init" << endl;
	}

	spread_table_with_selection::Spread_table_description = Spread_table_description;

	if (Spread_table_description->f_space) {
		PA = Get_projective_space(Spread_table_description->space_label);
	}
	else {
		cout << "spread_table_with_selection::do_spread_table_init "
				"please specify a projective space using -space <label>" << endl;
		exit(1);
	}


	if (Spread_table_description->f_rk) {
		dimension_of_spread_elements = Spread_table_description->rk;
	}
	else {
		cout << "spread_table_with_selection::do_spread_table_init "
				"please specify the vector space dimension of the spread elements using -rk <rk>" << endl;
		exit(1);
	}
	algebra::basic_algebra::matrix_group *Mtx;


	n = PA->A->matrix_group_dimension();
	Mtx = PA->A->get_matrix_group();
	F = Mtx->GFq;
	q = F->q;
	if (f_v) {
		cout << "spread_table_with_selection::do_spread_table_init n=" << n
				<< " k=" << dimension_of_spread_elements
				<< " q=" << q << endl;
	}



	SD = NEW_OBJECT(geometry::finite_geometries::spread_domain);

	if (f_v) {
		cout << "spread_table_with_selection::do_spread_table_init "
				"before SD->init_spread_domain" << endl;
	}

	SD->init_spread_domain(
			F,
			n, dimension_of_spread_elements,
			verbose_level - 1);

	if (f_v) {
		cout << "spread_table_with_selection::do_spread_table_init "
				"after SD->init_spread_domain" << endl;
	}





	if (Spread_table_description->f_control) {


		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"before Spread_classify->init" << endl;
		}

		spreads::spread_classify_description *Descr;

		Descr = NEW_OBJECT(spreads::spread_classify_description);

		Descr->f_poset_classification_control = true;
		Descr->poset_classification_control_label = Spread_table_description->control_label;

		Descr->f_projective_space = false;
		Descr->f_recoordinatize = true;


		Spread_classify = NEW_OBJECT(spreads::spread_classify);

		//spreads::spread_classify *Spread_classify;

		Spread_classify->init(Descr, SD, PA, verbose_level - 1);

		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"after Spread_classify->init" << endl;
		}

	#if 0
		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"before Spread_classify->init2" << endl;
		}
		Spread_classify->init2(Control, verbose_level);
		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"after Spread_classify->init2" << endl;
		}
	#endif


	}
	else {

		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"we don't allocate Spread_classify object since -control is not given" << endl;
		}

	}




	//spreads::spread_table_with_selection *Spread_table_with_selection;

	//Spread_table_with_selection = NEW_OBJECT(spreads::spread_table_with_selection);


	if (Spread_table_description->f_restricted_table) {
		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"restricted table " << Spread_table_description->restricted_table_label << endl;
			cout << "spread_table_with_selection::do_spread_table_init "
					"subset " << Spread_table_description->restricted_table_subset << endl;
		}

		init_restricted_table(verbose_level);

	}
	else {

		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"before init" << endl;
		}
		init(
			verbose_level);
		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"after init" << endl;
		}

		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"before compute_spread_table" << endl;
		}
		compute_spread_table(verbose_level);
		if (f_v) {
			cout << "spread_table_with_selection::do_spread_table_init "
					"after compute_spread_table" << endl;
		}

	}





	if (f_v) {
		cout << "spread_table_with_selection::do_spread_table_init done" << endl;
	}


}


void spread_table_with_selection::init_restricted_table(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table" << endl;
	}

	F = PA->F;
	q = F->q;

	spread_size = SD->spread_size;
	size_of_packing = q * q + q + 1;

	nb_lines = SD->nCkq;
	//nb_lines = Spread_classify->A2->degree;

	if (Spread_table_description->f_path) {
		spread_table_with_selection::path_to_spread_tables = Spread_table_description->path;
	}
	else {
		spread_table_with_selection::path_to_spread_tables = "";
	}


	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table q=" << q << endl;
		cout << "spread_table_with_selection::init_restricted_table nb_lines=" << nb_lines << endl;
		cout << "spread_table_with_selection::init_restricted_table spread_size=" << spread_size << endl;
		cout << "spread_table_with_selection::init_restricted_table size_of_packing=" << size_of_packing << endl;
	}


	spreads::spread_table_with_selection *Old_table;

	Old_table = Get_spread_table(Spread_table_description->restricted_table_label);

	long int *subset;
	int subset_sz;
	other::data_structures::vector_builder *vb_Subset;

	vb_Subset = Get_vector(Spread_table_description->restricted_table_subset);
	subset = vb_Subset->v;
	subset_sz = vb_Subset->len;
	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"subset " << Spread_table_description->restricted_table_subset << " has size " << subset_sz << endl;
		Lint_vec_print(cout, subset, subset_sz);
		cout << endl;
	}

	long int *Spreads_selected;
	int *isomorphism_type_of_spread;
	int i, a;

	Spreads_selected = NEW_lint(subset_sz * spread_size);
	isomorphism_type_of_spread = NEW_int(subset_sz);
	for (i = 0; i < subset_sz; i++) {
		a = subset[i];
		Lint_vec_copy(Old_table->get_spread(a), Spreads_selected + i * spread_size, spread_size);
		isomorphism_type_of_spread[i] = Old_table->Spread_table->iso_types_of_spreads[a];
	}

	Spread_table = NEW_OBJECT(geometry::finite_geometries::spread_table);



	Spread_table->nb_spreads = subset_sz;

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"before Spread_tables->init" << endl;
	}

	Spread_table->init(
			Spread_table_description,
				verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"after Spread_tables->init" << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"We will use " << subset_sz << " spreads of the original "
				<< Old_table->Spread_table->nb_spreads << endl;
	}


	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"before Spread_table->init_spread_table" << endl;
	}

	Spread_table->init_spread_table(
			subset_sz,
			Spreads_selected, isomorphism_type_of_spread,
			verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"after Spread_tables->init_spread_table" << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table "
				"before Spread_table->save" << endl;
	}

	Spread_table->save(verbose_level);

	sorted_packing = NEW_int(size_of_packing);
	dual_packing = NEW_int(size_of_packing);


	if (f_v) {
		cout << "spread_table_with_selection::init_restricted_table done" << endl;
	}
}

void spread_table_with_selection::init(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::init" << endl;
	}

	if (Spread_table_description->f_iso_types) {
		Int_vec_scan(Spread_table_description->iso_types_string,
				select_spread, select_spread_nb);
	}
	else {
		cout << "spread_table_with_selection::init please specify the isomorphism "
				"types of spreads using -iso_types <list>" << endl;
		exit(1);
	}


	F = PA->F;
	q = F->q;

	spread_size = SD->spread_size;
	size_of_packing = q * q + q + 1;
	nb_lines = SD->nCkq;
	//nb_lines = Spread_classify->A2->degree;

	if (Spread_table_description->f_path) {
		spread_table_with_selection::path_to_spread_tables = Spread_table_description->path;
	}
	else {
		spread_table_with_selection::path_to_spread_tables = "";
	}


	if (f_v) {
		cout << "spread_table_with_selection::init q=" << q << endl;
		cout << "spread_table_with_selection::init nb_lines=" << nb_lines << endl;
		cout << "spread_table_with_selection::init spread_size=" << spread_size << endl;
		cout << "spread_table_with_selection::init size_of_packing=" << size_of_packing << endl;
	}


	if (Spread_table_description->f_iso_types) {
		cout << "spread_table_with_selection::init selected spreads are "
				"from the following orbits: ";
		Int_vec_print(cout,
				select_spread,
				select_spread_nb);
		cout << endl;
	}


	Spread_table = NEW_OBJECT(geometry::finite_geometries::spread_table);



	if (f_v) {
		cout << "spread_table_with_selection::init "
				"before predict_spread_table_length" << endl;
	}
	predict_spread_table_length(
			Spread_classify->A,
			Spread_classify->A->Strong_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "spread_table_with_selection::init "
				"after predict_spread_table_length" << endl;
		cout << "spread_table_with_selection::init "
				"total_nb_of_spreads = " << total_nb_of_spreads << endl;
	}

	Spread_table->nb_spreads = total_nb_of_spreads;

	if (f_v) {
		cout << "spread_table_with_selection::init "
				"before Spread_tables->init" << endl;
	}

	Spread_table->init(
			Spread_table_description,
				verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::init "
				"after Spread_tables->init" << endl;
	}

	if (f_v) {
		cout << "We will use " << nb_spread_reps << " isomorphism types of spreads, "
				"this will give a total number of " << Spread_table->nb_spreads
				<< " labeled spreads" << endl;
	}

	sorted_packing = NEW_int(size_of_packing);
	dual_packing = NEW_int(size_of_packing);


	if (f_v) {
		cout << "spread_table_with_selection::init done" << endl;
	}
}

void spread_table_with_selection::compute_spread_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table" << endl;
	}




	if (Spread_table->files_exist(verbose_level)) {
		if (f_v) {
			cout << "spread_table_with_selection::compute_spread_table files exist, "
					"reading" << endl;
		}

		Spread_table->load(verbose_level);

		if (f_v) {
			cout << "spread_table_with_selection::compute_spread_table "
					"after Spread_tables->load" << endl;
		}
	}
	else {

		if (f_v) {
			cout << "spread_table_with_selection::compute_spread_table "
					"files do not exist, computing the spread table" << endl;
		}

		if (f_v) {
			cout << "spread_table_with_selection::compute_spread_table "
					"before compute_spread_table_from_scratch" << endl;
		}
		compute_spread_table_from_scratch(verbose_level - 1);
		if (f_v) {
			cout << "spread_table_with_selection::compute_spread_table "
					"after compute_spread_table_from_scratch" << endl;
		}
	}


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table "
				"before create_action_on_spreads" << endl;
	}
	create_action_on_spreads(verbose_level);
	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table "
				"after create_action_on_spreads" << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table done" << endl;
	}
}

void spread_table_with_selection::compute_spread_table_from_scratch(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch" << endl;
	}

	int i, j;
	long int **Sets;
	int nb_spreads;
	int *Prev;
	int *Label;
	int *First;
	int *Len;
	int *isomorphism_type_of_spread;
	other::data_structures::sorting Sorting;


	nb_spreads = Spread_table->nb_spreads;

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before make_spread_table" << endl;
	}


	make_spread_table(
			Spread_classify->A, Spread_classify->A2, Spread_classify->A->Strong_gens,
			Sets,
			Prev, Label, First, Len,
			isomorphism_type_of_spread,
			verbose_level);

	// does not sort the spread table




	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"after make_spread_table" << endl;
	}


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch before "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}
	tmp_isomorphism_type_of_spread = isomorphism_type_of_spread;

	// for packing_swap_func

	int *original_position;
	int *original_position_inv;

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before NEW_int(total_nb_of_spreads)" << endl;
	}


	original_position = NEW_int(total_nb_of_spreads);
	original_position_inv = NEW_int(total_nb_of_spreads);
	for (i = 0; i < total_nb_of_spreads; i++) {
		original_position[i] = i;
	}

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Sorting.Heapsort_general_with_log" << endl;
	}
	Sorting.Heapsort_general_with_log(
			Sets, original_position, total_nb_of_spreads,
			spread_table_with_selection_compare_func,
			spread_table_with_selection_swap_func,
			this);
	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch after "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}
	for (i = 0; i < total_nb_of_spreads; i++) {
		j = original_position[i];
		original_position_inv[j] = i;
	}

	long int *Spread_table_sorted;

	Spread_table_sorted = NEW_lint(nb_spreads * spread_size);
	for (i = 0; i < nb_spreads; i++) {
		Lint_vec_copy(Sets[i], Spread_table_sorted + i * spread_size, spread_size);
	}

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->init" << endl;
	}

	Spread_table->init(
			Spread_table_description,
			verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"after Spread_table->init" << endl;
	}

	string fname;

	fname = Spread_table->prefix + "_gens.tex";
	{
		std::ofstream ost(fname);

		layer1_foundations::other::l1_interfaces::latex_interface Latex;


		Latex.head_easy(ost);

		Spread_classify->A->Strong_gens->print_generators_as_permutations_tex(
				ost, Spread_classify->A2);

		Latex.foot(ost);

	}


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->init_spread_table" << endl;
	}

	Spread_table->init_spread_table(
			nb_spreads,
			Spread_table_sorted, isomorphism_type_of_spread,
			verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"after Spread_tables->init_spread_table" << endl;
	}

	long int *Dual_spread_idx;
	long int *self_dual_spread_idx;
	int nb_self_dual_spreads;

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->compute_dual_spreads" << endl;
	}
	Spread_table->compute_dual_spreads(
			Sets,
				Dual_spread_idx,
				self_dual_spread_idx,
				nb_self_dual_spreads,
				verbose_level);
	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"after Spread_table->compute_dual_spreads" << endl;
	}


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->init_tables" << endl;
	}

	Spread_table->init_tables(
			nb_spreads,
			Spread_table_sorted, isomorphism_type_of_spread,
			Dual_spread_idx,
			self_dual_spread_idx, nb_self_dual_spreads,
			verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"after Spread_table->init_tables" << endl;
	}


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"preparing schreier_table" << endl;
	}

	int *schreier_table;

	schreier_table = NEW_int(nb_spreads * 4);

	for (i = 0; i < nb_spreads; i++) {
		schreier_table[i * 4 + 0] = original_position[i];
		schreier_table[i * 4 + 1] = original_position_inv[i];
		//schreier_table[i * 4 + 2] = Prev[i];
		//schreier_table[i * 4 + 3] = Label[i];
		if (Prev[original_position[i]] == -1) {
			schreier_table[i * 4 + 2] = -1;
		}
		else {
			schreier_table[i * 4 + 2] = original_position_inv[Prev[original_position[i]]];
		}
		schreier_table[i * 4 + 3] = Label[original_position[i]];
	}

	FREE_int(Prev);
	FREE_int(Label);
	FREE_int(First);
	FREE_int(Len);


	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->init_schreier_table" << endl;
	}
	Spread_table->init_schreier_table(
			schreier_table, verbose_level);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"before Spread_table->save" << endl;
	}

	Spread_table->save(verbose_level);

#if 1
	if (nb_spreads < 10000) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"We are computing the adjacency matrix" << endl;
		compute_adjacency_matrix(verbose_level - 1);
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"The adjacency matrix has been computed" << endl;
	}
	else {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch "
				"We are NOT computing the adjacency matrix" << endl;
	}
#endif


	for (i = 0; i < nb_spreads; i++) {
		FREE_lint(Sets[i]);
	}
	FREE_plint(Sets);

	if (f_v) {
		cout << "spread_table_with_selection::compute_spread_table_from_scratch done" << endl;
	}
}

void spread_table_with_selection::create_action_on_spreads(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::create_action_on_spreads" << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::create_action_on_spreads "
				"creating action A_on_spreads" << endl;
	}
	A_on_spreads = Spread_classify->A2->Induced_action->create_induced_action_on_sets(
			Spread_table->nb_spreads, spread_size,
			Spread_table->Spread_table,
			verbose_level - 2);

	if (f_v) {
		cout << "spread_table_with_selection::create_action_on_spreads "
				"created action on spreads of degree " << A_on_spreads->degree << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::create_action_on_spreads "
				"creating action A_on_spreads done" << endl;
	}
}

int spread_table_with_selection::find_spread(
		long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "spread_table_with_selection::find_spread" << endl;
	}
	if (A_on_spreads == NULL) {
		cout << "spread_table_with_selection::find_spread A_on_spreads == NULL" << endl;
		exit(1);
	}
	idx = A_on_spreads->G.on_sets->find_set(set, verbose_level);
	return idx;
}

long int *spread_table_with_selection::get_spread(
		int spread_idx)
{
	return Spread_table->get_spread(spread_idx);
}

void spread_table_with_selection::find_spreads_containing_two_lines(
		std::vector<int> &v,
		int line1, int line2, int verbose_level)
{
	Spread_table->find_spreads_containing_two_lines(
			v,
			line1, line2,
			verbose_level);
}

int spread_table_with_selection::test_if_packing_is_self_dual(
		int *packing, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = false;
	int i, a, b;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "spread_table_with_selection::test_if_packing_is_self_dual" << endl;
	}
	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		sorted_packing[i] = a;
	}
	Sorting.int_vec_heapsort(
			sorted_packing, size_of_packing);

	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		b = Spread_table->dual_spread_idx[a];
		dual_packing[i] = b;
	}
	Sorting.int_vec_heapsort(
			dual_packing, size_of_packing);
	if (Sorting.int_vec_compare(
			sorted_packing, dual_packing, size_of_packing) == 0) {
		ret = true;
	}

	if (f_v) {
		cout << "spread_table_with_selection::test_if_packing_is_self_dual done" << endl;
	}
	return ret;
}


void spread_table_with_selection::predict_spread_table_length(
		actions::action *A, groups::strong_generators *Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_do_it, idx, no;
	algebra::ring_theory::longinteger_object go, stab_go;
	algebra::ring_theory::longinteger_domain D;
	combinatorics::knowledge_base::knowledge_base K;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "spread_table_with_selection::predict_spread_table_length" << endl;
	}


	total_nb_of_spreads = 0;

	Strong_gens->group_order(go);
	if (f_v) {
		cout << "spread_table_with_selection::predict_spread_table_length go = " << go << endl;
	}


	nb_iso_types_of_spreads = K.Spread_nb_reps(
			q, Spread_classify->SD->k /* dimension_of_spread_elements */);
	if (f_v) {
		cout << "spread_table_with_selection::predict_spread_table_length "
				"nb_iso_types_of_spreads = " << nb_iso_types_of_spreads << endl;
	}

	spread_reps = NEW_lint(nb_iso_types_of_spreads * spread_size);
	spread_reps_idx = NEW_int(nb_iso_types_of_spreads);
	spread_orbit_length = NEW_lint(nb_iso_types_of_spreads);
	nb_spread_reps = 0;


	for (no = 0; no < nb_iso_types_of_spreads; no++) {

		data_structures_groups::vector_ge *gens;
		string stab_order;

		actions::action_global AcGl;

		AcGl.stabilizer_of_spread_representative(
				A,
				q,
				Spread_classify->SD->k /* dimension_of_spread_elements */, no, gens, stab_order,
				0 /*verbose_level*/);


		if (Sorting.int_vec_search_linear(
				select_spread,
				select_spread_nb, no, idx)) {
			f_do_it = true;
		}
		else {
			f_do_it = false;
		}

		if (f_do_it) {
			if (f_v) {
				cout << "spread_table_with_selection::predict_spread_table_length spread " << no << " is selected" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "spread_table_with_selection::predict_spread_table_length spread " << no << " is skipped" << endl;
			}
		}

		if (f_do_it) {
			long int *rep;
			int sz;

			rep = K.Spread_representative(
					q, Spread_classify->SD->k /* dimension_of_spread_elements*/, no, sz);
			Lint_vec_copy(rep,
					spread_reps + nb_spread_reps * spread_size,
					spread_size);


			spread_reps_idx[nb_spread_reps] = no;


			stab_go.create_from_base_10_string(stab_order);
			//Stab->group_order(stab_go);

			spread_orbit_length[nb_spread_reps] = D.quotient_as_lint(go, stab_go);
			if (f_v) {
				cout << "spread orbit " << no
						<< " has group order "
						<< stab_go << " orbit_length = "
						<< spread_orbit_length[nb_spread_reps] << endl;
			}


			total_nb_of_spreads += spread_orbit_length[nb_spread_reps];
			nb_spread_reps++;


		}


	} // next no



	if (f_v) {
		cout << "spread_table_with_selection::predict_spread_table_length done, "
				"total_nb_of_spreads = " << total_nb_of_spreads << endl;
	}
}


void spread_table_with_selection::make_spread_table(
		actions::action *A, actions::action *A2,
		groups::strong_generators *Strong_gens,
		long int **&Sets, int *&Prev,
		int *&Label, int *&First, int *&Len,
		int *&isomorphism_type_of_spread,
		int verbose_level)
// does not sort the table
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int nb_spreads1;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "spread_table_with_selection::make_spread_table "
				"nb_spread_reps = " << nb_spread_reps << endl;
		cout << "spread_table_with_selection::make_spread_table "
				"total_nb_of_spreads = " << total_nb_of_spreads << endl;
		cout << "spread_table_with_selection::make_spread_table "
				"verbose_level = " << verbose_level << endl;
	}
	Sets = NEW_plint(total_nb_of_spreads);
	Prev = NEW_int(total_nb_of_spreads);
	Label = NEW_int(total_nb_of_spreads);
	First = NEW_int(nb_spread_reps);
	Len = NEW_int(nb_spread_reps);
	isomorphism_type_of_spread = NEW_int(total_nb_of_spreads);

	orbits_schreier::orbit_of_sets *SetOrb;

	SetOrb = NEW_OBJECTS(orbits_schreier::orbit_of_sets, nb_spread_reps);

	for (i = 0; i < nb_spread_reps; i++) {

		if (f_v) {
			cout << "spread_table_with_selection::make_spread_table "
				"Spread " << i << " / "
				<< nb_spread_reps << " computing orbits" << endl;
		}


		SetOrb[i].init(A, A2,
				spread_reps + i * spread_size,
				spread_size, Strong_gens->gens,
				verbose_level);


		if (f_v) {
			cout << "spread_table_with_selection::make_spread_table Spread "
				<< spread_reps_idx[i] << " = " << i << " / "
				<< nb_spread_reps << " has orbit length "
				<< SetOrb[i].used_length << endl;
		}


	} // next i

	nb_spreads1 = 0;

	for (i = 0; i < nb_spread_reps; i++) {

		First[i] = nb_spreads1;
		Len[i] = SetOrb[i].used_length;

		for (j = 0; j < SetOrb[i].used_length; j++) {

			Sets[nb_spreads1] = NEW_lint(spread_size);

			Lint_vec_copy(SetOrb[i].Sets[j], Sets[nb_spreads1], spread_size);

			Prev[nb_spreads1] = First[i] + SetOrb[i].Extra[j * 2 + 0];
			Label[nb_spreads1] = SetOrb[i].Extra[j * 2 + 1];

			isomorphism_type_of_spread[nb_spreads1] = i;


			nb_spreads1++;

		} // next j
	} // next i

	if (f_v) {
		cout << "spread_table_with_selection::make_spread_table "
				"We found " << nb_spreads1 << " spreads in total" << endl;
	}

	if (nb_spreads1 != total_nb_of_spreads) {
		cout << "spread_table_with_selection::make_spread_table "
				"nb_spreads1 != total_nb_of_spreads" << endl;
		exit(1);
	}

	FREE_OBJECTS(SetOrb);

#if 0
	if (f_v) {
		cout << "spread_table_with_selection::make_spread_table before "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}
	tmp_isomorphism_type_of_spread = isomorphism_type_of_spread;
		// for packing_swap_func
	Sorting.Heapsort_general(Sets, total_nb_of_spreads,
			packing_spread_compare_func,
			packing_swap_func,
			this);
	if (f_v) {
		cout << "spread_table_with_selection::make_spread_table after "
				"sorting spread table of size " << total_nb_of_spreads << endl;
	}
#endif


	if (false) {
		cout << "spread_table_with_selection::make_spread_table "
				"The labeled spreads are:" << endl;
		for (i = 0; i < total_nb_of_spreads; i++) {
			cout << i << " : ";
			Lint_vec_print(cout, Sets[i], spread_size /* + 1*/);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "spread_table_with_selection::make_spread_table done" << endl;
	}
}

void spread_table_with_selection::compute_covered_points(
	long int *&points_covered_by_starter,
	int &nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
// points_covered_by_starter are the lines that
// are contained in the spreads chosen for the starter
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a, s;

	if (f_v) {
		cout << "spread_table_with_selection::compute_covered_points" << endl;
	}
	points_covered_by_starter = NEW_lint(starter_size * spread_size);
	for (i = 0; i < starter_size; i++) {
		s = starter[i];
		for (j = 0; j < spread_size; j++) {
			a = Spread_table->Spread_table[s * spread_size + j];
			points_covered_by_starter[i * spread_size + j] = a;
		}
	}
#if 0
	cout << "covered lines:" << endl;
	int_vec_print(cout, covered_lines, starter_size * spread_size);
	cout << endl;
#endif
	if (f_v) {
		cout << "spread_table_with_selection::compute_covered_points done" << endl;
	}
}

void spread_table_with_selection::compute_free_points2(
	long int *&free_points2,
	int &nb_free_points2, long int *&free_point_idx,
	long int *points_covered_by_starter,
	int nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
// free_points2 are actually the free lines,
// i.e., the lines that are not
// yet part of the partial packing
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "spread_table_with_selection::compute_free_points2" << endl;
	}
	free_point_idx = NEW_lint(nb_lines);
	free_points2 = NEW_lint(nb_lines);
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
		cout << "spread_table_with_selection::compute_free_points2 done" << endl;
	}
}

void spread_table_with_selection::compute_live_blocks2(
		layer4_classification::solvers_package::exact_cover *EC,
		int starter_case,
	long int *&live_blocks2, int &nb_live_blocks2,
	long int *points_covered_by_starter,
	int nb_points_covered_by_starter,
	long int *starter, int starter_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "spread_table_with_selection::compute_live_blocks2" << endl;
	}
	live_blocks2 = NEW_lint(Spread_table->nb_spreads);
	nb_live_blocks2 = 0;
	for (i = 0; i < Spread_table->nb_spreads; i++) {
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
		cout << "spread_table_with_selection::compute_live_blocks2 done" << endl;
	}

	if (f_v) {
		cout << "spread_table_with_selection::compute_live_blocks2 STARTER_CASE "
			<< starter_case << " / " << EC->starter_nb_cases
			<< " : Found " << nb_live_blocks2 << " live spreads" << endl;
	}
}

void spread_table_with_selection::compute_adjacency_matrix(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_table_with_selection::compute_adjacency_matrix" << endl;
	}

	Spread_table->compute_adjacency_matrix(
			Bitvec,
			verbose_level);


	if (f_v) {
		cout << "spread_table_with_selection::compute_adjacency_matrix done" << endl;
	}
}



int spread_table_with_selection::is_adjacent(
		int i, int j)
{
	int k;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (i == j) {
		return false;
	}
#if 1
	if (Bitvec) {
		k = Combi.ij2k(i, j, Spread_table->nb_spreads);
		if (Bitvec->s_i(k)) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		if (Spread_table->test_if_spreads_are_line_disjoint(i, j)) {
			return true;
		}
		else {
			return false;
		}
	}
#else
	if (Spread_tables->test_if_spreads_are_line_disjoint(i, j)) {
		return true;
	}
	else {
		return false;
	}
#endif
}


// #############################################################################
// global functions:
// #############################################################################


static int spread_table_with_selection_compare_func(
		void *data, int i, int j, void *extra_data)
{
	spread_table_with_selection *S = (spread_table_with_selection *) extra_data;
	long int **Sets = (long int **) data;
	int ret;
	other::data_structures::sorting Sorting;

	ret = Sorting.lint_vec_compare(Sets[i], Sets[j], S->spread_size);
	return ret;
}

static void spread_table_with_selection_swap_func(
		void *data, int i, int j, void *extra_data)
{
	spread_table_with_selection *S = (spread_table_with_selection *) extra_data;
	int *d = S->tmp_isomorphism_type_of_spread;
	long int **Sets = (long int **) data;
	long int *p;
	int a;

	p = Sets[i];
	Sets[i] = Sets[j];
	Sets[j] = p;

	a = d[i];
	d[i] = d[j];
	d[j] = a;
}




}}}

