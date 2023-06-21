/*
 * spread_tables.cpp
 *
 *  Created on: Feb 24, 2019
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


static int spread_table_compare_func(void *a, void *b, void *data);


spread_tables::spread_tables()
{
	q = 0;
	d = 4; // = 4
	F = NULL;
	P = NULL; // PG(3,q)
	Gr = NULL; // Gr_{4,2}
	nb_lines = 0;
	spread_size = 0;
	nb_iso_types_of_spreads = 0;

	//std::string prefix;

	//std::string fname_dual_line_idx;
	//std::string fname_self_dual_lines;
	//std::string fname_spreads;
	//std::string fname_isomorphism_type_of_spreads;
	//std::string fname_dual_spread;
	//std::string fname_self_dual_spreads;
	//std::string fname_schreier_table;


	dual_line_idx = NULL;
	self_dual_lines = NULL;
	nb_self_dual_lines = 0;

	nb_spreads = 0;
	spread_table = NULL;
	spread_iso_type = NULL;
	dual_spread_idx = NULL;
	self_dual_spreads = NULL;
	nb_self_dual_spreads = 0;

	schreier_table = NULL;

}

spread_tables::~spread_tables()
{
#if 0
	if (P) {
		FREE_OBJECT(P);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
#endif
	if (dual_line_idx) {
		FREE_int(dual_line_idx);
	}
	if (self_dual_lines) {
		FREE_int(self_dual_lines);
	}
	if (spread_table) {
		FREE_lint(spread_table);
	}
	if (spread_iso_type) {
		FREE_int(spread_iso_type);
	}
	if (dual_spread_idx) {
		FREE_lint(dual_spread_idx);
	}
	if (schreier_table) {
		FREE_int(schreier_table);
	}
}

void spread_tables::init(projective_space *P,
		int f_load,
		int nb_iso_types_of_spreads,
		std::string &path_to_spread_tables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "spread_tables::init" << endl;
	}

	if (P->Subspaces->n != 3) {
		cout << "spread_tables::init P->Subspaces->n != 3" << endl;
		exit(1);
	}
	spread_tables::P = P;
	spread_tables::F = P->Subspaces->F;
	Gr = P->Subspaces->Grass_lines;
	q = F->q;
	d = 4;

	nb_lines = Gr->nCkq->as_int();
	spread_size = q * q + 1;
	spread_tables::nb_iso_types_of_spreads = nb_iso_types_of_spreads;

	if (f_v) {
		cout << "spread_tables::init nb_lines=" << nb_lines << endl;
		cout << "spread_tables::init spread_size=" << spread_size << endl;
		cout << "spread_tables::init nb_iso_types_of_spreads="
				<< nb_iso_types_of_spreads << endl;
	}


	prefix = path_to_spread_tables + "spread_" + std::to_string(NT.i_power_j(q, 2));

	if (f_v) {
		cout << "spread_tables::init prefix=" << spread_tables::prefix << endl;
	}


	create_file_names(verbose_level);

	if (f_v) {
		cout << "spread_tables::init before Gr->compute_dual_line_idx" << endl;
	}
	Gr->compute_dual_line_idx(dual_line_idx,
			self_dual_lines, nb_self_dual_lines,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "spread_tables::init after Gr->compute_dual_line_idx" << endl;
	}

	if (f_load) {
		if (f_v) {
			cout << "spread_tables::init before load" << endl;
		}
		load(verbose_level);
		if (f_v) {
			cout << "spread_tables::init after load" << endl;
		}
	}


	if (f_v) {
		cout << "spread_tables::init done" << endl;
	}
}

void spread_tables::create_file_names(int verbose_level)
{
	fname_dual_line_idx.assign(prefix);
	fname_dual_line_idx.append("_dual_line_idx.csv");

	fname_self_dual_lines.assign(prefix);
	fname_self_dual_lines.append("_self_dual_line_idx.csv");

	fname_spreads.assign(prefix);
	fname_spreads.append("_spreads.csv");

	fname_isomorphism_type_of_spreads.assign(prefix);
	fname_isomorphism_type_of_spreads.append("_spreads_iso.csv");

	fname_dual_spread.assign(prefix);
	fname_dual_spread.append("_dual_spread_idx.csv");

	fname_self_dual_spreads.assign(prefix);
	fname_self_dual_spreads.append("_self_dual_spreads.csv");

	fname_schreier_table.assign(prefix);
	fname_schreier_table.append("_schreier_table.csv");

}
void spread_tables::init_spread_table(int nb_spreads,
		long int *spread_table, int *spread_iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::init_spread_table" << endl;
	}
	spread_tables::nb_spreads = nb_spreads;
	spread_tables::spread_table = spread_table;
	spread_tables::spread_iso_type = spread_iso_type;
	if (f_v) {
		cout << "spread_tables::init_spread_table done" << endl;
	}
}

void spread_tables::init_tables(int nb_spreads,
		long int *spread_table, int *spread_iso_type,
		long int *dual_spread_idx,
		long int *self_dual_spreads, int nb_self_dual_spreads,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::init_tables" << endl;
	}
	spread_tables::nb_spreads = nb_spreads;
	spread_tables::spread_table = spread_table;
	spread_tables::spread_iso_type = spread_iso_type;
	spread_tables::dual_spread_idx = dual_spread_idx;
	spread_tables::self_dual_spreads = self_dual_spreads;
	spread_tables::nb_self_dual_spreads = nb_self_dual_spreads;
	if (f_v) {
		cout << "spread_tables::init_tables done" << endl;
	}
}

void spread_tables::init_schreier_table(int *schreier_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::init_schreier_table" << endl;
	}
	spread_tables::schreier_table = schreier_table;
	if (f_v) {
		cout << "spread_tables::init_schreier_table done" << endl;
	}
}

void spread_tables::init_reduced(
		int nb_select, int *select,
		spread_tables *old_spread_table,
		std::string &path_to_spread_tables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "spread_tables::init_reduced, nb_select=" << nb_select << endl;
	}

	P = old_spread_table->P;
	F = P->Subspaces->F;
	Gr = P->Subspaces->Grass_lines;
	q = F->q;
	d = 4; // = 4


	prefix = path_to_spread_tables + "reduced_spread_" + std::to_string(NT.i_power_j(q, 2));

	create_file_names(verbose_level);


	nb_lines = Gr->nCkq->as_int();
	spread_size = old_spread_table->spread_size;
	nb_iso_types_of_spreads = old_spread_table->nb_iso_types_of_spreads;


	nb_spreads = nb_select;
	if (f_v) {
		cout << "spread_tables::init_reduced allocating spread_table" << endl;
	}
	spread_table = NEW_lint(nb_spreads * spread_size);
	if (f_v) {
		cout << "spread_tables::init_reduced allocating spread_iso_type" << endl;
	}
	spread_iso_type = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		a = select[i];
		Lint_vec_copy(old_spread_table->spread_table + a * spread_size,
				spread_table + i * spread_size, spread_size);
		spread_iso_type[i] = old_spread_table->spread_iso_type[a];
	}

	data_structures::sorting Sorting;
	int *select_sorted;
	int *select_original_idx;

	select_sorted = NEW_int(nb_select);
	select_original_idx = NEW_int(nb_select);
	Int_vec_copy(select, select_sorted, nb_select);

	if (f_v) {
		cout << "spread_tables::init_reduced "
				"sorting good spreads" << endl;
	}
	Sorting.int_vec_heapsort_with_log(select_sorted, select_original_idx, nb_select);
	for (i = 0; i < nb_spreads; i++) {
		select_original_idx[i] = i;
	}
	if (f_v) {
		cout << "spread_tables::init_reduced "
				"sorting good spreads done" << endl;
	}
	int idx;

	if (f_v) {
		cout << "spread_tables::init_reduced computing dual_spread_idx" << endl;
	}
	dual_spread_idx = NEW_lint(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		a = select[i];
		if (!Sorting.int_vec_search(select_sorted, nb_spreads, old_spread_table->dual_spread_idx[a], idx)) {
			//cout << "spread_tables::init_reduced unable to find the dual spread in the selection" << endl;
			//exit(1);
			idx = -1;
		}
		else {
			idx = select_original_idx[idx];
		}
		dual_spread_idx[i] = idx;
	}
	if (f_v) {
		cout << "spread_tables::init_reduced computing dual_spread_idx done" << endl;
	}

	FREE_int(select_sorted);
	FREE_int(select_original_idx);

	if (f_v) {
		cout << "spread_tables::init_reduced computing self_dual_spreads" << endl;
	}
	nb_self_dual_spreads = 0;
	self_dual_spreads = NEW_lint(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		idx = dual_spread_idx[i];
		if (idx == i) {
			self_dual_spreads[nb_self_dual_spreads++] = i;
		}
	}
	if (f_v) {
		cout << "spread_tables::init_reduced computing self_dual_spreads done" << endl;
	}



	if (f_v) {
		cout << "spread_tables::init_reduced done" << endl;
	}
}

long int *spread_tables::get_spread(int spread_idx)
{
	return spread_table + spread_idx * spread_size;
}

void spread_tables::find_spreads_containing_two_lines(std::vector<int> &v,
		int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::find_spreads_containing_two_lines" << endl;
		cout << "spread_tables::find_spreads_containing_two_lines line1 = " << line1 << endl;
		cout << "spread_tables::find_spreads_containing_two_lines line2 = " << line2 << endl;
	}
	int spread_idx;
	long int *S;
	int i;
	int f_found_line1;
	int f_found_line2;

	for (spread_idx = 0; spread_idx < nb_spreads; spread_idx++) {
		S = get_spread(spread_idx);
		f_found_line1 = false;
		for (i = 0; i < spread_size; i++) {
			if (S[i] == line1) {
				f_found_line1 = true;
				break;
			}
		}
		f_found_line2 = false;
		for (i = 0; i < spread_size; i++) {
			if (S[i] == line2) {
				f_found_line2 = true;
				break;
			}
		}
		if (f_found_line1 && f_found_line2) {
			v.push_back(spread_idx);
		}
	}
	if (f_v) {
		cout << "spread_tables::find_spreads_containing_two_lines done" << endl;
	}
}

void spread_tables::classify_self_dual_spreads(int *&type,
		data_structures::set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "spread_tables::classify_self_dual_spreads" << endl;
	}
	type = NEW_int(nb_iso_types_of_spreads);
	Int_vec_zero(type, nb_iso_types_of_spreads);
	for (i = 0; i < nb_self_dual_spreads; i++) {
		a = spread_iso_type[i];
		type[a]++;
	}
	SoS = NEW_OBJECT(data_structures::set_of_sets);
	SoS->init_basic_with_Sz_in_int(
			nb_self_dual_spreads /* underlying_set_size */,
			nb_iso_types_of_spreads /* nb_sets */,
			type, 0 /* verbose_level */);
	for (a = 0; a < nb_iso_types_of_spreads; a++) {
		SoS->Set_size[a] = 0;
	}
	for (i = 0; i < nb_self_dual_spreads; i++) {
		a = spread_iso_type[i];
		SoS->Sets[a][SoS->Set_size[a]++] = i;
	}

	if (f_v) {
		cout << "spread_tables::classify_self_dual_spreads done" << endl;
	}
}

int spread_tables::files_exist(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spread_tables::files_exist testing whether file exists: " << fname_spreads << endl;
	}
	if (Fio.file_size(fname_spreads) > 0) {
		return true;
	}
	else {
		return false;
	}
}

void spread_tables::save(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spread_tables::save" << endl;
	}

	if (f_v) {
		cout << "spread_tables::save "
				"writing file " << fname_spreads << endl;
	}

	Fio.lint_matrix_write_csv(fname_spreads,
			spread_table, nb_spreads, spread_size);
	if (f_v) {
		cout << "spread_tables::save "
				"written file " << fname_spreads << endl;
	}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_isomorphism_type_of_spreads
				<< endl;
	}
	string label;

	label.assign("isomorphism_type_of_spread");
	Fio.int_vec_write_csv(
			spread_iso_type, nb_spreads,
			fname_isomorphism_type_of_spreads,
			label);
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_isomorphism_type_of_spreads
				<< endl;
	}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_dual_spread
				<< endl;
	}
	label.assign("dual_spread_idx");
	Fio.lint_vec_write_csv(
			dual_spread_idx, nb_spreads,
			fname_dual_spread,
			label);
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_dual_spread
				<< endl;
	}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_self_dual_spreads
				<< endl;
	}
	label.assign("self_dual_spreads");
	Fio.lint_vec_write_csv(
			self_dual_spreads, nb_self_dual_spreads,
			fname_self_dual_spreads,
			label);
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_self_dual_spreads
				<< endl;
	}



	if (schreier_table) {
		if (f_v) {
			cout << "spread_tables::save "
					"writing file " << fname_schreier_table << endl;
		}

		Fio.int_matrix_write_csv(fname_schreier_table,
				schreier_table, nb_spreads, 4);
		if (f_v) {
			cout << "spread_tables::save "
					"written file " << fname_schreier_table << endl;
		}
	}
	else {
		cout << "spread_tables::save no schreier table" << endl;
	}


	if (f_v) {
		cout << "spread_tables::save done" << endl;
	}
}

void spread_tables::load(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spread_tables::load" << endl;
	}

	if (f_v) {
		cout << "spread_tables::load "
				"reading file " << fname_spreads << endl;
	}

	Fio.lint_matrix_read_csv(fname_spreads,
			spread_table, nb_spreads, b,
			0 /* verbose_level */);
	if (b != spread_size) {
		cout << "spread_tables::load b != spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load "
				"read file " << fname_spreads << endl;
	}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_isomorphism_type_of_spreads
				<< endl;
	}
	Fio.int_matrix_read_csv(fname_isomorphism_type_of_spreads,
			spread_iso_type, a, b,
			0 /* verbose_level */);
	if (a != nb_spreads) {
		cout << "spread_tables::load a != nb_spreads" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_isomorphism_type_of_spreads
				<< endl;
	}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_dual_spread
				<< endl;
	}
	Fio.lint_matrix_read_csv(fname_dual_spread,
			dual_spread_idx, a, b,
			0 /* verbose_level */);
	if (a != nb_spreads) {
		cout << "spread_tables::load a != nb_spreads" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_dual_spread
				<< endl;
	}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_self_dual_spreads
				<< endl;
	}
	Fio.lint_matrix_read_csv(fname_self_dual_spreads,
			self_dual_spreads, nb_self_dual_spreads, b,
			0 /* verbose_level */);
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_self_dual_spreads
				<< endl;
	}


	if (Fio.file_size(fname_schreier_table) > 0) {
		if (f_v) {
			cout << "spread_tables::load, "
					"reading file " << fname_schreier_table
					<< endl;
		}
		Fio.int_matrix_read_csv(fname_schreier_table,
				schreier_table, a, b,
				0 /* verbose_level */);
		if (a != nb_spreads) {
			cout << "spread_tables::load a != nb_spreads" << endl;
			exit(1);
		}
		if (b != 4) {
			cout << "spread_tables::load b != 4" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "spread_tables::load, "
					"read file " << fname_schreier_table
					<< endl;
		}
	}
	else {
		cout << "spread_tables::load no schreier tables" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "spread_tables::load done" << endl;
	}
}


void spread_tables::compute_adjacency_matrix(
		data_structures::bitvector *&Bitvec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, N2; //, cnt;

	if (f_v) {
		cout << "spread_tables::compute_adjacency_matrix" << endl;
	}

	N2 = ((long int) nb_spreads * (long int) nb_spreads) >> 1;

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(N2);

	k = 0;
	//cnt = 0;
	for (i = 0; i < nb_spreads; i++) {

		for (j = i + 1; j < nb_spreads; j++) {


			if (test_if_spreads_are_disjoint(i, j)) {
				Bitvec->m_i(k, 1);
				//cnt++;
			}
			else {
				Bitvec->m_i(k, 0);
			}

			k++;
			if ((k & ((1 << 21) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k << " / " << N2 << endl;
			}
		}
	}





	{
		graph_theory::colored_graph *CG;
		std::string fname;
		orbiter_kernel_system::file_io Fio;
		string label;
		string label_tex;

		CG = NEW_OBJECT(graph_theory::colored_graph);
		int *color;

		label.assign(prefix);
		label_tex.assign(prefix);

		color = NEW_int(nb_spreads);
		Int_vec_zero(color, nb_spreads);

		CG->init(nb_spreads, 1, 1,
				color, Bitvec,
				false,
				label, label_tex,
				verbose_level);

		fname.assign(prefix);
		fname.append("_disjoint_spreads.colored_graph");

		CG->save(fname, verbose_level);

		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_int(color);
		FREE_OBJECT(CG);
	}


	if (f_v) {
		cout << "spread_tables::compute_adjacency_matrix done" << endl;
	}
}

int spread_tables::test_if_spreads_are_disjoint(int a, int b)
{
	long int *p, *q;
	data_structures::sorting Sorting;
	int ret;

	p = spread_table + a * spread_size;
	q = spread_table + b * spread_size;
	ret = Sorting.test_if_sets_are_disjoint(p, q, spread_size, spread_size);
	return ret;
}

void spread_tables::compute_dual_spreads(long int **Sets,
		long int *&Dual_spread_idx,
		long int *&self_dual_spread_idx,
		int &nb_self_dual_spreads,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *dual_spread;
	int i, j;
	long int a, b;
	int idx;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "spread_tables::compute_dual_spreads" << endl;
	}

	dual_spread = NEW_lint(spread_size);
	Dual_spread_idx = NEW_lint(nb_spreads);
	self_dual_spread_idx = NEW_lint(nb_spreads);

	nb_self_dual_spreads = 0;

	for (i = 0; i < nb_spreads; i++) {

		for (j = 0; j < spread_size; j++) {
			a = spread_table[i * spread_size + j];
			b = dual_line_idx[a];
			dual_spread[j] = b;
		}

		if (false) {
			cout << "spread_tables::compute_dual_spreads spread "
					<< i << " / " << nb_spreads << endl;
			Lint_vec_print(cout,
					spread_table + i * spread_size, spread_size);
			cout << endl;
			Lint_vec_print(cout, dual_spread, spread_size);
			cout << endl;
		}
		Sorting.lint_vec_heapsort(dual_spread, spread_size);
		//dual_spread[0] = int_vec_hash(dual_spread + 1, spread_size);
		if (false) {
			Lint_vec_print(cout, dual_spread, spread_size);
			cout << endl;
		}

		long int v[1];

		v[0] = spread_size /*+ 1*/;

		if (Sorting.vec_search((void **)Sets,
				spread_table_compare_func, (void *) v,
				nb_spreads, dual_spread, idx,
				0 /* verbose_level */)) {
			if (false) {
				cout << "spread_tables::compute_dual_spreads Dual "
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
			Lint_vec_print(cout, dual_spread, spread_size);
			cout << endl;
			exit(1);
		}
	}

	FREE_lint(dual_spread);
	if (f_v) {
		cout << "spread_tables::compute_dual_spreads we found "
				<< nb_self_dual_spreads << " self dual spreads" << endl;
		cout << "They are: ";
		Lint_vec_print(cout, self_dual_spread_idx, nb_self_dual_spreads);
		cout << endl;
	}


	if (f_v) {
		cout << "spread_tables::compute_dual_spreads done" << endl;
	}

}

int spread_tables::test_if_pair_of_sets_are_adjacent(
		long int *set1, int sz1,
		long int *set2, int sz2,
		int verbose_level)
{
	int f_v = false; // (verbose_level >= 1);
	long int s1, s2;
	int i, j;

	if (f_v) {
		cout << "spread_tables::test_if_pair_of_sets_are_adjacent" << endl;
	}
	for (i = 0; i < sz1; i++) {
		s1 = set1[i];
		for (j = 0; j < sz2; j++) {
			s2 = set2[j];
			if (!test_if_spreads_are_disjoint(s1, s2)) {
				return false;
			}
		}
	}
	return true;
}

int spread_tables::test_if_set_of_spreads_is_line_disjoint(
		long int *set, int len)
{
	int i, j, ret;
	long int a, b;

	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = i + 1; j < len; j++) {
			b = set[j];
			if (!test_if_spreads_are_disjoint(a, b)) {
				break;
			}
		}
		if (j < len) {
			break;
		}
	}
	if (i < len) {
		//cout << "is NOT a partial packing" << endl;
		ret = false;
	}
	else {
		ret = true;
		//cout << "IS a partial packing" << endl;
	}
	return ret;

}

int spread_tables::test_if_set_of_spreads_is_line_disjoint_and_complain_if_not(
		long int *set, int len)
{
	int i, j, ret;
	long int a, b;

	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = i + 1; j < len; j++) {
			b = set[j];
			if (!test_if_spreads_are_disjoint(a, b)) {
				cout << "elements i=" << i << " j=" << j << " corresponding to spreads " << a << " and " << b << " are not line disjoint" << endl;
				exit(1);
			}
		}
		if (j < len) {
			break;
		}
	}
	if (i < len) {
		//cout << "is NOT a partial packing" << endl;
		ret = false;
	}
	else {
		ret = true;
		//cout << "IS a partial packing" << endl;
	}
	return ret;

}

void spread_tables::make_exact_cover_problem(
		solvers::diophant *&Dio,
		long int *live_point_index, int nb_live_points,
		long int *live_blocks, int nb_live_blocks,
		int nb_needed,
		int verbose_level)
// points are actually lines and lines are actually spreads
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::make_exact_cover_problem" << endl;
	}

	int s, u, i, j, a;
	int nb_rows = nb_live_points;
	int nb_cols = nb_live_blocks;

	Dio = NEW_OBJECT(solvers::diophant);
	Dio->open(nb_rows, nb_cols, verbose_level - 1);
	Dio->f_has_sum = true;
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		Dio->RHS_low[i] = 1;
	}

	Dio->fill_coefficient_matrix_with(0);


	for (j = 0; j < nb_cols; j++) {
		s = live_blocks[j];
		for (a = 0; a < spread_size; a++) {
			i = spread_table[s * spread_size + a];
			u = live_point_index[i];
			if (u == -1) {
				cout << "spread_tables::make_exact_cover_problem "
						"live_point_index[i] == -1" << endl;
				exit(1);
			}
			Dio->Aij(u, j) = 1;
		}
	}
	for (j = 0; j < nb_cols; j++) {
		Dio->x_max[j] = 1;
		Dio->x_min[j] = 0;
	}


	if (f_v) {
		cout << "spread_tables::make_exact_cover_problem done" << endl;
	}
}

void spread_tables::compute_list_of_lines_from_packing(
		long int *list_of_lines,
		long int *packing, int sz_of_packing,
		int verbose_level)
// list_of_lines[sz_of_packing * spread_size]
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "spread_tables::compute_list_of_lines_from_packing" << endl;
	}
	for (i = 0; i < sz_of_packing; i++) {
		a = packing[i];
		Lint_vec_copy(spread_table + a * spread_size,
				list_of_lines + i * spread_size, spread_size);
	}
	if (f_v) {
		cout << "spread_tables::compute_list_of_lines_from_packing done" << endl;
	}
}

void spread_tables::compute_iso_type_invariant(
		int *Partial_packings, int nb_pp, int sz,
		int *&Iso_type_invariant,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;

	if (f_v) {
		cout << "spread_tables::compute_iso_type_invariant" << endl;
	}

	Iso_type_invariant = NEW_int(nb_pp * nb_iso_types_of_spreads);
	Int_vec_zero(Iso_type_invariant, nb_pp * nb_iso_types_of_spreads);
	for (i = 0; i < nb_pp; i++) {
		for (j = 0; j < sz; j++) {
			a = Partial_packings[i * sz + j];
			b = spread_iso_type[a];
			Iso_type_invariant[i * nb_iso_types_of_spreads + b]++;
		}
	}

	if (f_v) {
		cout << "spread_tables::compute_iso_type_invariant done" << endl;
	}
}

void spread_tables::report_one_spread(std::ostream &ost, int a)
{
	long int *p;
	long int b;
	int i;

	p = spread_table + a * spread_size;
	for (i = 0; i < spread_size; i++) {
		ost << "$";
		b = p[i];
		Gr->print_single_generator_matrix_tex(ost, b);
		ost << "_{" << b << "}";
		ost << "$";
		if (i < spread_size - 1) {
			ost << ", ";
		}
	}
}


static int spread_table_compare_func(void *a, void *b, void *data)
// used in spread_table
{
	int *A = (int *)a;
	int *B = (int *)b;
	int *p = (int *) data;
	int n = *p;
	int i;

	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return 1;
		}
		if (A[i] > B[i]) {
			return -1;
		}
	}
	return 0;
}




}}}


