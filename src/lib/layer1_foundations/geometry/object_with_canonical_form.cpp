// object_with_canonical_form.cpp
// 
// Anton Betten
//
// December 23, 2017
//
//
// 
//
//

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {


object_with_canonical_form::object_with_canonical_form()
{
	P = NULL;
	type = t_PTS;
	//input_fname = NULL;
	input_idx = 0;
	f_has_known_ago = FALSE;
	known_ago = 0;
	//set_as_string = NULL;

	set = NULL;
	sz = 0;

	set2 = NULL;
	sz2 = 0;

	v = 0;
	b = 0;

	f_partition = FALSE;
	partition = NULL;

	design_k = 0;
	design_sz = 0;
	SoS = NULL;
	C = NULL;
}

object_with_canonical_form::~object_with_canonical_form()
{
	if (set) {
		FREE_lint(set);
	}
	if (set2) {
		FREE_lint(set2);
	}
#if 0
	if (partition) {
		FREE_int(partition);
	}
#endif
	if (SoS) {
		FREE_OBJECT(SoS);
	}
	if (C) {
		FREE_OBJECT(C);
	}
}

void object_with_canonical_form::print(ostream &ost)
{

	cout << "set_as_string: " << set_as_string << endl;
	if (type == t_PTS) {
		ost << "set of points of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << endl;
	}
	else if (type == t_LNS) {
		ost << "set of lines of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << endl;
	}
	else if (type == t_PNL) {
		ost << "set of points of size " << sz
				<< " and a set of lines of size " << sz2 << ": ";
		Lint_vec_print(ost, set, sz);
		ost << ", ";
		Lint_vec_print(ost, set2, sz2);
		ost << endl;
	}
	else if (type == t_PAC) {
		ost << "packing:" << endl;
		SoS->print_table_tex(ost);
		ost << endl;
	}
	else if (type == t_INC) {
		ost << "incidence structure:" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;
	}
	else if (type == t_LS) {
		ost << "large set:" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;
	}
}

void object_with_canonical_form::print_rows(std::ostream &ost,
		int f_show_incma, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::print_rows" << endl;
	}

	//print_tex(ost);

	if (f_show_incma) {

		combinatorics::encoded_combinatorial_object *Enc;

		encode_incma(Enc, verbose_level);

		//Enc->latex_set_system_by_columns(ost, verbose_level);

		Enc->latex_set_system_by_rows(ost, verbose_level);

		//Enc->latex_incma(ost, verbose_level);

		FREE_OBJECT(Enc);
	}

	if (f_v) {
		cout << "object_with_canonical_form::print_rows done" << endl;
	}
}

void object_with_canonical_form::print_tex_detailed(std::ostream &ost,
		int f_show_incma, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::print_tex_detailed" << endl;
	}

	if (f_v) {
		cout << "object_with_canonical_form::print_tex_detailed before print_tex" << endl;
	}
	print_tex(ost, verbose_level);
	if (f_v) {
		cout << "object_with_canonical_form::print_tex_detailed after print_tex" << endl;
	}

	if (f_show_incma) {

		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed f_show_incma" << endl;
		}

		combinatorics::encoded_combinatorial_object *Enc;

		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed before encode_incma" << endl;
		}
		encode_incma(Enc, verbose_level);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed after encode_incma" << endl;
		}

		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed before Enc->latex_set_system_by_columns" << endl;
		}
		Enc->latex_set_system_by_columns(ost, verbose_level);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed after Enc->latex_set_system_by_columns" << endl;
		}

		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed before Enc->latex_set_system_by_rows" << endl;
		}
		Enc->latex_set_system_by_rows(ost, verbose_level);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed after Enc->latex_set_system_by_rows" << endl;
		}

		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed before Enc->latex_incma" << endl;
		}
		Enc->latex_incma(ost, verbose_level);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed after Enc->latex_incma" << endl;
		}
		ost << "\\\\" << endl;

		FREE_OBJECT(Enc);
	}

	if (f_v) {
		cout << "object_with_canonical_form::print_tex_detailed done" << endl;
	}
}

void object_with_canonical_form::print_tex(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::print_tex" << endl;
	}

	if (type == t_PTS) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_PTS" << endl;
		}
		ost << "set of points of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
		//P->print_set_numerical(ost, set, sz);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex before P->Reporting->print_set_of_points" << endl;
		}
		P->Reporting->print_set_of_points(ost, set, sz);
		if (f_v) {
			cout << "object_with_canonical_form::print_tex after P->Reporting->print_set_of_points" << endl;
		}
	}
	else if (type == t_LNS) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_LNS" << endl;
		}
		ost << "set of lines of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
	}
	else if (type == t_PNL) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_PNL" << endl;
		}
		ost << "set of points of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
		ost << "and a set of lines of size " << sz2 << ": ";
		Lint_vec_print(ost, set2, sz2);
		ost << "\\\\" << endl;
	}
	else if (type == t_PAC) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_PAC" << endl;
		}
		ost << "packing: \\\\" << endl;
		SoS->print_table_tex(ost);
		ost << endl;
	}
	else if (type == t_INC) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_INC" << endl;
		}
		ost << "incidence structure: \\\\" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
#if 0
		object_with_canonical_form::set = NEW_lint(data_sz);
		Orbiter->Lint_vec.copy(data, object_with_canonical_form::set, data_sz);
		object_with_canonical_form::sz = data_sz;
		object_with_canonical_form::v = v;
		object_with_canonical_form::b = b;
#endif
	}
	else if (type == t_LS) {
		if (f_v) {
			cout << "object_with_canonical_form::print_tex t_LS" << endl;
		}
		ost << "large set: \\\\" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;

		int nb_designs = b / design_sz;
		int i;

		for (i = 0; i < nb_designs; i++) {
			Lint_vec_print(ost, set + i * design_sz, design_sz);
			ost << "\\\\" << endl;
		}
#if 0
		object_with_canonical_form::set = NEW_lint(data_sz);
		Orbiter->Lint_vec.copy(data, object_with_canonical_form::set, data_sz);
		object_with_canonical_form::sz = data_sz;
		object_with_canonical_form::v = v;
		object_with_canonical_form::b = data_sz;
		object_with_canonical_form::design_k = k;
		object_with_canonical_form::design_sz = design_sz;
#endif

	}

	if (f_v) {
		cout << "object_with_canonical_form::print_tex done" << endl;
	}
}

void object_with_canonical_form::get_packing_as_set_system(long int *&Sets,
		int &nb_sets, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "object_with_canonical_form::get_packing_as_set_system" << endl;
	}
	nb_sets = SoS->nb_sets;
	set_size = SoS->Set_size[0];
	Sets = NEW_lint(nb_sets * set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < set_size; j++) {
			Sets[i * set_size + j] = SoS->Sets[i][j];
		}
	}
	if (f_v) {
		cout << "object_with_canonical_form::get_packing_as_set_system done" << endl;
	}
}


void object_with_canonical_form::init_point_set(
		long int *set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_point_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PTS;
	object_with_canonical_form::set = NEW_lint(sz);
	Lint_vec_copy(set, object_with_canonical_form::set, sz);
	object_with_canonical_form::sz = sz;
	if (f_v) {
		cout << "object_with_canonical_form::init_point_set done" << endl;
	}
}

void object_with_canonical_form::init_point_set_from_string(
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_point_set_from_string" << endl;
	}

	type = t_PTS;

	Get_lint_vector_from_label(set_text, set, sz, verbose_level);

	if (f_v) {
		cout << "object_with_canonical_form::init_point_set_from_string done" << endl;
	}
}


void object_with_canonical_form::init_line_set(
		long int *set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_line_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_LNS;
	object_with_canonical_form::set = NEW_lint(sz);
	Lint_vec_copy(set, object_with_canonical_form::set, sz);
	object_with_canonical_form::sz = sz;
	if (f_v) {
		cout << "object_with_canonical_form::init_line_set done" << endl;
	}
}

void object_with_canonical_form::init_line_set_from_string(
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_line_set_from_string" << endl;
	}

	type = t_LNS;

	Lint_vec_scan(set_text, set, sz);

	if (f_v) {
		cout << "object_with_canonical_form::init_line_set_from_string done" << endl;
	}
}

void object_with_canonical_form::init_points_and_lines(
	long int *set, int sz,
	long int *set2, int sz2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_points_and_lines" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PNL;

	object_with_canonical_form::set = NEW_lint(sz);
	Lint_vec_copy(set, object_with_canonical_form::set, sz);
	object_with_canonical_form::sz = sz;

	object_with_canonical_form::set2 = NEW_lint(sz2);
	Lint_vec_copy(set2, object_with_canonical_form::set2, sz2);
	object_with_canonical_form::sz2 = sz2;

	if (f_v) {
		cout << "object_with_canonical_form::init_points_and_lines done" << endl;
	}
}

void object_with_canonical_form::init_points_and_lines_from_string(
	std::string &set_text,
	std::string &set2_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_points_and_lines_from_string" << endl;
	}

	type = t_PNL;

	Lint_vec_scan(set_text, set, sz);

	Lint_vec_scan(set2_text, set2, sz2);

	if (f_v) {
		cout << "object_with_canonical_form::init_points_and_lines_from_string done" << endl;
	}
}

void object_with_canonical_form::init_packing_from_set(
		long int *packing, int sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, q, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	q = P->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	if (sz != size_of_packing * size_of_spread) {
		cout << "object_with_canonical_form::init_packing_from_set "
			"sz != size_of_packing * size_of_spread" << endl;
		exit(1);
	}
	SoS = NEW_OBJECT(data_structures::set_of_sets);

	SoS->init_basic_constant_size(P->N_lines, 
		size_of_packing /* nb_sets */, 
		size_of_spread /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		Lint_vec_copy(packing + i * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
#if 0
	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_set it is" << endl;
		SoS->print_table();
	}
#endif
	
	
	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_set done" << endl;
	}
}


void object_with_canonical_form::init_packing_from_string(
		std::string &packing_text,
		int q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_string" << endl;
	}
	type = t_PAC;

	long int *packing;
	int sz;
	int N_lines;



	Lint_vec_scan(packing_text, packing, sz);

	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	N_lines = size_of_spread * size_of_packing;
	if (sz != N_lines) {
		cout << "object_with_canonical_form::init_packing_from_string "
			"sz != N_lines" << endl;
		exit(1);
	}
	SoS = NEW_OBJECT(data_structures::set_of_sets);

	SoS->init_basic_constant_size(N_lines,
		size_of_packing /* nb_sets */,
		size_of_spread /* constant_size */,
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		Lint_vec_copy(packing + i * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
#if 0
	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_string it is" << endl;
		SoS->print_table();
	}
#endif


	FREE_lint(packing);

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_string done" << endl;
	}
}

void object_with_canonical_form::init_packing_from_set_of_sets(
		data_structures::set_of_sets *SoS, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_set_of_sets" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	//object_in_projective_space::set = NEW_int(sz);
	//int_vec_copy(set, object_in_projective_space::set, sz);
	//object_in_projective_space::sz = sz;

	object_with_canonical_form::SoS = SoS->copy();

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_set_of_sets done" << endl;
	}
}


void object_with_canonical_form::init_packing_from_spread_table(
	long int *data,
	long int *Spread_table, int nb_spreads, int spread_size,
	int q,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, size_of_spread, size_of_packing;
	int N_lines;

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_spread_table" << endl;
		}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	//q = P->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	if (spread_size != size_of_spread) {
		cout << "object_with_canonical_form::init_packing_from_spread_table "
				"spread_size != size_of_spread" << endl;
		exit(1);
	}
	N_lines = size_of_spread * size_of_packing;

	SoS = NEW_OBJECT(data_structures::set_of_sets);

	SoS->init_basic_constant_size(N_lines,
		size_of_packing /* nb_sets */,
		size_of_spread /* constant_size */,
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		a = data[i];
		Lint_vec_copy(Spread_table + a * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
	if (verbose_level >= 5) {
		cout << "object_with_canonical_form::init_packing_from_spread_table Sos:" << endl;
		SoS->print_table();
	}

	// test if the object is a packing:
	SoS->sort_all(FALSE /*verbose_level*/);
	int *M;
	int j;
	SoS->pairwise_intersection_matrix(M, 0 /*verbose_level*/);
	for (i = 0; i < SoS->nb_sets; i++) {
		for (j = i + 1; j < SoS->nb_sets; j++) {
			if (M[i * SoS->nb_sets + j]) {
				cout << "object_with_canonical_form::init_packing_from_spread_table not a packing, spreads "
						<< i << " and " << j << " meet in "
						<< M[i * SoS->nb_sets + j] << " lines" << endl;
				cout << "object_with_canonical_form::init_packing_from_spread_table Sos:" << endl;
				SoS->print_table();
				exit(1);

			}
		}
	}
	FREE_int(M);

	if (f_v) {
		cout << "object_with_canonical_form::init_packing_from_spread_table done" << endl;
		}
}

void object_with_canonical_form::init_incidence_geometry(
	long int *data, int data_sz, int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry" << endl;
	}
	if (nb_flags != data_sz) {
		cout << "object_with_canonical_form::init_incidence_geometry nb_flags != data_sz" << endl;
	}
	object_with_canonical_form::P = NULL;
	type = t_INC;
	object_with_canonical_form::set = NEW_lint(data_sz);
	Lint_vec_copy(data, object_with_canonical_form::set, data_sz);
	object_with_canonical_form::sz = data_sz;
	object_with_canonical_form::v = v;
	object_with_canonical_form::b = b;
	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry done" << endl;
	}
}

void object_with_canonical_form::init_incidence_geometry_from_vector(
	std::vector<int> &Flags, int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry" << endl;
	}
	if (nb_flags != Flags.size()) {
		cout << "object_with_canonical_form::init_incidence_geometry nb_flags != Flags.size()" << endl;
	}

	object_with_canonical_form::P = NULL;

	type = t_INC;

	object_with_canonical_form::set = NEW_lint(Flags.size());

	int i;

	for (i = 0; i < Flags.size(); i++) {
		set[i] = Flags[i];
	}
	object_with_canonical_form::sz = Flags.size();
	object_with_canonical_form::v = v;
	object_with_canonical_form::b = b;
	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry done" << endl;
	}
}

void object_with_canonical_form::init_incidence_geometry_from_string(
	std::string &data,
	int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string" << endl;
	}
	long int *flags;
	int data_sz;

	Lint_vec_scan(data, flags, data_sz);

	if (nb_flags != data_sz) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string nb_flags != data_sz" << endl;
	}
	object_with_canonical_form::P = NULL;
	type = t_INC;
	object_with_canonical_form::set = NEW_lint(data_sz);
	Lint_vec_copy(flags, object_with_canonical_form::set, data_sz);
	object_with_canonical_form::sz = data_sz;
	object_with_canonical_form::v = v;
	object_with_canonical_form::b = b;

	FREE_lint(flags);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string done" << endl;
	}
}

void object_with_canonical_form::init_incidence_geometry_from_string_of_row_ranks(
	std::string &data,
	int v, int b, int r,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string" << endl;
	}
	long int *row_ranks;
	long int *flags;
	int *row_set;
	int data_sz;
	int nb_flags;
	int i, h, a;
	combinatorics::combinatorics_domain Combi;

	Lint_vec_scan(data, row_ranks, data_sz);

	if (v != data_sz) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string v != data_sz" << endl;
	}

	flags = NEW_lint(v * r);
	row_set = NEW_int(r);
	nb_flags = 0;
	for (i = 0; i < v; i++) {
		Combi.unrank_k_subset(row_ranks[i], row_set, b, r);
		for (h = 0; h < r; h++) {
			a = i * b + row_set[h];
			flags[nb_flags++] = a;
		}

	}

	object_with_canonical_form::P = NULL;
	type = t_INC;
	object_with_canonical_form::set = NEW_lint(nb_flags);
	Lint_vec_copy(flags, object_with_canonical_form::set, nb_flags);
	object_with_canonical_form::sz = nb_flags;
	object_with_canonical_form::v = v;
	object_with_canonical_form::b = b;

	FREE_int(row_set);
	FREE_lint(row_ranks);
	FREE_lint(flags);

	if (f_v) {
		cout << "object_with_canonical_form::init_incidence_geometry_from_string done" << endl;
	}
}


void object_with_canonical_form::init_large_set(
	long int *data, int data_sz, int v, int b, int k, int design_sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_large_set" << endl;
	}

	if (data_sz != b) {
		cout << "object_with_canonical_form::init_large_set data_sz != b" << endl;
		exit(1);
	}
	object_with_canonical_form::P = NULL;
	type = t_LS;
	object_with_canonical_form::set = NEW_lint(data_sz);
	Lint_vec_copy(data, object_with_canonical_form::set, data_sz);
	object_with_canonical_form::sz = data_sz;
	object_with_canonical_form::v = v;
	object_with_canonical_form::b = data_sz;
	object_with_canonical_form::design_k = k;
	object_with_canonical_form::design_sz = design_sz;
	if (f_v) {
		cout << "object_with_canonical_form::init_large_set done" << endl;
	}
}

void object_with_canonical_form::init_large_set_from_string(
	std::string &data_text, int v, int k, int design_sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::init_large_set_from_string" << endl;
	}
	object_with_canonical_form::P = NULL;

	type = t_LS;

	Lint_vec_scan(data_text, set, sz);

	object_with_canonical_form::v = v;
	object_with_canonical_form::b = sz;
	object_with_canonical_form::design_k = k;
	object_with_canonical_form::design_sz = design_sz;
	if (f_v) {
		cout << "object_with_canonical_form::init_large_set_from_string done" << endl;
	}
}


void object_with_canonical_form::encoding_size(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size" << endl;
	}
	if (type == t_PTS) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_point_set" << endl;
		}
		encoding_size_point_set(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_LNS) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_line_set" << endl;
		}
		encoding_size_line_set(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_PNL) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_points_and_lines" << endl;
		}
		encoding_size_points_and_lines(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_PAC) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_packing" << endl;
		}
		encoding_size_packing(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_INC) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_packing" << endl;
		}
		encoding_size_incidence_geometry(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_LS) {

		if (f_v) {
			cout << "object_with_canonical_form::encoding_size "
					"before encoding_size_large_set" << endl;
		}
		encoding_size_large_set(
				nb_rows, nb_cols, verbose_level);

	}
	else {
		cout << "object_with_canonical_form::encoding_size "
				"unknown type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "object_in_projective_space::encoding_size done" << endl;
	}
}

void object_with_canonical_form::encoding_size_point_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_point_set" << endl;
	}


	C = NEW_OBJECT(data_structures::tally);

	C->init_lint(set, sz, TRUE, 0);
	if (C->second_nb_types > 1) {
		cout << "object_with_canonical_form::encoding_size_point_set "
				"The set is a multiset:" << endl;
		C->print(FALSE /*f_backwards*/);
	}


	if (f_v) {
		cout << "The type of the set is:" << endl;
		C->print(FALSE /*f_backwards*/);
		cout << "C->second_nb_types = " << C->second_nb_types << endl;
	}

	int nb_rows0, nb_cols0;

	nb_rows0 = P->N_points;
	nb_cols0 = P->N_lines;

	nb_rows0 += P->N_lines;
	nb_cols0 += P->Nb_subspaces[2];


	nb_rows = nb_rows0 + 1;
	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_point_set "
				"nb_rows=" << nb_rows << endl;
	}
	nb_cols = nb_cols0 + C->second_nb_types;
	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_point_set "
				"nb_cols=" << nb_cols << endl;
	}
	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_point_set "
				"before FREE_OBJECT(C)" << endl;
	}
	FREE_OBJECT(C);
	C = NULL;
	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_point_set "
				"done" << endl;
	}

}

void object_with_canonical_form::encoding_size_line_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_line_set" << endl;
	}


	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + 1;

}

void object_with_canonical_form::encoding_size_points_and_lines(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_points_and_lines" << endl;
	}


	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + 1;

}

void object_with_canonical_form::encoding_size_packing(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_packing" << endl;
	}

	nb_rows = P->N_points + SoS->nb_sets;
	nb_cols = P->N_lines + 1;

}

void object_with_canonical_form::encoding_size_large_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_designs;

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_large_set" << endl;
	}

	nb_designs = b / design_sz;
	if (nb_designs * design_sz != b) {
		cout << "object_with_canonical_form::encoding_size_large_set "
				"design_sz does not divide b" << endl;
		exit(1);
	}

	nb_rows = v + nb_designs;
	nb_cols = b + 1;

}

void object_with_canonical_form::encoding_size_incidence_geometry(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encoding_size_packing" << endl;
	}

	nb_rows = v;
	nb_cols = b;

}

void object_with_canonical_form::canonical_form_given_canonical_labeling(
		int *canonical_labeling,
		data_structures::bitvector *&B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::canonical_form_given_canonical_labeling" << endl;
	}

	combinatorics::encoded_combinatorial_object *Enc;

	encode_incma(Enc, verbose_level - 1);
	if (f_v) {
		cout << "object_with_canonical_form::canonical_form_given_canonical_labeling "
				"after OiP->encode_incma" << endl;
	}

	Enc->canonical_form_given_canonical_labeling(canonical_labeling,
				B,
				verbose_level);


	FREE_OBJECT(Enc);


	if (f_v) {
		cout << "object_with_canonical_form::canonical_form_given_canonical_labeling done" << endl;
	}
}

void object_with_canonical_form::encode_incma(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_incma" << endl;
	}
	if (type == t_PTS) {
		
		encode_point_set(Enc, verbose_level);

	}
	else if (type == t_LNS) {
		
		encode_line_set(Enc, verbose_level);

	}
	else if (type == t_PNL) {

		encode_points_and_lines(Enc, verbose_level);

	}
	else if (type == t_PAC) {
		
		encode_packing(Enc, verbose_level);

	}
	else if (type == t_INC) {

		encode_incidence_geometry(Enc, verbose_level);

	}
	else if (type == t_LS) {

		encode_large_set(Enc, verbose_level);

	}
	else {
		cout << "object_with_canonical_form::encode_incma "
				"unknown type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_incma done" << endl;
	}
}

void object_with_canonical_form::encode_point_set(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_point_set" << endl;
	}
	int i, j;
	int f_vvv = FALSE; // (verbose_level >= 3);
	

	C = NEW_OBJECT(data_structures::tally);

	if (f_v) {
		cout << "object_with_canonical_form::encode_point_set set=";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}
	C->init_lint(set, sz, TRUE, 0);
	if (C->second_nb_types > 1) {
		cout << "object_with_canonical_form::encode_point_set "
				"The set is a multiset:" << endl;
		C->print(FALSE /*f_backwards*/);
	}


	if (f_v) {
		cout << "The type of the set is:" << endl;
		C->print(FALSE /*f_backwards*/);
		cout << "C->second_nb_types = " << C->second_nb_types << endl;
	}


	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->N_points;
	nb_cols0 = P->N_lines;

	if (P->n >= 3) {
		nb_rows0 += P->N_lines;
		nb_cols0 += P->Nb_subspaces[2];
	}

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + C->second_nb_types;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);



	//Enc->incidence_matrix_projective_space_top_left(P, verbose_level);

	if (P->n >= 3) {
		Enc->extended_incidence_matrix_projective_space_top_left(P, verbose_level);
	}
	else {
		Enc->incidence_matrix_projective_space_top_left(P, verbose_level);
	}

	// last columns:
	for (j = 0; j < C->second_nb_types; j++) {

		int h, f2, l2, m, idx, f, l;

		f2 = C->second_type_first[j];
		l2 = C->second_type_len[j];
		m = C->second_data_sorted[f2 + 0];
		if (f_vvv) {
			cout << "j=" << j << " f2=" << f2 << " l2=" << l2
					<< " multiplicity=" << m << endl;
		}
		for (h = 0; h < l2; h++) {
			idx = C->second_sorting_perm_inv[f2 + h];
			f = C->type_first[idx];
			l = C->type_len[idx];
			i = C->data_sorted[f + 0];
			if (f_vvv) {
				cout << "h=" << h << " idx=" << idx << " f=" << f
						<< " l=" << l << " i=" << i << endl;
			}
			if (i > P->N_points) {
				cout << "object_with_canonical_form::encode_point_set i > P->N_points" << endl;
				cout << "i = " << i << endl;
				cout << "P->N_points = " << P->N_points << endl;
				cout << "h=" << h << " idx=" << idx << " f=" << f
						<< " l=" << l << " i=" << i << endl;
				exit(1);
			}
			Enc->set_incidence_ij(i, Enc->nb_cols0 + j);
		}
	}

	if (f_v) {
		cout << "object_with_canonical_form::encode_point_set bottom right entries" << endl;
	}
	// bottom right entries:
	for (j = 0; j < C->second_nb_types; j++) {
		Enc->set_incidence_ij(Enc->nb_rows0, Enc->nb_cols0 + j);
	}

	if (f_v) {
		cout << "object_with_canonical_form::encode_point_set partition" << endl;
	}


	Enc->partition[P->N_points - 1] = 0;
	Enc->partition[nb_rows0 - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;

	Enc->partition[nb_rows + P->N_lines - 1] = 0;
	Enc->partition[nb_rows + Enc->nb_cols0 - 1] = 0;

	for (j = 0; j < C->second_nb_types; j++) {
		Enc->partition[nb_rows + Enc->nb_cols0 + j] = 0;
	}
	if (f_vvv) {
		cout << "object_with_canonical_form::encode_point_set "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_point_set "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_line_set(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_line_set" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);
	
	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->N_points;
	nb_cols0 = P->N_lines;

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + 1;

	//int N;
	
	//N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	Enc->incidence_matrix_projective_space_top_left(P, verbose_level);

	// last rows:
	for (i = 0; i < 1; i++) {
		int h;

		for (h = 0; h < sz; h++) {
			j = set[h];
			Enc->set_incidence_ij(nb_rows0 + i, j);
		}
	}

	// bottom right entry:
	Enc->set_incidence_ij(nb_rows0, nb_cols0);

	Enc->partition[nb_rows0 - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 + 1 - 1] = 0;

	if (f_vvv) {
		cout << "object_with_canonical_form::encode_line_set "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_line_set "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_points_and_lines(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_points_and_lines" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);

	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->N_points;
	nb_cols0 = P->N_lines;

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + 1;

	//int N;

	//N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	Enc->incidence_matrix_projective_space_top_left(P, verbose_level);

	// lines go in the last row:
	for (i = 0; i < 1; i++) {
		int h;

		for (h = 0; h < sz2; h++) {
			j = set2[h];
			Enc->set_incidence_ij(nb_rows0 + i, j);
		}
	}

	// points go in the last column:
	int h;

	for (h = 0; h < sz; h++) {
		i = set[h];
		Enc->set_incidence_ij(i, nb_cols0);
	}

	// bottom right entry:
	Enc->set_incidence_ij(nb_rows0, nb_cols0);

	Enc->partition[nb_rows0 - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 + 1 - 1] = 0;
	if (f_vvv) {
		cout << "object_with_canonical_form::encode_points_and_lines "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_points_and_lines "
				"done" << endl;
	}
}


void object_with_canonical_form::encode_packing(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_packing" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);
	

	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->N_points;
	nb_cols0 = P->N_lines;

	nb_rows = nb_rows0 + SoS->nb_sets;
	nb_cols = nb_cols0 + 1;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	Enc->incidence_matrix_projective_space_top_left(P, verbose_level);

	// last rows:
	for (i = 0; i < SoS->nb_sets; i++) {
		int h;

		for (h = 0; h < SoS->Set_size[i]; h++) {
			j = SoS->Sets[i][h];
			Enc->set_incidence_ij(nb_rows0 + i, j);
		}
	}
	// bottom right entries:
	for (i = 0; i < SoS->nb_sets; i++) {
		Enc->set_incidence_ij(nb_rows0 + i, nb_cols0);
	}

	Enc->partition[nb_rows0 - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 - 1] = 0;
	Enc->partition[nb_rows + nb_cols0 + 1 - 1] = 0;
	if (f_vvv) {
		cout << "object_with_canonical_form::encode_packing "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_packing "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_large_set(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_large_set" << endl;
	}
	int i, j, a, h;
	int f_vvv = (verbose_level >= 3);


	int nb_rows, nb_cols;
	int nb_designs;

	nb_designs = b / design_sz;

	nb_rows = v + nb_designs;
	nb_cols = b + 1;

	//int N;
	//int L;

	//N = nb_rows + nb_cols;
	//L = nb_rows * nb_cols;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);


	combinatorics::combinatorics_domain Combi;

	int *block;

	block = NEW_int(design_k);

	for (j = 0; j < sz; j++) {
		a = set[j];
		Combi.unrank_k_subset(a, block, v, design_k);
		for (h = 0; h < design_k; h++) {
			i = block[h];
			Enc->set_incidence_ij(i, j);
		}
	}

	// last rows:
	for (i = 0; i < nb_designs; i++) {

		for (h = 0; h < design_sz; h++) {
			Enc->set_incidence_ij(v + i, i * design_sz + h);
		}
	}
	// bottom right entries:
	for (i = 0; i < nb_designs; i++) {
		Enc->set_incidence_ij(v + i, b);
	}

	Enc->partition[v - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + b - 1] = 0;
	Enc->partition[nb_rows + b + 1 - 1] = 0;
	if (f_vvv) {
		cout << "object_with_canonical_form::encode_large_set "
				"partition:" << endl;
		Enc->print_partition();
	}

	FREE_int(block);

	if (f_v) {
		cout << "object_with_canonical_form::encode_large_set "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_incidence_geometry(
		combinatorics::encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_incidence_geometry" << endl;
	}
	int i, a;
	int f_vvv = (verbose_level >= 3);


	int nb_rows, nb_cols;
	nb_rows = v;
	nb_cols = b;

	int N;

	N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(combinatorics::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	for (i = 0; i < sz; i++) {
		a = set[i];
		if (a >= nb_rows * nb_cols) {
			cout << "object_with_canonical_form::encode_incidence_geometry a >= nb_rows* nb_cols" << endl;
			cout << "nb_rows = " << nb_rows << endl;
			cout << "nb_cols = " << nb_cols << endl;
			cout << "a = " << a << endl;
			exit(1);
		}
		Enc->set_incidence(a);
	}


	if (f_partition) {
		Int_vec_copy(partition, Enc->partition, N);
	}
	else {
		Enc->partition[nb_rows - 1] = 0;
		Enc->partition[N - 1] = 0;
	}

	if (f_vvv) {
		cout << "object_with_canonical_form::encode_incidence_geometry "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_incidence_geometry "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_incma_and_make_decomposition(
		combinatorics::encoded_combinatorial_object *&Enc,
		incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_incma_and_make_decomposition" << endl;
	}
	if (type == t_PTS) {
		
		encode_point_set(Enc, verbose_level);

	}
	else if (type == t_LNS) {
		
		encode_line_set(Enc, verbose_level);

	}
	else if (type == t_PNL) {

		encode_points_and_lines(Enc, verbose_level);

	}
	else if (type == t_PAC) {
		
		encode_packing(Enc, verbose_level);

	}
	else if (type == t_INC) {

		encode_incidence_geometry(Enc, verbose_level);

	}
	else if (type == t_LS) {

		encode_large_set(Enc, verbose_level);

	}
	else {
		cout << "object_with_canonical_form::encode_incma_and_make_decomposition unknown type" << endl;
		exit(1);
	}

	Inc = NEW_OBJECT(incidence_structure);
	Inc->init_by_matrix(Enc->nb_rows, Enc->nb_cols,
			Enc->get_Incma(), verbose_level - 2);




	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(Enc->nb_rows + Enc->nb_cols, 0);
	Stack->subset_continguous(Inc->nb_points(), Inc->nb_lines());
	Stack->split_cell(0);

	if (type == t_PTS) {
		
		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_PTS split1" << endl;
		}
		Stack->subset_continguous(
				Inc->nb_points() + P->N_lines,
				Enc->nb_cols - P->N_lines);
		Stack->split_cell(0);
		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_PTS split2" << endl;
		}
		if (Enc->nb_rows - Inc->nb_points()) {
			Stack->subset_continguous(
					Inc->nb_points(),
					Enc->nb_rows - Inc->nb_points());
			Stack->split_cell(0);
		}

	}
	
	else if (type == t_LNS) {
		
		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_LNS" << endl;
		}
		Stack->subset_continguous(P->N_points, 1);
		Stack->split_cell(0);
		Stack->subset_continguous(
				Inc->nb_points() + P->N_lines,
				Enc->nb_cols - P->N_lines);
		Stack->split_cell(0);

	}

	else if (type == t_PNL) {

		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_PNL" << endl;
		}
		Stack->subset_continguous(P->N_points, 1);
		Stack->split_cell(0);
		Stack->subset_continguous(
				Inc->nb_points() + P->N_lines,
				Enc->nb_cols - P->N_lines);
		Stack->split_cell(0);

	}

	else if (type == t_PAC) {
		
		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_PAC" << endl;
		}
		Stack->subset_continguous(P->N_points, Enc->nb_rows - P->N_points);
		Stack->split_cell(0);
		Stack->subset_continguous(
				Inc->nb_points() + P->N_lines,
				Enc->nb_cols - P->N_lines);
		Stack->split_cell(0);

	}
	else if (type == t_INC) {

		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_INC" << endl;
		}
		Stack->subset_continguous(v, b);
		Stack->split_cell(0);

	}
	else if (type == t_LS) {

		if (f_v) {
			cout << "object_with_canonical_form::encode_incma_and_make_decomposition t_LS" << endl;
		}
		Stack->subset_continguous(v, Enc->nb_rows - v);
		Stack->split_cell(0);
		Stack->subset_continguous(
				v + b,
				Enc->nb_cols - b);
		Stack->split_cell(0);

	}
	else {
		cout << "object_with_canonical_form::encode_incma_and_make_decomposition "
				"unknown type " << type << endl;
		exit(1);
	}
	
	if (f_v) {
		cout << "object_with_canonical_form::encode_incma_and_make_decomposition done" << endl;
	}
}

void object_with_canonical_form::encode_object(
		long int *&encoding, int &encoding_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object" << endl;
	}
	if (type == t_PTS) {
		
		encode_object_points(encoding, encoding_sz, verbose_level);

	}
	else if (type == t_LNS) {
		
		encode_object_lines(encoding, encoding_sz, verbose_level);

	}
	else if (type == t_PNL) {

		encode_object_points_and_lines(encoding, encoding_sz, verbose_level);

	}
	else if (type == t_PAC) {
		
		encode_object_packing(encoding, encoding_sz, verbose_level);

	}
	else if (type == t_INC) {

		encode_object_incidence_geometry(encoding, encoding_sz, verbose_level);

	}
	else if (type == t_LS) {

		encode_object_large_set(encoding, encoding_sz, verbose_level);

	}
	else {
		cout << "object_with_canonical_form::encode_object "
				"unknown type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_object "
				"encoding_sz=" << encoding_sz << endl;
	}
	if (f_v) {
		cout << "object_with_canonical_form::encode_object "
				"done" << endl;
	}
}

void object_with_canonical_form::encode_object_points(
		long int *&encoding, int &encoding_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_points" << endl;
	}
	encoding_sz = sz;
	encoding = NEW_lint(sz);
	Lint_vec_copy(set, encoding, sz);
}

void object_with_canonical_form::encode_object_lines(
		long int *&encoding, int &encoding_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_lines" << endl;
	}
	encoding_sz = sz;
	encoding = NEW_lint(sz);
	Lint_vec_copy(set, encoding, sz);
}

void object_with_canonical_form::encode_object_points_and_lines(
		long int *&encoding, int &encoding_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_points_and_lines" << endl;
	}
	encoding_sz = sz;
	encoding = NEW_lint(sz);
	Lint_vec_copy(set, encoding, sz);
}

void object_with_canonical_form::encode_object_packing(
		long int *&encoding, int &encoding_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_packing" << endl;
	}
	int i, h;
	
	encoding_sz = SoS->total_size();
	encoding = NEW_lint(encoding_sz);
	h = 0;
	for (i = 0; i < SoS->nb_sets; i++) {
		Lint_vec_copy(SoS->Sets[i], encoding + h, SoS->Set_size[i]);
		h += SoS->Set_size[i];
	}
	if (h != encoding_sz) {
		cout << "object_with_canonical_form::encode_object_packing "
				"h != encoding_sz" << endl;
		exit(1);
	}
}

void object_with_canonical_form::encode_object_incidence_geometry(
		long int *&encoding, int &encoding_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_incidence_geometry" << endl;
	}
	encoding_sz = sz;
	encoding = NEW_lint(sz);
	Lint_vec_copy(set, encoding, sz);
}

void object_with_canonical_form::encode_object_large_set(
		long int *&encoding, int &encoding_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::encode_object_large_set" << endl;
	}
	encoding_sz = sz;
	encoding = NEW_lint(sz);
	Lint_vec_copy(set, encoding, sz);
}

#if 0
void object_with_canonical_form::klein(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::klein" << endl;
	}
	if (type != t_LNS) {
		if (f_v) {
			cout << "object_with_canonical_form::klein "
					"not of type t_LNS" << endl;
		}
		return;
	}
	if (P->n != 3) {
		if (f_v) {
			cout << "object_with_canonical_form::klein "
					"not in three space" << endl;
		}
		return;
	}


	projective_space *P5;
	grassmann *Gr;
	long int *pts_klein;
	long int i, N;
	
	ring_theory::longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;



	P5 = NEW_OBJECT(projective_space);
	
	P5->init(5, P->F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	pts_klein = NEW_lint(sz);
	
	if (f_v) {
		cout << "object_with_canonical_form::klein "
				"before P3->klein_correspondence" << endl;
	}
	P->klein_correspondence(P5, 
		set, sz, pts_klein, 0/*verbose_level*/);


	N = P5->nb_rk_k_subspaces_as_lint(3);
	if (f_v) {
		cout << "object_with_canonical_form::klein N = " << N << endl;
	}

	

	Gr = NEW_OBJECT(grassmann);

	Gr->init(6, 3, P->F, 0 /* verbose_level */);

	if (f_v) {
		cout << "object_with_canonical_form::klein "
				"before plane_intersection_type_fast" << endl;
	}
	P5->plane_intersection_type_slow(Gr, pts_klein, sz, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level /*- 3*/);

	if (f_v) {
		cout << "object_with_canonical_form::klein "
				"We found " << nb_planes << " planes." << endl;

		tally C;

		C.init(nb_pts_on_plane, nb_planes, FALSE, 0);
		cout << "plane types are: ";
		C.print(TRUE /* f_backwards*/);
		cout << endl;
#if 0
		for (i = 0; i < nb_planes; i++) {
			if (nb_pts_on_plane[i] >= 3) {
				cout << setw(3) << i << " / " << nb_planes << " : " << R[i] 
					<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
				int_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
				cout << endl;
			}
		}
#endif
	}
	if (f_v) {
		cout << "before FREE_OBJECTS(R);" << endl;
	}
	FREE_OBJECTS(R);
	if (f_v) {
		cout << "before FREE_int(Pts_on_plane[i]);" << endl;
	}
	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	if (f_v) {
		cout << "before FREE_pint(Pts_on_plane);" << endl;
	}
	FREE_plint(Pts_on_plane);
	if (f_v) {
		cout << "before FREE_int(nb_pts_on_plane);" << endl;
	}
	FREE_int(nb_pts_on_plane);

	
	
	FREE_lint(pts_klein);
	FREE_OBJECT(P5);
	FREE_OBJECT(Gr);
	if (f_v) {
		cout << "object_with_canonical_form::klein done" << endl;
	}
}
#endif

void object_with_canonical_form::run_nauty(
		int f_compute_canonical_form,
		data_structures::bitvector *&Canonical_form,
		data_structures::nauty_output *&NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_with_canonical_form::run_nauty" << endl;
	}
	//int L;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::file_io Fio;
	nauty_interface Nau;
	combinatorics::encoded_combinatorial_object *Enc;

	if (f_v) {
		cout << "object_with_canonical_form::run_nauty" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty "
				"before encode_incma" << endl;
	}
	encode_incma(Enc, verbose_level - 1);
	if (f_v) {
		cout << "object_with_canonical_form::run_nauty "
				"after encode_incma" << endl;
	}
	if (verbose_level > 5) {
		cout << "object_with_canonical_form::run_nauty Incma:" << endl;
		//int_matrix_print_tight(Incma, nb_rows, nb_cols);
	}



	NO = NEW_OBJECT(data_structures::nauty_output);


	//L = Enc->nb_rows * Enc->nb_cols;

	if (verbose_level > 5) {
		cout << "object_with_canonical_form::run_nauty "
				"before NO->allocate" << endl;
	}

	NO->allocate(Enc->canonical_labeling_len, verbose_level - 2);

	if (f_v) {
		cout << "object_with_canonical_form::run_nauty "
				"before Nau.nauty_interface_matrix_int" << endl;
	}
	int t0, t1, dt, tps;
	double delta_t_in_sec;
	orbiter_kernel_system::os_interface Os;

	tps = Os.os_ticks_per_second();
	t0 = Os.os_ticks();


	Nau.nauty_interface_matrix_int(
		Enc,
		NO,
		verbose_level - 3);

	Int_vec_copy_to_lint(NO->Base, NO->Base_lint, NO->Base_length);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) dt / (double) tps;

	if (f_v) {
		cout << "object_with_canonical_form::run_nauty "
				"after Nau.nauty_interface_matrix_int, "
				"Ago=" << *NO->Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
	}
	if (verbose_level > 5) {
		int h;
		//int degree = nb_rows +  nb_cols;

		for (h = 0; h < NO->Aut_counter; h++) {
			cout << "aut generator " << h << " / " << NO->Aut_counter << " : " << endl;
			//Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}




	if (f_compute_canonical_form) {


		Enc->compute_canonical_form(Canonical_form,
				NO->canonical_labeling, verbose_level);

	}


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty before FREE_OBJECT(Enc)" << endl;
	}
	FREE_OBJECT(Enc);


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty done" << endl;
	}


}


void object_with_canonical_form::canonical_labeling(
		data_structures::nauty_output *NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	combinatorics::encoded_combinatorial_object *Enc;
	nauty_interface Nau;


	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling"
				<< endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling "
				"before encode_incma" << endl;
	}
	encode_incma(Enc, verbose_level - 1);
	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling "
				"after encode_incma" << endl;
	}
	if (verbose_level > 5) {
		cout << "object_with_canonical_form::canonical_labeling "
				"Incma:" << endl;
		Enc->print_incma();
	}

	if (f_vv) {
		cout << "object_with_canonical_form::canonical_labeling "
				"initializing Aut, Base, "
				"Transversal_length" << endl;
	}



	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling "
				"calling nauty_interface_matrix_int" << endl;
	}


	int t0, t1, dt;
	double delta_t_in_sec;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	Nau.nauty_interface_matrix_int(
			Enc,
			NO,
			verbose_level - 3);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) t1 / (double) dt;

	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << NO->Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
	}


	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << NO->Ago << endl;
	}
	FREE_OBJECT(Enc);
	if (f_v) {
		cout << "object_with_canonical_form::canonical_labeling done"
				<< endl;
	}
}

void object_with_canonical_form::run_nauty_basic(
		data_structures::nauty_output *&NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic"
				<< endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	int nb_rows, nb_cols;


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic before OiP->encoding_size" << endl;
	}
	encoding_size(nb_rows, nb_cols, 0 /*verbose_level*/);
	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic after OiP->encoding_size" << endl;
		cout << "object_with_canonical_form::run_nauty_basic nb_rows=" << nb_rows << endl;
		cout << "object_with_canonical_form::run_nauty_basic nb_cols=" << nb_cols << endl;
	}

	data_structures::bitvector *Canonical_form;


	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic "
				"before OwCF->run_nauty" << endl;
	}
	run_nauty(
			FALSE /* f_compute_canonical_form */, Canonical_form,
			NO,
			verbose_level);
	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic "
				"after OwCF->run_nauty" << endl;
	}

	if (f_v) {
		cout << "object_with_canonical_form::run_nauty_basic done" << endl;
	}
}




}}}



