// any_combinatorial_object.cpp
// 
// Anton Betten
//
// December 23, 2017
//
// previously: object_with_canonical_form.cpp
// 
//
//

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace canonical_form_classification {


any_combinatorial_object::any_combinatorial_object()
{
	Record_birth();
	P = NULL;
	f_has_label = false;
	//std::string label;
	type = t_PTS;
	//input_fname = NULL;
	input_idx = 0;
	f_has_known_ago = false;
	known_ago = 0;
	//set_as_string = NULL;

	set = NULL;
	sz = 0;

	set2 = NULL;
	sz2 = 0;

	v = 0;
	b = 0;

	f_partition = false;
	partition = NULL;

	design_k = 0;
	design_sz = 0;
	SoS = NULL;
	original_data = NULL;

	f_extended_incma = false;

	m = n = max_val = 0;

	C = NULL;
}

any_combinatorial_object::~any_combinatorial_object()
{
	Record_death();
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
	if (original_data) {
		FREE_lint(original_data);
	}
	if (SoS) {
		FREE_OBJECT(SoS);
	}
	if (C) {
		FREE_OBJECT(C);
	}
}

void any_combinatorial_object::set_label(
		std::string &object_label)
{
	f_has_label = true;
	label = object_label;
}


void any_combinatorial_object::print_brief(
		std::ostream &ost)
{

	cout << "any_combinatorial_object: "
			"set_as_string: " << set_as_string << endl;
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
	else if (type == t_MMX) {
		ost << "multi matrix:" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;
	}
}

void any_combinatorial_object::print_rows(
		std::ostream &ost,
		canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::print_rows" << endl;
	}

	//print_tex(ost);

	if (Report_options->f_show_incidence_matrices) {

		encoded_combinatorial_object *Enc;

		encode_incma(Enc, verbose_level);

		//Enc->latex_set_system_by_columns(ost, verbose_level);

		Enc->latex_set_system_by_rows(ost, verbose_level);

		//Enc->latex_incma(ost, verbose_level);

		FREE_OBJECT(Enc);
	}

	if (f_v) {
		cout << "any_combinatorial_object::print_rows done" << endl;
	}
}


void any_combinatorial_object::print_tex_detailed(
		std::ostream &ost,
		canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::print_tex_detailed" << endl;
	}

	if (f_v) {
		cout << "any_combinatorial_object::print_tex_detailed "
				"before print_tex" << endl;
	}
	print_tex(ost, verbose_level);
	if (f_v) {
		cout << "any_combinatorial_object::print_tex_detailed "
				"after print_tex" << endl;
	}

	if (Report_options->f_show_incidence_matrices) {

		if (f_v) {
			cout << "any_combinatorial_object::print_tex_detailed f_show_incma" << endl;
		}

		print_incidence_matrices(ost, Report_options, verbose_level);

	}

	if (f_v) {
		cout << "any_combinatorial_object::print_tex_detailed done" << endl;
	}
}

void any_combinatorial_object::print_incidence_matrices(
		std::ostream &ost,
		canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices" << endl;
	}


	ost << "\\subsubsection*{any\\_combinatorial\\_object::print\\_incidence\\_matrices}" << endl;

	encoded_combinatorial_object *Enc;

	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"before encode_incma" << endl;
	}
	encode_incma(Enc, verbose_level);
	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"after encode_incma" << endl;
	}

	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"before Enc->latex_set_system_by_rows_and_columns" << endl;
	}
	Enc->latex_set_system_by_rows_and_columns(ost, verbose_level);
	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"after Enc->latex_set_system_by_rows_and_columns" << endl;
	}



	other::graphics::draw_incidence_structure_description *Draw_incidence_options;

	if (Report_options->f_incidence_draw_options) {
		Draw_incidence_options = Get_draw_incidence_structure_options(
				Report_options->incidence_draw_options_label);
	}
	else {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"please use -incidence_draw_options" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"before Enc->latex_incma" << endl;
	}
	Enc->latex_incma(
			ost,
			Draw_incidence_options,
			verbose_level);
	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices "
				"after Enc->latex_incma" << endl;
	}
	ost << "\\\\" << endl;

	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "any_combinatorial_object::print_incidence_matrices done" << endl;
	}

}


std::string any_combinatorial_object::stringify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::stringify" << endl;
	}

	string s;

	if (type == t_PTS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PTS" << endl;
		}
	}
	else if (type == t_LNS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_LNS" << endl;
		}
	}
	else if (type == t_PNL) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PNL" << endl;
		}
	}
	else if (type == t_PAC) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PAC" << endl;
		}

		//original_data = NEW_lint(size_of_packing);
		//Lint_vec_copy(data, original_data, size_of_packing);

		int size_of_packing;

		size_of_packing = SoS->nb_sets;

		s = Lint_vec_stringify(original_data, size_of_packing);

	}
	else if (type == t_INC) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_INC" << endl;
		}
	}
	else if (type == t_LS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_LS" << endl;
		}

		//int nb_designs = b / design_sz;

		s = Lint_vec_stringify(set, b);



	}
	else if (type == t_MMX) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_MMX" << endl;
		}

	}


	if (f_v) {
		cout << "any_combinatorial_object::stringify done" << endl;
	}
	return s;
}

void any_combinatorial_object::print_tex(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::print_tex" << endl;
	}

	//ost << "\\subsubsection*{any\\_combinatorial\\_object::print\\_tex}" << endl;

	if (type == t_PTS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PTS" << endl;
		}
		ost << "Set of points of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
		//P->print_set_numerical(ost, set, sz);
		if (f_v) {
			cout << "any_combinatorial_object::print_tex "
					"before P->Reporting->print_set_of_points" << endl;
		}
		P->Reporting->print_set_of_points(ost, set, sz);
		if (f_v) {
			cout << "any_combinatorial_object::print_tex "
					"after P->Reporting->print_set_of_points" << endl;
		}
	}
	else if (type == t_LNS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_LNS" << endl;
		}
		ost << "Set of lines of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
	}
	else if (type == t_PNL) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PNL" << endl;
		}
		ost << "Set of points of size " << sz << ": ";
		Lint_vec_print(ost, set, sz);
		ost << "\\\\" << endl;
		ost << "and a set of lines of size " << sz2 << ": ";
		Lint_vec_print(ost, set2, sz2);
		ost << "\\\\" << endl;
	}
	else if (type == t_PAC) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_PAC" << endl;
		}
		ost << "Packing: \\\\" << endl;
		SoS->print_table_tex(ost);
		ost << endl;
	}
	else if (type == t_INC) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_INC" << endl;
		}
		ost << "Incidence structure by flags: \\\\" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;
		Lint_vec_print_fully(ost, set, sz);
		ost << "\\\\" << endl;
#if 0
		object_with_canonical_form::set = NEW_lint(data_sz);
		Lint_vec_copy(data, object_with_canonical_form::set, data_sz);
		object_with_canonical_form::sz = data_sz;
		object_with_canonical_form::v = v;
		object_with_canonical_form::b = b;
#endif
	}
	else if (type == t_LS) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_LS" << endl;
		}
		ost << "Large set: \\\\" << endl;
		//SoS->print_table_tex(ost);
		//ost << endl;

		int nb_designs = b / design_sz;

		for (int i = 0; i < nb_designs; i++) {
			Lint_vec_print(ost, set + i * design_sz, design_sz);
			ost << "\\\\" << endl;
		}

		//combinatorics::other_combinatorics::combinatorics_domain Combi;

		combinatorics::design_theory::design_theory_global Design;

		ost << "Large set: \\\\" << endl;

		Design.report_large_set(
				ost, set, nb_designs,
				v, design_k, design_sz, verbose_level);

		if (v <= 10) {
			Design.report_large_set_compact(
					ost, set, nb_designs,
					v, design_k, design_sz, verbose_level);
		}

#if 0
		int j, h;
		int a;
		int *the_block;

		the_block = NEW_int(design_k);

		for (i = 0; i < nb_designs; i++) {
			for (j = 0; j < design_sz; j++) {
				a = set[i * design_sz + j];


				Combi.unrank_k_subset(a, the_block, v, design_k);
				for (h = 0; h < design_k; h++) {
					ost << the_block[h];
					if (h < design_k - 1) {
						ost << ", ";
					}
				}
				ost << "\\\\" << endl;

				}
			//Lint_vec_print(ost, set + i * design_sz, design_sz);
			ost << "\\\\" << endl;
		}
		FREE_int(the_block);
#endif


	}
	else if (type == t_MMX) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_MMX" << endl;
		}
		ost << "Multi matrix: \\\\" << endl;


		other::l1_interfaces::latex_interface Latex;


		std::string *headers_row;
		std::string *headers_col;
		std::string *Table;

		headers_row = new std::string[m];
		headers_col = new std::string[n];
		Table = new std::string[m * n];

		int i, j;

		for (i = 0; i < m; i++) {
			headers_row[i] = std::to_string(set[i]);
		}
		for (j = 0; j < n; j++) {
			headers_col[j] = std::to_string(set[m + j]);
		}
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				Table[i * n + j] = std::to_string(set[m + n + i * n + j]);
			}
		}
		ost << "$$" << endl;
		Latex.print_table_of_strings_with_headers_rc(
				ost, headers_row, headers_col, Table, m, n);
		ost << "$$" << endl;

		delete [] headers_row;
		delete [] headers_col;
		delete [] Table;

	}


	if (f_v) {
		cout << "any_combinatorial_object::print_tex done" << endl;
	}
}

void any_combinatorial_object::get_packing_as_set_system(
		long int *&Sets,
		int &nb_sets, int &set_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "any_combinatorial_object::get_packing_as_set_system" << endl;
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
		cout << "any_combinatorial_object::get_packing_as_set_system done" << endl;
	}
}


void any_combinatorial_object::init_input_fname(
		std::string &input_fname,
		int input_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_input_fname" << endl;
	}

	any_combinatorial_object::input_fname = input_fname;
	any_combinatorial_object::input_idx = input_idx;

	if (f_v) {
		cout << "any_combinatorial_object::init_input_fname done" << endl;
	}

}

void any_combinatorial_object::init_point_set(
		long int *set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_point_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PTS;
	any_combinatorial_object::set = NEW_lint(sz);
	Lint_vec_copy(set, any_combinatorial_object::set, sz);
	any_combinatorial_object::sz = sz;
	if (f_v) {
		cout << "any_combinatorial_object::init_point_set done" << endl;
	}
}

void any_combinatorial_object::init_point_set_from_string(
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_point_set_from_string" << endl;
	}

	type = t_PTS;

	Get_lint_vector_from_label(set_text, set, sz, verbose_level);

	if (f_v) {
		cout << "any_combinatorial_object::init_point_set_from_string done" << endl;
	}
}


void any_combinatorial_object::init_line_set(
		long int *set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_line_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_LNS;
	any_combinatorial_object::set = NEW_lint(sz);
	Lint_vec_copy(set, any_combinatorial_object::set, sz);
	any_combinatorial_object::sz = sz;
	if (f_v) {
		cout << "any_combinatorial_object::init_line_set done" << endl;
	}
}

void any_combinatorial_object::init_line_set_from_string(
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_line_set_from_string" << endl;
	}

	type = t_LNS;

	Lint_vec_scan(set_text, set, sz);

	if (f_v) {
		cout << "any_combinatorial_object::init_line_set_from_string done" << endl;
	}
}

void any_combinatorial_object::init_points_and_lines(
	long int *set, int sz,
	long int *set2, int sz2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_points_and_lines" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PNL;

	any_combinatorial_object::set = NEW_lint(sz);
	Lint_vec_copy(set, any_combinatorial_object::set, sz);
	any_combinatorial_object::sz = sz;

	any_combinatorial_object::set2 = NEW_lint(sz2);
	Lint_vec_copy(set2, any_combinatorial_object::set2, sz2);
	any_combinatorial_object::sz2 = sz2;

	if (f_v) {
		cout << "any_combinatorial_object::init_points_and_lines done" << endl;
	}
}

void any_combinatorial_object::init_points_and_lines_from_string(
	std::string &set_text,
	std::string &set2_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_points_and_lines_from_string" << endl;
	}

	type = t_PNL;

	Lint_vec_scan(set_text, set, sz);

	Lint_vec_scan(set2_text, set2, sz2);

	if (f_v) {
		cout << "any_combinatorial_object::init_points_and_lines_from_string done" << endl;
	}
}

void any_combinatorial_object::init_packing_from_set(
		long int *packing, int sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, q, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_set" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	q = P->Subspaces->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	if (sz != size_of_packing * size_of_spread) {
		cout << "any_combinatorial_object::init_packing_from_set "
			"sz != size_of_packing * size_of_spread" << endl;
		exit(1);
	}
	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	SoS->init_basic_constant_size(
			P->Subspaces->N_lines,
		size_of_packing /* nb_sets */, 
		size_of_spread /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		Lint_vec_copy(packing + i * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
#if 0
	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_set it is" << endl;
		SoS->print_table();
	}
#endif
	
	
	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_set done" << endl;
	}
}


void any_combinatorial_object::init_packing_from_string(
		std::string &packing_text,
		int q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_string" << endl;
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
		cout << "any_combinatorial_object::init_packing_from_string "
			"sz != N_lines" << endl;
		exit(1);
	}
	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	SoS->init_basic_constant_size(
			N_lines,
		size_of_packing /* nb_sets */,
		size_of_spread /* constant_size */,
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		Lint_vec_copy(packing + i * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
#if 0
	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_string it is" << endl;
		SoS->print_table();
	}
#endif


	FREE_lint(packing);

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_string done" << endl;
	}
}

void any_combinatorial_object::init_packing_from_set_of_sets(
		other::data_structures::set_of_sets *SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_set_of_sets" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	//object_in_projective_space::set = NEW_int(sz);
	//int_vec_copy(set, object_in_projective_space::set, sz);
	//object_in_projective_space::sz = sz;

	any_combinatorial_object::SoS = SoS->copy();

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_set_of_sets done" << endl;
	}
}


void any_combinatorial_object::init_packing_from_spread_table(
	long int *data,
	long int *Spread_table, int nb_spreads, int spread_size,
	int q,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, size_of_spread, size_of_packing;
	int N_lines;

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_spread_table" << endl;
	}
	//object_with_canonical_form::P = P;
	type = t_PAC;
	//q = P->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;

	original_data = NEW_lint(size_of_packing);
	Lint_vec_copy(data, original_data, size_of_packing);

	if (spread_size != size_of_spread) {
		cout << "any_combinatorial_object::init_packing_from_spread_table "
				"spread_size != size_of_spread" << endl;
		exit(1);
	}

	N_lines = size_of_spread * size_of_packing;

	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	SoS->init_basic_constant_size(
			N_lines,
		size_of_packing /* nb_sets */,
		size_of_spread /* constant_size */,
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		a = data[i];
		Lint_vec_copy(
				Spread_table + a * size_of_spread,
				SoS->Sets[i], size_of_spread);
	}
	if (verbose_level >= 5) {
		cout << "any_combinatorial_object::init_packing_from_spread_table "
				"Sos:" << endl;
		SoS->print_table();
	}

	// test if the object is a packing:
	SoS->sort_all(false /*verbose_level*/);

	int *M;
	int j;

	SoS->pairwise_intersection_matrix(M, 0 /*verbose_level*/);

	for (i = 0; i < SoS->nb_sets; i++) {
		for (j = i + 1; j < SoS->nb_sets; j++) {
			if (M[i * SoS->nb_sets + j]) {
				cout << "any_combinatorial_object::init_packing_from_spread_table "
						"not a packing, spreads "
						<< i << " and " << j << " meet in "
						<< M[i * SoS->nb_sets + j] << " lines" << endl;
				cout << "any_combinatorial_object::init_packing_from_spread_table "
						"Sos:" << endl;
				SoS->print_table();
				exit(1);

			}
		}
	}
	FREE_int(M);

	if (f_v) {
		cout << "any_combinatorial_object::init_packing_from_spread_table done" << endl;
	}
}

void any_combinatorial_object::init_design_from_block_orbits(
		other::data_structures::set_of_sets *Block_orbits,
		long int *Solution, int width,
		int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_orbits" << endl;
	}

	if (width != Block_orbits->nb_sets) {
		cout << "any_combinatorial_object::init_design_from_block_orbits "
				"width != Block_orbits->nb_sets" << endl;
		exit(1);
	}
	int h, len, nb_flags;

	nb_flags = 0;
	for (h = 0; h < width; h++) {
		if (Solution[h] == 0) {
			continue;
		}
		len = Block_orbits->Set_size[h];
		nb_flags += len;
		if (f_v) {
			cout << "block orbit " << h << ", number of flags = " << len << endl;
		}
	}

	v = Block_orbits->underlying_set_size;
	b = nb_flags / k;

	if (b * k != nb_flags) {
		cout << "any_combinatorial_object::init_design_from_block_orbits "
				"k does not divide the number of flags" << endl;
		cout << "nb_flags=" << nb_flags << endl;
		cout << "k=" << k << endl;
		exit(1);
	}

	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_orbits v = " << v << endl;
		cout << "any_combinatorial_object::init_design_from_block_orbits b = " << b << endl;
		cout << "any_combinatorial_object::init_design_from_block_orbits nb_flags = " << nb_flags << endl;
	}

	any_combinatorial_object::P = NULL;

	type = t_INC;

	any_combinatorial_object::set = NEW_lint(nb_flags);

	int *incma;
	int i, j, l, u, a;

	incma = NEW_int(v * b);
	Int_vec_zero(incma, v * b);

	j = 0;
	for (h = 0; h < width; h++) {
		if (Solution[h] == 0) {
			continue;
		}
		len = Block_orbits->Set_size[h];
		l = len / k;
		if (l * k != len) {
			cout << "any_combinatorial_object::init_design_from_block_orbits "
					"l * k != len" << endl;
			exit(1);
		}
		for (a = 0; a < l; a++, j++) {
			for (u = 0; u < k; u++) {
				i = Block_orbits->Sets[h][a * k + u];
				if (i < 0 || i >= v) {
					cout << "any_combinatorial_object::init_design_from_block_orbits "
							"i is out of range" << endl;
					exit(1);
				}
				incma[i * b + j] = 1;
			}
		}
	}
	int f;

	f = 0;
	for (i = 0; i < v * b; i++) {
		if (incma[i]) {
			set[f++] = i;
		}
	}
	if (f != nb_flags) {
		cout << "any_combinatorial_object::init_design_from_block_orbits "
				"f != nb_flags" << endl;
		exit(1);
	}
	FREE_int(incma);
	any_combinatorial_object::sz = nb_flags;
	//any_combinatorial_object::v = v;
	//any_combinatorial_object::b = b;


	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_orbits done" << endl;
	}
}

void any_combinatorial_object::init_design_from_block_table(
		long int *Block_table, int v, int nb_blocks, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_table" << endl;
	}

	any_combinatorial_object::v = v;
	b = nb_blocks;
	int nb_flags = b * k;
	sz = nb_flags;

	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_table v = " << v << endl;
		cout << "any_combinatorial_object::init_design_from_block_table b = " << b << endl;
		cout << "any_combinatorial_object::init_design_from_block_table k = " << k << endl;
		cout << "any_combinatorial_object::init_design_from_block_table nb_flags = " << nb_flags << endl;
	}

	any_combinatorial_object::P = NULL;

	type = t_INC;

	any_combinatorial_object::set = NEW_lint(nb_flags);

	int i, j, h, f;

	f = 0;
	for (j = 0; j < b; j++) {
		for (h = 0; h < k; h++) {
			f = j * k + h;
			i = Block_table[f];
			set[f++] = i * b + j;
		}
	}
	if (f != nb_flags) {
		cout << "any_combinatorial_object::init_design_from_block_table "
				"f != nb_flags" << endl;
		exit(1);
	}
	any_combinatorial_object::sz = nb_flags;
	//any_combinatorial_object::v = v;
	//any_combinatorial_object::b = b;


	if (f_v) {
		cout << "any_combinatorial_object::init_design_from_block_table done" << endl;
	}
}

void any_combinatorial_object::init_incidence_geometry(
	long int *data, int data_sz, int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry" << endl;
	}
	if (nb_flags != data_sz) {
		cout << "any_combinatorial_object::init_incidence_geometry "
				"nb_flags != data_sz" << endl;
	}
	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(data_sz);
	Lint_vec_copy(data, any_combinatorial_object::set, data_sz);
	any_combinatorial_object::sz = data_sz;
	any_combinatorial_object::v = v;
	any_combinatorial_object::b = b;
	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry done" << endl;
	}
}

void any_combinatorial_object::init_incidence_geometry_from_vector(
	std::vector<int> &Flags, int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry" << endl;
	}
	if (nb_flags != Flags.size()) {
		cout << "any_combinatorial_object::init_incidence_geometry "
				"nb_flags != Flags.size()" << endl;
	}

	any_combinatorial_object::P = NULL;

	type = t_INC;

	any_combinatorial_object::set = NEW_lint(Flags.size());

	int i;

	for (i = 0; i < Flags.size(); i++) {
		set[i] = Flags[i];
	}
	any_combinatorial_object::sz = Flags.size();
	any_combinatorial_object::v = v;
	any_combinatorial_object::b = b;
	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry done" << endl;
	}
}

void any_combinatorial_object::init_incidence_geometry_from_string(
	std::string &data,
	int v, int b, int nb_flags,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string" << endl;
	}
	long int *flags;
	int data_sz;

	Lint_vec_scan(data, flags, data_sz);

	if (nb_flags != data_sz) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string "
				"nb_flags != data_sz" << endl;
	}
	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(data_sz);
	Lint_vec_copy(flags, any_combinatorial_object::set, data_sz);
	any_combinatorial_object::sz = data_sz;
	any_combinatorial_object::v = v;
	any_combinatorial_object::b = b;

	FREE_lint(flags);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string done" << endl;
	}
}

void any_combinatorial_object::init_incidence_geometry_from_string_of_row_ranks(
	std::string &data,
	int v, int b, int r,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string" << endl;
	}
	long int *row_ranks;
	long int *flags;
	int *row_set;
	int data_sz;
	int nb_flags;
	int i, h, a;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Lint_vec_scan(data, row_ranks, data_sz);

	if (v != data_sz) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string "
				"v != data_sz" << endl;
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

	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(nb_flags);
	Lint_vec_copy(flags, any_combinatorial_object::set, nb_flags);
	any_combinatorial_object::sz = nb_flags;
	any_combinatorial_object::v = v;
	any_combinatorial_object::b = b;

	FREE_int(row_set);
	FREE_lint(row_ranks);
	FREE_lint(flags);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_geometry_from_string done" << endl;
	}
}


void any_combinatorial_object::init_large_set(
	long int *data, int data_sz,
	int v, int b, int k, int design_sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_large_set" << endl;
	}

	if (data_sz != b) {
		cout << "any_combinatorial_object::init_large_set "
				"data_sz != b" << endl;
		exit(1);
	}
	any_combinatorial_object::P = NULL;
	type = t_LS;
	any_combinatorial_object::set = NEW_lint(data_sz);
	Lint_vec_copy(data, any_combinatorial_object::set, data_sz);
	any_combinatorial_object::sz = data_sz;
	any_combinatorial_object::v = v;
	any_combinatorial_object::b = data_sz;
	any_combinatorial_object::design_k = k;
	any_combinatorial_object::design_sz = design_sz;
	if (f_v) {
		cout << "any_combinatorial_object::init_large_set done" << endl;
	}
}

void any_combinatorial_object::init_large_set_from_string(
	std::string &data_text, int v, int k, int design_sz,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_large_set_from_string" << endl;
	}
	any_combinatorial_object::P = NULL;

	type = t_LS;

	Lint_vec_scan(data_text, set, sz);

	any_combinatorial_object::v = v;
	any_combinatorial_object::b = sz;
	any_combinatorial_object::design_k = k;
	any_combinatorial_object::design_sz = design_sz;
	if (f_v) {
		cout << "any_combinatorial_object::init_large_set_from_string done" << endl;
	}
}

void any_combinatorial_object::init_graph_by_adjacency_matrix_text(
		std::string &adjacency_matrix_text, int N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix_text" << endl;
	}

	int N2;

	N2 = (N * (N - 1)) >> 1;

	long int *adjacency_matrix;
	int adj_sz;
	//int nb_edges;
	//int i;

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix_text "
				"N=" << N << " N2=" << N2 << endl;
	}
	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix_text "
				"adjacency_matrix_text: " << adjacency_matrix_text << endl;
	}

	Lint_vec_scan(adjacency_matrix_text, adjacency_matrix, adj_sz);

	if (adj_sz != N2) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix_text "
				"size of adjacency matrix is incorrect" << endl;
		exit(1);
	}


	init_graph_by_adjacency_matrix(
			adjacency_matrix, adj_sz, N,
			verbose_level);


	FREE_lint(adjacency_matrix);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix done" << endl;
	}

}



void any_combinatorial_object::init_graph_by_adjacency_matrix(
		long int *adjacency_matrix, int adj_sz, int N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix" << endl;
	}

	int N2;

	N2 = (N * (N - 1)) >> 1;

	int nb_edges;
	int i;

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix "
				"N=" << N << " N2=" << N2 << endl;
	}

	if (adj_sz != N2) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix "
				"size of adjacency matrix is incorrect" << endl;
		exit(1);
	}

	nb_edges = 0;
	for (i = 0; i < N2; i++) {
		if (adjacency_matrix[i]) {
			nb_edges++;
		}
	}
	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix "
				"nb_edges=" << nb_edges << endl;
	}

	long int *flags;
	int data_sz;
	int j, h, k;

	data_sz = nb_edges * 2;
	flags = NEW_lint(data_sz);
	k = 0;
	h = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++, h++) {
			if (adjacency_matrix[h]) {
				flags[2 * k + 0] = i * nb_edges + k;
				flags[2 * k + 1] = j * nb_edges + k;
				k++;
			}
		}
	}

	if (h != N2) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix "
				"h != N2" << endl;
		exit(1);
	}
	if (k != nb_edges) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix "
				"k != nb_edges" << endl;
		exit(1);
	}
	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(data_sz);
	Lint_vec_copy(flags, any_combinatorial_object::set, data_sz);
	any_combinatorial_object::sz = data_sz;
	any_combinatorial_object::v = N;
	any_combinatorial_object::b = nb_edges;

	FREE_lint(flags);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_adjacency_matrix done" << endl;
	}

}

void any_combinatorial_object::init_graph_by_object(
		combinatorics::graph_theory::colored_graph *CG,
		int verbose_level)
// A graph is converted into an incidence geometry
// with v points and nb_edges lines of size 2.
// The flags of the incidence geometry are stored.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_object" << endl;
	}

	int N, nb_edges;

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_object "
				"CG->nb_points = " << CG->nb_points << endl;
	}


	N = CG->nb_points;

	nb_edges = CG->get_nb_edges(0 /*verbose_level */);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_object "
				"nb_edges = " << nb_edges << endl;
	}

	long int *flags;
	int data_sz;
	int i, j, k;

	data_sz = nb_edges * 2;
	flags = NEW_lint(data_sz);
	k = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			if (CG->is_adjacent(i, j)) {
				flags[2 * k + 0] = i * nb_edges + k;
				flags[2 * k + 1] = j * nb_edges + k;
				k++;
			}
		}
	}
	if (k != nb_edges) {
		cout << "any_combinatorial_object::init_graph_by_object "
				"k != nb_edges" << endl;
		exit(1);
	}
	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(data_sz);
	Lint_vec_copy(flags, any_combinatorial_object::set, data_sz);
	any_combinatorial_object::sz = data_sz;
	any_combinatorial_object::v = N;
	any_combinatorial_object::b = nb_edges;

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_object "
				"v=" << v << " b=" << b << endl;
	}

	FREE_lint(flags);

	if (f_v) {
		cout << "any_combinatorial_object::init_graph_by_object done" << endl;
	}
}


void any_combinatorial_object::init_incidence_structure_from_design_object(
		combinatorics::design_theory::design_object *Design_object,
		int verbose_level)
// A design with v points and b blocks is converted into an incidence geometry
// with 2 * v points and 2 + k * b blocks.
// Two blocks have size v, and all remaining blocks have size 2.
// The flags of this expanded incidence geometry are stored.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object" << endl;
	}

	int v, b, k;



	v = Design_object->v;
	b = Design_object->b;
	k = Design_object->k;

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object v = " << v << endl;
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object k = " << k << endl;
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object b = " << b << endl;
	}


	long int *flags;
	int nb_flags;
	int i, j, col, cur_flag;
	int V, B;

	nb_flags = v * 2 + 2 * k * b;
	V = 2 * v;
	B = 2 + k * b;

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object "
				"nb_flags = " << nb_flags << endl;
	}

	flags = NEW_lint(nb_flags);
	cur_flag = 0;
	col = 0;
	for (i = 0; i < v; i++) {
		flags[cur_flag++] = i * B + col;
	}
	col++;
	for (i = 0; i < v; i++) {
		flags[cur_flag++] = (v + i) * B + col;
	}
	col++;
	for (j = 0; j < b; j++) {
		for (i = 0; i < v; i++) {
			if (Design_object->incma[i * b + j]) {
				flags[cur_flag++] = i * B + col;
				flags[cur_flag++] = (v + j) * B + col;
				col++;
			}
		}
	}
	if (col != B) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object "
				"col != B" << endl;
		cout << "col=" << col << endl;
		cout << "B=" << B << endl;
		exit(1);
	}
	if (cur_flag != nb_flags) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object "
				"cur_flag != nb_flags" << endl;
		exit(1);
	}
	any_combinatorial_object::P = NULL;
	type = t_INC;
	any_combinatorial_object::set = NEW_lint(nb_flags);
	Lint_vec_copy(flags, any_combinatorial_object::set, nb_flags);
	any_combinatorial_object::sz = nb_flags;
	any_combinatorial_object::v = V;
	any_combinatorial_object::b = B;

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object "
				"V=" << V << " B=" << B << endl;
	}

	FREE_lint(flags);

	if (f_v) {
		cout << "any_combinatorial_object::init_incidence_structure_from_design_object done" << endl;
	}
}



void any_combinatorial_object::init_multi_matrix(
		std::string &data1,
		std::string &data2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix" << endl;
	}
	long int *entries1;
	int nb_entries1;
	long int *entries2;
	int nb_entries2;

	Lint_vec_scan(data1, entries1, nb_entries1);
	Lint_vec_scan(data2, entries2, nb_entries2);

	if (nb_entries1 != 3) {
		cout << "any_combinatorial_object::init_multi_matrix "
				"nb_entries1 != 3" << endl;
		exit(1);
	}

	any_combinatorial_object::P = NULL;
	type = t_MMX;
	any_combinatorial_object::set = NEW_lint(nb_entries2);
	Lint_vec_copy(entries2, any_combinatorial_object::set, nb_entries2);
	any_combinatorial_object::sz = nb_entries2;
	any_combinatorial_object::m = entries1[0];
	any_combinatorial_object::n = entries1[1];
	any_combinatorial_object::max_val = entries1[2];
	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix "
				"m = " << m << endl;
		cout << "any_combinatorial_object::init_multi_matrix "
				"n = " << n << endl;
		cout << "any_combinatorial_object::init_multi_matrix "
				"max_val = " << max_val << endl;
	}

	if (nb_entries2 != m + n + m * n) {
		cout << "any_combinatorial_object::init_multi_matrix "
				"nb_entries2 != m + n + m * n" << endl;
		exit(1);
	}

	FREE_lint(entries1);
	FREE_lint(entries2);

	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix done" << endl;
	}
}

void any_combinatorial_object::init_multi_matrix_from_data(
		int nb_V, int nb_B, int *V, int *B, int *scheme,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix_from_data" << endl;
	}


	sz = nb_V + nb_B + nb_V * nb_B;

	P = NULL;
	type = t_MMX;
	set = NEW_lint(sz);

	Int_vec_copy_to_lint(V, set, nb_V);
	Int_vec_copy_to_lint(B, set + nb_V, nb_B);
	Int_vec_copy_to_lint(scheme, set + nb_V + nb_B, nb_V * nb_B);
	m = nb_V;
	n = nb_B;
	max_val = Lint_vec_maximum(set, sz);

	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix_from_data "
				"m=" << m << " n=" << n
				<< " max_val=" << max_val << endl;
	}

	if (f_v) {
		cout << "any_combinatorial_object::init_multi_matrix_from_data done" << endl;
	}
}


void any_combinatorial_object::encoding_size(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size" << endl;
	}
	if (type == t_PTS) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_point_set" << endl;
		}
		encoding_size_point_set(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_LNS) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_line_set" << endl;
		}
		encoding_size_line_set(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_PNL) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_points_and_lines" << endl;
		}
		encoding_size_points_and_lines(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_PAC) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_packing" << endl;
		}
		encoding_size_packing(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_INC) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_packing" << endl;
		}
		encoding_size_incidence_geometry(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_LS) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_large_set" << endl;
		}
		encoding_size_large_set(
				nb_rows, nb_cols, verbose_level);

	}
	else if (type == t_MMX) {

		if (f_v) {
			cout << "any_combinatorial_object::encoding_size "
					"before encoding_size_multi_matrix" << endl;
		}
		encoding_size_multi_matrix(
				nb_rows, nb_cols, verbose_level);

	}
	else {
		cout << "any_combinatorial_object::encoding_size "
				"unknown type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "any_combinatorial_object::encoding_size done" << endl;
	}
}

void any_combinatorial_object::encoding_size_point_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_point_set" << endl;
	}


	C = NEW_OBJECT(other::data_structures::tally);

	C->init_lint(set, sz, true, 0);
	if (C->second_nb_types > 1) {
		cout << "any_combinatorial_object::encoding_size_point_set "
				"The set is a multiset:" << endl;
		C->print(false /*f_backwards*/);
	}


	if (f_v) {
		cout << "The type of the set is:" << endl;
		C->print(false /*f_backwards*/);
		cout << "C->second_nb_types = " << C->second_nb_types << endl;
	}

	int nb_rows0, nb_cols0;
	//int f_extended_incma = false;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;

	if (f_extended_incma) {
		// space for the lines vs planes incidence matrix:
		nb_rows0 += P->Subspaces->N_lines;
		nb_cols0 += P->Subspaces->Nb_subspaces[2];
	}


	// for the decoration:
	nb_rows = nb_rows0 + 1;
	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_point_set "
				"nb_rows=" << nb_rows << endl;
	}
	nb_cols = nb_cols0 + C->second_nb_types;
	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_point_set "
				"nb_cols=" << nb_cols << endl;
	}
	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_point_set "
				"before FREE_OBJECT(C)" << endl;
	}
	FREE_OBJECT(C);
	C = NULL;
	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_point_set "
				"done" << endl;
	}

}

void any_combinatorial_object::encoding_size_line_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_line_set" << endl;
	}


	nb_rows = P->Subspaces->N_points + 1;
	nb_cols = P->Subspaces->N_lines + 1;

}

void any_combinatorial_object::encoding_size_points_and_lines(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_points_and_lines" << endl;
	}


	nb_rows = P->Subspaces->N_points + 1;
	nb_cols = P->Subspaces->N_lines + 1;

}

void any_combinatorial_object::encoding_size_packing(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_packing" << endl;
	}

	nb_rows = P->Subspaces->N_points + SoS->nb_sets;
	nb_cols = P->Subspaces->N_lines + 1;

}

void any_combinatorial_object::encoding_size_large_set(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_designs;

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_large_set" << endl;
	}

	nb_designs = b / design_sz;
	if (nb_designs * design_sz != b) {
		cout << "any_combinatorial_object::encoding_size_large_set "
				"design_sz does not divide b" << endl;
		exit(1);
	}

	nb_rows = v + nb_designs;
	nb_cols = b + 1;

}


void any_combinatorial_object::encoding_size_incidence_geometry(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_incidence_geometry" << endl;
	}

	nb_rows = v;
	nb_cols = b;

}

void any_combinatorial_object::encoding_size_multi_matrix(
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_designs;

	if (f_v) {
		cout << "any_combinatorial_object::encoding_size_multi_matrix" << endl;
	}

	nb_designs = b / design_sz;
	if (nb_designs * design_sz != b) {
		cout << "any_combinatorial_object::encoding_size_multi_matrix "
				"design_sz does not divide b" << endl;
		exit(1);
	}

	nb_rows = m + n + max_val + 1;
	nb_cols = m + n + m * n;

}


void any_combinatorial_object::canonical_form_given_canonical_labeling(
		int *canonical_labeling,
		other::data_structures::bitvector *&B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::canonical_form_given_canonical_labeling" << endl;
	}

	encoded_combinatorial_object *Enc;

	encode_incma(Enc, verbose_level - 1);
	if (f_v) {
		cout << "any_combinatorial_object::canonical_form_given_canonical_labeling "
				"after OiP->encode_incma" << endl;
	}

	Enc->canonical_form_given_canonical_labeling(
			canonical_labeling,
				B,
				verbose_level);


	FREE_OBJECT(Enc);


	if (f_v) {
		cout << "any_combinatorial_object::canonical_form_given_canonical_labeling done" << endl;
	}
}

void any_combinatorial_object::encode_incma(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_incma" << endl;
	}
	if (type == t_PTS) {
		
		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_PTS" << endl;
		}
		encode_point_set(Enc, verbose_level);

	}
	else if (type == t_LNS) {
		
		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_LNS" << endl;
		}
		encode_line_set(Enc, verbose_level);

	}
	else if (type == t_PNL) {

		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_PNL" << endl;
		}
		encode_points_and_lines(Enc, verbose_level);

	}
	else if (type == t_PAC) {
		
		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_PAC" << endl;
		}
		encode_packing(Enc, verbose_level);

	}
	else if (type == t_INC) {

		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_INC" << endl;
		}
		encode_incidence_geometry(Enc, verbose_level);

	}
	else if (type == t_LS) {

		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_LS" << endl;
		}
		encode_large_set(Enc, verbose_level);

	}
	else if (type == t_MMX) {

		if (f_v) {
			cout << "any_combinatorial_object::encode_incma type == t_MMX" << endl;
		}
		encode_multi_matrix(Enc, verbose_level);

	}
	else {
		cout << "any_combinatorial_object::encode_incma "
				"unknown type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_incma done" << endl;
	}
}

void any_combinatorial_object::encode_point_set(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_point_set" << endl;
	}
	int i, j;
	int f_vvv = false; // (verbose_level >= 3);
	//int f_extended_incma = false;
	

	C = NEW_OBJECT(other::data_structures::tally);

	if (f_v) {
		cout << "any_combinatorial_object::encode_point_set set=";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}
	C->init_lint(set, sz, true, 0);
	if (C->second_nb_types > 1) {
		cout << "any_combinatorial_object::encode_point_set "
				"The set is a multiset:" << endl;
		C->print(false /*f_backwards*/);
	}


	if (f_v) {
		cout << "The type of the set is:" << endl;
		C->print(false /*f_backwards*/);
		cout << "C->second_nb_types = " << C->second_nb_types << endl;
	}


	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;

	if (P->Subspaces->n >= 3 && f_extended_incma) {
		nb_rows0 += P->Subspaces->N_lines;
		nb_cols0 += P->Subspaces->Nb_subspaces[2];
	}

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + C->second_nb_types;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);



	//Enc->incidence_matrix_projective_space_top_left(P, verbose_level);

	if (P->Subspaces->n >= 3 && f_extended_incma) {
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
			if (i > P->Subspaces->N_points) {
				cout << "any_combinatorial_object::encode_point_set "
						"i > P->N_points" << endl;
				cout << "i = " << i << endl;
				cout << "P->N_points = " << P->Subspaces->N_points << endl;
				cout << "h=" << h << " idx=" << idx << " f=" << f
						<< " l=" << l << " i=" << i << endl;
				exit(1);
			}
			Enc->set_incidence_ij(i, Enc->nb_cols0 + j);
		}
	}

	if (f_v) {
		cout << "any_combinatorial_object::encode_point_set "
				"bottom right entries" << endl;
	}
	// bottom right entries:
	for (j = 0; j < C->second_nb_types; j++) {
		Enc->set_incidence_ij(Enc->nb_rows0, Enc->nb_cols0 + j);
	}

	if (f_v) {
		cout << "any_combinatorial_object::encode_point_set partition" << endl;
	}


	Enc->partition[P->Subspaces->N_points - 1] = 0;
	Enc->partition[nb_rows0 - 1] = 0;
	Enc->partition[nb_rows - 1] = 0;

	Enc->partition[nb_rows + P->Subspaces->N_lines - 1] = 0;
	Enc->partition[nb_rows + Enc->nb_cols0 - 1] = 0;

	for (j = 0; j < C->second_nb_types; j++) {
		Enc->partition[nb_rows + Enc->nb_cols0 + j] = 0;
	}
	if (f_vvv) {
		cout << "any_combinatorial_object::encode_point_set "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_point_set "
				"done" << endl;
	}
}

void any_combinatorial_object::encode_line_set(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_line_set" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);
	
	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + 1;

	//int N;
	
	//N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
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
		cout << "any_combinatorial_object::encode_line_set "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_line_set "
				"done" << endl;
	}
}

void any_combinatorial_object::encode_points_and_lines(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_points_and_lines" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);

	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;

	nb_rows = nb_rows0 + 1;
	nb_cols = nb_cols0 + 1;

	//int N;

	//N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
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
		cout << "any_combinatorial_object::encode_points_and_lines "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_points_and_lines "
				"done" << endl;
	}
}


void any_combinatorial_object::encode_packing(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_packing" << endl;
	}
	int i, j;
	int f_vvv = (verbose_level >= 3);
	

	int nb_rows0, nb_cols0;
	int nb_rows, nb_cols;

	nb_rows0 = P->Subspaces->N_points;
	nb_cols0 = P->Subspaces->N_lines;

	nb_rows = nb_rows0 + SoS->nb_sets;
	nb_cols = nb_cols0 + 1;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
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
		cout << "any_combinatorial_object::encode_packing "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_packing "
				"done" << endl;
	}
}

void any_combinatorial_object::encode_large_set(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_large_set" << endl;
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

	Enc = NEW_OBJECT(encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);


	combinatorics::other_combinatorics::combinatorics_domain Combi;

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
		cout << "any_combinatorial_object::encode_large_set "
				"partition:" << endl;
		Enc->print_partition();
	}

	FREE_int(block);

	if (f_v) {
		cout << "any_combinatorial_object::encode_large_set "
				"done" << endl;
	}
}


void any_combinatorial_object::encode_incidence_geometry(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_incidence_geometry" << endl;
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_incidence_geometry v=" << v << endl;
		cout << "any_combinatorial_object::encode_incidence_geometry b=" << b << endl;
	}
	int i, a;
	int f_vvv = (verbose_level >= 3);


	int nb_rows, nb_cols;
	nb_rows = v;
	nb_cols = b;

	int N;

	N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	for (i = 0; i < sz; i++) {
		a = set[i];
		if (a >= nb_rows * nb_cols) {
			cout << "any_combinatorial_object::encode_incidence_geometry "
					"a >= nb_rows* nb_cols" << endl;
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
		cout << "any_combinatorial_object::encode_incidence_geometry "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_incidence_geometry "
				"done" << endl;
	}
}



void any_combinatorial_object::encode_multi_matrix(
		encoded_combinatorial_object *&Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::encode_multi_matrix" << endl;
	}
	int i, a;
	int f_vvv = (verbose_level >= 3);


	int nb_rows, nb_cols;
	nb_rows = m + n + max_val + 1;
	nb_cols = m + n + m * n;

	int N;

	N = nb_rows + nb_cols;

	Enc = NEW_OBJECT(encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	// encode row block lengths:
	for (i = 0; i < m; i++) {
		a = set[i];
		if (f_v) {
			cout << "any_combinatorial_object::encode_multi_matrix "
					"i=" << i << " a=" << a << endl;
		}
		if (a > max_val) {
			cout << "any_combinatorial_object::encode_multi_matrix "
					"a > max_val" << endl;
			exit(1);
		}
		Enc->set_incidence(i * nb_cols + i);
		Enc->set_incidence((m + n + a) * nb_cols + i);
	}

	// encode col block lengths:
	for (i = 0; i < n; i++) {
		a = set[m + i];
		if (f_v) {
			cout << "any_combinatorial_object::encode_multi_matrix "
					"i=" << i << " a=" << a << endl;
		}
		if (a > max_val) {
			cout << "any_combinatorial_object::encode_multi_matrix "
					"a > max_val" << endl;
			exit(1);
		}
		Enc->set_incidence((m + i) * nb_cols + m + i);
		Enc->set_incidence((m + n + a) * nb_cols + m + i);
	}

	int h, j;

	// encode matrix entries:
	for (h = 0; h < m * n; h++) {

		i = h / n;
		j = h % n;

		a = set[m + n + h];
		if (f_v) {
			cout << "any_combinatorial_object::encode_multi_matrix h=" << h << " a=" << a << endl;
		}
		if (a > max_val) {
			cout << "any_combinatorial_object::encode_multi_matrix a > max_val" << endl;
			exit(1);
		}
		Enc->set_incidence((i) * nb_cols + m + n + h);
		Enc->set_incidence((m + j) * nb_cols + m + n + h);
		Enc->set_incidence((m + n + a) * nb_cols + m + n + h);
	}



	Enc->partition[m - 1] = 0;
	Enc->partition[m + n - 1] = 0;
	for (i = 0; i <= max_val; i++) {
		Enc->partition[m + n + i] = 0;
	}
	//Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + m - 1] = 0;
	Enc->partition[nb_rows + m + n - 1] = 0;
	Enc->partition[N - 1] = 0;

	if (f_v) {
		cout << "any_combinatorial_object::encode_multi_matrix "
				"Enc=" << endl;
		Enc->print_incma();
	}

	if (f_vvv) {
		cout << "any_combinatorial_object::encode_multi_matrix "
				"partition:" << endl;
		Enc->print_partition();
	}
	if (f_v) {
		cout << "any_combinatorial_object::encode_multi_matrix "
				"done" << endl;
	}
}


void any_combinatorial_object::collinearity_graph(
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::collinearity_graph" << endl;
	}
	int i, j, c;

	encoded_combinatorial_object *Enc;

	encode_incidence_geometry(
			Enc,
			verbose_level);

	N = v;
	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

	for (c = 0; c < Enc->nb_cols; c++) {
		for (i = 0; i < Enc->nb_rows; i++) {
			if (!Enc->get_incidence_ij(i, c)) {
				continue;
			}
			for (j = i + 1; j < Enc->nb_rows; j++) {
				if (!Enc->get_incidence_ij(j, c)) {
					continue;
				}
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	if (f_v) {
		cout << "any_combinatorial_object::collinearity_graph done" << endl;
	}

}

void any_combinatorial_object::print()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "any_combinatorial_object::print" << endl;
	}
	encoded_combinatorial_object *Enc;
	//geometry::incidence_structure *Inc;
	//data_structures::partitionstack *Stack;

	encode_incma(Enc, verbose_level);

#if 0
	encode_incma_and_make_decomposition(
			Enc,
			Inc,
			Stack,
			verbose_level);
#endif

	Enc->print_incma();
	FREE_OBJECT(Enc);
	//FREE_OBJECT(Inc);
	//FREE_OBJECT(Stack);


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
		false /* f_init_incidence_structure */, 
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

		C.init(nb_pts_on_plane, nb_planes, false, 0);
		cout << "plane types are: ";
		C.print(true /* f_backwards*/);
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





}}}}




