/*
 * tactical_decomposition_domain.cpp
 *
 *  Created on: Feb 2, 2025
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {


tactical_decomposition_domain::tactical_decomposition_domain()
{
	Record_birth();


}



tactical_decomposition_domain::~tactical_decomposition_domain()
{
	Record_death();

}

void tactical_decomposition_domain::do_widor(
			std::string &widor_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tactical_decomposition_domain::do_widor" << endl;
	}


	other::data_structures::string_tools ST;


	std::string fname_out;

	fname_out = ST.without_extension(widor_fname) + ".tdo";

	other::orbiter_kernel_system::file_io Fio;

	{
		geo_parameter GP;
		tdo_scheme_synthetic G;
		ifstream f(widor_fname);
		ofstream g(fname_out);

		int i;

		for (i = 0; ; i++) {
			if (f.eof()) {
				break;
			}
			if (!GP.input(f)) {
				break;
			}
			if (f_v) {
				cout << "read decomposition " << i
					<< " v=" << GP.v << " b=" << GP.b << endl;
			}
			GP.convert_single_to_stack(verbose_level - 1);
			if (f_v) {
				cout << "after convert_single_to_stack" << endl;
			}

			std::string label;

			if (GP.label.length()) {
				label = GP.label;
			}
			else {
				label = std::to_string(i);;
			}
			GP.write(g, label);
			if (f_v) {
				cout << "after write" << endl;
			}
			GP.init_tdo_scheme(G, verbose_level - 1);
			if (f_v) {
				cout << "after init_tdo_scheme" << endl;
			}
			if (f_vv) {
				GP.print_schemes(G);
			}
		}
		g << "-1 " << i << endl;
	}
	if (f_v) {
		cout << "Written file " << fname_out
			<< " of size " << Fio.file_size(fname_out) << endl;
	}


}

void tactical_decomposition_domain::do_tdo_refinement(
		tactical_decompositions::tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::do_tdo_refinement" << endl;
	}

	tactical_decompositions::tdo_refinement *R;

	R = NEW_OBJECT(tactical_decompositions::tdo_refinement);

	R->init(Descr, verbose_level);
	R->main_loop(verbose_level);

	FREE_OBJECT(R);

	if (f_v) {
		cout << "tactical_decomposition_domain::do_tdo_refinement done" << endl;
	}
}

void tactical_decomposition_domain::do_tdo_print(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int cnt;
	int f_widor = false;
	int f_doit = false;

	if (f_v) {
		cout << "tactical_decomposition_domain::do_tdo_print" << endl;
	}

	cout << "opening file " << fname << " for reading" << endl;
	ifstream f(fname);



	tactical_decompositions::geo_parameter GP;
	tactical_decompositions::tdo_scheme_synthetic G;



	for (cnt = 0; ; cnt++) {
		if (f.eof()) {
			cout << "eof reached" << endl;
			break;
		}
		if (f_widor) {
			if (!GP.input(f)) {
				//cout << "GP.input returns false" << endl;
				break;
			}
		}
		else {
			if (!GP.input_mode_stack(f, verbose_level - 1)) {
				//cout << "GP.input_mode_stack returns false" << endl;
				break;
			}
		}

		f_doit = true;

		if (!f_doit) {
			continue;
		}
		//cout << "before convert_single_to_stack" << endl;
		//GP.convert_single_to_stack();
		//cout << "after convert_single_to_stack" << endl;
		//GP.write(g, label);
		if (f_vv) {
			cout << "tactical_decomposition_domain::do_tdo_print "
					"before init_tdo_scheme" << endl;
		}
		if (f_v) {
			cout << "tactical_decomposition_domain::do_tdo_print decomposition " << cnt << endl;
		}
		GP.init_tdo_scheme(G, verbose_level - 1);
		if (f_vv) {
			cout << "combinatorics_domain::do_tdo_print "
					"after init_tdo_scheme" << endl;
		}
		GP.print_schemes(G);

#if 0
		if (f_C) {
			GP.print_C_source();
		}
#endif
		if (true /* f_tex */) {
			GP.print_scheme_tex(cout, G, ROW_SCHEME);
			GP.print_scheme_tex(cout, G, COL_SCHEME);
		}
	}


	if (f_v) {
		cout << "tactical_decomposition_domain::do_tdo_print done" << endl;
	}
}

void tactical_decomposition_domain::convert_stack_to_tdo(
		std::string &stack_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	string fname;
	string fname_out;
	string label;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "tactical_decomposition_domain::convert_stack_to_tdo" << endl;
	}
	fname = stack_fname;
	ST.chop_off_extension(fname);
	fname_out = fname + ".tdo";

	if (f_v) {
		cout << "reading stack file " << stack_fname << endl;
	}
	{
		tactical_decompositions::geo_parameter GP;
		tactical_decompositions::tdo_scheme_synthetic G;
		ifstream f(stack_fname);
		ofstream g(fname_out);
		for (i = 0; ; i++) {
			if (f.eof()) {
				if (f_v) {
					cout << "end of file reached" << endl;
				}
				break;
			}
			if (!GP.input(f)) {
				if (f_v) {
					cout << "GP.input returns false" << endl;
				}
				break;
			}
			if (f_v) {
				cout << "read decomposition " << i
							<< " v=" << GP.v << " b=" << GP.b << endl;
			}
			GP.convert_single_to_stack(verbose_level - 1);
			if (f_v) {
				cout << "after convert_single_to_stack" << endl;
			}
			if (strlen(GP.label.c_str())) {
				GP.write(g, GP.label);
			}
			else {
				string s;

				s = std::to_string(i);
				GP.write(g, s);
			}

			if (f_v) {
				cout << "after write" << endl;
			}
			GP.init_tdo_scheme(G, verbose_level - 1);
			if (f_v) {
				cout << "after init_tdo_scheme" << endl;
			}
			if (f_vv) {
				GP.print_schemes(G);
			}
		}
		g << "-1 " << i << endl;
	}
	if (f_v) {
		other::orbiter_kernel_system::file_io Fio;
		cout << "written file " << fname_out
				<< " of size " << Fio.file_size(fname_out) << endl;
		cout << "tactical_decomposition_domain::convert_stack_to_tdo done" << endl;
	}
}

void tactical_decomposition_domain::do_parameters_maximal_arc(
		int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 2;
	int v[2], b[2], aij[4];
	int Q;
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "tactical_decomposition_domain::do_parameters_maximal_arc "
				"q=" << q << " r=" << r << endl;
	}

	Q = q * q;
	v[0] = q * (r - 1) + r;
	v[1] = Q + q * (2 - r) - r + 1;
	b[0] = Q - Q / r + q * 2 - q / r + 1;
	b[1] = Q / r + q / r - q;
	aij[0] = q + 1;
	aij[1] = 0;
	aij[2] = q - q / r + 1;
	aij[3] = q / r;
	fname = "max_arc_q" + std::to_string(q)
			+ "_r" + std::to_string(r)
			+ ".stack";

	Fio.write_decomposition_stack(
			fname, m, n, v, b, aij, verbose_level - 1);
}

void tactical_decomposition_domain::do_parameters_arc(
		int q, int s, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 1;
	int v[2], b[1], aij[2];
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "tactical_decomposition_domain::do_parameters_maximal_arc "
				"q=" << q << " s=" << s << " r=" << r << endl;
	}

	v[0] = s;
	v[1] = q * q + q + 1 - s;
	b[0] = q * q + q + 1;
	aij[0] = q + 1;
	aij[1] = q + 1;
	fname = "arc_q" + std::to_string(q)
			+ "_s" + std::to_string(s)
			+ "_r" + std::to_string(r)
			+ ".stack";

	Fio.write_decomposition_stack(
			fname, m, n, v, b, aij, verbose_level - 1);
}


void tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space_old(
		std::string &fname_base,
		geometry::projective_geometry::projective_space *P,
		long int *points, int nb_points,
		long int *lines, int nb_lines,
		std::vector<std::string> &file_names,
		int verbose_level)
// creates incidence_structure and data_structures::partitionstack objects
// called from quartic_curve_from_surface::TDO_decomposition
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space_old" << endl;
	}
	{

		geometry::other_geometry::incidence_structure *Inc;

		Inc = NEW_OBJECT(geometry::other_geometry::incidence_structure);

		Inc->init_projective_space(P, verbose_level);


		combinatorics::tactical_decompositions::decomposition *Decomp;

		Decomp = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);
		Decomp->init_incidence_structure(
				Inc,
				verbose_level);


		Decomp->Stack->split_cell_front_or_back_lint(
				points, nb_points, true /* f_front*/,
				verbose_level);

		Decomp->Stack->split_line_cell_front_or_back_lint(
				lines, nb_lines, true /* f_front*/,
				verbose_level);



		if (f_v) {
			cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space_old "
					"before Decomp->compute_TDO_safe_and_write_files" << endl;
		}
		Decomp->compute_TDO_safe_and_write_files(
				Decomp->N /* depth */,
				fname_base, file_names,
				verbose_level);
		if (f_v) {
			cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space_old "
					"after Decomp->compute_TDO_safe_and_write_files" << endl;
		}



		//FREE_OBJECT(Stack);
		FREE_OBJECT(Decomp);
		FREE_OBJECT(Inc);
	}
	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space_old done" << endl;
	}

}

combinatorics::tactical_decompositions::decomposition_scheme *tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space(
		geometry::projective_geometry::projective_space *P,
		long int *points, int nb_points,
		long int *lines, int nb_lines,
		int verbose_level)
// returns NULL if the space is too large
// called from
// surface_object_with_group::compute_tactical_decompositions
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space" << endl;
	}


	int nb_rows, nb_cols;

	nb_rows = P->Subspaces->N_points;
	nb_cols = P->Subspaces->N_lines;

	if (nb_rows + nb_cols > 50000) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space "
				"the space is too large" << endl;
		return NULL;
	}


	combinatorics::tactical_decompositions::decomposition *Decomposition;

	Decomposition = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);


	Decomposition->init_decomposition_of_projective_space(
			P,
			points, nb_points,
			lines, nb_lines,
			verbose_level);


	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space "
				"before Decomposition_scheme->compute_TDO" << endl;
	}
	Decomposition->compute_TDO(
			verbose_level - 1);
	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space "
				"after Decomposition_scheme->compute_TDO" << endl;
	}



	combinatorics::tactical_decompositions::decomposition_scheme *Decomposition_scheme;

	Decomposition_scheme = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition_scheme);

	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space "
				"before Decomposition_scheme->init_row_and_col_schemes" << endl;
	}
	Decomposition_scheme->init_row_and_col_schemes(
			Decomposition,
		verbose_level);
	if (f_v) {
		cout << "tactical_decomposition_domain::compute_TDO_decomposition_of_projective_space "
				"after Decomposition_scheme->init_row_and_col_schemes" << endl;
	}

	return Decomposition_scheme;

}

void tactical_decomposition_domain::refine_the_partition(
		int v, int k, int b, long int *Blocks_coded,
		int &b_reduced,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::refine_the_partition" << endl;
	}

	//combinatorics::other_combinatorics::combinatorics_domain Combi;
	combinatorics::design_theory::design_theory_global Design;

	//int N = k * b;
	int *M;
	//int i, j;
	int *R;

	R = NEW_int(v);

	Design.compute_incidence_matrix(v, b, k, Blocks_coded,
			M, verbose_level);

	{
		geometry::other_geometry::incidence_structure *Inc;


		Inc = NEW_OBJECT(geometry::other_geometry::incidence_structure);

		Inc->init_by_matrix(v, b, M, 0 /* verbose_level */);

		combinatorics::tactical_decompositions::decomposition *Decomposition;


		Decomposition = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);

		Decomposition->init_incidence_structure(
				Inc,
				verbose_level);

#if 0
		data_structures::partitionstack *Stack;
		Stack = NEW_OBJECT(data_structures::partitionstack);

		Stack->allocate_with_two_classes(v + b, v, b, 0 /* verbose_level */);
#endif


		while (true) {

			int ht0, ht1;

			ht0 = Decomposition->Stack->ht;

			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"before refine_column_partition_safe" << endl;
			}
			Decomposition->refine_column_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"before refine_row_partition_safe" << endl;
			}
			Decomposition->refine_row_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"after refine_row_partition_safe" << endl;
			}
			ht1 = Decomposition->Stack->ht;
			if (ht1 == ht0) {
				break;
			}
		}

		int f_labeled = true;

		Decomposition->print_partitioned(cout, f_labeled);
		Decomposition->get_and_print_decomposition_schemes();
		Decomposition->Stack->print_classes(cout);


		int f_print_subscripts = false;
		if (f_v) {
			cout << "Decomposition:\\\\" << endl;
			cout << "Row scheme:\\\\" << endl;
			Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
				f_print_subscripts);
			cout << "Column scheme:\\\\" << endl;
			Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
				f_print_subscripts);
		}

		other::data_structures::set_of_sets *Row_classes;
		other::data_structures::set_of_sets *Col_classes;

		Decomposition->Stack->get_row_classes(Row_classes, verbose_level);
		if (f_v) {
			cout << "Row classes:\\\\" << endl;
			Row_classes->print_table_tex(cout);
		}


		Decomposition->Stack->get_column_classes(Col_classes, verbose_level);
		if (f_v) {
			cout << "Col classes:\\\\" << endl;
			Col_classes->print_table_tex(cout);
		}

		if (Row_classes->nb_sets > 1) {
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"The row partition splits" << endl;
			}
		}

		if (Col_classes->nb_sets > 1) {
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"The col partition splits" << endl;
			}

			int idx;
			int j, a;

			idx = Col_classes->find_smallest_class();

			b_reduced = Col_classes->Set_size[idx];

			for (j = 0; j < b_reduced; j++) {
				a = Col_classes->Sets[idx][j];
				Blocks_coded[j] = Blocks_coded[a];
			}
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"reducing from " << b << " down to " << b_reduced << endl;
			}
		}
		else {
			if (f_v) {
				cout << "tactical_decomposition_domain::refine_the_partition "
						"The col partition does not split" << endl;
			}
			b_reduced = b;
		}


		FREE_OBJECT(Inc);
		FREE_OBJECT(Decomposition);
		FREE_OBJECT(Row_classes);
		FREE_OBJECT(Col_classes);
	}

	FREE_int(R);
	FREE_int(M);

	if (f_v) {
		cout << "tactical_decomposition_domain::refine_the_partition done" << endl;
	}

}

std::string tactical_decomposition_domain::stringify_row_scheme(
		std::string label_base,
		int nb_V, int nb_B,
		int *V, int *B, int *the_scheme,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_row_scheme" << endl;
	}

	string g;

	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	int extra_row_level = -1;
	int extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
	}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
	}
	g = label_base + " " + std::to_string(m + n) + " " + std::to_string(m) + " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g += std::to_string(f) + " ";
	}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g += std::to_string(f) + " ";
	}
	g += std::to_string(-1) + " ";
	col_level = 2;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0) {
			ii = 0;
		}
		else {
			ii = i + 1;
		}
		for (j = 0; j < nb_B; j++) {
			if (j == 0) {
				jj = 1;
			}
			else {
				jj = nb_V + j;
			}
			g += std::to_string(row_level) + " "
				+ std::to_string(ii) + " " + std::to_string(jj) + " "
				+ std::to_string(the_scheme[i * nb_B + j]) + " ";
		}
	}
	g += std::to_string(-1) + " ";
	g += std::to_string(row_level) + " "
		+ std::to_string(col_level) + " "
		+ std::to_string(lambda_level) + " "
		+ std::to_string(extra_row_level) + " "
		+ std::to_string(extra_col_level) + " ";

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_row_scheme done" << endl;
	}

	return g;
}

std::string tactical_decomposition_domain::stringify_col_scheme(
		std::string &label_base, int nb_V, int nb_B,
		int *V, int *B, int *the_scheme,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_col_scheme" << endl;
	}

	string g;
	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	int extra_row_level = -1;
	int extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
	}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
	}
	g = label_base + " " + std::to_string(m + n) + " " + std::to_string(m) + " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g += std::to_string(f) + " ";
	}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g += std::to_string(f) + " ";
	}
	g += std::to_string(-1) + " ";
	col_level = nb_V + nb_B;
	row_level = 2;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0) {
			ii = 0;
		}
		else {
			ii = i + 1;
		}
		for (j = 0; j < nb_B; j++) {
			if (j == 0) {
				jj = 1;
			}
			else {
				jj = nb_V + j;
			}
			g += std::to_string(col_level) + " "
				+ std::to_string(jj) + " " + std::to_string(ii) + " "
				+ std::to_string(the_scheme[i * nb_B + j]) + " ";
		}
	}
	g += std::to_string(-1) + " ";
	g += std::to_string(row_level) + " "
		+ std::to_string(col_level) + " "
		+ std::to_string(lambda_level) + " "
		+ std::to_string(extra_row_level) + " "
		+ std::to_string(extra_col_level) + " ";

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_col_scheme done" << endl;
	}

	return g;
}

std::string tactical_decomposition_domain::stringify_tdo_line_type(
		std::string &label_base, int m, int n,
		int nb_line_types, int *lines, int *multiplicities,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_tdo_line_type" << endl;
	}

	string g;
	int a, j, row_level, col_level, lambda_level, extra_row_level, extra_col_level;

	g = label_base + " " + std::to_string(m + n) + " " + std::to_string(m) + " ";
	a = multiplicities[0];
	for (j = 1; j < nb_line_types; j++) {
		g += std::to_string(m + a) + " ";
		a += multiplicities[j];
	}
	g += std::to_string(-1) + " ";
	col_level = 2 + nb_line_types - 1;
	row_level = 1;
	lambda_level = 2;
	extra_row_level = -1;
	extra_col_level = -1;
	for (j = 0; j < nb_line_types; j++) {
		g += std::to_string(col_level) + " " + std::to_string(1 + j)
				+ " " + std::to_string(0) + " " + std::to_string(lines[j]) + " ";
	}
	g += std::to_string(-1) + " ";
	g += std::to_string(row_level) + " "
		+ std::to_string(col_level) + " "
		+ std::to_string(lambda_level) + " "
		+ std::to_string(extra_row_level) + " "
		+ std::to_string(extra_col_level) + " ";

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_tdo_line_type done" << endl;
	}
	return g;
}

std::string tactical_decomposition_domain::stringify_td_scheme(
		std::string &label_base, int nb_V, int nb_B,
		int *V, int *B, int *the_scheme,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_td_scheme" << endl;
	}

	string g;
	int i, j, ii, jj, m, n, f, row_level, col_level, lambda_level, a, b, c;
	int extra_row_level = -1;
	int extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
	}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
	}
	g = label_base + " " + std::to_string(m + n) + " " + std::to_string(m) + " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g += std::to_string(f) + " ";
	}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g += std::to_string(f) + " ";
	}
	g += std::to_string(-1) + " ";
	col_level = nb_V + nb_B;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0) {
			ii = 0;
		}
		else {
			ii = i + 1;
		}
		for (j = 0; j < nb_B; j++) {
			if (j == 0) {
				jj = 1;
			}
			else {
				jj = nb_V + j;
			}
			g += std::to_string(row_level) + " "
				+ std::to_string(ii) + " " + std::to_string(jj) + " "
				+ std::to_string(the_scheme[i * nb_B + j]) + " ";
		}
	}
	for (i = 0; i < nb_V; i++) {
		if (i == 0) {
			ii = 0;
		}
		else {
			ii = i + 1;
		}
		for (j = 0; j < nb_B; j++) {
			if (j == 0) {
				jj = 1;
			}
			else {
				jj = nb_V + j;
			}
			a = V[i];
			b = B[j];
			c = (a * the_scheme[i * nb_B + j]) / b;
			if (b * c != (a * the_scheme[i * nb_B + j])) {
				cout << "tactical_decomposition_domain::stringify_td_scheme "
						"not tactical in (" << i << "," << j << ")-spot" << endl;
				exit(1);
			}
			g += std::to_string(col_level) + " "
				+ std::to_string(jj) + " " + std::to_string(ii) + " "
				+ std::to_string(c) + " ";
		}
	}
	g += std::to_string(-1) + " ";
	g += std::to_string(row_level) + " " + std::to_string(col_level) + " "
		+ std::to_string(lambda_level) + " "
		+ std::to_string(extra_row_level) + " "
		+ std::to_string(extra_col_level) + " ";

	if (f_v) {
		cout << "tactical_decomposition_domain::stringify_td_scheme done" << endl;
	}
	return g;
}



#if 0
// tdo_start.C
// Anton Betten
//
// started:  Dec 26 2006

#include "orbiter.h"

INT t0;

const BYTE *version = "tdo_start Jan 30 2008";

BYTE buf[BUFSIZE];

void print_usage();
int main(int argc, char **argv);
void create_all_linetypes(BYTE *label_base, INT m, INT verbose_level);
void write_tdo_line_type(ofstream &g, BYTE *label_base, INT m, INT n,
	INT nb_line_types, INT *lines, INT *multiplicities);
void write_row_scheme(ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme);
void write_col_scheme(ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme);
void write_td_scheme(ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme);

void print_usage()
{
	cout << "usage: tdo_start.out [options] <tdo_file>\n";
	cout << "where options can be:\n";
	cout << "-v <n>" << endl;
	cout << "  verbose level <n>" << endl;
	cout << "-conf <m> <n> <r> <k>" << endl;
	cout << "  create a TDO for a configuration m_r n_k" << endl;
	cout << "-linearspace <n> <i_1> <a_1> <i_2> <a_2> ... -1 <file_name>" << endl;
	cout << "  create TDO file for linear spaces on n points" << endl;
	cout << "  with a_j (> 0) lines of size i_j." << endl;
	cout << "  Note that \\sum_{j}a_j{i_j \\choose 2} = {n \\choose 2} " << endl;
	cout << "  is required. The output is written into the specified file," << endl;
	cout << "  as <file_name>.tdo" << endl;
	cout << "-all <n>" << endl;
	cout << "  Create a TDO-file with all possible line cases for" << endl;
	cout << "  linear spaces on <n> points" << endl;
	cout << "" << endl;
	cout << "-rowscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <r_{1,1}> <r_{1,2}> ... <r_{m,n}>" << endl;
	cout << "-colscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <k_{1,1}> <k_{1,2}> ... <k_{m,n}>" << endl;
	cout << "-tdscheme <m> <n> <V_1> ... <V_m> <B_1> ... <B_n> <r_{1,1}> <r_{1,2}> ... <r_{m,n}>" << endl;
}

int main(int argc, char **argv)
{
	BYTE fname_out[1000];
	t0 = os_ticks();
	INT verbose_level = 0;
	INT f_conf = FALSE;
	INT f_all = FALSE;
	INT f_linearspace = FALSE;
	INT f_rowscheme = FALSE;
	INT f_colscheme = FALSE;
	INT f_tdscheme = FALSE;
	INT nb_V, nb_B;
	INT *V, *B, *the_scheme;
	INT row_level, col_level, lambda_level, extra_row_level, extra_col_level;
	INT i, m, n, r, k, a, m2, a2, ii, jj;
	INT nb_lines;
	INT line_size[1000];
	INT line_multiplicity[1000];
	BYTE *label_base;
	BYTE label[1000];

	cout << version << endl;
	if (argc <= 1) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		if (strcmp(argv[i], "-conf") == 0) {
			f_conf = TRUE;
			m = atoi(argv[++i]);
			n = atoi(argv[++i]);
			r = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-conf " << m << " " << n << " " << r << " " << k << endl;
		}
		if (strcmp(argv[i], "-linearspace") == 0) {
			f_linearspace = TRUE;
			m = atoi(argv[++i]);
			n = 0;
			for (nb_lines = 0; ; nb_lines++) {
				a = atoi(argv[++i]);
				if (a == -1)
					break;
				line_size[nb_lines] = a;
				a = atoi(argv[++i]);
				line_multiplicity[nb_lines] = a;
				n += a;
				}
			cout << "-linearspace " << m << " " << n << endl;
		}
		if (strcmp(argv[i], "-all") == 0) {
			f_all = TRUE;
			m = atoi(argv[++i]);
			cout << "-all " << m << endl;
		}
		if (strcmp(argv[i], "-rowscheme") == 0) {
			f_rowscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			V = new INT[nb_V];
			B = new INT[nb_B];
			the_scheme = new INT[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
		}
		if (strcmp(argv[i], "-colscheme") == 0) {
			f_colscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			cout << "-colscheme " << nb_V << " " << nb_B << endl;
			V = new INT[nb_V];
			B = new INT[nb_B];
			the_scheme = new INT[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			cout << "V:" << endl;
			for (ii = 0; ii < nb_V; ii++) {
				cout << V[ii] << " ";
				}
			cout << endl;
			cout << "B:" << endl;
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				cout << B[jj] << " ";
				}
			cout << endl;
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
			cout << "scheme:" << endl;
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					cout << setw(3) << the_scheme[ii * nb_B + jj];
					}
				cout << endl;
				}
			}
		if (strcmp(argv[i], "-tdscheme") == 0) {
			f_tdscheme = TRUE;
			nb_V = atoi(argv[++i]);
			nb_B = atoi(argv[++i]);
			V = new INT[nb_V];
			B = new INT[nb_B];
			the_scheme = new INT[nb_V * nb_B];
			for (ii = 0; ii < nb_V; ii++) {
				V[ii] = atoi(argv[++i]);
				}
			for (jj = 0; jj < nb_B; jj++) {
				B[jj] = atoi(argv[++i]);
				}
			for (ii = 0; ii < nb_V; ii++) {
				for (jj = 0; jj < nb_B; jj++) {
					the_scheme[ii * nb_B + jj] = atoi(argv[++i]);
					}
				}
		}
	}
	label_base = argv[argc - 1];
	sprintf(fname_out, "%s.tdo", label_base);
	{

	if (f_linearspace) {
		m2 = binomial2(m);
		for (i = 0; i < nb_lines; i++) {
			a = line_size[i];
			a2 = binomial2(a);
			a2 *= line_multiplicity[i];
			m2 -= a2;
			}
		if (m2 < 0) {
			cout << "error in the line type" << endl;
			exit(1);
			}
		if (m2 > 0) {
			cout << "need " << m2 << " additional 2-lines" << endl;
			exit(1);
			}
		}

	if (f_conf) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		sprintf(label, "%s.0", label_base);
		g << label << " " << m + n << " " << m << " " << -1 << " "
			<< 2 << " " << 0 << " " << 1 << " " << r << " "
			<< 2 << " " << 1 << " " << 0 << " " << k << " "
			<< -1 << " ";
		row_level = col_level = lambda_level = 2;
		extra_row_level = -1;
		extra_col_level = -1;
		g << row_level << " "
			<< col_level << " "
			<< lambda_level << " "
			<< extra_row_level << " "
			<< extra_col_level << " "
			<< endl;
		g << -1 << endl;
		}
	if (f_linearspace) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_tdo_line_type(g, label_base, m, n,
			nb_lines, line_size, line_multiplicity);
		g << -1 << endl;
		}

	if (f_all) {
		create_all_linetypes(label_base, m, verbose_level);
		}
	if (f_rowscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_row_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}
	if (f_colscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_col_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}
	if (f_tdscheme) {
		cout << "opening file " << fname_out << " for writing" << endl;
		ofstream g(fname_out);
		write_td_scheme(g, label_base, nb_V, nb_B, V, B, the_scheme);
		g << -1 << endl;
		}

	}
	cout << "time: ";
	time_check(cout, t0);
	cout << endl;
}

void create_all_linetypes(BYTE *label_base, INT m, INT verbose_level)
{
	BYTE fname[1000];
	BYTE label[1000];
	INT nb_line_types;
	INT *lines;
	INT *multiplicities;
	INT *types;
	INT nb_eqns = 2, nb_vars = m;
	INT nb_sol, m2, i, /* k, */ j, a, n, Nb_sol;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);

	cout << "create all line types of linear spaces on " << m << " points" << endl;
	lines = new INT[m];
	multiplicities = new INT[m];

	diophant D;

	D.open(nb_eqns, nb_vars);

	for (i = 2; i <= m; i++) {
		//cout << "i = " << i << " : " << 0 * nb_vars + i - 2 << " : " << binomial2(i) << endl;
		D.A[0 * nb_vars + nb_vars - i] = binomial2(i);
		}
	D.A[0 * nb_vars + m - 1] = 0;
	for (i = 0; i < nb_vars; i++) {
		D.A[1 * nb_vars + i] = 1;
		}
	m2 = binomial2(m);
	D.RHS[0] = m2;
	D.RHS[1] = m2;
	D.type[0] = t_EQ;
	D.type[1] = t_EQ;
	D.sum = m2;

	if (f_v) {
		D.print();
		}
	nb_sol = 0;
	if (D.solve_first_betten(f_vvv)) {

		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D.solve_next_betten(f_vvv))
				break;
			}
		}
	if (f_v) {
		cout << "found " << nb_sol << " line types" << endl;
		}
	Nb_sol = nb_sol;
	types = new INT[Nb_sol * nb_vars];
	nb_sol = 0;
	if (D.solve_first_betten(f_vvv)) {

		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			for (i = 0; i < nb_vars; i++) {
				types[nb_sol * nb_vars + i] = D.x[i];
				}

			nb_sol++;
			if (!D.solve_next_betten(f_vvv))
				break;
			}
		}

	//diophant_close(D);

	sprintf(fname, "%s.tdo", label_base);
	{
		cout << "opening file " << fname << " for writing" << endl;
		ofstream g(fname);


		for (i = 0; i < Nb_sol; i++) {
			//k = Nb_sol - 1 - i;
			nb_line_types = 0;
			for (j = 0; j < nb_vars - 1; j++) {
				a = types[i * nb_vars + j];
				if (a == 0)
					continue;
				lines[nb_line_types] = m - j;
				multiplicities[nb_line_types] = a;
				nb_line_types++;
				}
			n = m2 - types[i * nb_vars + nb_vars - 1];
			sprintf(label, "%s.%ld", label_base, i + 1);
			write_tdo_line_type(g, label, m, n,
				nb_line_types, lines, multiplicities);
			}
		g << -1 << endl;
	}


	delete [] lines;
	delete [] multiplicities;
	delete [] types;
}

void write_tdo_line_type(ofstream &g, BYTE *label_base, INT m, INT n,
	INT nb_line_types, INT *lines, INT *multiplicities)
{
	INT a, j, row_level, col_level, lambda_level, extra_row_level, extra_col_level;

	g << label_base << " " << m + n << " " << m << " ";
	a = multiplicities[0];
	for (j = 1; j < nb_line_types; j++) {
		g << m + a << " ";
		a += multiplicities[j];
		}
	g << -1 << " ";
	col_level = 2 + nb_line_types - 1;
	row_level = 1;
	lambda_level = 2;
	extra_row_level = -1;
	extra_col_level = -1;
	for (j = 0; j < nb_line_types; j++) {
		g << col_level << " " << 1 + j << " " << 0 << " " << lines[j] << " ";
		}
	g << -1 << " ";
	g << row_level << " "
		<< col_level << " "
		<< lambda_level << " "
		<< extra_row_level << " "
		<< extra_col_level << " "
		<< endl;
}

void write_row_scheme(
		ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme)
{
	INT i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	INT extra_row_level = -1;
	INT extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = 2;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else
				jj = nb_V + j;
			g << row_level << " "
				<< ii << " " << jj << " "
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " "
		<< col_level << " "
		<< lambda_level << " "
		<< extra_row_level << " "
		<< extra_col_level << " "
		<< endl;
}

void write_col_scheme(ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme)
{
	INT i, j, ii, jj, m, n, f, row_level, col_level, lambda_level;
	INT extra_row_level = -1;
	INT extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = nb_V + nb_B;
	row_level = 2;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else
				jj = nb_V + j;
			g << col_level << " "
				<< jj << " " << ii << " "
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " "
		<< col_level << " "
		<< lambda_level << " "
		<< extra_row_level << " "
		<< extra_col_level << " "
		<< endl;
}

void write_td_scheme(ofstream &g, BYTE *label_base, INT nb_V, INT nb_B,
	INT *V, INT *B, INT *the_scheme)
{
	INT i, j, ii, jj, m, n, f, row_level, col_level, lambda_level, a, b, c;
	INT extra_row_level = -1;
	INT extra_col_level = -1;

	m = 0;
	for (i = 0; i < nb_V; i++) {
		m += V[i];
		}
	n = 0;
	for (j = 0; j < nb_B; j++) {
		n += B[j];
		}
	g << label_base << " " << m + n << " " << m << " ";
	f = 0;
	for (i = 1; i < nb_V; i++) {
		f += V[i - 1];
		g << f << " ";
		}
	f = m;
	for (j = 1; j < nb_B; j++) {
		f += B[j - 1];
		g << f << " ";
		}
	g << -1 << " ";
	col_level = nb_V + nb_B;
	row_level = nb_V + nb_B;
	lambda_level = 2;
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else
				jj = nb_V + j;
			g << row_level << " "
				<< ii << " " << jj << " "
				<< the_scheme[i * nb_B + j] << " ";
			}
		}
	for (i = 0; i < nb_V; i++) {
		if (i == 0)
			ii = 0;
		else
			ii = i + 1;
		for (j = 0; j < nb_B; j++) {
			if (j == 0)
				jj = 1;
			else
				jj = nb_V + j;
			a = V[i];
			b = B[j];
			c = (a * the_scheme[i * nb_B + j]) / b;
			if (b * c != (a * the_scheme[i * nb_B + j])) {
				cout << "not tactical in (" << i << "," << j << ")-spot" << endl;
				exit(1);
				}
			g << col_level << " "
				<< jj << " " << ii << " "
				<< c << " ";
			}
		}
	g << -1 << " ";
	g << row_level << " " << col_level << " "
		<< lambda_level << " "
		<< extra_row_level << " "
		<< extra_col_level << " "
		<< endl;
}

#endif

}}}}

