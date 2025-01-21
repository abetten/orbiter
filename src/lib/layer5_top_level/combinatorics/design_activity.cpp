/*
 * design_activity.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_activity::design_activity()
{
	Record_birth();
	Descr = NULL;

}

design_activity::~design_activity()
{
	Record_death();

}

void design_activity::perform_activity(
		design_activity_description *Descr,
		design_create *DC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::perform_activity" << endl;
	}

	design_activity::Descr = Descr;

	if (Descr->f_load_table) {
		do_load_table(
				DC,
				Descr->load_table_label,
				Descr->load_table_group,
				Descr->load_table_H_label,
				Descr->load_table_H_group_order,
				Descr->load_table_H_gens,
				Descr->load_table_selected_orbit_length,
				verbose_level);
	}
	else if (Descr->f_canonical_form) {
		do_canonical_form(Descr->Canonical_form_Descr,
				verbose_level);
	}
	else if (Descr->f_extract_solutions_by_index_csv) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_csv" << endl;
		}

		do_extract_solutions_by_index(
				DC,
				Descr->extract_solutions_by_index_label,
				Descr->extract_solutions_by_index_group,
				Descr->extract_solutions_by_index_fname_solutions_in,
				Descr->extract_solutions_by_index_col_label,
				Descr->extract_solutions_by_index_fname_solutions_out,
				Descr->extract_solutions_by_index_prefix,
				true /* f_csv */,
				verbose_level);

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_csv done" << endl;
		}
	}
	else if (Descr->f_extract_solutions_by_index_txt) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_txt" << endl;
		}

		do_extract_solutions_by_index(
				DC,
				Descr->extract_solutions_by_index_label,
				Descr->extract_solutions_by_index_group,
				Descr->extract_solutions_by_index_fname_solutions_in,
				Descr->extract_solutions_by_index_col_label,
				Descr->extract_solutions_by_index_fname_solutions_out,
				Descr->extract_solutions_by_index_prefix,
				false /* f_csv */,
				verbose_level);

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_txt done" << endl;
		}
	}
	else if (Descr->f_export_flags) {
		if (f_v) {
			cout << "design_activity::perform_activity f_export_flags" << endl;
		}
		do_export_flags(
				DC,
				verbose_level);

	}
	else if (Descr->f_export_incidence_matrix) {
		if (f_v) {
			cout << "design_activity::perform_activity f_export_incidence_matrix" << endl;
		}
		do_export_incidence_matrix_csv(DC, verbose_level);
	}
	else if (Descr->f_export_incidence_matrix_latex) {
		if (f_v) {
			cout << "design_activity::perform_activity f_export_incidence_matrix_latex" << endl;
		}

		other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description;

		Draw_incidence_structure_description = Get_draw_incidence_structure_options(Descr->export_incidence_matrix_latex_draw_options);

		do_export_incidence_matrix_latex(DC, Draw_incidence_structure_description, verbose_level);
	}
	else if (Descr->f_intersection_matrix) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_intersection_matrix" << endl;
		}

		//int f_save = false;

		do_intersection_matrix(
				DC,
				Descr->f_save,
				verbose_level);
	}
	else if (Descr->f_export_blocks) {
		if (f_v) {
			cout << "design_activity::perform_activity export_blocks" << endl;
		}
		do_export_blocks(
				DC,
				verbose_level);
	}
	else if (Descr->f_row_sums) {
		if (f_v) {
			cout << "design_activity::perform_activity row_sums" << endl;
		}
		do_row_sums(
				DC,
				verbose_level);
	}
	else if (Descr->f_tactical_decomposition) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_tactical_decomposition" << endl;
		}
		do_tactical_decomposition(
				DC,
				verbose_level);
	}
	else if (Descr->f_orbits_on_blocks) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_orbits_on_blocks" << endl;
		}
		int *Pair_orbits;
		int degree;

		if (Descr->orbits_on_blocks_sz != 2) {
			cout << "Descr->orbits_on_blocks_sz != 2" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "design_activity::perform_activity "
					"before do_pair_orbits_on_blocks" << endl;
		}
		do_pair_orbits_on_blocks(
				DC,
				Descr->orbits_on_blocks_control,
				Pair_orbits, degree,
				verbose_level);
		if (f_v) {
			cout << "design_activity::perform_activity "
					"after do_pair_orbits_on_blocks" << endl;
		}
	}
	else if (Descr->f_one_point_extension) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_one_point_extension" << endl;
		}
		int *Pair_orbits;
		int degree;

		if (f_v) {
			cout << "design_activity::perform_activity "
					"before do_pair_orbits_on_blocks" << endl;
		}
		do_pair_orbits_on_blocks(
				DC,
				Descr->one_point_extension_control,
				Pair_orbits, degree,
				verbose_level);
		if (f_v) {
			cout << "design_activity::perform_activity "
					"after do_pair_orbits_on_blocks" << endl;
		}

		int orbit_idx;

		orbit_idx = Descr->one_point_extension_pair_orbit_idx;

		int *Adj;
		int n;
		int i, j, u, v;

		n = 1 + DC->v + DC->b;
		Adj = NEW_int(n * n);
		Int_vec_zero(Adj, n * n);
		for (i = 0; i < DC->v; i++) {
			u = 0;
			v = 1 + i;
			Adj[u * n + v] = Adj[v * n + u] = 1;
		}
		for (i = 0; i < DC->v; i++) {
			for (j = 0; j < DC->b; j++) {
				u = 1 + i;
				v = 1 + DC->v + j;
				Adj[u * n + v] = Adj[v * n + u] = DC->incma[i * DC->b + j];
			}
		}
		for (i = 0; i < DC->b; i++) {
			u = 1 + DC->v + i;
			for (j = i + 1; j < DC->b; j++) {
				v = 1 + DC->v + j;
				if (Pair_orbits[i * degree + j] == orbit_idx) {
					Adj[u * n + v] = Adj[v * n + u] = 1;
				}
			}
		}


		string fname;

		fname = DC->label_txt + "_one_pt_ext_orb" + std::to_string(orbit_idx) + ".csv";
		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, Adj, n, n);

		cout << "design_activity::perform_activity "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	}

	if (f_v) {
		cout << "design_activity::perform_activity done" << endl;
	}

}


void design_activity::do_extract_solutions_by_index(
		design_create *DC,
		std::string &label,
		std::string &group_label,
		std::string &fname_in,
		std::string &col_label,
		std::string &fname_out,
		std::string &prefix_text,
		int f_csv_format,
		int verbose_level)
// does not need DC. This should be an activity for the design_table
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;

	groups::any_group *AG;

	AG = Get_any_group(group_label);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index "
				"before Combi.load_design_table" << endl;
	}

	Combi.load_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index "
				"after Combi.load_design_table" << endl;
	}

	int *prefix;
	int prefix_sz;

	Int_vec_scan(prefix_text, prefix, prefix_sz);

	other::orbiter_kernel_system::file_io Fio;
	int *Sol_idx;
	int nb_sol;
	int sol_width = 0;

	if (f_csv_format) {

#if 0
		data_structures::string_tools ST;

		//int *Sol_idx_1;
		int i, j;


		std::string *Column;
		int len;

		if (f_v) {
			cout << "design_activity::do_extract_solutions_by_index "
					"before Fio.Csv_file_support->read_column_of_strings" << endl;
		}
		Fio.Csv_file_support->read_column_of_strings(
				fname_in, col_label,
				Column, len,
				verbose_level);

		//Fio.Csv_file_support->int_matrix_read_csv(
		//		fname_in, Sol_idx_1, nb_sol, sol_width, verbose_level);
		if (f_v) {
			cout << "design_activity::do_extract_solutions_by_index "
					"after Fio.Csv_file_support->read_column_of_strings" << endl;
		}

		nb_sol = len;

		if (nb_sol == 0) {
			sol_width = 0;
		}
		else {
			for (i = 0; i < nb_sol; i++) {
				std::string s;

				ST.drop_quotes(
						Column[i], s);
				Column[i] = s;
			}
			int *data;
			Int_vec_scan(Column[0], data, sol_width);
			FREE_int(data);
		}

		delete [] Column;

#endif

		int *Sol_idx_1;

		Fio.Csv_file_support->read_table_of_strings_as_matrix(
				fname_in, col_label,
				Sol_idx_1, nb_sol, sol_width, verbose_level - 1);


		int i;

		if (f_v) {
			cout << "design_activity::do_extract_solutions_by_index "
					"Sol_idx_1 has size " << nb_sol << " x " << sol_width << endl;
		}
		Sol_idx = NEW_int(nb_sol * (prefix_sz + sol_width));
		for (i = 0; i < nb_sol; i++) {
			Int_vec_copy(prefix, Sol_idx + i * (prefix_sz + sol_width), prefix_sz);
#if 0
			for (j = 0; j < prefix_sz; j++) {
				Sol_idx[i * (prefix_sz + sol_width) + j] = prefix[j];
			}
#endif
			//int *data;
			//Int_vec_scan(Column[i], data, sol_width);
			Int_vec_copy(
					Sol_idx_1 + i * sol_width,
					Sol_idx + i * (prefix_sz + sol_width) + prefix_sz,
					sol_width);
			//FREE_int(data);
#if 0
			for (j = 0; j < sol_width; j++) {
				Sol_idx[i * (prefix_sz + sol_width) + prefix_sz + j] = Sol_idx_1[i * sol_width + j];
			}
#endif
		}
		FREE_int(Sol_idx_1);
		sol_width += prefix_sz;
	}
	else {
		other::data_structures::set_of_sets *SoS;
		int underlying_set_size = 0;
		int i, j;

		SoS = NEW_OBJECT(other::data_structures::set_of_sets);
		SoS->init_from_orbiter_file(underlying_set_size,
				fname_in, verbose_level);
		nb_sol = SoS->nb_sets;

		if (nb_sol) {
			if (!SoS->has_constant_size_property()) {
				cout << "design_activity::do_extract_solutions_by_index "
						"the sets have different sizes" << endl;
				exit(1);
			}
			sol_width = SoS->Set_size[0];

			Sol_idx = NEW_int(nb_sol * (prefix_sz + sol_width));
			for (i = 0; i < nb_sol; i++) {
				Int_vec_copy(prefix, Sol_idx + i * (prefix_sz + sol_width), prefix_sz);
#if 0
				for (j = 0; j < prefix_sz; j++) {
					Sol_idx[i * (prefix_sz + sol_width) + j] = prefix[j];
				}
#endif
				for (j = 0; j < sol_width; j++) {
					Sol_idx[i * (prefix_sz + sol_width) + prefix_sz + j] = SoS->Sets[i][j];
				}
			}
			sol_width += prefix_sz;
		}
		else {
			Sol_idx = NEW_int(1);
		}
		FREE_OBJECT(SoS);
	}


	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index "
				"before T->extract_solutions_by_index" << endl;
	}

	T->extract_solutions_by_index(
			nb_sol, sol_width, Sol_idx,
			fname_out,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index "
				"after T->extract_solutions_by_index" << endl;
	}



	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index done" << endl;
	}
}



void design_activity::do_create_table(
		design_create *DC,
		std::string &label,
		std::string &group_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_create_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;


	groups::any_group *AG;

	AG = Get_any_group(group_label);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"after Combi.create_design_table" << endl;
	}

	if (f_v) {
		cout << "design_activity::do_create_table done" << endl;
	}
}


void design_activity::do_load_table(
		design_create *DC,
		std::string &label,
		std::string &group_label,
		std::string &H_label,
		std::string &H_go_text,
		std::string &H_generators_data,
		int selected_orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_load_table" << endl;
	}

	groups::any_group *AG;

	AG = Get_any_group(group_label);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"before Combi.load_design_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;

	Combi.load_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"after Combi.load_design_table" << endl;
	}


	large_set_classify *LS;

	LS = NEW_OBJECT(large_set_classify);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"before LS->init" << endl;
	}
	LS->init(DC,
			T,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_create_table "
				"after LS->init" << endl;
	}



	groups::strong_generators *H_gens;
	H_gens = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "design_activity::do_load_table "
				"before H_gens->init_from_data_with_go" << endl;
	}
	H_gens->init_from_data_with_go(
			DC->A, H_generators_data,
			H_go_text,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table "
				"after H_gens->init_from_data_with_go" << endl;
	}


#if 0
	large_set_was *LSW;

	LSW = NEW_OBJECT(large_set_was);


	if (f_v) {
		cout << "design_activity::do_load_table "
				"before LSW->init" << endl;
	}
	LSW->init(LS,
			H_gens, H_label,
			selected_orbit_length,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table "
				"after LSW->init" << endl;
	}

#endif

	if (f_v) {
		cout << "design_activity::do_load_table done" << endl;
	}
}

void design_activity::do_canonical_form(
		combinatorics::canonical_form_classification::classification_of_objects_description
			*Canonical_form_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}


#if 0
	combinatorics::classification_of_objects *OC;

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}

	OC = NEW_OBJECT(classification_of_objects);

	if (f_v) {
		cout << "design_activity::do_canonical_form "
				"before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_Descr,
			false,
			NULL,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_canonical_form "
				"after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);


#endif

	if (f_v) {
		cout << "design_activity::do_canonical_form done" << endl;
	}

}


#if 0
void design_activity::do_export_inc(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_inc" << endl;
	}

	string fname;

	fname = DC->label_txt + "_inc.txt";

	if (f_v) {
		cout << "design_activity::do_export_inc "
				"fname=" << fname << endl;
	}


	{
		ofstream ost(fname);

		int h;
		ost << DC->v << " " << DC->b << " " << DC->nb_inc << endl;
		for (h = 0; h < DC->v * DC->b; h++) {
			if (DC->incma[h]) {
				ost << h << " ";
			}
		}
		ost << endl;
		ost << "-1" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	//8 8 24
	//0 1 2 8 11 12 16 21 22 25 27 29 33 36 39 42 44 46 50 53 55 59 62 63
	//-1 1
	//48



	if (f_v) {
		cout << "design_activity::do_export_inc done" << endl;
	}
}
#endif

void design_activity::do_export_flags(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_flags" << endl;
	}

	string fname;

	fname = DC->label_txt + "_flags.txt";

	if (f_v) {
		cout << "design_activity::do_export_flags "
				"fname=" << fname << endl;
	}


	{
		ofstream ost(fname);

		int h;
		ost << DC->v << " " << DC->b << " " << DC->nb_inc << endl;
		for (h = 0; h < DC->v * DC->b; h++) {
			if (DC->incma[h]) {
				ost << h << " ";
			}
		}
		ost << endl;
		ost << "-1" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	//8 8 24
	//0 1 2 8 11 12 16 21 22 25 27 29 33 36 39 42 44 46 50 53 55 59 62 63
	//-1 1
	//48



	if (f_v) {
		cout << "design_activity::do_export_flags done" << endl;
	}
}




void design_activity::do_export_incidence_matrix_csv(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_csv" << endl;
	}

	string fname;

	fname = DC->label_txt + "_incma.csv";

	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_csv "
				"fname=" << fname << endl;
	}

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->int_matrix_write_csv(
			fname, DC->incma, DC->v, DC->b);
	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_csv done" << endl;
	}
}

void design_activity::do_export_incidence_matrix_latex(
		design_create *DC,
		other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_latex" << endl;
	}

	string fname;

	fname = DC->label_txt + "_incma.tex";

	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_latex "
				"fname=" << fname << endl;
	}

	int nb_rows, nb_cols;

	nb_rows = DC->v;
	nb_cols = DC->b;

	combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

	Enc = NEW_OBJECT(combinatorics::canonical_form_classification::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	int i, j, f;

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < DC->b; j++) {
			f = i * DC->b + j;
			if (DC->incma[f]) {
				Enc->set_incidence(f);
			}
		}
	}

	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + nb_cols - 1] = 0;


	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_latex "
				"partition:" << endl;
		Enc->print_partition();
	}

	other::orbiter_kernel_system::file_io Fio;


	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(ost);


		Enc->latex_incma(
				ost,
				Draw_incidence_structure_description,
				verbose_level);

		ost << endl;

		ost << "\\bigskip" << endl;

		ost << endl;


		Enc->latex_incma_as_01_matrix(
				ost,
				verbose_level);

		ost << endl;

		ost << "\\bigskip" << endl;

		ost << endl;




		ost << "\\noindent Blocks: \\\\" << endl;

		Enc->latex_set_system_by_columns(
				ost,
				verbose_level);

		L.foot(ost);
	}



	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "design_activity::do_export_incidence_matrix_latex done" << endl;
	}
}



void design_activity::do_intersection_matrix(
		design_create *DC,
		int f_save,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_intersection_matrix" << endl;
	}


	if (!DC->f_has_incma) {
		cout << "design_activity::do_intersection_matrix "
				"the incidence matrix of the design is not available" << endl;
		exit(1);
	}

	int *AAt;
	int i, j, h, cnt;

	AAt = NEW_int(DC->v * DC->v);
	for (i = 0; i < DC->v; i++) {
		for (j = 0; j < DC->v; j++) {
			cnt = 0;
			for (h = 0; h < DC->b; h++) {
				if (DC->incma[i * DC->b + h] && DC->incma[j * DC->b + h]) {
					cnt++;
				}
			}
			AAt[i * DC->v + j] = cnt;
		}

	}

	algebra::basic_algebra::algebra_global Algebra;
	int coeff_I, coeff_J;

	if (Algebra.is_lc_of_I_and_J(
			AAt, DC->v, coeff_I, coeff_J, 0 /* verbose_level*/)) {
		cout << "Is a linear combination of I and J with coefficients "
				"coeff(I)=" << coeff_I << " and coeff(J-I)=" << coeff_J << endl;
	}
	else {
		cout << "Is *not* a linear combination of I and J" << endl;

	}

	if (f_save) {

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = DC->label_txt + "_AAt.csv";

		if (f_v) {
			cout << "design_activity::do_intersection_matrix "
					"fname=" << fname << endl;
		}

		{
			ofstream ost(fname);

			Fio.Csv_file_support->int_matrix_write_csv(
					fname, AAt, DC->v, DC->v);

		}

		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
	}



	if (f_v) {
		cout << "design_activity::do_intersection_matrix done" << endl;
	}
}


void design_activity::do_export_blocks(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_blocks" << endl;
	}

	string fname;

	fname = DC->label_txt + "_blocks.csv";

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int v = DC->v;
	int k = DC->k;
	int b = DC->b;


	other::orbiter_kernel_system::file_io Fio;


	#if 0

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, DC->set, 1, b);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
#endif


	fname = DC->label_txt + "_blocks.csv";

	int *Blocks;


	if (DC->f_has_set) {
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"before Combi.compute_blocks_from_coding" << endl;
		}
		Combi.compute_blocks_from_coding(
				v, b, k, DC->set, Blocks, verbose_level);
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"after Combi.compute_blocks_from_coding" << endl;
		}
	}
	else if (DC->f_has_incma) {
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"before Combi.compute_blocks_from_incma" << endl;
		}
		Combi.compute_blocks_from_incma(
				v, b, k, DC->incma,
					Blocks, verbose_level);
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"after Combi.compute_blocks_from_incma" << endl;
		}
	}
	else {
		cout << "design_activity::do_export_blocks "
				"we neither have a set nor an incma" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "design_activity::do_export_blocks "
				"b = " << b << endl;
		cout << "design_activity::do_export_blocks "
				"k = " << k << endl;
	}

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Blocks, b, k);
#endif

	string *Table;
	string headings;
	int nb_rows = b;
	int nb_cols = 2;
	int i;

	Table = new string [nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Int_vec_stringify(Blocks + i * k, k) + "\"";
	}
	headings = "row,block";



	Fio.Csv_file_support->write_table_of_strings(
			fname,
				nb_rows, nb_cols, Table,
				headings,
				verbose_level);

	if (f_v) {
		cout << "design_activity::do_export_blocks Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	delete [] Table;

	FREE_int(Blocks);


	if (f_v) {
		cout << "design_activity::do_export_blocks done" << endl;
	}
}

void design_activity::do_row_sums(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_row_sums" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;


	int i, j;
	int *R;

	R = NEW_int(DC->v);

	for (i = 0; i < DC->v; i++) {
		R[i] = 0;
		for (j = 0; j < DC->b; j++) {
			if (DC->incma[i * DC->b + j]) {
				R[i]++;
			}
		}
	}

	other::data_structures::tally T;

	T.init(R, DC->v, false, 0);
	if (f_v) {
		cout << "distribution of row sums: ";
		T.print(true /* f_backwards */);
		cout << endl;
	}

	FREE_int(R);



	if (f_v) {
		cout << "design_activity::do_row_sums done" << endl;
	}
}

void design_activity::do_tactical_decomposition(
		design_create *DC,
		int verbose_level)
// Computes the TDO and prints it in latex to cout
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_tactical_decomposition" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;



	{
		geometry::other_geometry::incidence_structure *Inc;


		Inc = NEW_OBJECT(geometry::other_geometry::incidence_structure);

		Inc->init_by_matrix(
				DC->v, DC->b, DC->incma,
				0 /* verbose_level */);

		combinatorics::tactical_decompositions::decomposition *Decomposition;


		Decomposition = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);

		Decomposition->init_incidence_structure(
				Inc,
				verbose_level);

#if 0
		data_structures::partitionstack *Stack;

		Stack = NEW_OBJECT(data_structures::partitionstack);

		Stack->allocate_with_two_classes(
				DC->v + DC->b, DC->v, DC->b,
				0 /* verbose_level */);
#endif


		while (true) {

			int ht0, ht1;

			ht0 = Decomposition->Stack->ht;

			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"before refine_column_partition_safe" << endl;
			}
			Decomposition->refine_column_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"before refine_row_partition_safe" << endl;
			}
			Decomposition->refine_row_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
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
		cout << "Decomposition:\\\\" << endl;
		cout << "Row scheme:\\\\" << endl;
		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts);
		cout << "Column scheme:\\\\" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts);

		other::data_structures::set_of_sets *Row_classes;
		other::data_structures::set_of_sets *Col_classes;

		Decomposition->Stack->get_row_classes(Row_classes, verbose_level);
		cout << "Row classes:\\\\" << endl;
		Row_classes->print_table_tex(cout);


		Decomposition->Stack->get_column_classes(Col_classes, verbose_level);
		cout << "Col classes:\\\\" << endl;
		Col_classes->print_table_tex(cout);

#if 0
		if (Row_classes->nb_sets > 1) {
			cout << "The row partition splits" << endl;
		}

		if (Col_classes->nb_sets > 1) {
			cout << "The col partition splits" << endl;
		}
#endif


		FREE_OBJECT(Inc);
		FREE_OBJECT(Decomposition);
		FREE_OBJECT(Row_classes);
		FREE_OBJECT(Col_classes);
	}

	if (f_v) {
		cout << "design_activity::do_tactical_decomposition" << endl;
	}

}

void design_activity::do_pair_orbits_on_blocks(
		design_create *DC,
		std::string &control_label,
		int *&Pair_orbits, int &degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks" << endl;
	}

	actions::action *A_on_blocks;

	long int *blocks;
	int nb_blocks;
	int block_sz;

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"before DC->compute_blocks_from_incidence_matrix" << endl;
	}


	DC->compute_blocks_from_incidence_matrix(
			blocks, nb_blocks, block_sz,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after DC->compute_blocks_from_incidence_matrix" << endl;
	}

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"before DC->A->Induced_action->create_induced_action_on_sets" << endl;
	}

	A_on_blocks = DC->A->Induced_action->create_induced_action_on_sets(
			nb_blocks, block_sz, blocks,
			verbose_level - 2);

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after DC->A->Induced_action->create_induced_action_on_sets" << endl;
	}



	orbits::orbits_global Orbits_global;
	poset_classification::poset_classification_control *Control;



	Control = Get_poset_classification_control(control_label);

	poset_classification::poset_classification *PC;

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"before creating any_group *AG" << endl;
	}

	groups::any_group *AG;

	AG = NEW_OBJECT(groups::any_group);
	AG->A_base = DC->A_base;
	AG->A = A_on_blocks;
	AG->label = Control->problem_label;
	AG->Subgroup_gens = DC->Sg;

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"AG->A_base=" << endl;
		AG->A_base->print_info();
		cout << "design_activity::do_pair_orbits_on_blocks "
				"AG->A=" << endl;
		AG->A->print_info();
	}


	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after creating any_group *AG" << endl;
	}

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"before Orbits_global.orbits_on_subsets" << endl;
	}
	Orbits_global.orbits_on_subsets(
			AG,
			Control,
			PC,
			2 /* size */,
			verbose_level - 2);
	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after Orbits_global.orbits_on_subsets" << endl;
	}

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"before computing orbits on pairs" << endl;
	}

	int *transporter;

	degree = AG->A->degree;
	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"degree == " << degree << endl;
	}

	transporter = NEW_int(AG->A_base->elt_size_in_int);
	Pair_orbits = NEW_int(degree * degree);
	Int_vec_mone(Pair_orbits, degree * degree);

	int i, j;
	long int set[2];
	long int canonical_set[2];
	int orbit_no;

	for (i = 0; i < degree; i++) {
		for (j = i + 1; j < degree; j++) {
			set[0] = i;
			set[1] = j;
			orbit_no = PC->trace_set(
					set, 2, 2,
				canonical_set, transporter,
				0 /*verbose_level - 1*/);
			Pair_orbits[i * degree + j] = orbit_no;
			Pair_orbits[j * degree + i] = orbit_no;
		}
	}

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after computing orbits on pairs" << endl;
	}
	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"Pair_orbits=" << endl;
		Int_matrix_print(Pair_orbits, degree, degree);
	}

	FREE_int(transporter);
	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks done" << endl;
	}
}




}}}




