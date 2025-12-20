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
		combinatorics::design_theory::design_object *Design_object,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::perform_activity" << endl;
		cout << "design_activity::perform_activity design = " << Design_object->label_txt << endl;
	}

	design_activity::Descr = Descr;

	if (Design_object->DC == NULL) {
		cout << "design_activity::perform_activity "
				"Design_object->DC == NULL" << endl;
		exit(1);
	}
	design_create *DC = (design_create *) Design_object->DC;

	if (Descr->f_load_table) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_load_table" << endl;
		}

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

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_canonical_form" << endl;
		}


		do_canonical_form(
				Descr->Canonical_form_Descr,
				verbose_level);
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_canonical_form done" << endl;
		}
	}
	else if (Descr->f_extract_solutions_by_index_csv) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_csv" << endl;
		}

		if (Design_object->DC == NULL) {
			cout << "design_activity::perform_activity "
					"Design_object->DC == NULL" << endl;
			exit(1);
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

		if (Design_object->DC == NULL) {
			cout << "design_activity::perform_activity "
					"Design_object->DC == NULL" << endl;
			exit(1);
		}
		design_create *DC = (design_create *) Design_object->DC;

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
		Design_object->do_export_flags(
				verbose_level);

	}
	else if (Descr->f_export_incidence_matrix) {
		if (f_v) {
			cout << "design_activity::perform_activity f_export_incidence_matrix" << endl;
		}
		Design_object->do_export_incidence_matrix_csv(
				verbose_level);
	}
	else if (Descr->f_export_incidence_matrix_latex) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_export_incidence_matrix_latex" << endl;
		}

		other::graphics::draw_incidence_structure_description
			*Draw_incidence_structure_description;

		Draw_incidence_structure_description =
				Get_draw_incidence_structure_options(
						Descr->export_incidence_matrix_latex_draw_options);

		Design_object->do_export_incidence_matrix_latex(
				Draw_incidence_structure_description, verbose_level);
	}
	else if (Descr->f_intersection_matrix) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_intersection_matrix" << endl;
		}

		//int f_save = false;

		Design_object->do_intersection_matrix(
				Descr->f_save,
				verbose_level);
	}
	else if (Descr->f_export_blocks) {
		if (f_v) {
			cout << "design_activity::perform_activity export_blocks" << endl;
		}
		Design_object->do_export_blocks(
				verbose_level);
	}
	else if (Descr->f_row_sums) {
		if (f_v) {
			cout << "design_activity::perform_activity row_sums" << endl;
		}
		Design_object->do_row_sums(
				verbose_level);
	}
	else if (Descr->f_tactical_decomposition) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_tactical_decomposition" << endl;
		}
		Design_object->do_tactical_decomposition(
				verbose_level);
	}
	else if (Descr->f_orbits_on_blocks) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_orbits_on_blocks" << endl;
		}

		if (Design_object->DC == NULL) {
			cout << "design_activity::perform_activity "
					"Design_object->DC == NULL" << endl;
			exit(1);
		}
		design_create *DC = (design_create *) Design_object->DC;

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

		if (Design_object->DC == NULL) {
			cout << "design_activity::perform_activity "
					"Design_object->DC == NULL" << endl;
			exit(1);
		}
		design_create *DC = (design_create *) Design_object->DC;

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

		n = 1 + Design_object->v + Design_object->b;
		Adj = NEW_int(n * n);
		Int_vec_zero(Adj, n * n);
		for (i = 0; i < Design_object->v; i++) {
			u = 0;
			v = 1 + i;
			Adj[u * n + v] = Adj[v * n + u] = 1;
		}
		for (i = 0; i < Design_object->v; i++) {
			for (j = 0; j < Design_object->b; j++) {
				u = 1 + i;
				v = 1 + Design_object->v + j;
				Adj[u * n + v] = Adj[v * n + u] =
						Design_object->incma[i * Design_object->b + j];
			}
		}
		for (i = 0; i < Design_object->b; i++) {
			u = 1 + Design_object->v + i;
			for (j = i + 1; j < Design_object->b; j++) {
				v = 1 + Design_object->v + j;
				if (Pair_orbits[i * degree + j] == orbit_idx) {
					Adj[u * n + v] = Adj[v * n + u] = 1;
				}
			}
		}


		string fname;

		fname = Design_object->label_txt + "_one_pt_ext_orb" + std::to_string(orbit_idx) + ".csv";
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

	Combi.load_design_table(
			DC,
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

		other::data_structures::set_of_sets *SoS;

		Fio.Csv_file_support->read_column_as_set_of_sets(
				fname_in, col_label,
				SoS,
				verbose_level - 1);


#if 0
		int *Sol_idx_1;

		Fio.Csv_file_support->read_table_of_strings_as_matrix(
				fname_in, col_label,
				Sol_idx_1, nb_sol, sol_width,
				verbose_level - 1);
#endif

		nb_sol = SoS->nb_sets;

		if (!SoS->has_constant_size_property()) {
			cout << "design_activity::do_extract_solutions_by_index "
					"the sets have different sizes" << endl;
			exit(1);
		}

		sol_width = SoS->get_constant_size();

		int i;

		if (f_v) {
			cout << "design_activity::do_extract_solutions_by_index "
					"solutions have size " << nb_sol << " x " << sol_width << endl;
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
			Lint_vec_copy_to_int(
					//Sol_idx_1 + i * sol_width,
					SoS->Sets[i],
					Sol_idx + i * (prefix_sz + sol_width) + prefix_sz,
					sol_width);
			//FREE_int(data);
#if 0
			for (j = 0; j < sol_width; j++) {
				Sol_idx[i * (prefix_sz + sol_width) + prefix_sz + j] = Sol_idx_1[i * sol_width + j];
			}
#endif
		}
		FREE_OBJECT(SoS);

		//FREE_int(Sol_idx_1);

		sol_width += prefix_sz;
	}
	else {
		other::data_structures::set_of_sets *SoS;
		int underlying_set_size = 0;
		int i, j;

		SoS = NEW_OBJECT(other::data_structures::set_of_sets);
		SoS->init_from_orbiter_file(
				underlying_set_size,
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

	Combi.create_design_table(
			DC->Design_object,
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
				"before DC->Design_object->compute_blocks_from_incidence_matrix" << endl;
	}


	DC->Design_object->compute_blocks_from_incidence_matrix(
			blocks, nb_blocks, block_sz,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_pair_orbits_on_blocks "
				"after DC->Design_object->compute_blocks_from_incidence_matrix" << endl;
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
	// ToDo:
	Orbits_global.orbits_on_subsets(
			AG->A_base,
			AG->A,
			AG->Subgroup_gens,
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




