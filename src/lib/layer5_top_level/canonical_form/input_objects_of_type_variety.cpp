/*
 * input_objects_of_type_variety.cpp
 *
 *  Created on: Oct 13, 2023
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


input_objects_of_type_variety::input_objects_of_type_variety()
{
	Classifier = NULL;

	skip_vector = NULL;
	skip_sz = 0;


	nb_objects_to_test = 0;

	idx_po_go = 0;
	idx_po_index = 0;

	idx_po = idx_so = 0;
	idx_eqn_algebraic = 0;
	idx_eqn_by_coefficients = 0;
	idx_eqn2_algebraic = 0;
	idx_eqn2_by_coefficients = 0;
	idx_pts = idx_bitangents = 0;

	Vo = NULL;

}

input_objects_of_type_variety::~input_objects_of_type_variety()
{
	if (Vo) {
		int i;

		for (i = 0; i < nb_objects_to_test; i++) {
			if (Vo[i]) {
				FREE_OBJECT(Vo[i]);
			}
		}
		FREE_pvoid((void **) Vo);
	}
}

void input_objects_of_type_variety::init(
		canonical_form_classifier *Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "input_objects_of_type_variety::init" << endl;
	}
	input_objects_of_type_variety::Classifier = Classifier;


#if 1
	if (Classifier->get_description()->f_skip) {
		if (f_v) {
			cout << "input_objects_of_type_variety::init "
					"f_skip" << endl;
		}
		Get_int_vector_from_label(
				Classifier->get_description()->skip_label,
				skip_vector, skip_sz,
				0 /* verbose_level */);
		data_structures::sorting Sorting;

		Sorting.int_vec_heapsort(skip_vector, skip_sz);
		if (f_v) {
			cout << "input_objects_of_type_variety::init "
					"skip list consists of " << skip_sz << " cases" << endl;
			cout << "The cases to be skipped are :";
			Int_vec_print(cout, skip_vector, skip_sz);
			cout << endl;
		}
	}
#endif


	if (f_v) {
		cout << "input_objects_of_type_variety::init "
				"before read_input_objects_from_list_of_csv_files" << endl;
	}
	read_input_objects_from_list_of_csv_files(verbose_level);
	if (f_v) {
		cout << "input_objects_of_type_variety::init "
				"after read_input_objects_from_list_of_csv_files" << endl;
	}

	if (f_v) {
		cout << "input_objects_of_type_variety::init done" << endl;
	}
}


int input_objects_of_type_variety::skip_this_one(
		int counter)
{
	data_structures::sorting Sorting;
	int idx;

	if (Classifier->get_description()->f_skip) {
		if (Sorting.int_vec_search(
				skip_vector, skip_sz, counter, idx)) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}

}


void input_objects_of_type_variety::count_nb_objects_to_test(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;

	if (f_v) {
		cout << "input_objects_of_type_variety::count_nb_objects_to_test" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < Classifier->get_description()->nb_files; cnt++) {

		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					<< cnt << " / " << Classifier->get_description()->nb_files << endl;
		}
		data_structures::string_tools ST;


		string fname;

		fname = ST.printf_d(Classifier->get_description()->fname_mask, cnt);

		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					<< cnt << " / " << Classifier->get_description()->nb_files
					<< " fname=" << fname << endl;
		}

		int nb;


		nb = Fio.count_number_of_data_lines_in_spreadsheet(
				fname, verbose_level - 2);

		nb_objects_to_test += nb;


		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					"file " << cnt << " / " << Classifier->get_description()->nb_files << " has  "
					<< nb << " objects, total = " << nb_objects_to_test << endl;
		}
	}

	if (f_v) {
		cout << "input_objects_of_type_variety::count_nb_objects_to_test "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}
}

void input_objects_of_type_variety::read_input_objects_from_list_of_csv_files(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);




	if (f_v) {
		cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files" << endl;
		if (Classifier->get_description()->f_has_nauty_output) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files f_has_nauty_output" << endl;
		}
	}


	count_nb_objects_to_test(verbose_level);


	Vo = (variety_object_with_action **) NEW_pvoid(nb_objects_to_test);

	int counter;

	counter = 0;

	int file_cnt;

	for (file_cnt = 0; file_cnt < Classifier->get_description()->nb_files; file_cnt++) {

		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"file " << file_cnt << " / " << Classifier->get_description()->nb_files << endl;
		}
		string fname;

		data_structures::string_tools ST;

		fname = ST.printf_d(Classifier->get_description()->fname_mask, file_cnt);

		data_structures::spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"S.nb_rows = " << S.nb_rows << endl;
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"S.nb_cols = " << S.nb_cols << endl;
		}

		int *Carry_through = NULL;
		int nb_carry_through = 0;

		if (Classifier->get_description()->carry_through.size()) {
			int i;

			//f_carry_through = true;
			nb_carry_through = Classifier->get_description()->carry_through.size();
			Carry_through = NEW_int(nb_carry_through);
			for (i = 0; i < nb_carry_through; i++) {
				Carry_through[i] = S.find_column(Classifier->get_description()->carry_through[i]);
			}
		}

		if (Classifier->get_description()->f_has_nauty_output) {

			//f_carry_through = true;
			if (f_v) {
				cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
						"f_has_nauty_output" << endl;
			}

			int nb_ct = nb_carry_through + 9;
			int *Carry_through2;

			Carry_through2 = NEW_int(nb_ct);

			int i;

			const char *headings[] = {
						"NO_N",
						"NO_ago",
						"NO_base_len",
						"NO_aut_cnt",
						"NO_base",
						"NO_tl",
						"NO_aut",
						"NO_cl",
						"NO_stats"
			};

			Int_vec_copy(Carry_through, Carry_through2, nb_carry_through);
			for (i = 0; i < 9; i++) {
				string s;

				s = headings[i];
				if (f_v) {
					cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
							"before S.find_column " << s << endl;
				}
				Carry_through2[nb_carry_through + i] = S.find_column(s);
			}
			FREE_int(Carry_through);
			Carry_through = Carry_through2;
			nb_carry_through = nb_ct;

		}

		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"nb_carry_through = " << nb_carry_through << endl;
		}

#if 0
		//ROW,CNT,PO,SO,PO_GO,PO_INDEX,Iso_idx,F_Fst,Idx_canonical,Idx_eqn,Eqn,Pts,Bitangents,
		//NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats,
		//nb_eqn,ago
		v.push_back(std::to_string(Vo->cnt));
		v.push_back(std::to_string(Vo->po));
		v.push_back(std::to_string(Vo->so));
		v.push_back(std::to_string(Vo->po_go));
		v.push_back(std::to_string(Vo->po_index));
		v.push_back(std::to_string(Canonical_form_classifier->Output->Iso_idx[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Output->F_first_time[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Output->Idx_canonical_form[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Output->Idx_equation[i]));
#endif


		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"before find_columns" << endl;
		}
		find_columns(&S, verbose_level - 2);
		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"after find_columns" << endl;
		}



		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"before read_all_varieties_from_spreadsheet" << endl;
		}
		read_all_varieties_from_spreadsheet(
				&S,
				Carry_through,
				nb_carry_through,
				file_cnt, counter,
				verbose_level);
		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files "
					"after read_all_varieties_from_spreadsheet" << endl;
		}


		if (Carry_through) {
			FREE_int(Carry_through);
		}

	} // next file_cnt

	if (f_v) {
		cout << "input_objects_of_type_variety::read_input_objects_from_list_of_csv_files done" << endl;
	}
}


void input_objects_of_type_variety::read_all_varieties_from_spreadsheet(
		data_structures::spreadsheet *S,
		int *Carry_through,
		int nb_carry_through,
		int file_cnt, int &counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet" << endl;
	}
	if (f_v) {
		cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
				"file = " << file_cnt << " / " << Classifier->get_description()->nb_files << endl;
	}

	int row;

	for (row = 0; row < S->nb_rows - 1; row++, counter++) {

		if (f_v) {
			cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
					"file = " << file_cnt << " / " << Classifier->get_description()->nb_files
					<< " row = " << row << " / " << S->nb_rows - 1 << endl;
			cout << "counter = " << counter << " / " << nb_objects_to_test << endl;
		}

#if 0
		if (skip_this_one(counter)) {
			if (f_v) {
				cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
						"skipping case counter = " << counter << endl;
			}
			Vo[counter] = NULL;
			continue;
		}
#endif




		if (f_v) {
			cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
					"before prepare_input_of_variety_type" << endl;
		}
		prepare_input_of_variety_type(
				row, counter,
				Carry_through,
				nb_carry_through,
				S,
				Vo[counter], verbose_level - 2);
		if (f_v) {
			cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
					"after prepare_input_of_variety_type" << endl;
		}

		if (idx_pts == -1) {
			if (f_v) {
				cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
						"before enumerate_points" << endl;
			}
			Vo[counter]->Variety_object->enumerate_points(
					verbose_level - 1);
			if (f_v) {
				cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet "
						"after enumerate_points" << endl;
			}
		}


	} // next row

	if (f_v) {
		cout << "input_objects_of_type_variety::read_all_varieties_from_spreadsheet done" << endl;
	}

}

void input_objects_of_type_variety::find_columns(
		data_structures::spreadsheet *S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "input_objects_of_type_variety::find_columns" << endl;
	}

	if (Classifier->get_description()->f_label_po_go) {
		idx_po_go = S->find_column(Classifier->get_description()->column_label_po_go);
	}
	else {
		idx_po_go = -1;
	}
	if (Classifier->get_description()->f_label_po_index) {
		idx_po_index = S->find_column(Classifier->get_description()->column_label_po_index);
	}
	else {
		idx_po_index = -1;
	}
	if (Classifier->get_description()->f_label_po) {
		idx_po = S->find_column(Classifier->get_description()->column_label_po);
	}
	else {
		idx_po = -1;
	}
	if (Classifier->get_description()->f_label_so) {
		idx_so = S->find_column(Classifier->get_description()->column_label_so);
	}
	else {
		idx_so = -1;
	}
	if (Classifier->get_description()->f_label_equation_algebraic) {
		idx_eqn_algebraic = S->find_column(Classifier->get_description()->column_label_eqn_algebraic);
	}
	else {
		idx_eqn_algebraic = -1;
	}
	if (Classifier->get_description()->f_label_equation_by_coefficients) {
		idx_eqn_by_coefficients = S->find_column(Classifier->get_description()->column_label_eqn_by_coefficients);
	}
	else {
		idx_eqn_by_coefficients = -1;
	}
	if (Classifier->get_description()->f_label_equation2_algebraic) {
		idx_eqn2_algebraic = S->find_column(Classifier->get_description()->column_label_eqn2_algebraic);
	}
	else {
		idx_eqn2_algebraic = -1;
	}
	if (Classifier->get_description()->f_label_equation2_by_coefficients) {
		idx_eqn2_by_coefficients = S->find_column(Classifier->get_description()->column_label_eqn2_by_coefficients);
	}
	else {
		idx_eqn2_by_coefficients = -1;
	}

	if (Classifier->get_description()->f_label_points) {
		idx_pts = S->find_column(Classifier->get_description()->column_label_pts);
	}
	else {
		idx_pts = -1;
	}

	if (Classifier->get_description()->f_label_lines) {
		idx_bitangents = S->find_column(Classifier->get_description()->column_label_bitangents);
	}
	else {
		idx_bitangents = -1;
	}

	if (f_v) {
		cout << "input_objects_of_type_variety::find_columns done" << endl;
	}
}

void input_objects_of_type_variety::prepare_input_of_variety_type(
		int row, int counter,
		int *Carry_through,
		int nb_carry_trough,
		data_structures::spreadsheet *S,
		variety_object_with_action *&Vo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type" << endl;
	}



	int i;
	int po_go, po_index;
	int po, so;


	po_go = S->get_lint(row + 1, idx_po_go);
	po_index = S->get_lint(row + 1, idx_po_index);
	po = S->get_lint(row + 1, idx_po);
	so = S->get_lint(row + 1, idx_so);


	data_structures::string_tools ST;



	algebraic_geometry::variety_description *VD;

	VD = NEW_OBJECT(algebraic_geometry::variety_description);


	VD->f_label_txt = true;
	VD->label_txt = "po" + std::to_string(po) + "_so" + std::to_string(so);

	VD->f_label_tex = false;
	VD->label_tex = "po" + std::to_string(po) + "\\_so" + std::to_string(so);

	VD->f_projective_space = false;
	//VD->projective_space_label;

	VD->f_has_projective_space_pointer = true;
	VD->Projective_space_pointer = Classifier->PA->P;

	VD->f_ring = false;
	VD->ring_label = "";

	VD->f_has_ring_pointer = true;
	VD->Ring_pointer = Classifier->Poly_ring;

	VD->f_has_equation_in_algebraic_form = false;
	VD->f_has_equation_by_coefficients = false;

	if (idx_eqn_algebraic >= 0) {
		string eqn_txt;

		eqn_txt = S->get_entry_ij(row + 1, idx_eqn_algebraic);
		ST.remove_specific_character(eqn_txt, '\"');

		VD->f_has_equation_in_algebraic_form = true;
		VD->equation_in_algebraic_form_text = eqn_txt;
	}
	else if (idx_eqn_by_coefficients >= 0) {
		string eqn_txt;

		eqn_txt = S->get_entry_ij(row + 1, idx_eqn_by_coefficients);
		ST.remove_specific_character(eqn_txt, '\"');

		VD->f_has_equation_by_coefficients = true;
		VD->equation_by_coefficients_text = eqn_txt;

	}


	VD->f_has_second_equation_in_algebraic_form = false;
	VD->f_has_second_equation_by_coefficients = false;

	if (idx_eqn2_algebraic >= 0) {
		string eqn_txt;
		eqn_txt = S->get_entry_ij(row + 1, idx_eqn2_algebraic);
		ST.remove_specific_character(eqn_txt, '\"');
		VD->f_has_second_equation_in_algebraic_form = true;
		VD->second_equation_in_algebraic_form_text = eqn_txt;
	}
	else {
		VD->f_has_second_equation_by_coefficients = true;
		string eqn_txt;
		eqn_txt = S->get_entry_ij(row + 1, idx_eqn2_by_coefficients);
		ST.remove_specific_character(eqn_txt, '\"');
		VD->second_equation_by_coefficients_text = "";
	}



	if (idx_pts >= 0) {
		string pts_txt;
		pts_txt = S->get_entry_ij(row + 1, idx_pts);
		ST.remove_specific_character(pts_txt, '\"');
		VD->f_has_points = true;
		VD->points_txt = pts_txt;
	}
	else {
		VD->f_has_points = false;
	}

	if (idx_bitangents >= 0) {
		string bitangents_txt;
		bitangents_txt = S->get_entry_ij(row + 1, idx_bitangents);
		ST.remove_specific_character(bitangents_txt, '\"');
		VD->f_has_bitangents = true;
		VD->bitangents_txt = bitangents_txt;
	}
	else {
		VD->f_has_bitangents = false;
	}


	Vo = NEW_OBJECT(variety_object_with_action);

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"before Vo->init" << endl;
	}
	Vo->init(
			Classifier->PA,
			counter, po_go, po_index, po, so,
			VD,
			verbose_level);
	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"after Vo->init" << endl;
	}


	for (i = 0; i < nb_carry_trough; i++) {

		string s;

		s = S->get_entry_ij(row + 1, Carry_through[i]);

		Vo->Carrying_through.push_back(s);

	}

	if (Classifier->get_description()->f_has_nauty_output) {

		Vo->f_has_nauty_output = true;
		Vo->nauty_output_index_start = nb_carry_trough - 9;

		if (f_v) {
			cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
					"f_has_nauty_output" << endl;
			cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
					"nauty_output_index_start=" << Vo->nauty_output_index_start << endl;
		}
	}

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type done" << endl;
	}

}




}}}


