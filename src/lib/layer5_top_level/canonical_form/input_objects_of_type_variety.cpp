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

	idx_po = idx_so = idx_eqn = idx_pts = idx_bitangents = 0;

	//Qco = NULL;
	Vo = NULL;

}

input_objects_of_type_variety::~input_objects_of_type_variety()
{
#if 0
	if (Qco) {
		int i;

		for (i = 0; i < nb_objects_to_test; i++) {
			if (Qco[i]) {
				FREE_OBJECT(Qco[i]);
			}
		}
		FREE_pvoid((void **) Qco);
	}
#endif
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

	if (Classifier->Descr->f_skip) {
		if (f_v) {
			cout << "input_objects_of_type_variety::init "
					"f_skip" << endl;
		}
		Get_int_vector_from_label(Classifier->Descr->skip_label,
				skip_vector, skip_sz, 0 /* verbose_level */);
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



	if (f_v) {
		cout << "input_objects_of_type_variety::init "
				"before read_input_objects" << endl;
	}
	read_input_objects(verbose_level);
	if (f_v) {
		cout << "input_objects_of_type_variety::init "
				"after read_input_objects" << endl;
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

	if (Classifier->Descr->f_skip) {
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

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < Classifier->Descr->nb_files; cnt++) {

		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					<< cnt << " / " << Classifier->Descr->nb_files << endl;
		}
		char str[1000];
		string fname;

		snprintf(str, sizeof(str), Classifier->Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		data_structures::spreadsheet S;

		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					<< cnt << " / " << Classifier->Descr->nb_files
					<< " fname=" << fname << endl;
		}
		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		nb_objects_to_test += S.nb_rows - 1;

		if (f_v) {
			cout << "input_objects_of_type_variety::count_nb_objects_to_test "
					"file " << cnt << " / " << Classifier->Descr->nb_files << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}
	}

	if (f_v) {
		cout << "input_objects_of_type_variety::count_nb_objects_to_test "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}
}

void input_objects_of_type_variety::read_input_objects(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;



	if (f_v) {
		cout << "input_objects_of_type_variety::read_input_objects" << endl;
	}


	count_nb_objects_to_test(verbose_level);


	//Qco = (quartic_curve_object_with_action **) NEW_pvoid(nb_objects_to_test);
	Vo = (variety_object_with_action **) NEW_pvoid(nb_objects_to_test);

	int counter;

	counter = 0;

	for (cnt = 0; cnt < Classifier->Descr->nb_files; cnt++) {

		char str[1000];
		string fname;
		int row;


		snprintf(str, sizeof(str), Classifier->Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		data_structures::spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		if (f_v) {
			cout << "input_objects_of_type_variety::read_input_objects "
					"S.nb_rows = " << S.nb_rows << endl;
			cout << "input_objects_of_type_variety::read_input_objects "
					"S.nb_cols = " << S.nb_cols << endl;
		}

		int *Carry_through = NULL;
		int nb_carry_through = 0;

		if (Classifier->Descr->carry_through.size()) {
			int i;

			nb_carry_through = Classifier->Descr->carry_through.size();
			Carry_through = NEW_int(nb_carry_through);
			for (i = 0; i < nb_carry_through; i++) {
				Carry_through[i] = S.find_column(Classifier->Descr->carry_through[i]);
			}
		}

#if 0
		int f_label_po_go;
		std::string column_label_po_go;
		int f_label_po_index;
		std::string column_label_po_index;
		int f_label_po;
		std::string column_label_po;
		int f_label_so;
		std::string column_label_so;
		int f_label_equation;
		std::string column_label_eqn;
		int f_label_points;
		std::string column_label_pts;
		int f_label_lines;
		std::string column_label_bitangents;
#endif

		if (Classifier->Descr->f_label_po_go) {
			idx_po_go = S.find_column(Classifier->Descr->column_label_po_go);
		}
		else {
			idx_po_go = -1;
		}
		if (Classifier->Descr->f_label_po_index) {
			idx_po_index = S.find_column(Classifier->Descr->column_label_po_index);
		}
		else {
			idx_po_index = -1;
		}
		if (Classifier->Descr->f_label_po) {
			idx_po = S.find_column(Classifier->Descr->column_label_po);
		}
		else {
			idx_po = -1;
		}
		if (Classifier->Descr->f_label_so) {
			idx_so = S.find_column(Classifier->Descr->column_label_so);
		}
		else {
			idx_so = -1;
		}
		if (Classifier->Descr->f_label_equation) {
			idx_eqn = S.find_column(Classifier->Descr->column_label_eqn);
		}
		else {
			idx_eqn = -1;
		}

		if (Classifier->Descr->f_label_points) {
			idx_pts = S.find_column(Classifier->Descr->column_label_pts);
		}
		else {
			idx_pts = -1;
		}

		if (Classifier->Descr->f_label_lines) {
			idx_bitangents = S.find_column(Classifier->Descr->column_label_bitangents);
		}
		else {
			idx_bitangents = -1;
		}

		for (row = 0; row < S.nb_rows - 1; row++, counter++) {

			if (f_v) {
				cout << "cnt = " << cnt << " / " << Classifier->Descr->nb_files
						<< " row = " << row << " / " << S.nb_rows - 1 << endl;
				cout << "counter = " << counter << " / " << nb_objects_to_test << endl;
			}

			if (skip_this_one(counter)) {
				if (f_v) {
					cout << "input_objects_of_type_variety::read_input_objects "
							"skipping case counter = " << counter << endl;
				}
				Vo[counter] = NULL;
				continue;
			}





			if (f_v) {
				cout << "input_objects_of_type_variety::read_input_objects "
						"before prepare_input_of_variety_type" << endl;
			}
			prepare_input_of_variety_type(
					row, counter,
					Carry_through,
					&S,
					Vo[counter], verbose_level - 2);
			if (f_v) {
				cout << "input_objects_of_type_variety::read_input_objects "
						"after prepare_input_of_variety_type" << endl;
			}

			if (idx_pts == -1) {
				if (f_v) {
					cout << "input_objects_of_type_variety::read_input_objects "
							"before enumerate_points" << endl;
				}
				Vo[counter]->Variety_object->enumerate_points(
						//Classifier->Poly_ring,
						verbose_level - 1);
				if (f_v) {
					cout << "input_objects_of_type_variety::read_input_objects "
							"after enumerate_points" << endl;
				}
			}



		} // next row


		if (Carry_through) {
			FREE_int(Carry_through);
		}

	} // next cnt

	if (f_v) {
		cout << "input_objects_of_type_variety::read_input_objects done" << endl;
	}
}


void input_objects_of_type_variety::prepare_input(
		int row, int counter,
		int *Carry_through,
		data_structures::spreadsheet *S,
		quartic_curve_object_with_action *&Qco,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input" << endl;
	}



	int i;
	int po_go, po_index;
	int po, so;


	po_go = S->get_lint(row + 1, idx_po_go);
	po_index = S->get_lint(row + 1, idx_po_index);
	po = S->get_lint(row + 1, idx_po);
	so = S->get_lint(row + 1, idx_so);
	string eqn_txt;
	string pts_txt;
	string bitangents_txt;

	eqn_txt = S->get_entry_ij(row + 1, idx_eqn);

	if (idx_pts >= 0) {
		pts_txt = S->get_entry_ij(row + 1, idx_pts);
	}
	else {
		pts_txt = "";
	}

	if (idx_bitangents >= 0) {
		bitangents_txt = S->get_entry_ij(row + 1, idx_bitangents);
	}
	else {
		bitangents_txt = "";
	}

	data_structures::string_tools ST;

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input "
				"row = " << row
				<< " before processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt
				<< " =" << bitangents_txt << endl;
	}


	ST.remove_specific_character(eqn_txt, '\"');
	ST.remove_specific_character(pts_txt, '\"');
	ST.remove_specific_character(bitangents_txt, '\"');

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input "
				"row = " << row << " after processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}



	Qco = NEW_OBJECT(quartic_curve_object_with_action);

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input "
				"before Qco->init" << endl;
	}


	Qco->init(
			counter, po_go, po_index, po, so,
			Classifier->Poly_ring,
			eqn_txt,
			pts_txt, bitangents_txt,
			verbose_level);
	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input "
				"after Qco->init" << endl;
	}


	for (i = 0; i < Classifier->Descr->carry_through.size(); i++) {

		string s;

		s = S->get_entry_ij(row + 1, Carry_through[i]);

		Qco->Carrying_through.push_back(s);

	}


	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input done" << endl;
	}

}

void input_objects_of_type_variety::prepare_input_of_variety_type(
		int row, int counter,
		int *Carry_through,
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
	string eqn_txt;
	string pts_txt;
	string bitangents_txt;

	eqn_txt = S->get_entry_ij(row + 1, idx_eqn);

	if (idx_pts >= 0) {
		pts_txt = S->get_entry_ij(row + 1, idx_pts);
	}
	else {
		pts_txt = "";
	}

	if (idx_bitangents >= 0) {
		bitangents_txt = S->get_entry_ij(row + 1, idx_bitangents);
	}
	else {
		bitangents_txt = "";
	}

	data_structures::string_tools ST;

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"row = " << row
				<< " before processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt
				<< " =" << bitangents_txt << endl;
	}


	ST.remove_specific_character(eqn_txt, '\"');
	ST.remove_specific_character(pts_txt, '\"');
	ST.remove_specific_character(bitangents_txt, '\"');

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"row = " << row << " after processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}



	Vo = NEW_OBJECT(variety_object_with_action);

	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"before Vo->init" << endl;
	}


	Vo->init(
			counter, po_go, po_index, po, so,
			Classifier->PA->P,
			Classifier->Poly_ring,
			eqn_txt,
			pts_txt, bitangents_txt,
			verbose_level);
	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type "
				"after Vo->init" << endl;
	}


	for (i = 0; i < Classifier->Descr->carry_through.size(); i++) {

		string s;

		s = S->get_entry_ij(row + 1, Carry_through[i]);

		Vo->Carrying_through.push_back(s);

	}


	if (f_v) {
		cout << "input_objects_of_type_variety::prepare_input_of_variety_type done" << endl;
	}

}




}}}


