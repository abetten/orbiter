/*
 * mapping.cpp
 *
 *  Created on: Sep 23, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {



mapping::mapping()
{
	Record_birth();
	Descr = NULL;

	Domain = NULL;
	Codomain = NULL;
	Ring = NULL;
	Formula = NULL;

	object_in_codomain_idx = -1;
	object_in_codomain_type = other::orbiter_kernel_system::symbol_table_object_type::t_nothing_object;
	object_in_codomain_cubic_surface = NULL;

	f_object_in_codomain = false;
	Variety_object = NULL;


	//std::string label_txt;
	//std::string label_tex;

	Image_pts = NULL;
	Image_pts_in_object = NULL;
	N_points_input = 0;

}

mapping::~mapping()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mapping::~mapping" << endl;
	}
	if (Image_pts) {
		FREE_lint(Image_pts);
	}
	if (f_object_in_codomain) {
		FREE_lint(Image_pts_in_object);
	}
	if (f_v) {
		cout << "mapping::~mapping done" << endl;
	}
}

void mapping::init(
		mapping_description *Descr,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mapping::init" << endl;
	}
	mapping::Descr = Descr;

	if (Descr->f_domain) {
		Domain = Get_projective_space(Descr->domain_label);
	}
	else {
		cout << "mapping::init need option -domain to specify the domain" << endl;
		exit(1);
	}

	if (Descr->f_codomain) {
		Codomain = Get_projective_space(Descr->codomain_label);
	}
	else {
		cout << "mapping::init need option -codomain to specify the codomain" << endl;
		exit(1);
	}

	if (Descr->f_ring) {
		Ring = Get_ring(Descr->ring_label);
	}
	else {
		cout << "mapping::init need option -ring to specify the polynomial ring" << endl;
		exit(1);
	}

	if (Descr->f_formula) {
		Formula = Get_symbol(Descr->formula_label);
	}
	else {
		cout << "mapping::init need option -ring to specify the polynomial ring" << endl;
		exit(1);
	}


	if (Descr->f_object_in_codomain_cubic_surface) {
		if (f_v) {
			cout << "mapping::init -object_in_codomain_cubic_surface "
					<< Descr->object_in_codomain_cubic_surface_label << endl;
		}

		object_in_codomain_idx = other::orbiter_kernel_system::Orbiter->find_symbol(Descr->object_in_codomain_cubic_surface_label);
		if (f_v) {
			cout << "mapping::init object_in_codomain_idx = "
					<< object_in_codomain_idx << endl;
		}

		object_in_codomain_type =
				other::orbiter_kernel_system::Orbiter->get_object_type(object_in_codomain_idx);

		if (object_in_codomain_type != other::orbiter_kernel_system::symbol_table_object_type::t_cubic_surface) {
			cout << "mapping::init object in codomain must be of type cubic surface" << endl;
			exit(1);
		}
		object_in_codomain_cubic_surface = Get_object_of_cubic_surface(Descr->object_in_codomain_cubic_surface_label);


		f_object_in_codomain = true;
		Variety_object = object_in_codomain_cubic_surface->SO->Variety_object;

	}
	else {
		object_in_codomain_idx = -1;
	}




	if (f_v) {
		cout << "mapping::init before evaluate_regular_map" << endl;
	}
	evaluate_regular_map(verbose_level);
	if (f_v) {
		cout << "mapping::init after evaluate_regular_map" << endl;
	}


	if (f_v) {
		cout << "mapping::init Image_pts:" << endl;
		Lint_vec_print(cout, Image_pts, N_points_input);
		cout << endl;
	}

	string fname_map;
	other::orbiter_kernel_system::file_io Fio;

	fname_map = Descr->formula_label + "_map.csv";


	string *Table;
	std::string *Col_headings;
	int nb_rows, nb_cols;
	int i;
	int *v;
	int *w;

	int input_len;
	int output_len;

	output_len = Formula->Formula_vector->len;

	geometry::projective_geometry::projective_space *P;

	P = Domain->P;

	input_len = P->Subspaces->n + 1;
	v = NEW_int(input_len);
	w = NEW_int(output_len);

	nb_rows = N_points_input;
	nb_cols = 5;

	Col_headings = new string[nb_cols];

	Table = new string[nb_rows * nb_cols];

	Col_headings[0] = "IN";
	Col_headings[1] = "INV";
	Col_headings[2] = "OUT";
	Col_headings[3] = "OUTV";
	Col_headings[4] = "SUB";

	for (i = 0; i < N_points_input; i++) {

		long int k;
		P->unrank_point(v, i);

		if (Image_pts[i] >= 0) {
			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified_lint(
				w, 1 /* stride */, output_len, Image_pts[i]);
		}
		else {
			Int_vec_zero(w, output_len);
		}

		if (f_object_in_codomain) {
			k = Image_pts_in_object[i];
		}
		else {
			k = -1;
		}

		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Int_vec_stringify(v, input_len) + "\"";
		Table[i * nb_cols + 2] = std::to_string(Image_pts[i]);
		Table[i * nb_cols + 3] = "\"" + Int_vec_stringify(w, output_len) + "\"";
		Table[i * nb_cols + 4] = std::to_string(k);
	}


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_map,
			nb_rows, nb_cols,
			Table, Col_headings,
			verbose_level - 2);

	delete [] Table;
	delete [] Col_headings;

	if (f_v) {
		cout << "Written file " << fname_map
				<< " of size " << Fio.file_size(fname_map) << endl;
	}



	if (f_v) {
		cout << "mapping::init done" << endl;
	}
}


void mapping::evaluate_regular_map(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "mapping::evaluate_regular_map" << endl;
	}

	int *v;
	int *w;
	int *w2;
	int h;
	long int i, j;

	geometry::projective_geometry::projective_space *P;

	P = Domain->P;


	if (Ring->get_nb_variables() != P->Subspaces->n + 1) {
		cout << "mapping::evaluate_regular_map "
				"number of variables does not match" << endl;
		cout << "number of variables in the ring = " << Ring->get_nb_variables() << endl;
		cout << "projective dimension of the domain plus one = " << P->Subspaces->n + 1 << endl;
		exit(1);
	}



	N_points_input = P->Subspaces->N_points;

	if (f_v) {
		cout << "mapping::evaluate_regular_map "
				"N_points_input = " << N_points_input << endl;
	}

	int len;

	len = Formula->Formula_vector->len;

	if (f_v) {
		cout << "mapping::evaluate_regular_map "
				"len = " << len << endl;
	}


	Image_pts = NEW_lint(N_points_input);

	if (f_object_in_codomain) {
		Image_pts_in_object = NEW_lint(N_points_input);
	}

	v = NEW_int(P->Subspaces->n + 1);
	w = NEW_int(len);
	w2 = NEW_int(len);


	other::data_structures::string_tools ST;
	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(symbol_table,
				Descr->substitute_text, verbose_level - 1);

	for (i = 0; i < N_points_input; i++) {

		P->unrank_point(v, i);

		if (f_v) {
			cout << "mapping::evaluate_regular_map "
					"point " << i << " / " << N_points_input << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << endl;
		}

		for (h = 0; h < P->Subspaces->n + 1; h++) {

			symbol_table[Ring->get_symbol(h)] = std::to_string(v[h]);

		}

		for (h = 0; h < len; h++) {

			w[h] = Formula->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					0 /*verbose_level*/);

		}

		Int_vec_copy(w, w2, len);



		if (!Int_vec_is_zero(w, len)) {
			P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
					w, 1 /* stride */, len, j);
		}
		else {
			j = -1;
		}

		if (f_v) {
			cout << "mapping::evaluate_regular_map "
					"point " << i << " / " << N_points_input << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << " maps to ";
			Int_vec_print(cout, w2, len);
			cout << " image rank = " << j;
			cout << endl;
		}

		Image_pts[i] = j;

		if (f_v) {
			cout << "mapping::evaluate_regular_map "
					"point " << i << " / " << N_points_input << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << " maps to ";
			Int_vec_print(cout, w2, len);
			cout << " image rank = " << Image_pts[i];
			//cout << " in subobject = " << k;
			cout << endl;
		}

		if (f_object_in_codomain) {
			int idx;


			if (Image_pts[i] != -1) {
				if (!Variety_object/*object_in_codomain_cubic_surface->SO*/->find_point(Image_pts[i], idx)) {
					cout << "mapping::evaluate_regular_map "
							"cannot find point " << Image_pts[i] << " on the variety" << endl;
					exit(1);
				}
				Image_pts_in_object[i] = idx;
				if (f_v) {
					cout << "mapping::evaluate_regular_map "
							"point " << i << " / " << N_points_input << " is ";
					Int_vec_print(cout, v, P->Subspaces->n + 1);
					cout << " maps to ";
					Int_vec_print(cout, w2, len);
					cout << " image rank = " << Image_pts[i];
					cout << " in variety = " << Image_pts_in_object[i];
					cout << endl;
				}
			}
			else {
				Image_pts_in_object[i] = -1;
			}
		}

	}

	FREE_int(v);
	FREE_int(w);
	FREE_int(w2);

	if (f_v) {
		cout << "mapping::evaluate_regular_map done" << endl;
	}
}



}}}

