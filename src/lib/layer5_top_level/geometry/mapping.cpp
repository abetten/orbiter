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
	Descr = NULL;

	Domain = NULL;
	Codomain = NULL;
	Ring = NULL;
	Formula = NULL;

	object_in_codomain_idx = -1;
	object_in_codomain_type = orbiter_kernel_system::symbol_table_object_type::t_nothing_object;
	object_in_codomain_cubic_surface = NULL;


	//std::string label_txt;
	//std::string label_tex;

	Image_pts = NULL;
	N_points_input = 0;

}

mapping::~mapping()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mapping::~mapping" << endl;
	}
	if (Image_pts) {
		FREE_lint(Image_pts);
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


	if (Descr->f_object_in_codomain) {
		if (f_v) {
			cout << "mapping::init -object_in_codomain " << Descr->object_in_codomain_label << endl;
		}

		object_in_codomain_idx = orbiter_kernel_system::Orbiter->find_symbol(Descr->object_in_codomain_label);
		if (f_v) {
			cout << "mapping::init object_in_codomain_idx = " << object_in_codomain_idx << endl;
		}

		object_in_codomain_type =
				orbiter_kernel_system::Orbiter->get_object_type(object_in_codomain_idx);

		if (object_in_codomain_type != orbiter_kernel_system::symbol_table_object_type::t_cubic_surface) {
			cout << "mapping::init object in codomain must be of type cubic surface" << endl;
			exit(1);
		}
		object_in_codomain_cubic_surface = Get_object_of_cubic_surface(Descr->object_in_codomain_label);


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
	orbiter_kernel_system::file_io Fio;

	fname_map = Descr->formula_label + "_map.csv";


	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_map, Image_pts, N_points_input, 1);
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
	long int i, j, k;

	geometry::projective_space *P;

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

	v = NEW_int(P->Subspaces->n + 1);
	w = NEW_int(len);
	w2 = NEW_int(len);


	data_structures::string_tools ST;
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


		if (Descr->f_object_in_codomain) {
			int idx;


			if (j != -1) {
				if (!object_in_codomain_cubic_surface->SO->find_point(j, idx)) {
					cout << "connot find point " << j << " in the cubic surface" << endl;
					exit(1);
				}
				k = idx;
			}
			else {
				k = -1;
			}

			if (f_v) {
				cout << "mapping::evaluate_regular_map "
						"point " << i << " / " << N_points_input << " is ";
				Int_vec_print(cout, v, P->Subspaces->n + 1);
				cout << " maps to ";
				Int_vec_print(cout, w2, len);
				cout << " image rank = " << j;
				cout << " in subobject = " << k;
				cout << endl;
			}


		}
		else {
			k = j;
		}

		Image_pts[i] = k;
	}

	FREE_int(v);
	FREE_int(w);
	FREE_int(w2);

	if (f_v) {
		cout << "mapping::evaluate_regular_map done" << endl;
	}
}



}}}

