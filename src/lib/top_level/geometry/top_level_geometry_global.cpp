/*
 * top_level_geometry_global.cpp
 *
 *  Created on: May 23, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


top_level_geometry_global::top_level_geometry_global()
{

}

top_level_geometry_global::~top_level_geometry_global()
{

}


void top_level_geometry_global::set_stabilizer_projective_space(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space" << endl;
	}

	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space before SubC->classify_substructures" << endl;
	}

	SubC->classify_substructures(PA->A, PA->A,
			PA->A->Strong_gens,
			intermediate_subset_size, verbose_level);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space after SubC->classify_substructures" << endl;
		cout << "top_level_geometry_global::set_stabilizer_projective_space We found " << SubC->nb_orbits << " orbits at level " << intermediate_subset_size << ":" << endl;
	}





	int nb_objects_to_test;
	int cnt;
	int row;

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		nb_objects_to_test += S.nb_rows - 1;
		if (f_v) {
			cout << "top_level_geometry_global::set_stabilizer_projective_space "
					"file " << cnt << " / " << nb << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}

	}

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}



	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "top_level_geometry_global::set_stabilizer_projective_space S.nb_rows = " << S.nb_rows << endl;
			cout << "top_level_geometry_global::set_stabilizer_projective_space S.nb_cols = " << S.nb_cols << endl;
		}

		int col_idx;

		col_idx = S.find_column(column_label);


		for (row = 0; row < S.nb_rows - 1; row++) {

			int j, t;
			//string eqn_txt;
			string pts_txt;
			//string bitangents_txt;
			//int *eqn;
			//int sz;
			long int *pts;
			int nb_pts;
			long int *canonical_pts;
			//long int *bitangents;
			//int nb_bitangents;

			if (f_v) {
				cout << "#############################################################################" << endl;
				cout << "cnt = " << cnt << " / " << nb << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

#if 0
			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer_projective_space token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
#endif
			j = col_idx;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer_projective_space token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
#if 0
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer_projective_space token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);
#endif

			string_tools ST;

			//ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			//ST.remove_specific_character(bitangents_txt, '\"');

#if 0
			if (FALSE) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}
#endif

			//Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			//Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			canonical_pts = NEW_lint(nb_pts);

			if (f_v) {
				cout << "row = " << row;
				//cout << " eqn=";
				//Orbiter->Int_vec.print(cout, eqn, sz);
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				//cout << " bitangents=";
				//Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}

			set_stabilizer_of_set(
						SubC,
						cnt, nb, row,
						//eqn,
						//sz,
						pts,
						nb_pts,
						canonical_pts,
						//bitangents,
						//nb_bitangents,
						verbose_level);

			//FREE_int(eqn);
			FREE_lint(pts);
			//FREE_lint(bitangents);
			FREE_lint(canonical_pts);

		} // row

	}

	FREE_OBJECT(SubC);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space done" << endl;
	}

}


void top_level_geometry_global::set_stabilizer_orthogonal_space(
		orthogonal_space_with_action *OA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space" << endl;
	}

	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space before SubC->classify_substructures" << endl;
	}

	SubC->classify_substructures(OA->A, OA->A,
			OA->A->Strong_gens,
			intermediate_subset_size, verbose_level - 5);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space after SubC->classify_substructures" << endl;
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space We found " << SubC->nb_orbits << " orbits at level " << intermediate_subset_size << ":" << endl;
	}





	int nb_objects_to_test;
	int cnt;
	int row;

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		nb_objects_to_test += S.nb_rows - 1;
		if (f_v) {
			cout << "top_level_geometry_global::set_stabilizer_orthogonal_space "
					"file " << cnt << " / " << nb << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}

	}

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}



	for (cnt = 0; cnt < nb; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "top_level_geometry_global::set_stabilizer_orthogonal_space S.nb_rows = " << S.nb_rows << endl;
			cout << "top_level_geometry_global::set_stabilizer_orthogonal_space S.nb_cols = " << S.nb_cols << endl;
		}

		int col_idx;

		col_idx = S.find_column(column_label);


		for (row = 0; row < S.nb_rows - 1; row++) {

			int j, t;
			string pts_txt;
			long int *pts;
			int nb_pts;
			long int *canonical_pts;

			if (f_v) {
				cout << "#############################################################################" << endl;
				cout << "cnt = " << cnt << " / " << nb << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = col_idx;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer_orthogonal_space token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(pts_txt, '\"');


			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);

			canonical_pts = NEW_lint(nb_pts);

			if (f_v) {
				cout << "row = " << row;
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				cout << endl;
			}

			if (f_v) {
				cout << "top_level_geometry_global::set_stabilizer_orthogonal_space "
						"before set_stabilizer_of_set" << endl;
			}
			set_stabilizer_of_set(
						SubC,
						cnt, nb, row,
						pts,
						nb_pts,
						canonical_pts,
						verbose_level);
			if (f_v) {
				cout << "top_level_geometry_global::set_stabilizer_orthogonal_space "
						"after set_stabilizer_of_set" << endl;
			}

			FREE_lint(pts);
			FREE_lint(canonical_pts);

		} // row

	}

	FREE_OBJECT(SubC);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_orthogonal_space done" << endl;
	}

}


void top_level_geometry_global::set_stabilizer_of_set(
		substructure_classifier *SubC,
		int cnt, int nb, int row,
		//int *eqn,
		//int sz,
		long int *pts,
		int nb_pts,
		long int *canonical_pts,
		//long int *bitangents,
		//int nb_bitangents,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set" << endl;
	}

	substructure_stats_and_selection *SubSt;

	SubSt = NEW_OBJECT(substructure_stats_and_selection);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set before SubSt->init" << endl;
	}
	SubSt->init(
			SubC,
			pts,
			nb_pts,
			verbose_level - 2);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set after SubSt->init" << endl;
	}
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set" << endl;
		cout << "stabilizer generators are:" << endl;
		SubSt->gens->print_generators_tex(cout);
	}




	int *transporter_to_canonical_form;
	strong_generators *Gens_stabilizer_original_set;

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer before handle_orbit" << endl;
	}

	transporter_to_canonical_form = NEW_int(SubC->A->elt_size_in_int);

	handle_orbit(
			SubSt,
			canonical_pts,
			transporter_to_canonical_form,
			Gens_stabilizer_original_set,
			verbose_level - 2);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer after handle_orbit" << endl;
		cout << "canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		SubC->A->element_print(transporter_to_canonical_form, cout);
		cout << "Stabilizer of the original set:" << endl;
		Gens_stabilizer_original_set->print_generators_tex();
	}

	strong_generators *Gens_stabilizer_canonical_form;

	Gens_stabilizer_canonical_form = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer before init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Gens_stabilizer_canonical_form->init_generators_for_the_conjugate_group_avGa(
			Gens_stabilizer_original_set, transporter_to_canonical_form,
			verbose_level - 2);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer after init_generators_for_the_conjugate_group_avGa" << endl;
	}

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer after handle_orbit" << endl;
		cout << "canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		longinteger_object go;

		Gens_stabilizer_canonical_form->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		SubC->A->element_print(transporter_to_canonical_form, cout);
		cout << "Stabilizer of the canonical form:" << endl;
		Gens_stabilizer_canonical_form->print_generators_tex();
	}


	FREE_OBJECT(SubSt);

	FREE_int(transporter_to_canonical_form);
	FREE_OBJECT(Gens_stabilizer_original_set);
	FREE_OBJECT(Gens_stabilizer_canonical_form);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set done" << endl;
	}

}

void top_level_geometry_global::handle_orbit(
		substructure_stats_and_selection *SubSt,
		long int *canonical_pts,
		int *transporter_to_canonical_form,
		strong_generators *&Gens_stabilizer_original_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit calling compute_stabilizer_function" << endl;
		}

	compute_stabilizer *CS;

	CS = NEW_OBJECT(compute_stabilizer);

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit before CS->init" << endl;
	}
	CS->init(SubSt,
			//SubSt->Pts, SubSt->nb_pts,
			canonical_pts,
			//SubSt->SubC->PC, SubSt->SubC->A, SubSt->SubC->A2,
			//SubSt->SubC->substructure_size, SubSt->selected_orbit,
			//SubSt->nb_interesting_subsets, SubSt->interesting_subsets,
			verbose_level);
	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit after CS->init" << endl;
	}


	SubSt->SubC->A->element_move(CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit done with compute_stabilizer" << endl;
		cout << "top_level_geometry_global::handle_orbit backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
		cout << "top_level_geometry_global::handle_orbit backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
		}


	FREE_OBJECT(CS);

	//overall_backtrack_nodes += CS->nodes;


	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit done" << endl;
	}
}

}}



