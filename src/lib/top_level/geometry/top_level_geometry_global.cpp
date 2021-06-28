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

	SubC->set_stabilizer_in_any_space(
			PA->A, PA->A, PA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			verbose_level);

#if 0
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

			SubC->set_stabilizer_of_set(
						cnt, nb, row,
						pts,
						nb_pts,
						canonical_pts,
						verbose_level);

			FREE_lint(pts);
			FREE_lint(canonical_pts);

		} // row

	}
#endif

	FREE_OBJECT(SubC);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space done" << endl;
	}

}


}}



