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


void top_level_geometry_global::set_stabilizer(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer" << endl;
	}

	poset_classification *PC;
	poset_classification_control *Control;
	poset_with_group_action *Poset;
	int nb_orbits;
	int j;

	Poset = NEW_OBJECT(poset_with_group_action);


	Control = NEW_OBJECT(poset_classification_control);

	Control->f_depth = TRUE;
	Control->depth = intermediate_subset_size;


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(PA->A, PA->A,
			PA->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			intermediate_subset_size,
			verbose_level);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	nb_orbits = PC->nb_orbits_at_level(intermediate_subset_size);

	cout << "We found " << nb_orbits << " orbits at level " << intermediate_subset_size << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {


		strong_generators *Strong_gens;

		PC->get_stabilizer_generators(
				Strong_gens,
				intermediate_subset_size, j, 0 /* verbose_level*/);

		longinteger_object go;

		Strong_gens->group_order(go);

		FREE_OBJECT(Strong_gens);

		cout << j << " : " << go << endl;


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
			cout << "top_level_geometry_global::set_stabilizer "
					"file " << cnt << " / " << nb << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}

	}

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer "
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
			cout << "top_level_geometry_global::set_stabilizer S.nb_rows = " << S.nb_rows << endl;
			cout << "top_level_geometry_global::set_stabilizer S.nb_cols = " << S.nb_cols << endl;
		}




		for (row = 0; row < S.nb_rows - 1; row++) {

			int j, t;
			string eqn_txt;
			string pts_txt;
			string bitangents_txt;
			int *eqn;
			int sz;
			long int *pts;
			int nb_pts;
			long int *canonical_pts;
			long int *bitangents;
			int nb_bitangents;

			if (f_v) {
				cout << "#############################################################################" << endl;
				cout << "cnt = " << cnt << " / " << nb << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
			j = 2;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "top_level_geometry_global::set_stabilizer token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			ST.remove_specific_character(bitangents_txt, '\"');

			if (FALSE) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}

			Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			canonical_pts = NEW_lint(nb_pts);


			if (f_v) {
				cout << "row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				//cout << " pts=";
				//Orbiter->Lint_vec.print(cout, pts, nb_pts);
				//cout << " bitangents=";
				//Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}

			set_stabilizer_of_set(
						PA,
						intermediate_subset_size,
						PC,
						cnt, nb, row,
						eqn,
						sz,
						pts,
						nb_pts,
						canonical_pts,
						bitangents,
						nb_bitangents,
						verbose_level);

			FREE_int(eqn);
			FREE_lint(pts);
			FREE_lint(bitangents);
			FREE_lint(canonical_pts);

		} // row

	}

}

void top_level_geometry_global::set_stabilizer_of_set(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		poset_classification *PC,
		int cnt, int nb, int row,
		int *eqn,
		int sz,
		long int *pts,
		int nb_pts,
		long int *canonical_pts,
		long int *bitangents,
		int nb_bitangents,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set" << endl;
	}
	int nCk;
	int *isotype;
	int *orbit_frequencies;
	int nb_orbits;
	tally *T;

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set before PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}

	PC->trace_all_k_subsets_and_compute_frequencies(
			pts, nb_pts, intermediate_subset_size, nCk, isotype, orbit_frequencies, nb_orbits,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_of_set after PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}




	T = NEW_OBJECT(tally);

	T->init(orbit_frequencies, nb_orbits, FALSE, 0);


	if (f_v) {
		cout << "cnt = " << cnt << " / " << nb << ", row = " << row << " eqn=";
		Orbiter->Int_vec.print(cout, eqn, sz);
		cout << " pts=";
		Orbiter->Lint_vec.print(cout, pts, nb_pts);
		cout << endl;
		cout << "orbit isotype=";
		Orbiter->Int_vec.print(cout, isotype, nCk);
		cout << endl;
		cout << "orbit frequencies=";
		Orbiter->Int_vec.print(cout, orbit_frequencies, nb_orbits);
		cout << endl;
		cout << "orbit frequency types=";
		T->print_naked(FALSE /* f_backwards */);
		cout << endl;
	}

	set_of_sets *SoS;
	int *types;
	int nb_types;
	int i, f, l, idx;
	int selected_type = -1;
	int selected_orbit = -1;
	int selected_frequency;
	longinteger_domain D;
	int j;



	SoS = T->get_set_partition_and_types(types, nb_types, verbose_level);

	longinteger_object go_min;


	for (i = 0; i < nb_types; i++) {
		f = T->type_first[i];
		l = T->type_len[i];
		cout << types[i];
		cout << " : ";
		Orbiter->Lint_vec.print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << " : ";


		for (j = 0; j < SoS->Set_size[i]; j++) {

			idx = SoS->Sets[i][j];

			longinteger_object go;

			PC->get_stabilizer_order(intermediate_subset_size, idx, go);

			if (types[i]) {

				// types[i] must be greater than zero
				// so the type really appears.

				if (selected_type == -1) {
					selected_type = j;
					selected_orbit = idx;
					selected_frequency = types[i];
					go.assign_to(go_min);
				}
				else {
					if (D.compare_unsigned(go, go_min) < 0) {
						selected_type = j;
						selected_orbit = idx;
						selected_frequency = types[i];
						go.assign_to(go_min);
					}
				}
			}

			cout << go;
			if (j < SoS->Set_size[i] - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	if (f_v) {
		cout << "selected_type = " << selected_type
			<< " selected_orbit = " << selected_orbit
			<< " selected_frequency = " << selected_frequency
			<< " go_min = " << go_min << endl;
	}

	strong_generators *gens;

	PC->get_stabilizer_generators(
		gens,
		intermediate_subset_size, selected_orbit, verbose_level);


	int *transporter_to_canonical_form;
	strong_generators *Gens_stabilizer_original_set;

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer before handle_orbit" << endl;
	}

	transporter_to_canonical_form = NEW_int(PA->A->elt_size_in_int);


	handle_orbit(*T,
			isotype,
			selected_orbit, selected_frequency, nCk,
			intermediate_subset_size,
			PC, PA->A, PA->A,
			pts, nb_pts,
			canonical_pts,
			transporter_to_canonical_form,
			Gens_stabilizer_original_set,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer after handle_orbit" << endl;
		cout << "canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		PA->A->element_print(transporter_to_canonical_form, cout);
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
			verbose_level);
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
		PA->A->element_print(transporter_to_canonical_form, cout);
		cout << "Stabilizer of the canonical form:" << endl;
		Gens_stabilizer_canonical_form->print_generators_tex();
	}




	FREE_int(transporter_to_canonical_form);
	FREE_OBJECT(gens);
	FREE_OBJECT(Gens_stabilizer_original_set);
	FREE_OBJECT(Gens_stabilizer_canonical_form);
	FREE_OBJECT(SoS);
	FREE_int(types);

	FREE_int(isotype);
	FREE_int(orbit_frequencies);
	FREE_OBJECT(T);


}

void top_level_geometry_global::handle_orbit(tally &C,
		int *isotype,
		int selected_orbit, int selected_frequency, int n_choose_k,
		int intermediate_subset_size,
		poset_classification *PC, action *A, action *A2,
		long int *pts,
		int nb_pts,
		long int *canonical_pts,
		int *transporter_to_canonical_form,
		strong_generators *&Gens_stabilizer_original_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// interesting_subsets are the lvl-subsets of the given set
	// which are of the chosen type.
	// There is nb_interesting_subsets of them.
	long int *interesting_subsets;
	int nb_interesting_subsets;

	int i, j;

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit" << endl;
		cout << "selected_orbit = " << selected_orbit << endl;
	}

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit we decide to go for subsets of size " << intermediate_subset_size << ", selected_frequency = " << selected_frequency << endl;
	}

	j = 0;
	interesting_subsets = NEW_lint(selected_frequency);
	for (i = 0; i < n_choose_k; i++) {
		if (isotype[i] == selected_orbit) {
			interesting_subsets[j++] = i;
			//cout << "subset of rank " << i << " is isomorphic to orbit " << orb_idx << " j=" << j << endl;
			}
		}
	if (j != selected_frequency) {
		cout << "j != nb_interesting_subsets" << endl;
		exit(1);
		}
	nb_interesting_subsets = selected_frequency;
#if 0
	if (f_vv) {
		print_interesting_subsets(nb_pts, intermediate_subset_size, nb_interesting_subsets, interesting_subsets);
		}
#endif


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit calling compute_stabilizer_function" << endl;
		}

	compute_stabilizer *CS;

	CS = NEW_OBJECT(compute_stabilizer);

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit before CS->init" << endl;
	}
	CS->init(pts, nb_pts,
			canonical_pts,
			PC, A, A2,
			intermediate_subset_size, selected_orbit,
			nb_interesting_subsets, interesting_subsets,
			verbose_level);
	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit after CS->init" << endl;
	}


	A->element_move(CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit done with compute_stabilizer" << endl;
		cout << "top_level_geometry_global::handle_orbit backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
		cout << "top_level_geometry_global::handle_orbit backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
		}


	FREE_OBJECT(CS);

	//overall_backtrack_nodes += CS->nodes;

	FREE_lint(interesting_subsets);

	if (f_v) {
		cout << "top_level_geometry_global::handle_orbit done" << endl;
	}
}

}}



