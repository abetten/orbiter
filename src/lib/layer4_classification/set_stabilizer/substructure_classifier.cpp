/*
 * substructure_classifier.cpp
 *
 *  Created on: Jun 9, 2021
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace set_stabilizer {


substructure_classifier::substructure_classifier()
{
	Record_birth();
	//std::string &fname_base_out;
	substructure_size = 0;
	PC = NULL;
	Control = NULL;
	A = NULL;
	A2 = NULL;
	Poset = NULL;
	nb_orbits = 0;
}


substructure_classifier::~substructure_classifier()
{
	Record_death();

}


void substructure_classifier::classify_substructures(
		std::string &fname_base_out,
		actions::action *A,
		actions::action *A2,
		groups::strong_generators *gens,
		int substructure_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_classifier::classify_substructures, "
				"substructure_size=" << substructure_size << endl;
		cout << "substructure_classifier::classify_substructures, "
				"fname_base_out=" << fname_base_out << endl;
		cout << "substructure_classifier::classify_substructures, "
				"action A=";
		A->print_info();
		cout << endl;
		cout << "substructure_classifier::classify_substructures, "
				"action A2=";
		A2->print_info();
		cout << endl;
		cout << "substructure_classifier::classify_substructures "
				"generators:" << endl;
		gens->print_generators_tex(cout);
	}

	substructure_classifier::fname_base_out.assign(fname_base_out);
	substructure_classifier::A = A;
	substructure_classifier::A2 = A2;
	substructure_classifier::substructure_size = substructure_size;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);


	Control = NEW_OBJECT(poset_classification::poset_classification_control);

	Control->f_depth = true;
	Control->depth = substructure_size;


	if (f_v) {
		cout << "substructure_classifier::classify_substructures "
				"control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(A, A2,
			gens,
			verbose_level);

	if (f_v) {
		cout << "substructure_classifier::classify_substructures "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			substructure_size,
			verbose_level);
	if (f_v) {
		cout << "substructure_classifier::classify_substructures "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	nb_orbits = PC->nb_orbits_at_level(substructure_size);

	cout << "We found " << nb_orbits << " orbits "
			"at level " << substructure_size << ":" << endl;

	int j;

	for (j = 0; j < nb_orbits; j++) {


		groups::strong_generators *Strong_gens;

		PC->get_stabilizer_generators(
				Strong_gens,
				substructure_size, j,
				0 /* verbose_level*/);

		algebra::ring_theory::longinteger_object go;

		Strong_gens->group_order(go);

		FREE_OBJECT(Strong_gens);

		cout << j << " : " << go << endl;


	}

	if (f_v) {
		cout << "substructure_classifier::classify_substructures done" << endl;
	}



}



void substructure_classifier::set_stabilizer_in_any_space(
		actions::action *A, actions::action *A2,
		groups::strong_generators *Strong_gens,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_in_any_space" << endl;
	}

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_in_any_space "
				"before SubC->classify_substructures" << endl;
	}

	classify_substructures(
			fname_base_out,
			A, A,
			Strong_gens,
			intermediate_subset_size, verbose_level - 5);

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_in_any_space "
				"after SubC->classify_substructures" << endl;
		cout << "substructure_classifier::set_stabilizer_in_any_space "
				"We found " << nb_orbits
				<< " orbits at level " << intermediate_subset_size << ":" << endl;
	}





	int nb_objects_to_test;
	int cnt;
	int row;

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < nb; cnt++) {

		other::data_structures::string_tools ST;

		string fname;

		fname = ST.printf_d(fname_mask, cnt);

		other::data_structures::spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		nb_objects_to_test += S.nb_rows - 1;
		if (f_v) {
			cout << "substructure_classifier::set_stabilizer_in_any_space "
					"file " << cnt << " / " << nb << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}

	}

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_in_any_space "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}


	int counter;

	counter = 0;

	for (cnt = 0; cnt < nb; cnt++) {

		other::data_structures::string_tools ST;

		string fname;

		fname = ST.printf_d(fname_mask, cnt);

		other::data_structures::spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "substructure_classifier::set_stabilizer_in_any_space "
					"S.nb_rows = " << S.nb_rows << endl;
			cout << "substructure_classifier::set_stabilizer_in_any_space "
					"S.nb_cols = " << S.nb_cols << endl;
		}

		int col_idx;

		col_idx = S.find_column(column_label);


		for (row = 0; row < S.nb_rows - 1; row++, counter++) {

			int j;
			string pts_txt;
			long int *pts;
			int nb_pts;
			long int *canonical_pts;

			if (f_v) {
				cout << "#############################################################################" << endl;
				cout << "cnt = " << cnt << " / " << nb
						<< " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = col_idx;
			pts_txt = S.get_entry_ij(row + 1, j);

			other::data_structures::string_tools ST;

			ST.remove_specific_character(pts_txt, '\"');


			Lint_vec_scan(pts_txt, pts, nb_pts);

			canonical_pts = NEW_lint(nb_pts);

			if (f_v) {
				cout << "row = " << row;
				cout << " pts=";
				Lint_vec_print(cout, pts, nb_pts);
				cout << endl;
			}



			std::string fname;

			fname = fname_out + "_cnt" + std::to_string(counter);


			if (f_v) {
				cout << "substructure_classifier::set_stabilizer_in_any_space "
						"before set_stabilizer_of_set" << endl;
			}
			set_stabilizer_of_set(
						fname,
						cnt, nb, row,
						pts,
						nb_pts,
						canonical_pts,
						verbose_level - 3);
			if (f_v) {
				cout << "substructure_classifier::set_stabilizer_in_any_space "
						"after set_stabilizer_of_set" << endl;
			}

			FREE_lint(pts);
			FREE_lint(canonical_pts);

		} // row

	}

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_in_any_space done" << endl;
	}

}


void substructure_classifier::set_stabilizer_of_set(
		std::string &fname_out,
		int cnt, int nb, int row,
		long int *pts,
		int nb_pts,
		long int *canonical_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_of_set" << endl;
	}

	substructure_stats_and_selection *SubSt;

	SubSt = NEW_OBJECT(substructure_stats_and_selection);

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_of_set "
				"before SubSt->init" << endl;
	}
	SubSt->init(
			fname_out,
			this,
			pts,
			nb_pts,
			verbose_level - 2);
	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_of_set "
				"after SubSt->init" << endl;
	}
	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_of_set" << endl;
		cout << "stabilizer generators are:" << endl;
		SubSt->gens->print_generators_tex(cout);
	}




	int *transporter_to_canonical_form;
	groups::strong_generators *Gens_stabilizer_original_set;

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer "
				"before handle_orbit" << endl;
	}

	transporter_to_canonical_form = NEW_int(A->elt_size_in_int);

	handle_orbit(
			SubSt,
			canonical_pts,
			transporter_to_canonical_form,
			Gens_stabilizer_original_set,
			verbose_level - 2);

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout, canonical_pts, nb_pts);
		algebra::ring_theory::longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		A->Group_element->element_print(transporter_to_canonical_form, cout);
		cout << "Stabilizer of the original set:" << endl;
		Gens_stabilizer_original_set->print_generators_tex();
	}

	groups::strong_generators *Gens_stabilizer_canonical_form;

	Gens_stabilizer_canonical_form = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer "
				"before init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Gens_stabilizer_canonical_form->init_generators_for_the_conjugate_group_avGa(
			Gens_stabilizer_original_set, transporter_to_canonical_form,
			verbose_level - 2);
	if (f_v) {
		cout << "substructure_classifier::set_stabilizer "
				"after init_generators_for_the_conjugate_group_avGa" << endl;
	}

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout, canonical_pts, nb_pts);
		algebra::ring_theory::longinteger_object go;

		Gens_stabilizer_canonical_form->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		A->Group_element->element_print(transporter_to_canonical_form, cout);
		cout << "Stabilizer of the canonical form:" << endl;
		Gens_stabilizer_canonical_form->print_generators_tex();
	}


	FREE_OBJECT(SubSt);

	FREE_int(transporter_to_canonical_form);
	FREE_OBJECT(Gens_stabilizer_original_set);
	FREE_OBJECT(Gens_stabilizer_canonical_form);

	if (f_v) {
		cout << "substructure_classifier::set_stabilizer_of_set done" << endl;
	}

}

void substructure_classifier::handle_orbit(
		substructure_stats_and_selection *SubSt,
		long int *canonical_pts,
		int *transporter_to_canonical_form,
		groups::strong_generators *&Gens_stabilizer_original_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "substructure_classifier::handle_orbit "
				"calling compute_stabilizer_function" << endl;
	}

	compute_stabilizer *CS;

	CS = NEW_OBJECT(compute_stabilizer);

	if (f_v) {
		cout << "substructure_classifier::handle_orbit "
				"before CS->init" << endl;
	}
	CS->init(SubSt,
			//SubSt->Pts, SubSt->nb_pts,
			canonical_pts,
			//SubSt->SubC->PC, SubSt->SubC->A, SubSt->SubC->A2,
			//SubSt->SubC->substructure_size, SubSt->selected_orbit,
			//SubSt->nb_interesting_subsets, SubSt->interesting_subsets,
			verbose_level);
	if (f_v) {
		cout << "substructure_classifier::handle_orbit after CS->init" << endl;
	}


	SubSt->SubC->A->Group_element->element_move(
			CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(groups::strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);

	if (f_v) {
		cout << "substructure_classifier::handle_orbit "
				"done with compute_stabilizer" << endl;
		cout << "substructure_classifier::handle_orbit "
				"backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
		cout << "substructure_classifier::handle_orbit "
				"backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
	}


	FREE_OBJECT(CS);

	//overall_backtrack_nodes += CS->nodes;


	if (f_v) {
		cout << "substructure_classifier::handle_orbit done" << endl;
	}
}




}}}


