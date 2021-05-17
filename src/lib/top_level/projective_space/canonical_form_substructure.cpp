/*
 * canonical_form_substructure.cpp
 *
 *  Created on: May 13, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



canonical_form_substructure::canonical_form_substructure()
{

	Canonical_form_classifier = NULL;

	cnt = 0;
	row = 0;
	counter = 0;
	eqn = NULL;
	sz = 0;
	pts = NULL;
	nb_pts = 0;
	bitangents = NULL;
	nb_bitangents = 0;

	canonical_pts = NULL;

	nCk = 0;
	isotype = NULL;
	orbit_frequencies = NULL;
	nb_orbits = 0;
	T = NULL;

	SoS = NULL;
	types = NULL;
	nb_types = 0;
	selected_type = 0;
	selected_orbit = 0;
	selected_frequency = 0;

	//longinteger_object go_min;

	gens = NULL;

	interesting_subsets = NULL;
	nb_interesting_subsets = 0;

	CS = NULL;

	Gens_stabilizer_original_set = NULL;
	Gens_stabilizer_canonical_form = NULL;



	Orb = NULL;

	gens_stab_of_canonical_equation = NULL;

	trans1 = NULL;
	trans2 = NULL;
	intermediate_equation = NULL;



	Elt = NULL;
	eqn2 = NULL;

	canonical_equation = NULL;
	transporter_to_canonical_form = NULL;
}

canonical_form_substructure::~canonical_form_substructure()
{
}


void canonical_form_substructure::classify_curve_with_substructure(
		canonical_form_classifier *Canonical_form_classifier,
		int counter, int cnt, int row,
		int *eqn,
		int sz,
		long int *pts,
		int nb_pts,
		long int *bitangents,
		int nb_bitangents,
		int *canonical_equation,
		int *transporter_to_canonical_form,
		longinteger_object &go_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	//int f_vvv = (verbose_level >= 5);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure" << endl;
	}

	canonical_form_substructure::Canonical_form_classifier = Canonical_form_classifier;
	canonical_form_substructure::counter = counter;
	canonical_form_substructure::cnt = cnt;
	canonical_form_substructure::row = row;

	canonical_form_substructure::eqn = eqn;
	canonical_form_substructure::sz = sz;
	canonical_form_substructure::pts = pts;
	canonical_form_substructure::nb_pts = nb_pts;


	canonical_form_substructure::bitangents = NEW_lint(nb_bitangents);
	Orbiter->Lint_vec.copy(bitangents, canonical_form_substructure::bitangents, nb_bitangents);
	canonical_form_substructure::nb_bitangents = nb_bitangents;
	canonical_form_substructure::canonical_equation = canonical_equation;
	canonical_form_substructure::transporter_to_canonical_form = transporter_to_canonical_form;

	//long int *canonical_pts;

	canonical_pts = NEW_lint(nb_pts);

	trans1 = NEW_int(Canonical_form_classifier->Descr->PA->A->elt_size_in_int);
	trans2 = NEW_int(Canonical_form_classifier->Descr->PA->A->elt_size_in_int);
	intermediate_equation = NEW_int(Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "row = " << row << " eqn=";
		Orbiter->Int_vec.print(cout, eqn, sz);
		cout << " pts=";
		Orbiter->Lint_vec.print(cout, pts, nb_pts);
		//cout << " bitangents=";
		//Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
		cout << endl;
	}

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}

	Canonical_form_classifier->PC->trace_all_k_subsets_and_compute_frequencies(
			pts, nb_pts, Canonical_form_classifier->Descr->substructure_size,
			nCk, isotype, orbit_frequencies, nb_orbits,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}




	T = NEW_OBJECT(tally);

	T->init(orbit_frequencies, nb_orbits, FALSE, 0);


	if (f_vv) {
		cout << "counter = " << counter << " cnt = " << cnt << " / "
				<< Canonical_form_classifier->Descr->nb_files
				<< ", row = " << row << " eqn=";
		Orbiter->Int_vec.print(cout, eqn, sz);

		cout << " pts=";
		Orbiter->Lint_vec.print(cout, pts, nb_pts);
		cout << endl;

#if 0
		cout << "orbit isotype=";
		Orbiter->Int_vec.print(cout, isotype, nCk);
		cout << endl;
		cout << "orbit frequencies=";
		Orbiter->Int_vec.print(cout, orbit_frequencies, nb_orbits);
		cout << endl;
#endif
		cout << " orbit frequency types=";
		T->print_naked(FALSE /* f_backwards */);
		cout << endl;
	}

	selected_type = -1;
	selected_orbit = -1;
	selected_frequency = 0;
	longinteger_domain D;
	int i, f, l, idx;
	int j;



	SoS = T->get_set_partition_and_types(types, nb_types, verbose_level);

	for (i = 0; i < nb_types; i++) {
		f = T->type_first[i];
		l = T->type_len[i];

		if (f_vv) {
			cout << types[i];
			cout << " : ";
			Orbiter->Lint_vec.print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << " : ";
		}


		for (j = 0; j < SoS->Set_size[i]; j++) {

			idx = SoS->Sets[i][j];

			longinteger_object go;

			Canonical_form_classifier->PC->get_stabilizer_order(
					Canonical_form_classifier->Descr->substructure_size,
					idx, go);

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

			if (f_vv) {
				cout << go;
				if (j < SoS->Set_size[i] - 1) {
					cout << ", ";
				}
			}
		}
		if (f_vv) {
			cout << endl;
		}
	}

	if (f_vv) {
		cout << "selected_type = " << selected_type
			<< " selected_orbit = " << selected_orbit
			<< " selected_frequency = " << selected_frequency
			<< " go_min = " << go_min << endl;
	}

	Canonical_form_classifier->PC->get_stabilizer_generators(
		gens,
		Canonical_form_classifier->Descr->substructure_size,
		selected_orbit, 0 /*verbose_level*/);


	if (f_vv) {
		cout << "canonical_form_substructure::classify_curve_with_substructure before handle_orbit" << endl;
	}



	handle_orbit(
			trans1,
			Gens_stabilizer_original_set,
			Gens_stabilizer_canonical_form,
			verbose_level - 2);




	if (FALSE) {
		cout << "canonical_form_substructure::classify_curve_with_substructure after handle_orbit" << endl;
		cout << "canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Canonical_form_classifier->Descr->PA->A->element_print(trans1, cout);
		//cout << "Stabilizer of the original set:" << endl;
		//Gens_stabilizer_original_set->print_generators_tex();
	}


	if (FALSE) {
		cout << "canonical_form_substructure::classify_curve_with_substructure after handle_orbit" << endl;
		cout << "canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		longinteger_object go;

		Gens_stabilizer_canonical_form->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Canonical_form_classifier->Descr->PA->A->element_print(trans1, cout);
		//cout << "Stabilizer of the canonical form:" << endl;
		//Gens_stabilizer_canonical_form->print_generators_tex();
	}

	longinteger_object go;

	Gens_stabilizer_canonical_form->group_order(go);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure canonical point set: ";
		Orbiter->Lint_vec.print(cout, canonical_pts, nb_pts);
		cout << "_{" << go << "}" << endl;
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure before AonHPD->compute_image_int_low_level" << endl;
	}
	Canonical_form_classifier->AonHPD->compute_image_int_low_level(
			trans1, eqn, intermediate_equation, verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure after AonHPD->compute_image_int_low_level" << endl;
	}

	Orb = NEW_OBJECT(orbit_of_equations);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before Orb->init" << endl;
	}
	Orb->init(Canonical_form_classifier->Descr->PA->A,
			Canonical_form_classifier->Descr->PA->F,
			Canonical_form_classifier->AonHPD,
			Gens_stabilizer_canonical_form /* A->Strong_gens*/,
			intermediate_equation,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after Orb->init" << endl;
		cout << "canonical_form_substructure::classify_curve_with_substructure found an orbit of length " << Orb->used_length << endl;
	}


	// we need to compute the canonical form!

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before Orb->get_canonical_form" << endl;
	}

	//strong_generators *gens_stab_of_canonical_equation;

	Orb->get_canonical_form(
				canonical_equation,
				trans2,
				gens_stab_of_canonical_equation,
				go,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after Orb->get_canonical_form" << endl;
	}


	Canonical_form_classifier->Descr->PA->A->element_mult(
			trans1, trans2, transporter_to_canonical_form, 0);


	gens_stab_of_canonical_equation->group_order(go_eqn);

	Canonical_form_classifier->Descr->PA->F->PG_element_normalize(
			canonical_equation, 1, Canonical_form_classifier->Poly_ring->get_nb_monomials());

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure canonical equation: ";
		Orbiter->Int_vec.print(cout, canonical_equation,
				Canonical_form_classifier->Poly_ring->get_nb_monomials());
		cout << "_{" << go_eqn << "}" << endl;
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure done" << endl;
	}
}


void canonical_form_substructure::handle_orbit(
		int *transporter_to_canonical_form,
		strong_generators *&Gens_stabilizer_original_set,
		strong_generators *&Gens_stabilizer_canonical_form,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// interesting_subsets are the lvl-subsets of the given set
	// which are of the chosen type.
	// There is nb_interesting_subsets of them.
	poset_classification *PC;
	action *A;
	action *A2;
	int intermediate_subset_size;

	int i, j;

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit" << endl;
		cout << "selected_orbit = " << selected_orbit << endl;
	}

	PC = Canonical_form_classifier->PC;
	A = Canonical_form_classifier->Descr->PA->A;
	A2 = Canonical_form_classifier->Descr->PA->A;
	intermediate_subset_size = Canonical_form_classifier->Descr->substructure_size;

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit we decide to go for subsets of size " << intermediate_subset_size << ", selected_frequency = " << selected_frequency << endl;
	}

	j = 0;
	interesting_subsets = NEW_lint(selected_frequency);
	for (i = 0; i < nCk; i++) {
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


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit calling compute_stabilizer_function" << endl;
		}


	CS = NEW_OBJECT(compute_stabilizer);

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit before CS->init" << endl;
	}
	CS->init(pts, nb_pts,
			canonical_pts,
			PC, A, A2,
			intermediate_subset_size, selected_orbit,
			nb_interesting_subsets, interesting_subsets,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit after CS->init" << endl;
	}


	A->element_move(CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(strong_generators);
	Gens_stabilizer_canonical_form = NEW_OBJECT(strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);




	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure before init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Gens_stabilizer_canonical_form->init_generators_for_the_conjugate_group_avGa(
			Gens_stabilizer_original_set, transporter_to_canonical_form,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure after init_generators_for_the_conjugate_group_avGa" << endl;
	}





	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit done with compute_stabilizer" << endl;
		cout << "canonical_form_substructure::handle_orbit backtrack_nodes_first_time = " << CS->backtrack_nodes_first_time << endl;
		cout << "canonical_form_substructure::handle_orbit backtrack_nodes_total_in_loop = " << CS->backtrack_nodes_total_in_loop << endl;
		}



	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit done" << endl;
	}
}

}}

