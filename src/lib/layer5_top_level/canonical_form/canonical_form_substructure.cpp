/*
 * canonical_form_substructure.cpp
 *
 *  Created on: May 13, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_substructure::canonical_form_substructure()
{


	Variety = NULL;

	SubSt = NULL;

	CS = NULL;

	Gens_stabilizer_original_set = NULL;
	Gens_stabilizer_canonical_form = NULL;



	Orb = NULL;

	trans1 = NULL;
	trans2 = NULL;
	intermediate_equation = NULL;



	Elt = NULL;
	eqn2 = NULL;

}

canonical_form_substructure::~canonical_form_substructure()
{
}


void canonical_form_substructure::classify_curve_with_substructure(
		canonical_form_of_variety *Variety,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 5);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"verbose_level=" << verbose_level << endl;
	}

	canonical_form_substructure::Variety = Variety;


	trans1 = NEW_int(Variety->Canonical_form_classifier->PA->A->elt_size_in_int);
	trans2 = NEW_int(Variety->Canonical_form_classifier->PA->A->elt_size_in_int);
	intermediate_equation = NEW_int(Variety->Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "fname_case_out = " << Variety->fname_case_out << endl;
		cout << "cnt = " << Variety->Vo->cnt << " eqn=";
		Int_vec_print(cout, Variety->Vo->Variety_object->eqn, Variety->Canonical_form_classifier->Poly_ring->get_nb_monomials());
		cout << " pts=";
		Lint_vec_print(cout,
				Variety->Vo->Variety_object->Point_sets->Sets[0],
				Variety->Vo->Variety_object->Point_sets->Set_size[0]);
		cout << endl;

		Variety->Canonical_form_classifier->Poly_ring->print_equation_tex(
				cout, Variety->Vo->Variety_object->eqn);
		cout << endl;

		//Canonical_form_classifier->Poly_ring->get_P()->print_set_of_points(cout, pts, nb_pts);


		//cout << " bitangents=";
		//Lint_vec_print(cout, bitangents, nb_bitangents);
	}





	SubSt = NEW_OBJECT(set_stabilizer::substructure_stats_and_selection);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before SubSt->init" << endl;
	}
	SubSt->init(
			Variety->fname_case_out,
			Variety->Canonical_form_classifier->Classification_of_varieties->SubC,
			Variety->Vo->Variety_object->Point_sets->Sets[0],
			Variety->Vo->Variety_object->Point_sets->Set_size[0],
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after SubSt->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before handle_orbit" << endl;
	}



	handle_orbit(
			trans1,
			Gens_stabilizer_original_set,
			Gens_stabilizer_canonical_form,
			verbose_level);




	if (false) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout,
				Variety->canonical_pts,
				Variety->Vo->Variety_object->Point_sets->Set_size[0]);
		ring_theory::longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Variety->Canonical_form_classifier->PA->A->Group_element->element_print(trans1, cout);
		//cout << "Stabilizer of the original set:" << endl;
		//Gens_stabilizer_original_set->print_generators_tex();
	}


	if (false) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout,
				Variety->canonical_pts,
				Variety->Vo->Variety_object->Point_sets->Set_size[0]);
		ring_theory::longinteger_object go;

		Gens_stabilizer_canonical_form->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Variety->Canonical_form_classifier->PA->A->Group_element->element_print(trans1, cout);
		//cout << "Stabilizer of the canonical form:" << endl;
		//Gens_stabilizer_canonical_form->print_generators_tex();
	}

	ring_theory::longinteger_object go;

	Gens_stabilizer_canonical_form->group_order(go);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"canonical point set: ";
		Lint_vec_print(cout,
				Variety->canonical_pts,
				Variety->Vo->Variety_object->Point_sets->Set_size[0]);
		cout << "_{" << go << "}" << endl;
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before AonHPD->compute_image_int_low_level" << endl;
	}
	Variety->Canonical_form_classifier->AonHPD->compute_image_int_low_level(
			trans1,
			Variety->Vo->Variety_object->eqn,
			intermediate_equation,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after AonHPD->compute_image_int_low_level" << endl;
	}

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before Orb->init" << endl;
	}
	Orb->init(
			Variety->Canonical_form_classifier->PA->A,
			Variety->Canonical_form_classifier->PA->F,
			Variety->Canonical_form_classifier->AonHPD,
			Gens_stabilizer_canonical_form /* A->Strong_gens*/,
			intermediate_equation,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after Orb->init" << endl;
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"found an orbit of length " << Orb->used_length << endl;
	}


	// we need to compute the canonical form!

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before Orb->get_canonical_form" << endl;
	}

	Orb->get_canonical_form(
			Variety->canonical_equation,
				trans2,
				Variety->gens_stab_of_canonical_equation,
				go,
				verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after Orb->get_canonical_form" << endl;
	}


	Variety->Canonical_form_classifier->PA->A->Group_element->element_mult(
			trans1, trans2, Variety->transporter_to_canonical_form, 0);


	Variety->gens_stab_of_canonical_equation->group_order(*Variety->go_eqn);

	Variety->Canonical_form_classifier->PA->F->Projective_space_basic->PG_element_normalize(
			Variety->canonical_equation, 1,
			Variety->Canonical_form_classifier->Poly_ring->get_nb_monomials());

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"canonical equation: ";
		Int_vec_print(cout,
				Variety->canonical_equation,
				Variety->Canonical_form_classifier->Poly_ring->get_nb_monomials());
		cout << "_{" << Variety->go_eqn << "}" << endl;
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure done" << endl;
	}
}


void canonical_form_substructure::handle_orbit(
		int *transporter_to_canonical_form,
		groups::strong_generators *&Gens_stabilizer_original_set,
		groups::strong_generators *&Gens_stabilizer_canonical_form,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// interesting_subsets are the lvl-subsets of the given set
	// which are of the chosen type.
	// There is nb_interesting_subsets of them.
	//poset_classification *PC;
	//action *A;
	//action *A2;
	//int intermediate_subset_size;

	//int i, j;

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit" << endl;
		cout << "fname = " << Variety->fname_case_out << endl;
		cout << "selected_orbit = " << SubSt->selected_orbit << endl;
	}


	//overall_backtrack_nodes = 0;
	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit "
				"calling compute_stabilizer_function" << endl;
		}


	CS = NEW_OBJECT(set_stabilizer::compute_stabilizer);

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit "
				"before CS->init" << endl;
	}
	CS->init(
			SubSt,
			Variety->canonical_pts,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit "
				"after CS->init" << endl;
	}


	Variety->Canonical_form_classifier->Classification_of_varieties->SubC->A->Group_element->element_move(
			CS->T1, transporter_to_canonical_form, 0);

	Gens_stabilizer_original_set = NEW_OBJECT(groups::strong_generators);
	Gens_stabilizer_canonical_form = NEW_OBJECT(groups::strong_generators);

	Gens_stabilizer_original_set->init_from_sims(CS->Stab, verbose_level);




	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Gens_stabilizer_canonical_form->init_generators_for_the_conjugate_group_avGa(
			Gens_stabilizer_original_set,
			transporter_to_canonical_form,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after init_generators_for_the_conjugate_group_avGa" << endl;
	}





	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit "
				"done with compute_stabilizer" << endl;
		cout << "canonical_form_substructure::handle_orbit "
				"backtrack_nodes_first_time = "
				<< CS->backtrack_nodes_first_time << endl;
		cout << "canonical_form_substructure::handle_orbit "
				"backtrack_nodes_total_in_loop = "
				<< CS->backtrack_nodes_total_in_loop << endl;
		}



	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit done" << endl;
	}
}

}}}


