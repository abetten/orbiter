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
namespace projective_geometry {



canonical_form_substructure::canonical_form_substructure()
{

	//std::string fname_case_out;

	//Canonical_form_classifier = NULL;
	//Canonical_form_of_variety = NULL;

	Variety = NULL;

	//Qco = NULL;

#if 0
	cnt = 0;
	row = 0;
	counter = 0;
	eqn = NULL;
	sz = 0;
	pts = NULL;
	nb_pts = 0;
	bitangents = NULL;
	nb_bitangents = 0;
#endif

	//canonical_pts = NULL;

	SubSt = NULL;

	CS = NULL;

	Gens_stabilizer_original_set = NULL;
	Gens_stabilizer_canonical_form = NULL;



	Orb = NULL;

	//gens_stab_of_canonical_equation = NULL;

	trans1 = NULL;
	trans2 = NULL;
	intermediate_equation = NULL;



	Elt = NULL;
	eqn2 = NULL;

	//canonical_equation = NULL;
	//transporter_to_canonical_form = NULL;
}

canonical_form_substructure::~canonical_form_substructure()
{
}


void canonical_form_substructure::classify_curve_with_substructure(
		canonical_form_of_variety *Variety,
		//canonical_form_classifier *Canonical_form_classifier,
		//std::string &fname_case_out,
		//quartic_curve_object *Qco,
		//ring_theory::longinteger_object &go_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 5);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"verbose_level=" << verbose_level << endl;
	}

	//canonical_form_substructure::fname_case_out.assign(fname_case_out);
	//canonical_form_substructure::Canonical_form_classifier = Canonical_form_classifier;

	//canonical_form_substructure::Qco = Qco;

	canonical_form_substructure::Variety = Variety;

#if 0
	canonical_form_substructure::counter = Qco->cnt;
	canonical_form_substructure::cnt = Qco->po;
	canonical_form_substructure::row = Qco->so;

	eqn = Qco->eqn;
	sz = Qco->sz;
	pts = Qco->pts;
	nb_pts = Qco->nb_pts;


	nb_bitangents = Qco->nb_bitangents;
	bitangents = NEW_lint(Qco->nb_bitangents);
	Lint_vec_copy(Qco->bitangents, bitangents, Qco->nb_bitangents);
#endif

	//canonical_form_substructure::canonical_equation =
	//		NEW_int(Canonical_form_classifier->Poly_ring->get_nb_monomials());
	//canonical_form_substructure::transporter_to_canonical_form =
	//		NEW_int(Canonical_form_classifier->Descr->PA->A->elt_size_in_int);

	//long int *canonical_pts;

	//canonical_pts = NEW_lint(Qco->nb_pts);

	trans1 = NEW_int(Variety->Canonical_form_classifier->Descr->PA->A->elt_size_in_int);
	trans2 = NEW_int(Variety->Canonical_form_classifier->Descr->PA->A->elt_size_in_int);
	intermediate_equation = NEW_int(Variety->Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "fname_case_out = " << Variety->fname_case_out << endl;
		cout << "cnt = " << Variety->Qco->cnt << " eqn=";
		Int_vec_print(cout, Variety->Qco->eqn, Variety->Qco->sz);
		cout << " pts=";
		Lint_vec_print(cout, Variety->Qco->pts, Variety->Qco->nb_pts);
		cout << endl;

		Variety->Canonical_form_classifier->Poly_ring->print_equation_tex(
				cout, Variety->Qco->eqn);
		cout << endl;

		//Canonical_form_classifier->Poly_ring->get_P()->print_set_of_points(cout, pts, nb_pts);


		//cout << " bitangents=";
		//Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
	}





	SubSt = NEW_OBJECT(set_stabilizer::substructure_stats_and_selection);

	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before SubSt->init" << endl;
	}
	SubSt->init(
			Variety->fname_case_out,
			Variety->Canonical_form_classifier->SubC,
			Variety->Qco->pts, Variety->Qco->nb_pts,
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




	if (FALSE) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout, Variety->canonical_pts, Variety->Qco->nb_pts);
		ring_theory::longinteger_object go;

		Gens_stabilizer_original_set->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Variety->Canonical_form_classifier->Descr->PA->A->element_print(trans1, cout);
		//cout << "Stabilizer of the original set:" << endl;
		//Gens_stabilizer_original_set->print_generators_tex();
	}


	if (FALSE) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after handle_orbit" << endl;
		cout << "canonical point set: ";
		Lint_vec_print(cout, Variety->canonical_pts, Variety->Qco->nb_pts);
		ring_theory::longinteger_object go;

		Gens_stabilizer_canonical_form->group_order(go);
		cout << "_{" << go << "}" << endl;
		cout << endl;
		cout << "transporter to canonical form:" << endl;
		Variety->Canonical_form_classifier->Descr->PA->A->element_print(trans1, cout);
		//cout << "Stabilizer of the canonical form:" << endl;
		//Gens_stabilizer_canonical_form->print_generators_tex();
	}

	ring_theory::longinteger_object go;

	Gens_stabilizer_canonical_form->group_order(go);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"canonical point set: ";
		Lint_vec_print(cout, Variety->canonical_pts, Variety->Qco->nb_pts);
		cout << "_{" << go << "}" << endl;
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before AonHPD->compute_image_int_low_level" << endl;
	}
	Variety->Canonical_form_classifier->AonHPD->compute_image_int_low_level(
			trans1, Variety->Qco->eqn, intermediate_equation, verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"after AonHPD->compute_image_int_low_level" << endl;
	}

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


	if (f_v) {
		cout << "canonical_form_substructure::classify_curve_with_substructure "
				"before Orb->init" << endl;
	}
	Orb->init(Variety->Canonical_form_classifier->Descr->PA->A,
			Variety->Canonical_form_classifier->Descr->PA->F,
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

	//strong_generators *gens_stab_of_canonical_equation;

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


	Variety->Canonical_form_classifier->Descr->PA->A->element_mult(
			trans1, trans2, Variety->transporter_to_canonical_form, 0);


	Variety->gens_stab_of_canonical_equation->group_order(*Variety->go_eqn);

	Variety->Canonical_form_classifier->Descr->PA->F->Projective_space_basic->PG_element_normalize(
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
		cout << "canonical_form_substructure::handle_orbit calling compute_stabilizer_function" << endl;
		}


	CS = NEW_OBJECT(set_stabilizer::compute_stabilizer);

	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit before CS->init" << endl;
	}
	CS->init(
			SubSt,
			Variety->canonical_pts,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_substructure::handle_orbit after CS->init" << endl;
	}


	Variety->Canonical_form_classifier->SubC->A->element_move(
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


