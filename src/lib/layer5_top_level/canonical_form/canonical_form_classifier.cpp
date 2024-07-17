/*
 * canonical_form_classifier.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_classifier::canonical_form_classifier()
{
	Descr = NULL;

	PA = NULL;

	Poly_ring = NULL;
	AonHPD = NULL;

	Input = NULL;

	Classification_of_varieties = NULL;

	Classification_of_varieties_nauty = NULL;

}

canonical_form_classifier::~canonical_form_classifier()
{
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
	if (Input) {
		FREE_OBJECT(Input);
	}
	if (Classification_of_varieties) {
		FREE_OBJECT(Classification_of_varieties);
	}
	if (Classification_of_varieties_nauty) {
		FREE_OBJECT(Classification_of_varieties_nauty);
	}
}

canonical_form_classifier_description *canonical_form_classifier::get_description()
{
	if (Descr == NULL) {
		cout << "canonical_form_classifier::get_description Descr == NULL" << endl;
		exit(1);
	}
	return Descr;
}

void canonical_form_classifier::init(
		canonical_form_classifier_description *Descr,
		int verbose_level)
// Prepare the projective space and the ring,
// Create the action_on_homogeneous_polynomials
// Prepare the input input_objects_of_type_variety
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init" << endl;
	}

	canonical_form_classifier::Descr = Descr;

	if (f_v) {
		cout << "canonical_form_classifier::init "
				"algorithm = ";
		if (Descr->f_algorithm_nauty) {
			cout << "nauty";

			if (Descr->f_has_nauty_output) {

				cout << ", has nauty output";
			}
			else {
				cout << ", needs to apply nauty";

			}
		}
		else if (Descr->f_algorithm_substructure) {
			cout << "substructure";
		}
		else {
			cout << "unknown" << endl;
		}
		cout << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::init before copying carry_through" << endl;
	}
	if (f_v) {
		cout << "canonical_form_classifier::init Descr->carry_through.size() = " << endl;
		cout << "canonical_form_classifier::init Descr->carry_through.size() = " << Descr->carry_through.size() << endl;
	}
	int i;

	for (i = 0; i < Descr->carry_through.size(); i++) {

		carry_through.push_back(Descr->carry_through[i]);

	}
	if (f_v) {
		cout << "canonical_form_classifier::init after copying carry_through" << endl;
	}



	if (!Descr->f_algorithm_nauty && !Descr->f_algorithm_substructure) {
		cout << "canonical_form_classifier::init "
				"please select an algorithm to use" << endl;
		exit(1);
	}



	if (!Descr->f_space) {
		cout << "canonical_form_classifier::init "
				"please use -space <label>  to specify the space" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "canonical_form_classifier::init before Get_projective_space" << endl;
	}
	PA = Get_projective_space(
					Descr->space_label);


#if 0
	if (!Descr->f_degree) {
		cout << "canonical_form_classifier::init "
				"please use -degree <d>  to specify the degree" << endl;
		exit(1);
	}
#endif

	if (!Descr->f_ring) {
		cout << "canonical_form_classifier::init "
				"please use -ring <label>  to specify the ring" << endl;
		exit(1);
	}


	if (!Descr->f_output_fname) {
		cout << "please use -output_fname" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "canonical_form_classifier::init before Get_ring" << endl;
	}
	Poly_ring = Get_ring(Descr->ring_label);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"polynomial degree " << Poly_ring->degree << endl;
		cout << "canonical_form_classifier::init "
				"polynomial number of variables " << Poly_ring->nb_variables << endl;
	}

	if (Poly_ring->nb_variables != PA->n + 1) {
		cout << "canonical_form_classifier::init "
				"polynomial number of variables must equal projective dimension plus one" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "canonical_form_classifier::init "
				"nb_monomials = " << Poly_ring->get_nb_monomials() << endl;
	}



	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before create_action_on_polynomials" << endl;
	}
	create_action_on_polynomials(verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after create_action_on_polynomials" << endl;
	}



	Input = NEW_OBJECT(input_objects_of_type_variety);


	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before Input->init" << endl;
	}
	Input->init(this, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after Input->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::init done" << endl;
	}
}


void canonical_form_classifier::init_direct(
		projective_geometry::projective_space_with_action *PA,
		ring_theory::homogeneous_polynomial_domain *Poly_ring,
		int nb_input_Vo,
		canonical_form::variety_object_with_action *Input_Vo,
		int verbose_level)
// Prepare the projective space and the ring,
// Create the action_on_homogeneous_polynomials
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init_direct" << endl;
	}


	canonical_form_classifier::PA = PA;

	canonical_form_classifier::Poly_ring = Poly_ring;


	Input = NULL;


	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before create_action_on_polynomials" << endl;
	}
	create_action_on_polynomials(verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after create_action_on_polynomials" << endl;
	}





	if (f_v) {
		cout << "canonical_form_classifier::init_direct done" << endl;
	}
}



void canonical_form_classifier::create_action_on_polynomials(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials" << endl;
	}

	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(
			PA->A, Poly_ring, verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"after AonHPD->init" << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials done" << endl;
	}



}


void canonical_form_classifier::classify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify" << endl;
	}


	Classification_of_varieties = NEW_OBJECT(classification_of_varieties);


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before Classification_of_varieties->init" << endl;
	}
	Classification_of_varieties->init(this, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after Classification_of_varieties->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}




}}}


