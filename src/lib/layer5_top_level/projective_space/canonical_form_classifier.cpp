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
namespace projective_geometry {



canonical_form_classifier::canonical_form_classifier()
{
	Descr = NULL;

	Poly_ring = NULL;
	AonHPD = NULL;

	Input = NULL;

	Output = NULL;

}

canonical_form_classifier::~canonical_form_classifier()
{
	if (Input) {
		FREE_OBJECT(Input);
	}
	if (Output) {
		FREE_OBJECT(Output);
	}
}

void canonical_form_classifier::init(
		canonical_form_classifier_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init" << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::init algorithm = ";
		if (Descr->f_algorithm_nauty) {
			cout << "nauty";
		}
		else if (Descr->f_algorithm_substructure) {
			cout << "substructure";
		}
		else {
			cout << "unknown" << endl;
		}
		cout << endl;
	}






	if (!Descr->f_algorithm_nauty && !Descr->f_algorithm_substructure) {
		cout << "canonical_form_classifier::init "
				"please select an algorithm to use" << endl;
		exit(1);
	}

	canonical_form_classifier::Descr = Descr;


	if (!Descr->f_degree) {
		cout << "canonical_form_classifier::init "
				"please use -degree <d>  to specify the degree" << endl;
		exit(1);
	}
	if (!Descr->f_output_fname) {
		cout << "please use -output_fname" << endl;
		exit(1);
	}

	Poly_ring = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before Poly_ring->init" << endl;
	}
	Poly_ring->init(
			Descr->PA->F,
			Descr->PA->n + 1,
			Descr->degree,
			t_PART,
			verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after Poly_ring->init" << endl;
	}
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"nb_monomials = " << Poly_ring->get_nb_monomials() << endl;
	}



	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(Descr->PA->A, Poly_ring, verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after AonHPD->init" << endl;
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




void canonical_form_classifier::classify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify" << endl;
	}


	Output = NEW_OBJECT(classification_of_varieties);


	if (f_v) {
		cout << "canonical_form_classifier::classify before Output->init" << endl;
	}
	Output->init(this, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify after Output->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}




}}}


