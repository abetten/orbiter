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

	Output = NULL;

}

canonical_form_classifier::~canonical_form_classifier()
{
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
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
		cout << "canonical_form_classifier::init "
				"algorithm = ";
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


	if (!Descr->f_space) {
		cout << "canonical_form_classifier::init "
				"please use -space <label>  to specify the space" << endl;
		exit(1);
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



	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(PA->A, Poly_ring, verbose_level - 3);
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
		cout << "canonical_form_classifier::classify "
				"before Output->init" << endl;
	}
	Output->init(this, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after Output->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}




}}}


