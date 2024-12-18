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
	Record_birth();
	Descr = NULL;

	f_nauty_control = false;
	Nauty_interface_control = NULL;

	Ring_with_action = NULL;

	Input = NULL;

	f_has_skip = false;
	skip_vector = NULL;
	skip_sz = 0;


	Classification_of_varieties_nauty = NULL;

}

canonical_form_classifier::~canonical_form_classifier()
{
	Record_death();

	int verbose_level = 1;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_classifier::~canonical_form_classifier before Ring_with_action" << endl;
	}
	if (Ring_with_action) {
		FREE_OBJECT(Ring_with_action);
	}
#if 0
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
#endif


#if 0
	if (Input) {
		FREE_OBJECT(Input);
	}
#endif


	if (f_v) {
		cout << "canonical_form_classifier::~canonical_form_classifier before Classification_of_varieties_nauty" << endl;
	}
	if (Classification_of_varieties_nauty) {
		FREE_OBJECT(Classification_of_varieties_nauty);
	}

	if (f_v) {
		cout << "canonical_form_classifier::~canonical_form_classifier before skip_vector" << endl;
	}
	if (skip_vector) {
		FREE_int(skip_vector);
	}
	if (f_v) {
		cout << "canonical_form_classifier::~canonical_form_classifier done" << endl;
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

void canonical_form_classifier::set_description(
		canonical_form_classifier_description *Descr)
{
	canonical_form_classifier::Descr = Descr;
}


int canonical_form_classifier::has_description()
{
	if (Descr == NULL) {
		return false;
	}
	return true;
}

void canonical_form_classifier::init_objects_from_list_of_csv_files(
		canonical_form_classifier_description *Descr,
		int verbose_level)
// called from orbits_create::init
// Prepare the projective space and the ring,
// Create the action_on_homogeneous_polynomials
// Prepare the input input_objects_of_type_variety
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files" << endl;
	}

	canonical_form_classifier::Descr = Descr;

	f_nauty_control = Descr->f_nauty_control;
	Nauty_interface_control = Descr->Nauty_interface_control;


	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"before copying carry_through" << endl;
	}
	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"Descr->carry_through.size() = " << endl;
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"Descr->carry_through.size() = " << Descr->carry_through.size() << endl;
	}
	int i;

	for (i = 0; i < Descr->carry_through.size(); i++) {

		carry_through.push_back(Descr->carry_through[i]);

	}
	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files after copying carry_through" << endl;
	}


#if 0
	if (!Descr->f_algorithm_nauty /*&& !Descr->f_algorithm_substructure*/) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"please select an algorithm to use" << endl;
		exit(1);
	}
#endif

	if (!Descr->f_nauty_control) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"please use -nauty_control <options> -end" << endl;
		exit(1);
	}
	else {
		if (f_v) {
			cout << "canonical_form_classifier::init_objects_from_list_of_csv_files nauty_control:" << endl;
			Descr->Nauty_interface_control->print();
		}
	}


	if (!Descr->f_space) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"please use -space <label>  to specify the space" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"before create_action_on_polynomials" << endl;
	}
	create_action_on_polynomials(verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"after create_action_on_polynomials" << endl;
	}



	Input = NEW_OBJECT(input_objects_of_type_variety);


	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"before Input->read_objects_from_list_of_csv_files" << endl;
	}
	Input->read_objects_from_list_of_csv_files(
			this, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
				"after Input->read_objects_from_list_of_csv_files" << endl;
	}

	if (Descr->f_skip) {
		if (f_v) {
			cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
					"before init_skip" << endl;
		}
		init_skip(
				Descr->skip_vector_label, verbose_level);
		if (f_v) {
			cout << "canonical_form_classifier::init_objects_from_list_of_csv_files "
					"after init_skip" << endl;
		}
	}

	if (f_v) {
		cout << "canonical_form_classifier::init_objects_from_list_of_csv_files done" << endl;
	}
}


void canonical_form_classifier::init_direct(
		int nb_input_Vo,
		canonical_form::variety_object_with_action *Input_Vo,
		std::string &fname_base_out,
		int f_nauty_control,
		other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
		int verbose_level)
// Prepare the projective space and the ring,
// Create the action_on_homogeneous_polynomials
// called from
// variety_activity::do_compute_group
//
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init_direct" << endl;
	}

	canonical_form_classifier::f_nauty_control = f_nauty_control;
	canonical_form_classifier::Nauty_interface_control = Nauty_interface_control;

	//canonical_form_classifier::PA = PA;

	//canonical_form_classifier::Poly_ring = Poly_ring;

	//induced_actions::action_on_homogeneous_polynomials *AonHPD;


	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before getting PA" << endl;
	}
	projective_geometry::projective_space_with_action *PA = Input_Vo->PA;
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after getting PA" << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before getting Poly_ring" << endl;
	}
	algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring = Input_Vo->Variety_object->Ring;
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after getting Poly_ring" << endl;
	}


	Ring_with_action = NEW_OBJECT(projective_geometry::ring_with_action);

	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before Ring_with_action->ring_with_action_init" << endl;
	}
	Ring_with_action->ring_with_action_init(PA, Poly_ring, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after Ring_with_action->ring_with_action_init" << endl;
	}

	Input = NEW_OBJECT(input_objects_of_type_variety);


	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before Input->init_direct" << endl;
	}
	Input->init_direct(
			nb_input_Vo,
			Input_Vo,
			fname_base_out,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after Input->init_direct" << endl;
	}


#if 0
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"before create_action_on_polynomials" << endl;
	}
	create_action_on_polynomials(verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init_direct "
				"after create_action_on_polynomials" << endl;
	}
#endif





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


	projective_geometry::projective_space_with_action *PA;

	algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring;



	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"before Get_projective_space" << endl;
	}
	PA = Get_projective_space(
					Descr->space_label);


	if (!Descr->f_ring) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"please use -ring <label>  to specify the ring" << endl;
		exit(1);
	}


	if (!Descr->f_output_fname) {
		cout << "please use -output_fname" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"before Get_ring" << endl;
	}
	Poly_ring = Get_ring(Descr->ring_label);
	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"polynomial degree " << Poly_ring->degree << endl;
		cout << "canonical_form_classifier::init "
				"polynomial number of variables " << Poly_ring->nb_variables << endl;
	}

	if (Poly_ring->nb_variables != PA->n + 1) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"polynomial number of variables must equal projective dimension plus one" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "canonical_form_classifier::init "
				"nb_monomials = " << Poly_ring->get_nb_monomials() << endl;
	}


	Ring_with_action = NEW_OBJECT(projective_geometry::ring_with_action);

	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"before Ring_with_action->ring_with_action_init" << endl;
	}
	Ring_with_action->ring_with_action_init(PA, Poly_ring, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials "
				"after Ring_with_action->ring_with_action_init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::create_action_on_polynomials done" << endl;
	}



}


void canonical_form_classifier::classify(
		input_objects_of_type_variety *Input,
		std::string &fname_base,
		int verbose_level)
// initializes Classification_of_varieties_nauty
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify" << endl;
	}


	Classification_of_varieties_nauty = NEW_OBJECT(classification_of_varieties_nauty);


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before Classification_of_varieties_nauty->prepare_for_classification" << endl;
	}

	Classification_of_varieties_nauty->prepare_for_classification(
			Input,
			this /*canonical_form_classifier *Classifier*/,
			verbose_level);


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after Classification_of_varieties_nauty->prepare_for_classification" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before Classification_of_varieties_nauty->compute_classification" << endl;
	}
	Classification_of_varieties_nauty->compute_classification(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after Classification_of_varieties_nauty->compute_classification" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}

void canonical_form_classifier::init_skip(
		std::string &skip_vector_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::init_skip" << endl;
	}

	f_has_skip = true;

	Get_int_vector_from_label(
			skip_vector_label,
			skip_vector, skip_sz,
			0 /* verbose_level */);

	other::data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(skip_vector, skip_sz);
	if (f_v) {
		cout << "canonical_form_classifier::init_skip "
				"skip list consists of " << skip_sz << " cases" << endl;
		cout << "The cases to be skipped are :";
		Int_vec_print(cout, skip_vector, skip_sz);
		cout << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::init_skip done" << endl;
	}
}

int canonical_form_classifier::skip_this_one(
		int counter)
{
	other::data_structures::sorting Sorting;
	int idx;

	if (f_has_skip) {
		if (Sorting.int_vec_search(
				skip_vector, skip_sz, counter, idx)) {
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}





}}}


