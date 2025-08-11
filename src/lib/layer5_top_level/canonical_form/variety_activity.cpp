/*
 * variety_activity.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


variety_activity::variety_activity()
{
	Record_birth();
	Descr = NULL;

	nb_input_Vo = 0;

	Input_Vo = NULL;

}

variety_activity::~variety_activity()
{
	Record_death();

}

void variety_activity::init(
		variety_activity_description *Descr,
		int nb_input_Vo,
		canonical_form::variety_object_with_action **Input_Vo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::init" << endl;
	}
	if (f_v) {
		cout << "variety_activity::init "
				"nb_input_Vo = " << nb_input_Vo << endl;
	}

	variety_activity::Descr = Descr;

	variety_activity::nb_input_Vo = nb_input_Vo;
	variety_activity::Input_Vo = Input_Vo;

	if (f_v) {
		cout << "variety_activity::init done" << endl;
	}
}


void variety_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::perform_activity" << endl;
	}

	if (Descr->f_compute_group) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-compute_group" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before do_compute_group" << endl;
		}

		do_compute_group(
				Descr->f_output_fname_base,
				Descr->output_fname_base,
				Descr->f_nauty_control,
				Descr->Nauty_interface_control,
				verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after do_compute_group" << endl;
		}
	}
	if (Descr->f_test_isomorphism) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-test_isomorphism" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before do_test_isomorphism" << endl;
		}

		do_test_isomorphism(
				Descr->f_output_fname_base,
				Descr->output_fname_base,
				Descr->f_nauty_control,
				Descr->Nauty_interface_control,
				verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after do_test_isomorphism" << endl;
		}
	}
	if (Descr->f_compute_set_stabilizer) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-compute_set_stabilizer" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before do_compute_set_stabilizer" << endl;
		}
		do_compute_set_stabilizer(
				Descr->f_output_fname_base,
				Descr->output_fname_base,
				Descr->f_nauty_control,
				Descr->Nauty_interface_control,
				verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after do_compute_set_stabilizer" << endl;
		}
	}
	if (Descr->f_report) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-report" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before Input_Vo[0]->do_report" << endl;
		}
		Input_Vo[0]->do_report(verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after Input_Vo[0]->do_report" << endl;
		}
	}
	if (Descr->f_export) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-export" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before Input_Vo[0]->do_export" << endl;
		}
		Input_Vo[0]->do_export(verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after Input_Vo[0]->do_export" << endl;
		}
	}
	if (Descr->f_apply_transformation_to_self) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-apply_transformation_to_self" << endl;
		}

		int f_inverse = false;

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before do_apply_transformation_to_self" << endl;
		}
		do_apply_transformation_to_self(
				f_inverse,
				Descr->apply_transformation_to_self_group_element,
				verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after do_apply_transformation_to_self" << endl;
		}


	}
	if (Descr->f_singular_points) {

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"-singular_points" << endl;
		}

		if (f_v) {
			cout << "variety_activity::perform_activity "
					"before do_singular_points" << endl;
		}
		do_singular_points(verbose_level);
		if (f_v) {
			cout << "variety_activity::perform_activity "
					"after do_singular_points" << endl;
		}
	}


	if (f_v) {
		cout << "variety_activity::perform_activity done" << endl;
	}
}

void variety_activity::do_compute_group(
		int f_has_output_fname_base,
		std::string &output_fname_base,
		int f_nauty_control,
		other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_compute_group" << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"nb_input_Vo = " << nb_input_Vo << endl;
	}

	if (nb_input_Vo == 0) {
		cout << "variety_activity::do_compute_group "
				"nb_input_Vo == 0" << endl;
		exit(1);
	}

	std::string fname_base;

	if (f_has_output_fname_base) {
		fname_base = output_fname_base;
	}
	else {
		fname_base = Input_Vo[0]->Variety_object->label_txt + "_c";
	}

	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			nb_input_Vo,
			Input_Vo,
			fname_base,
			f_nauty_control,
			Nauty_interface_control,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Classifier->init_direct" << endl;
	}



	canonical_form::canonical_form_global Canonical_form_global;

	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}
	Canonical_form_global.compute_group_and_tactical_decomposition(
			Classifier,
			Input_Vo[0],
			Classification_of_varieties_nauty,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}


	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "variety_activity::do_compute_group done" << endl;
	}
}


void variety_activity::do_test_isomorphism(
		int f_has_output_fname_base,
		std::string &output_fname_base,
		int f_nauty_control,
		other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_test_isomorphism" << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_test_isomorphism "
				"nb_input_Vo = " << nb_input_Vo << endl;
	}

	if (nb_input_Vo != 2) {
		cout << "variety_activity::do_test_isomorphism "
				"nb_input_Vo != 2" << endl;
		exit(1);
	}

	std::string fname_base;

	if (f_has_output_fname_base) {
		fname_base = output_fname_base;
	}
	else {
		fname_base = "Iso_" + Input_Vo[0]->Variety_object->label_txt + "_and_" + Input_Vo[1]->Variety_object->label_txt;
	}

	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (f_v) {
		cout << "variety_activity::do_test_isomorphism "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			nb_input_Vo,
			Input_Vo,
			fname_base,
			f_nauty_control,
			Nauty_interface_control,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_test_isomorphism "
				"after Classifier->init_direct" << endl;
	}



	canonical_form::canonical_form_global Canonical_form_global;

	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;


	if (f_v) {
		cout << "variety_activity::do_test_isomorphism "
				"before Canonical_form_global.compute_isomorphism" << endl;
	}
	Canonical_form_global.compute_isomorphism(
			Classifier,
			Classification_of_varieties_nauty,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_test_isomorphism "
				"after Canonical_form_global.compute_isomorphism" << endl;
	}


	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "variety_activity::do_test_isomorphism done" << endl;
	}
}


void variety_activity::do_compute_set_stabilizer(
		int f_has_output_fname_base,
		std::string &output_fname_base,
		int f_nauty_control,
		other::l1_interfaces::nauty_interface_control *Nauty_interface_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer" << endl;
	}

	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"nb_input_Vo = " << nb_input_Vo << endl;
	}

	if (nb_input_Vo == 0) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"nb_input_Vo == 0" << endl;
		exit(1);
	}

	std::string fname_base;

	if (f_has_output_fname_base) {
		fname_base = output_fname_base;
	}
	else {
		fname_base = Input_Vo[0]->Variety_object->label_txt + "_cs";
	}

	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			nb_input_Vo,
			Input_Vo,
			fname_base,
			f_nauty_control,
			Nauty_interface_control,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"after Classifier->init_direct" << endl;
	}



	canonical_form::canonical_form_global Canonical_form_global;

	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;


	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"before Canonical_form_global.compute_set_stabilizer_and_tactical_decomposition" << endl;
	}
	Canonical_form_global.compute_set_stabilizer_and_tactical_decomposition(
			Classifier,
			Input_Vo[0],
			Classification_of_varieties_nauty,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer "
				"after Canonical_form_global.compute_set_stabilizer_and_tactical_decomposition" << endl;
	}


	FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "variety_activity::do_compute_set_stabilizer done" << endl;
	}
}

void variety_activity::do_apply_transformation_to_self(
		int f_inverse,
		std::string &transformation_coded,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_apply_transformation_to_self" << endl;
	}


	actions::action *A;

	A = Input_Vo[0]->PA->A;

	int *Elt1;
	int *Elt2;
	int *Elt3;

	//A = PA->A;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	int *v;
	int sz;

	Get_int_vector_from_label(transformation_coded, v, sz, verbose_level - 3);

	if (sz != A->make_element_size) {
		cout << "variety_activity::do_apply_transformation_to_self sz != A->make_element_size" << endl;
		cout << "sz=" << sz << endl;
		cout << "A->make_element_size=" << A->make_element_size << endl;
		exit(1);
	}

	A->Group_element->make_element(
			Elt1, v,
			verbose_level);

	if (f_inverse) {
		A->Group_element->element_invert(
				Elt1, Elt2, 0 /*verbose_level*/);
	}
	else {
		A->Group_element->element_move(
				Elt1, Elt2, 0 /*verbose_level*/);
	}

	//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

	A->Group_element->element_invert(
			Elt2, Elt3, 0 /*verbose_level*/);

	if (f_v) {
		cout << "variety_activity::do_apply_transformation_to_self "
				"applying the transformation given by:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt2);
		cout << endl;
		cout << "$$" << endl;
		cout << "variety_activity::do_apply_transformation_to_self "
				"The inverse is:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt3);
		cout << endl;
		cout << "$$" << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_apply_transformation_to_self "
				"before Input_Vo[0]->apply_transformation_to_self" << endl;
	}
	Input_Vo[0]->apply_transformation_to_self(
			Elt2,
			Input_Vo[0]->PA->A,
			Input_Vo[0]->PA->A_on_lines,
			verbose_level);

	if (f_v) {
		cout << "variety_activity::do_apply_transformation_to_self "
				"after Input_Vo[0]->apply_transformation_to_self" << endl;
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "variety_activity::do_apply_transformation_to_self done" << endl;
	}
}

void variety_activity::do_singular_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_activity::do_singular_points" << endl;
	}

	geometry::projective_geometry::projective_space *P;


	P = Input_Vo[0]->Variety_object->Projective_space;

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before getting Poly_ring" << endl;
	}
	algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring = Input_Vo[0]->Variety_object->Ring;
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after getting Poly_ring" << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"before Poly_ring->compute_singular_points_projectively" << endl;
	}
	Poly_ring->compute_singular_points_projectively(
			P,
			Input_Vo[0]->Variety_object->eqn,
			Input_Vo[0]->Variety_object->Singular_points,
			verbose_level);
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"after Poly_ring->compute_singular_points_projectively" << endl;
	}

	Input_Vo[0]->Variety_object->f_has_singular_points = true;

	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"number of singular points = " << Input_Vo[0]->Variety_object->Singular_points.size() << endl;
	}
	if (f_v) {
		cout << "variety_activity::do_compute_group "
				"The singular points are: " << endl;
		Lint_vec_stl_print_fully(cout, Input_Vo[0]->Variety_object->Singular_points);
		cout << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = Input_Vo[0]->Variety_object->label_txt + "_singular_pts.csv";

	Fio.Csv_file_support->vector_lint_write_csv(
			fname,
			Input_Vo[0]->Variety_object->Singular_points);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "variety_activity::do_singular_points done" << endl;
	}
}


}}}
