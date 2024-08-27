/*
 * variety_object_with_action.cpp
 *
 *  Created on: Dec 11, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



variety_object_with_action::variety_object_with_action()
{

	PA = NULL;

	cnt = 0;
	po_go = 0;
	po_index = 0;
	po = 0;
	so = 0;

	f_has_nauty_output = false;
	nauty_output_index_start = 0;

	Variety_object = NULL;

	f_has_automorphism_group = false;
	Stab_gens = NULL;

	TD = NULL;

}

variety_object_with_action::~variety_object_with_action()
{
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
	if (f_has_automorphism_group && Stab_gens) {
		FREE_OBJECT(Stab_gens);
	}
	if (TD) {
		FREE_OBJECT(TD);
	}
}

void variety_object_with_action::init(
		projective_geometry::projective_space_with_action *PA,
		int cnt, int po_go, int po_index, int po, int so,
		algebraic_geometry::variety_description *VD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::init" << endl;
	}

	data_structures::string_tools ST;


	variety_object_with_action::PA = PA;
	variety_object_with_action::cnt = cnt;
	variety_object_with_action::po_go = po_go;
	variety_object_with_action::po_index = po_index;
	variety_object_with_action::po = po;
	variety_object_with_action::so = so;


	if (VD->f_projective_space) {
		VD->f_has_projective_space_pointer = true;
		VD->Projective_space_pointer = Get_projective_space(VD->projective_space_label)->P;
	}

	if (VD->f_has_bitangents == false) {
		VD->f_has_bitangents = true;
		VD->bitangents_txt = "";
	}



	Variety_object = NEW_OBJECT(algebraic_geometry::variety_object);


	if (f_v) {
		cout << "variety_object_with_action::init "
				"before Variety_object->init" << endl;
	}
	Variety_object->init(
			VD,
			verbose_level);
	if (f_v) {
		cout << "variety_object_with_action::init "
				"after Variety_object->init" << endl;
	}

	int i;

	for (i = 0; i < VD->transformations.size(); i++) {
		if (f_v) {
			cout << "variety_object_with_action::init -transform " << VD->transformations[i] << endl;
		}

		int *data;
		int *Elt;
		int sz;

		Elt = NEW_int(PA->A->elt_size_in_int);

		Int_vec_scan(VD->transformations[i], data, sz);
		PA->A->Group_element->make_element(Elt, data, 0 /* verbose_level */);

		if (f_v) {
			cout << "variety_object_with_action::init before apply_transformation" << endl;
		}

		apply_transformation(
				Elt,
				PA->A,
				PA->A_on_lines,
				verbose_level - 2);

		if (f_v) {
			cout << "variety_object_with_action::init after apply_transformation" << endl;
		}

		FREE_int(data);

	}


	if (f_v) {
		cout << "variety_object_with_action::init before compute_tactical_decompositions" << endl;
	}
	compute_tactical_decompositions(
			verbose_level);
	if (f_v) {
		cout << "variety_object_with_action::init after compute_tactical_decompositions" << endl;
	}


	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::init done" << endl;
	}
}

void variety_object_with_action::apply_transformation(
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int verbose_level)
// Creates an action on the homogeneous polynomials on the fly
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation" << endl;
	}
	//data_structures::sorting Sorting;


#if 0
	cnt = old_one->cnt;
	po = old_one->po;
	so = old_one->so;

	Variety_object = NEW_OBJECT(algebraic_geometry::variety_object);

#endif


	algebraic_geometry::variety_object *old_Variety_object;

	old_Variety_object = Variety_object;

	Variety_object = NEW_OBJECT(algebraic_geometry::variety_object);


	Variety_object->Descr = old_Variety_object->Descr;
	Variety_object->Projective_space = old_Variety_object->Projective_space;
	Variety_object->Ring = old_Variety_object->Ring;
	Variety_object->label_txt = old_Variety_object->label_txt;
	Variety_object->label_tex = old_Variety_object->label_tex;

	Variety_object->eqn = NEW_int(old_Variety_object->Ring->get_nb_monomials());


	actions::action *A_on_equations;
	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	A_on_equations = A->Induced_action->induced_action_on_homogeneous_polynomials(
			Variety_object->Ring,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	AonHPD = A_on_equations->G.OnHP;

	if (f_v) {
		cout << "created action A_on_equations" << endl;
		A_on_equations->print_info();
	}

	AonHPD->compute_image_int_low_level(
		Elt,
		old_Variety_object->eqn,
		Variety_object->eqn,
		verbose_level - 2);


	FREE_OBJECT(A_on_equations);

	actions::action_global AG;


	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"before AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}
	Variety_object->Point_sets = AG.set_of_sets_copy_and_apply(
			A,
			Elt,
			old_Variety_object->Point_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"after AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}

	// we are sorting the points:

	Variety_object->Point_sets->sort();

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"before AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}
	Variety_object->Line_sets = AG.set_of_sets_copy_and_apply(
			A_on_lines,
			Elt,
			old_Variety_object->Line_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"after AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}

	// We are not sorting the lines because the lines are often in the Schlaefli ordering

	FREE_OBJECT(old_Variety_object);


#if 0
	variety_description *Descr;

	geometry::projective_space *Projective_space;

	ring_theory::homogeneous_polynomial_domain *Ring;


	std::string label_txt;
	std::string label_tex;



	int *eqn; // [Ring->get_nb_monomials()]
	//int *eqn2; // [Ring->get_nb_monomials()]


	// the partition into points and lines
	// must be invariant under the group.
	// must be sorted if find_point() or identify_lines() is invoked.

	data_structures::set_of_sets *Point_sets;

	data_structures::set_of_sets *Line_sets;

#endif


#if 0
	Variety_object->allocate_points(
			old_one->Variety_object->nb_pts,
			verbose_level);

	Int_vec_copy(eqn2, Quartic_curve_object->eqn15, 15);

	int i;

	for (i = 0; i < old_one->Quartic_curve_object->nb_pts; i++) {
		Quartic_curve_object->Pts[i] =
				A->Group_element->element_image_of(
				old_one->Quartic_curve_object->Pts[i],
				Elt, 0 /* verbose_level */);
	}

	// after mapping, the points are not in increasing order
	// Therefore, we sort the points:

	Sorting.lint_vec_heapsort(
			Quartic_curve_object->Pts,
			old_one->Quartic_curve_object->nb_pts);


	for (i = 0; i < 28; i++) {
		Quartic_curve_object->bitangents28[i] =
				A_on_lines->Group_element->element_image_of(
				old_one->Quartic_curve_object->bitangents28[i],
				Elt, 0 /* verbose_level */);
	}

	// We don't sort the lines because the lines are often in the Schlaefli ordering
#endif

	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation done" << endl;
	}
}

void variety_object_with_action::compute_tactical_decompositions(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions" << endl;
	}


	if (f_has_automorphism_group && Stab_gens) {

		TD = NEW_OBJECT(apps_combinatorics::variety_with_TDO_and_TDA);


		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions "
					"before TD->init_and_compute_tactical_decompositions" << endl;
		}
		TD->init_and_compute_tactical_decompositions(PA, Variety_object, Stab_gens, verbose_level);
		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions "
					"after TD->init_and_compute_tactical_decompositions" << endl;
		}
	}
	else {
		cout << "variety_object_with_action::compute_tactical_decompositions the automorphism group is not available" << endl;
		TD = NULL;
	}


	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions done" << endl;
	}


}



void variety_object_with_action::print(
		std::ostream &ost)
{
	ost << "cnt=" << cnt;
	ost << " po=" << po;
	ost << " so=" << so;

	Variety_object->print(ost);
}

std::string variety_object_with_action::stringify_Pts()
{
	std::string s;


	s = Lint_vec_stringify(
			Variety_object->Point_sets->Sets[0],
			Variety_object->Point_sets->Set_size[0]);

	return s;

}

std::string variety_object_with_action::stringify_bitangents()
{
	std::string s;

	s = Lint_vec_stringify(
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0]);
	return s;

}


}}}




