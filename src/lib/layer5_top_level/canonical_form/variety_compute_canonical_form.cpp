/*
 * variety_compute_canonical_form.cpp
 *
 *  Created on: Jan 26, 2023
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



variety_compute_canonical_form::variety_compute_canonical_form()
{
	Record_birth();
	Canonical_form_classifier = NULL;

	Ring_with_action = NULL;

	Classification_of_varieties_nauty = NULL;

	//std::string fname_case_out;

	counter = 0;
	Vo = NULL;

	canonical_pts = NULL;
	canonical_equation = NULL;
	transporter_to_canonical_form = NULL;
	gens_stab_of_canonical_equation = NULL;

	Variety_stabilizer_compute = NULL;

	go_eqn = NULL;

	Canonical_object = NULL;


}

variety_compute_canonical_form::~variety_compute_canonical_form()
{
	Record_death();
	if (canonical_equation) {
		FREE_int(canonical_equation);
	}
	if (transporter_to_canonical_form) {
		FREE_int(transporter_to_canonical_form);
	}
	if (canonical_pts) {
		FREE_lint(canonical_pts);
	}
	if (Variety_stabilizer_compute) {
		FREE_OBJECT(Variety_stabilizer_compute);
	}
	if (go_eqn) {
		FREE_OBJECT(go_eqn);
	}
	if (Canonical_object) {
		FREE_OBJECT(Canonical_object);
	}
}


void variety_compute_canonical_form::init(
		canonical_form_classifier *Canonical_form_classifier,
		projective_geometry::ring_with_action *Ring_with_action,
		classification_of_varieties_nauty *Classification_of_varieties_nauty,
		std::string &fname_case_out,
		int counter,
		variety_object_with_action *Vo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::init "
				" counter=" << counter
				<< " verbose_level=" << verbose_level
				<< endl;
	}

	variety_compute_canonical_form::Canonical_form_classifier = Canonical_form_classifier;
	variety_compute_canonical_form::Ring_with_action = Ring_with_action;
	variety_compute_canonical_form::Classification_of_varieties_nauty = Classification_of_varieties_nauty;
	variety_compute_canonical_form::fname_case_out.assign(fname_case_out);
	variety_compute_canonical_form::counter = counter;
	variety_compute_canonical_form::Vo = Vo;

	canonical_equation =
			NEW_int(Ring_with_action->Poly_ring->get_nb_monomials());
	transporter_to_canonical_form =
			NEW_int(Ring_with_action->PA->A->elt_size_in_int);


	canonical_pts = NEW_lint(Vo->Variety_object->Point_sets->Set_size[0]);

	go_eqn = NEW_OBJECT(algebra::ring_theory::longinteger_object);



	if (f_v) {
		cout << "variety_compute_canonical_form::init done" << endl;
	}
}


void variety_compute_canonical_form::compute_canonical_form_nauty_new(
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &f_found_canonical_form,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
// called from classification_of_varieties_nauty::main_loop
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"input_counter = " << counter << " / " << Classification_of_varieties_nauty->Input->nb_objects_to_test << endl;
	}


	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"before classify_using_nauty_new" << endl;
	}

	classify_using_nauty_new(
			Nauty_control,
			f_found_canonical_form,
			idx_canonical_form,
			idx_equation,
			f_found_eqn,
			verbose_level - 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"after classify_using_nauty_new" << endl;
	}
	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"input_counter = " << counter << " / " << Classification_of_varieties_nauty->Input->nb_objects_to_test << endl;
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"f_found_canonical_form=" << f_found_canonical_form << endl;
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"idx_canonical_form=" << idx_canonical_form << endl;
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"idx_equation=" << idx_equation << endl;
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"f_found_eqn=" << f_found_eqn << endl;
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new "
				"group_order=" << Variety_stabilizer_compute->Stab_gens_variety->group_order_as_lint() << endl;
	}

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_form_nauty_new done" << endl;
	}
}

void variety_compute_canonical_form::classify_using_nauty_new(
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int &f_found_canonical_form,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"verbose_level=" << verbose_level << endl;
	}

	algebra::ring_theory::longinteger_object go;






	Variety_stabilizer_compute = NEW_OBJECT(variety_stabilizer_compute);

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"before Variety_stabilizer_compute->init" << endl;
	}
	Variety_stabilizer_compute->init(
			Ring_with_action, verbose_level - 6);
	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"after Variety_stabilizer_compute->init" << endl;
	}


	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"input_counter = " << counter << " / " << Classification_of_varieties_nauty->Input->nb_objects_to_test << endl;
	}
	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"before Variety_stabilizer_compute->compute_canonical_form_of_variety" << endl;
	}
	Variety_stabilizer_compute->compute_canonical_form_of_variety(
			Vo,
			Nauty_control,
			verbose_level - 2);
	// Computes the canonical labeling of the Levi graph associated with
	// the set of rational points of the variety.
	// Computes the stabilizer of the set of rational points of the variety.
	// Computes the orbit of the original equation under the stabilizer of the set.
	// The orbit is stored as a sorted table of equations.
	// The original equation is stored at position position_of_original_object
	// The canonical equation is stored at position 0
	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"after Variety_stabilizer_compute->compute_canonical_form_of_variety" << endl;
	}

	algebra::ring_theory::longinteger_object set_stab_order;

	Variety_stabilizer_compute->Set_stab->group_order(set_stab_order);

	// set_stab_order needs to be initialized

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"before Variety_stabilizer_compute->Orb->get_canonical_form" << endl;
	}
	Variety_stabilizer_compute->Orb->get_canonical_form(
			canonical_equation,
			transporter_to_canonical_form,
			gens_stab_of_canonical_equation,
			set_stab_order /* full_group_order */,
			verbose_level);
	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"after Variety_stabilizer_compute->Orb->get_canonical_form" << endl;
	}

	// gens_stab_of_canonical_equation maps the root node to the canonical equation.


	if (Nauty_control->f_save_orbit_of_equations) {
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"f_save_orbit_of_equations" << endl;
		}

		string fname;

		fname = Nauty_control->save_orbit_of_equations_prefix + fname_case_out + ".csv";

		Variety_stabilizer_compute->save_table_of_equations(
				fname,
				verbose_level);

		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"f_save_orbit_of_equations done" << endl;
		}


	}


	if (f_v) {
		algebra::ring_theory::longinteger_object go;

		gens_stab_of_canonical_equation->group_order(go);
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"The stabilizer of the variety is a group of order " << go << endl;
		//gens_stab_of_canonical_equation->print_generators_tex(cout);
	}


	//Variety_stabilizer_compute->Stab_gens_variety->group_order(go);

	//goi = go.as_int();

	//FREE_OBJECT(gens_stab_of_canonical_equation);

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"Variety_stabilizer_compute->NO->N = " << Variety_stabilizer_compute->NO->N << endl;
	}

	Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len =
			Variety_stabilizer_compute->NO->N; //canonical_labeling_len;

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"canonical_labeling_len=" <<
				Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len << endl;
	}
	if (Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len == 0) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"canonical_labeling_len == 0, error" << endl;
		exit(1);
	}



	if (Canonical_form_classifier->Classification_of_varieties_nauty->CB->n == 0) {
		Canonical_form_classifier->Classification_of_varieties_nauty->CB->init(
				Canonical_form_classifier->Classification_of_varieties_nauty->Input->nb_objects_to_test,
				Variety_stabilizer_compute->Canonical_form->get_allocated_length(),
				verbose_level - 2);
	}
	//int f_found;
	//int idx;

	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"before CB->canonical_form_search_and_add_if_new" << endl;
	}
	Canonical_form_classifier->Classification_of_varieties_nauty->CB->canonical_form_search_and_add_if_new(
			Variety_stabilizer_compute->Canonical_form,
			Variety_stabilizer_compute /* void *extra_data */,
			f_found_canonical_form,
			idx_canonical_form,
			verbose_level);
	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"after CB->canonical_form_search_and_add_if_new" << endl;
		cout << "variety_compute_canonical_form::classify_using_nauty_new "
				"CB->nb_types = "
				<< Canonical_form_classifier->Classification_of_varieties_nauty->CB->nb_types << endl;

	}
	// if f_found is true: idx_canonical_form is where the canonical form was found.
	// if f_found is false: idx_canonical_form is where the new canonical form was added.




	if (f_found_canonical_form) {
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form at idx_canonical_form = "
					<< idx_canonical_form << endl;
		}


#if 0
		int *alpha; // [Canonical_form_classifier->Output->canonical_labeling_len]
		int *gamma; // [Canonical_form_classifier->Output->canonical_labeling_len]


		alpha = NEW_int(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
		gamma = NEW_int(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
#endif
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"before handle_repeated_canonical_form_of_set_new" << endl;
		}
		handle_repeated_canonical_form_of_set_new(
				idx_canonical_form,
				Variety_stabilizer_compute,
				//alpha, gamma,
				idx_canonical_form,
				idx_equation,
				f_found_eqn,
				verbose_level);
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"after handle_repeated_canonical_form_of_set_new" << endl;
		}

		//FREE_int(alpha);
		//FREE_int(gamma);


	} // if f_found_canonical_form
	else {
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " The canonical form is new and has been added" << endl;
		}

		// The canonical form has already been added,
		// at position idx_canonical_form,
		// with Canonical_form_nauty as extra_data.
		idx_equation = Variety_stabilizer_compute->Orb->position_of_original_object;
		if (f_v) {
			cout << "variety_compute_canonical_form::classify_using_nauty_new "
					"idx_equation = Variety_stabilizer_compute->Orb->position_of_original_object = " << idx_equation << endl;
		}
	}


	if (f_v) {
		cout << "variety_compute_canonical_form::classify_using_nauty_new done" << endl;
	}
}



void variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new(
		int idx,
		variety_stabilizer_compute *C,
		//int *alpha, int *gamma,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
// called from variety_compute_canonical_form::classify_using_nauty_new
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
				"verbose_level=" << verbose_level << endl;
	}

	int *alpha; // [Canonical_form_classifier->Output->canonical_labeling_len]
	int *gamma; // [Canonical_form_classifier->Output->canonical_labeling_len]


	alpha = NEW_int(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
	gamma = NEW_int(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);



	idx_equation = -1;
	f_found_eqn = false;

	if (f_v) {
		cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
				"starting loop over idx1" << endl;
	}

	for (idx_canonical_form = idx; idx_canonical_form >= 0; idx_canonical_form--) {



		// test if entry at idx_canonical_form is equal to C.
		// if not, break

		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"before Canonical_form_classifier->CB->compare_at "
					"idx_canonical_form = " << idx_canonical_form << endl;
		}
		if (Canonical_form_classifier->Classification_of_varieties_nauty->CB->compare_at(
				C->Canonical_form->get_data(), idx_canonical_form) != 0) {
			if (f_v) {
				cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
						"at idx_canonical_form = " << idx_canonical_form
						<< " is not equal, break" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"canonical form at " << idx_canonical_form << " is equal" << endl;
		}


		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"before find_equation_new" << endl;
		}

		f_found_eqn = find_equation_new(
				C,
				alpha, gamma,
				idx_canonical_form, idx_equation,
				verbose_level);

		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"after find_equation_new" << endl;
			if (f_found_eqn) {
				cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
						"We found the equation" << endl;
			}
			else {
				cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
						"We did not find the equation" << endl;
			}
		}

		if (f_found_eqn) {
			break;
		}


	}

	FREE_int(alpha);
	FREE_int(gamma);



	if (f_found_eqn) {
		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"we found the equation at index " << idx_equation << endl;
		}
	}
	else {
		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"we found the canonical form but we did "
					"not find the equation" << endl;
		}

		// add the canonical form at position idx

		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"before add_object_and_compute_canonical_equation_new" << endl;
		}
		add_object_and_compute_canonical_equation_new(
				C, idx,
				verbose_level);
		if (f_v) {
			cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new "
					"after add_object_and_compute_canonical_equation_new" << endl;
		}

	}

	if (f_v) {
		cout << "variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new done" << endl;
	}

}




int variety_compute_canonical_form::find_equation_new(
		variety_stabilizer_compute *B,
		int *alpha, int *gamma,
		int idx_rational_point_set, int &found_at,
		int verbose_level)
// gets the variety_stabilizer_compute object from
// Canonical_form_classifier->Output->CB->Type_extra_data[idx1]
// called from variety_compute_canonical_form::handle_repeated_canonical_form_of_set_new
// Computes Elt, which maps A to B, where A is the set of rational points
// whose canonical form is stored in CB at idx1
// Vo->Variety_object->eqn is the equation defining the set B
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"verbose_level=" << verbose_level << endl;
	}



	variety_stabilizer_compute *A;
	A = (variety_stabilizer_compute *)
			Canonical_form_classifier->Classification_of_varieties_nauty->CB->Type_extra_data[idx_rational_point_set];


	canonical_form_global Canonical_form_global;

	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"before Canonical_form_global.find_isomorphism_between_set_of_rational_points" << endl;
	}
	Canonical_form_global.find_isomorphism_between_set_of_rational_points(
			A,
			B,
			alpha, gamma,
			verbose_level - 2);
	// find gamma which maps the points A to the points B.
	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"after Canonical_form_global.find_isomorphism_between_set_of_rational_points" << endl;
	}


	// gamma maps A to B.
	// So, in the contragredient action,
	// gamma maps the equation of B to an equation in the orbit of the equation A,
	// which is what we want.


	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"before Ring_with_action->lift_mapping" << endl;
	}
	Ring_with_action->lift_mapping(
			gamma,
			Canonical_form_classifier->Classification_of_varieties_nauty->Elt_gamma,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"after Ring_with_action->lift_mapping" << endl;
	}

	if (f_v) {
		cout << "The isomorphism from B to A is given by:" << endl;
		Ring_with_action->PA->A->Group_element->element_print(
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt_gamma, cout);
	}



	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"before Ring_with_action->apply" << endl;
	}
	Ring_with_action->apply(
			Canonical_form_classifier->Classification_of_varieties_nauty->Elt_gamma,
			Vo->Variety_object->eqn
				/* int *eqn_in, the equation defining the set B */,
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2
				/* int *eqn_out image of B under gamma */,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new "
				"after Ring_with_action->apply" << endl;
	}


	//FREE_int(Mtx);

	// now, eqn2 is the image of the curve B (under gamma)
	// and belongs to the orbit of equations associated with A.

	Ring_with_action->PA->F->Projective_space_basic->PG_element_normalize_from_front(
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2, 1,
			Ring_with_action->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "The mapped equation is:";
		Ring_with_action->Poly_ring->print_equation_simple(
				cout, Canonical_form_classifier->Classification_of_varieties_nauty->eqn2);
		cout << endl;
	}


	int f_found = false;

	int idx_equation;

	if (!A->Orb->search_equation(
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2 /*image of the curve B */,
			idx_equation,
			verbose_level - 1)) {
		// need to map points and bitangents under gamma:
		if (f_v) {
			cout << "variety_compute_canonical_form::find_equation_new "
					"we found the canonical form but we did not find "
					"the equation at idx_rational_point_set=" << idx_rational_point_set << endl;
		}
		f_found = false;
		found_at = -1;
	}
	else {
		if (f_v) {
			cout << "variety_compute_canonical_form::find_equation_new "
					"After A->Orb->search_equation, cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form and the equation "
							"at idx_equation " << idx_equation << ", idx_rational_point_set=" << idx_rational_point_set << endl;
		}
		f_found = true;
		found_at = idx_equation;

#if 1
		A->Orb->get_transporter_from_a_to_b(
				idx_equation /* idx_a */, A->Orb->position_of_original_object /* idx_b */,
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt_delta,
				verbose_level);

		Canonical_form_classifier->Ring_with_action->PA->A->Group_element->mult(
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt_gamma,
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt_delta,
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt_phi);

#endif

	}

	if (f_v) {
		cout << "variety_compute_canonical_form::find_equation_new done" << endl;
	}
	return f_found;
}



void variety_compute_canonical_form::add_object_and_compute_canonical_equation_new(
		variety_stabilizer_compute *C,
		int idx, int verbose_level)
// adds the canonical form at position idx, using Classification_of_varieties_nauty
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::add_object_and_compute_canonical_equation_new" << endl;
	}


	if (f_v) {
		cout << "variety_compute_canonical_form::add_object_and_compute_canonical_equation_new "
				"before Canonical_form_classifier->Classification_of_varieties_nauty->CB->add_at_idx" << endl;
	}
	Classification_of_varieties_nauty->CB->add_at_idx(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			idx,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "variety_compute_canonical_form::add_object_and_compute_canonical_equation_new "
				"after Canonical_form_classifier->Classification_of_varieties_nauty->CB->add_at_idx" << endl;
	}


	if (f_v) {
		cout << "variety_compute_canonical_form::add_object_and_compute_canonical_equation_new done" << endl;
	}

}



void variety_compute_canonical_form::compute_canonical_object(
		int verbose_level)
// applies transporter_to_canonical_form to Vo to compute canonical_equation
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_object" << endl;
	}


	Canonical_object = NEW_OBJECT(variety_object_with_action);


	actions::action *A;
	actions::action *A_on_lines;

	A = Ring_with_action->PA->A;
	A_on_lines = Ring_with_action->PA->A_on_lines;

	//Canonical_object->Variety_object = Vo;

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_object "
				"before Vo->apply_transformation_to_self" << endl;
	}
	Vo->apply_transformation_to_self(
			transporter_to_canonical_form,
			A,
			A_on_lines,
			verbose_level);
	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_object "
				"after Vo->apply_transformation_to_self" << endl;
	}

	Canonical_object = NULL; // ToDo

	if (f_v) {
		cout << "variety_compute_canonical_form::compute_canonical_object done" << endl;
	}
}


std::string variety_compute_canonical_form::stringify_csv_entry_one_line_nauty_new(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::stringify_csv_entry_one_line_nauty_new" << endl;
	}

	vector<string> v;
	int j;
	string line;

	if (f_v) {
		cout << "variety_compute_canonical_form::stringify_csv_entry_one_line_nauty_new "
				"before prepare_csv_entry_one_line_nauty" << endl;
	}
	prepare_csv_entry_one_line_nauty_new(
			v, i, verbose_level);
	if (f_v) {
		cout << "variety_compute_canonical_form::stringify_csv_entry_one_line_nauty_new "
				"after prepare_csv_entry_one_line_nauty" << endl;
	}

	for (j = 0; j < v.size(); j++) {
		line += v[j];
		if (j < v.size() - 1) {
			line += ",";
		}
	}
	if (f_v) {
		cout << "variety_compute_canonical_form::stringify_csv_entry_one_line_nauty_new done" << endl;
	}
	return line;
}

void variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new(
		std::vector<std::string> &v, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new" << endl;
	}
	//header = "ROW,Q,FO,PO,SO,Iso_idx,F_Fst,Idx_canonical,Idx_eqn,Eqn,Eqn2,nb_pts_on_curve,pts_on_curve,Bitangents";


	//v.push_back(std::to_string(Vo->cnt)); // ROW
	v.push_back(std::to_string(Vo->PA->P->Subspaces->F->q)); // Q
	v.push_back(std::to_string(Vo->cnt)); // FO
	v.push_back(std::to_string(Vo->po)); // PO
	v.push_back(std::to_string(Vo->so)); // SO
	//v.push_back(std::to_string(Vo->po_go));
	//v.push_back(std::to_string(Vo->po_index));
	if (false /* Canonical_form_classifier->Input->skip_this_one(i) */) {
		if (f_v) {
			cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
					"case input_counter = " << i << " was skipped" << endl;
		}
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));

	}
	else {
		v.push_back(std::to_string(Classification_of_varieties_nauty->Iso_idx[i]));
		v.push_back(std::to_string(Classification_of_varieties_nauty->F_first_time[i]));
		v.push_back(std::to_string(Classification_of_varieties_nauty->Idx_canonical_form[i]));
		v.push_back(std::to_string(Classification_of_varieties_nauty->Idx_equation[i]));
		//v.push_back(std::to_string(Canonical_form_classifier->Output->Tally->rep_idx[i]));
	}





	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Classification_of_varieties_nauty->Input->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Eqn2;
	string s_Pts;
	string s_nb_Pts;
	string s_Bitangents;

	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
				"before Vo->Variety_object->stringify" << endl;
	}


	Vo->Variety_object->stringify(
			s_Eqn, /*s_Eqn2,*/ s_nb_Pts, s_Pts, s_Bitangents);

	s_Eqn2 = "";


	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Canonical_form_classifier->Classification_of_varieties_nauty->Input->nb_objects_to_test
				<< " pushing strings" << endl;
	}

	v.push_back("\"" + s_Eqn + "\"");
	v.push_back("\"" + s_Eqn2 + "\"");
	v.push_back(s_nb_Pts);
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Canonical_form_classifier->Classification_of_varieties_nauty->Input->nb_objects_to_test
				<< " after pushing strings" << endl;
	}

	if (Canonical_form_classifier->carry_through.size()) {

		if (f_v) {
			cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
					"pushing Carrying_through" << endl;
		}

		int j;

		for (j = 0; j < Canonical_form_classifier->carry_through.size(); j++) {
			v.push_back(Vo->Carrying_through[j]);
		}
	}

	if (false /* Canonical_form_classifier->Input->skip_this_one(i)*/) {
		if (f_v) {
			cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new "
					"case input_counter = " << i << " was skipped" << endl;
		}

		int l;

		// we don't have NO, so we create an empty one here:
		other::l1_interfaces::nauty_output NO;

		l = NO.get_output_size(verbose_level);
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + std::to_string(-1) + "\"");
		}

	}
	else {

		int l;
		std::vector<std::string> NO_stringified;

		Variety_stabilizer_compute->NO->stringify_as_vector(
				NO_stringified,
				verbose_level);

		l = NO_stringified.size();
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + NO_stringified[j] + "\"");
		}
		v.push_back("\"" + std::to_string(Variety_stabilizer_compute->Orb->used_length) + "\"");
		v.push_back("\"" + Variety_stabilizer_compute->Stab_gens_variety->group_order_stringify() + "\"");


	}


	if (f_v) {
		cout << "variety_compute_canonical_form::prepare_csv_entry_one_line_nauty_new done" << endl;
	}


}


}}}


