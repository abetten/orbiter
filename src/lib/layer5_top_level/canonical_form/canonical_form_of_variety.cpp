/*
 * canonical_form_of_variety.cpp
 *
 *  Created on: Jan 26, 2023
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



canonical_form_of_variety::canonical_form_of_variety()
{
	Canonical_form_classifier = NULL;

	//std::string fname_case_out;

	counter = 0;
	Vo = NULL;

	canonical_pts = NULL;
	canonical_equation = NULL;
	transporter_to_canonical_form = NULL;
	gens_stab_of_canonical_equation = NULL;

	Stabilizer_of_set_of_rational_points = NULL;

	go_eqn = NULL;

	Canonical_object = NULL;


}

canonical_form_of_variety::~canonical_form_of_variety()
{
	if (canonical_equation) {
		FREE_int(canonical_equation);
	}
	if (transporter_to_canonical_form) {
		FREE_int(transporter_to_canonical_form);
	}
	if (canonical_pts) {
		FREE_lint(canonical_pts);
	}
	if (Stabilizer_of_set_of_rational_points) {
		FREE_OBJECT(Stabilizer_of_set_of_rational_points);
	}
	if (go_eqn) {
		FREE_OBJECT(go_eqn);
	}
	if (Canonical_object) {
		FREE_OBJECT(Canonical_object);
	}
}


void canonical_form_of_variety::init(
		canonical_form_classifier *Canonical_form_classifier,
		std::string &fname_case_out,
		int counter,
		variety_object_with_action *Vo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::init "
				" counter=" << counter
				<< " verbose_level=" << verbose_level
				<< endl;
	}

	canonical_form_of_variety::Canonical_form_classifier = Canonical_form_classifier;
	canonical_form_of_variety::fname_case_out.assign(fname_case_out);
	canonical_form_of_variety::counter = counter;
	canonical_form_of_variety::Vo = Vo;

	canonical_equation =
			NEW_int(Canonical_form_classifier->Poly_ring->get_nb_monomials());
	transporter_to_canonical_form =
			NEW_int(Canonical_form_classifier->PA->A->elt_size_in_int);


	canonical_pts = NEW_lint(Vo->Variety_object->Point_sets->Set_size[0]);

	go_eqn = NEW_OBJECT(ring_theory::longinteger_object);



	if (f_v) {
		cout << "canonical_form_of_variety::init done" << endl;
	}
}


void canonical_form_of_variety::compute_canonical_form_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"verbose_level=" << verbose_level << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"before classify_using_nauty" << endl;
	}

	classify_using_nauty(
			verbose_level);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"after classify_using_nauty" << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"f_found_canonical_form=" << Stabilizer_of_set_of_rational_points->f_found_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"idx_canonical_form=" << Stabilizer_of_set_of_rational_points->idx_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"idx_equation=" << Stabilizer_of_set_of_rational_points->idx_equation << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"f_found_eqn=" << Stabilizer_of_set_of_rational_points->f_found_eqn << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"group_order=" << Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order_as_lint() << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty done" << endl;
	}
}

void canonical_form_of_variety::compute_canonical_form_nauty_new(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"verbose_level=" << verbose_level << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"before classify_using_nauty_new" << endl;
	}

	classify_using_nauty_new(
			verbose_level);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"after classify_using_nauty_new" << endl;
	}
	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"f_found_canonical_form=" << Stabilizer_of_set_of_rational_points->f_found_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"idx_canonical_form=" << Stabilizer_of_set_of_rational_points->idx_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"idx_equation=" << Stabilizer_of_set_of_rational_points->idx_equation << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"f_found_eqn=" << Stabilizer_of_set_of_rational_points->f_found_eqn << endl;
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new "
				"group_order=" << Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order_as_lint() << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty_new done" << endl;
	}
}


void canonical_form_of_variety::classify_using_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"verbose_level=" << verbose_level << endl;
	}

	ring_theory::longinteger_object go;


	long int *alpha; // [Canonical_form_classifier->Output->canonical_labeling_len]
	int *gamma; // [Canonical_form_classifier->Output->canonical_labeling_len]





	Stabilizer_of_set_of_rational_points = NEW_OBJECT(stabilizer_of_set_of_rational_points);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"before Canonical_form_nauty->init" << endl;
	}
	Stabilizer_of_set_of_rational_points->init(Canonical_form_classifier, verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"after Canonical_form_nauty->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"before Canonical_form_nauty->compute_canonical_form_of_variety" << endl;
	}
	Stabilizer_of_set_of_rational_points->compute_canonical_form_of_variety(
			this,
			verbose_level - 2);
	// Computes the canonical labeling of the graph associated with
	// the set of rational points of the variety.
	// Computes the stabilizer of the set of rational points of the variety.
	// Computes the orbit of the equation under the stabilizer of the set.
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"after Canonical_form_nauty->compute_canonical_form_of_variety" << endl;
	}


	Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order(go);

	//goi = go.as_int();

	//FREE_OBJECT(gens_stab_of_canonical_equation);

	Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len =
			Stabilizer_of_set_of_rational_points->NO->N; //canonical_labeling_len;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"canonical_labeling_len=" <<
				Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len << endl;
	}
	if (Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len == 0) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"canonical_labeling_len == 0, error" << endl;
		exit(1);
	}
	alpha = NEW_lint(Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len);
	gamma = NEW_int(Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len);


	if (Canonical_form_classifier->Classification_of_varieties->CB->n == 0) {
		Canonical_form_classifier->Classification_of_varieties->CB->init(
				Canonical_form_classifier->Input->nb_objects_to_test,
				Stabilizer_of_set_of_rational_points->Canonical_form->get_allocated_length(),
				verbose_level - 2);
	}
	//int f_found;
	//int idx;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"before Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	Canonical_form_classifier->Classification_of_varieties->CB->search_and_add_if_new(
			Stabilizer_of_set_of_rational_points->Canonical_form->get_data(),
			Stabilizer_of_set_of_rational_points /* void *extra_data */,
			Stabilizer_of_set_of_rational_points->f_found_canonical_form,
			Stabilizer_of_set_of_rational_points->idx_canonical_form,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty "
				"after Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	// if f_found is true: idx_canonical_form is where the canonical form was found.
	// if f_found is false: idx_canonical_form is where the new canonical form was added.


	if (Stabilizer_of_set_of_rational_points->f_found_canonical_form) {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form at idx_canonical_form = "
					<< Stabilizer_of_set_of_rational_points->idx_canonical_form << endl;
		}




		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty "
					"before handle_repeated_canonical_form_of_set" << endl;
		}
		handle_repeated_canonical_form_of_set(
				Stabilizer_of_set_of_rational_points->idx_canonical_form,
				Stabilizer_of_set_of_rational_points,
				alpha, gamma,
				Stabilizer_of_set_of_rational_points->idx_canonical_form,
				Stabilizer_of_set_of_rational_points->idx_equation,
				Stabilizer_of_set_of_rational_points->f_found_eqn,
				verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty "
					"after handle_repeated_canonical_form_of_set" << endl;
		}


	} // if f_found_canonical_form
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " The canonical form is new" << endl;
		}

		// The canonical form has already been added,
		// at position idx_canonical_form,
		// with Canonical_form_nauty as extra_data.
	}

	FREE_lint(alpha);
	FREE_int(gamma);


	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty done" << endl;
	}
}



void canonical_form_of_variety::classify_using_nauty_new(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"verbose_level=" << verbose_level << endl;
	}

	ring_theory::longinteger_object go;


	long int *alpha; // [Canonical_form_classifier->Output->canonical_labeling_len]
	int *gamma; // [Canonical_form_classifier->Output->canonical_labeling_len]





	Stabilizer_of_set_of_rational_points = NEW_OBJECT(stabilizer_of_set_of_rational_points);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"before Canonical_form_nauty->init" << endl;
	}
	Stabilizer_of_set_of_rational_points->init(Canonical_form_classifier, verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"after Canonical_form_nauty->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"before Canonical_form_nauty->compute_canonical_form_of_variety" << endl;
	}
	Stabilizer_of_set_of_rational_points->compute_canonical_form_of_variety(
			this,
			verbose_level - 2);
	// Computes the canonical labeling of the graph associated with
	// the set of rational points of the variety.
	// Computes the stabilizer of the set of rational points of the variety.
	// Computes the orbit of the equation under the stabilizer of the set.
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"after Canonical_form_nauty->compute_canonical_form_of_variety" << endl;
	}


	Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order(go);

	//goi = go.as_int();

	//FREE_OBJECT(gens_stab_of_canonical_equation);

	Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len =
			Stabilizer_of_set_of_rational_points->NO->N; //canonical_labeling_len;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"canonical_labeling_len=" <<
				Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len << endl;
	}
	if (Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len == 0) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"canonical_labeling_len == 0, error" << endl;
		exit(1);
	}
	alpha = NEW_lint(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
	gamma = NEW_int(Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);


	if (Canonical_form_classifier->Classification_of_varieties_nauty->CB->n == 0) {
		Canonical_form_classifier->Classification_of_varieties_nauty->CB->init(
				Canonical_form_classifier->Classification_of_varieties_nauty->nb_objects_to_test,
				/*Canonical_form_classifier->Input->nb_objects_to_test,*/
				Stabilizer_of_set_of_rational_points->Canonical_form->get_allocated_length(),
				verbose_level - 2);
	}
	//int f_found;
	//int idx;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"before Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	Canonical_form_classifier->Classification_of_varieties_nauty->CB->search_and_add_if_new(
			Stabilizer_of_set_of_rational_points->Canonical_form->get_data(),
			Stabilizer_of_set_of_rational_points /* void *extra_data */,
			Stabilizer_of_set_of_rational_points->f_found_canonical_form,
			Stabilizer_of_set_of_rational_points->idx_canonical_form,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new "
				"after Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	// if f_found is true: idx_canonical_form is where the canonical form was found.
	// if f_found is false: idx_canonical_form is where the new canonical form was added.


	if (Stabilizer_of_set_of_rational_points->f_found_canonical_form) {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty_new "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form at idx_canonical_form = "
					<< Stabilizer_of_set_of_rational_points->idx_canonical_form << endl;
		}




		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty_new "
					"before handle_repeated_canonical_form_of_set_new" << endl;
		}
		handle_repeated_canonical_form_of_set_new(
				Stabilizer_of_set_of_rational_points->idx_canonical_form,
				Stabilizer_of_set_of_rational_points,
				alpha, gamma,
				Stabilizer_of_set_of_rational_points->idx_canonical_form,
				Stabilizer_of_set_of_rational_points->idx_equation,
				Stabilizer_of_set_of_rational_points->f_found_eqn,
				verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty_new "
					"after handle_repeated_canonical_form_of_set_new" << endl;
		}


	} // if f_found_canonical_form
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_using_nauty_new "
					"After search_and_add_if_new, "
					"cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " The canonical form is new" << endl;
		}

		// The canonical form has already been added,
		// at position idx_canonical_form,
		// with Canonical_form_nauty as extra_data.
	}

	FREE_lint(alpha);
	FREE_int(gamma);


	if (f_v) {
		cout << "canonical_form_of_variety::classify_using_nauty_new done" << endl;
	}
}



void canonical_form_of_variety::handle_repeated_canonical_form_of_set(
		int idx,
		stabilizer_of_set_of_rational_points *C,
		long int *alpha, int *gamma,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
				"verbose_level=" << verbose_level << endl;
	}

	idx_equation = -1;
	f_found_eqn = false;

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
				"starting loop over idx1" << endl;
	}

	for (idx_canonical_form = idx; idx_canonical_form >= 0; idx_canonical_form--) {



		// test if entry at idx1 is equal to C.
		// if not, break

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"before Canonical_form_classifier->CB->compare_at "
					"idx_canonical_form = " << idx_canonical_form << endl;
		}
		if (Canonical_form_classifier->Classification_of_varieties->CB->compare_at(
				C->Canonical_form->get_data(), idx_canonical_form) != 0) {
			if (f_v) {
				cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
						"at idx_canonical_form = " << idx_canonical_form
						<< " is not equal, break" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"canonical form at " << idx_canonical_form << " is equal" << endl;
		}


		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"before find_equation" << endl;
		}

		f_found_eqn = find_equation(
				C,
				alpha, gamma,
				idx_canonical_form, idx_equation,
				verbose_level);

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"after find_equation" << endl;
			if (f_found_eqn) {
				cout << "We found the equation" << endl;
			}
			else {
				cout << "We did not find the equation" << endl;
			}
		}

		if (f_found_eqn) {
			break;
		}


	}


	if (f_found_eqn) {
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"we found the equation at index " << idx_equation << endl;
		}
	}
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"we found the canonical form but we did "
					"not find the equation" << endl;
		}

		// add the canonical form at position idx

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"before add_object_and_compute_canonical_equation" << endl;
		}
		add_object_and_compute_canonical_equation(C, idx, verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set "
					"after add_object_and_compute_canonical_equation" << endl;
		}

	}

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set done" << endl;
	}

}



void canonical_form_of_variety::handle_repeated_canonical_form_of_set_new(
		int idx,
		stabilizer_of_set_of_rational_points *C,
		long int *alpha, int *gamma,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
				"verbose_level=" << verbose_level << endl;
	}

	idx_equation = -1;
	f_found_eqn = false;

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
				"starting loop over idx1" << endl;
	}

	for (idx_canonical_form = idx; idx_canonical_form >= 0; idx_canonical_form--) {



		// test if entry at idx1 is equal to C.
		// if not, break

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"before Canonical_form_classifier->CB->compare_at "
					"idx_canonical_form = " << idx_canonical_form << endl;
		}
		if (Canonical_form_classifier->Classification_of_varieties_nauty->CB->compare_at(
				C->Canonical_form->get_data(), idx_canonical_form) != 0) {
			if (f_v) {
				cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
						"at idx_canonical_form = " << idx_canonical_form
						<< " is not equal, break" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"canonical form at " << idx_canonical_form << " is equal" << endl;
		}


		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"before find_equation_new" << endl;
		}

		f_found_eqn = find_equation_new(
				C,
				alpha, gamma,
				idx_canonical_form, idx_equation,
				verbose_level);

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"after find_equation_new" << endl;
			if (f_found_eqn) {
				cout << "We found the equation" << endl;
			}
			else {
				cout << "We did not find the equation" << endl;
			}
		}

		if (f_found_eqn) {
			break;
		}


	}


	if (f_found_eqn) {
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"we found the equation at index " << idx_equation << endl;
		}
	}
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"we found the canonical form but we did "
					"not find the equation" << endl;
		}

		// add the canonical form at position idx

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"before add_object_and_compute_canonical_equation_new" << endl;
		}
		add_object_and_compute_canonical_equation_new(C, idx, verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new "
					"after add_object_and_compute_canonical_equation_new" << endl;
		}

	}

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_canonical_form_of_set_new done" << endl;
	}

}



int canonical_form_of_variety::find_equation(
		stabilizer_of_set_of_rational_points *C,
		long int *alpha, int *gamma,
		int idx1, int &found_at,
		int verbose_level)
// gets the canonical_form_nauty object from
// Canonical_form_classifier->Output->CB->Type_extra_data[idx1]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"verbose_level=" << verbose_level << endl;
	}

	int *alpha_inv;
	int *beta_inv;
	int i, j;
	int f_found = false;

	stabilizer_of_set_of_rational_points *C1;
	C1 = (stabilizer_of_set_of_rational_points *)
			Canonical_form_classifier->Classification_of_varieties->CB->Type_extra_data[idx1];

	alpha_inv = C1->NO->canonical_labeling;
	//alpha_inv = C1->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"alpha_inv = " << endl;
		Int_vec_print(cout,
				alpha_inv,
				Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len);
		cout << endl;
	}

	beta_inv = C->NO->canonical_labeling;
	//beta_inv = C->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"beta_inv = " << endl;
		Int_vec_print(
				cout,
				beta_inv,
				Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len);
		cout << endl;
	}

	// compute gamma = alpha * beta^-1 (left to right multiplication),
	// which maps the points on curve C1 to the points on curve C


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"computing alpha" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len; i++) {
		j = alpha_inv[i];
		alpha[j] = i;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"computing gamma" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len; i++) {
		gamma[i] = beta_inv[alpha[i]];
	}
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"gamma = " << endl;
		Int_vec_print(
				cout,
				gamma,
				Canonical_form_classifier->Classification_of_varieties->canonical_labeling_len);
		cout << endl;
	}


	// gamma maps C1 to C.
	// So, in the contragredient action,
	// it maps the equation of C to an equation in the orbit of the equation C1,
	// which is what we want.

	// turn gamma into a matrix



	int d = Canonical_form_classifier->PA->P->Subspaces->n + 1;
	int *Mtx; // [d * d + 1]


	Mtx = NEW_int(d * d + 1);


	int frobenius;
	linear_algebra::linear_algebra_global LA;

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"before LA.reverse_engineer_semilinear_map" << endl;
	}

	// works for any dimension:

	LA.reverse_engineer_semilinear_map(
			Canonical_form_classifier->PA->P->Subspaces->F,
			Canonical_form_classifier->PA->P->Subspaces->n,
			gamma, Mtx, frobenius,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"after LA.reverse_engineer_semilinear_map" << endl;
	}

	Mtx[d * d] = frobenius;

	Canonical_form_classifier->PA->A->Group_element->make_element(
			Canonical_form_classifier->Classification_of_varieties->Elt, Mtx,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "The isomorphism from C to C1 is given by:" << endl;
		Canonical_form_classifier->PA->A->Group_element->element_print(
				Canonical_form_classifier->Classification_of_varieties->Elt, cout);
	}



	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"before substitute_semilinear" << endl;
	}
	Canonical_form_classifier->Poly_ring->substitute_semilinear(
			Vo->Variety_object->eqn /* coeff_in */,
			Canonical_form_classifier->Classification_of_varieties->eqn2 /* coeff_out */,
			Canonical_form_classifier->PA->A->is_semilinear_matrix_group(),
			frobenius, Mtx,
			0/*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"after substitute_semilinear" << endl;
	}

	FREE_int(Mtx);

	// now, eqn2 is the image of the curve C
	// and belongs to the orbit of equations associated with C1.

	Canonical_form_classifier->PA->F->Projective_space_basic->PG_element_normalize_from_front(
			Canonical_form_classifier->Classification_of_varieties->eqn2, 1,
			Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "The mapped equation is:";
		Canonical_form_classifier->Poly_ring->print_equation_simple(
				cout, Canonical_form_classifier->Classification_of_varieties->eqn2);
		cout << endl;
	}




	int idx2;

	if (!C1->Orb->search_equation(
			Canonical_form_classifier->Classification_of_varieties->eqn2 /*new_object */, idx2,
			true)) {
		// need to map points and bitangents under gamma:
		if (f_v) {
			cout << "canonical_form_of_variety::find_equation "
					"we found the canonical form but we did not find "
					"the equation at idx1=" << idx1 << endl;
		}

	}
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::find_equation "
					"After C1->Orb->search_equation, cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form and the equation "
							"at idx2 " << idx2 << ", idx1=" << idx1 << endl;
		}
		f_found = true;
		found_at = idx2;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation done" << endl;
	}
	return f_found;
}




int canonical_form_of_variety::find_equation_new(
		stabilizer_of_set_of_rational_points *C,
		long int *alpha, int *gamma,
		int idx1, int &found_at,
		int verbose_level)
// gets the canonical_form_nauty object from
// Canonical_form_classifier->Output->CB->Type_extra_data[idx1]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"verbose_level=" << verbose_level << endl;
	}

	int *alpha_inv;
	int *beta_inv;
	int i, j;
	int f_found = false;

	stabilizer_of_set_of_rational_points *C1;
	C1 = (stabilizer_of_set_of_rational_points *)
			Canonical_form_classifier->Classification_of_varieties_nauty->CB->Type_extra_data[idx1];

	alpha_inv = C1->NO->canonical_labeling;
	//alpha_inv = C1->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"alpha_inv = " << endl;
		Int_vec_print(cout,
				alpha_inv,
				Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
		cout << endl;
	}

	beta_inv = C->NO->canonical_labeling;
	//beta_inv = C->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"beta_inv = " << endl;
		Int_vec_print(
				cout,
				beta_inv,
				Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
		cout << endl;
	}

	// compute gamma = alpha * beta^-1 (left to right multiplication),
	// which maps the points on curve C1 to the points on curve C


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"computing alpha" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len; i++) {
		j = alpha_inv[i];
		alpha[j] = i;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"computing gamma" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len; i++) {
		gamma[i] = beta_inv[alpha[i]];
	}
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"gamma = " << endl;
		Int_vec_print(
				cout,
				gamma,
				Canonical_form_classifier->Classification_of_varieties_nauty->canonical_labeling_len);
		cout << endl;
	}


	// gamma maps C1 to C.
	// So, in the contragredient action,
	// it maps the equation of C to an equation in the orbit of the equation C1,
	// which is what we want.

	// turn gamma into a matrix



	int d = Canonical_form_classifier->PA->P->Subspaces->n + 1;
	int *Mtx; // [d * d + 1]


	Mtx = NEW_int(d * d + 1);


	int frobenius;
	linear_algebra::linear_algebra_global LA;

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"before LA.reverse_engineer_semilinear_map" << endl;
	}

	// works for any dimension:

	LA.reverse_engineer_semilinear_map(
			Canonical_form_classifier->PA->P->Subspaces->F,
			Canonical_form_classifier->PA->P->Subspaces->n,
			gamma, Mtx, frobenius,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"after LA.reverse_engineer_semilinear_map" << endl;
	}

	Mtx[d * d] = frobenius;

	Canonical_form_classifier->PA->A->Group_element->make_element(
			Canonical_form_classifier->Classification_of_varieties_nauty->Elt, Mtx,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "The isomorphism from C to C1 is given by:" << endl;
		Canonical_form_classifier->PA->A->Group_element->element_print(
				Canonical_form_classifier->Classification_of_varieties_nauty->Elt, cout);
	}



	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"before substitute_semilinear" << endl;
	}
	Canonical_form_classifier->Poly_ring->substitute_semilinear(
			Vo->Variety_object->eqn /* coeff_in */,
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2 /* coeff_out */,
			Canonical_form_classifier->PA->A->is_semilinear_matrix_group(),
			frobenius, Mtx,
			0/*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new "
				"after substitute_semilinear" << endl;
	}

	FREE_int(Mtx);

	// now, eqn2 is the image of the curve C
	// and belongs to the orbit of equations associated with C1.

	Canonical_form_classifier->PA->F->Projective_space_basic->PG_element_normalize_from_front(
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2, 1,
			Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "The mapped equation is:";
		Canonical_form_classifier->Poly_ring->print_equation_simple(
				cout, Canonical_form_classifier->Classification_of_varieties_nauty->eqn2);
		cout << endl;
	}




	int idx2;

	if (!C1->Orb->search_equation(
			Canonical_form_classifier->Classification_of_varieties_nauty->eqn2 /*new_object */, idx2,
			true)) {
		// need to map points and bitangents under gamma:
		if (f_v) {
			cout << "canonical_form_of_variety::find_equation_new "
					"we found the canonical form but we did not find "
					"the equation at idx1=" << idx1 << endl;
		}

	}
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::find_equation_new "
					"After C1->Orb->search_equation, cnt = " << Vo->cnt
					<< " po = " << Vo->po
					<< " so = " << Vo->so
					<< " We found the canonical form and the equation "
							"at idx2 " << idx2 << ", idx1=" << idx1 << endl;
		}
		f_found = true;
		found_at = idx2;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation_new done" << endl;
	}
	return f_found;
}





void canonical_form_of_variety::add_object_and_compute_canonical_equation(
		stabilizer_of_set_of_rational_points *C,
		int idx, int verbose_level)
// adds the canonical form at position idx
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"before Canonical_form_classifier->CB->add_at_idx" << endl;
	}
	Canonical_form_classifier->Classification_of_varieties->CB->add_at_idx(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			idx,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"after Canonical_form_classifier->CB->add_at_idx" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation done" << endl;
	}

}


void canonical_form_of_variety::add_object_and_compute_canonical_equation_new(
		stabilizer_of_set_of_rational_points *C,
		int idx, int verbose_level)
// adds the canonical form at position idx, using Classification_of_varieties_nauty
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation_new" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation_new "
				"before Canonical_form_classifier->Classification_of_varieties_nauty->CB->add_at_idx" << endl;
	}
	Canonical_form_classifier->Classification_of_varieties_nauty->CB->add_at_idx(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			idx,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation_new "
				"after Canonical_form_classifier->Classification_of_varieties_nauty->CB->add_at_idx" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation_new done" << endl;
	}

}




void canonical_form_of_variety::compute_canonical_form_substructure(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_substructure" << endl;
	}



	if (Vo->Variety_object->Point_sets->Set_size[0] >= Canonical_form_classifier->get_description()->substructure_size) {

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"nb_pts is sufficient" << endl;
		}

		//ring_theory::longinteger_object go_eqn;




		canonical_form_substructure *CFS;

		CFS = NEW_OBJECT(canonical_form_substructure);

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"before CFS->classify_curve_with_substructure" << endl;
		}

		CFS->classify_curve_with_substructure(
				this,
				verbose_level);

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"after CFS->classify_curve_with_substructure" << endl;
		}

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"storing CFS in CFS_table[counter], "
					"counter = " << counter << endl;
		}
		Canonical_form_classifier->Classification_of_varieties->CFS_table[counter] = CFS;
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"storing canonical_equation in "
					"Canonical_forms[counter], "
					"counter = " << counter << endl;
		}
		Int_vec_copy(canonical_equation,
				Canonical_form_classifier->Classification_of_varieties->Canonical_equation +
				counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
				Canonical_form_classifier->Poly_ring->get_nb_monomials());
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"storing group order in Goi[]" << endl;
		}
		Canonical_form_classifier->Classification_of_varieties->Goi[counter] = go_eqn->as_lint();
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"Goi[counter] = " << Canonical_form_classifier->Classification_of_varieties->Goi[counter] << endl;
		}

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"after CFS->classify_curve_with_substructure" << endl;
		}
	}
	else {


		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"too small for substructure algorithm. "
					"Skipping" << endl;
		}

		Canonical_form_classifier->Classification_of_varieties->CFS_table[counter] = NULL;
		Int_vec_zero(
				Canonical_form_classifier->Classification_of_varieties->Canonical_equation +
				counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
				Canonical_form_classifier->Poly_ring->get_nb_monomials());
		Canonical_form_classifier->Classification_of_varieties->Goi[counter] = -1;

	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_substructure done" << endl;
	}
}


void canonical_form_of_variety::compute_canonical_object(
		int verbose_level)
// applies transporter_to_canonical_form to Vo to compute canonical_equation
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object" << endl;
	}


	Canonical_object = NEW_OBJECT(variety_object_with_action);


	actions::action *A;
	actions::action *A_on_lines;

	A = Canonical_form_classifier->PA->A;
	A_on_lines = Canonical_form_classifier->PA->A_on_lines;

	//Canonical_object->Variety_object = Vo;

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object "
				"before Vo->apply_transformation" << endl;
	}
	Vo->apply_transformation(
			transporter_to_canonical_form,
			A,
			A_on_lines,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object "
				"after Vo->apply_transformation" << endl;
	}

	Canonical_object = NULL; // ToDo

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object done" << endl;
	}
}

std::string canonical_form_of_variety::stringify_csv_entry_one_line(
		int i, int verbose_level)
{

	vector<string> v;
	int j;
	string line;

	prepare_csv_entry_one_line(
			v, i, verbose_level);
	for (j = 0; j < v.size(); j++) {
		line = v[j];
		if (j < v.size()) {
			line += ",";
		}
	}
	return line;
}

void canonical_form_of_variety::prepare_csv_entry_one_line(
		std::vector<std::string> &v,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line" << endl;
	}

	v.push_back(std::to_string(Vo->cnt));
	v.push_back(std::to_string(Vo->po));
	v.push_back(std::to_string(Vo->so));
	v.push_back(std::to_string(Vo->po_go));
	v.push_back(std::to_string(Vo->po_index));
	v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties->Tally->rep_idx[i]));


	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Eqn2;
	string s_nb_Pts;
	string s_Pts;
	string s_Bitangents;
	string s_transporter;
	string s_Eqn_canonical;
	string s_Pts_canonical;
	string s_Bitangents_canonical;
	std::string s_tl, s_gens, s_go;


	Vo->Variety_object->stringify(
			s_Eqn, /*s_Eqn2,*/ s_nb_Pts, s_Pts, s_Bitangents);

	s_Eqn2 = "";

	s_transporter = Canonical_form_classifier->PA->A->Group_element->stringify(
			transporter_to_canonical_form);

	s_Eqn_canonical = Int_vec_stringify(
			canonical_equation, 15);

	s_Pts_canonical = Canonical_object->stringify_Pts();

	s_Bitangents_canonical = Canonical_object->stringify_bitangents();

	if (gens_stab_of_canonical_equation) {

		if (false) {
			cout << "classification_of_varieties::prepare_csv_entry_one_line "
					"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
					<< " before gens_stab_of_canonical_equation->stringify" << endl;
		}
		gens_stab_of_canonical_equation->stringify(s_tl, s_gens, s_go);
	}
	else {
		cout << "gens_stab_of_canonical_equation is not available" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " writing data" << endl;
	}

	v.push_back("\"" + s_Eqn + "\"");
	v.push_back("\"" + s_Eqn2 + "\"");
	v.push_back("\"" + s_nb_Pts + "\"");
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
	v.push_back("\"" + s_transporter + "\"");
	v.push_back("\"" + s_Eqn_canonical + "\"");
	v.push_back("\"" + s_Pts_canonical + "\"");
	v.push_back("\"" + s_Bitangents_canonical + "\"");
	v.push_back("\"" + s_tl + "\"");
	v.push_back("\"" + s_gens + "\"");
	v.push_back(s_go);


	if (Canonical_form_classifier->carry_through.size()) {
		int j;

		for (j = 0; j < Canonical_form_classifier->carry_through.size(); j++) {
			v.push_back(Vo->Carrying_through[j]);
		}
	}

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line done" << endl;
	}


}

std::string canonical_form_of_variety::stringify_csv_entry_one_line_nauty(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty" << endl;
	}

	vector<string> v;
	int j;
	string line;

	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty "
				"before prepare_csv_entry_one_line_nauty" << endl;
	}
	prepare_csv_entry_one_line_nauty(
			v, i, verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty "
				"after prepare_csv_entry_one_line_nauty" << endl;
	}

	for (j = 0; j < v.size(); j++) {
		line += v[j];
		if (j < v.size() - 1) {
			line += ",";
		}
	}
	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty done" << endl;
	}
	return line;
}


std::string canonical_form_of_variety::stringify_csv_entry_one_line_nauty_new(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty_new" << endl;
	}

	vector<string> v;
	int j;
	string line;

	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty_new "
				"before prepare_csv_entry_one_line_nauty" << endl;
	}
	prepare_csv_entry_one_line_nauty_new(
			v, i, verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty_new "
				"after prepare_csv_entry_one_line_nauty" << endl;
	}

	for (j = 0; j < v.size(); j++) {
		line += v[j];
		if (j < v.size() - 1) {
			line += ",";
		}
	}
	if (f_v) {
		cout << "canonical_form_of_variety::stringify_csv_entry_one_line_nauty_new done" << endl;
	}
	return line;
}





void canonical_form_of_variety::prepare_csv_entry_one_line_nauty(
		std::vector<std::string> &v, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty" << endl;
	}


	v.push_back(std::to_string(Vo->cnt));
	v.push_back(std::to_string(Vo->po));
	v.push_back(std::to_string(Vo->so));
	v.push_back(std::to_string(Vo->po_go));
	v.push_back(std::to_string(Vo->po_index));
	if (Canonical_form_classifier->Input->skip_this_one(i)) {
		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty "
					"case input_counter = " << i << " was skipped" << endl;
		}
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));

	}
	else {
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties->Iso_idx[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties->F_first_time[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties->Idx_canonical_form[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties->Idx_equation[i]));
		//v.push_back(std::to_string(Canonical_form_classifier->Output->Tally->rep_idx[i]));
	}





	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Eqn2;
	string s_Pts;
	string s_nb_Pts;
	string s_Bitangents;

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty "
				"before Vo->Variety_object->stringify" << endl;
	}


	Vo->Variety_object->stringify(
			s_Eqn, /*s_Eqn2,*/ s_nb_Pts, s_Pts, s_Bitangents);

	s_Eqn2 = "";


	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " pushing strings" << endl;
	}

	v.push_back("\"" + s_Eqn + "\"");
	v.push_back("\"" + s_Eqn2 + "\"");
	v.push_back(s_nb_Pts);
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " after pushing strings" << endl;
	}

	if (Canonical_form_classifier->carry_through.size()) {

		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty "
					"pushing Carrying_through" << endl;
		}

		int j;

		for (j = 0; j < Canonical_form_classifier->carry_through.size(); j++) {
			v.push_back(Vo->Carrying_through[j]);
		}
	}

	if (Canonical_form_classifier->Input->skip_this_one(i)) {
		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty "
					"case input_counter = " << i << " was skipped" << endl;
		}

		int l;

		// we don't have NO, so we create an empty one here:
		l1_interfaces::nauty_output NO;

		l = NO.get_output_size(verbose_level);
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + std::to_string(-1) + "\"");
		}

	}
	else {

		int l;
		std::vector<std::string> NO_stringified;

		Stabilizer_of_set_of_rational_points->NO->stringify_as_vector(
				NO_stringified,
				verbose_level);

		l = NO_stringified.size();
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + NO_stringified[j] + "\"");
		}
		v.push_back("\"" + std::to_string(Stabilizer_of_set_of_rational_points->Orb->used_length) + "\"");
		v.push_back("\"" + Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order_stringify() + "\"");


	}


	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty done" << endl;
	}


}


void canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new(
		std::vector<std::string> &v, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new" << endl;
	}


	v.push_back(std::to_string(Vo->cnt));
	v.push_back(std::to_string(Vo->po));
	v.push_back(std::to_string(Vo->so));
	v.push_back(std::to_string(Vo->po_go));
	v.push_back(std::to_string(Vo->po_index));
	if (false /* Canonical_form_classifier->Input->skip_this_one(i) */) {
		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new "
					"case input_counter = " << i << " was skipped" << endl;
		}
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));
		v.push_back(std::to_string(-1));

	}
	else {
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties_nauty->Iso_idx[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties_nauty->F_first_time[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties_nauty->Idx_canonical_form[i]));
		v.push_back(std::to_string(Canonical_form_classifier->Classification_of_varieties_nauty->Idx_equation[i]));
		//v.push_back(std::to_string(Canonical_form_classifier->Output->Tally->rep_idx[i]));
	}





	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Canonical_form_classifier->Classification_of_varieties_nauty->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Eqn2;
	string s_Pts;
	string s_nb_Pts;
	string s_Bitangents;

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new "
				"before Vo->Variety_object->stringify" << endl;
	}


	Vo->Variety_object->stringify(
			s_Eqn, /*s_Eqn2,*/ s_nb_Pts, s_Pts, s_Bitangents);

	s_Eqn2 = "";


	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Canonical_form_classifier->Classification_of_varieties_nauty->nb_objects_to_test
				<< " pushing strings" << endl;
	}

	v.push_back("\"" + s_Eqn + "\"");
	v.push_back("\"" + s_Eqn2 + "\"");
	v.push_back(s_nb_Pts);
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty_new "
				"i=" << i << " / " << Canonical_form_classifier->Classification_of_varieties_nauty->nb_objects_to_test
				<< " after pushing strings" << endl;
	}

	if (Canonical_form_classifier->carry_through.size()) {

		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new "
					"pushing Carrying_through" << endl;
		}

		int j;

		for (j = 0; j < Canonical_form_classifier->carry_through.size(); j++) {
			v.push_back(Vo->Carrying_through[j]);
		}
	}

	if (false /* Canonical_form_classifier->Input->skip_this_one(i)*/) {
		if (f_v) {
			cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new "
					"case input_counter = " << i << " was skipped" << endl;
		}

		int l;

		// we don't have NO, so we create an empty one here:
		l1_interfaces::nauty_output NO;

		l = NO.get_output_size(verbose_level);
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + std::to_string(-1) + "\"");
		}

	}
	else {

		int l;
		std::vector<std::string> NO_stringified;

		Stabilizer_of_set_of_rational_points->NO->stringify_as_vector(
				NO_stringified,
				verbose_level);

		l = NO_stringified.size();
		int j;

		for (j = 0; j < l; j++) {
			v.push_back("\"" + NO_stringified[j] + "\"");
		}
		v.push_back("\"" + std::to_string(Stabilizer_of_set_of_rational_points->Orb->used_length) + "\"");
		v.push_back("\"" + Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order_stringify() + "\"");


	}


	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty_new done" << endl;
	}


}



}}}


