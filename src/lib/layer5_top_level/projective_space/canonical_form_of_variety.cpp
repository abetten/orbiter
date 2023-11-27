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
namespace projective_geometry {



canonical_form_of_variety::canonical_form_of_variety()
{
	Canonical_form_classifier = NULL;

	//std::string fname_case_out;

	Qco = NULL;

	canonical_pts = NULL;
	canonical_equation = NULL;
	transporter_to_canonical_form = NULL;
	gens_stab_of_canonical_equation = NULL;

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
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_object_with_action *Qco,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::init "
				"verbose_level=" << verbose_level << endl;
	}

	canonical_form_of_variety::Canonical_form_classifier = Canonical_form_classifier;
	canonical_form_of_variety::fname_case_out.assign(fname_case_out);
	canonical_form_of_variety::Qco = Qco;

	canonical_equation =
			NEW_int(Canonical_form_classifier->Poly_ring->get_nb_monomials());
	transporter_to_canonical_form =
			NEW_int(Canonical_form_classifier->PA->A->elt_size_in_int);


	canonical_pts = NEW_lint(Qco->Quartic_curve_object->nb_pts);

	go_eqn = NEW_OBJECT(ring_theory::longinteger_object);



	if (f_v) {
		cout << "canonical_form_of_variety::init done" << endl;
	}
}


void canonical_form_of_variety::classify_curve_nauty(
		int &f_found_canonical_form,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"verbose_level=" << verbose_level << endl;
	}

	canonical_form_nauty *C;
	ring_theory::longinteger_object go;


	long int *alpha;
	int *gamma;





	C = NEW_OBJECT(canonical_form_nauty);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before C->init" << endl;
	}
	C->init(Canonical_form_classifier, verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after C->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before C->canonical_form_of_quartic_curve" << endl;
	}
	C->canonical_form_of_quartic_curve(
			this,
			verbose_level - 2);
	// Computes the canonical labeling of the graph associated with
	// the set of rational points of the curve.
	// Computes the stabilizer of the set of rational points of the curve.
	// Computes the orbit of the equation under the stabilizer of the set.
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after C->canonical_form_of_quartic_curve" << endl;
	}

	C->Stab_gens_quartic->group_order(go);

	//FREE_OBJECT(gens_stab_of_canonical_equation);

	Canonical_form_classifier->Output->canonical_labeling_len = C->canonical_labeling_len;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"canonical_labeling_len=" <<
				Canonical_form_classifier->Output->canonical_labeling_len << endl;
	}
	if (Canonical_form_classifier->Output->canonical_labeling_len == 0) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"canonical_labeling_len == 0, error" << endl;
		exit(1);
	}
	alpha = NEW_lint(Canonical_form_classifier->Output->canonical_labeling_len);
	gamma = NEW_int(Canonical_form_classifier->Output->canonical_labeling_len);


	if (Canonical_form_classifier->Output->CB->n == 0) {
		Canonical_form_classifier->Output->CB->init(
				Canonical_form_classifier->Input->nb_objects_to_test,
				C->Canonical_form->get_allocated_length(),
				verbose_level - 2);
	}
	//int f_found;
	//int idx;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	Canonical_form_classifier->Output->CB->search_and_add_if_new(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			f_found_canonical_form,
			idx_canonical_form,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	// if f_found is true: idx_canonical_form is where the canonical form was found.
	// if f_found is false: idx_canonical_form is where the new canonical form was added.


	if (f_found_canonical_form) {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Qco->cnt
					<< " po = " << Qco->po
					<< " so = " << Qco->so
					<< " We found the canonical form at idx_canonical_form = " << idx_canonical_form << endl;
		}




		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"before handle_repeated_canonical_form_of_set" << endl;
		}
		handle_repeated_canonical_form_of_set(
				idx_canonical_form,
				C,
				alpha, gamma,
				idx_canonical_form,
				idx_equation,
				f_found_eqn,
				verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"after handle_repeated_canonical_form_of_set" << endl;
		}


	} // if f_found_canonical_form
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Qco->cnt
					<< " po = " << Qco->po
					<< " so = " << Qco->so
					<< " The canonical form is new" << endl;
		}

		// The canonical form has already been added,
		// at position idx_canonical_form,
		// with C as extra_data.
	}

	FREE_lint(alpha);
	FREE_int(gamma);


	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty done" << endl;
	}
}


void canonical_form_of_variety::handle_repeated_canonical_form_of_set(
		int idx,
		canonical_form_nauty *C,
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
		if (Canonical_form_classifier->Output->CB->compare_at(
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

int canonical_form_of_variety::find_equation(
		canonical_form_nauty *C,
		long int *alpha, int *gamma,
		int idx1, int &found_at,
		int verbose_level)
// relies on reverse_engineer_semilinear_map,
// which only works in a projective plane.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"verbose_level=" << verbose_level << endl;
	}

	long int *alpha_inv;
	long int *beta_inv;
	int i, j;
	int f_found = false;

	canonical_form_nauty *C1;
	C1 = (canonical_form_nauty *)
			Canonical_form_classifier->Output->CB->Type_extra_data[idx1];

	alpha_inv = C1->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"alpha_inv = " << endl;
		Lint_vec_print(cout,
				alpha_inv,
				Canonical_form_classifier->Output->canonical_labeling_len);
		cout << endl;
	}

	beta_inv = C->canonical_labeling;
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"beta_inv = " << endl;
		Lint_vec_print(
				cout,
				beta_inv,
				Canonical_form_classifier->Output->canonical_labeling_len);
		cout << endl;
	}

	// compute gamma = alpha * beta^-1 (left to right multiplication),
	// which maps the points on curve C1 to the points on curve C


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"computing alpha" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Output->canonical_labeling_len; i++) {
		j = alpha_inv[i];
		alpha[j] = i;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"computing gamma" << endl;
	}
	for (i = 0; i < Canonical_form_classifier->Output->canonical_labeling_len; i++) {
		gamma[i] = beta_inv[alpha[i]];
	}
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"gamma = " << endl;
		Int_vec_print(
				cout,
				gamma,
				Canonical_form_classifier->Output->canonical_labeling_len);
		cout << endl;
	}


	// gamma maps C1 to C.
	// So, in the contragredient action,
	// it maps the equation of C to an equation in the orbit of the equation C1,
	// which is what we want.

	// turn gamma into a matrix


	if (Canonical_form_classifier->PA->n != 2) {
		cout << "canonical_form_of_variety::find_equation "
				"need projective dimension to be 2." << endl;
		exit(1);
	}

	int Mtx[10];
		// We are in a plane, so we have 3 x 3 matrices,
		// possibly plus a field automorphism


	int frobenius;
	linear_algebra::linear_algebra_global LA;

	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"before LA.reverse_engineer_semilinear_map" << endl;
	}

	LA.reverse_engineer_semilinear_map(
			Canonical_form_classifier->PA->P->Subspaces->F,
			Canonical_form_classifier->PA->P->Subspaces->n,
			gamma, Mtx, frobenius,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"after LA.reverse_engineer_semilinear_map" << endl;
	}

	Mtx[9] = frobenius;

	Canonical_form_classifier->PA->A->Group_element->make_element(
			Canonical_form_classifier->Output->Elt, Mtx,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "The isomorphism from C to C1 is given by:" << endl;
		Canonical_form_classifier->PA->A->Group_element->element_print(
				Canonical_form_classifier->Output->Elt, cout);
	}



	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"before substitute_semilinear" << endl;
	}
	Canonical_form_classifier->Poly_ring->substitute_semilinear(
			Qco->Quartic_curve_object->eqn15 /* coeff_in */,
			Canonical_form_classifier->Output->eqn2 /* coeff_out */,
			Canonical_form_classifier->PA->A->is_semilinear_matrix_group(),
			frobenius, Mtx,
			0/*verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::find_equation "
				"after substitute_semilinear" << endl;
	}

	// now, eqn2 is the image of the curve C
	// and belongs to the orbit of equations associated with C1.

	Canonical_form_classifier->PA->F->Projective_space_basic->PG_element_normalize_from_front(
			Canonical_form_classifier->Output->eqn2, 1,
			Canonical_form_classifier->Poly_ring->get_nb_monomials());


	if (f_v) {
		cout << "The mapped equation is:";
		Canonical_form_classifier->Poly_ring->print_equation_simple(
				cout, Canonical_form_classifier->Output->eqn2);
		cout << endl;
	}




	int idx2;

	if (!C1->Orb->search_equation(
			Canonical_form_classifier->Output->eqn2 /*new_object */, idx2,
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
					"After C1->Orb->search_equation, cnt = " << Qco->cnt
					<< " po = " << Qco->po
					<< " so = " << Qco->so
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

void canonical_form_of_variety::add_object_and_compute_canonical_equation(
		canonical_form_nauty *C,
		int idx, int verbose_level)
// adds the canonical form at position idx
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation" << endl;
	}

#if 0
	quartic_curve_object *Qc2;

	Qc2 = NEW_OBJECT(quartic_curve_object);

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"before Qc2->init_image_of" << endl;
	}
	Qc2->init_image_of(Qco,
			Canonical_form_classifier->Elt,
			Canonical_form_classifier->Descr->PA->A,
			Canonical_form_classifier->Descr->PA->A_on_lines,
			Canonical_form_classifier->eqn2,
				verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"after Qc2->init_image_of" << endl;
	}

	FREE_OBJECT(Qco);
	Qco = Qc2;

	canonical_form_nauty *C2;
	ring_theory::longinteger_object go;


	C2 = NEW_OBJECT(canonical_form_nauty);

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"we will recompute the quartic curve "
				"from the canonical equation." << endl;
	}
	C2->init(Canonical_form_classifier, verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"before C2->canonical_form_of_quartic_curve" << endl;
	}
	C2->canonical_form_of_quartic_curve(
			this,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"after C2->canonical_form_of_quartic_curve" << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"before Canonical_form_classifier->CB->add_at_idx" << endl;
	}
	Canonical_form_classifier->CB->add_at_idx(
			C2->Canonical_form->get_data(),
			C2 /* void *extra_data */,
			idx,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"after Canonical_form_classifier->CB->add_at_idx" << endl;
	}
#else

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"before Canonical_form_classifier->CB->add_at_idx" << endl;
	}
	Canonical_form_classifier->Output->CB->add_at_idx(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			idx,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation "
				"after Canonical_form_classifier->CB->add_at_idx" << endl;
	}

#endif

	if (f_v) {
		cout << "canonical_form_of_variety::add_object_and_compute_canonical_equation done" << endl;
	}

}

void canonical_form_of_variety::compute_canonical_form_nauty(
		int counter,
		int &f_found_canonical_form,
		int &idx_canonical_form,
		int &idx_equation,
		int &f_found_eqn,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty "
				"verbose_level=" << verbose_level << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form "
				"before classify_curve_nauty" << endl;
	}

	classify_curve_nauty(
			f_found_canonical_form,
			idx_canonical_form,
			idx_equation,
			f_found_eqn,
			0 /*verbose_level - 2 */);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form "
				"after classify_curve_nauty" << endl;
		cout << "canonical_form_of_variety::compute_canonical_form "
				"f_found_canonical_form=" << f_found_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form "
				"idx_canonical_form=" << idx_canonical_form << endl;
		cout << "canonical_form_of_variety::compute_canonical_form "
				"idx_equation=" << idx_equation << endl;
		cout << "canonical_form_of_variety::compute_canonical_form "
				"f_found_eqn=" << f_found_eqn << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_nauty done" << endl;
	}
}


void canonical_form_of_variety::compute_canonical_form_substructure(
		int counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_substructure" << endl;
	}



	if (Qco->Quartic_curve_object->nb_pts >= Canonical_form_classifier->Descr->substructure_size) {

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
		Canonical_form_classifier->Output->CFS_table[counter] = CFS;
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"storing canonical_equation in "
					"Canonical_forms[counter], "
					"counter = " << counter << endl;
		}
		Int_vec_copy(canonical_equation,
				Canonical_form_classifier->Output->Canonical_equation +
				counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
				Canonical_form_classifier->Poly_ring->get_nb_monomials());
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"storing group order in Goi[]" << endl;
		}
		Canonical_form_classifier->Output->Goi[counter] = go_eqn->as_lint();
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form_substructure "
					"Goi[counter] = " << Canonical_form_classifier->Output->Goi[counter] << endl;
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

		Canonical_form_classifier->Output->CFS_table[counter] = NULL;
		Int_vec_zero(
				Canonical_form_classifier->Output->Canonical_equation +
				counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
				Canonical_form_classifier->Poly_ring->get_nb_monomials());
		Canonical_form_classifier->Output->Goi[counter] = -1;

	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form_substructure done" << endl;
	}
}


void canonical_form_of_variety::compute_canonical_object(
		int verbose_level)
// applies transporter_to_canonical_form to Qco to compute canonical_equation
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object" << endl;
	}


	Canonical_object = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_object_with_action);


	actions::action *A;
	actions::action *A_on_lines;

	A = Canonical_form_classifier->PA->A;
	A_on_lines = Canonical_form_classifier->PA->A_on_lines;


	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object "
				"before Qco_canonical->init_image_of" << endl;
	}
	Canonical_object->init_image_of(
			Qco,
			transporter_to_canonical_form,
			A,
			A_on_lines,
			canonical_equation,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object "
				"after Qco_canonical->init_image_of" << endl;
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_object done" << endl;
	}
}

void canonical_form_of_variety::prepare_csv_entry_one_line(
		std::vector<std::string> &v,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line" << endl;
	}

	v.push_back(std::to_string(Qco->cnt));
	v.push_back(std::to_string(Qco->po));
	v.push_back(std::to_string(Qco->so));
	v.push_back(std::to_string(Qco->po_go));
	v.push_back(std::to_string(Qco->po_index));
	v.push_back(std::to_string(Canonical_form_classifier->Output->Tally->rep_idx[i]));


	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Pts;
	string s_Bitangents;
	string s_transporter;
	string s_Eqn_canonical;
	string s_Pts_canonical;
	string s_Bitangents_canonical;
	std::string s_tl, s_gens, s_go;


	Qco->Quartic_curve_object->stringify(
			s_Eqn, s_Pts, s_Bitangents);

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
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
	v.push_back("\"" + s_transporter + "\"");
	v.push_back("\"" + s_Eqn_canonical + "\"");
	v.push_back("\"" + s_Pts_canonical + "\"");
	v.push_back("\"" + s_Bitangents_canonical + "\"");
	v.push_back("\"" + s_tl + "\"");
	v.push_back("\"" + s_gens + "\"");
	v.push_back(s_go);


	if (Canonical_form_classifier->Descr->carry_through.size()) {
		int j;

		for (j = 0; j < Canonical_form_classifier->Descr->carry_through.size(); j++) {
			v.push_back(Qco->Carrying_through[j]);
		}
	}

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line done" << endl;
	}


}

void canonical_form_of_variety::prepare_csv_entry_one_line_nauty(
		std::vector<std::string> &v, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty" << endl;
	}

	v.push_back(std::to_string(Qco->cnt));
	v.push_back(std::to_string(Qco->po));
	v.push_back(std::to_string(Qco->so));
	v.push_back(std::to_string(Qco->po_go));
	v.push_back(std::to_string(Qco->po_index));
	v.push_back(std::to_string(Canonical_form_classifier->Output->Iso_idx[i]));
	v.push_back(std::to_string(Canonical_form_classifier->Output->F_first_time[i]));
	v.push_back(std::to_string(Canonical_form_classifier->Output->Idx_canonical_form[i]));
	v.push_back(std::to_string(Canonical_form_classifier->Output->Idx_equation[i]));
	//v.push_back(std::to_string(Canonical_form_classifier->Output->Tally->rep_idx[i]));





	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " preparing string data" << endl;
	}
	string s_Eqn;
	string s_Pts;
	string s_Bitangents;
#if 0
	string s_transporter;
	string s_Eqn_canonical;
	string s_Pts_canonical;
	string s_Bitangents_canonical;
	std::string s_tl, s_gens, s_go;
#endif


	Qco->Quartic_curve_object->stringify(
			s_Eqn, s_Pts, s_Bitangents);

#if 0
	s_transporter = Canonical_form_classifier->PA->A->Group_element->stringify(
			transporter_to_canonical_form);

	s_Eqn_canonical = Int_vec_stringify(
			canonical_equation, 15);

	s_Pts_canonical = Canonical_object->stringify_Pts();

	s_Bitangents_canonical = Canonical_object->stringify_bitangents();

	if (gens_stab_of_canonical_equation) {

		if (false) {
			cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
					"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
					<< " before gens_stab_of_canonical_equation->stringify" << endl;
		}
		gens_stab_of_canonical_equation->stringify(s_tl, s_gens, s_go);
	}
	else {
		cout << "gens_stab_of_canonical_equation is not available" << endl;
	}
#endif

	if (f_v) {
		cout << "classification_of_varieties::prepare_csv_entry_one_line_nauty "
				"i=" << i << " / " << Canonical_form_classifier->Input->nb_objects_to_test
				<< " writing data" << endl;
	}

	v.push_back("\"" + s_Eqn + "\"");
	v.push_back("\"" + s_Pts + "\"");
	v.push_back("\"" + s_Bitangents + "\"");
#if 0
	v.push_back("\"" + s_transporter + "\"");
	v.push_back("\"" + s_Eqn_canonical + "\"");
	v.push_back("\"" + s_Pts_canonical + "\"");
	v.push_back("\"" + s_Bitangents_canonical + "\"");
	v.push_back("\"" + s_tl + "\"");
	v.push_back("\"" + s_gens + "\"");
	v.push_back(s_go);
#endif

	if (Canonical_form_classifier->Descr->carry_through.size()) {
		int j;

		for (j = 0; j < Canonical_form_classifier->Descr->carry_through.size(); j++) {
			v.push_back(Qco->Carrying_through[j]);
		}
	}

	if (f_v) {
		cout << "canonical_form_of_variety::prepare_csv_entry_one_line_nauty done" << endl;
	}


}


}}}


