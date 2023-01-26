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
}


void canonical_form_of_variety::init(
		canonical_form_classifier *Canonical_form_classifier,
		std::string &fname_case_out,
		quartic_curve_object *Qco,
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
			NEW_int(Canonical_form_classifier->Descr->PA->A->elt_size_in_int);

	//long int *canonical_pts;

	canonical_pts = NEW_lint(Qco->nb_pts);

	go_eqn = NEW_OBJECT(ring_theory::longinteger_object);

	if (f_v) {
		cout << "canonical_form_of_variety::init done" << endl;
	}
}


void canonical_form_of_variety::classify_curve_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"verbose_level=" << verbose_level << endl;
	}

	canonical_form_nauty *C;
	ring_theory::longinteger_object go;

	groups::strong_generators *gens_stab_of_canonical_equation;

	long int *alpha;
	int *gamma;





	C = NEW_OBJECT(canonical_form_nauty);

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before C->init" << endl;
	}
	C->init(Canonical_form_classifier, verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after C->init" << endl;
	}


	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before C->canonical_form_of_quartic_curve" << endl;
	}
	C->canonical_form_of_quartic_curve(
			Qco,
			canonical_equation,
			transporter_to_canonical_form,
			gens_stab_of_canonical_equation,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after C->canonical_form_of_quartic_curve" << endl;
	}

	C->Stab_gens_quartic->group_order(go);

	FREE_OBJECT(gens_stab_of_canonical_equation);

	Canonical_form_classifier->canonical_labeling_len = C->canonical_labeling_len;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"canonical_labeling_len=" <<
				Canonical_form_classifier->canonical_labeling_len << endl;
	}
	if (Canonical_form_classifier->canonical_labeling_len == 0) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"canonical_labeling_len == 0, error" << endl;
		exit(1);
	}
	alpha = NEW_lint(Canonical_form_classifier->canonical_labeling_len);
	gamma = NEW_int(Canonical_form_classifier->canonical_labeling_len);


	if (Canonical_form_classifier->CB->n == 0) {
		Canonical_form_classifier->CB->init(
				Canonical_form_classifier->nb_objects_to_test,
				C->Canonical_form->get_allocated_length(),
				verbose_level);
	}
	int f_found;
	int idx;

	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"before Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}
	Canonical_form_classifier->CB->search_and_add_if_new(
			C->Canonical_form->get_data(),
			C /* void *extra_data */,
			f_found,
			idx,
			verbose_level - 2);
	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty "
				"after Canonical_form_classifier->CB->search_and_add_if_new" << endl;
	}


	if (!f_found) {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Qco->cnt
					<< " po = " << Qco->po
					<< " so = " << Qco->so
					<< " The canonical form is new" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"After search_and_add_if_new, "
					"cnt = " << Qco->cnt
					<< " po = " << Qco->po
					<< " so = " << Qco->so
					<< " We found the canonical form at idx = " << idx << endl;
		}

		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"before handle_repeated_object" << endl;
		}
		handle_repeated_object(
				idx,
				C,
				alpha, gamma,
				verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::classify_curve_nauty "
					"after handle_repeated_object" << endl;
		}


	} // if f_found

	FREE_lint(alpha);
	FREE_int(gamma);


	if (f_v) {
		cout << "canonical_form_of_variety::classify_curve_nauty done" << endl;
	}
}


void canonical_form_of_variety::handle_repeated_object(
		int idx,
		canonical_form_nauty *C,
		long int *alpha, int *gamma,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_object "
				"verbose_level=" << verbose_level << endl;
	}


	long int *alpha_inv;
	long int *beta_inv;
	int i, j;

	//long int *canonical_labeling;




	int idx1;
	int found_at = -1;

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_object "
				"starting loop over idx1" << endl;
	}

	for (idx1 = idx; idx1 >= 0; idx1--) {



		// test if entry at idx1 is equal to C.
		// if not, break

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before Canonical_form_classifier->CB->compare_at idx1 = " << idx1 << endl;
		}
		if (Canonical_form_classifier->CB->compare_at(
				C->Canonical_form->get_data(), idx1) != 0) {
			if (f_v) {
				cout << "canonical_form_of_variety::handle_repeated_object "
						"at idx1 = " << idx1 << " is not equal, break" << endl;
			}
			break;
		}
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"canonical form at " << idx1 << " is equal" << endl;
		}


		canonical_form_nauty *C1;
		C1 = (canonical_form_nauty *) Canonical_form_classifier->CB->Type_extra_data[idx1];

		alpha_inv = C1->canonical_labeling;
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"alpha_inv = " << endl;
			Lint_vec_print(cout, alpha_inv, Canonical_form_classifier->canonical_labeling_len);
			cout << endl;
		}

		beta_inv = C->canonical_labeling;
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"beta_inv = " << endl;
			Lint_vec_print(cout, beta_inv, Canonical_form_classifier->canonical_labeling_len);
			cout << endl;
		}

		// compute gamma = beta * alpha^-1


		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"computing alpha" << endl;
		}
		for (i = 0; i < Canonical_form_classifier->canonical_labeling_len; i++) {
			j = alpha_inv[i];
			alpha[j] = i;
		}

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"computing gamma" << endl;
		}
		for (i = 0; i < Canonical_form_classifier->canonical_labeling_len; i++) {
			gamma[i] = beta_inv[alpha[i]];
		}
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"gamma = " << endl;
			Int_vec_print(cout, gamma, Canonical_form_classifier->canonical_labeling_len);
			cout << endl;
		}


		// gamma maps C1 to C.
		// So, in the contragredient action,
		// it maps the equation of C to the equation of C1,
		// which is what we want.

		// turn gamma into a matrix


		int Mtx[10];
			// We are in a plane, so we have 3 x 3 matrices,
			// possibly plus a field automorphism


		//int Mtx_inv[10];
		int frobenius;

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before PA->P->reverse_engineer_semilinear_map" << endl;
		}
		Canonical_form_classifier->Descr->PA->P->reverse_engineer_semilinear_map(
				gamma, Mtx, frobenius,
			verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"after PA->P->reverse_engineer_semilinear_map" << endl;
		}

		Mtx[9] = frobenius;

		Canonical_form_classifier->Descr->PA->A->make_element(
				Canonical_form_classifier->Elt, Mtx, 0 /* verbose_level*/);

		if (f_v) {
			cout << "The isomorphism from C to C1 is given by:" << endl;
			Canonical_form_classifier->Descr->PA->A->element_print(
					Canonical_form_classifier->Elt, cout);
		}



		//int frobenius_inv;

		//frobenius_inv = NT.int_negate(Mtx[3 * 3], PA->F->e);


		//PA->F->matrix_inverse(Mtx, Mtx_inv, 3, 0 /* verbose_level*/);

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before substitute_semilinear" << endl;
		}
		Canonical_form_classifier->Poly_ring->substitute_semilinear(
				C->Qco->eqn /* coeff_in */,
				Canonical_form_classifier->eqn2 /* coeff_out */,
				Canonical_form_classifier->Descr->PA->A->is_semilinear_matrix_group(),
				frobenius, Mtx,
				0/*verbose_level*/);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"after substitute_semilinear" << endl;
		}

		Canonical_form_classifier->Descr->PA->F->PG_element_normalize_from_front(
				Canonical_form_classifier->eqn2, 1,
				Canonical_form_classifier->Poly_ring->get_nb_monomials());


		if (f_v) {
			cout << "The mapped equation is:";
			Canonical_form_classifier->Poly_ring->print_equation_simple(
					cout, Canonical_form_classifier->eqn2);
			cout << endl;
		}




		int idx2;

		if (!C1->Orb->search_equation(
				Canonical_form_classifier->eqn2 /*new_object */, idx2,
				TRUE)) {
			// need to map points and bitangents under gamma:
			if (f_v) {
				cout << "canonical_form_of_variety::handle_repeated_object "
						"we found the canonical form but we did not find "
						"the equation at idx1=" << idx1 << endl;
			}


		}
		else {
			if (f_v) {
				cout << "canonical_form_of_variety::handle_repeated_object "
						"After C1->Orb->search_equation, cnt = " << Qco->cnt
						<< " po = " << Qco->po
						<< " so = " << Qco->so
						<< " We found the canonical form and the equation "
								"at idx2 " << idx2 << ", idx1=" << idx1 << endl;
			}
			found_at = idx1;
			break;
		}


	}


	if (found_at == -1) {

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"we found the canonical form but we did "
					"not find the equation" << endl;
		}


		quartic_curve_object *Qc2;

		Qc2 = NEW_OBJECT(quartic_curve_object);

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before Qc2->init_image_of" << endl;
		}
		Qc2->init_image_of(Qco,
				Canonical_form_classifier->Elt,
				Canonical_form_classifier->Descr->PA->A,
				Canonical_form_classifier->Descr->PA->A_on_lines,
				Canonical_form_classifier->eqn2,
					verbose_level - 2);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"after Qc2->init_image_of" << endl;
		}

		canonical_form_nauty *C2;
		ring_theory::longinteger_object go;


		C2 = NEW_OBJECT(canonical_form_nauty);

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"we will recompute the quartic curve "
					"from the canonical equation." << endl;
		}
		C2->init(Canonical_form_classifier, verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before C2->canonical_form_of_quartic_curve" << endl;
		}
		C2->canonical_form_of_quartic_curve(
				Qc2,
				canonical_equation,
				transporter_to_canonical_form,
				gens_stab_of_canonical_equation,
				verbose_level - 2);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"after C2->canonical_form_of_quartic_curve" << endl;
		}

		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"before Canonical_form_classifier->CB->add_at_idx" << endl;
		}
		Canonical_form_classifier->CB->add_at_idx(
				C2->Canonical_form->get_data(),
				C2 /* void *extra_data */,
				idx,
				0 /* verbose_level*/);
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"after Canonical_form_classifier->CB->add_at_idx" << endl;
		}


	} // if (found_at == -1)
	else {
		if (f_v) {
			cout << "canonical_form_of_variety::handle_repeated_object "
					"we found the equation at index " << found_at << endl;
		}
	}

	if (f_v) {
		cout << "canonical_form_of_variety::handle_repeated_object done" << endl;
	}

}



void canonical_form_of_variety::compute_canonical_form(
		int counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form "
				"verbose_level=" << verbose_level << endl;
	}

	if (Canonical_form_classifier->Descr->f_algorithm_nauty) {
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"f_algorithm_nauty" << endl;
		}


		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"before classify_curve_nauty" << endl;
		}
		classify_curve_nauty(verbose_level);
		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"after classify_curve_nauty" << endl;
		}

		Int_vec_copy(
				canonical_equation,
				Canonical_form_classifier->Canonical_forms + counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
				Canonical_form_classifier->Poly_ring->get_nb_monomials());


		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"f_algorithm_nauty done" << endl;
		}
	}
	else if (Canonical_form_classifier->Descr->f_algorithm_substructure) {


		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"f_algorithm_substructure" << endl;
		}



		if (Qco->nb_pts >= Canonical_form_classifier->Descr->substructure_size) {

			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"nb_pts is sufficient" << endl;
			}

			//ring_theory::longinteger_object go_eqn;




			canonical_form_substructure *CFS;

			CFS = NEW_OBJECT(canonical_form_substructure);

			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"before CFS->classify_curve_with_substructure" << endl;
			}

			CFS->classify_curve_with_substructure(
					this,
					verbose_level);

			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"after CFS->classify_curve_with_substructure" << endl;
			}

			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"storing CFS in CFS_table[counter], "
						"counter = " << counter << endl;
			}
			Canonical_form_classifier->CFS_table[counter] = CFS;
			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"storing canonical_equation in "
						"Canonical_forms[counter], "
						"counter = " << counter << endl;
			}
			Int_vec_copy(canonical_equation,
					Canonical_form_classifier->Canonical_forms + counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
					Canonical_form_classifier->Poly_ring->get_nb_monomials());
			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"storing group order in Goi[]" << endl;
			}
			Canonical_form_classifier->Goi[counter] = go_eqn->as_lint();
			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"Goi[counter] = " << Canonical_form_classifier->Goi[counter] << endl;
			}

			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"after CFS->classify_curve_with_substructure" << endl;
			}
		}
		else {


			if (f_v) {
				cout << "canonical_form_of_variety::compute_canonical_form "
						"too small for substructure algorithm. "
						"Skipping" << endl;
			}

			Canonical_form_classifier->CFS_table[counter] = NULL;
			Int_vec_zero(
					Canonical_form_classifier->Canonical_forms + counter * Canonical_form_classifier->Poly_ring->get_nb_monomials(),
					Canonical_form_classifier->Poly_ring->get_nb_monomials());
			Canonical_form_classifier->Goi[counter] = -1;

		}

		if (f_v) {
			cout << "canonical_form_of_variety::compute_canonical_form "
					"f_algorithm_substructure done" << endl;
		}

	}
	else {
		cout << "canonical_form_of_variety::compute_canonical_form "
				"please select which algorithm to use" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "canonical_form_of_variety::compute_canonical_form done" << endl;
	}

}



}}}


