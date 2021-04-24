/*
 * canonical_form_classifier.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



canonical_form_classifier::canonical_form_classifier()
{
	Descr = NULL;
	Poly_ring = NULL;
	AonHPD = NULL;
	nb_objects_to_test = 0;
	CB = NULL;
}

canonical_form_classifier::~canonical_form_classifier()
{

}

void canonical_form_classifier::count_nb_objects_to_test(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;

	if (f_v) {
		cout << "canonical_form_classifier::count_nb_objects_to_test" << endl;
	}

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < Descr->nb_files; cnt++) {

		char str[1000];
		string fname;

		sprintf(str, Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		nb_objects_to_test += S.nb_rows - 1;

		if (f_v) {
			cout << "canonical_form_classifier::count_nb_objects_to_test "
					"file " << cnt << " / " << Descr->nb_files << " has  "
					<< S.nb_rows - 1 << " objects" << endl;
		}
	}

	if (f_v) {
		cout << "canonical_form_classifier::count_nb_objects_to_test "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}
}


void canonical_form_classifier::classify(canonical_form_classifier_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify" << endl;
	}

	canonical_form_classifier::Descr = Descr;

	int cnt;

	if (!Descr->f_degree) {
		cout << "canonical_form_classifier::classify please use -degree <d>  to specify the degree" << endl;
		exit(1);
	}

	Poly_ring = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly_ring->init(Descr->PA->F, Descr->PA->n + 1, Descr->degree, FALSE, t_PART, verbose_level);


	AonHPD = NEW_OBJECT(action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(Descr->PA->A, Poly_ring, verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after AonHPD->init" << endl;
	}




	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before count_nb_objects_to_test" << endl;
	}

	count_nb_objects_to_test(verbose_level);


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}


	number_theory_domain NT;

	CB = NEW_OBJECT(classify_bitvectors);

	int canonical_labeling_len;
	long int *alpha;
	int *gamma;

	int *Elt;
	int *eqn2;

	Elt = NEW_int(Descr->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Poly_ring->get_nb_monomials());


	for (cnt = 0; cnt < Descr->nb_files; cnt++) {
		char str[1000];
		string fname;
		int row;

		sprintf(str, Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "canonical_form_classifier::classify S.nb_rows = " << S.nb_rows << endl;
			cout << "canonical_form_classifier::classify S.nb_cols = " << S.nb_cols << endl;
		}

		int j, t;
		string eqn_txt;
		string pts_txt;
		string bitangents_txt;
		int *eqn;
		int sz;
		long int *pts;
		int nb_pts;
		long int *bitangents;
		int nb_bitangents;



		for (row = 0; row < S.nb_rows - 1; row++) {

			if (f_v) {
				cout << "cnt = " << cnt << " / " << Descr->nb_files << " row = " << row << " / " << S.nb_rows - 1 << endl;
			}

			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
			j = 2;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			ST.remove_specific_character(bitangents_txt, '\"');

			if (f_v) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}

			Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			if (f_v) {
				cout << "row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				cout << " bitangents=";
				Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}


			canonical_form *C;
			longinteger_object go;


			C = NEW_OBJECT(canonical_form);

			C->quartic_curve(
					Descr->PA,
					Poly_ring,
					AonHPD,
					row, eqn, sz,
					pts, nb_pts,
					bitangents, nb_bitangents,
					verbose_level);

			C->Stab_gens_quartic->group_order(go);

			canonical_labeling_len = C->canonical_labeling_len;
			alpha = NEW_lint(canonical_labeling_len);
			gamma = NEW_int(canonical_labeling_len);


			if (CB->n == 0) {
				CB->init(nb_objects_to_test,
						C->Canonical_form->get_allocated_length(),
						verbose_level);
			}
			int f_found;
			int idx;

			CB->search_and_add_if_new(C->Canonical_form->get_data(), C /* void *extra_data */, f_found, idx, verbose_level);


			if (!f_found) {
				if (f_v) {
					cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " The canonical form is new" << endl;
				}
			}
			else {
				if (f_v) {
					cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " We found the canonical form at idx = " << idx << endl;
				}




				long int *alpha_inv;
				long int *beta_inv;
				int i;

				//long int *canonical_labeling;




				int idx1;
				int found_at = -1;

				if (f_v) {
					cout << "starting loop over idx1" << endl;
				}

				for (idx1 = idx; idx1 >= 0; idx1--) {



					// test if entry at idx1 is equal to C.
					// if not, break

					if (f_v) {
						cout << "canonical_form_classifier::classify before CB->compare_at idx1 = " << idx1 << endl;
					}
					if (CB->compare_at(C->Canonical_form->get_data(), idx1) != 0) {
						if (f_v) {
							cout << "canonical_form_classifier::classify at idx1 = " << idx1 << " is not equal, break" << endl;
						}
						break;
					}
					if (f_v) {
						cout << "canonical_form_classifier::classify canonical form at " << idx1 << " is equal" << endl;
					}


					canonical_form *C1;
					C1 = (canonical_form *) CB->Type_extra_data[idx1];

					alpha_inv = C1->canonical_labeling;

					beta_inv = C->canonical_labeling;

					// compute gamma = beta * alpha^-1


					if (f_v) {
						cout << "canonical_form_classifier::classify computing alpha" << endl;
					}
					for (i = 0; i < canonical_labeling_len; i++) {
						j = alpha_inv[i];
						alpha[j] = i;
					}

					if (f_v) {
						cout << "canonical_form_classifier::classify computing gamma" << endl;
					}
					for (i = 0; i < canonical_labeling_len; i++) {
						gamma[i] = beta_inv[alpha[i]];
					}


					// gamma maps C1 to C.
					// So, in the contragredient action, it maps the equation of C to the equation of C1,
					// which is what we want.

					// turn gamma into a matrix


					int Mtx[10];
					//int Mtx_inv[10];
					int frobenius;

					if (f_v) {
						cout << "canonical_form_classifier::classify before PA->P->reverse_engineer_semilinear_map" << endl;
					}
					Descr->PA->P->reverse_engineer_semilinear_map(
						gamma, Mtx, frobenius,
						0 /*verbose_level*/);
					if (f_v) {
						cout << "canonical_form_classifier::classify after PA->P->reverse_engineer_semilinear_map" << endl;
					}

					Mtx[9] = frobenius;

					Descr->PA->A->make_element(Elt, Mtx, 0 /* verbose_level*/);

					if (f_v) {
						cout << "The isomorphism from C to C1 is given by:" << endl;
						Descr->PA->A->element_print(Elt, cout);
					}



					//int frobenius_inv;

					//frobenius_inv = NT.int_negate(Mtx[3 * 3], PA->F->e);


					//PA->F->matrix_inverse(Mtx, Mtx_inv, 3, 0 /* verbose_level*/);

					if (f_v) {
						cout << "canonical_form_classifier::classify before substitute_semilinear" << endl;
					}
					Poly_ring->substitute_semilinear(C->eqn /* coeff_in */, eqn2 /* coeff_out */,
							Descr->PA->A->is_semilinear_matrix_group(), frobenius, Mtx, 0/*verbose_level*/);
					if (f_v) {
						cout << "canonical_form_classifier::classify after substitute_semilinear" << endl;
					}

					Descr->PA->F->PG_element_normalize_from_front(eqn2, 1, Poly_ring->get_nb_monomials());


					if (f_v) {
						cout << "The mapped equation is:";
						Poly_ring->print_equation_simple(cout, eqn2);
						cout << endl;
					}




					int idx2;

					if (!C1->Orb->search_equation(eqn2 /*new_object */, idx2, TRUE)) {
						// need to map points and bitangents under gamma:
						if (f_v) {
							cout << "we found the canonical form but we did not find the equation at idx1=" << idx1 << endl;
						}


					}
					else {
						if (f_v) {
							cout << "After search_and_add_if_new, cnt = " << cnt << " row = " << row << " We found the canonical form and the equation at idx2 " << idx2 << ", idx1=" << idx1 << endl;
						}
						found_at = idx1;
						break;
					}


				}


				if (found_at == -1) {

					if (f_v) {
						cout << "we found the canonical form but we did not find the equation" << endl;
					}

					long int *pts2;
					//int nb_pts;
					long int *bitangents2;
					//int nb_bitangents;
					int i;

					pts2 = NEW_lint(nb_pts);
					bitangents2 = NEW_lint(nb_bitangents);

					for (i = 0; i < nb_pts; i++) {
						pts2[i] = Descr->PA->A->element_image_of(pts[i], Elt, 0 /* verbose_level */);
					}
					for (i = 0; i < nb_bitangents; i++) {
						bitangents2[i] = Descr->PA->A_on_lines->element_image_of(bitangents[i], Elt, 0 /* verbose_level */);
					}

					canonical_form *C2;
					longinteger_object go;


					C2 = NEW_OBJECT(canonical_form);

					if (f_v) {
						cout << "we recompute the quartic curve from the canonical equation." << endl;
					}
					if (f_v) {
						cout << "canonical_form_classifier::classify before C2->quartic_curve" << endl;
					}
					C2->quartic_curve(
							Descr->PA,
							Poly_ring,
							AonHPD,
							row, eqn2, sz,
							pts2, nb_pts,
							bitangents2, nb_bitangents,
							verbose_level);
					if (f_v) {
						cout << "canonical_form_classifier::classify after C2->quartic_curve" << endl;
					}

					if (f_v) {
						cout << "After search_and_add_if_new, adding at " << idx << endl;
					}
					CB->add_at_idx(C2->Canonical_form->get_data(), C2 /* void *extra_data */, idx, 0 /* verbose_level*/);


				} // if (found_at == -1)
				else {
					if (f_v) {
						cout << "we found the equation at index " << found_at << endl;
					}

				}

			} // if f_found

			FREE_lint(alpha);
			FREE_int(gamma);

		} // next row


	} // next cnt

	if (f_v) {
		cout << "canonical_form_classifier::classify The number of isomorphism types is " << CB->nb_types << endl;
	}

	FREE_int(eqn2);
	FREE_int(Elt);


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}



}}

