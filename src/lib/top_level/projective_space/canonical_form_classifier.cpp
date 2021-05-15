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
	canonical_labeling_len = 0;
	alpha = NULL;
	gamma = NULL;

	PC = NULL;
	Control = NULL;
	Poset = NULL;
	nb_orbits = 0;


	Elt = NULL;
	eqn2 = NULL;

	canonical_equation = NULL;
	transporter_to_canonical_form = NULL;
	//longinteger_object go_eqn;

	CFS_table = NULL;
	counter = 0;
	Canonical_forms = NULL;
	Goi = NULL;

	Classification_of_quartic_curves = NULL;

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

	Elt = NEW_int(Descr->PA->A->elt_size_in_int);
	transporter_to_canonical_form = NEW_int(Descr->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Poly_ring->get_nb_monomials());
	canonical_equation = NEW_int(Poly_ring->get_nb_monomials());



	Canonical_forms = NEW_int(nb_objects_to_test * Poly_ring->get_nb_monomials());
	Goi = NEW_lint(nb_objects_to_test);

	if (Descr->f_algorithm_nauty) {
		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"before classify_nauty" << endl;
		}
		classify_nauty(verbose_level);
		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"after classify_nauty" << endl;
		}
	}
	else if (Descr->f_algorithm_substructure) {

		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"before classify_with_substructure" << endl;
		}
		classify_with_substructure(verbose_level);
		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"after classify_with_substructure" << endl;
		}
	}
	else {
		cout << "canonical_form_classifier::classify please select which algorithm to use" << endl;
		exit(1);
	}

	//FREE_int(eqn2);
	//FREE_int(Elt);
	//FREE_int(canonical_equation);
	//FREE_int(transporter_to_canonical_form);


	int i, j;

	cout << "canonical forms:" << endl;
	for (i = 0; i < nb_objects_to_test; i++) {
		cout << setw(2) << i << " : ";
		Orbiter->Int_vec.print(cout, Canonical_forms + i * Poly_ring->get_nb_monomials(), Poly_ring->get_nb_monomials());
		cout << " : " << Goi[i] << endl;
	}

	Classification_of_quartic_curves = NEW_OBJECT(tally_vector_data);

	Classification_of_quartic_curves->init(Canonical_forms, nb_objects_to_test, Poly_ring->get_nb_monomials(), verbose_level);

	cout << "Classification of curves:" << endl;
	//Classification_of_quartic_curves->print();

	for (i = 0; i < Classification_of_quartic_curves->nb_types; i++) {

		//h = int_vec_hash(Reps + i * data_set_sz, data_set_sz);

		cout << i << " : " << Classification_of_quartic_curves->Frequency[i] << " x ";
		Orbiter->Int_vec.print(cout,
				Classification_of_quartic_curves->Reps + i * Classification_of_quartic_curves->data_set_sz,
				Classification_of_quartic_curves->data_set_sz);
		cout << " : ";
		j = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[i]];
		cout << Goi[j] << " : ";
		Orbiter->Int_vec.print(cout,
				Classification_of_quartic_curves->sorting_perm_inv + Classification_of_quartic_curves->type_first[i],
				Classification_of_quartic_curves->Frequency[i]);
		cout << endl;
#if 0
		cout << "for elements ";
		int_vec_print(cout, sorting_perm_inv + type_first[i], Frequency[i]);
		cout << endl;
#endif
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify done" << endl;
	}
}


void canonical_form_classifier::classify_nauty(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify_nauty" << endl;
	}


	CB = NEW_OBJECT(classify_bitvectors);




	main_loop(verbose_level);

	if (f_v) {
		cout << "canonical_form_classifier::classify_nauty The number of isomorphism types is " << CB->nb_types << endl;
	}


}

void canonical_form_classifier::classify_with_substructure(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure" << endl;
	}



	int j;

	Poset = NEW_OBJECT(poset);


	Control = NEW_OBJECT(poset_classification_control);

	Control->f_depth = TRUE;
	Control->depth = Descr->substructure_size;


	if (f_v) {
		cout << "projective_space_activity::set_stabilizer control=" << endl;
		Control->print();
	}


	Poset->init_subset_lattice(Descr->PA->A, Descr->PA->A,
			Descr->PA->A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "projective_space_activity::set_stabilizer "
				"before Poset->orbits_on_k_sets_compute" << endl;
	}
	PC = Poset->orbits_on_k_sets_compute(
			Control,
			Descr->substructure_size,
			verbose_level);
	if (f_v) {
		cout << "projective_space_activity::set_stabilizer "
				"after Poset->orbits_on_k_sets_compute" << endl;
	}

	nb_orbits = PC->nb_orbits_at_level(Descr->substructure_size);

	cout << "We found " << nb_orbits << " orbits at level " << Descr->substructure_size << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {


		strong_generators *Strong_gens;

		PC->get_stabilizer_generators(
				Strong_gens,
				Descr->substructure_size, j, 0 /* verbose_level*/);

		longinteger_object go;

		Strong_gens->group_order(go);

		FREE_OBJECT(Strong_gens);

		cout << j << " : " << go << endl;


	}


	CFS_table = (canonical_form_substructure **) NEW_pvoid(nb_objects_to_test);




	main_loop(verbose_level);



	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure done" << endl;
	}

}


void canonical_form_classifier::main_loop(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;



	if (f_v) {
		cout << "canonical_form_classifier::main_loop" << endl;
	}




	counter = 0;

	for (cnt = 0; cnt < Descr->nb_files; cnt++) {
		char str[1000];
		string fname;
		int row;

		sprintf(str, Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		spreadsheet S;

		S.read_spreadsheet(fname, verbose_level);

		if (f_v) {
			cout << "canonical_form_classifier::classify_nauty S.nb_rows = " << S.nb_rows << endl;
			cout << "canonical_form_classifier::classify_nauty S.nb_cols = " << S.nb_cols << endl;
		}




		for (row = 0; row < S.nb_rows - 1; row++, counter++) {

			if (f_v) {
				cout << "cnt = " << cnt << " / " << Descr->nb_files << " row = " << row << " / " << S.nb_rows - 1 << endl;
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

			j = 1;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify_nauty token[t] == NULL" << endl;
			}
			eqn_txt.assign(S.tokens[t]);
			j = 2;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify_nauty token[t] == NULL" << endl;
			}
			pts_txt.assign(S.tokens[t]);
			j = 3;
			t = S.Table[(row + 1) * S.nb_cols + j];
			if (S.tokens[t] == NULL) {
				cout << "canonical_form_classifier::classify_nauty token[t] == NULL" << endl;
			}
			bitangents_txt.assign(S.tokens[t]);

			string_tools ST;

			ST.remove_specific_character(eqn_txt, '\"');
			ST.remove_specific_character(pts_txt, '\"');
			ST.remove_specific_character(bitangents_txt, '\"');

			if (FALSE) {
				cout << "row = " << row << " eqn=" << eqn_txt << " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
			}

			Orbiter->Int_vec.scan(eqn_txt, eqn, sz);
			Orbiter->Lint_vec.scan(pts_txt, pts, nb_pts);
			Orbiter->Lint_vec.scan(bitangents_txt, bitangents, nb_bitangents);

			if (FALSE) {
				cout << "row = " << row << " eqn=";
				Orbiter->Int_vec.print(cout, eqn, sz);
				cout << " pts=";
				Orbiter->Lint_vec.print(cout, pts, nb_pts);
				cout << " bitangents=";
				Orbiter->Lint_vec.print(cout, bitangents, nb_bitangents);
				cout << endl;
			}

			if (Descr->f_algorithm_nauty) {
				if (f_v) {
					cout << "canonical_form_classifier::main_loop "
							"before classify_curve_nauty" << endl;
				}
				classify_curve_nauty(cnt, row,
						eqn, sz, pts, nb_pts, bitangents, nb_bitangents,
						verbose_level);
				if (f_v) {
					cout << "canonical_form_classifier::main_loop "
							"after classify_curve_nauty" << endl;
				}
			}
			else if (Descr->f_algorithm_substructure) {





				if (nb_pts >= Descr->substructure_size) {

					if (f_v) {
						cout << "canonical_form_classifier::main_loop "
								"before CFS->classify_curve_with_substructure" << endl;
					}

					longinteger_object go_eqn;

					canonical_form_substructure *CFS;

					CFS = NEW_OBJECT(canonical_form_substructure);

					CFS->classify_curve_with_substructure(
							this,
							counter, cnt, row,
							eqn,
							sz,
							pts,
							nb_pts,
							bitangents,
							nb_bitangents,
							canonical_equation,
							transporter_to_canonical_form,
							go_eqn,
							verbose_level);

					CFS_table[counter] = CFS;
					Orbiter->Int_vec.copy(CFS->canonical_equation,
							Canonical_forms + counter * Poly_ring->get_nb_monomials(),
							Poly_ring->get_nb_monomials());
					Goi[counter] = go_eqn.as_lint();

					if (f_v) {
						cout << "canonical_form_classifier::main_loop "
								"after CFS->classify_curve_with_substructure" << endl;
					}
				}
				else {


					if (f_v) {
						cout << "canonical_form_classifier::main_loop "
								"too small for substructure algorithm. Skipping" << endl;
					}

					CFS_table[counter] = NULL;
					Orbiter->Int_vec.zero(
							Canonical_forms + counter * Poly_ring->get_nb_monomials(),
							Poly_ring->get_nb_monomials());
					Goi[counter] = -1;

				}
			}
			else {
				cout << "canonical_form_classifier::main_loop please select which algorithm to use" << endl;
				exit(1);
			}

#if 0
			FREE_int(eqn);
			FREE_lint(pts);
			FREE_lint(bitangents);
#endif

		} // next row


	} // next cnt

	if (f_v) {
		cout << "canonical_form_classifier::main_loop done" << endl;
	}
}


void canonical_form_classifier::classify_curve_nauty(int cnt, int row,
		int *eqn,
		int sz,
		long int *pts,
		int nb_pts,
		long int *bitangents,
		int nb_bitangents,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_classifier::classify_curve_nauty" << endl;
	}

	canonical_form_nauty *C;
	longinteger_object go;

	strong_generators *gens_stab_of_canonical_equation;






	C = NEW_OBJECT(canonical_form_nauty);

	C->quartic_curve(
			Descr->PA,
			Poly_ring,
			AonHPD,
			row, eqn, sz,
			pts, nb_pts,
			bitangents, nb_bitangents,
			canonical_equation,
			transporter_to_canonical_form,
			gens_stab_of_canonical_equation,
			verbose_level);

	C->Stab_gens_quartic->group_order(go);

	FREE_OBJECT(gens_stab_of_canonical_equation);

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
		int i, j;

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
				cout << "canonical_form_classifier::classify_curve_nauty before CB->compare_at idx1 = " << idx1 << endl;
			}
			if (CB->compare_at(C->Canonical_form->get_data(), idx1) != 0) {
				if (f_v) {
					cout << "canonical_form_classifier::classify_curve_nauty at idx1 = " << idx1 << " is not equal, break" << endl;
				}
				break;
			}
			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty canonical form at " << idx1 << " is equal" << endl;
			}


			canonical_form_nauty *C1;
			C1 = (canonical_form_nauty *) CB->Type_extra_data[idx1];

			alpha_inv = C1->canonical_labeling;

			beta_inv = C->canonical_labeling;

			// compute gamma = beta * alpha^-1


			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty computing alpha" << endl;
			}
			for (i = 0; i < canonical_labeling_len; i++) {
				j = alpha_inv[i];
				alpha[j] = i;
			}

			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty computing gamma" << endl;
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
				cout << "canonical_form_classifier::classify_curve_nauty before PA->P->reverse_engineer_semilinear_map" << endl;
			}
			Descr->PA->P->reverse_engineer_semilinear_map(
				gamma, Mtx, frobenius,
				0 /*verbose_level*/);
			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty after PA->P->reverse_engineer_semilinear_map" << endl;
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
				cout << "canonical_form_classifier::classify_curve_nauty before substitute_semilinear" << endl;
			}
			Poly_ring->substitute_semilinear(C->eqn /* coeff_in */, eqn2 /* coeff_out */,
					Descr->PA->A->is_semilinear_matrix_group(), frobenius, Mtx, 0/*verbose_level*/);
			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty after substitute_semilinear" << endl;
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

			canonical_form_nauty *C2;
			longinteger_object go;


			C2 = NEW_OBJECT(canonical_form_nauty);

			if (f_v) {
				cout << "we recompute the quartic curve from the canonical equation." << endl;
			}
			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty before C2->quartic_curve" << endl;
			}
			C2->quartic_curve(
					Descr->PA,
					Poly_ring,
					AonHPD,
					row, eqn2, sz,
					pts2, nb_pts,
					bitangents2, nb_bitangents,
					canonical_equation,
					transporter_to_canonical_form,
					gens_stab_of_canonical_equation,
					verbose_level);
			if (f_v) {
				cout << "canonical_form_classifier::classify_curve_nauty after C2->quartic_curve" << endl;
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

	if (f_v) {
		cout << "canonical_form_classifier::classify_curve_nauty done" << endl;
	}

}


void canonical_form_classifier::report(std::string &fname_base, int verbose_level)
{

	string label;
	string fname;


	label.assign(fname_base);
	label.append("_canonical");

	fname.assign(label);
	fname.append(".tex");


	{
		ofstream ost(fname);
		latex_interface L;

		L.head_easy(ost);


		report2(ost, fname_base, verbose_level);

		L.foot(ost);
	}
	file_io Fio;

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;



}

void canonical_form_classifier::report2(std::ostream &ost, std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	std::string label;
	std::string fname;

	if (f_v) {
		cout << "canonical_form_classifier::report2" << endl;
	}

	label.assign(fname_base);
	label.append("_canonical");

	fname.assign(label);
	fname.append("_data.csv");

	int i, j;
	int nb_cols = 5 + nb_orbits + 15;

	long int *Table;

	Table = NEW_lint(nb_objects_to_test * nb_cols);

	for (i = 0; i < nb_objects_to_test; i++) {

		cout << "i=" << i << endl;


		if (CFS_table[i]) {
			Table[i * nb_cols + 0] = i;
			Table[i * nb_cols + 1] = CFS_table[i]->cnt;
			Table[i * nb_cols + 2] = CFS_table[i]->row;
			Table[i * nb_cols + 3] = CFS_table[i]->nb_pts;
			Table[i * nb_cols + 4] = nb_orbits;

			//cout << "i=" << i << " getting orbit_frequencies" << endl;

			for (j = 0; j < nb_orbits; j++) {
				Table[i * nb_cols + 5 + j] = CFS_table[i]->orbit_frequencies[j];
			}

			//cout << "i=" << i << " getting orbit_frequencies part 3" << endl;

			Table[i * nb_cols + 5 + nb_orbits + 0] = CFS_table[i]->nb_types;
			Table[i * nb_cols + 5 + nb_orbits + 1] = CFS_table[i]->selected_type;
			Table[i * nb_cols + 5 + nb_orbits + 2] = CFS_table[i]->selected_orbit;
			Table[i * nb_cols + 5 + nb_orbits + 3] = CFS_table[i]->selected_frequency;
			Table[i * nb_cols + 5 + nb_orbits + 4] = CFS_table[i]->go_min.as_lint();
			Table[i * nb_cols + 5 + nb_orbits + 5] = CFS_table[i]->Gens_stabilizer_original_set->group_order_as_lint();
			Table[i * nb_cols + 5 + nb_orbits + 6] = CFS_table[i]->CS->reduced_set_size;
			Table[i * nb_cols + 5 + nb_orbits + 7] = CFS_table[i]->nb_interesting_subsets;
			Table[i * nb_cols + 5 + nb_orbits + 8] = CFS_table[i]->CS->nb_interesting_subsets_reduced;
			Table[i * nb_cols + 5 + nb_orbits + 9] = CFS_table[i]->CS->nb_interesting_subsets_rr;
			Table[i * nb_cols + 5 + nb_orbits + 10] = CFS_table[i]->CS->nb_orbits;
			Table[i * nb_cols + 5 + nb_orbits + 11] = CFS_table[i]->CS->nb_interesting_orbits;
			Table[i * nb_cols + 5 + nb_orbits + 12] = CFS_table[i]->CS->nb_interesting_points;
			Table[i * nb_cols + 5 + nb_orbits + 13] = CFS_table[i]->Orb->used_length;
			Table[i * nb_cols + 5 + nb_orbits + 14] = CFS_table[i]->gens_stab_of_canonical_equation->group_order_as_lint();
		}
		else {
			Orbiter->Lint_vec.zero(Table + i * nb_cols, nb_cols);
			Table[i * nb_cols + 0] = i;
		}

	}
	if (f_v) {
		cout << "canonical_form_classifier::report2 finished collecting Table" << endl;
	}

	file_io Fio;

	Fio.lint_matrix_write_csv(fname, Table, nb_objects_to_test, nb_cols);

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "canonical_form_classifier::report2 done" << endl;
	}

}

#if 0
int cnt;
int row;
int counter;
int *eqn;
int sz;
long int *pts;
int nb_pts;
long int *bitangents;
int nb_bitangents;

long int *canonical_pts;

int nCk;
int *isotype;
int *orbit_frequencies;
int nb_orbits;
tally *T;

set_of_sets *SoS;
int *types;
int nb_types;
int selected_type;
int selected_orbit;
int selected_frequency;

longinteger_object go_min;

strong_generators *gens;

strong_generators *Gens_stabilizer_original_set;
strong_generators *Gens_stabilizer_canonical_form;


orbit_of_equations *Orb;

strong_generators *gens_stab_of_canonical_equation;

int *trans1;
int *trans2;
int *intermediate_equation;



int *Elt;
int *eqn2;

int *canonical_equation;
int *transporter_to_canonical_form;
#endif


}}

