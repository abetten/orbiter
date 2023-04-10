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
namespace projective_geometry {



canonical_form_classifier::canonical_form_classifier()
{
	Descr = NULL;

	skip_vector = NULL;
	skip_sz = 0;

	Poly_ring = NULL;
	AonHPD = NULL;
	nb_objects_to_test = 0;

	idx_po = idx_so = idx_eqn = idx_pts = idx_bitangents = 0;

	CB = NULL;
	canonical_labeling_len = 0;
	//alpha = NULL;
	//gamma = NULL;

	SubC = NULL;

	Elt = NULL;
	eqn2 = NULL;

	//canonical_equation = NULL;
	//transporter_to_canonical_form = NULL;
	//longinteger_object go_eqn;

	CFS_table = NULL;

	Variety_table = NULL;

	counter = 0;
	Canonical_forms = NULL;
	Goi = NULL;

	Classification_of_quartic_curves = NULL;

	transversal = NULL;
	frequency = NULL;
	nb_types = 0;

}

canonical_form_classifier::~canonical_form_classifier()
{

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
		cout << "canonical_form_classifier::init algorithm = ";
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


	if (!Descr->f_degree) {
		cout << "canonical_form_classifier::init "
				"please use -degree <d>  to specify the degree" << endl;
		exit(1);
	}
	if (!Descr->f_output_fname) {
		cout << "please use -output_fname" << endl;
		exit(1);
	}

	Poly_ring = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"before Poly_ring->init" << endl;
	}
	Poly_ring->init(
			Descr->PA->F,
			Descr->PA->n + 1,
			Descr->degree,
			t_PART,
			verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after Poly_ring->init" << endl;
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
	AonHPD->init(Descr->PA->A, Poly_ring, verbose_level - 3);
	if (f_v) {
		cout << "canonical_form_classifier::init "
				"after AonHPD->init" << endl;
	}


	if (Descr->f_skip) {
		if (f_v) {
			cout << "canonical_form_classifier::init "
					"f_skip" << endl;
		}
		Get_int_vector_from_label(Descr->skip_label,
				skip_vector, skip_sz, 0 /* verbose_level */);
		data_structures::sorting Sorting;

		Sorting.int_vec_heapsort(skip_vector, skip_sz);
		if (f_v) {
			cout << "canonical_form_classifier::init "
					"skip list consists of " << skip_sz << " cases" << endl;
			cout << "The cases to be skipped are :";
			Int_vec_print(cout, skip_vector, skip_sz);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "canonical_form_classifier::init done" << endl;
	}
}


int canonical_form_classifier::skip_this_one(int counter)
{
	data_structures::sorting Sorting;
	int idx;

	if (Descr->f_skip) {
		if (Sorting.int_vec_search(skip_vector, skip_sz, counter, idx)) {
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
void canonical_form_classifier::count_nb_objects_to_test(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int cnt;

	if (f_v) {
		cout << "canonical_form_classifier::count_nb_objects_to_test" << endl;
	}

	nb_objects_to_test = 0;


	for (cnt = 0; cnt < Descr->nb_files; cnt++) {

		if (f_v) {
			cout << "canonical_form_classifier::count_nb_objects_to_test "
					<< cnt << " / " << Descr->nb_files << endl;
		}
		char str[1000];
		string fname;

		snprintf(str, sizeof(str), Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		data_structures::spreadsheet S;

		if (f_v) {
			cout << "canonical_form_classifier::count_nb_objects_to_test "
					<< cnt << " / " << Descr->nb_files
					<< " fname=" << fname << endl;
		}
		S.read_spreadsheet(fname, 0 /*verbose_level*/);

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


void canonical_form_classifier::classify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify" << endl;
	}




	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before count_nb_objects_to_test" << endl;
	}

	count_nb_objects_to_test(verbose_level - 2);


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}

	Variety_table = (canonical_form_of_variety **) NEW_pvoid(nb_objects_to_test);

	Elt = NEW_int(Descr->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Poly_ring->get_nb_monomials());



	Canonical_forms = NEW_int(nb_objects_to_test * Poly_ring->get_nb_monomials());
	Goi = NEW_lint(nb_objects_to_test);

	if (Descr->f_algorithm_nauty) {
		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"before classify_nauty" << endl;
		}
		classify_nauty(verbose_level - 1);
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
		classify_with_substructure(verbose_level - 1);
		if (f_v) {
			cout << "canonical_form_classifier::classify "
					"after classify_with_substructure" << endl;
		}
	}
	else {
		cout << "canonical_form_classifier::classify "
				"please select which algorithm to use" << endl;
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
		Int_vec_print(cout,
				Canonical_forms + i * Poly_ring->get_nb_monomials(),
				Poly_ring->get_nb_monomials());
		cout << " : " << Goi[i] << endl;
	}

	Classification_of_quartic_curves = NEW_OBJECT(data_structures::tally_vector_data);

	Classification_of_quartic_curves->init(Canonical_forms,
			nb_objects_to_test, Poly_ring->get_nb_monomials(),
			verbose_level);


	Classification_of_quartic_curves->get_transversal(
			transversal, frequency, nb_types, verbose_level);


	cout << "Classification of curves:" << endl;


	cout << "transversal:" << endl;
	Int_vec_print(cout, transversal, nb_types);
	cout << endl;

	//Classification_of_quartic_curves->print();

	for (i = 0; i < Classification_of_quartic_curves->nb_types; i++) {

		//h = int_vec_hash(Reps + i * data_set_sz, data_set_sz);

		cout << i << " : " << Classification_of_quartic_curves->Frequency[i] << " x ";
		Int_vec_print(cout,
				Classification_of_quartic_curves->Reps + i * Classification_of_quartic_curves->data_set_sz,
				Classification_of_quartic_curves->data_set_sz);
		cout << " : ";
		j = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[i]];
		cout << Goi[j] << " : ";
		Int_vec_print(cout,
				Classification_of_quartic_curves->sorting_perm_inv + Classification_of_quartic_curves->type_first[i],
				Classification_of_quartic_curves->Frequency[i]);
		cout << endl;
#if 0
		cout << "for elements ";
		int_vec_print(cout, sorting_perm_inv + type_first[i], Frequency[i]);
		cout << endl;
#endif
	}


#if 0
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before write_canonical_forms_csv" << endl;
	}
	write_canonical_forms_csv(
			Descr->fname_base_out,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after write_canonical_forms_csv" << endl;
	}


	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"before generate_source_code" << endl;
	}

	generate_source_code(
			Descr->fname_base_out,
			verbose_level);

	if (f_v) {
		cout << "canonical_form_classifier::classify "
				"after generate_source_code" << endl;
	}
#endif

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


	CB = NEW_OBJECT(data_structures::classify_bitvectors);




	if (f_v) {
		cout << "canonical_form_classifier::classify_nauty "
				"before main_loop" << endl;
	}
	main_loop(verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify_nauty "
				"after main_loop" << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::classify_nauty "
				"The number of isomorphism types is " << CB->nb_types << endl;
	}


}

void canonical_form_classifier::classify_with_substructure(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure, "
				"Descr->substructure_size=" << Descr->substructure_size << endl;
	}




	SubC = NEW_OBJECT(set_stabilizer::substructure_classifier);

	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure "
				"before SubC->classify_substructures" << endl;
	}

	SubC->classify_substructures(
			Descr->fname_base_out,
			Descr->PA->A,
			Descr->PA->A,
			Descr->PA->A->Strong_gens,
			Descr->substructure_size,
			verbose_level - 3);

	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure "
				"after SubC->classify_substructures" << endl;
		cout << "canonical_form_classifier::classify_with_substructure "
				"We found " << SubC->nb_orbits
				<< " orbits at level " << Descr->substructure_size << ":" << endl;
	}



	CFS_table = (canonical_form_substructure **) NEW_pvoid(nb_objects_to_test);



	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure "
				"before main_loop" << endl;
	}
	main_loop(verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::classify_with_substructure "
				"after main_loop" << endl;
	}



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

	string fname_case_out;


	counter = 0;

	for (cnt = 0; cnt < Descr->nb_files; cnt++) {
		char str[1000];
		string fname;
		int row;


		snprintf(str, sizeof(str), Descr->fname_mask.c_str(), cnt);
		fname.assign(str);

		data_structures::spreadsheet S;

		S.read_spreadsheet(fname, 0 /*verbose_level*/);

		if (f_v) {
			cout << "canonical_form_classifier::classify_nauty "
					"S.nb_rows = " << S.nb_rows << endl;
			cout << "canonical_form_classifier::classify_nauty "
					"S.nb_cols = " << S.nb_cols << endl;
		}




		idx_po = S.find_column(Descr->column_label_po);
		idx_so = S.find_column(Descr->column_label_so);
		idx_eqn = S.find_column(Descr->column_label_eqn);
		idx_pts = S.find_column(Descr->column_label_pts);
		idx_bitangents = S.find_column(Descr->column_label_bitangents);

		for (row = 0; row < S.nb_rows - 1; row++, counter++) {

			if (f_v) {
				cout << "cnt = " << cnt << " / " << Descr->nb_files
						<< " row = " << row << " / " << S.nb_rows - 1 << endl;
				cout << "counter = " << counter << " / " << nb_objects_to_test << endl;
			}

			if (skip_this_one(counter)) {
				if (f_v) {
					cout << "canonical_form_classifier::main_loop skipping case counter = " << counter << endl;
				}
				continue;
			}


			snprintf(str, sizeof(str), "_cnt%d", counter);

			fname_case_out.assign(Descr->fname_base_out);
			fname_case_out.append(str);

			quartic_curve_object *Qco;

			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"before prepare_input" << endl;
			}
			prepare_input(row,
					&S,
					Qco, verbose_level - 2);
			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"after prepare_input" << endl;
			}

			canonical_form_of_variety *Variety;

			Variety = NEW_OBJECT(canonical_form_of_variety);

			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"before Variety->init" << endl;
			}
			Variety->init(
					this,
					fname_case_out,
					Qco,
					verbose_level - 2);
			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"after Variety->init" << endl;
			}

			Variety_table[counter] = Variety;

			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"before Variety->compute_canonical_form" << endl;
			}
			Variety->compute_canonical_form(counter, verbose_level - 1);
			if (f_v) {
				cout << "canonical_form_classifier::main_loop "
						"after Variety->compute_canonical_form" << endl;
			}

			// Don't free Qco, because it is now stored in Variety_table[]
			//FREE_OBJECT(Qco);

		} // next row


	} // next cnt

	if (f_v) {
		cout << "canonical_form_classifier::main_loop done" << endl;
	}
}

void canonical_form_classifier::prepare_input(int row,
		data_structures::spreadsheet *S,
		quartic_curve_object *&Qco, int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "canonical_form_classifier::prepare_input" << endl;
	}



	int t;
	int po, so;


	po = S->get_lint(row + 1, idx_po);
	so = S->get_lint(row + 1, idx_so);
	string eqn_txt;
	string pts_txt;
	string bitangents_txt;

	t = S->Table[(row + 1) * S->nb_cols + idx_eqn];
	if (S->tokens[t] == NULL) {
		cout << "canonical_form_classifier::prepare_input "
				"token[t] == NULL" << endl;
	}
	eqn_txt.assign(S->tokens[t]);
	t = S->Table[(row + 1) * S->nb_cols + idx_pts];
	if (S->tokens[t] == NULL) {
		cout << "canonical_form_classifier::prepare_input "
				"token[t] == NULL" << endl;
	}
	pts_txt.assign(S->tokens[t]);
	t = S->Table[(row + 1) * S->nb_cols + idx_bitangents];
	if (S->tokens[t] == NULL) {
		cout << "canonical_form_classifier::prepare_input "
				"token[t] == NULL" << endl;
	}
	bitangents_txt.assign(S->tokens[t]);

	data_structures::string_tools ST;

	if (f_v) {
		cout << "canonical_form_classifier::prepare_input "
				"row = " << row
				<< " before processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt
				<< " =" << bitangents_txt << endl;
	}


	ST.remove_specific_character(eqn_txt, '\"');
	ST.remove_specific_character(pts_txt, '\"');
	ST.remove_specific_character(bitangents_txt, '\"');

	if (f_v) {
		cout << "canonical_form_classifier::prepare_input "
				"row = " << row << " after processing, eqn=" << eqn_txt
				<< " pts_txt=" << pts_txt << " =" << bitangents_txt << endl;
	}



	Qco = NEW_OBJECT(quartic_curve_object);

	if (f_v) {
		cout << "canonical_form_classifier::prepare_input "
				"before Qco->init" << endl;
	}
	Qco->init(
			counter, po, so,
			eqn_txt,
			pts_txt, bitangents_txt,
			verbose_level);
	if (f_v) {
		cout << "canonical_form_classifier::prepare_input "
				"after Qco->init" << endl;
	}

	if (f_v) {
		cout << "canonical_form_classifier::prepare_input done" << endl;
	}

}



void canonical_form_classifier::write_canonical_forms_csv(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	std::string fname;
	int i, j;
	data_structures::sorting Sorting;

	//int nb_orbits;
	int nb_monomials;

	actions::action *A;
	actions::action *A_on_lines;

	if (f_v) {
		cout << "canonical_form_classifier::write_canonical_forms_csv" << endl;
	}
	fname.assign(fname_base);
	fname.append("_canonical_form.csv");


	//nb_orbits = Classification_of_quartic_curves->nb_types;
	nb_monomials = Poly_ring->get_nb_monomials();
	if (f_v) {
		cout << "canonical_form_classifier::write_canonical_forms_csv "
				"nb_monomials = " << nb_monomials << endl;
	}


	A = Descr->PA->A;
	A_on_lines = Descr->PA->A_on_lines;


	{
		ofstream ost(fname.c_str());

		ost << "ROW,CNT,PO,SO,SourceRow,Eqn,Pts,Bitangents,"
				"Transporter,CanEqn,CanPts,CanLines,AutTl,AutGens,Ago" << endl;
		for (i = 0; i < nb_objects_to_test; i++) {

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test << endl;
			}


#if 0
			// CFS_table is only available if f_algorithm_substructure is true
			if (CFS_table == NULL) {
				continue;
			}

			if (CFS_table[i] == NULL) {
				continue;
			}
#endif

			ost << i;
			ost << ",";
			ost << Variety_table[i]->Qco->cnt;
			ost << ",";
			ost << Variety_table[i]->Qco->po;
			ost << ",";
			ost << Variety_table[i]->Qco->so;
			ost << ",";

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing equation" << endl;
			}
			{
				string str;
				Int_vec_create_string_with_quotes(str,
						Variety_table[i]->Qco->eqn, nb_monomials);
				ost << str;
			}
			ost << ",";

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing pts" << endl;
			}
			{
				string str;
				Lint_vec_create_string_with_quotes(str,
						Variety_table[i]->Qco->pts,
						Variety_table[i]->Qco->nb_pts);
				ost << str;
			}
			ost << ",";

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing bitangents" << endl;
			}
			{
				string str;
				Lint_vec_create_string_with_quotes(str,
						Variety_table[i]->Qco->bitangents,
						Variety_table[i]->Qco->nb_bitangents);
				ost << str;
			}
			ost << ",";

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing transporter_to_canonical_form" << endl;
			}
			{
				string str;
				Int_vec_create_string_with_quotes(str,
						Variety_table[i]->transporter_to_canonical_form,
						A->make_element_size);
				ost << str;
			}
			ost << ",";

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing canonical_equation" << endl;
			}

			{
				string str;
				Int_vec_create_string_with_quotes(str,
						Variety_table[i]->canonical_equation,
						nb_monomials);
				ost << str;
			}
			ost << ",";


			long int *Pts_orig;
			long int *Pts_canonical;

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " mapping points" << endl;
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"transporter_to_canonical_form=";
				Int_vec_print(cout,
						Variety_table[i]->transporter_to_canonical_form,
						A->make_element_size);
				cout << endl;
			}
			Pts_orig = Variety_table[i]->Qco->pts;
			Pts_canonical = NEW_lint(Variety_table[i]->Qco->nb_pts);

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " computing Pts_canonical" << endl;
			}

			for (j = 0; j < Variety_table[i]->Qco->nb_pts; j++) {
				Pts_canonical[j] = A->Group_element->element_image_of(Pts_orig[j],
						Variety_table[i]->transporter_to_canonical_form,
						0 /* verbose_level */);
			}
			Sorting.lint_vec_heapsort(Pts_canonical, Variety_table[i]->Qco->nb_pts);


			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing Pts_canonical" << endl;
			}
			{
				string str;
				orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str,
						Pts_canonical, Variety_table[i]->Qco->nb_pts);
				ost << str;
			}
			ost << ",";


			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " mapping bitangents" << endl;
			}
			long int *bitangents_orig;
			long int *bitangents_canonical;

			bitangents_orig = Variety_table[i]->Qco->bitangents;
			bitangents_canonical = NEW_lint(Variety_table[i]->Qco->nb_bitangents);

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test << " bitangents_orig:" << endl;
				Lint_vec_print(cout, bitangents_orig, Variety_table[i]->Qco->nb_bitangents);
				cout << endl;
			}

			for (j = 0; j < Variety_table[i]->Qco->nb_bitangents; j++) {
				bitangents_canonical[j] = A_on_lines->Group_element->element_image_of(
						bitangents_orig[j],
						Variety_table[i]->transporter_to_canonical_form,
						0 /* verbose_level */);
			}

			//Sorting.lint_vec_heapsort(bitangents_canonical, CFS_table[i]->nb_bitangents);

			if (f_v) {
				cout << "canonical_form_classifier::write_canonical_forms_csv "
						"i=" << i << " / " << nb_objects_to_test
						<< " writing bitangents_canonical" << endl;
			}
			{
				string str;
				orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str,
						bitangents_canonical,
						Variety_table[i]->Qco->nb_bitangents);
				ost << str;
			}
			ost << ",";

			groups::strong_generators *gens;

			gens = Variety_table[i]->gens_stab_of_canonical_equation;


			if (gens) {
				if (f_v) {
					cout << "canonical_form_classifier::write_canonical_forms_csv "
							"i=" << i << " / " << nb_objects_to_test
							<< " writing tl" << endl;
				}

				{
					string str;
					Int_vec_create_string_with_quotes(str,
							gens->tl, A->base_len());
					ost << str;
				}
				ost << ",";

				if (f_v) {
					cout << "canonical_form_classifier::write_canonical_forms_csv "
							"i=" << i << " / " << nb_objects_to_test
							<< " writing gens" << endl;
				}

				{
					string str;

					gens->get_gens_data_as_string_with_quotes(str,
							0 /*verbose_level*/);
					ost << str;
				}
				ost << ",";
				ring_theory::longinteger_object go;

				if (f_v) {
					cout << "canonical_form_classifier::write_canonical_forms_csv "
							"i=" << i << " / " << nb_objects_to_test
							<< " writing go" << endl;
				}

				gens->group_order(go);
				ost << go;
			}
			else {
				cout << "gens is not available" << endl;
			}
			ost << endl;

		}
		ost << "END" << endl;
	}


	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname.c_str()) << endl;
	if (f_v) {
		cout << "canonical_form_classifier::write_canonical_forms_csv done" << endl;
	}
}



void canonical_form_classifier::generate_source_code(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	std::string fname;
	int orbit_index;
	int i, j;

	int nb_orbits;
	int nb_monomials;

	actions::action *A;
	actions::action *A_on_lines;

	if (f_v) {
		cout << "canonical_form_classifier::generate_source_code" << endl;
	}


#if 0
	if (!Descr->f_algorithm_substructure) {
		cout << "canonical_form_classifier::generate_source_code skipping because "
				"we did not use substructure classification" << endl;
		return;
	}
#endif

	fname.assign(fname_base);
	fname.append(".cpp");


	nb_orbits = Classification_of_quartic_curves->nb_types;
	nb_monomials = Poly_ring->get_nb_monomials();


	A = Descr->PA->A;
	A_on_lines = Descr->PA->A_on_lines;

	{
		ofstream f(fname.c_str());

		f << "static int " << fname_base << "_nb_reps = "
				<< nb_orbits << ";" << endl;
		f << "static int " << fname_base << "_size = "
				<< nb_monomials << ";" << endl;



		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing reps" << endl;
		}
		f << "// the equations:" << endl;
		f << "static int " << fname_base << "_reps[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			int *equation;

			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];


			//canonical_form_substructure *CFS = Variety_table[idx];


			if (Variety_table[idx]) {
				//equation = Classification_of_quartic_curves->Reps + orbit_index * Classification_of_quartic_curves->data_set_sz;
				equation = Variety_table[idx]->canonical_equation;

				f << "\t";
				for (i = 0; i < nb_monomials; i++) {
					f << equation[i];
					f << ", ";
				}
				f << endl;
			}
			else {
				f << "\t";
				for (i = 0; i < nb_monomials; i++) {
					f << 0;
					f << ", ";
				}
				f << "// problem" << endl;

			}

		}
		f << "};" << endl;



		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing stab_order" << endl;
		}
		f << "// the stabilizer orders:" << endl;
		f << "static const char *" << fname_base << "_stab_order[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {

			ring_theory::longinteger_object ago;

			int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];


			ago.create(Goi[idx], __FILE__, __LINE__);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;





		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing Bitangents" << endl;
		}
		f << "// the 28 bitangents:" << endl;
		f << "static long int " << fname_base << "_Bitangents[] = { " << endl;


		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

			//canonical_form_substructure *CFS = Variety_table[idx];


			if (Variety_table[idx]) {
				long int *bitangents_orig;
				long int *bitangents_canonical;

				bitangents_orig = Variety_table[idx]->Qco->bitangents;
				bitangents_canonical = NEW_lint(Variety_table[idx]->Qco->nb_bitangents);
				for (j = 0; j < Variety_table[idx]->Qco->nb_bitangents; j++) {
					bitangents_canonical[j] =
							A_on_lines->Group_element->element_image_of(
									bitangents_orig[j],
							Variety_table[idx]->transporter_to_canonical_form, 0 /* verbose_level */);
				}




				f << "\t";
				for (j = 0; j < 28; j++) {
					f << bitangents_canonical[j];
					f << ", ";
				}
				f << endl;
			}
			else {
				f << "\t";
				for (j = 0; j < 28; j++) {
					f << 0;
					f << ", ";
				}
				f << "// problem" << endl;

			}

		}
		f << "};" << endl;

		f << "static int " << fname_base << "_make_element_size = "
				<< A->make_element_size << ";" << endl;

		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;



			stab_gens_first = NEW_int(nb_orbits);
			stab_gens_len = NEW_int(nb_orbits);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {


				groups::strong_generators *gens;

				int idx;

				idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

				//canonical_form_substructure *CFS = CFS_table[idx];
				//gens = CFS->Gens_stabilizer_canonical_form;
				if (Variety_table[idx]) {
					gens = Variety_table[idx]->gens_stab_of_canonical_equation;


					if (gens) {
						stab_gens_first[orbit_index] = fst;
						stab_gens_len[orbit_index] = gens->gens->len;
						fst += stab_gens_len[orbit_index];
					}
					else {
						cout << "canonical_form_classifier::generate_source_code gens not available" << endl;
						stab_gens_first[orbit_index] = fst;
						stab_gens_len[orbit_index] = 0;
						fst += stab_gens_len[orbit_index];
					}
				}
				else {
					stab_gens_first[orbit_index] = fst;
					stab_gens_len[orbit_index] = 0;
					fst += 0;

				}
			}


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens_fst" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_fst[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_first[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens_len" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_len[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_len[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens" << endl;
			}
			f << "static int " << fname_base << "_stab_gens[] = {" << endl;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				int j;

				for (j = 0; j < stab_gens_len[orbit_index]; j++) {
					if (f_vv) {
						cout << "canonical_form_classifier::generate_source_code "
								"before extract_strong_generators_in_order "
								"generator " << j << " / "
								<< stab_gens_len[orbit_index] << endl;
					}
					f << "\t";

					groups::strong_generators *gens;

					int idx;

					idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

					//canonical_form_substructure *CFS = CFS_table[idx];
					//gens = CFS->Gens_stabilizer_canonical_form;
					if (Variety_table[idx]) {
						gens = Variety_table[idx]->gens_stab_of_canonical_equation;


						if (gens) {
							A->Group_element->element_print_for_make_element(gens->gens->ith(j), f);
							f << endl;
						}
						else {
							cout << "canonical_form_classifier::generate_source_code gens are not available" << endl;
						}
					}
					else {
						f << "// problem" << endl;
					}
				}
			}
			f << "};" << endl;


			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname.c_str()) << endl;
	if (f_v) {
		cout << "canonical_form_classifier::generate_source_code done" << endl;
	}
}


void canonical_form_classifier::report(
		poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_classifier::report" << endl;
	}

	string label;
	string fname;

	if (Opt->f_fname) {
		label.assign(Opt->fname);
	}
	else {
		label.assign("report");
	}


	fname.assign(label);
	fname.append("_orbits.tex");



	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;

		L.head_easy(ost);


		report2(ost, verbose_level);

		L.foot(ost);
	}



	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}


	{
		string fname_data;

		fname_data.assign(label);
		fname_data.append("_canonical_form_data.csv");


		if (f_v) {
			cout << "canonical_form_classifier::report "
					"before export_canonical_form_data" << endl;
		}
		export_canonical_form_data(fname_data, verbose_level);
		if (f_v) {
			cout << "canonical_form_classifier::report "
					"after export_canonical_form_data" << endl;
		}

		if (f_v) {
			cout << "Written file " << fname_data << " of size "
					<< Fio.file_size(fname_data) << endl;
		}
	}
	if (f_v) {
		cout << "canonical_form_classifier::report done" << endl;
	}

}

void canonical_form_classifier::report2(std::ostream &ost, int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_classifier::report2" << endl;
	}


	int orbit_index;
	//int i, j;

	int nb_orbits;
	int nb_monomials;

	//actions::action *A;
	//actions::action *A_on_lines;

	if (f_v) {
		cout << "canonical_form_classifier::report2" << endl;
	}


	nb_orbits = Classification_of_quartic_curves->nb_types;
	nb_monomials = Poly_ring->get_nb_monomials();


	//A = Descr->PA->A;
	//A_on_lines = Descr->PA->A_on_lines;


	int idx;

	{


		ost << "Classification\\\\" << endl;
		ost << "$q=" << Descr->PA->F->q << "$\\\\" << endl;
		ost << "Number of isomorphism classes: " << nb_orbits << "\\\\" << endl;


		std::vector<long int> Ago;

		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];


			Ago.push_back(Goi[idx]);
		}

		data_structures::tally_lint T;

		T.init_vector_lint(Ago,
				false /* f_second */, 0 /* verbose_level */);
		ost << "Automorphism group order statistic: " << endl;
		//ost << "$";
		T.print_file_tex(ost, true /* f_backwards */);
		ost << "\\\\" << endl;


		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;


		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing reps" << endl;
		}
		ost << "The isomorphism classes are:\\\\" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			int *equation;

			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			ost << "Isomorphism class " << orbit_index << " / " << nb_orbits << ":\\\\" << endl;


			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

			//canonical_form_substructure *CFS = CFS_table[idx];



			if (Variety_table[idx]) {


				ost << "Number of rational points over $\\bbF_{" << Descr->PA->F->q << "}$: ";
				ost << Variety_table[idx]->Qco->nb_pts;
				ost << "\\\\" << endl;


				//equation = Classification_of_quartic_curves->Reps + orbit_index * Classification_of_quartic_curves->data_set_sz;
				equation = Variety_table[idx]->canonical_equation;

				ost << "Canonical equation:" << endl;
				ost << "\\begin{eqnarray*}" << endl;

				//Poly_ring->print_equation_tex(ost, equation);

				Poly_ring->print_equation_with_line_breaks_tex(ost, equation, 7, "\\\\");

				ost << "\\end{eqnarray*}" << endl;

				Int_vec_print(ost, equation, nb_monomials);
				ost << "\\\\" << endl;

			}
			else {
				ost << "Data not available.\\\\" << endl;

			}

			ring_theory::longinteger_object ago;

			//int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];


			ago.create(Goi[idx], __FILE__, __LINE__);

			ost << "Stabilizer order: ";

			ago.print_not_scientific(ost);
			ost << "\\\\" << endl;

			groups::strong_generators *gens;

			if (Variety_table[idx]) {
				gens = Variety_table[idx]->gens_stab_of_canonical_equation;

				ost << "The stabilizer: \\\\" << endl;
				gens->print_generators_tex(ost);
			}
			else {
				ost << "Data not available.\\\\" << endl;

			}




		}


#if 0
		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing stab_order" << endl;
		}
		f << "// the stabilizer orders:" << endl;
		f << "static const char *" << fname_base << "_stab_order[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {

			ring_theory::longinteger_object ago;

			int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];


			ago.create(Goi[idx], __FILE__, __LINE__);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;





		if (f_v) {
			cout << "canonical_form_classifier::generate_source_code "
					"preparing Bitangents" << endl;
		}
		f << "// the 28 bitangents:" << endl;
		f << "static long int " << fname_base << "_Bitangents[] = { " << endl;


		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			int idx;

			idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

			canonical_form_substructure *CFS = CFS_table[idx];


			if (CFS) {
				long int *bitangents_orig;
				long int *bitangents_canonical;

				bitangents_orig = CFS->Qco->bitangents;
				bitangents_canonical = NEW_lint(CFS->Qco->nb_bitangents);
				for (j = 0; j < CFS->Qco->nb_bitangents; j++) {
					bitangents_canonical[j] = A_on_lines->element_image_of(bitangents_orig[j],
							CFS->transporter_to_canonical_form, 0 /* verbose_level */);
				}




				f << "\t";
				for (j = 0; j < 28; j++) {
					f << bitangents_canonical[j];
					f << ", ";
				}
				f << endl;
			}
			else {
				f << "\t";
				for (j = 0; j < 28; j++) {
					f << 0;
					f << ", ";
				}
				f << "// problem" << endl;

			}

		}
		f << "};" << endl;

		f << "static int " << fname_base << "_make_element_size = "
				<< A->make_element_size << ";" << endl;

		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;



			stab_gens_first = NEW_int(nb_orbits);
			stab_gens_len = NEW_int(nb_orbits);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {


				groups::strong_generators *gens;

				int idx;

				idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

				canonical_form_substructure *CFS = CFS_table[idx];
				//gens = CFS->Gens_stabilizer_canonical_form;
				if (CFS) {
					gens = CFS->gens_stab_of_canonical_equation;


					stab_gens_first[orbit_index] = fst;
					stab_gens_len[orbit_index] = gens->gens->len;
					fst += stab_gens_len[orbit_index];
				}
				else {
					stab_gens_first[orbit_index] = fst;
					stab_gens_len[orbit_index] = 0;
					fst += 0;

				}
			}


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens_fst" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_fst[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_first[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens_len" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_len[] = { " << endl << "\t";
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				f << stab_gens_len[orbit_index];
				if (orbit_index < nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;


			if (f_v) {
				cout << "canonical_form_classifier::generate_source_code "
						"preparing stab_gens" << endl;
			}
			f << "static int " << fname_base << "_stab_gens[] = {" << endl;
			for (orbit_index = 0;
					orbit_index < nb_orbits;
					orbit_index++) {
				int j;

				for (j = 0; j < stab_gens_len[orbit_index]; j++) {
					if (f_vv) {
						cout << "canonical_form_classifier::generate_source_code "
								"before extract_strong_generators_in_order "
								"generator " << j << " / "
								<< stab_gens_len[orbit_index] << endl;
					}
					f << "\t";

					groups::strong_generators *gens;

					int idx;

					idx = Classification_of_quartic_curves->sorting_perm_inv[Classification_of_quartic_curves->type_first[orbit_index]];

					canonical_form_substructure *CFS = CFS_table[idx];
					//gens = CFS->Gens_stabilizer_canonical_form;
					if (CFS) {
						gens = CFS->gens_stab_of_canonical_equation;


						A->element_print_for_make_element(gens->gens->ith(j), f);
						f << endl;
					}
					else {
						f << "// problem" << endl;
					}
				}
			}
			f << "};" << endl;


			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
#endif

	}


	if (f_v) {
		cout << "canonical_form_classifier::report2 done" << endl;
	}
}

void canonical_form_classifier::export_canonical_form_data(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "canonical_form_classifier::export_canonical_form_data" << endl;
	}

	int i, j;
	//int nb_cols = 5 + SubC->nb_orbits + 15;
	int nb_cols = 22;

	//long int *Table;

	data_structures::spreadsheet S;

	S.init_empty_table(nb_objects_to_test + 1, nb_cols);

	//Table = NEW_lint(nb_objects_to_test * nb_cols);

	S.fill_entry_with_text(0, 0, "Line");
	S.fill_entry_with_text(0, 1, "CNT");
	S.fill_entry_with_text(0, 2, "PO");
	S.fill_entry_with_text(0, 3, "SO");
	S.fill_entry_with_text(0, 4, "nb_pts");
	S.fill_entry_with_text(0, 5, "nb_sub_orbs");
	S.fill_entry_with_text(0, 6, "frequencies");
	S.fill_entry_with_text(0, 7, "nb_types");
	S.fill_entry_with_text(0, 8, "selected_type");
	S.fill_entry_with_text(0, 9, "selected_orbit");
	S.fill_entry_with_text(0, 10, "selected_frequency");
	S.fill_entry_with_text(0, 11, "go_min");
	S.fill_entry_with_text(0, 12, "set_stabilizer_order");
	S.fill_entry_with_text(0, 13, "reduced_set_size");
	S.fill_entry_with_text(0, 14, "nb_interesting_subsets");
	S.fill_entry_with_text(0, 15, "nb_interesting_subsets_reduced");
	S.fill_entry_with_text(0, 16, "nb_interesting_subsets_rr");
	S.fill_entry_with_text(0, 17, "nb_orbits");
	S.fill_entry_with_text(0, 18, "nb_interesting_orbits");
	S.fill_entry_with_text(0, 19, "nb_interesting_points");
	S.fill_entry_with_text(0, 20, "orbit_length_under_set_stab");
	S.fill_entry_with_text(0, 21, "stab_of_eqn");

	j = 1;
	for (i = 0; i < nb_objects_to_test; i++, j++) {

		cout << "canonical_form_classifier::export_data i=" << i << " / " << nb_objects_to_test << endl;


		S.set_entry_lint(j, 0, i);

		if (Variety_table[i]) {

			S.set_entry_lint(j, 1, Variety_table[i]->Qco->cnt);
			S.set_entry_lint(j, 2, Variety_table[i]->Qco->po);
			S.set_entry_lint(j, 3, Variety_table[i]->Qco->so);
			S.set_entry_lint(j, 4, Variety_table[i]->Qco->nb_pts);
			S.set_entry_lint(j, 5, SubC->nb_orbits);

			//cout << "i=" << i << " getting orbit_frequencies" << endl;

			string str;

			Int_vec_create_string_with_quotes(str, CFS_table[i]->SubSt->orbit_frequencies, SubC->nb_orbits);

			S.fill_entry_with_text(j, 6, str);

			//cout << "i=" << i << " getting orbit_frequencies part 3" << endl;

			if (CFS_table) {
				S.set_entry_lint(j, 7, CFS_table[i]->SubSt->nb_types);
				S.set_entry_lint(j, 8, CFS_table[i]->SubSt->selected_type);
				S.set_entry_lint(j, 9, CFS_table[i]->SubSt->selected_orbit);
				S.set_entry_lint(j, 10, CFS_table[i]->SubSt->selected_frequency);
				S.set_entry_lint(j, 11, CFS_table[i]->SubSt->gens->group_order_as_lint());
				S.set_entry_lint(j, 12, CFS_table[i]->Gens_stabilizer_original_set->group_order_as_lint());
				S.set_entry_lint(j, 13, CFS_table[i]->CS->Stab_orbits->reduced_set_size);
				S.set_entry_lint(j, 14, CFS_table[i]->SubSt->nb_interesting_subsets);
				S.set_entry_lint(j, 15, CFS_table[i]->CS->Stab_orbits->nb_interesting_subsets_reduced);
				S.set_entry_lint(j, 16, CFS_table[i]->CS->nb_interesting_subsets_rr);
				S.set_entry_lint(j, 17, CFS_table[i]->CS->Stab_orbits->nb_orbits);
				S.set_entry_lint(j, 18, CFS_table[i]->CS->Stab_orbits->nb_interesting_orbits);
				S.set_entry_lint(j, 19, CFS_table[i]->CS->Stab_orbits->nb_interesting_points);
				S.set_entry_lint(j, 20, CFS_table[i]->Orb->used_length);
			}
			S.set_entry_lint(j, 21, Variety_table[i]->gens_stab_of_canonical_equation->group_order_as_lint());

		}
		else {
			//Lint_vec_zero(Table + i * nb_cols, nb_cols);
			//Table[i * nb_cols + 0] = i;
			S.set_entry_lint(j, 1, 0);
			S.set_entry_lint(j, 2, 0);
			S.set_entry_lint(j, 3, 0);
			S.set_entry_lint(j, 4, 0);
			S.set_entry_lint(j, 5, 0);
			S.fill_entry_with_text(j, 6, "");

			int h;

			for (h = 7; h <= 21; h++) {
				S.set_entry_lint(j, h, 0);
			}
		}

	}
	if (f_v) {
		cout << "canonical_form_classifier::export_canonical_form_data finished collecting Table" << endl;
	}

#if 1
	orbiter_kernel_system::file_io Fio;

	S.save(fname, 0 /* verbose_level*/);

	//Fio.lint_matrix_write_csv(fname, Table, nb_objects_to_test, nb_cols);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
#endif

	if (f_v) {
		cout << "canonical_form_classifier::export_canonical_form_data done" << endl;
	}

}


}}}


