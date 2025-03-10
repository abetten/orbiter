/*
 * semifield_classify_with_substructure.cpp
 *
 *  Created on: May 15, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {

static void semifield_print_function_callback(
		std::ostream &ost, int orbit_idx,
		invariant_relations::classification_step *Step, void *print_function_data);


semifield_classify_with_substructure::semifield_classify_with_substructure()
{
	Record_birth();
	Descr = NULL;
	PA = NULL;
	Mtx = NULL;
	//F = NULL;
	Control = NULL;
	t0 = 0;

	identify_semifields_from_file_Po = NULL;
	identify_semifields_from_file_m = 0;

	f_trace_record_prefix = false;
	//trace_record_prefix = NULL;
	f_FstLen = false;
	//fname_FstLen = NULL;
	f_Data = false;
	//fname_Data = NULL;

	p = e = e1 = n = k = q = k2 = 0;

	Sub = NULL;
	L2 = NULL;


	nb_existing_cases = 0;
	Existing_cases = NULL;
	Existing_cases_fst = NULL;
	Existing_cases_len = NULL;


	nb_non_unique_cases = 0;
	Non_unique_cases = NULL;
	Non_unique_cases_fst = NULL;
	Non_unique_cases_len = NULL;
	Non_unique_cases_go = NULL;

	Semifields = NULL;

}

semifield_classify_with_substructure::~semifield_classify_with_substructure()
{
	Record_death();

}

void semifield_classify_with_substructure::init(
		semifield_classify_description *Descr,
		projective_geometry::projective_space_with_action *PA,
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify_with_substructure::init" << endl;
	}
	algebra::number_theory::number_theory_domain NT;


	semifield_classify_with_substructure::Descr = Descr;

	semifield_classify_with_substructure::PA = PA;
	Mtx = PA->A->get_matrix_group();
	semifield_classify_with_substructure::Control = Control;

	NT.factor_prime_power(Descr->order, p, e);
	if (f_v) {
		cout << "order = " << Descr->order << " = " << p << "^" << e << endl;
	}

	if (Descr->f_dim_over_kernel) {
		if (e % Descr->dim_over_kernel) {
			cout << "dim_over_kernel does not divide e" << endl;
			exit(1);
		}
		e1 = e / Descr->dim_over_kernel;
		n = 2 * Descr->dim_over_kernel;
		k = Descr->dim_over_kernel;
		q = NT.i_power_j(p, e1);
		if (f_v) {
			cout << "order=" << Descr->order << " n=" << n
				<< " k=" << k << " q=" << q << endl;
		}
	}
	else {
		n = 2 * e;
		k = e;
		q = p;
		if (f_v) {
			cout << "order=" << Descr->order << " n=" << n
				<< " k=" << k << " q=" << q << endl;
		}
	}
	k2 = k * k;




	Sub = NEW_OBJECT(semifield_substructure);

	Sub->SCWS = this;
	Sub->start_column = 4;


	Sub->SC = NEW_OBJECT(semifield_classify);

	if (!Descr->f_level_two_prefix) {
		Descr->f_level_two_prefix = true;
		Descr->level_two_prefix.assign("L2");
	}
	if (!Descr->f_level_three_prefix) {
		Descr->f_level_three_prefix = true;
		Descr->level_three_prefix.assign("L3");
	}

	if (f_v) {
		cout << "semifield_classify_with_substructure::init before Sub->SC->init" << endl;
	}
	Sub->SC->init(
			PA, k, Control,
			Descr->level_two_prefix,
			Descr->level_three_prefix,
			verbose_level - 1);
	if (f_v) {
		cout << "semifield_classify_with_substructure::init after Sub->SC->init" << endl;
	}




	if (Descr->f_test_semifield) {
		long int *data = NULL;
		int data_len = 0;
		int i;

		cout << "semifield_classify_with_substructure::init f_test_semifield" << endl;
		Lint_vec_scan(Descr->test_semifield_data, data, data_len);
		cout << "input semifield:" << endl;
		for (i = 0; i < data_len; i++) {
			cout << i << " : " << data[i] << endl;
		}
		if (Sub->SC->test_partial_semifield_numerical_data(
				data, data_len, verbose_level)) {
			cout << "the set satisfies the partial semifield condition" << endl;
		}
		else {
			cout << "the set does not satisfy the partial semifield condition" << endl;
		}
		exit(0);
	}


	L2 = NEW_OBJECT(semifield_level_two);

	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"before L2->init" << endl;
	}
	L2->init(Sub->SC, verbose_level);
	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"after L2->init" << endl;
	}


#if 1
	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"before L2->compute_level_two" << endl;
	}
	L2->compute_level_two(4, verbose_level);
	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"after L2->compute_level_two" << endl;
	}
#else
	L2->read_level_info_file(verbose_level);
#endif

	Sub->L3 = NEW_OBJECT(semifield_lifting);

	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"before L3->init_level_three" << endl;
	}
	Sub->L3->init_level_three(L2,
			true /* f_prefix */, Sub->SC->level_three_prefix,
			verbose_level);
	if (f_v) {
		cout << "semifield_classify_with_substructure::init "
				"after L3->init_level_three" << endl;
	}



	if (f_v) {
		cout << "semifield_classify_with_substructure::init done" << endl;
	}
}

void semifield_classify_with_substructure::read_data(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::read_data" << endl;
	}

	if (f_v) {
		cout << "before reading files " << fname_FstLen
			<< " and " << fname_Data << endl;
	}




	other::data_structures::tally C;
	int mtx_n;
	int i, a;


	// We need 64 bit integers for the Data array
	// because semifields of order 64 need 6 x 6 matrices,
	// which need to be encoded using 36 bits.

	if (sizeof(long int) < 8) {
		cout << "sizeof(long int) < 8" << endl;
		exit(1);
	}
	Fio.Csv_file_support->int_matrix_read_csv(
			fname_FstLen,
			Sub->FstLen, Sub->nb_orbits_at_level_3, mtx_n,
			verbose_level);
	Sub->Len = NEW_int(Sub->nb_orbits_at_level_3);
	for (i = 0; i < Sub->nb_orbits_at_level_3; i++) {
		Sub->Len[i] = Sub->FstLen[i * 2 + 1];
	}
	Fio.Csv_file_support->lint_matrix_read_csv(
			fname_Data, Sub->Data,
			Sub->nb_solutions, Sub->data_size,
			verbose_level);


	if (f_v) {
		cout << "Read " << Sub->nb_solutions
			<< " solutions arising from "
			<< Sub->nb_orbits_at_level_3 << " orbits" << endl;
	}





	C.init(Sub->Len, Sub->nb_orbits_at_level_3, false, 0);
	if (f_v) {
		cout << "classification of Len:" << endl;
		C.print_bare(true);
		cout << endl;
	}

	if (f_v) {
		cout << "computing existing cases:" << endl;
	}


	Existing_cases = NEW_int(Sub->nb_orbits_at_level_3);
	nb_existing_cases = 0;

	for (i = 0; i < Sub->nb_orbits_at_level_3; i++) {
		if (Sub->Len[i]) {
			Existing_cases[nb_existing_cases++] = i;
		}
	}
	Existing_cases_fst = NEW_int(nb_existing_cases);
	Existing_cases_len = NEW_int(nb_existing_cases);
	for (i = 0; i < nb_existing_cases; i++) {
		a = Existing_cases[i];
		Existing_cases_fst[i] = Sub->FstLen[2 * a + 0];
		Existing_cases_len[i] = Sub->FstLen[2 * a + 1];
	}
	if (f_v) {
		cout << "There are " << nb_existing_cases
			<< " cases which exist" << endl;
	}

	if (f_v) {
		cout << "computing non-unique cases:" << endl;
	}

	Non_unique_cases = NEW_int(nb_existing_cases);
	nb_non_unique_cases = 0;
	for (i = 0; i < nb_existing_cases; i++) {
		a = Existing_cases[i];
		if (Existing_cases_len[i] > 1) {
			Non_unique_cases[nb_non_unique_cases++] = a;
		}
	}

	if (f_v) {
		cout << "There are " << nb_non_unique_cases
			<< " cases which have more than one solution" << endl;
	}
	Non_unique_cases_fst = NEW_int(nb_non_unique_cases);
	Non_unique_cases_len = NEW_int(nb_non_unique_cases);
	Non_unique_cases_go = NEW_lint(nb_non_unique_cases);
	for (i = 0; i < nb_non_unique_cases; i++) {
		a = Non_unique_cases[i];
		Non_unique_cases_fst[i] = Sub->FstLen[2 * a + 0];
		Non_unique_cases_len[i] = Sub->FstLen[2 * a + 1];
		Non_unique_cases_go[i] =
			Sub->L3->Stabilizer_gens[a].group_order_as_lint();
	}

	{
		other::data_structures::tally C;

		C.init(Non_unique_cases_len, nb_non_unique_cases, false, 0);
		if (f_v) {
			cout << "classification of Len amongst the non-unique cases:" << endl;
			C.print_bare(true);
			cout << endl;
		}
	}
	{
		other::data_structures::tally C;

		C.init_lint(Non_unique_cases_go, nb_non_unique_cases, false, 0);
		if (f_v) {
			cout << "classification of group orders amongst "
					"the non-unique cases:" << endl;
			C.print_bare(true);
			cout << endl;
		}
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::read_data "
				"before Sub->compute_cases" << endl;
	}
	Sub->compute_cases(
			nb_non_unique_cases, Non_unique_cases, Non_unique_cases_go,
			verbose_level);
	if (f_v) {
		cout << "semifield_classify_with_substructure::read_data "
				"after Sub->compute_cases" << endl;
	}

	if (f_v) {
		cout << "semifield_classify_with_substructure::read_data done" << endl;
	}
}

void semifield_classify_with_substructure::create_fname_for_classification(
		std::string &fname)
{
	fname = "semifields_" + std::to_string(Sub->SC->order) + "_classification.bin";
}

void semifield_classify_with_substructure::create_fname_for_flag_orbits(
		std::string &fname)
{
	fname = "semifields_" + std::to_string(Sub->SC->order) + "_flag_orbits.bin";
}

void semifield_classify_with_substructure::classify_semifields(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::classify_semifields" << endl;
	}

	Semifields = NEW_OBJECT(invariant_relations::classification_step);

	Sub->do_classify(verbose_level);

	{
		string fname;

		create_fname_for_classification(fname);
		{
			ofstream fp(fname, ios::binary);

			Semifields->write_file(fp, verbose_level);
		}
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	{
		string fname;

		create_fname_for_flag_orbits(fname);
		{
			ofstream fp(fname, ios::binary);

			Sub->Flag_orbits->write_file(fp, verbose_level);
		}
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "semifield_classify_with_substructure::classify_semifields done" << endl;
	}
}

void semifield_classify_with_substructure::load_classification(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::load_classification" << endl;
	}

	Semifields = NEW_OBJECT(invariant_relations::classification_step);

	algebra::ring_theory::longinteger_object go;

	Sub->SC->A->group_order(go);


	{
		string fname;

		create_fname_for_classification(fname);
		{
			ifstream fp(fname, ios::binary);

			cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;
			Semifields->read_file(fp, Sub->SC->A, Sub->SC->AS, go, verbose_level);
		}
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::load_classification "
				"Semifields->nb_orbits = " << Semifields->nb_orbits << endl;
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::load_classification done" << endl;
	}
}

void semifield_classify_with_substructure::load_flag_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::load_flag_orbits" << endl;
	}



	{
		string fname;

		create_fname_for_flag_orbits(fname);
		{
			ifstream fp(fname, ios::binary);

			cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;
			Sub->Flag_orbits->read_file(fp, Sub->SC->A, Sub->SC->AS, verbose_level);
		}
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::load_flag_orbits "
				"Sub->Flag_orbits->nb_flag_orbits = "
				<< Sub->Flag_orbits->nb_flag_orbits << endl;
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::load_flag_orbits done" << endl;
	}
}


void semifield_classify_with_substructure::identify_semifield(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifield" << endl;
	}

	if (Descr->f_identify_semifield) {
		long int *data = NULL;
		int data_len = 0;
		cout << "f_identify_semifield" << endl;
		Lint_vec_scan(Descr->identify_semifield_data, data, data_len);
		cout << "input semifield:" << endl;
		for (i = 0; i < data_len; i++) {
			cout << i << " : " << data[i] << endl;
		}


		int t, rk, trace_po, fo, po;
		int *transporter;

		transporter = NEW_int(Sub->SC->A->elt_size_in_int);

		for (t = 0; t < 6; t++) {

			long int data_out[6];

			cout << "Knuth operation " << t << " / " << 6 << ":" << endl;
			Sub->SC->knuth_operation(t,
					data, data_out,
					verbose_level);

			Lint_vec_print(cout, data_out, k);
			cout << endl;
			for (i = 0; i < k; i++) {
				Sub->SC->matrix_unrank(data_out[i], Sub->Basis1 + i * k2);
			}
			Sub->SC->basis_print(Sub->Basis1, k);

			if (Sub->identify(
					data_out,
					rk, trace_po, fo, po,
					transporter,
					verbose_level)) {
				cout << "The given semifield has been identified "
						"as semifield orbit " << po << endl;
				cout << "rk=" << rk << endl;
				cout << "trace_po=" << trace_po << endl;
				cout << "fo=" << fo << endl;
				cout << "po=" << po << endl;
				cout << "isotopy:" << endl;
				Sub->SC->A->Group_element->element_print_quick(transporter, cout);
				cout << endl;
			}
			else {
				cout << "The given semifield cannot be identified" << endl;
			}
		}
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifield done" << endl;
	}
}

void semifield_classify_with_substructure::identify_semifields_from_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifields_from_file" << endl;
	}

	identify_semifields_from_file_Po = NULL;

	if (Descr->f_identify_semifields_from_file) {
		cout << "f_identify_semifield_from_file" << endl;

		long int *Data;
		int n;

		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->identify_semifields_from_file_fname, Data,
				identify_semifields_from_file_m, n, verbose_level);
		if (n != Sub->SC->k) {
			cout << "n != Sub->SC->k" << endl;
			exit(1);
		}
		int t, rk, trace_po, fo, po;
		int *transporter;


		transporter = NEW_int(Sub->SC->A->elt_size_in_int);
		identify_semifields_from_file_Po = NEW_int(identify_semifields_from_file_m * 6);

		for (i = 0; i < identify_semifields_from_file_m; i++) {

			for (t = 0; t < 6; t++) {

				long int data_out[6];

				Sub->SC->knuth_operation(t,
						Data + i * n, data_out,
						verbose_level);


				if (Sub->identify(
						data_out,
						rk, trace_po, fo, po,
						transporter,
						verbose_level)) {
					cout << "Identify " << i << " / " << identify_semifields_from_file_m
							<< " : The given semifield has been identified "
							"as semifield orbit " << po << endl;
					cout << "rk=" << rk << endl;
					cout << "trace_po=" << trace_po << endl;
					cout << "fo=" << fo << endl;
					cout << "po=" << po << endl;
					cout << "isotopy:" << endl;
					Sub->SC->A->Group_element->element_print_quick(transporter, cout);
					cout << endl;
					identify_semifields_from_file_Po[i * 6 + t] = po;
				}
				else {
					cout << "The given semifield cannot be identified" << endl;
					identify_semifields_from_file_Po[i * 6 + t] = -1;
				}
			}

		}
		string fname;
		other::data_structures::string_tools ST;

		fname = Descr->identify_semifields_from_file_fname;
		ST.chop_off_extension(fname);
		fname += "_identification.csv";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, identify_semifields_from_file_Po,
				identify_semifields_from_file_m, 6);
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_"
				"semifields_from_file done" << endl;
	}
}

void semifield_classify_with_substructure::latex_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::latex_report" << endl;
	}
	string author, fname, extra_praeamble;
	string title;

	title = "Isotopy classes of semifields of order " + std::to_string(Descr->order);


	author = "Orbiter";

	fname = "Semifields_" + std::to_string(Descr->order) + ".tex";

	if (f_v) {
		cout << "writing latex file " << fname << endl;
		}

	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;


		//latex_head_easy(ost);
		L.head(
				ost,
			false /* f_book */,
			true /* f_title */,
			title,
			author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (Semifields == NULL) {
			cout << "semifield_classify_with_substructure::latex_report "
					"Semifields == NULL" << endl;
			exit(1);
		}
		long int *Go;


		other::data_structures::tally C;

		Go = NEW_lint(Semifields->nb_orbits);
		for (i = 0; i < Semifields->nb_orbits; i++) {
			Go[i] = Semifields->Orbit[i].gens->group_order_as_lint();
		}

		C.init_lint(Go, Semifields->nb_orbits, false, 0);

		ost << "\\section*{Summary}" << endl;
		ost << "Classification by stabilizer order:\\\\" << endl;
		ost << "$$" << endl;
		C.print_array_tex(ost, true /*f_backwards */);
		ost << "$$" << endl;

		if (f_v) {
			cout << "semifield_classify_with_substructure::latex_report "
					"before L2->print_representatives" << endl;
			}

		L2->report(ost, verbose_level);

		if (f_v) {
			cout << "semifield_classify_with_substructure::latex_report "
					"after L2->print_representatives" << endl;
			}

		if (f_v) {
			cout << "semifield_classify_with_substructure::latex_report "
					"before Semifields->print_latex" << endl;
			}

		Semifields->print_latex(
				ost,
			title,
			true /* f_print_stabilizer_gens */,
			true,
			semifield_print_function_callback,
			Sub);


		if (f_v) {
			cout << "semifield_classify_with_substructure::latex_report "
					"after Semifields->print_latex" << endl;
			}

		if (Descr->f_identify_semifields_from_file) {
			ost << "\\clearpage" << endl;
			ost << "\\section*{Identification of Rua types}" << endl;
			ost << "The $i$-th row, $j$-th column of the table is the number $c$ "
					"of the isotopy class in the list above which corresponds "
					"to the $j$-th element in the Knuth orbit of the $i$-th "
					"class in the R\\'ua labeling.\\\\" << endl;
			//fp << "$$" << endl;
			L.print_integer_matrix_tex_block_by_block(
					ost,
					identify_semifields_from_file_Po,
					identify_semifields_from_file_m, 6, 40 /* block_width */);

			//print_integer_matrix_with_standard_labels_and_offset_tex(
			//	fp, identify_semifields_from_file_Po, identify_semifields_from_file_m, 6,
			//	1 /* m_offset */, 1 /* n_offset */);
			//fp << "$$" << endl;
		}


		if (f_v) {
			cout << "semifield_classify_with_substructure::latex_report "
					"substructures of dimension two" << endl;
			}

		int *Po2;
		int *PO2;
		int orbit_idx;

		Po2 = NEW_int(Sub->N2);
		PO2 = NEW_int(Semifields->nb_orbits * Sub->N2);

		ost << "\\clearpage" << endl;
		ost << "\\section*{Substructures of dimension two}" << endl;
		ost << "\\begin{enumerate}" << endl;
		for (orbit_idx = 0; orbit_idx < Semifields->nb_orbits; orbit_idx++) {


			if (f_v) {
				cout << "orbit " << orbit_idx << " / " << Semifields->nb_orbits << ":" << endl;
			}
			Lint_vec_copy(Semifields->Rep_ith(orbit_idx), Sub->data1, Sub->SC->k);

			if (f_v) {
				cout << "before Sub->all_two_dimensional_subspaces" << endl;
			}
			Sub->all_two_dimensional_subspaces(
					Po2, verbose_level - 3);
			if (f_v) {
				cout << "after Sub->all_two_dimensional_subspaces" << endl;
			}

			Int_vec_copy(Po2, PO2 + orbit_idx * Sub->N2, Sub->N2);
			ost << "\\item" << endl;
			ost << orbit_idx << " / " << Semifields->nb_orbits << endl;
			ost << " has  type ";
			other::data_structures::tally C;

			C.init(Po2, Sub->N2, false, 0);
			ost << "$";
			C.print_bare_tex(ost, false /* f_backwards */);
			ost << "$";
			ost << "\\\\" << endl;
		}
		ost << "\\end{enumerate}" << endl;

		{
		string fname;

		fname = "Semifields_" + std::to_string(Descr->order) + "_2structure.tex";
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, PO2, Semifields->nb_orbits, Sub->N2);
		}
		FREE_int(Po2);
		FREE_int(PO2);

		L.foot(ost);
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "writing latex file " << fname << " done" << endl;
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::latex_report" << endl;
	}
}

void semifield_classify_with_substructure::generate_source_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify_with_substructure::generate_source_code" << endl;
	}
	string fname_base;

	fname_base = "semifields_" + std::to_string(Descr->order);

	if (f_v) {
		cout << "before Semifields->generate_source_code " << fname_base << endl;
		}

	Semifields->generate_source_code(fname_base, verbose_level);

	if (f_v) {
		cout << "after Semifields->generate_source_code " << fname_base << endl;
		}
	if (f_v) {
		cout << "semifield_classify_with_substructure::generate_source_code done" << endl;
	}
}



void semifield_classify_with_substructure::decomposition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify_with_substructure::decomposition" << endl;
	}
	semifield_flag_orbit_node *F1;
	invariant_relations::flag_orbits *F2;
	//int N1, N2, N3;
	int i1, i2, i3;
	int h1, h2;
	int po_up;
	int f_success;

	F1 = Sub->L3->Flag_orbits;
	F2 = Sub->Flag_orbits;
	//N1 = L2->nb_orbits;
	//N2 = Sub->L3->nb_orbits;
	//N3 = Semifields->nb_orbits;
	i1 = 8;
	i3 = 322;


#if 0
	cout << "F1:" << endl;
	cout << "Sub->L3->nb_flag_orbits=" << Sub->L3->nb_flag_orbits << endl;
	for (h1 = 0; h1 < Sub->L3->nb_flag_orbits; h1++) {
		if (F1[h1].f_fusion_node) {
			int f;

			f = F1[h1].fusion_with;
			po_up = F1[f].upstep_orbit;
		}
		else {
			po_up = F1[h1].upstep_orbit;
		}
		if (h1 > 5576873 - 139) {
			cout << h1 << " : " << F1[h1].downstep_primary_orbit << " : " << po_up << endl;
		}
	}

	cout << "F2:" << endl;
	for (h2 = 0; h2 < F2->nb_flag_orbits; h2++) {
		if (F2->Flag_orbit_node[h2].f_fusion_node) {
			int f;

			f = F2->Flag_orbit_node[h2].fusion_with;
			po_up = F2->Flag_orbit_node[f].upstep_primary_orbit;

		}
		else {
			po_up = F2->Flag_orbit_node[h2].upstep_primary_orbit;
		}
		cout << h2 << " : " << F2->Flag_orbit_node[h2].downstep_primary_orbit << " : " << po_up << endl;

	}
#endif

	cout << "searching for chains from i1=" << i1 << " to i3=" << i3 << endl;

	for (h1 = 0; h1 < Sub->L3->nb_flag_orbits; h1++) {
		if (F1[h1].downstep_primary_orbit != i1) {
			continue;
		}
		if (F1[h1].f_fusion_node) {
			int f;

			f = F1[h1].fusion_with;
			po_up = F1[f].upstep_orbit;
		}
		else {
			po_up = F1[h1].upstep_orbit;
		}
		i2 = po_up;
		f_success = false;
		//cout << "searching i1=" << i1 << " h1=" << h1 << " i2=" << i2 << endl;
		for (h2 = 0; h2 < F2->nb_flag_orbits; h2++) {
			if (F2->Flag_orbit_node[h2].downstep_primary_orbit != i2) {
				continue;
			}
			if (F2->Flag_orbit_node[h2].f_fusion_node) {
				int f;

				f = F2->Flag_orbit_node[h2].fusion_with;
				po_up = F2->Flag_orbit_node[f].upstep_primary_orbit;

			}
			else {
				po_up = F2->Flag_orbit_node[h2].upstep_primary_orbit;
			}
			if (po_up == i3) {
				f_success = true;
				//cout << i1 << " - " << h1 << " - " << i2 << " - " << h2 << " - " << i3 << endl;
			}
		}
		if (f_success) {
			long int pt;

			pt = F1[h1].pt;
			cout << "i1=" << i1 << " h1=" << h1 << " i2=" << i2 << " pt=" << pt << endl;
			if (F1[h1].f_fusion_node) {
				int f;

				f = F1[h1].fusion_with;
				cout << "f=" << f << endl;
				cout << "fusion elt:" << endl;
				Sub->SC->A->Group_element->element_print_quick(F1[h1].fusion_elt, cout);
			}
			for (h2 = 0; h2 < F2->nb_flag_orbits; h2++) {
				if (F2->Flag_orbit_node[h2].downstep_primary_orbit != i2) {
					continue;
				}
				if (F2->Flag_orbit_node[h2].f_fusion_node) {
					int f;

					f = F2->Flag_orbit_node[h2].fusion_with;
					po_up = F2->Flag_orbit_node[f].upstep_primary_orbit;

				}
				else {
					po_up = F2->Flag_orbit_node[h2].upstep_primary_orbit;
				}
				if (po_up == i3) {
					cout << i2 << " - " << h2 << " - " << i3 << endl;
				}
			}

		}
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::decomposition done" << endl;
	}
}

//##############################################################################
// global functions:
//##############################################################################


static void semifield_print_function_callback(
		std::ostream &ost, int orbit_idx,
		invariant_relations::classification_step *Step, void *print_function_data)
{
	semifield_substructure *Sub = (semifield_substructure *) print_function_data;
	semifield_classify *SC;
	semifield_classify_with_substructure *SCWS;
	other::l1_interfaces::latex_interface L;
	long int *R;
	long int a;
	int i, j;


	SC = Sub->SC;
	SCWS = Sub->SCWS;
	R = Step->Rep_ith(orbit_idx);
	for (j = 0; j < Step->representation_sz; j++) {
		a = R[j];
		SC->matrix_unrank(a, SC->test_Basis);
		ost << "$";
		ost << "\\left[";
		L.print_integer_matrix_tex(ost,
			SC->test_Basis, SC->k, SC->k);
		ost << "\\right]";
		ost << "$";
		if (j < Step->representation_sz - 1) {
			ost << ", " << endl;
		}
	}
	ost << "\\\\" << endl;
	for (j = 0; j < Step->representation_sz; j++) {
		a = R[j];
		SC->matrix_unrank(a, SC->test_Basis);
		ost << "$";
		Int_vec_print(ost, SC->test_Basis, SC->k2);
		ost << "$";
		ost << "\\\\" << endl;
	}
	if (SCWS->Descr->f_identify_semifields_from_file) {
		ost << "R\\'ua type: ";
		for (i = 0; i < SCWS->identify_semifields_from_file_m; i++) {
			for (j = 0; j < 6; j++) {
				if (SCWS->identify_semifields_from_file_Po[i * 6 + j] == orbit_idx) {
					ost << "$" << i << "." << j << "$; ";
				}
			}
		}
		ost << "\\\\" << endl;
	}

	ost << endl;
}




}}}


