/*
 * semifield_classify_with_substructure.cpp
 *
 *  Created on: May 15, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_classify_with_substructure::semifield_classify_with_substructure()
{
	t0 = 0;
	argc = 0;
	argv = NULL;
	f_poly = FALSE;
	poly = NULL;
	f_order = FALSE;
	order = 0;
	f_dim_over_kernel = FALSE;
	dim_over_kernel = 0;
	f_prefix = FALSE;
	prefix = "";
	f_orbits_light = FALSE;
	f_test_semifield = FALSE;
	test_semifield_data = NULL;
	f_identify_semifield = FALSE;
	identify_semifield_data = NULL;
	f_identify_semifields_from_file = FALSE;
	identify_semifields_from_file_fname = NULL;
	f_load_classification = FALSE;

	identify_semifields_from_file_Po = NULL;
	identify_semifields_from_file_m = 0;

	f_trace_record_prefix = FALSE;
	trace_record_prefix = NULL;
	f_FstLen = FALSE;
	fname_FstLen = NULL;
	f_Data = FALSE;
	fname_Data = NULL;

	p = e = e1 = n = k = q = k2 = 0;

	F = NULL;
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

}

void semifield_classify_with_substructure::read_arguments(
		int argc, const char **argv, int &verbose_level)
{
	int i;

	t0 = os_ticks();
	semifield_classify_with_substructure::argc = argc;
	semifield_classify_with_substructure::argv = argv;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = atoi(argv[++i]);
			cout << "-order " << order << endl;
			}
		else if (strcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = atoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
			}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
			}
		else if (strcmp(argv[i], "-orbits_light") == 0) {
			f_orbits_light = TRUE;
			cout << "-orbits_light " << endl;
			}
		else if (strcmp(argv[i], "-test_semifield") == 0) {
			f_test_semifield = TRUE;
			test_semifield_data = argv[++i];
			cout << "-test_semifield " << test_semifield_data << endl;
			}
		else if (strcmp(argv[i], "-identify_semifield") == 0) {
			f_identify_semifield = TRUE;
			identify_semifield_data = argv[++i];
			cout << "-identify_semifield " << identify_semifield_data << endl;
			}
		else if (strcmp(argv[i], "-identify_semifields_from_file") == 0) {
			f_identify_semifields_from_file = TRUE;
			identify_semifields_from_file_fname = argv[++i];
			cout << "-identify_semifields_from_file "
					<< identify_semifields_from_file_fname << endl;
			}
		else if (strcmp(argv[i], "-trace_record_prefix") == 0) {
			f_trace_record_prefix = TRUE;
			trace_record_prefix = argv[++i];
			cout << "-trace_record_prefix " << trace_record_prefix << endl;
			}
		else if (strcmp(argv[i], "-FstLen") == 0) {
			f_FstLen = TRUE;
			fname_FstLen = argv[++i];
			cout << "-FstLen " << fname_FstLen << endl;
			}
		else if (strcmp(argv[i], "-Data") == 0) {
			f_Data = TRUE;
			fname_Data = argv[++i];
			cout << "-Data " << fname_Data << endl;
			}
		else if (strcmp(argv[i], "-load_classification") == 0) {
			f_load_classification = TRUE;
			cout << "-load_classification " << endl;
			}
		}

}

void semifield_classify_with_substructure::init(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify_with_substructure::init" << endl;
	}
	number_theory_domain NT;

	NT.factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "dim_over_kernel does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = NT.i_power_j(p, e1);
		cout << "order=" << order << " n=" << n
			<< " k=" << k << " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "order=" << order << " n=" << n
			<< " k=" << k << " q=" << q << endl;
		}
	k2 = k * k;



	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);


	Sub = NEW_OBJECT(semifield_substructure);

	Sub->SCWS = this;
	Sub->start_column = 4;


	Sub->SC = NEW_OBJECT(semifield_classify);
	cout << "before SC->init" << endl;
	Sub->SC->init(argc, argv, order, n, k, F,
			4 /* MINIMUM(verbose_level - 1, 2) */);
	cout << "after SC->init" << endl;


	if (f_test_semifield) {
		long int *data = NULL;
		int data_len = 0;
		int i;

		cout << "f_test_semifield" << endl;
		lint_vec_scan(test_semifield_data, data, data_len);
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
	cout << "before L2->init" << endl;
	L2->init(Sub->SC, verbose_level);
	cout << "after L2->init" << endl;


#if 1
	cout << "before L2->compute_level_two" << endl;
	L2->compute_level_two(verbose_level);
	cout << "after L2->compute_level_two" << endl;
#else
	L2->read_level_info_file(verbose_level);
#endif

	Sub->L3 = NEW_OBJECT(semifield_lifting);

	cout << "before L3->init_level_three" << endl;
	Sub->L3->init_level_three(L2,
			Sub->SC->f_level_three_prefix, Sub->SC->level_three_prefix,
			verbose_level);
	cout << "after L3->init_level_three" << endl;

	cout << "before L3->recover_level_three_from_file" << endl;
	//L3->compute_level_three(verbose_level);
	Sub->L3->recover_level_three_from_file(TRUE /* f_read_flag_orbits */, verbose_level);
	cout << "after L3->recover_level_three_from_file" << endl;


	if (f_v) {
		cout << "semifield_classify_with_substructure::init done" << endl;
	}
}

void semifield_classify_with_substructure::read_data(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::read_data" << endl;
	}

	if (f_v) {
		cout << "before reading files " << fname_FstLen
			<< " and " << fname_Data << endl;
		}




	classify C;
	int mtx_n;
	int i, a;


	// We need 64 bit integers for the Data array
	// because semifields of order 64 need 6 x 6 matrices,
	// which need to be encoded using 36 bits.

	if (sizeof(long int) < 8) {
		cout << "sizeof(long int) < 8" << endl;
		exit(1);
		}
	Fio.int_matrix_read_csv(fname_FstLen,
		Sub->FstLen, Sub->nb_orbits_at_level_3, mtx_n, verbose_level);
	Sub->Len = NEW_int(Sub->nb_orbits_at_level_3);
	for (i = 0; i < Sub->nb_orbits_at_level_3; i++) {
		Sub->Len[i] = Sub->FstLen[i * 2 + 1];
		}
	Fio.lint_matrix_read_csv(fname_Data, Sub->Data,
			Sub->nb_solutions, Sub->data_size, verbose_level);


	if (f_v) {
		cout << "Read " << Sub->nb_solutions
			<< " solutions arising from "
			<< Sub->nb_orbits_at_level_3 << " orbits" << endl;
		}





	C.init(Sub->Len, Sub->nb_orbits_at_level_3, FALSE, 0);
	if (f_v) {
		cout << "classification of Len:" << endl;
		C.print_naked(TRUE);
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
	Non_unique_cases_go = NEW_int(nb_non_unique_cases);
	for (i = 0; i < nb_non_unique_cases; i++) {
		a = Non_unique_cases[i];
		Non_unique_cases_fst[i] = Sub->FstLen[2 * a + 0];
		Non_unique_cases_len[i] = Sub->FstLen[2 * a + 1];
		Non_unique_cases_go[i] =
			Sub->L3->Stabilizer_gens[a].group_order_as_int();
		}

	{
		classify C;

		C.init(Non_unique_cases_len, nb_non_unique_cases, FALSE, 0);
		if (f_v) {
			cout << "classification of Len amongst the non-unique cases:" << endl;
			C.print_naked(TRUE);
			cout << endl;
		}
	}
	{
	classify C;

	C.init(Non_unique_cases_go, nb_non_unique_cases, FALSE, 0);
	if (f_v) {
		cout << "classification of group orders amongst "
				"the non-unique cases:" << endl;
		C.print_naked(TRUE);
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

void semifield_classify_with_substructure::create_fname_for_classification(char *fname)
{
	sprintf(fname, "semifields_%d_classification.bin", Sub->SC->order);
}

void semifield_classify_with_substructure::create_fname_for_flag_orbits(char *fname)
{
	sprintf(fname, "semifields_%d_flag_orbits.bin", Sub->SC->order);
}

void semifield_classify_with_substructure::classify_semifields(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::classify_semifields" << endl;
	}

	Semifields = NEW_OBJECT(classification_step);

	Sub->do_classify(verbose_level);

	{
	char fname[1000];
	create_fname_for_classification(fname);
	{
		ofstream fp(fname, ios::binary);

		Semifields->write_file(fp, verbose_level);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	{
	char fname[1000];
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

void semifield_classify_with_substructure::load_classification(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::load_classification" << endl;
	}

	Semifields = NEW_OBJECT(classification_step);


	{
	char fname[1000];
	create_fname_for_classification(fname);
	{
		ifstream fp(fname, ios::binary);

		cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;
		Semifields->read_file(fp, Sub->SC->A, Sub->SC->AS, verbose_level);
	}
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::load_classification done" << endl;
	}
}

void semifield_classify_with_substructure::load_flag_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "semifield_classify_with_substructure::load_flag_orbits" << endl;
	}



	{
	char fname[1000];
	create_fname_for_flag_orbits(fname);
	{
		ifstream fp(fname, ios::binary);

		cout << "Reading file " << fname << " of size " << Fio.file_size(fname) << endl;
		Sub->Flag_orbits->read_file(fp, Sub->SC->A, Sub->SC->AS, verbose_level);
	}
	}


	if (f_v) {
		cout << "semifield_classify_with_substructure::load_flag_orbits done" << endl;
	}
}


void semifield_classify_with_substructure::identify_semifield(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifield" << endl;
	}

	if (f_identify_semifield) {
		long int *data = NULL;
		int data_len = 0;
		cout << "f_identify_semifield" << endl;
		lint_vec_scan(identify_semifield_data, data, data_len);
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

			lint_vec_print(cout, data_out, k);
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
				Sub->SC->A->element_print_quick(transporter, cout);
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
	file_io Fio;
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifields_from_file" << endl;
	}

	identify_semifields_from_file_Po = NULL;

	if (f_identify_semifields_from_file) {
		cout << "f_identify_semifield_from_file" << endl;

		long int *Data;
		int n;

		Fio.lint_matrix_read_csv(identify_semifields_from_file_fname, Data,
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
					Sub->SC->A->element_print_quick(transporter, cout);
					cout << endl;
					identify_semifields_from_file_Po[i * 6 + t] = po;
				}
				else {
					cout << "The given semifield cannot be identified" << endl;
					identify_semifields_from_file_Po[i * 6 + t] = -1;
				}
			}

		}
		char fname[1000];

		strcpy(fname, identify_semifields_from_file_fname);
		chop_off_extension(fname);
		sprintf(fname + strlen(fname), "_identification.csv");
		Fio.int_matrix_write_csv(fname, identify_semifields_from_file_Po,
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
	file_io Fio;
	int i;

	if (f_v) {
		cout << "semifield_classify_with_substructure::latex_report" << endl;
	}
	char title[1000];
	char author[1000];
	char fname[1000];
	sprintf(title, "Isotopy classes of semifields of order %d", order);
	sprintf(author, "Anton Betten");
	sprintf(fname, "Semifields_%d.tex", order);

	if (f_v) {
		cout << "writing latex file " << fname << endl;
		}

	{
		ofstream fp(fname);
		latex_interface L;


		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title,
			author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);


		int *Go;


		classify C;

		Go = NEW_int(Semifields->nb_orbits);
		for (i = 0; i < Semifields->nb_orbits; i++) {
			Go[i] = Semifields->Orbit[i].gens->group_order_as_int();
		}

		C.init(Go, Semifields->nb_orbits, FALSE, 0);

		fp << "\\section*{Summary}" << endl;
		fp << "Classification by stabilizer order:\\\\" << endl;
		fp << "$$" << endl;
		C.print_array_tex(fp, TRUE /*f_backwards */);
		fp << "$$" << endl;


		L2->print_representatives(fp, verbose_level);


		Semifields->print_latex(fp,
			title,
			TRUE /* f_print_stabilizer_gens */,
			TRUE,
			semifield_print_function_callback,
			Sub);

		if (f_identify_semifields_from_file) {
			fp << "\\clearpage" << endl;
			fp << "\\section*{Identification of Rua types}" << endl;
			fp << "The $i$-th row, $j$-th column of the table is the number $c$ "
					"of the isotopy class in the list above which corresponds "
					"to the $j$-th element in the Knuth orbit of the $i$-th "
					"class in the R\\'ua labeling.\\\\" << endl;
			//fp << "$$" << endl;
			print_integer_matrix_tex_block_by_block(fp,
					identify_semifields_from_file_Po,
					identify_semifields_from_file_m, 6, 40 /* block_width */);

			//print_integer_matrix_with_standard_labels_and_offset_tex(
			//	fp, identify_semifields_from_file_Po, identify_semifields_from_file_m, 6,
			//	1 /* m_offset */, 1 /* n_offset */);
			//fp << "$$" << endl;
		}

		int *Po2;
		int *PO2;
		int orbit_idx;

		Po2 = NEW_int(Sub->N2);
		PO2 = NEW_int(Semifields->nb_orbits * Sub->N2);

		fp << "\\clearpage" << endl;
		fp << "\\section*{Substructures of dimension two}" << endl;
		fp << "\\begin{enumerate}" << endl;
		for (orbit_idx = 0; orbit_idx < Semifields->nb_orbits; orbit_idx++) {


			if (f_v) {
				cout << "orbit " << orbit_idx << " / " << Semifields->nb_orbits << ":" << endl;
			}
			lint_vec_copy(Semifields->Rep_lint_ith(orbit_idx), Sub->data1, Sub->SC->k);

			Sub->all_two_dimensional_subspaces(
					Po2, verbose_level - 3);

			int_vec_copy(Po2, PO2 + orbit_idx * Sub->N2, Sub->N2);
			fp << "\\item" << endl;
			fp << orbit_idx << " / " << Semifields->nb_orbits << endl;
			fp << " has  type ";
			classify C;

			C.init(Po2, Sub->N2, FALSE, 0);
			fp << "$";
			C.print_naked_tex(fp, FALSE /* f_backwards */);
			fp << "$";
			fp << "\\\\" << endl;
		}
		fp << "\\end{enumerate}" << endl;

		{
		char fname[1000];

		sprintf(fname, "Semifields_%d_2structure.tex", order);
		Fio.int_matrix_write_csv(fname, PO2, Semifields->nb_orbits, Sub->N2);
		}
		FREE_int(Po2);
		FREE_int(PO2);

		L.foot(fp);
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
		cout << "semifield_classify_with_substructure::generate_"
				"source_code" << endl;
	}
	char fname_base[1000];
	sprintf(fname_base, "semifields_%d", order);

	if (f_v) {
		cout << "before Semifields->generate_source_code " << fname_base << endl;
		}

	Semifields->generate_source_code(fname_base,
			verbose_level);

	if (f_v) {
		cout << "after Semifields->generate_source_code " << fname_base << endl;
		}
	if (f_v) {
		cout << "semifield_classify_with_substructure::generate_"
				"source_code done" << endl;
	}
}





void semifield_print_function_callback(ostream &ost, int orbit_idx,
		classification_step *Step, void *print_function_data)
{
	semifield_substructure *Sub = (semifield_substructure *) print_function_data;
	semifield_classify *SC;
	semifield_classify_with_substructure *SCWS;
	long int *R;
	long int a;
	int i, j;


	SC = Sub->SC;
	SCWS = Sub->SCWS;
	R = Step->Rep_lint_ith(orbit_idx);
	for (j = 0; j < Step->representation_sz; j++) {
		a = R[j];
		SC->matrix_unrank(a, SC->test_Basis);
		ost << "$";
		ost << "\\left[";
		print_integer_matrix_tex(ost,
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
		int_vec_print(ost, SC->test_Basis, SC->k2);
		ost << "$";
		ost << "\\\\" << endl;
	}
	if (SCWS->f_identify_semifields_from_file) {
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




}}


