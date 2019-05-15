/*
 * isomorph_testing.cpp
 *
 *  Created on: May 8, 2019
 *      Author: betten
 *
 *  originally created on March 7, 2018
 *
 */


#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;




class trace_record;
class semifield_substructure;
class semifield_classify_with_substructure;

// global data:

int t0; // the system time when the program started


void save_trace_record(
		trace_record *T,
		int f_trace_record_prefix, const char *trace_record_prefix,
		int iso, int f, int po, int so, int N);
void semifield_print_function_callback(ostream &ost, int i,
		classification_step *Step, void *print_function_data);


#define MAX_FILES 1000

class trace_record {
public:
	int coset;
	int trace_po;
	int f_skip;
	int solution_idx;
	int nb_sol;
	int go;
	int pos;
	int so;
	int orbit_len;
	int f2;
	trace_record();
	~trace_record();
};

class semifield_classify_with_substructure {
public:

	int argc;
	const char **argv;

	int f_poly;
	const char *poly;
	int f_order;
	int order;
	int f_dim_over_kernel;
	int dim_over_kernel;
	int f_prefix;
	const char *prefix;
	int f_orbits_light;
	int f_test_semifield;
	const char *test_semifield_data;
	int f_identify_semifield;
	const char *identify_semifield_data;
	int f_identify_semifields_from_file;
	const char *identify_semifields_from_file_fname;

	int *identify_semifields_from_file_Po;
	int identify_semifields_from_file_m;


	int f_trace_record_prefix;
	const char *trace_record_prefix;
	int f_FstLen;
	const char *fname_FstLen;
	int f_Data;
	const char *fname_Data;

	int p, e, e1, n, k, q, k2;

	finite_field *F;
	semifield_substructure *Sub;
	semifield_level_two *L2;


	int nb_existing_cases;
	int *Existing_cases;
	int *Existing_cases_fst;
	int *Existing_cases_len;


	int nb_non_unique_cases;
	int *Non_unique_cases;
	int *Non_unique_cases_fst;
	int *Non_unique_cases_len;
	int *Non_unique_cases_go;


	classification_step *Semifields;


	semifield_classify_with_substructure();
	~semifield_classify_with_substructure();
	void read_arguments(int argc, const char **argv, int &verbose_level);
	void init(int verbose_level);
	void read_data(int verbose_level);
	void classify_semifields(int verbose_level);
	void identify_semifield(int verbose_level);
	void identify_semifields_from_file(int verbose_level);
	void latex_report(int verbose_level);
	void generate_source_code(int verbose_level);
};

class semifield_substructure {
public:
	semifield_classify_with_substructure *SCWS;
	semifield_classify *SC;
	semifield_lifting *L3;
	grassmann *Gr;
	int *Non_unique_cases_with_non_trivial_group;
	int nb_non_unique_cases_with_non_trivial_group;

	int *Need_orbits_fst;
	int *Need_orbits_len;

	trace_record *TR;
	int N;
	int f;
	long int *Data;
	int nb_solutions;
	int data_size;
	int start_column;
	int *FstLen;
	int *Len;
	int nb_orbits_at_level_3;
	int nb_orb_total; // = sum_i Nb_orb[i]
	orbit_of_subspaces ***All_Orbits; // [nb_non_unique_cases_with_non_trivial_group]
	int *Nb_orb; // [nb_non_unique_cases_with_non_trivial_group]
		// Nb_orb[i] is the number of orbits in All_Orbits[i]
	int **Orbit_idx; // [nb_non_unique_cases_with_non_trivial_group]
		// Orbit_idx[i][j] = b
		// means that the j-th solution of Nontrivial case i belongs to orbt All_Orbits[i][b]
	int **Position; // [nb_non_unique_cases_with_non_trivial_group]
		// Position[i][j] = a
		// means that the j-th solution of Nontrivial case i is the a-th element in All_Orbits[i][b]
		// where Orbit_idx[i][j] = b
	int *Fo_first; // [nb_orbits_at_level_3]
	int nb_flag_orbits;
	flag_orbits *Flag_orbits; // [nb_flag_orbits]
	int *f_processed; // [nb_flag_orbits]
	int nb_processed;
	long int *data1;
	long int *data2;
	int *Basis1;
	int *Basis2;
	int *B;
	int *v1;
	int *v2;
	int *v3;
	int *transporter1;
	int *transporter2;
	int *transporter3;
	int *Elt1;
	vector_ge *coset_reps;

	semifield_substructure();
	~semifield_substructure();
	void compute_cases(
			int nb_non_unique_cases,
			int *Non_unique_cases, int *Non_unique_cases_go,
			int verbose_level);
	void compute_orbits(int verbose_level);
	void compute_flag_orbits(int verbose_level);
	void do_classify(int verbose_level);
	void loop_over_all_subspaces(int verbose_level);
	int find_semifield_in_table(
		int po,
		long int *given_data,
		int &idx,
		int verbose_level);
	int identify(long int *data,
			int &rk, int &trace_po, int &fo, int &po,
			int *transporter,
			int verbose_level);
};




int main(int argc, const char **argv)
{
	int verbose_level = 0;
	semifield_classify_with_substructure SCWS;


	t0 = os_ticks();

	SCWS.read_arguments(argc, argv, verbose_level);

	if (!SCWS.f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}


	SCWS.init(verbose_level);

	SCWS.read_data(verbose_level);

	SCWS.Sub->compute_orbits(verbose_level);

	SCWS.Sub->compute_flag_orbits(verbose_level);

	SCWS.classify_semifields(verbose_level);

	SCWS.identify_semifield(verbose_level);


	SCWS.identify_semifields_from_file(
				verbose_level);


	SCWS.latex_report(
				verbose_level);


	SCWS.generate_source_code(verbose_level);



#if 0
	cout << "before freeing Gr" << endl;
	FREE_OBJECT(Sub.Gr);
	cout << "before freeing transporter1" << endl;
	FREE_int(Sub.transporter1);
	FREE_int(Sub.transporter2);
	FREE_int(Sub.transporter3);
	cout << "before freeing Basis1" << endl;
	FREE_int(Sub.Basis1);
	FREE_int(Sub.Basis2);
	FREE_int(Sub.B);
	cout << "before freeing Flag_orbits" << endl;
	FREE_OBJECT(Sub.Flag_orbits);

	cout << "before freeing L3" << endl;
	FREE_OBJECT(Sub.L3);
	cout << "before freeing L2" << endl;
	FREE_OBJECT(L2);
	cout << "before freeing SC" << endl;
	FREE_OBJECT(Sub.SC);
	cout << "before freeing F" << endl;
	FREE_OBJECT(F);
	cout << "before leaving scope" << endl;
	}
	cout << "after leaving scope" << endl;

#endif



	the_end(t0);
}




semifield_classify_with_substructure::semifield_classify_with_substructure()
{
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

void semifield_classify_with_substructure::read_arguments(int argc, const char **argv, int &verbose_level)
{
	int i;

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
			cout << "-identify_semifield_from_file " << identify_semifields_from_file_fname << endl;
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

void semifield_classify_with_substructure::classify_semifields(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_classify_with_substructure::classify_semifields" << endl;
	}

	Semifields = NEW_OBJECT(classification_step);

	Sub->do_classify(verbose_level);

	if (f_v) {
		cout << "semifield_classify_with_substructure::classify_semifields done" << endl;
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
		Fio.int_matrix_write_csv(fname, identify_semifields_from_file_Po, identify_semifields_from_file_m, 6);
	}
	if (f_v) {
		cout << "semifield_classify_with_substructure::identify_semifields_from_file done" << endl;
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
	sprintf(title, "Semifields of order %d", order);
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

		fp << "Classification by stabilizer order:\\\\" << endl;
		fp << "$$" << endl;
		C.print_array_tex(fp, TRUE /*f_backwards */);
		fp << "$$" << endl;

		Semifields->print_latex(fp,
			title,
			TRUE /* f_print_stabilizer_gens */,
			TRUE,
			semifield_print_function_callback,
			&Sub);

		if (f_identify_semifields_from_file) {
			fp << "\\clearpage" << endl;
			fp << "Identification:\\\\" << endl;
			//fp << "$$" << endl;
			print_integer_matrix_tex_block_by_block(fp,
					identify_semifields_from_file_Po,
					identify_semifields_from_file_m, 6, 40 /* block_width */);

			//print_integer_matrix_with_standard_labels_and_offset_tex(
			//	fp, identify_semifields_from_file_Po, identify_semifields_from_file_m, 6,
			//	1 /* m_offset */, 1 /* n_offset */);
			//fp << "$$" << endl;
		}

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
		cout << "semifield_classify_with_substructure::generate_source_code" << endl;
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
		cout << "semifield_classify_with_substructure::generate_source_code done" << endl;
	}
}

semifield_substructure::semifield_substructure()
{
	SCWS = NULL;
	SC = NULL;
	L3 = NULL;
	Gr = NULL;
	Non_unique_cases_with_non_trivial_group = NULL;
	nb_non_unique_cases_with_non_trivial_group = 0;

	Need_orbits_fst = NULL;
	Need_orbits_len = NULL;

	TR = NULL;
	N = 0;
	f = 0;
	Data = NULL;
	nb_solutions = 0;
	data_size = 0;
	start_column = 0;
	FstLen = NULL;
	Len = NULL;
	nb_orbits_at_level_3 = 0;
	nb_orb_total = 0;
	All_Orbits = NULL;
	Nb_orb = NULL;
	Orbit_idx = NULL;
	Position = NULL;
	Fo_first = NULL;
	nb_flag_orbits = 0;
	Flag_orbits = NULL;
	f_processed = NULL;
	nb_processed = 0;
	data1 = NULL;
	data2 = NULL;
	Basis1 = NULL;
	Basis2 = NULL;
	B = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	transporter1 = NULL;
	transporter2 = NULL;
	transporter3 = NULL;
	Elt1 = NULL;
	coset_reps = NULL;

}

semifield_substructure::~semifield_substructure()
{

}

void semifield_substructure::compute_cases(
		int nb_non_unique_cases, int *Non_unique_cases, int *Non_unique_cases_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, sum;

	if (f_v) {
		cout << "computing non-unique cases with non-trivial group:" << endl;
		}

	Non_unique_cases_with_non_trivial_group = NEW_int(nb_non_unique_cases);
	nb_non_unique_cases_with_non_trivial_group = 0;
	for (i = 0; i < nb_non_unique_cases; i++) {
		a = Non_unique_cases[i];
		if (Non_unique_cases_go[i] > 1) {
			Non_unique_cases_with_non_trivial_group
				[nb_non_unique_cases_with_non_trivial_group++] = a;
			}
		}
	if (f_v) {
		cout << "There are " << nb_non_unique_cases_with_non_trivial_group
			<< " cases with more than one solution and with a "
				"non-trivial group" << endl;
		cout << "They are:" << endl;
		int_matrix_print(Non_unique_cases_with_non_trivial_group,
			nb_non_unique_cases_with_non_trivial_group / 10 + 1, 10);
		}


	Need_orbits_fst = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	Need_orbits_len = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	for (i = 0; i < nb_non_unique_cases_with_non_trivial_group; i++) {
		a = Non_unique_cases_with_non_trivial_group[i];
		Need_orbits_fst[i] = FstLen[2 * a + 0];
		Need_orbits_len[i] = FstLen[2 * a + 1];
		}


	sum = 0;
	for (i = 0; i < nb_non_unique_cases_with_non_trivial_group; i++) {
		sum += Need_orbits_len[i];
		}
	{
	classify C;

	C.init(Need_orbits_len,
			nb_non_unique_cases_with_non_trivial_group, FALSE,
			0);
	if (f_v) {
		cout << "classification of Len amongst the need orbits cases:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}
	}
	if (f_v) {
		cout << "For a total of " << sum << " semifields" << endl;
		}
}


void semifield_substructure::compute_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	int a, o, fst, len;

	if (f_v) {
		cout << "semifield_substructure::compute_orbits" << endl;
		}
	if (f_v) {
		cout << "computing all orbits:" << endl;
		}



	All_Orbits =
			new orbit_of_subspaces **[nb_non_unique_cases_with_non_trivial_group];
	Nb_orb = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	Position = NEW_pint(nb_non_unique_cases_with_non_trivial_group);
	Orbit_idx = NEW_pint(nb_non_unique_cases_with_non_trivial_group);

	nb_orb_total = 0;
	for (o = 0; o < nb_non_unique_cases_with_non_trivial_group; o++) {



		a = Non_unique_cases_with_non_trivial_group[o];

		fst = Need_orbits_fst[o];
		len = Need_orbits_len[o];

		All_Orbits[o] = new orbit_of_subspaces *[len];

		if (f_v) {
			cout << "case " << o << " / "
				<< nb_non_unique_cases_with_non_trivial_group
				<< " with " << len
				<< " semifields" << endl;
			}

		int *f_reached;
		int *position;
		int *orbit_idx;

		f_reached = NEW_int(len);
		position = NEW_int(len);
		orbit_idx = NEW_int(len);

		int_vec_zero(f_reached, len);
		int_vec_mone(position, len);

		int cnt, f;

		cnt = 0;

		for (f = 0; f < len; f++) {
			if (f_reached[f]) {
				continue;
				}
			orbit_of_subspaces *Orb;
			long int *input_data;


			input_data = Data + (fst + f) * data_size + start_column;

			if (f_v) {
				cout << "case " << o << " / "
					<< nb_non_unique_cases_with_non_trivial_group
					<< " is original case " << a << " at "
					<< fst << " with " << len
					<< " semifields. Computing orbit of semifield " << f << " / " << len << endl;
				cout << "Orbit rep "
					<< f << ":" << endl;
				lint_vec_print(cout, input_data, SC->k);
				cout << endl;
				}
			if (f_vvv) {
				cout << "The stabilizer is:" << endl;
				L3->Stabilizer_gens[a].print_generators_ost(cout);
			}

			SC->compute_orbit_of_subspaces(input_data,
				&L3->Stabilizer_gens[a],
				Orb,
				verbose_level - 4);
			if (f_v) {
				cout << "case " << o << " / "
					<< nb_non_unique_cases_with_non_trivial_group
					<< " is original case " << a << " at "
					<< fst << " with " << len
					<< " semifields. Orbit of semifield " << f << " / " << len << " has length "
					<< Orb->used_length << endl;
				}

			int idx, g, c;

			c = 0;
			for (g = 0; g < len; g++) {
				if (f_reached[g]) {
					continue;
					}
				if (FALSE) {
					cout << "testing subspace " << g << " / " << len << ":" << endl;
				}
				if (Orb->find_subspace_lint(
						Data + (fst + g) * data_size + start_column,
						idx, 0 /*verbose_level*/)) {
					f_reached[g] = TRUE;
					position[g] = idx;
					orbit_idx[g] = cnt;
					c++;
					}
				}
			if (c != Orb->used_length) {
				cout << "c != Orb->used_length" << endl;
				cout << "Orb->used_length=" << Orb->used_length << endl;
				cout << "c=" << c << endl;
				exit(1);
				}
			All_Orbits[o][cnt] = Orb;

			cnt++;

			}

		if (f_v) {
			cout << "case " << o << " / "
				<< nb_non_unique_cases_with_non_trivial_group
				<< " with " << len << " semifields done, there are "
				<< cnt << " orbits in this case." << endl;
			}

		Nb_orb[o] = cnt;
		nb_orb_total += cnt;

		Position[o] = position;
		Orbit_idx[o] = orbit_idx;

		FREE_int(f_reached);



		}

	if (f_v) {
		cout << "Number of orbits:" << endl;
		for (o = 0; o < nb_non_unique_cases_with_non_trivial_group; o++) {
			cout << o << " : " << Nb_orb[o] << endl;
			}
		cout << "Total number of orbits = " << nb_orb_total << endl;
		}
	if (f_v) {
		cout << "semifield_substructure::compute_orbits done" << endl;
		}
}


void semifield_substructure::compute_flag_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int o, idx, h, fst, g;
	sorting Sorting;

	if (f_v) {
		cout << "semifield_substructure::compute_flag_orbits" << endl;
		}
	if (f_v) {
		cout << "Counting number of flag orbits:" << endl;
		}
	nb_flag_orbits = 0;
	for (o = 0; o < nb_orbits_at_level_3; o++) {

		if (FALSE) {
			cout << "orbit " << o << " number of semifields = "
				<< Len[o] << " group order = "
				<< L3->Stabilizer_gens[o].group_order_as_int() << endl;
			}
		if (Len[o] == 0) {
			}
		else if (Len[o] == 1) {
			nb_flag_orbits += 1;
			}
		else if (L3->Stabilizer_gens[o].group_order_as_int() == 1) {
			nb_flag_orbits += Len[o];
			}
		else {
			if (!Sorting.int_vec_search(
				Non_unique_cases_with_non_trivial_group,
				nb_non_unique_cases_with_non_trivial_group, o, idx)) {
				cout << "cannot find orbit " << o
						<< " in the list " << endl;
				exit(1);
				}
			if (f_vv) {
				cout << "Found orbit " << o
						<< " at position " << idx << endl;
				}
			nb_flag_orbits += Nb_orb[idx];
			}

		} // next o
	if (f_v) {
		cout << "nb_flag_orbits = " << nb_flag_orbits << endl;
		}


	if (f_v) {
		cout << "Computing Flag_orbits:" << endl;
		}


	Fo_first = NEW_int(nb_orbits_at_level_3);

	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init_lint(
		SC->A, SC->AS,
		nb_orbits_at_level_3 /* nb_primary_orbits_lower */,
		SC->k /* pt_representation_sz */,
		nb_flag_orbits /* nb_flag_orbits */,
		verbose_level);


	h = 0;
	for (o = 0; o < nb_orbits_at_level_3; o++) {

		long int *data;

		Fo_first[o] = h;
		fst = FstLen[2 * o + 0];

		if (Len[o] == 0) {
			// nothing to do here
		}
		else if (Len[o] == 1) {
			data = Data +
					(fst + 0) * data_size + start_column;
			Flag_orbits->Flag_orbit_node[h].init_lint(
					Flag_orbits,
				h /* flag_orbit_index */,
				o /* downstep_primary_orbit */,
				0 /* downstep_secondary_orbit */,
				1 /* downstep_orbit_len */,
				FALSE /* f_long_orbit */,
				data /* int *pt_representation */,
				&L3->Stabilizer_gens[o],
				0 /*verbose_level - 2 */);
			h++;
		}
		else if (L3->Stabilizer_gens[o].group_order_as_int() == 1) {
			for (g = 0; g < Len[o]; g++) {
				data = Data +
						(fst + g) * data_size + start_column;
				Flag_orbits->Flag_orbit_node[h].init_lint(
						Flag_orbits,
					h /* flag_orbit_index */,
					o /* downstep_primary_orbit */,
					g /* downstep_secondary_orbit */,
					1 /* downstep_orbit_len */,
					FALSE /* f_long_orbit */,
					data /* int *pt_representation */,
					&L3->Stabilizer_gens[o],
					0 /*verbose_level - 2*/);
				h++;
			}
		}
		else {
			if (!Sorting.int_vec_search(
					Non_unique_cases_with_non_trivial_group,
					nb_non_unique_cases_with_non_trivial_group, o, idx)) {
				cout << "cannot find orbit " << o << " in the list " << endl;
				exit(1);
			}
			if (FALSE) {
				cout << "Found orbit " << o
					<< " at position " << idx << endl;
			}
			for (g = 0; g < Nb_orb[idx]; g++) {
				orbit_of_subspaces *Orb;
				strong_generators *gens;
				longinteger_object go;

				Orb = All_Orbits[idx][g];
				data = Orb->Subspaces_lint[Orb->position_of_original_subspace];
				L3->Stabilizer_gens[o].group_order(go);
				gens = Orb->stabilizer_orbit_rep(
					go /* full_group_order */, 0 /*verbose_level - 1*/);
				Flag_orbits->Flag_orbit_node[h].init_lint(
						Flag_orbits,
					h /* flag_orbit_index */,
					o /* downstep_primary_orbit */,
					g /* downstep_secondary_orbit */,
					Orb->used_length /* downstep_orbit_len */,
					FALSE /* f_long_orbit */,
					data /* int *pt_representation */,
					gens,
					0 /*verbose_level - 2*/);
				h++;
			}
		}


	}
	if (h != nb_flag_orbits) {
		cout << "h != nb_flag_orbits" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Finished initializing flag orbits. "
				"Number of flag orbits = " << nb_flag_orbits << endl;
	}
	if (f_v) {
		cout << "semifield_substructure::compute_flag_orbits done" << endl;
		}
}


void semifield_substructure::do_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_substructure::do_classify" << endl;
		}
	if (f_v) {
		cout << "Computing classification:" << endl;
	}

	int depth0 = 3;





	int po, so;
	longinteger_object go;
	int f_skip = FALSE;
	int i;


	Basis1 = NEW_int(SC->k * SC->k2);
	Basis2 = NEW_int(SC->k * SC->k2);
	B = NEW_int(SC->k2);
	Gr = NEW_OBJECT(grassmann);
	transporter1 = NEW_int(SC->A->elt_size_in_int);
	transporter2 = NEW_int(SC->A->elt_size_in_int);
	transporter3 = NEW_int(SC->A->elt_size_in_int);


	Gr->init(SC->k, depth0, SC->F, 0 /* verbose_level */);
	N = Gr->nb_of_subspaces(0 /* verbose_level */);
	data1 = NEW_lint(SC->k);
	data2 = NEW_lint(SC->k);
	v1 = NEW_int(SC->k);
	v2 = NEW_int(SC->k);
	v3 = NEW_int(SC->k);



	f_processed = NEW_int(nb_flag_orbits);
	int_vec_zero(f_processed, nb_flag_orbits);
	nb_processed = 0;

	Elt1 = NEW_int(SC->A->elt_size_in_int);



	SC->A->group_order(go);

	SCWS->Semifields->init_lint(SC->A, SC->AS,
			nb_flag_orbits, SC->k, go, verbose_level);


	Flag_orbits->nb_primary_orbits_upper = 0;

	for (f = 0; f < nb_flag_orbits; f++) {


		double progress;

		if (f_processed[f]) {
			continue;
		}

		progress = ((double) nb_processed * 100. ) /
				(double) nb_flag_orbits;

		if (f_v) {
			time_check(cout, t0);
			cout << " : Defining new orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< nb_flag_orbits << " progress="
				<< progress << "% "
				"nb semifields = " << Flag_orbits->nb_primary_orbits_upper << endl;

		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;



		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		lint_vec_copy(
				Flag_orbits->Pt_lint + f * Flag_orbits->pt_representation_sz,
				data1, SC->k);
		if (f_v) {
			cout << "data1=";
			lint_vec_print(cout, data1, SC->k);
			cout << endl;
		}

		strong_generators *Aut_gens;
		longinteger_object go;

		Aut_gens = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		coset_reps = NEW_OBJECT(vector_ge);
		coset_reps->init(SC->A);


		for (i = 0; i < SC->k; i++) {
			SC->matrix_unrank(data1[i], Basis1 + i * SC->k2);
		}
		if (f_v) {
			cout << "Basis1=" << endl;
			int_matrix_print(Basis1, SC->k, SC->k2);
			SC->basis_print(Basis1, SC->k);
		}
		f_skip = FALSE;
		for (i = 0; i < SC->k; i++) {
			v3[i] = Basis1[2 * SC->k2 + i * SC->k + 0];
		}
		if (!SC->F->is_unit_vector(v3, SC->k, SC->k - 1)) {
			cout << "flag orbit " << f << " / "
					<< nb_flag_orbits
					<< " 1st col of third matrix is = ";
			int_vec_print(cout, v3, SC->k);
			cout << " which is not the (k-1)-th unit vector, "
					"so we skip" << endl;
			f_skip = TRUE;
		}

		if (f_skip) {
			if (f_v) {
				cout << "flag orbit " << f << " / "
						<< nb_flag_orbits
					<< ", first vector is not the unit vector, "
					"so we skip" << endl;
			}
			f_processed[f] = TRUE;
			nb_processed++;
			continue;
		}
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces" << endl;
		}

		//trace_record *TR;

		TR = NEW_OBJECTS(trace_record, N);


		if (f_v) {
			time_check(cout, t0);
			cout << " : flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces, "
				"before loop_over_all_subspaces" << endl;
		}


		loop_over_all_subspaces(
				verbose_level - 3);


		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces, "
				"after loop_over_all_subspaces" << endl;
		}


		save_trace_record(
				TR,
				SCWS->f_trace_record_prefix, SCWS->trace_record_prefix,
				Flag_orbits->nb_primary_orbits_upper,
			f, po, so, N);


		FREE_OBJECTS(TR);

		int cl;
		longinteger_object go1, Cl, ago, ago1;
		longinteger_domain D;

		Aut_gens->group_order(go);
		cl = coset_reps->len;
		Cl.create(cl);
		D.mult(go, Cl, ago);
		if (f_v) {
			cout << "Semifield "
					<< Flag_orbits->nb_primary_orbits_upper
					<< ", Ago = starter * number of cosets = " << ago
					<< " = " << go << " * " << Cl
					<< " created from flag orbit " << f << " / "
				<< nb_flag_orbits << " progress="
				<< progress << "%" << endl;
		}

		strong_generators *Stab;

		Stab = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", semifield isotopy class " << Flag_orbits->nb_primary_orbits_upper <<
				"computing stabilizer" << endl;
		}
		Stab->init_group_extension(Aut_gens,
				coset_reps, cl /* index */,
				verbose_level);
		Stab->group_order(ago1);
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", semifield isotopy class " << Flag_orbits->nb_primary_orbits_upper <<
				" computing stabilizer done, order = " <<  ago1 << endl;
		}


		SCWS->Semifields->Orbit[Flag_orbits->nb_primary_orbits_upper].init_lint(
				SCWS->Semifields,
			Flag_orbits->nb_primary_orbits_upper,
			Stab, data1, verbose_level);

		FREE_OBJECT(Aut_gens);

		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;



	} // next f


	SCWS->Semifields->nb_orbits = Flag_orbits->nb_primary_orbits_upper;


	if (f_v) {
		cout << "Computing classification done, we found "
				<< SCWS->Semifields->nb_orbits
				<< " semifields" << endl;
		time_check(cout, t0);
		cout << endl;
	}
	if (f_v) {
		cout << "semifield_substructure::do_classify done" << endl;
		}
}

void semifield_substructure::loop_over_all_subspaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	finite_field *F;
	int rk, i, f2;
	int k, k2;
	int f_skip;
	int trace_po;
	int ret;
	sorting Sorting;


#if 0
	if (f == 1) {
		verbose_level += 10;
		f_v = f_vv = f_vvv = TRUE;
		cout << "CASE 1, STARTS HERE" << endl;
	}
#endif

	if (f_v) {
		cout << "loop_over_all_subspaces" << endl;
	}


	k = SC->k;
	k2 = SC->k2;
	F = SC->F;

	for (rk = 0; rk < N; rk++) {

		trace_record *T;

		T = TR + rk;

		T->coset = rk;


		if (f_vv) {
			cout << "flag orbit " << f << " / "
				<< nb_flag_orbits << ", subspace "
				<< rk << " / " << N << ":" << endl;
		}

		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data1[i], Basis1 + i * k2);
		}
		if (f_vvv) {
			SC->basis_print(Basis1, k);
		}


		// unrank the subspace:
		Gr->unrank_int_here_and_extend_basis(B, rk,
				0 /* verbose_level */);

		// multiply the matrices to get the matrices
		// adapted to the subspace:
		// the first three matrices are the generators
		// for the subspace.
		F->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
				0 /* verbose_level */);


		if (f_vvv) {
			cout << "base change matrix B=" << endl;
			int_matrix_print_bitwise(B, k, k);

			cout << "Basis2 = B * Basis1 (before trace)=" << endl;
			int_matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		if (f_vv) {
			cout << "before trace_to_level_three" << endl;
		}
		ret = L3->trace_to_level_three(
			Basis2,
			k /* basis_sz */,
			transporter1,
			trace_po,
			verbose_level - 3);

		f_skip = FALSE;

		if (ret == FALSE) {
			cout << "flag orbit " << f << " / "
				<< nb_flag_orbits << ", subspace "
				<< rk << " / " << N << " : trace_to_level_three return FALSE" << endl;

			f_skip = TRUE;
		}

		T->trace_po = trace_po;

		if (f_vvv) {
			cout << "After trace, trace_po = "
					<< trace_po << endl;
			cout << "Basis2 (after trace)=" << endl;
			int_matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		for (i = 0; i < k; i++) {
			v1[i] = Basis2[0 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v2[i] = Basis2[1 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v3[i] = Basis2[2 * k2 + i * k + 0];
		}
		if (f_skip == FALSE) {
			if (!F->is_unit_vector(v1, k, 0) ||
					!F->is_unit_vector(v2, k, 1) ||
					!F->is_unit_vector(v3, k, k - 1)) {
				f_skip = TRUE;
			}
		}
		T->f_skip = f_skip;

		if (f_skip) {
			if (f_vv) {
				cout << "skipping this case " << trace_po
					<< " because pivot is not "
						"in the last row." << endl;
			}
		}
		else {

			F->Gauss_int_with_given_pivots(
				Basis2 + 3 * k2,
				FALSE /* f_special */,
				TRUE /* f_complete */,
				SC->desired_pivots + 3,
				k - 3 /* nb_pivots */,
				k - 3, k2,
				0 /*verbose_level - 2*/);
			if (f_vvv) {
				cout << "Basis2 after RREF(2)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);
			}

			for (i = 0; i < k; i++) {
				data2[i] = SC->matrix_rank(Basis2 + i * k2);
			}
			if (f_vvv) {
				cout << "data2=";
				lint_vec_print(cout, data2, k);
				cout << endl;
			}

			int solution_idx;

			if (f_vvv) {
				cout << "before find_semifield_in_table" << endl;
			}
			if (!find_semifield_in_table(
				trace_po,
				data2 /* given_data */,
				solution_idx,
				verbose_level)) {

				cout << "flag orbit " << f << " / "
					<< nb_flag_orbits << ", subspace "
					<< rk << " / " << N << ":" << endl;

				cout << "find_semifield_in_table returns FALSE" << endl;

				cout << "trace_po=" << trace_po << endl;
				cout << "data2=";
				lint_vec_print(cout, data2, k);
				cout << endl;

				cout << "Basis2 after RREF(2)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);

				exit(1);
			}


			T->solution_idx = solution_idx;
			T->nb_sol = Len[trace_po];

			if (f_vvv) {
				cout << "solution_idx=" << solution_idx << endl;
			}

			int go;
			go = L3->Stabilizer_gens[trace_po].group_order_as_int();

			T->go = go;
			T->pos = -1;
			T->so = -1;
			T->orbit_len = -1;
			T->f2 = -1;

			if (f_vv) {
				cout << "flag orbit " << f << " / "
					<< nb_flag_orbits << ", subspace "
					<< rk << " / " << N << " trace_po="
					<< trace_po << " go=" << go
					<< " solution_idx=" << solution_idx << endl;
			}


			if (go == 1) {
				if (f_vv) {
					cout << "This starter case has a trivial "
							"group order" << endl;
				}

				f2 = Fo_first[trace_po] + solution_idx;

				T->so = solution_idx;
				T->orbit_len = 1;
				T->f2 = f2;

				if (f2 == f) {
					if (f_vv) {
						cout << "We found an automorphism" << endl;
					}
					coset_reps->append(transporter1);
				}
				else {
					if (!f_processed[f2]) {
						if (f_vv) {
							cout << "We are identifying with "
									"flag orbit " << f2 << ", which is "
									"po=" << trace_po << " so="
									<< solution_idx << endl;
						}
						Flag_orbits->Flag_orbit_node[f2].f_fusion_node
							= TRUE;
						Flag_orbits->Flag_orbit_node[f2].fusion_with
							= f;
						Flag_orbits->Flag_orbit_node[f2].fusion_elt
							= NEW_int(SC->A->elt_size_in_int);
						SC->A->element_invert(
							transporter1,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt,
							0);
						f_processed[f2] = TRUE;
						nb_processed++;
						if (f_vv) {
							cout << "Flag orbit f2 = " << f2
								<< " has been fused with flag orbit "
								<< f << endl;
						}
					}
					else {
						if (f_vv) {
							cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
											"nothing to do here" << endl;
						}
					}
				}
			}
			else {
				if (Len[trace_po] == 1) {
					if (f_vv) {
						cout << "This starter case has only "
								"one solution" << endl;
					}
					f2 = Fo_first[trace_po] + 0;

					T->pos = -1;
					T->so = 0;
					T->orbit_len = 1;
					T->f2 = f2;


					if (f2 == f) {
						if (f_vv) {
							cout << "We found an automorphism" << endl;
						}
						coset_reps->append(transporter1);
					}
					else {
						if (!f_processed[f2]) {
							if (f_vv) {
								cout << "We are identifying with po="
										<< trace_po << " so=" << 0
										<< ", which is flag orbit "
										<< f2 << endl;
							}
							Flag_orbits->Flag_orbit_node[f2].f_fusion_node
								= TRUE;
							Flag_orbits->Flag_orbit_node[f2].fusion_with
								= f;
							Flag_orbits->Flag_orbit_node[f2].fusion_elt
								= NEW_int(SC->A->elt_size_in_int);
							SC->A->element_invert(transporter1,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
							f_processed[f2] = TRUE;
							nb_processed++;
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has been fused with "
										"flag orbit " << f << endl;
							}
						}
						else {
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
										"nothing to do here" << endl;
							}
						}
					}
				}
				else {
					// now we have a starter_case with more
					// than one solution and
					// with a non-trivial group.
					// Those cases are collected in
					// Non_unique_cases_with_non_trivial_group
					// [nb_non_unique_cases_with_non_trivial_group];

					int non_unique_case_idx, orbit_idx, position;
					orbit_of_subspaces *Orb;

					if (!Sorting.int_vec_search(
						Non_unique_cases_with_non_trivial_group,
						nb_non_unique_cases_with_non_trivial_group,
						trace_po, non_unique_case_idx)) {
						cout << "cannot find in Non_unique_cases_with_"
								"non_trivial_group array" << endl;
						exit(1);
					}
					orbit_idx = Orbit_idx[non_unique_case_idx][solution_idx];
					position = Position[non_unique_case_idx][solution_idx];
					Orb = All_Orbits[non_unique_case_idx][orbit_idx];
					f2 = Fo_first[trace_po] + orbit_idx;



					T->pos = position;
					T->so = orbit_idx;
					T->orbit_len = Orb->used_length;
					T->f2 = f2;



					if (f_vv) {
						cout << "orbit_idx = " << orbit_idx
								<< " position = " << position
								<< " f2 = " << f2 << endl;
					}
					Orb->get_transporter(position, transporter2,
							0 /*verbose_level */);
					SC->A->element_invert(transporter2, Elt1, 0);
					SC->A->element_mult(transporter1, Elt1,
							transporter3, 0);

					if (f2 == f) {
						if (f_vv) {
							cout << "We found an automorphism" << endl;
							}
						coset_reps->append(transporter3);
					}
					else {
						if (!f_processed[f2]) {
							if (f_vv) {
								cout << "We are identifying with po="
									<< trace_po << " so=" << solution_idx
									<< ", which is flag orbit "
									<< f2 << endl;
							}
							Flag_orbits->Flag_orbit_node[f2].f_fusion_node
								= TRUE;
							Flag_orbits->Flag_orbit_node[f2].fusion_with
								= f;
							Flag_orbits->Flag_orbit_node[f2].fusion_elt =
								NEW_int(SC->A->elt_size_in_int);
							SC->A->element_invert(transporter3,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
							f_processed[f2] = TRUE;
							nb_processed++;
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has been fused with flag "
									"orbit " << f << endl;
							}
						}
						else {
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
											"nothing to do here" << endl;
							}
						}
					}

				}
			}
		} // if !f_skip
	} // next rk
	if (f_v) {
		cout << "loop_over_all_subspaces done" << endl;
	}

}

int semifield_substructure::identify(long int *data,
		int &rk, int &trace_po, int &fo, int &po,
		int *transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	finite_field *F;
	int i, f2;
	int k, k2;
	int f_skip;
	//int trace_po;
	int ret;
	sorting Sorting;
	int solution_idx;



	if (f_v) {
		cout << "semifield_substructure::identify" << endl;
	}


	k = SC->k;
	k2 = SC->k2;
	F = SC->F;

	for (rk = 0; rk < N; rk++) {

		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data[i], Basis1 + i * k2);
		}
		if (f_vvv) {
			SC->basis_print(Basis1, k);
		}


		// unrank the subspace:
		Gr->unrank_int_here_and_extend_basis(B, rk,
				0 /* verbose_level */);

		// multiply the matrices to get the matrices
		// adapted to the subspace:
		// the first three matrices are the generators
		// for the subspace.
		F->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
				0 /* verbose_level */);


		if (f_vvv) {
			cout << "semifield_substructure::identify "
					"base change matrix B=" << endl;
			int_matrix_print_bitwise(B, k, k);

			cout << "semifield_substructure::identify "
					"Basis2 = B * Basis1 (before trace)=" << endl;
			int_matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		if (f_vv) {
			cout << "semifield_substructure::identify "
					"before trace_to_level_three" << endl;
		}
		ret = L3->trace_to_level_three(
			Basis2,
			k /* basis_sz */,
			transporter1,
			trace_po,
			verbose_level - 3);

		f_skip = FALSE;

		if (ret == FALSE) {
			cout << "semifield_substructure::identify trace_to_level_three return FALSE" << endl;

			f_skip = TRUE;
		}


		if (f_vvv) {
			cout << "semifield_substructure::identify After trace, trace_po = "
					<< trace_po << endl;
			cout << "semifield_substructure::identify Basis2 (after trace)=" << endl;
			int_matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		for (i = 0; i < k; i++) {
			v1[i] = Basis2[0 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v2[i] = Basis2[1 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v3[i] = Basis2[2 * k2 + i * k + 0];
		}
		if (f_skip == FALSE) {
			if (!F->is_unit_vector(v1, k, 0) ||
					!F->is_unit_vector(v2, k, 1) ||
					!F->is_unit_vector(v3, k, k - 1)) {
				f_skip = TRUE;
			}
		}

		if (f_skip) {
			if (f_vv) {
				cout << "semifield_substructure::identify skipping this case " << trace_po
					<< " because pivot is not "
						"in the last row." << endl;
			}
		}
		else {

			F->Gauss_int_with_given_pivots(
				Basis2 + 3 * k2,
				FALSE /* f_special */,
				TRUE /* f_complete */,
				SC->desired_pivots + 3,
				k - 3 /* nb_pivots */,
				k - 3, k2,
				0 /*verbose_level - 2*/);
			if (f_vvv) {
				cout << "semifield_substructure::identify "
						"Basis2 after RREF(2)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);
			}

			for (i = 0; i < k; i++) {
				data2[i] = SC->matrix_rank(Basis2 + i * k2);
			}
			if (f_vvv) {
				cout << "semifield_substructure::identify data2=";
				lint_vec_print(cout, data2, k);
				cout << endl;
			}

			//int solution_idx;

			if (f_vvv) {
				cout << "semifield_substructure::identify "
						"before find_semifield_in_table" << endl;
			}
			if (!find_semifield_in_table(
				trace_po,
				data2 /* given_data */,
				solution_idx,
				verbose_level)) {

				cout << "semifield_substructure::identify "
						"find_semifield_in_table returns FALSE" << endl;

				cout << "semifield_substructure::identify "
						"trace_po=" << trace_po << endl;
				cout << "semifield_substructure::identify data2=";
				lint_vec_print(cout, data2, k);
				cout << endl;

				cout << "semifield_substructure::identify "
						"Basis2 after RREF(2)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);

				return FALSE;
			}

			int go;
			go = L3->Stabilizer_gens[trace_po].group_order_as_int();

			//T->solution_idx = solution_idx;
			//T->nb_sol = Len[trace_po];
			if (go == 1) {
				if (f_vv) {
					cout << "This starter case has a trivial "
							"group order" << endl;
				}

				f2 = Fo_first[trace_po] + solution_idx;

				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}
			}
			else if (Len[trace_po] == 1) {
				f2 = Fo_first[trace_po] + 0;
				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}
			}
			else {
				// now we have a starter_case with more
				// than one solution and
				// with a non-trivial group.
				// Those cases are collected in
				// Non_unique_cases_with_non_trivial_group
				// [nb_non_unique_cases_with_non_trivial_group];

				int non_unique_case_idx, orbit_idx, position;
				orbit_of_subspaces *Orb;

				if (!Sorting.int_vec_search(
					Non_unique_cases_with_non_trivial_group,
					nb_non_unique_cases_with_non_trivial_group,
					trace_po, non_unique_case_idx)) {
					cout << "semifield_substructure::identify "
							"cannot find in Non_unique_cases_with_"
							"non_trivial_group array" << endl;
					exit(1);
				}
				orbit_idx = Orbit_idx[non_unique_case_idx][solution_idx];
				position = Position[non_unique_case_idx][solution_idx];
				Orb = All_Orbits[non_unique_case_idx][orbit_idx];
				f2 = Fo_first[trace_po] + orbit_idx;

				if (f_vv) {
					cout << "orbit_idx = " << orbit_idx
							<< " position = " << position
							<< " f2 = " << f2 << endl;
				}
				Orb->get_transporter(position, transporter2,
						0 /*verbose_level */);
				SC->A->element_invert(transporter2, Elt1, 0);
				SC->A->element_mult(transporter1, Elt1,
						transporter3, 0);
				SC->A->element_move(transporter3,
						transporter1, 0);
				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}

			} // go != 1

			po = Flag_orbits->Flag_orbit_node[fo].upstep_primary_orbit;


			if (f_vvv) {
				cout << "semifield_substructure::identify done "
						"solution_idx=" << solution_idx
						<< "trace_po=" << trace_po
						<< "f2=" << f2
						<< "fo=" << fo
						<< "po=" << po
						<< endl;
			}
			return TRUE;
		} // end else


	} // next rk

	if (f_v) {
		cout << "semifield_substructure::identify done" << endl;
	}
	return FALSE;
}

int semifield_substructure::find_semifield_in_table(
	int po,
	long int *given_data,
	int &idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fst, len, g;



	if (f_v) {
		cout << "find_semifield_in_table" << endl;
		}

	idx = -1;

	if (f_v) {
		cout << "searching for: ";
		lint_vec_print(cout, given_data, SC->k);
		cout << endl;
		}
	fst = FstLen[2 * po + 0];
	len = FstLen[2 * po + 1];
	if (f_v) {
		cout << "find_semifield_in_table po = " << po
				<< " len = " << len << endl;
		}
	// make sure that the first three integers
	// agree with what is stored
	// in the table for orbit po:
	if (len == 0) {
		cout << "find_semifield_in_table len == 0" << endl;
		return FALSE;
		}

	if (lint_vec_compare(Data + fst * data_size + start_column,
			given_data, 3)) {
		cout << "find_semifield_in_table the first three entries "
				"of given_data do not match with what "
				"is in Data" << endl;
		exit(1);
	}

	if (len == 1) {
		if (lint_vec_compare(Data + fst * data_size + start_column,
				given_data, SC->k)) {
			cout << "find_semifield_in_table len is 1 and the first six "
				"entries of given_data do not match with what "
				"is in Data" << endl;
			exit(1);
			}
		idx = 0;
	}
	else {
		for (g = 0; g < len; g++) {
			if (lint_vec_compare(Data + (fst + g) * data_size + start_column,
					given_data, SC->k) == 0) {
				idx = g;
				break;
				}
		}
		if (g == len) {
			cout << "find_semifield_in_table cannot find the "
					"semifield in the table" << endl;
			for (g = 0; g < len; g++) {
				cout << g << " : " << fst + g << " : ";
				lint_vec_print(cout, Data + (fst + g) * data_size + start_column, SC->k);
				cout << " : ";
				cout << lint_vec_compare(Data + (fst + g) * data_size + start_column,
						given_data, SC->k);
				cout << endl;
			}
			return FALSE;
		}
	}

	if (f_v) {
		cout << "find_semifield_in_table done, idx = " << idx << endl;
	}
	return TRUE;
}



trace_record::trace_record()
{
	coset = 0;
	trace_po = 0;
	f_skip = FALSE;
	solution_idx = -1;
	nb_sol = -1;
	go = -1;
	pos = -1;
	so = -1;
	orbit_len = 0;
	f2 = -1;
}

trace_record::~trace_record()
{

}




void save_trace_record(
		trace_record *T,
		int f_trace_record_prefix, const char *trace_record_prefix,
		int iso, int f, int po, int so, int N)
{
	int *M;
	int w = 10;
	char fname[1000];
	const char *column_label[] = {
		"coset",
		"trace_po",
		"f_skip",
		"solution_idx",
		"nb_sol",
		"go",
		"pos",
		"so",
		"orbit_len",
		"f2"
		};
	int i;
	file_io Fio;

	M = NEW_int(N * w);
	for (i = 0; i < N; i++) {
		M[i * w + 0] = T[i].coset;
		M[i * w + 1] = T[i].trace_po;
		M[i * w + 2] = T[i].f_skip;
		M[i * w + 3] = T[i].solution_idx;
		M[i * w + 4] = T[i].nb_sol;
		M[i * w + 5] = T[i].go;
		M[i * w + 6] = T[i].pos;
		M[i * w + 7] = T[i].so;
		M[i * w + 8] = T[i].orbit_len;
		M[i * w + 9] = T[i].f2;
		}

	if (f_trace_record_prefix) {
		sprintf(fname, "%strace_record_%03d_f%05d_po%d_so%d.csv", trace_record_prefix, iso, f, po, so);
	}
	else {
		sprintf(fname, "trace_record_%03d_f%05d_po%d_so%d.csv", iso, f, po, so);
	}
	Fio.int_matrix_write_csv_with_labels(fname, M, N, w, column_label);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}

void semifield_print_function_callback(ostream &ost, int i,
		classification_step *Step, void *print_function_data)
{
	semifield_substructure *Sub = (semifield_substructure *) print_function_data;
	semifield_classify *SC;
	long int *R;
	long int a;
	int j;


	SC = Sub->SC;
	R = Step->Rep_lint_ith(i);
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
	ost << endl;
}



