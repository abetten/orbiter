/*
 * create_code.cpp
 *
 *  Created on: Aug 10, 2022
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


create_code::create_code()
{
	description = NULL;

	//std::string label_txt;
	//std::string label_tex;

	f_field = FALSE;
	F = NULL;

	genma = NULL;
	checkma = NULL;
	n = 0;
	nmk = 0;
	k = 0;
	d = 0;

	Create_BCH_code = NULL;

}

create_code::~create_code()
{
}

void create_code::init(
		create_code_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::init" << endl;
	}
	create_code::description = description;

	// preprocessing stage:

	if (description->f_field) {
		if (f_v) {
			cout << "create_code::init f_field" << endl;
		}

		f_field = TRUE;


		F = Get_object_of_type_finite_field(description->field_label);

	}

	// main stage:

	if (description->f_linear_code_through_generator_matrix) {
		if (f_v) {
			cout << "create_code::init f_linear_code_through_generator_matrix" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		int nb_rows, nb_cols;

		Get_matrix(
				description->linear_code_through_generator_matrix_label_genma,
				genma, nb_rows, nb_cols);

		n = nb_cols;
		k = nb_rows;
		nmk = n - k;


		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}

		char str[1000];

		snprintf(str, sizeof(str), "by_genma_n%d_k%d", n, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "by\\_genma\\_n%d\\_k%d", n, k);
		label_tex.assign(str);


		if (f_v) {
			cout << "create_code::init f_linear_code_through_generator_matrix done" << endl;
		}
	}
	else if (description->f_linear_code_from_projective_set) {
		if (f_v) {
			cout << "create_code::init f_linear_code_from_projective_set" << endl;
			cout << "create_code::init nmk = " << description->linear_code_from_projective_set_nmk << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		long int *v;
		int sz;


		Get_lint_vector_from_label(
				description->linear_code_from_projective_set_set,
				v, sz, verbose_level);

		if (f_v) {
			cout << "create_code::init projective set:" << endl;
			Lint_vec_print(cout, v, sz);
			cout << endl;
		}

		nmk = description->linear_code_from_projective_set_nmk;

		n = sz;
		k = n - nmk;

		int *Col;
		int i, j;


		checkma = NEW_int(nmk * n);
		Col = NEW_int(nmk);

		for (j = 0; j < n; j++) {

			F->PG_element_unrank_modified(Col, 1, nmk, v[j]);
			for (i = 0; i < nmk; i++) {
				checkma[i * n + j] = Col[i];
			}
		}

		if (f_v) {
			cout << "create_code::init checkma from projective set:" << endl;
			Int_matrix_print(checkma, nmk, n);
		}



		create_genma_from_checkma(verbose_level);

		char str[1000];

		snprintf(str, sizeof(str), "proj_set_n%d_k%d", n, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "proj\\_set\\_n%d\\_k%d", n, k);
		label_tex.assign(str);


		FREE_int(Col);



		if (f_v) {
			cout << "create_code::init f_linear_code_from_from_projective_set done" << endl;
		}
	}
	else if (description->f_linear_code_by_columns_of_parity_check) {
		if (f_v) {
			cout << "create_code::init f_linear_code_by_columns_of_parity_check" << endl;
			cout << "linear_code_by_columns_of_parity_check_nmk="
					<< description->linear_code_by_columns_of_parity_check_nmk << endl;
			cout << "linear_code_by_columns_of_parity_check_set="
					<< description->linear_code_by_columns_of_parity_check_set << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		coding_theory::coding_theory_domain Codes;

		long int *set;
		int sz;


		Get_lint_vector_from_label(
				description->linear_code_by_columns_of_parity_check_set,
				set, sz, verbose_level);



		n = sz;
		nmk = description->linear_code_by_columns_of_parity_check_nmk;
		k = n - nmk;

		Codes.do_linear_code_through_columns_of_parity_check(
				F,
				n,
				set,
				nmk /*k*/,
				genma,
				verbose_level);

		if (f_v) {
			cout << "create_code::init genma:" << endl;
			Int_matrix_print(genma, k, n);
		}

		FREE_lint(set);

		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}


		if (f_v) {
			cout << "create_code::init f_linear_code_by_columns_of_parity_check done" << endl;
		}
	}
	else if (description->f_first_order_Reed_Muller) {
		if (f_v) {
			cout << "create_code::init f_first_order_Reed_Muller" << endl;
			cout << "m = " << description->first_order_Reed_Muller_m << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}


#if 0
		CODE_RM_3_1_GENMA="\
		11111111\
		01010101\
		00110011\
		00001111"

		CODE_RM_4_1_GENMA="\
		1111111111111111\
		0101010101010101\
		0011001100110011\
		0000111100001111\
		0000000011111111"
#endif

		int m;
		int i;
		long int a;
		coding_theory::coding_theory_domain Codes;
		number_theory::number_theory_domain NT;
		long int *v;

		m = description->first_order_Reed_Muller_m;

		n = NT.i_power_j(2, m);
		if (f_v) {
			cout << "create_code::init f_first_order_Reed_Muller" << endl;
			cout << "n = " << n << endl;
		}

		v = NEW_lint(n);
		a = 1;
		for (i = 0; i < n; i++) {
			v[i] = a;
			a += 2;
		}

		nmk = n - m - 1;
		k = n - nmk;

		Codes.do_linear_code_through_columns_of_parity_check(
				F,
				n,
				v,
				nmk /*k*/,
				genma,
				verbose_level);

		if (f_v) {
			cout << "create_code::init genma:" << endl;
			Int_matrix_print(genma, k, n);
		}

		FREE_lint(v);

		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}


		char str[1000];

		snprintf(str, sizeof(str), "RM_m%d", m);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "RM\\_m%d", m);
		label_tex.assign(str);



		if (f_v) {
			cout << "create_code::init f_first_order_Reed_Muller done" << endl;
		}
	}
	else if (description->f_BCH) {
		if (f_v) {
			cout << "create_code::init f_BCH" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}


		n = description->BCH_n;
		d = description->BCH_d;

		Create_BCH_code = NEW_OBJECT(coding_theory::create_BCH_code);

		Create_BCH_code->init(F,
				n,
				d,
				verbose_level);

		k = Create_BCH_code->k;
		nmk = n - k;

		genma = NEW_int(k * n);

		Int_vec_copy(Create_BCH_code->Genma, genma, k * n);

		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}


		char str[1000];

		snprintf(str, sizeof(str), "BCH_n%d_d%d", n, d);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "BCH\\_n%d\\_d%d", n, d);
		label_tex.assign(str);

		Create_BCH_code->do_report(verbose_level);


		if (f_v) {
			cout << "create_code::init f_BCH done" << endl;
		}
	}
	else if (description->f_Reed_Solomon) {
		if (f_v) {
			cout << "create_code::init f_Reed_Solomon" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		//int Reed_Solomon_n;
		//int Reed_Solomon_d;

		if (f_v) {
			cout << "create_code::init f_Reed_Solomon done" << endl;
		}
	}
	else if (description->f_Gilbert_Varshamov) {
		if (f_v) {
			cout << "create_code::init f_Gilbert_Varshamov" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		n = description->Gilbert_Varshamov_n;
		k = description->Gilbert_Varshamov_k;
		d = description->Gilbert_Varshamov_d;

		coding_theory::coding_theory_domain Coding;


		Coding.make_gilbert_varshamov_code(
				n, k, d,
				F,
				genma, checkma,
				verbose_level);

		nmk = n - k;

		char str[1000];

		snprintf(str, sizeof(str), "GV_n%d_k%d_d%d", n, k, d);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "GV\\_n%d\\_k%d\\_d%d", n, k, d);
		label_tex.assign(str);

		if (f_v) {
			cout << "create_code::init f_Gilbert_Varshamov done" << endl;
		}
	}

	int i;

	for (i = 0; i < description->Modifications.size(); i++) {

		if (f_v) {
			cout << "create_code::init applying modification:" << endl;
		}
		description->Modifications[i].apply(this, verbose_level);
	}


	if (f_v) {
		cout << "create_code::init we have created the following code:" << endl;

		if (n < 100) {
			cout << "genma:" << endl;
			Int_matrix_print(genma, k, n);

			cout << "checkma:" << endl;
			Int_matrix_print(checkma, nmk, n);
		}
		else {
			cout << "Too big to print." << endl;
		}

		cout << "n=" << n << endl;
		cout << "k=" << k << endl;
		cout << "d=" << d << endl;
		cout << "label_txt=" << label_txt << endl;
		cout << "label_tex=" << label_tex << endl;
	}

	if (f_v) {
		cout << "create_code::init label_txt = " << label_txt << endl;
		cout << "create_code::init label_tex = " << label_tex << endl;
		cout << "create_code::init done" << endl;
	}
}

void create_code::dual_code(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::dual_code" << endl;
	}

	int k1, nmk1;
	int *genma1;
	int *checkma1;

	k1 = nmk;
	nmk1 = k;
	genma1 = checkma;
	checkma1 = genma;

	nmk = nmk1;
	k = k1;
	genma = genma1;
	checkma = checkma1;

	label_txt.append("_dual");
	label_tex.append("\\_dual");


	if (f_v) {
		cout << "create_code::dual_code done" << endl;
	}
}


void create_code::export_magma(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_magma n=" << n << " k=" << k << endl;
	}

	{
		ofstream ost(fname);
		int i, j, a;

		ost << "K<w> := GF(" << F->q << ");" << endl;
		ost << "V := VectorSpace(K, " << n << ");" << endl;
		ost << "C := LinearCode(sub<V |" << endl;
		for (i = 0; i < k; i++) {
			ost << "[";
			for (j = 0; j < n; j++) {
				a = genma[i * n + j];
				if (F->e == 1) {
					ost << a;
				}
				else {
					if (a <= 1) {
						ost << a;
					}
					else {
						ost << "w^" << F->log_alpha(a);
					}
				}
				if (j < n - 1) {
					ost << ",";
				}
			}
			ost << "]";
			if (i < k - 1) {
				ost << "," << endl;
			}
			else {
				ost << ">);" << endl;
			}
		}
	}

	if (f_v) {
		cout << "create_code::export_magma done" << endl;
	}
}

void create_code::create_genma_from_checkma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma n=" << n << " nmk=" << nmk << endl;
	}
	int *M;
	int rk;

	M = NEW_int(n * n);

	Int_vec_copy(checkma, M, nmk * n);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma before F->Linear_algebra->perp_standard" << endl;
	}

	rk = F->Linear_algebra->perp_standard(n, nmk, M, 0 /*verbose_level*/);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma after F->Linear_algebra->perp_standard" << endl;
	}


	if (rk != nmk) {
		cout << "create_code::create_genma_from_checkma warning: rk != nmk. Adjusting for that." << endl;
		//exit(1);
	}
	nmk = rk;
	k = n - nmk;

	genma = NEW_int(k * n);
	Int_vec_copy(M + nmk * n, genma, k * n);

	FREE_int(M);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma done" << endl;
	}

}


void create_code::create_checkma_from_genma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma n=" << n << " k=" << k << endl;
	}
	int *M;
	int rk;


	M = NEW_int(n * n);


	Int_vec_copy(genma, M, k * n);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma before F->Linear_algebra->perp_standard" << endl;
	}

	rk = F->Linear_algebra->perp_standard(n, k, M, 0 /*verbose_level*/);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma after F->Linear_algebra->perp_standard" << endl;
	}


	if (rk != k) {
		cout << "create_code::create_checkma_from_genma warning: rk != k. Adjusting for that." << endl;
		//exit(1);
	}
	k = rk;
	nmk = n - k;

	checkma = NEW_int(nmk * n);
	Int_vec_copy(M + k * n, checkma, nmk * n);

	FREE_int(M);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma done" << endl;
	}

}

void create_code::export_codewords(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_codewords" << endl;
	}

	number_theory::number_theory_domain NT;
	coding_theory::coding_theory_domain Code;
	long int *codewords;
	long int N;

	N = NT.i_power_j(2, k);

	codewords = NEW_lint(N);

	Code.codewords_affine(F, n, k,
			genma, // [k * n]
			codewords, // q^k
			verbose_level);



	if (f_v) {
		cout << "Codewords : ";
		Lint_vec_print_fully(cout, codewords, N);
		cout << endl;
	}

	orbiter_kernel_system::file_io Fio;

	Fio.lint_matrix_write_csv(fname, codewords, N, 1);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "create_code::export_codewords done" << endl;
	}

}

void create_code::export_codewords_by_weight(std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_codewords_by_weight" << endl;
	}

	number_theory::number_theory_domain NT;
	coding_theory::coding_theory_domain Code;
	long int *codewords;
	long int N;

	N = NT.i_power_j(2, k);

	codewords = NEW_lint(N);

	Code.codewords_affine(F, n, k,
			genma, // [k * n]
			codewords, // q^k
			verbose_level);


	data_structures::sorting Sorting;


	Sorting.lint_vec_heapsort(codewords, N);

	if (f_v) {
		cout << "Codewords : ";
		Lint_vec_print_fully(cout, codewords, N);
		cout << endl;
	}


	int *Wt;
	int *word;
	int h, i, w;

	word = NEW_int(n);
	Wt = NEW_int(N);

	geometry::geometry_global Gg;

	if (f_v) {
		cout << "create_code::export_codewords_by_weight computing weights" << endl;
	}
	for (h = 0; h < N; h++) {
		Gg.AG_element_unrank(F->q, word, 1, n, codewords[h]);
		w = 0;
		for (i = 0; i < n; i++) {
			if (word[i]) {
				w++;
			}
		}
		Wt[h] = w;
	}


	data_structures::tally *T;
	data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;

	T = NEW_OBJECT(data_structures::tally);
	T->init(Wt, N, FALSE, 0 /* verbose_level */);

	SoS = T->get_set_partition_and_types(types, nb_types, 0 /* verbose_level */);

	for (i = 0; i < nb_types; i++) {

		if (f_v) {
			cout << "create_code::export_codewords_by_weight we found "
					<< SoS->Set_size[i] << " codewords of weight " << types[i] << endl;
		}

		long int *codewords_of_weight;
		long int nb;
		int j, a;

		nb = SoS->Set_size[i];
		codewords_of_weight = NEW_lint(nb);

		for (j = 0; j < nb; j++) {
			a = SoS->Sets[i][j];
			codewords_of_weight[j] = codewords[a];
		}


		orbiter_kernel_system::file_io Fio;
		string fname;
		char str[1000];

		fname.assign(fname_base);
		snprintf(str, sizeof(str), "_of_weight_%d.csv", types[i]);
		fname.append(str);


		Fio.lint_matrix_write_csv(fname, codewords_of_weight, nb, 1);

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

		FREE_lint(codewords_of_weight);

	}


	if (f_v) {
		cout << "create_code::export_codewords_by_weight done" << endl;
	}

}


void create_code::export_genma(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_genma" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, genma, k, n);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "create_code::export_genma done" << endl;
	}

}

void create_code::export_checkma(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_checkma" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, checkma, nmk, n);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "create_code::export_checkma done" << endl;
	}

}

void create_code::weight_enumerator(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::weight_enumerator" << endl;
	}

	coding_theory::coding_theory_domain Codes;



	Codes.do_weight_enumerator(F,
			genma, k, n,
			FALSE /* f_normalize_from_the_left */,
			FALSE /* f_normalize_from_the_right */,
			verbose_level);

	if (f_v) {
		cout << "create_code::weight_enumerator done" << endl;
	}

}

void create_code::fixed_code(
	long int *perm, int n,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::fixed_code n = " << n << endl;
	}
	if (n != create_code::n) {
		cout << "create_code::fixed_code the length of the permutation does not match" << endl;
		exit(1);
	}

	long int t0, t1, dt;
	long int N;
	int *msg;
	int *word;
	geometry::geometry_global Gg;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "create_code::fixed_code" << endl;
	}
	N = Gg.nb_AG_elements(k, F->q);
	if (f_v) {
		cout << N << " messages" << endl;
	}
	msg = NEW_int(k);
	word = NEW_int(n);
	int h, i, a, j, b, cnt;
	vector<long int> V;

	cnt = 0;
	for (h = 0; h < N; h++) {
		if ((h % ONE_MILLION) == 0) {
			t1 = Os.os_ticks();
			dt = t1 - t0;
			cout << setw(10) << h << " / " << setw(10) << N << " : ";
			Os.time_check_delta(cout, dt);
			cout << endl;
		}
		Gg.AG_element_unrank(F->q, msg, 1, k, h);
		F->Linear_algebra->mult_vector_from_the_left(msg, genma, word, k, n);
		for (i = 0; i < n; i++) {
			a = word[i];
			j = perm[i];
			b = word[j];
			if (a != b) {
				break;
			}
		}
		if (i == n) {
			V.push_back(h);
			Int_vec_print(cout, word, n);
			cout << endl;
			cnt++;
		}
	}
	if (f_v) {
		cout << "create_code::fixed_code we found " << cnt << " fixed words" << endl;
	}
	int *M;
	int rk;

	M = NEW_int(cnt * n);
	for (i = 0; i < N; i++) {
		Gg.AG_element_unrank(F->q, msg, 1, k, V[i]);
		F->Linear_algebra->mult_vector_from_the_left(msg, genma, word, k, n);
		Int_vec_copy(word, M + i * n, n);
	}
	rk = F->Linear_algebra->Gauss_easy(M, cnt, n);
	if (f_v) {
		cout << "create_code::fixed_code The fix subcode has dimension " << rk << endl;
		Int_matrix_print(M, rk, n);
		cout << endl;
		Int_vec_print_fully(cout, M, rk * n);
		cout << endl;
	}



	if (f_v) {
		cout << "create_code::fixed_code done" << endl;
	}
}





}}}
