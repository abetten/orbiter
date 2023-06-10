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

	f_field = false;
	F = NULL;

	f_has_generator_matrix = false;
	genma = NULL;

	f_has_check_matrix = false;
	checkma = NULL;
	n = 0;
	nmk = 0;
	k = 0;
	d = 0;

	Create_BCH_code = NULL;
	Create_RS_code = NULL;

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

		f_field = true;


		F = Get_finite_field(description->field_label);

	}

	// main stage:

	if (description->f_generator_matrix) {
		if (f_v) {
			cout << "create_code::init f_generator_matrix" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		int nb_rows, nb_cols;

		Get_matrix(
				description->generator_matrix_label_genma,
				genma, nb_rows, nb_cols);

		f_has_generator_matrix = true;
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

		label_txt = "by_genma_n" + std::to_string(n) + "_k" + std::to_string(k);

		label_tex = "by\\_genma\\_n" + std::to_string(n) + "\\_k" + std::to_string(k);


		if (f_v) {
			cout << "create_code::init f_generator_matrix done" << endl;
		}
	}
	else if (description->f_basis) {
		if (f_v) {
			cout << "create_code::init f_basis" << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		long int *v;
		int sz;


		Get_lint_vector_from_label(
				description->basis_label,
				v, sz, verbose_level);

		if (f_v) {
			cout << "create_code::init using basis v=";
			Lint_vec_print(cout, v, sz);
			cout << endl;
		}

		int i;
		int nb_rows, nb_cols;
		geometry::geometry_global Gg;

		nb_rows = sz;
		nb_cols = description->basis_n;
		genma = NEW_int(nb_rows * nb_cols);

		if (f_v) {
			cout << "create_code::init nb_rows=" << nb_rows
					<< " nb_cols=" << nb_cols << endl;
		}

		for (i = 0; i < nb_rows; i++) {
			Gg.AG_element_unrank(F->q,
					genma + i * nb_cols, 1, nb_cols, v[i]);
		}

		f_has_generator_matrix = true;
		n = nb_cols;
		k = nb_rows;
		nmk = n - k;


		if (f_v) {
			cout << "create_code::init "
					"before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init "
					"after create_checkma_from_genma" << endl;
		}

		label_txt = "by_basis_n" + std::to_string(n) + "_k" + std::to_string(k);

		label_tex = "by\\_basis\\_n" + std::to_string(n) + "\\_k" + std::to_string(k);


		if (f_v) {
			cout << "create_code::init f_basis done" << endl;
		}
	}
	else if (description->f_projective_set) {
		if (f_v) {
			cout << "create_code::init f_projective_set" << endl;
			cout << "create_code::init nmk = " << description->projective_set_nmk << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		long int *v;
		int sz;


		Get_lint_vector_from_label(
				description->projective_set_set,
				v, sz, verbose_level);

		if (f_v) {
			cout << "create_code::init projective set:" << endl;
			Lint_vec_print(cout, v, sz);
			cout << endl;
		}

		nmk = description->projective_set_nmk;

		n = sz;
		k = n - nmk;

		int *Col;
		int i, j;


		checkma = NEW_int(nmk * n);
		Col = NEW_int(nmk);

		for (j = 0; j < n; j++) {

			F->Projective_space_basic->PG_element_unrank_modified(
					Col, 1, nmk, v[j]);
			for (i = 0; i < nmk; i++) {
				checkma[i * n + j] = Col[i];
			}
		}
		f_has_check_matrix = true;

		if (f_v) {
			cout << "create_code::init checkma from projective set:" << endl;
			Int_matrix_print(checkma, nmk, n);
		}



		create_genma_from_checkma(verbose_level);

		label_txt = "proj_set_n" + std::to_string(n) + "_k" + std::to_string(k);

		label_tex = "proj\\_set\\_n" + std::to_string(n) + "\\_k" + std::to_string(k);


		FREE_int(Col);



		if (f_v) {
			cout << "create_code::init f_projective_set done" << endl;
		}
	}
	else if (description->f_columns_of_generator_matrix) {
		if (f_v) {
			cout << "create_code::init f_columns_of_generator_matrix" << endl;
			cout << "columns_of_generator_matrix_nmk="
					<< description->columns_of_generator_matrix_k << endl;
			cout << "columns_of_generator_matrix_set="
					<< description->columns_of_generator_matrix_set << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		coding_theory::coding_theory_domain Codes;

		long int *set;
		int sz;


		Get_lint_vector_from_label(
				description->columns_of_generator_matrix_set,
				set, sz, verbose_level);



		n = sz;
		k = description->columns_of_generator_matrix_k;
		nmk = n - k;

		Codes.do_linear_code_through_columns_of_generator_matrix(
				F,
				n,
				set,
				k,
				genma,
				verbose_level);
		f_has_generator_matrix = true;

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
	else if (description->f_Reed_Muller) {
		if (f_v) {
			cout << "create_code::init f_Reed_Muller" << endl;
			cout << "m = " << description->Reed_Muller_m << endl;
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

		m = description->Reed_Muller_m;

		n = NT.i_power_j(2, m);
		if (f_v) {
			cout << "create_code::init f_Reed_Muller" << endl;
			cout << "n = " << n << endl;
		}

		// create the column ranks:
		// step size is 2 so we always have a one in the least significant bit.
		v = NEW_lint(n);
		a = 1;
		for (i = 0; i < n; i++) {
			v[i] = a;
			a += 2;
		}

		//nmk = n - m - 1;
		//k = n - nmk
		k = m + 1;
		nmk = n - k;

		Codes.do_linear_code_through_columns_of_generator_matrix(
				F,
				n,
				v,
				k,
				genma,
				verbose_level);
		f_has_generator_matrix = true;

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


		label_txt = "RM_n" + std::to_string(m);

		label_txt = "RM\\_n" + std::to_string(m);



		if (f_v) {
			cout << "create_code::init f_Reed_Muller done" << endl;
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
		f_has_generator_matrix = true;

		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}


		label_txt = "BCH_n" + std::to_string(n) + "_d" + std::to_string(d);

		label_tex = "BCH\\_n" + std::to_string(n) + "\\_d" + std::to_string(d);




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

		n = description->Reed_Solomon_n;
		d = description->Reed_Solomon_d;

		Create_RS_code = NEW_OBJECT(coding_theory::create_RS_code);

		Create_RS_code->init(F,
				n,
				d,
				verbose_level);

		k = Create_RS_code->k;
		nmk = n - k;

		genma = NEW_int(k * n);

		Int_vec_copy(Create_RS_code->Genma, genma, k * n);
		f_has_generator_matrix = true;

		if (f_v) {
			cout << "create_code::init before create_checkma_from_genma" << endl;
		}
		create_checkma_from_genma(verbose_level);
		if (f_v) {
			cout << "create_code::init after create_checkma_from_genma" << endl;
		}


		label_txt = "RS_n" + std::to_string(n) + "_d" + std::to_string(d);

		label_tex = "RS\\_n" + std::to_string(n) + "\\_d" + std::to_string(d);


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


		if (f_v) {
			cout << "create_code::init "
					"before Coding.make_gilbert_varshamov_code" << endl;
		}
		Coding.make_gilbert_varshamov_code(
				n, k, d,
				F,
				genma, checkma,
				verbose_level);
		if (f_v) {
			cout << "create_code::init "
					"after Coding.make_gilbert_varshamov_code" << endl;
		}
		f_has_generator_matrix = true;
		f_has_check_matrix = true;

		nmk = n - k;

		label_txt = "GV_n" + std::to_string(n) + "_k" + std::to_string(k) + "_d" + std::to_string(d);

		label_tex = "GV\\_n" + std::to_string(n) + "\\_k" + std::to_string(k) + "\\_d" + std::to_string(d);

		if (f_v) {
			cout << "create_code::init f_Gilbert_Varshamov done" << endl;
		}
	}
	else if (description->f_long_code) {
		if (f_v) {
			cout << "create_code::init f_long_code" << endl;
		}

		if (!f_field) {
			cout << "please use option -field "
					"to specify the field of linearity" << endl;
			exit(1);
		}

		//int f_long_code;
		//int long_code_n;
		//std::vector<std::string> long_code_generators;


		int i;
		int nb_rows, nb_cols;
		geometry::geometry_global Gg;

		nb_rows = description->long_code_generators.size();
		nb_cols = description->long_code_n;
		genma = NEW_int(nb_rows * nb_cols);
		Int_vec_zero(genma, nb_rows * nb_cols);

		if (f_v) {
			cout << "create_code::init nb_rows=" << nb_rows
					<< " nb_cols=" << nb_cols << endl;
		}

		for (i = 0; i < nb_rows; i++) {
			long int *v;
			int sz;
			int h;


			Get_lint_vector_from_label(
					description->long_code_generators[i],
					v, sz, verbose_level);

			if (f_v) {
				cout << "create_code::init long_code row " << i << " v=";
				Lint_vec_print(cout, v, sz);
				cout << endl;
			}

			for (h = 0; h < sz; h++) {
				genma[i * nb_cols + v[h]] = 1;
			}

			FREE_lint(v);

			//Gg.AG_element_unrank(F->q, genma + i * nb_cols, 1, nb_cols, v[i]);
		}

		if (f_v) {
			cout << "create_code::init genma:" << endl;
			Int_matrix_print(genma, nb_rows, nb_cols);
		}

		f_has_generator_matrix = true;
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

		label_txt = "long_code_n" + std::to_string(n) + "_k" + std::to_string(k);

		label_tex = "long\\_code\\_n" + std::to_string(n) + "\\_k" + std::to_string(k);


		if (f_v) {
			cout << "create_code::init f_long_code done" << endl;
		}
	}
	else if (description->f_ttpA) {
		if (f_v) {
			cout << "create_code::init f_ttpA" << endl;
			cout << "create_code::init field = " << description->ttpA_field_label << endl;
		}

		if (!f_field) {
			cout << "please use option -field to specify the field of linearity" << endl;
			exit(1);
		}

		//long int *v;
		//int sz;


		field_theory::finite_field *FQ;

		FQ = Get_finite_field(description->ttpA_field_label);

		coding_theory::ttp_codes TTP;

		int nb_rows, nb_cols;

		TTP.twisted_tensor_product_codes(
			FQ,
			F /* Fq */,
			true /* f_construction_A */, true /* f_hyperoval */,
			false /* f_construction_B */,
			checkma, nb_rows, nb_cols,
			verbose_level);

		n = nb_cols;
		nmk = nb_rows;

		f_has_check_matrix = true;



		create_genma_from_checkma(verbose_level);

		label_txt = "ttpA_n" + std::to_string(n) + "_k" + std::to_string(k);

		label_tex = "ttpA\\_n" + std::to_string(n) + "\\_k" + std::to_string(k);




		if (f_v) {
			cout << "create_code::init f_ttpA done" << endl;
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
			if (f_has_generator_matrix) {
				cout << "genma:" << endl;
				Int_matrix_print(genma, k, n);
			}
			else {
				cout << "generator matrix is not available" << endl;
			}

			if (f_has_check_matrix) {
				cout << "checkma:" << endl;
				Int_matrix_print(checkma, nmk, n);
			}
			else {
				cout << "check matrix is not available" << endl;
			}
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

	int f_has_generator_matrix_save;
	int f_has_check_matrix_save;

	f_has_generator_matrix_save = f_has_generator_matrix;
	f_has_check_matrix_save = f_has_check_matrix;

	f_has_generator_matrix = f_has_check_matrix_save;
	f_has_check_matrix = f_has_generator_matrix_save;



	label_txt.append("_dual");
	label_tex.append("\\_dual");


	if (f_v) {
		cout << "create_code::dual_code done" << endl;
	}
}


void create_code::export_magma(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_magma n=" << n << " k=" << k << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::export_magma "
				"generator matrix is not available" << endl;
		exit(1);
	}

	interfaces::magma_interface M;

	if (f_v) {
		cout << "create_code::export_magma "
				"before M.export_linear_code" << endl;
	}
	M.export_linear_code(
			fname,
			F,
			genma, n, k,
			verbose_level);
	if (f_v) {
		cout << "create_code::export_magma "
				"after M.export_linear_code" << endl;
	}

	if (f_v) {
		cout << "create_code::export_magma done" << endl;
	}
}

void create_code::create_genma_from_checkma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma "
				"n=" << n << " nmk=" << nmk << endl;
	}
	int *M;
	int rk;

	if (!f_has_check_matrix) {
		cout << "create_code::create_genma_from_checkma "
				"does not have check matrix" << endl;
		exit(1);
	}

	M = NEW_int(n * n);

	Int_vec_copy(checkma, M, nmk * n);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma "
				"before F->Linear_algebra->perp_standard" << endl;
	}

	rk = F->Linear_algebra->perp_standard(n, nmk, M, 0 /*verbose_level*/);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma "
				"after F->Linear_algebra->perp_standard" << endl;
	}


	if (rk != nmk) {
		cout << "create_code::create_genma_from_checkma "
				"warning: rk != nmk. Adjusting for that." << endl;
		//exit(1);
	}
	nmk = rk;
	k = n - nmk;

	genma = NEW_int(k * n);
	Int_vec_copy(M + nmk * n, genma, k * n);
	f_has_generator_matrix = true;

	FREE_int(M);

	if (f_v) {
		cout << "create_code::create_genma_from_checkma done" << endl;
	}

}


void create_code::create_checkma_from_genma(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma "
				"n=" << n << " k=" << k << endl;
	}
	int *M;
	int rk;


	M = NEW_int(n * n);


	if (!f_has_generator_matrix) {
		cout << "create_code::create_checkma_from_genma "
				"does not have generator matrix" << endl;
		exit(1);
	}

	Int_vec_copy(genma, M, k * n);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma "
				"before F->Linear_algebra->perp_standard" << endl;
	}

	rk = F->Linear_algebra->perp_standard(n, k, M, 0 /*verbose_level*/);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma "
				"after F->Linear_algebra->perp_standard" << endl;
	}


	if (rk != k) {
		cout << "create_code::create_checkma_from_genma "
				"warning: rk != k. Adjusting for that." << endl;
		//exit(1);
	}
	k = rk;
	nmk = n - k;

	checkma = NEW_int(nmk * n);
	Int_vec_copy(M + k * n, checkma, nmk * n);
	f_has_check_matrix = true;

	FREE_int(M);

	if (f_v) {
		cout << "create_code::create_checkma_from_genma done" << endl;
	}

}

void create_code::export_codewords(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_codewords" << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::export_codewords "
				"generator matrix is not available" << endl;
		exit(1);
	}

	number_theory::number_theory_domain NT;
	coding_theory::coding_theory_domain Code;
	long int *codewords;
	long int N;

	N = NT.i_power_j(2, k);

	codewords = NEW_lint(N);

	if (f_v) {
		cout << "create_code::export_codewords "
				"before Code.codewords_affine" << endl;
	}
	Code.codewords_affine(F, n, k,
			genma, // [k * n]
			codewords, // q^k
			verbose_level);
	if (f_v) {
		cout << "create_code::export_codewords "
				"after Code.codewords_affine" << endl;
	}



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

void create_code::export_codewords_long(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_codewords_long" << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::export_codewords_long "
				"generator matrix is not available" << endl;
		exit(1);
	}

	number_theory::number_theory_domain NT;
	coding_theory::coding_theory_domain Code;
	int *codewords;
	long int N;

	if (f_v) {
		cout << "create_code::export_codewords_long "
				"before Code.codewords_table" << endl;
	}
	Code.codewords_table(F, n, k,
			genma, // [k * n]
			codewords, // q^k
			N,
			verbose_level);
	if (f_v) {
		cout << "create_code::export_codewords_long "
				"after Code.codewords_table" << endl;
	}



	if (f_v) {
		cout << "export_codewords_long : ";
		Int_matrix_print(codewords, N, n);
		cout << endl;
	}

	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, codewords, N, n);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_int(codewords);

	if (f_v) {
		cout << "create_code::export_codewords_long done" << endl;
	}

}


void create_code::export_codewords_by_weight(
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_codewords_by_weight" << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::export_codewords_by_weight "
				"generator matrix is not available" << endl;
		exit(1);
	}

	coding_theory::coding_theory_domain Code;

	long int *codewords;
	long int N;

	if (f_v) {
		cout << "create_code::export_codewords_by_weight "
				"before Code.make_codewords_sorted" << endl;
	}
	Code.make_codewords_sorted(F,
				n, k,
				genma, // [k * n]
				codewords, // q^k
				N,
				verbose_level);
	if (f_v) {
		cout << "create_code::export_codewords_by_weight "
				"after Code.make_codewords_sorted" << endl;
	}

	if (false) {
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
		cout << "create_code::export_codewords_by_weight "
				"computing weights" << endl;
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
	T->init(Wt, N, false, 0 /* verbose_level */);

	SoS = T->get_set_partition_and_types(types, nb_types, 0 /* verbose_level */);

	for (i = 0; i < nb_types; i++) {

		if (f_v) {
			cout << "create_code::export_codewords_by_weight we found "
					<< SoS->Set_size[i] << " codewords "
							"of weight " << types[i] << endl;
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


void create_code::export_genma(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_genma" << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::export_genma "
				"generator matrix is not available" << endl;
		exit(1);
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

void create_code::export_checkma(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_checkma" << endl;
	}

	if (!f_has_check_matrix) {
		cout << "create_code::export_checkma "
				"check matrix is not available" << endl;
		exit(1);
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

void create_code::export_checkma_as_projective_set(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::export_checkma_as_projective_set" << endl;
	}

	if (!f_has_check_matrix) {
		cout << "create_code::export_checkma_as_projective_set "
				"check matrix is not available" << endl;
		exit(1);
	}


	long int *Rk;
	int *v;
	int i, j, a;

	Rk = NEW_lint(n);
	v = NEW_int(nmk);
	for (j = 0; j < n; j++) {
		for (i = 0; i < nmk; i++) {
			a = checkma[i * n + j];
			v[i] = a;
		}
		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, nmk, Rk[j]);
	}
	orbiter_kernel_system::file_io Fio;

	Fio.lint_matrix_write_csv(fname, Rk, n, 1);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "create_code::export_checkma_as_projective_set done" << endl;
	}

}


void create_code::weight_enumerator(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::weight_enumerator" << endl;
	}

	if (!f_has_generator_matrix) {
		cout << "create_code::weight_enumerator "
				"generator matrix is not available" << endl;
		exit(1);
	}


	coding_theory::coding_theory_domain Codes;



	if (f_v) {
		cout << "create_code::weight_enumerator "
				"before Codes.do_weight_enumerator" << endl;
	}
	Codes.do_weight_enumerator(F,
			genma, k, n,
			false /* f_normalize_from_the_left */,
			false /* f_normalize_from_the_right */,
			verbose_level);
	if (f_v) {
		cout << "create_code::weight_enumerator "
				"after Codes.do_weight_enumerator" << endl;
	}

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


	if (!f_has_generator_matrix) {
		cout << "create_code::fixed_code generator matrix "
				"is not available" << endl;
		exit(1);
	}



	if (n != create_code::n) {
		cout << "create_code::fixed_code the length of "
				"the permutation does not match" << endl;
		exit(1);
	}

	coding_theory::coding_theory_domain Codes;

	int subcode_k;
	int *subcode_genma;


	if (f_v) {
		cout << "create_code::fixed_code "
				"before Codes.fixed_code" << endl;
	}
	Codes.fixed_code(
				F,
				n, k, genma,
				perm,
				subcode_genma, subcode_k,
				verbose_level);
	if (f_v) {
		cout << "create_code::fixed_code "
				"after Codes.fixed_code" << endl;
	}

	if (f_v) {
		cout << "create_code::fixed_code "
				"The fix subcode has dimension " << subcode_k << endl;
		Int_matrix_print(subcode_genma, subcode_k, n);
		cout << endl;
		Int_vec_print_fully(cout, subcode_genma, subcode_k * n);
		cout << endl;
	}


	if (f_v) {
		cout << "create_code::fixed_code done" << endl;
	}
}


void create_code::make_diagram(
		int f_embellish, int embellish_radius,
		int f_metric_balls, int radius_of_metric_ball,
		coding_theory::code_diagram *&Diagram,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::code_diagram" << endl;
	}



	long int *Words;
	long int nb_words;


	coding_theory::coding_theory_domain Code;


	if (f_v) {
		cout << "create_code::code_diagram "
				"before Code.make_codewords_sorted" << endl;
	}
	Code.make_codewords_sorted(F,
				n, k,
				genma, // [k * n]
				Words, // q^k
				nb_words,
				verbose_level);
	if (f_v) {
		cout << "create_code::code_diagram "
				"after Code.make_codewords_sorted" << endl;
	}

	if (false) {
		cout << "Codewords : ";
		Lint_vec_print_fully(cout, Words, nb_words);
		cout << endl;
	}




	Diagram = NEW_OBJECT(coding_theory::code_diagram);

	if (f_v) {
		cout << "create_code::code_diagram "
				"before Diagram->init" << endl;
	}

	Diagram->init(
			label_txt,
			Words, nb_words, n, verbose_level);

	if (f_v) {
		cout << "create_code::code_diagram "
				"after Diagram->init" << endl;
	}



	if (f_v) {
		cout << "create_code::code_diagram done" << endl;
	}

}


void create_code::polynomial_representation_of_boolean_function(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::polynomial_representation_of_boolean_function" << endl;
	}



	long int *Words;
	long int nb_words;


	coding_theory::coding_theory_domain Codes;


	if (f_v) {
		cout << "create_code::code_diagram "
				"before Codes.make_codewords_sorted" << endl;
	}
	Codes.make_codewords_sorted(F,
				n, k,
				genma, // [k * n]
				Words, // q^k
				nb_words,
				verbose_level);
	if (f_v) {
		cout << "create_code::polynomial_representation_of_boolean_function "
				"after Codes.make_codewords_sorted" << endl;
	}

	if (false) {
		cout << "Codewords : ";
		Lint_vec_print_fully(cout, Words, nb_words);
		cout << endl;
	}



	if (f_v) {
		cout << "create_code::polynomial_representation_of_boolean_function "
				"before Code.code_diagram" << endl;
	}
	Codes.polynomial_representation_of_boolean_function(
			F,
			label_txt,
			Words,
			nb_words, n,
			verbose_level);
	if (f_v) {
		cout << "create_code::polynomial_representation_of_boolean_function "
				"after Code.code_diagram" << endl;
	}


	if (f_v) {
		cout << "create_code::polynomial_representation_of_boolean_function done" << endl;
	}

}


void create_code::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::report" << endl;
	}


	string label;
	//coding_theory_domain Codes;
	l1_interfaces::latex_interface Li;
	orbiter_kernel_system::file_io Fio;

	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "_code_n%d_k%d_q%d.tex",
				n,
				k,
				F->q
				);
		fname.assign(label_txt);
		fname.append(str);

		snprintf(str, 1000, "Linear code");
		title.assign(str);



		{
			ofstream ost(fname);
			number_theory::number_theory_domain NT;


			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			report2(ost, verbose_level);

			L.foot(ost);


		}

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}


	if (f_v) {
		cout << "create_code::report done" << endl;
	}
}

void create_code::report2(std::ofstream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_code::report2" << endl;
	}

	if (!f_field) {
		cout << "The code is not linear" << endl;
		exit(1);
	}

	if (f_has_generator_matrix) {

		if (description->f_BCH) {
			Create_BCH_code->do_report(verbose_level);
		}
		else if (description->f_Reed_Solomon) {
			Create_RS_code->do_report(verbose_level);
		}
		else {
			l1_interfaces::latex_interface Li;

			ost << "The generator matrix is:" << endl;
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			Li.int_matrix_print_tex(ost, genma, k, n);
			ost << "\\right]" << endl;
			ost << "$$" << endl;
		}
	}

	if (f_v) {
		cout << "create_code::report2 done" << endl;
	}
}




}}}
