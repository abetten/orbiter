/*
 * ring_theory_global.cpp
 *
 *  Created on: Jan 10, 2022
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace ring_theory {


ring_theory_global::ring_theory_global()
{
	Record_birth();

}

ring_theory_global::~ring_theory_global()
{
	Record_death();

}

void ring_theory_global::Monomial_ordering_type_as_string(
		monomial_ordering_type Monomial_ordering_type,
		std::string &s)
{
	if (Monomial_ordering_type == t_LEX) {
		s.assign("LEX");
	}
	else if (Monomial_ordering_type == t_PART) {
		s.assign("PART");
	}
	else {
		cout << "ring_theory_global::Monomial_ordering_type_as_string unknown type" << endl;
		exit(1);
	}
}

void ring_theory_global::write_code_for_division(
		algebra::field_theory::finite_field *F,
		std::string &label_code,
		std::string &A_coeffs,
		std::string &B_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::write_code_for_division" << endl;
	}


	string fname_code;

	fname_code = "crc_" + label_code + ".cpp";

	{

		std::ofstream ost(fname_code);


		string str;
		other::orbiter_kernel_system::os_interface Os;

		Os.get_date(str);

		string label_of_parameters;
		string name_of_function;
		string name_of_array_of_polynomials;






		int *data_A;
		int *data_B;
		int sz_A, sz_B;


		Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);
		Get_int_vector_from_label(B_coeffs, data_B, sz_B, verbose_level);


		int w;

		w = F->log10_of_q;

		unipoly_domain FX(F);
		unipoly_object poly_A, poly_B, poly_Q, poly_R;


		int da = sz_A - 1;
		int db = sz_B - 1;
		int i;

		FX.create_object_of_degree(poly_A, da);

		for (i = 0; i <= da; i++) {
			FX.s_i(poly_A, i) = data_A[i];
		}

		FX.create_object_of_degree(poly_B, da);

		for (i = 0; i <= db; i++) {
			FX.s_i(poly_B, i) = data_B[i];
		}


		FREE_int(data_A);
		FREE_int(data_B);



		if (f_v) {
			cout << "A(X)=";
			FX.print_object(poly_A, cout);
			cout << endl;
		}


		if (f_v) {
			cout << "B(X)=";
			FX.print_object(poly_B, cout);
			cout << endl;
			}


		name_of_function = "divide_" + label_code;



		label_of_parameters = "q" + std::to_string(F->q) + "_n" + std::to_string(da) + "_r" + std::to_string(db);



		name_of_array_of_polynomials = "crc_poly_table_" + label_code;




		ost << "/*" << endl;
		ost << " * " << fname_code << endl;
		ost << " *" << endl;
		ost << " *  Created on: " << str << endl;
		ost << " *      Author: Orbiter" << endl;
		ost << " * crc code parameters: " << label_of_parameters << endl;
		ost << " */" << endl;
		ost << endl;
		//ost << "#include \"orbiter.h\"" << endl;
		ost << "#include <iostream>" << endl;
		ost << endl;
		ost << "using namespace std;" << endl;
		//ost << "using namespace orbiter;" << endl;
		ost << endl;
		ost << "void " << name_of_function << "(const unsigned char *A, unsigned char *R);" << endl;
		ost << endl;
		ost << "int main(int argc, char **argv)" << endl;
		ost << "{" << endl;
		ost << "\t" << endl;



		FX.create_object_of_degree(poly_Q, da);

		FX.create_object_of_degree(poly_R, da);

		//FX.division_with_remainder(
		//	poly_A, poly_B,
		//	poly_Q, poly_R,
		//	verbose_level);


		int *ra = (int *) poly_A;
		int *rb = (int *) poly_B;
		int *A = ra + 1;
		int *B = rb + 1;

		//int da, db;

		if (f_v) {
			cout << "ring_theory_global::write_code_for_division" << endl;
		}
		if (da != FX.degree(poly_A)) {
			cout << "ring_theory_global::write_code_for_division "
					"da != FX.degree(poly_A)" << endl;
			exit(1);
		}
		if (db != FX.degree(poly_B)) {
			cout << "ring_theory_global::write_code_for_division "
					"db != FX.degree(poly_B)" << endl;
			exit(1);
		}

		int dq = da - db;


		int j;
		int a, b;


		ost << "\tconst unsigned char A[] = {" << endl;


		ost << "\t\t";
		for (i = 0; i <= da; i++) {
			a = A[i];
			ost << a << ",";
			if (((i + 1) % 25) == 0) {
				ost << endl;
				ost << "\t\t";
			}
		}
		ost << endl;

		ost << "\t};" << endl;

		ost << endl;

		ost << "\tunsigned char R[" << db + 1 << "] = {" << endl;
		ost << "\t\t";
		for (i = 0; i <= db; i++) {
			ost << "0";
			if (i < db) {
				ost << ",";
			}
		}
		ost << "};" << endl;

		ost << endl;

		ost << "\t" << endl;
		ost << "\t" << name_of_function << "(A, R);" << endl;

		ost << endl;


		ost << "\tint i;" << endl;
		ost << "\tfor (i = 0; i < " << db << "; i++) {" << endl;
		ost << "\t\tcout << (int) R[i] << \",\";" << endl;
		ost << "\t}" << endl;
		ost << "\tcout << endl;" << endl;

		ost << endl;

		ost << "}" << endl;
		ost << endl;



		ost << "\t// crc code parameters: " << label_of_parameters << endl;
		ost << "\t// the size of the array " << name_of_array_of_polynomials << " is  " << F->q - 1 << " x " << db + 1 << endl;
		ost << "const unsigned char " << name_of_array_of_polynomials << "[] = {" << endl;


		for (i = 1; i < F->q; i++) {
			ost << "\t";
			for (j = 0; j <= db; j++) {
				a = B[j];
				b = F->mult(a, i);
				ost << setw(w) << b << ",";
			}
			ost << endl;
		}

		ost << "};" << endl;
		ost << endl;


		ost << "void " << name_of_function << "(const unsigned char *in, unsigned char *out)" << endl;
		ost << "{" << endl;


		ost << "\tunsigned char R[" << da + 1 << "];" << endl;
		ost << "\tint i, j, ii, jj;" << endl;
		ost << "\tint x;" << endl;

		// copy input over to R[]:

		ost << "\tfor (i = 0; i < " << da + 1 << "; i++) {" << endl;
		ost << "\t\tR[i] = in[i];" << endl;
		ost << "\t}" << endl;


		//Orbiter->Int_vec.zero(Q, dq + 1);

		ost << endl;



		ost << "\tfor (i = " << da << ", j = " << dq << "; i >= " << db << "; i--, j--) {" << endl;
		ost << "\t\tx = R[i];" << endl;
		ost << "\t\tif (x == 0) {" << endl;
		ost << "\t\t\tcontinue;" << endl;
		ost << "\t\t}" << endl;
		ost << "\t\t//cout << \"i=\" << i << \" x=\" << x << endl;" << endl;
		ost << "\t\tx--;" << endl;
		ost << "\t\tfor (ii = i, jj = " << db << "; jj >= 0; ii--, jj--) {" << endl;
		ost << "\t\t\tR[ii] ^= " << name_of_array_of_polynomials << "[x * " << db + 1 << " + jj];" << endl;
		ost << "\t\t}" << endl;
		ost << "\t}" << endl;

#if 0
		for (i = da, j = dq; i >= db; i--, j--) {
			x = R[i];
			c = F->mult(x, pivot_inv);
			Q[j] = c;
			c = F->negate(c);
			//cout << "i=" << i << " c=" << c << endl;
			for (ii = i, jj = db; jj >= 0; ii--, jj--) {
				d = B[jj];
				d = F->mult(c, d);
				R[ii] = F->add(d, R[ii]);
			}
			if (R[i] != 0) {
				cout << "unipoly::write_code_for_division: R[i] != 0" << endl;
				exit(1);
			}
			//cout << "i=" << i << endl;
			//cout << "q="; print_object((unipoly_object)
			// rq, cout); cout << endl;
			//cout << "r="; print_object(r, cout); cout << endl;
		}
#endif


		// copy output over from R[] to out:

		ost << endl;

		ost << "\tfor (i = " << db - 1 << "; i >= 0; i--) {" << endl;
		ost << "\t\tout[i] = R[i];" << endl;
		ost << "\t}" << endl;

		ost << "}" << endl;

	}

	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_code << " of size " << Fio.file_size(fname_code) << endl;

	if (f_v) {
		cout << "ring_theory_global::write_code_for_division done" << endl;
	}
}


void ring_theory_global::polynomial_division(
		algebra::field_theory::finite_field *F,
		std::string &A_coeffs,
		std::string &B_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division" << endl;
	}

	int *data_A;
	int *data_B;
	int sz_A, sz_B;


	Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);
	Get_int_vector_from_label(B_coeffs, data_B, sz_B, verbose_level);




	unipoly_domain FX(F);
	unipoly_object A, B, Q, R;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}


	if (f_v) {
		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
		}

	FX.create_object_of_degree(Q, da);

	FX.create_object_of_degree(R, da);


	if (f_v) {
		cout << "ring_theory_global::polynomial_division "
				"before FX.division_with_remainder" << endl;
	}

	FX.division_with_remainder(
		A, B,
		Q, R,
		verbose_level);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division "
				"after FX.division_with_remainder" << endl;
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;

		cout << "Q(X)=";
		FX.print_object(Q, cout);
		cout << endl;

		cout << "R(X)=";
		FX.print_object(R, cout);
		cout << endl;
	}

	FREE_int(data_A);
	FREE_int(data_B);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division done" << endl;
	}
}

void ring_theory_global::extended_gcd_for_polynomials(
		algebra::field_theory::finite_field *F,
		std::string &A_coeffs,
		std::string &B_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::extended_gcd_for_polynomials" << endl;
	}



	int *data_A;
	int *data_B;
	int sz_A, sz_B;


	Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);
	Get_int_vector_from_label(B_coeffs, data_B, sz_B, verbose_level);


	algebra::number_theory::number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A, B, U, V, G;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, da);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= F->q) {
			data_B[i] = NT.mod(data_B[i], F->q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;
	}

	FX.create_object_of_degree(U, da);

	FX.create_object_of_degree(V, da);

	FX.create_object_of_degree(G, da);


	if (f_v) {
		cout << "ring_theory_global::extended_gcd_for_polynomials "
				"before FX.extended_gcd" << endl;
	}

	{
		FX.extended_gcd(
			A, B,
			U, V, G, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::extended_gcd_for_polynomials "
				"after FX.extended_gcd" << endl;
	}

	if (f_v) {
		cout << "U(X)=";
		FX.print_object(U, cout);
		cout << endl;

		cout << "V(X)=";
		FX.print_object(V, cout);
		cout << endl;

		cout << "G(X)=";
		FX.print_object(G, cout);
		cout << endl;

		cout << "deg G(X) = " << FX.degree(G) << endl;
	}

	if (FX.degree(G) == 0) {
		int c, cv, d;

		c = FX.s_i(G, 0);
		if (c != 1) {
			if (f_v) {
				cout << "ring_theory_global::extended_gcd_for_polynomials "
						"before normalizing:" << endl;
			}
			cv = F->inverse(c);
			if (f_v) {
				cout << "cv=" << cv << endl;
			}

			d = FX.degree(U);
			for (i = 0; i <= d; i++) {
				FX.s_i(U, i) = F->mult(cv, FX.s_i(U, i));
			}

			d = FX.degree(V);
			for (i = 0; i <= d; i++) {
				FX.s_i(V, i) = F->mult(cv, FX.s_i(V, i));
			}

			d = FX.degree(G);
			for (i = 0; i <= d; i++) {
				FX.s_i(G, i) = F->mult(cv, FX.s_i(G, i));
			}



			if (f_v) {
				cout << "ring_theory_global::extended_gcd_for_polynomials "
						"after normalizing:" << endl;

				cout << "U(X)=";
				FX.print_object(U, cout);
				cout << endl;

				cout << "V(X)=";
				FX.print_object(V, cout);
				cout << endl;

				cout << "G(X)=";
				FX.print_object(G, cout);
				cout << endl;
			}
		}

	}

	if (f_v) {
		cout << "ring_theory_global::extended_gcd_for_polynomials done" << endl;
	}
}


void ring_theory_global::polynomial_mult_mod(
		algebra::field_theory::finite_field *F,
		std::string &A_coeffs,
		std::string &B_coeffs,
		std::string &M_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_mult_mod" << endl;
	}

	int *data_A;
	int *data_B;
	int *data_M;
	int sz_A, sz_B, sz_M;


	Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);
	Get_int_vector_from_label(B_coeffs, data_B, sz_B, verbose_level);
	Get_int_vector_from_label(M_coeffs, data_M, sz_M, verbose_level);

	algebra::number_theory::number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A, B, M, C;


	int da = sz_A - 1;
	int db = sz_B - 1;
	int dm = sz_M - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(B, db);

	for (i = 0; i <= db; i++) {
		if (data_B[i] < 0 || data_B[i] >= F->q) {
			data_B[i] = NT.mod(data_B[i], F->q);
		}
		FX.s_i(B, i) = data_B[i];
	}

	FX.create_object_of_degree(M, dm);

	for (i = 0; i <= dm; i++) {
		if (data_M[i] < 0 || data_M[i] >= F->q) {
			data_M[i] = NT.mod(data_M[i], F->q);
		}
		FX.s_i(M, i) = data_M[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "B(X)=";
		FX.print_object(B, cout);
		cout << endl;

		cout << "M(X)=";
		FX.print_object(M, cout);
		cout << endl;
	}

	FX.create_object_of_degree(C, da + db);



	if (f_v) {
		cout << "ring_theory_global::polynomial_mult_mod "
				"before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(A, B, C, M, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_mult_mod "
				"after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "C(X)=";
		FX.print_object(C, cout);
		cout << endl;

		cout << "deg C(X) = " << FX.degree(C) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_mult_mod done" << endl;
	}
}

void ring_theory_global::polynomial_power_mod(
		algebra::field_theory::finite_field *F,
		std::string &A_coeffs,
		std::string &power_text,
		std::string &M_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod" << endl;
	}

	int *data_A;
	long int n;
	int *data_M;
	int sz_A, sz_M;
	other::data_structures::string_tools ST;


	Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);


	n = ST.strtolint(power_text);

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod n = " << n << endl;
	}


	Get_int_vector_from_label(M_coeffs, data_M, sz_M, verbose_level);

	algebra::number_theory::number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A, M;


	int da = sz_A - 1;
	int dm = sz_M - 1;
	int i;

	FX.create_object_of_degree(A, dm);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
		}
		FX.s_i(A, i) = data_A[i];
	}

	FX.create_object_of_degree(M, dm);

	for (i = 0; i <= dm; i++) {
		if (data_M[i] < 0 || data_M[i] >= F->q) {
			data_M[i] = NT.mod(data_M[i], F->q);
		}
		FX.s_i(M, i) = data_M[i];
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;


		cout << "M(X)=";
		FX.print_object(M, cout);
		cout << endl;
	}

	//FX.create_object_of_degree(C, dm + 1);



	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod "
				"before FX.mult_mod" << endl;
	}

	{
		FX.power_mod(A, M, n, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod "
				"after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;

		cout << "deg A(X) = " << FX.degree(A) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod done" << endl;
	}
}

void ring_theory_global::polynomial_find_roots(
		algebra::field_theory::finite_field *F,
		std::string &polynomial_find_roots_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_find_roots" << endl;
	}

	int *data_A;
	int sz_A;

	Get_int_vector_from_label(polynomial_find_roots_label, data_A, sz_A, verbose_level);

	algebra::number_theory::number_theory_domain NT;




	unipoly_domain FX(F);
	unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
		}
		FX.s_i(A, i) = data_A[i];
	}


	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "ring_theory_global::polynomial_find_roots "
				"before FX.mult_mod" << endl;
	}

	{
		int a, b;

		for (a = 0; a < F->q; a++) {
			b = FX.substitute_scalar_in_polynomial(
				A, a, 0 /* verbose_level*/);
			if (b == 0) {
				cout << "a root is " << a << endl;
			}
		}
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_find_roots "
				"after FX.mult_mod" << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_find_roots done" << endl;
	}
}

void ring_theory_global::sift_polynomials(
		algebra::field_theory::finite_field *F,
		long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	unipoly_object p;
	longinteger_domain ZZ;
	long int rk;
	int f_is_irred, f_is_primitive;
	int f, i;
	long int len, idx;
	long int *Table;
	longinteger_object Q, m1, Qm1;

	if (f_v) {
		cout << "ring_theory_global::sift_polynomials" << endl;
	}
	unipoly_domain D(F);

	len = rk1 - rk0;
	Table = NEW_lint(len * 3);
	for (rk = rk0; rk < rk1; rk++) {

		idx = rk - rk0;

		D.create_object_by_rank(p, rk,
				0 /* verbose_level*/);
		if (f_v) {
			cout << "rk=" << rk << " poly=";
			D.print_object(p, cout);
			cout << endl;
		}


		int nb_primes;
		longinteger_object *primes;
		int *exponents;


		f = D.degree(p);
		Q.create(F->q);
		m1.create(-1);
		ZZ.power_int(Q, f);
		ZZ.add(Q, m1, Qm1);
		if (f_v) {
			cout << "ring_theory_global::sift_polynomials Qm1 = " << Qm1 << endl;
		}
		ZZ.factor_into_longintegers(Qm1, nb_primes,
				primes, exponents, verbose_level - 2);
		if (f_v) {
			cout << "ring_theory_global::sift_polynomials after factoring "
					<< Qm1 << " nb_primes=" << nb_primes << endl;
			cout << "primes:" << endl;
			for (i = 0; i < nb_primes; i++) {
				cout << i << " : " << primes[i] << endl;
			}
		}

		f_is_irred = D.is_irreducible(p, verbose_level - 2);
		f_is_primitive = D.is_primitive(p,
				Qm1,
				nb_primes, primes,
				verbose_level);

		if (f_v) {
			cout << "rk=" << rk << " poly=";
			D.print_object(p, cout);
			cout << " is_irred=" << f_is_irred << " is_primitive=" << f_is_primitive << endl;
		}

		Table[idx * 3 + 0] = rk;
		Table[idx * 3 + 1] = f_is_irred;
		Table[idx * 3 + 2] = f_is_primitive;


		D.delete_object(p);

		FREE_int(exponents);
		FREE_OBJECTS(primes);
	}
	for (idx = 0; idx < len; idx++) {
		rk = Table[idx * 3 + 0];
		D.create_object_by_rank(p, rk,
				0 /* verbose_level*/);
		f_is_irred = Table[idx * 3 + 1];
		f_is_primitive = Table[idx * 3 + 2];
		if (f_v) {
			cout << "rk=" << rk;
			cout << " is_irred=" << f_is_irred << " is_primitive=" << f_is_primitive;
			cout << " poly=";
			D.print_object(p, cout);
			cout << endl;
		}
		D.delete_object(p);
	}

	if (f_v) {
		cout << "ring_theory_global::sift_polynomials done" << endl;
	}

}

void ring_theory_global::mult_polynomials(
		algebra::field_theory::finite_field *F,
		long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::mult_polynomials" << endl;
	}
	unipoly_domain D(F);

	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "polynomial_mult_" + std::to_string(rk0) + "_" + std::to_string(rk1) + ".tex";
		title = "Polynomial Mult";



		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

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


			if (f_v) {
				cout << "ring_theory_global::mult_polynomials before report" << endl;
			}
			//report(ost, verbose_level);

			long int rk2;
			D.mult_easy_with_report(rk0, rk1, rk2, ost, verbose_level);
			ost << "$" << rk0 << " \\otimes " << rk1 << " = " << rk2 << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::mult_polynomials after report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::mult_polynomials written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::mult_polynomials done" << endl;
	}

}



void ring_theory_global::polynomial_division_coefficient_table_with_report(
		std::string &prefix,
		algebra::field_theory::finite_field *F,
		int *coeff_table0, int coeff_table0_len,
		int *coeff_table1, int coeff_table1_len,
		int *&coeff_table_q, int &coeff_table_q_len,
		int *&coeff_table_r, int &coeff_table_r_len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_coefficient_table_with_report" << endl;
	}
	unipoly_domain D(F);

	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = prefix + "_polynomial_division.tex";
		title = "Polynomial Division";




		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

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


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_coefficient_table_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}


			ost << "{\\scriptsize" << endl;
			ost << "data:" << endl;
			ost << "$$" << endl;
			L.int_vec_print_as_matrix(
					ost,
					coeff_table0, coeff_table0_len, 20, true /*f_tex*/);
			ost << "$$" << endl;
			ost << "}" << endl;


			ost << "CRC:" << endl;
			ost << "$$" << endl;
			L.int_vec_print_as_matrix(
					ost,
					coeff_table1, coeff_table1_len, 10, true /*f_tex*/);
			ost << "$$" << endl;

			int f_report = true;

			D.division_with_remainder_based_on_tables_with_report(
					coeff_table0, coeff_table0_len,
					coeff_table1, coeff_table1_len,
					coeff_table_q, coeff_table_q_len,
					coeff_table_r, coeff_table_r_len,
					ost, f_report, verbose_level);

			ost << "{\\scriptsize" << endl;
			ost << "Quotient:" << endl;
			ost << "$$" << endl;
			L.int_vec_print_as_matrix(
					ost,
					coeff_table_q, coeff_table_q_len, 20, true /*f_tex*/);
			ost << "$$" << endl;
			ost << "}" << endl;

			ost << "Remainder:" << endl;
			ost << "$$" << endl;
			L.int_vec_print_as_matrix(
					ost,
					coeff_table_r, coeff_table_r_len, 20, true /*f_tex*/);
			ost << "$$" << endl;


			//ost << "$" << rk0 << " / " << rk1 << " = " << rk2
			//		<< "$ Remainder $" << rk3 << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_coefficient_table_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_coefficient_table_with_report "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_coefficient_table_with_report done" << endl;
	}

}


void ring_theory_global::polynomial_division_with_report(
		algebra::field_theory::finite_field *F,
		long int rk0, long int rk1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_with_report" << endl;
	}
	unipoly_domain D(F);

	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "polynomial_division_" + std::to_string(rk0) + "_" + std::to_string(rk1) + ".tex";
		title = "Polynomial Division";




		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

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


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk2, rk3;
			D.division_with_remainder_numerically_with_report(
					rk0, rk1, rk2, rk3, ost, verbose_level);
			ost << "$" << rk0 << " / " << rk1 << " = " << rk2 << "$ "
					"Remainder $" << rk3 << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_with_report "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_with_report done" << endl;
	}

}


void ring_theory_global::assemble_monopoly(
		algebra::field_theory::finite_field *F,
		int length,
		std::string &coefficient_vector_text,
		std::string &exponent_vector_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::assemble_monopoly" << endl;
	}

	int *A;
	int *V;
	int A_sz;
	int V_sz;

	Get_int_vector_from_label(coefficient_vector_text,
			A, A_sz, verbose_level);
	Get_int_vector_from_label(exponent_vector_text,
			V, V_sz, verbose_level);

	if (A_sz != V_sz) {
		cout << "ring_theory_global::assemble_monopoly A_sz != V_sz" << endl;
		exit(1);
	}

	int h, a, v;
	int *poly;

	poly = NEW_int(length);
	Int_vec_zero(poly, length);
	for (h = 0; h < A_sz; h++) {
		a = A[h];
		v = V[h];
		poly[a] = v;
	}
	int coeff;
	int f_first;

	f_first = true;

	for (h = 0; h < length; h++) {
		coeff = poly[h];
		if (coeff) {
			if (!f_first) {
				cout << "+";
			}
			f_first = false;
			cout << coeff;
			if (h) {
				cout << "*X";
				if (h > 1) {
					cout << "^" << h;
				}
			}
		}
	}
	cout << endl;

	FREE_int(poly);

	if (f_v) {
		cout << "ring_theory_global::assemble_monopoly done" << endl;
	}
}



void ring_theory_global::polynomial_division_from_file_with_report(
		algebra::field_theory::finite_field *F,
		std::string &input_file, long int rk1,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_with_report" << endl;
	}
	unipoly_domain D(F);
	{

		string fname;
		string author;
		string title;
		string extra_praeamble;

		fname = "polynomial_division_" + input_file + "_" + std::to_string(rk1);
		title = "Polynomial Division";



		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

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


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk_q, rk_r;

			D.division_with_remainder_from_file_with_report(
					input_file, rk1,
					rk_q, rk_r, ost, verbose_level);

			ost << "$ / " << rk1 << " = " << rk_q << "$ Remainder $"
					<< rk_r << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_from_file_with_report "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_with_report done" << endl;
	}

}

void ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report(
		algebra::field_theory::finite_field *F,
		std::string &input_file, long int rk1,
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report" << endl;
	}
	unipoly_domain D(F);
	{


		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "polynomial_division_all_errors_k" + std::to_string(k) + "_" + std::to_string(rk1);
		title = "Polynomial Division";




		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

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


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int *rk_q, *rk_r;
			int N, n, h;
			combinatorics::other_combinatorics::combinatorics_domain Combi;

			int *set;

			set = NEW_int(k);



			D.division_with_remainder_from_file_all_k_bit_error_patterns(input_file,
					rk1, k,
					rk_q, rk_r, n, N, ost, verbose_level);

			ost << "$" << input_file << " / " << rk1 << "$\\\\" << endl;

			for (h = 0; h < N; h++) {
				Combi.unrank_k_subset(h, set, n, k);
				ost << h << " : ";
				Int_vec_print(ost, set, k);
				ost << " : ";
				ost << rk_r[h] << "\\\\" << endl;
			}

			FREE_lint(rk_q);
			FREE_lint(rk_r);

			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}

			FREE_int(set);

			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_with_report done" << endl;
	}

}

void ring_theory_global::create_irreducible_polynomial(
		algebra::field_theory::finite_field *F,
		unipoly_domain *Fq,
		unipoly_object *&Beta, int n,
		long int *cyclotomic_set, int cylotomic_set_size,
		unipoly_object *&min_poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial n=" << n << endl;
	}
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"before allocating min_poly etc" << endl;
	}

	int degree = cylotomic_set_size;

	combinatorics::coding_theory::cyclic_codes Cyclic_codes;

	min_poly = NEW_OBJECTS(unipoly_object, degree + 2);
	unipoly_object *tmp = NEW_OBJECTS(unipoly_object, degree + 1);
	unipoly_object *linear_factor = NEW_OBJECTS(unipoly_object, 2);
	unipoly_object Pc, Pd;

	int i, j, h, r;

	// create the polynomial linear_factor = X - a:
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"creating linear_factor = X-a" << endl;
	}
	for (i = 0; i < 2; i++) {
		if (i == 1) {
			Fq->create_object_by_rank(linear_factor[i], 1, verbose_level);
		}
		else {
			Fq->create_object_by_rank(linear_factor[i], 0, verbose_level);
		}
	}
	for (i = 0; i <= degree; i++) {
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"creating generator[" << i << "]" << endl;
		}
		Fq->create_object_by_rank(min_poly[i], 0, verbose_level);
		Fq->create_object_by_rank(tmp[i], 0, verbose_level);
	}
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"creating generator[0]" << endl;
	}
	Fq->create_object_by_rank(min_poly[0], 1, verbose_level);

	// now coeffs has degree 1
	// and generator has degree 0

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"coeffs:" << endl;
		Cyclic_codes.print_polynomial(*Fq, 1, linear_factor);
		cout << endl;
		cout << "ring_theory_global::create_irreducible_polynomial "
				"generator:" << endl;
		Cyclic_codes.print_polynomial(*Fq, 0, min_poly);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"creating Pc" << endl;
	}
	Fq->create_object_by_rank(Pc, 0, verbose_level);
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"creating Pd" << endl;
	}
	Fq->create_object_by_rank(Pd, 0, verbose_level);

	r = 0;
	for (h = 0; h < cylotomic_set_size; h++) {
		i = cyclotomic_set[h];
		if (f_v) {
			cout << "h=" << h << ", i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"before Fq.assign beta" << endl;
		}
		Fq->assign(Beta[i], linear_factor[0], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"before Fq.negate" << endl;
		}
		Fq->negate(linear_factor[0]);
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"root: " << i << " : ";
			Fq->print_object(linear_factor[0], cout);
			//cout << " : ";
			//print_polynomial(Fq, 2, coeffs);
			cout << endl;
		}


		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"before Fq.assign(min_poly[j], tmp[j])" << endl;
		}
		for (j = 0; j <= r; j++) {
			Fq->assign(min_poly[j], tmp[j], verbose_level);
		}

		//cout << "tmp:" << endl;
		//print_polynomial(Fq, r, tmp);
		//cout << endl;

		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"before Fq.assign(tmp[j], min_poly[j + 1])" << endl;
		}

		for (j = 0; j <= r; j++) {
			Fq->assign(tmp[j], min_poly[j + 1], verbose_level);
		}

		Fq->delete_object(min_poly[0]);
		Fq->create_object_by_rank(min_poly[0], 0, verbose_level);

		//cout << "generator after shifting up:" << endl;
		//print_polynomial(Fq, r + 1, generator);
		//cout << endl;

		for (j = 0; j <= r; j++) {
			if (f_v) {
				cout << "ring_theory_global::create_irreducible_polynomial "
						"j=" << j << endl;
			}
			if (f_v) {
				cout << "ring_theory_global::create_irreducible_polynomial "
						"before Fq.mult(tmp[j], linear_factor[0], Pc)" << endl;
			}
			Fq->mult(tmp[j], linear_factor[0], Pc, verbose_level - 1);
			if (f_v) {
				cout << "ring_theory_global::create_irreducible_polynomial "
						"before Fq.add" << endl;
			}
			Fq->add(Pc, min_poly[j], Pd);
			if (f_v) {
				cout << "ring_theory_global::create_irreducible_polynomial "
						"before "
						"Fq.assign" << endl;
			}
			Fq->assign(Pd, min_poly[j], verbose_level);
		}
		r++;
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"r=" << r << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial "
					"current polynomial: ";
			Cyclic_codes.print_polynomial(*Fq, r, min_poly);
			cout << endl;
		}

	}

	if (r != degree) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"r != degree" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"The minimum polynomial is: ";
		Cyclic_codes.print_polynomial(*Fq, r, min_poly);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial "
				"done" << endl;
	}

}

void ring_theory_global::compute_nth_roots_as_polynomials(
		algebra::field_theory::finite_field *F,
		unipoly_domain *FpX,
		unipoly_domain *Fq, unipoly_object *&Beta,
		int n1, int n2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//unipoly_object M;
	unipoly_object beta;

	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials " << endl;
	}

#if 0
	Fp.finite_field_init(p, false /* f_without_tables */, verbose_level - 1);

	algebra_global Algebra;
	unipoly_domain FpX(&Fp);
	FpX.create_object_by_rank_string(M,
			Algebra.get_primitive_polynomial(p, field_degree, 0),
			verbose_level - 2);
#endif

	int m, r;
	int i;
	longinteger_object Qm1, Index;
	longinteger_domain D;
	algebra::number_theory::number_theory_domain NT;

	m = NT.order_mod_p(F->q, n1);
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code "
				"order of q mod n is m=" << m << endl;
	}
	D.create_qnm1(Qm1, F->q, m);

	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(Qm1, n2, Index, r);
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code "
				"Index = " << Index << endl;
	}

	int subgroup_index;

	subgroup_index = Index.as_int();
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code "
				"subgroup_index = " << subgroup_index << endl;
	}

	//b = (q - 1) / n;
	if (r != 0) {
		cout << "ring_theory_global::make_cyclic_code "
				"n does not divide q^m-1" << endl;
		exit(1);
	}

#if 0
	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials "
				"choosing the following irreducible "
				"and primitive polynomial:" << endl;
		FpX.print_object(M, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials "
				"creating unipoly_domain Fq modulo M" << endl;
	}
	//unipoly_domain Fq(this, M, verbose_level);  // Fq = Fp[X] modulo factor polynomial M
	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials "
				"extension field created" << endl;
	}
#endif

	Beta = new unipoly_object[n2];

	for (i = 0; i < n2; i++) {
		Fq->create_object_by_rank(Beta[i], 0, verbose_level);

	}

	//Fq->create_object_by_rank(c, 0, verbose_level);
	Fq->create_object_by_rank(beta, F->p, verbose_level); // the element alpha
	//Fq->create_object_by_rank(beta_i, 1, verbose_level);
	if (subgroup_index != 1) {
		//Fq.power_int(beta, b);
		if (f_v) {
			cout << "\\alpha = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"before Fq->power_int" << endl;
		}
		Fq->power_int(beta, subgroup_index, verbose_level - 1);
		if (f_v) {
			cout << "\\beta = \\alpha^" << Index << " = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
	}
	else {
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"subgroup_index is one" << endl;
		}
	}


	for (i = 0; i < n2; i++) {
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"before Fq.assign beta" << endl;
		}
		Fq->assign(beta, Beta[i], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials "
					"before Fq.power_int" << endl;
		}
		Fq->power_int(Beta[i], i, verbose_level);
	}


	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials "
				"done" << endl;
	}

}

void ring_theory_global::compute_powers(
		algebra::field_theory::finite_field *F,
		unipoly_domain *Fq,
		int n, int start_idx,
		unipoly_object *&Beta, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::compute_powers" << endl;
	}
	int i;
	unipoly_object beta;

	Beta = new unipoly_object[n];

	for (i = 0; i < n; i++) {
		Fq->create_object_by_rank(Beta[i], 0, verbose_level);

	}

	Fq->create_object_by_rank(
			beta, F->p, verbose_level); // the element alpha

	if (start_idx != 1) {

		if (f_v) {
			cout << "\\alpha = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"before Fq->power_int" << endl;
		}
		Fq->power_int(beta, start_idx, verbose_level - 1);
		if (f_v) {
			cout << "\\beta = \\alpha^" << start_idx << " = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
	}
	else {
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"subgroup_index is one" << endl;
		}
	}


	for (i = 0; i < n; i++) {
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"before Fq.assign beta" << endl;
		}
		Fq->assign(beta, Beta[i], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::compute_powers "
					"before Fq.power_int" << endl;
		}
		Fq->power_int(Beta[i], i, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::compute_powers done" << endl;
	}

}


void ring_theory_global::make_all_irreducible_polynomials_of_degree_d(
		algebra::field_theory::finite_field *F,
		int d,
		std::vector<std::vector<int> > &Table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	algebra::number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

#if 0
	cnt = count_all_irreducible_polynomials_of_degree_d(F, d, verbose_level - 2);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"cnt = " << cnt << endl;
	}

	nb = cnt;

	Table = NEW_int(nb * (d + 1));
#endif

	//NT.factor_prime_power(F->q, p, e);


#if 0
	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				" q=" << F->q << " p=" << F->p << " e=" << F->e << endl;
	}

	unipoly_domain FX(F);

	string poly;
	knowledge_base::knowledge_base K;

	K.get_primitive_polynomial(poly, F->q, d, 0 /* verbose_level */);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial is " << poly << endl;
	}

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(
			m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, 0 /* verbose_level */);

	int *Frobenius;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"before FX.Frobenius_matrix" << endl;
	}
	FX.Frobenius_matrix(
			Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		Int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"before compute_normal_basis" << endl;
	}
	FX.compute_normal_basis(
			d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		Int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}
#else

	algebra::field_theory::normal_basis *Nor;

	Nor = NEW_OBJECT(algebra::field_theory::normal_basis);

	Nor->init(F, d, verbose_level);

#endif

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	cnt = 0;

	Combi.int_vec_first_regular_word(Nor->v, d, F->q);
	while (true) {
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, Nor->v, d);
			cout << endl;
		}

		F->Linear_algebra->mult_vector_from_the_right(
				Nor->Normal_basis, Nor->v, Nor->w, d, d);
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : w = ";
			Int_vec_print(cout, Nor->w, d);
			cout << endl;
		}

		Nor->FX->delete_object(Nor->g);
		Nor->FX->create_object_of_degree(Nor->g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) Nor->g)[1 + i] = Nor->w[i];
		}

		Nor->FX->minimum_polynomial_extension_field(
				Nor->g, Nor->m, Nor->minpol, d, Nor->Frobenius,
				verbose_level - 3);
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, Nor->v, d);
			cout << " irreducible polynomial = ";
			Nor->FX->print_object(Nor->minpol, cout);
			cout << endl;
		}



		std::vector<int> T;

		for (i = 0; i <= d; i++) {
			T.push_back(((int *)Nor->minpol)[1 + i]);
		}
		Table.push_back(T);


		cnt++;


		if (!Combi.int_vec_next_regular_word(Nor->v, d, F->q)) {
			break;
		}

	}

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << F->q << endl;
	}

	FREE_OBJECT(Nor);
#if 0
	FREE_int(Frobenius);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);
#endif

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << " done" << endl;
	}
}

int ring_theory_global::count_all_irreducible_polynomials_of_degree_d(
		algebra::field_theory::finite_field *F,
		int d,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	algebra::number_theory::number_theory_domain NT;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d " << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"p=" << F->p << " e=" << F->e << endl;
	}
	if (F->e > 1) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"e=" << F->e << " is greater than one" << endl;
	}

	unipoly_domain FX(F);

	string poly;
	combinatorics::knowledge_base::knowledge_base K;

	K.get_primitive_polynomial(poly, F->q, d, 0 /* verbose_level */);

	unipoly_object m;
	unipoly_object g;
	unipoly_object minpol;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, 0 /* verbose_level */);

	int *Frobenius;
	//int *F2;
	int *Normal_basis;
	int *v;
	int *w;

	//Frobenius = NEW_int(d * d);
	//F2 = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	FX.Frobenius_matrix(Frobenius, m, verbose_level - 3);
	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius_matrix = " << endl;
		Int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

#if 0
	F->mult_matrix_matrix(Frobenius, Frobenius, F2, d, d, d,
			0 /* verbose_level */);
	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"Frobenius^2 = " << endl;
		int_matrix_print(F2, d, d);
		cout << endl;
	}
#endif

	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 3);

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		Int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;
	Combi.int_vec_first_regular_word(v, d, F->q);
	while (true) {
		if (f_vv) {
			cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, v, d);
			cout << endl;
		}

		F->Linear_algebra->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : w = ";
			Int_vec_print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(
				g, m, minpol, d,
				Frobenius, verbose_level - 3);
		if (f_vv) {
			cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}
		if (FX.degree(minpol) != d) {
			cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial does not have degree d"
					<< endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}
		if (!FX.is_irreducible(minpol, verbose_level)) {
			cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
					"The polynomial is not irreducible" << endl;
			FX.print_object(minpol, cout);
			cout << endl;
			exit(1);
		}


		cnt++;

		if (!Combi.int_vec_next_regular_word(v, d, F->q)) {
			break;
		}

	}

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt << " irreducible polynomials "
				"of degree " << d << " over " << "F_" << F->q << endl;
	}

	FREE_int(Frobenius);
	//FREE_int(F2);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);

	if (f_v) {
		cout << "ring_theory_global::count_all_irreducible_polynomials_of_degree_d done" << endl;
	}
	return cnt;
}

void ring_theory_global::do_make_table_of_irreducible_polynomials(
		algebra::field_theory::finite_field *F,
		int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::do_make_table_of_irreducible_polynomials" << endl;
		cout << "deg=" << deg << endl;
		cout << "q=" << F->q << endl;
	}

	int nb;
	std::vector<std::vector<int>> Table;

	make_all_irreducible_polynomials_of_degree_d(
			F, deg,
			Table, verbose_level);

	nb = Table.size();

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << F->q << " are:" << endl;

	other::orbiter_kernel_system::Orbiter->Int_vec->vec_print(Table);


	int *T;
	int i, j;

	T = NEW_int(Table.size() * (deg + 1));
	for (i = 0; i < Table.size(); i++) {
		for (j = 0; j < deg + 1; j++) {
			T[i * (deg + 1) + j] = Table[i][j];
		}
	}


	ring_theory::unipoly_domain FX(F);


	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		fname = "Irred_q" + std::to_string(F->q) + "_d" + std::to_string(deg) + ".tex";
		title = "Irreducible Polynomials of Degree "
				+ std::to_string(deg) + " over F" + std::to_string(F->q);


		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;
			geometry::other_geometry::geometry_global GG;
			long int rk;

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


			if (f_v) {
				cout << "ring_theory_global::do_make_table_of_irreducible_polynomials "
						"before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "There are " << Table.size() << " irreducible polynomials of "
					"degree " << deg << " over the field F" << F->q << ":\\\\" << endl;
			ost << "The coefficients in increasing order are:\\\\" << endl;
			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;
			//ost << "\\begin{multicols}{2}" << endl;

			long int *Rk;
			int *mtx;

			mtx = NEW_int(deg * deg);

			Rk = NEW_lint(Table.size());

			for (i = 0; i < Table.size(); i++) {
				rk = GG.AG_element_rank(F->q, T + i * (deg + 1), 1, deg + 1);
				Rk[i] = rk;
			}


			ost << "\\noindent" << endl;
			for (i = 0; i < Table.size(); i++) {

				ost << i << " : ";

				ost << "$";
				for (j = 0; j <= deg; j++) {
					ost << T[i * (deg + 1) + j];
				}
				ost << "$ : $";
				rk = GG.AG_element_rank(F->q, T + i * (deg + 1), 1, deg + 1);
				ost << rk;
				ost << "$";



				ring_theory::unipoly_object m;


				FX.create_object_by_rank(
					m, rk,
					0 /*verbose_level*/);

				string s;

				s = FX.stringify_object(m);

				ost << " : ";

				ost << "$" << s << "$";


				FX.make_companion_matrix(
						m, mtx, verbose_level);


				s = Int_vec_stringify(mtx, deg * deg);

				ost << " : ";

				ost << "$" << s << "$";

				ost << "\\\\" << endl;
			}
			//ost << "\\end{multicols}" << endl;


			if (f_v) {
				cout << "ring_theory_global::do_make_table_of_irreducible_polynomials "
						"after report" << endl;
			}

			ost << "companion matrices only:\\\\" << endl;

			for (i = 0; i < Table.size(); i++) {

				rk = GG.AG_element_rank(F->q, T + i * (deg + 1), 1, deg + 1);


				ring_theory::unipoly_object m;


				FX.create_object_by_rank(
					m, rk,
					0 /*verbose_level*/);


				FX.make_companion_matrix(
						m, mtx, verbose_level);

				string s;

				s = Int_vec_stringify(mtx, deg * deg);

				ost << "$" << s << ", $";

				ost << "\\\\" << endl;


			}

			FREE_int(mtx);
			FREE_lint(Rk);

			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::do_make_table_of_irreducible_polynomials "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_int(T);

	//int_matrix_print(Table, nb, deg + 1);

	//FREE_int(Table);

	if (f_v) {
		cout << "ring_theory_global::do_make_table_of_irreducible_polynomials done" << endl;
	}
}



#if 0
void ring_theory_global::do_search_for_primitive_polynomial_in_range(
		int p_min, int p_max,
		int deg_min, int deg_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::do_search_for_primitive_polynomial_in_range" << endl;
		cout << "p_min=" << p_min << endl;
		cout << "p_max=" << p_max << endl;
		cout << "deg_min=" << deg_min << endl;
		cout << "deg_max=" << deg_max << endl;
	}


	if (deg_min == deg_max && p_min == p_max) {
		char *poly;



		poly = search_for_primitive_polynomial_of_given_degree(
				p_min, deg_min, verbose_level);

		cout << "poly = " << poly << endl;

	}
	else {

		search_for_primitive_polynomials(p_min, p_max,
				deg_min, deg_max,
				verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::do_search_for_primitive_polynomial_in_range done" << endl;
	}
}
#endif

char *ring_theory_global::search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::field_theory::finite_field Fp;

	if (f_v) {
		cout << "ring_theory_global::search_for_primitive_polynomial_of_given_degree" << endl;
	}
	Fp.finite_field_init_small_order(p,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0 /*verbose_level*/);
	unipoly_domain FX(&Fp);

	unipoly_object m;
	longinteger_object rk;

	FX.create_object_by_rank(m, 0, verbose_level);

	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
	}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);

	char *s;
	int i, j;
	if (f_v) {
		cout << "found a primitive polynomial. The rank is " << rk << endl;
	}

	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
	}
	s[rk.len()] = 0;

	if (f_v) {
		cout << "created string " << s << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::search_for_primitive_polynomial_of_given_degree done" << endl;
	}
	return s;
}


void ring_theory_global::search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, q;
	algebra::number_theory::number_theory_domain NT;


	other::orbiter_kernel_system::Orbiter->longinteger_f_print_scientific = false;


	if (f_v) {
		cout << "ring_theory_global::search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
	}
	for (q = p_min; q <= p_max; q++) {


		if (!NT.is_prime_power(q)) {
			continue;
		}

		if (f_v) {
			cout << "ring_theory_global::search_for_primitive_polynomials "
					"considering the coefficient field of order " << q << endl;
		}

		{
			algebra::field_theory::finite_field Fq;
			Fq.finite_field_init_small_order(q,
					false /* f_without_tables */,
					false /* f_compute_related_fields */,
					0 /*verbose_level*/);
			unipoly_domain FX(&Fq);

			unipoly_object m;
			longinteger_object rk;

			FX.create_object_by_rank(m, 0, verbose_level);

			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
				}
				FX.get_a_primitive_polynomial(m, d, verbose_level);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
			}
			if (f_v) {
				cout << "ring_theory_global::search_for_primitive_polynomials "
						"before FX.delete_object(m)" << endl;
			}
			FX.delete_object(m);
			if (f_v) {
				cout << "ring_theory_global::search_for_primitive_polynomials "
						"after FX.delete_object(m)" << endl;
			}
		}
	}
	if (f_v) {
		cout << "ring_theory_global::search_for_primitive_polynomials done" << endl;
	}
}


void ring_theory_global::factor_cyclotomic(
		int n, int q, int d,
	int *coeffs, int f_poly,
	std::string &poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p, e, m, i, j, Q, a, b, c, cv, ccv, t, r1, r2, len;
	int field_degree, subgroup_index;
	algebra::field_theory::finite_field FQ;
	algebra::field_theory::finite_field Fq;
	algebra::number_theory::number_theory_domain NT;

	NT.factor_prime_power(q, p, e);
	if (f_v) {
		cout << "ring_theory_global::factor_cyclotomic q=" << q << " p=" << q
			<< " e=" << e << " n=" << n << endl;
	}
	m = NT.order_mod_p(q, n);
	if (f_v) {
		cout << "ring_theory_global::factor_cyclotomic order mod q is m=" << m << endl;
	}
	field_degree = e * m;
	Q = NT.i_power_j(p, field_degree);


	if (f_poly) {
		Fq.init_override_polynomial_small_order(
				q, poly,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				verbose_level - 1);
	}
	else {
		Fq.finite_field_init_small_order(
				q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				verbose_level - 2);
	}
	FQ.finite_field_init_small_order(
			Q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			verbose_level - 2);

	FQ.compute_subfields(verbose_level);

	subgroup_index = (Q - 1) / (q - 1);

	unipoly_domain FQX(&FQ);
	unipoly_object quo, rem, h, Xma;

	FQX.create_object_of_degree(h, d);

	if (e > 1) {
		cout << "ring_theory_global::factor_cyclotomic "
				"embedding the coefficients into the larger field" << endl;
		for (i = 0; i <= d; i++) {
			c = coeffs[i];
			if (c == 0) {
				t = 0;
			}
			else {
				a = Fq.log_alpha(c);
				t = a * subgroup_index;
				t = FQ.alpha_power(t);
			}
			FQX.s_i(h, i) = t;
		}
	}
	else {
		for (i = 0; i <= d; i++) {
			FQX.s_i(h, i) = coeffs[i];
		}
	}

	if (f_v) {
		cout << "ring_theory_global::factor_cyclotomic "
				"the polynomial is: ";
		FQX.print_object(h, cout);
		cout << endl;
	}


	FQX.create_object_of_degree(quo, d);
	FQX.create_object_of_degree(rem, d);

	int *roots;
	int *roots2;
	int nb_roots = 0;
	int beta = (Q - 1) / n, Beta;

	Beta = FQ.alpha_power(beta);

	if (f_v) {
		cout << "ring_theory_global::factor_cyclotomic "
				"the primitive n-th root of unity we choose "
				"is beta = alpha^" << beta << " = " << Beta << endl;
	}

	roots = NEW_int(n);
	roots2 = NEW_int(n);
	for (a = 0; a < n; a++) {
		FQX.create_object_of_degree(Xma, 1);
		t = FQ.power(Beta, a);
		FQX.s_i(Xma, 0) = FQ.negate(t);
		FQX.s_i(Xma, 1) = 1;
		FQX.division_with_remainder(h, Xma, quo, rem, 0);
		b = FQX.s_i(rem, 0);
		if (b == 0) {
			cout << "ring_theory_global::factor_cyclotomic "
					"zero Beta^" << a << " log "
				<< FQ.log_alpha(t) << endl;
			roots[nb_roots++] = a;
		}
	}

	exit(1);

	longinteger_domain D;
	longinteger_object C, N, A, B, G, U, V;
	other::data_structures::sorting Sorting;

	for (c = 0; c < n; c++) {
		if (NT.gcd_lint(c, n) != 1)
			continue;
		C.create(c);
		N.create(n);
		D.extended_gcd(C, N, G, U, V, false);
		cv = U.as_int();
		ccv= c * cv;
		cout << c << " : " << cv << " : ";
		if (ccv < 0) {
			if ((-ccv % n) != n - 1) {
				cout << "ring_theory_global::factor_cyclotomic "
						"error: c=" << c << " cv=" << cv << endl;
				exit(1);
			}
		}
		else if ((ccv % n) != 1) {
			cout << "ring_theory_global::factor_cyclotomic "
					"error: c=" << c << " cv=" << cv << endl;
			exit(1);
		}
		for (i = 0; i < nb_roots; i++) {
			roots2[i] = (cv * roots[i]) % n;
			while (roots2[i] < 0) {
				roots2[i] += n;
			}
		}
		Sorting.int_vec_quicksort_increasingly(roots2, nb_roots);
		t = 0;
		for (i = 0; i < nb_roots; i++) {
			r1 = roots2[i];
			for (j = i + 1; j < i + nb_roots; j++) {
				r2 = roots2[j % nb_roots];
				if (r2 != r1 + 1) {
					break;
				}
			}
			len = j - i - 1;
			t = MAXIMUM(t, len);
		}
		for (i = 0; i < nb_roots; i++) {
			cout << roots2[i] << " ";
		}
		cout << " : " << t << endl;
	}
}

void ring_theory_global::oval_polynomial(
		algebra::field_theory::finite_field *F,
	int *S, unipoly_domain &D, unipoly_object &poly,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, v[3], x; //, y;
	int *map;

	if (f_v) {
		cout << "ring_theory_global::oval_polynomial" << endl;
	}
	map = NEW_int(F->q);
	for (i = 0; i < F->q; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[2 + i]);
		if (v[2] != 1) {
			cout << "ring_theory_global::oval_polynomial "
					"not an affine point" << endl;
			exit(1);
		}
		x = v[0];
		//y = v[1];
		//cout << "map[" << i << "] = " << xx << endl;
		map[i] = x;
	}
	if (f_v) {
		cout << "the map" << endl;
		for (i = 0; i < F->q; i++) {
			cout << map[i] << " ";
		}
		cout << endl;
	}

	D.create_Dickson_polynomial(poly, map);

	FREE_int(map);
	if (f_v) {
		cout << "ring_theory_global::oval_polynomial done" << endl;
	}
}

void ring_theory_global::print_longinteger_after_multiplying(
		std::ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len, 0 /* verbose_level */);
	ost << a;
}






#if 0
void finite_field::do_ideal(int n,
		long int *set_in, int set_size, int degree,
		long int *&set_out, int &size_out,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	homogeneous_polynomial_domain *HPD;
	int *Kernel;
	int *w1;
	int *w2;
	long int *Pts;
	int nb_pts;
	int r, h, ns;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::do_ideal" << endl;
	}

	size_out = 0;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "finite_field::do_ideal before HPD->init" << endl;
	}
	HPD->init(this, n + 1, degree,
		false /* f_init_incidence_structure */,
		Monomial_ordering_type,
		verbose_level - 2);
	if (f_v) {
		cout << "finite_field::do_ideal after HPD->init" << endl;
	}

	Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
	w1 = NEW_int(HPD->get_nb_monomials());
	w2 = NEW_int(HPD->get_nb_monomials());

	if (f_v) {
		cout << "finite_field::do_ideal before HPD->vanishing_ideal" << endl;
	}
	HPD->vanishing_ideal(set_in, set_size,
			r, Kernel, 0 /*verbose_level */);
	if (f_v) {
		cout << "finite_field::do_ideal after HPD->vanishing_ideal" << endl;
	}

	ns = HPD->get_nb_monomials() - r; // dimension of null space
	cout << "The system has rank " << r << endl;
	cout << "The ideal has dimension " << ns << endl;
	cout << "and is generated by:" << endl;
	Int_matrix_print(Kernel, ns, HPD->get_nb_monomials());
	cout << "corresponding to the following basis "
			"of polynomials:" << endl;
	for (h = 0; h < ns; h++) {
		HPD->print_equation(cout, Kernel + h * HPD->get_nb_monomials());
		cout << endl;
		}

	cout << "looping over all generators of the ideal:" << endl;
	for (h = 0; h < ns; h++) {
		cout << "generator " << h << " / " << ns << ":" << endl;

		vector<long int> Points;
		int i;

		HPD->enumerate_points(Kernel + h * HPD->get_nb_monomials(),
				Points, verbose_level);
		nb_pts = Points.size();

		Pts = NEW_lint(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			Pts[i] = Points[i];
		}


		cout << "We found " << nb_pts << " points on the generator of the ideal" << endl;
		cout << "They are : ";
		Lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);


		if (h == 0) {
			size_out = HPD->get_nb_monomials();
			set_out = NEW_lint(size_out);
			//int_vec_copy(Kernel + h * HPD->nb_monomials, set_out, size_out);
			int u;
			for (u = 0; u < size_out; u++) {
				set_out[u] = Kernel[h * HPD->get_nb_monomials() + u];
			}
			//break;
		}
		FREE_lint(Pts);

	}

#if 0
	int N;
	int *Pts;
	cout << "looping over all elements of the ideal:" << endl;
	N = Gg.nb_PG_elements(ns - 1, q);
	for (h = 0; h < N; h++) {
		PG_element_unrank_modified(w1, 1, ns, h);
		cout << "element " << h << " / " << N << " w1=";
		int_vec_print(cout, w1, ns);
		mult_vector_from_the_left(w1, Kernel, w2, ns, HPD->get_nb_monomials());
		cout << " w2=";
		int_vec_print(cout, w2, HPD->get_nb_monomials());
		HPD->enumerate_points(w2, Pts, nb_pts, verbose_level);
		cout << " We found " << nb_pts << " points on this curve" << endl;
		}
#endif

	FREE_OBJECT(HPD);
	FREE_int(Kernel);
	FREE_int(w1);
	FREE_int(w2);
}
#endif

void ring_theory_global::parse_equation_easy(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &equation_text,
		int *&coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::parse_equation_easy" << endl;
	}

	std::string name_of_formula;
	std::string name_of_formula_tex;

	name_of_formula = "formula";
	name_of_formula_tex = "formula";

	string managed_variables;
	string equation_parameters;
	string equation_parameters_tex;
	string equation_parameter_values;

	int nb_coeffs;
	int f_has_managed_variables = true;

	managed_variables = Poly->list_of_variables();
	if (f_v) {
		cout << "ring_theory_global::parse_equation_easy "
				"managed_variables = " << managed_variables << endl;
	}

	parse_equation(Poly,
			name_of_formula,
			name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			equation_text,
			equation_parameters,
			equation_parameters_tex,
			equation_parameter_values,
			coeffs, nb_coeffs,
			verbose_level - 1);

	if (nb_coeffs != Poly->get_nb_monomials()) {
		cout << "ring_theory_global::parse_equation_easy "
				"nb_coeffs != Poly->get_nb_monomials()" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation_easy done" << endl;
	}
}


void ring_theory_global::parse_equation_with_parameters(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		int *&coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::parse_equation_with_parameters" << endl;
	}

	std::string name_of_formula;
	std::string name_of_formula_tex;

	name_of_formula = "formula";
	name_of_formula_tex = "formula";

	string managed_variables;

	int nb_coeffs;
	int f_has_managed_variables = true;

	managed_variables = Poly->list_of_variables();
	if (f_v) {
		cout << "ring_theory_global::parse_equation_with_parameters "
				"managed_variables = " << managed_variables << endl;
	}

	parse_equation(Poly,
			name_of_formula,
			name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			equation_text,
			equation_parameters,
			equation_parameters_tex,
			equation_parameter_values,
			coeffs, nb_coeffs,
			verbose_level - 1);

	if (nb_coeffs != Poly->get_nb_monomials()) {
		cout << "ring_theory_global::parse_equation_with_parameters "
				"nb_coeffs != Poly->get_nb_monomials()" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation_with_parameters done" << endl;
	}
}

void ring_theory_global::parse_equation(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		int *&coeffs, int &nb_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::parse_equation" << endl;
		cout << "ring_theory_global::parse_equation "
				"name_of_formula=" << name_of_formula << endl;
		cout << "ring_theory_global::parse_equation "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		cout << "ring_theory_global::parse_equation "
				"f_has_managed_variables=" << f_has_managed_variables << endl;
		cout << "ring_theory_global::parse_equation "
				"managed_variables=" << managed_variables << endl;
		cout << "ring_theory_global::parse_equation "
				"equation_text=" << equation_text << endl;
		cout << "ring_theory_global::parse_equation "
				"equation_parameters=" << equation_parameters << endl;
		cout << "ring_theory_global::parse_equation "
				"equation_parameters_tex=" << equation_parameters_tex << endl;
		cout << "ring_theory_global::parse_equation "
				"equation_parameter_values=" << equation_parameter_values << endl;
	}


	string label_txt;
	string label_tex;

	label_txt = name_of_formula + "_" + equation_parameters;
	label_tex = name_of_formula_tex + "\\_" + equation_parameters_tex;

	// create a symbolic object containing the general formula:

	algebra::expression_parser::symbolic_object_builder_description *Descr1;


	Descr1 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
	Descr1->f_field_pointer = true;
	Descr1->field_pointer = Poly->get_F();
	Descr1->f_text = true;
	Descr1->text_txt = equation_text;
	Descr1->f_managed_variables = f_has_managed_variables;
	Descr1->managed_variables = managed_variables;




	algebra::expression_parser::symbolic_object_builder *SB1;

	SB1 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);


	string s1;

	s1 = name_of_formula + "_raw";

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"before SB1->init" << endl;
	}

	SB1->init(Descr1, s1, verbose_level - 1);

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"after SB1->init" << endl;
	}

	algebra::expression_parser::formula_vector *Formula_vector_after_sub;


	if (equation_parameter_values.length()) {

		// create a second symbolic object containing the specific values
		// to be substituted.


		if (f_v) {
			cout << "ring_theory_global::parse_equation "
					"substituting parameter values" << endl;
			cout << "ring_theory_global::parse_equation "
					"equation_parameters = " << equation_parameters << endl;
			cout << "ring_theory_global::parse_equation "
					"equation_parameter_values = " << equation_parameter_values << endl;
		}


		if (f_v) {
			cout << "ring_theory_global::parse_equation "
					"before perform_substitution" << endl;
		}

		perform_substitution(
				Poly,
				name_of_formula,
				name_of_formula_tex,
				f_has_managed_variables,
				managed_variables,
				equation_parameters,
				equation_parameter_values,
				SB1,
				Formula_vector_after_sub,
				verbose_level - 2);

		if (f_v) {
			cout << "ring_theory_global::parse_equation "
					"after perform_substitution" << endl;
		}

	}
	else {

		if (f_v) {
			cout << "ring_theory_global::parse_equation "
					"no parameter values" << endl;
		}

		Formula_vector_after_sub = SB1->Formula_vector;
	}
	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"Formula_vector_after_sub->f_has_managed_variables = "
				<< Formula_vector_after_sub->f_has_managed_variables << endl;
	}


	algebra::expression_parser::formula_vector *Formula_vector_after_expand;

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"before simplify_and_expand" << endl;
	}

	simplify_and_expand(
			Poly,
			name_of_formula,
			name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			Formula_vector_after_sub,
			Formula_vector_after_expand,
			verbose_level - 2);

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"after simplify_and_expand" << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"Formula_vector_after_expand->f_has_managed_variables = "
				<< Formula_vector_after_expand->f_has_managed_variables << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"before collect_variables" << endl;
	}
	Formula_vector_after_expand->V[0].collect_variables(
			verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"after collect_variables" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"Formula_vector_after_expand->f_has_managed_variables = "
				<< Formula_vector_after_expand->f_has_managed_variables << endl;
	}

	// assemble the equation as a vector of coefficients
	// in the ordering of the polynomial ring:

	//int *coeffs;
	//int nb_coeffs;

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"before Formula_vector_after_expand->V[0].collect_coefficients_of_equation" << endl;
	}
	Formula_vector_after_expand->V[0].collect_coefficients_of_equation(
			Poly,
			coeffs, nb_coeffs,
			verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"after Formula_vector_after_expand->V[0].collect_coefficients_of_equation" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation "
				"Formula_vector_after_expand->f_has_managed_variables = "
				<< Formula_vector_after_expand->f_has_managed_variables << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::parse_equation done" << endl;
	}


}

void ring_theory_global::simplify_and_expand(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		algebra::expression_parser::formula_vector *Formula_vector_after_sub,
		algebra::expression_parser::formula_vector *&Formula_vector_after_expand,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand" << endl;
	}
	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"Formula_vector_after_sub->f_has_managed_variables = "
				<< Formula_vector_after_sub->f_has_managed_variables << endl;
	}

	// Perform simplification

	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"before Formula_vector_after_sub->V[0].simplify" << endl;
	}
	Formula_vector_after_sub->V[0].simplify(verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"after Formula_vector_after_sub->V[0].simplify" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"Formula_vector_after_sub->f_has_managed_variables = "
				<< Formula_vector_after_sub->f_has_managed_variables << endl;
	}


	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand



	Formula_vector_after_expand = NEW_OBJECT(algebra::expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector_after_expand->expand(
			Formula_vector_after_sub,
			Poly->get_F(),
			name_of_formula, name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			f_write_trees_during_expand,
			verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"after Formula_vector->expand" << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"Formula_vector_after_expand->f_has_managed_variables = "
				<< Formula_vector_after_expand->f_has_managed_variables << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand "
				"Formula_vector_after_expand->f_has_managed_variables = "
				<< Formula_vector_after_expand->f_has_managed_variables << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::simplify_and_expand done" << endl;
	}

}

void ring_theory_global::perform_substitution(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		std::string &equation_parameters,
		std::string &equation_parameter_values,
		algebra::expression_parser::symbolic_object_builder *SB1,
		algebra::expression_parser::formula_vector *&Formula_vector_after_sub,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::perform_substitution" << endl;
	}

	// create a second symbolic object containing the specific values
	// to be substituted.

	algebra::expression_parser::symbolic_object_builder_description *Descr2;


	Descr2 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder_description);
	Descr2->f_field_pointer = true;
	Descr2->field_pointer = Poly->get_F();
	Descr2->f_text = true;
	Descr2->text_txt = equation_parameter_values;
	Descr2->f_managed_variables = f_has_managed_variables;
	Descr2->managed_variables = managed_variables;



	algebra::expression_parser::symbolic_object_builder *SB2;

	SB2 = NEW_OBJECT(algebra::expression_parser::symbolic_object_builder);

	string s2;

	s2 = name_of_formula + "_param_values";


	if (f_v) {
		cout << "ring_theory_global::perform_substitution "
				"before SB2->init" << endl;
	}

	SB2->init(Descr2, s2, verbose_level);

	if (f_v) {
		cout << "ring_theory_global::perform_substitution "
				"after SB2->init" << endl;
	}


	// Perform the substitution.
	// Create temporary object Formula_vector_after_sub

	algebra::expression_parser::symbolic_object_builder *O_target = SB1;
	algebra::expression_parser::symbolic_object_builder *O_source = SB2;

	//O_target = Get_symbol(Descr->substitute_target);
	//O_source = Get_symbol(Descr->substitute_source);




	Formula_vector_after_sub = NEW_OBJECT(algebra::expression_parser::formula_vector);

	if (f_v) {
		cout << "ring_theory_global::perform_substitution "
				"before Formula_vector_after_sub->substitute" << endl;
	}
	Formula_vector_after_sub->substitute(
			O_source->Formula_vector,
			O_target->Formula_vector,
			equation_parameters /*Descr->substitute_variables*/,
			name_of_formula, name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			verbose_level - 2);
	if (f_v) {
		cout << "ring_theory_global::perform_substitution "
				"after Formula_vector_after_sub->substitute" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::perform_substitution done" << endl;
	}
}

void ring_theory_global::test_unipoly(
		algebra::field_theory::finite_field *F)
{
	ring_theory::unipoly_object m, a, b, c;
	ring_theory::unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;

	ring_theory::unipoly_domain FX(F);

	FX.create_object_by_rank(m, 7, 0);
	FX.create_object_by_rank(a, 5, 0);
	FX.create_object_by_rank(b, 55, 0);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	ring_theory::unipoly_domain Fq(F, m, verbose_level);
	Fq.create_object_by_rank(c, 2, 0);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i, 0);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c, verbose_level);
			Fq.print_object(c, cout); cout << endl;

			FX.mult(elts[i], elts[j], a, verbose_level);
			FX.print_object(a, cout); cout << endl;
		}
	}

}

void ring_theory_global::test_unipoly2(
		algebra::field_theory::finite_field *F)
{
	int i;

	ring_theory::unipoly_domain FX(F);

	ring_theory::unipoly_object a;

	FX.create_object_by_rank(a, 0, 0);
	for (i = 1; i < F->q; i++) {
		FX.minimum_polynomial(a, i, F->p, true);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}

}

void ring_theory_global::test_longinteger()
{
	ring_theory::longinteger_domain D;
	int x[] = {15, 14, 12, 8};
	ring_theory::longinteger_object a, b, q, r;
	int verbose_level = 0;

	D.multiply_up(a, x, 4, verbose_level);
	cout << "a=" << a << endl;
	b.create(2);
	while (!a.is_zero()) {
		D.integral_division(a, b, q, r, verbose_level);
		//cout << a << " = " << q << " * " << b << " + " << r << endl;
		cout << r << endl;
		q.assign_to(a);
	}

	D.multiply_up(a, x, 4, verbose_level);
	cout << "a=" << a << endl;

	int *rep, len;
	D.base_b_representation(a, 2, rep, len);
	b.create_from_base_b_representation(2, rep, len);
	cout << "b=" << b << endl;
	FREE_int(rep);
}

void ring_theory_global::test_longinteger2()
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, c, d, e;
	int r;
	int verbose_level = 0;

	a.create_from_base_10_string("562949953421311", verbose_level);
	D.integral_division_by_int(a, 127, b, r);
	cout << a << " = " << b << " * 127 + " << r << endl;
	c.create_from_base_10_string("270549121", verbose_level);
	D.integral_division(b, c, d, e, verbose_level);
	cout << b << " = " << d << " * " << c << " + " << e << endl;
}

void ring_theory_global::test_longinteger3()
{
	int i, j;
	combinatorics::other_combinatorics::combinatorics_domain D;
	ring_theory::longinteger_object a, b, c, d, e;

	for (i = 0; i < 10; i++) {
		for (j = 0; j < 10; j++) {
			D.binomial(a, i, j, false);
			a.print(cout);
			cout << " ";
		}
		cout << endl;
	}
}

void ring_theory_global::test_longinteger4()
{
	int n = 6, q = 2, k, x, d = 3;
	combinatorics::other_combinatorics::combinatorics_domain D;
	ring_theory::longinteger_object a;

	for (k = 0; k <= n; k++) {
		for (x = 0; x <= n; x++) {
			if (x > 0 && x < d) {
				continue;
			}
			if (q == 2 && EVEN(d) && ODD(x)) {
				continue;
			}
			D.krawtchouk(a, n, q, k, x);
			a.print(cout);
			cout << " ";
		}
		cout << endl;
	}
}

void ring_theory_global::test_longinteger5()
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, u, v, g;
	int verbose_level = 2;

	a.create(9548);
	b.create(254774);
	D.extended_gcd(a, b, g, u, v, verbose_level);

	g.print(cout);
	cout << " = ";
	u.print(cout);
	cout << " * ";
	a.print(cout);
	cout << " + ";
	v.print(cout);
	cout << " * ";
	b.print(cout);
	cout << endl;

}

void ring_theory_global::test_longinteger6()
{
	int verbose_level = 2;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b;

	a.create(7411);
	b.create(9283);
	D.jacobi(a, b, verbose_level);


}

void ring_theory_global::test_longinteger7()
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b;
	int i, j;
	int mult[15];

	a.create(15);
	for (i = 0; i < 15; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < 10000; i++) {
		D.random_number_less_than_n(a, b);
		j = b.as_int();
		mult[j]++;
		//cout << b << endl;
	}
	for (i = 0; i < 15; i++) {
		cout << i << " : " << mult[i] << endl;
	}

}

void ring_theory_global::test_longinteger8()
{
	int verbose_level = 2;
	combinatorics::cryptography::cryptography_domain Crypto;
	ring_theory::longinteger_object a, b, one;
	int nb_solovay_strassen_tests = 100;
	int f_miller_rabin_test = true;

	one.create(1);
	a.create(197659);
	Crypto.find_probable_prime_above(a, nb_solovay_strassen_tests,
		f_miller_rabin_test, verbose_level);
}

void ring_theory_global::longinteger_collect_setup(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities)
{
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
}

void ring_theory_global::longinteger_collect_free(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities)
{
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
	}
}

void ring_theory_global::longinteger_collect_add(
		int &nb_agos,
		ring_theory::longinteger_object *&agos, int *&multiplicities,
		ring_theory::longinteger_object &ago)
{
	int j, c, h, f_added;
	ring_theory::longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	ring_theory::longinteger_domain D;

	f_added = false;
	for (j = 0; j < nb_agos; j++) {
		c = D.compare_unsigned(ago, agos[j]);
		//cout << "comparing " << ago << " with "
		//<< agos[j] << " yields " << c << endl;
		if (c >= 0) {
			if (c == 0) {
				multiplicities[j]++;
			}
			else {
				tmp_agos = agos;
				tmp_multiplicities = multiplicities;
				agos = NEW_OBJECTS(ring_theory::longinteger_object, nb_agos + 1);
				multiplicities = NEW_int(nb_agos + 1);
				for (h = 0; h < j; h++) {
					tmp_agos[h].swap_with(agos[h]);
					multiplicities[h] = tmp_multiplicities[h];
				}
				ago.swap_with(agos[j]);
				multiplicities[j] = 1;
				for (h = j; h < nb_agos; h++) {
					tmp_agos[h].swap_with(agos[h + 1]);
					multiplicities[h + 1] = tmp_multiplicities[h];
				}
				nb_agos++;
				if (tmp_agos) {
					FREE_OBJECTS(tmp_agos);
					FREE_int(tmp_multiplicities);
				}
			}
			f_added = true;
			break;
		}
	}
	if (!f_added) {
		// add at the end (including the case that the list is empty)
		tmp_agos = agos;
		tmp_multiplicities = multiplicities;
		agos = NEW_OBJECTS(ring_theory::longinteger_object, nb_agos + 1);
		multiplicities = NEW_int(nb_agos + 1);
		for (h = 0; h < nb_agos; h++) {
			tmp_agos[h].swap_with(agos[h]);
			multiplicities[h] = tmp_multiplicities[h];
		}
		ago.swap_with(agos[nb_agos]);
		multiplicities[nb_agos] = 1;
		nb_agos++;
		if (tmp_agos) {
			FREE_OBJECTS(tmp_agos);
			FREE_int(tmp_multiplicities);
		}
	}
}

void ring_theory_global::longinteger_collect_print(
		std::ostream &ost,
		int &nb_agos,
		ring_theory::longinteger_object *&agos,
		int *&multiplicities)
{
	int j;

	ost << "(";
	for (j = 0; j < nb_agos; j++) {
		ost << agos[j];
		if (multiplicities[j] == 1) {
		}
		else if (multiplicities[j] >= 10) {
			ost << "^{" << multiplicities[j] << "}";
		}
		else  {
			ost << "^" << multiplicities[j];
		}
		if (j < nb_agos - 1) {
			ost << ", ";
		}
	}
	ost << ")" << endl;
}


void ring_theory_global::make_table_of_monomials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::make_table_of_monomials" << endl;
	}


	algebra::expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(name_of_formula);

	if (Object->Formula_vector->len != Poly->get_nb_monomials()) {
		cout << "ring_theory_global::make_table_of_monomials "
				"the number of objects must match the number "
				"of monomials in the polynomial ring." << endl;
		cout << "Object->Formula_vector->len = " << Object->Formula_vector->len << endl;
		cout << "nb_monomials=" << Poly->get_nb_monomials() << endl;
		exit(1);
	}

	string *Table;
	int nb_cols;
	int nb_rows;
	int i;
	int f_latex = false;

	nb_rows = Poly->get_nb_monomials();
	nb_cols = 3;

	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		Table[i * nb_cols + 0] =
				Poly->get_monomial_symbols_latex(i);
		Table[i * nb_cols + 1] =
				Poly->get_monomial_symbols(i);
		Table[i * nb_cols + 2] =
				Object->Formula_vector->V[i].string_representation_formula(
						f_latex, 0 /*verbose_level*/);
	}

	std::string Col_headings[3];

	Col_headings[0] = "Mono";
	Col_headings[1] = "Mono";
	Col_headings[2] = "Coeff";
	string fname;

	fname = name_of_formula + "_coefficients.csv";

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "ring_theory_global::make_table_of_monomials "
				"written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "ring_theory_global::make_table_of_monomials done" << endl;
	}
}

std::string ring_theory_global::stringify_expression(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::stringify_expression" << endl;
	}


	algebra::expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(name_of_formula);

	string s;
	string s_monomial;
	string s_coeff;
	int f_first = true;

	int i;
	int nb_monomials;

	nb_monomials = Poly->get_nb_monomials();

	if (Object->Formula_vector->len != nb_monomials) {
		cout << "ring_theory_global::stringify_expression "
				"Object->Formula_vector->len != nb_monomials, using default printer instead" << endl;
		for (i = 0; i < Object->Formula_vector->len; i++) {
			s_coeff =
					Object->Formula_vector->V[i].string_representation_formula(
							false /*f_latex*/, 0 /*verbose_level*/);
			if (s_coeff.length()) {

				if (!f_first) {
					s += ", \n";
				}
				s += s_coeff;
				f_first = false;
			}
		}
		return s;
	}
	for (i = 0; i < Object->Formula_vector->len; i++) {
		s_monomial =
				Poly->get_monomial_symbols(i);
		s_coeff =
				Object->Formula_vector->V[i].string_representation_formula(
						false /*f_latex*/, 0 /*verbose_level*/);
		if (s_coeff.length()) {

			if (!f_first) {
				s += " + ";
			}
			s += "(" + s_coeff + ") * " + s_monomial;
			f_first = false;
		}
	}

	if (f_v) {
		cout << "ring_theory_global::stringify_expression done" << endl;
	}
	return s;
}

std::string ring_theory_global::stringify_expression_latex(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::stringify_expression_latex" << endl;
	}


	algebra::expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(name_of_formula);

	string s;
	string s_monomial;
	string s_coeff;
	int f_first = true;

	int i;
	int nb_monomials;

	nb_monomials = Poly->get_nb_monomials();

	if (Object->Formula_vector->len != nb_monomials) {
		cout << "ring_theory_global::stringify_expression_latex "
				"Object->Formula_vector->len != nb_monomials, using default printer instead" << endl;
		for (i = 0; i < Object->Formula_vector->len; i++) {
			s_coeff =
					Object->Formula_vector->V[i].string_representation_formula(
							false /*f_latex*/, 0 /*verbose_level*/);
			if (s_coeff.length()) {

				if (!f_first) {
					s += ", \n";
				}
				s += s_coeff;
				f_first = false;
			}
		}
		return s;
	}
	for (i = 0; i < Object->Formula_vector->len; i++) {
		s_monomial =
				Poly->get_monomial_symbols(i);
		s_coeff =
				Object->Formula_vector->V[i].string_representation_formula(
						true /*f_latex*/, 0 /*verbose_level*/);
		if (s_coeff.length()) {

			if (!f_first) {
				s += " + ";
			}
			s += "(" + s_coeff + ")  " + s_monomial;
			f_first = false;
		}
	}

	if (f_v) {
		cout << "ring_theory_global::stringify_expression_latex done" << endl;
	}
	return s;
}



void ring_theory_global::do_export_partials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ring_theory_global::do_export_partials" << endl;
	}

	homogeneous_polynomial_domain *Poly_reduced_degree;
	//int *gradient;

	Poly_reduced_degree = NEW_OBJECT(homogeneous_polynomial_domain);


	if (f_v) {
		cout << "ring_theory_global::do_export_partials "
				"before Poly_reduced_degree->init" << endl;
	}
	Poly_reduced_degree->init(
			Poly->get_F(),
			Poly->nb_variables, Poly->degree - 1,
			Poly->Monomial_ordering_type,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "ring_theory_global::do_export_partials "
				"after Poly_reduced_degree->init" << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::do_export_partials "
				"before Poly->compute_and_export_partials" << endl;
	}
	Poly->compute_and_export_partials(
			Poly_reduced_degree,
			verbose_level);
	if (f_v) {
		cout << "ring_theory_global::do_export_partials "
				"after Poly->compute_and_export_partials" << endl;
	}

	FREE_OBJECT(Poly_reduced_degree);

	if (f_v) {
		cout << "ring_theory_global::do_export_partials done" << endl;
	}
}


}}}}



