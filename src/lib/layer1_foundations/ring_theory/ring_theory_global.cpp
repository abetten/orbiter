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
namespace ring_theory {


ring_theory_global::ring_theory_global()
{

}

ring_theory_global::~ring_theory_global()
{

}

void ring_theory_global::Monomial_ordering_type_as_string(
		monomial_ordering_type Monomial_ordering_type, std::string &s)
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
		field_theory::finite_field *F,
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

	fname_code.assign("crc_");
	fname_code.append(label_code);
	fname_code.append(".cpp");

	{

		std::ofstream ost(fname_code);


		string str;
		orbiter_kernel_system::os_interface Os;

		Os.get_date(str);

		string label_of_parameters;
		string name_of_function;
		string name_of_array_of_polynomials;
		char str2[1024];






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


		name_of_function.assign("divide");
		name_of_function.append("_");
		name_of_function.append(label_code);


		snprintf(str2, sizeof(str2), "q%d_n%d_r%d", F->q, da, db);

		label_of_parameters.assign(str2);



		name_of_array_of_polynomials.assign("crc_poly_table_");
		name_of_array_of_polynomials.append(label_code);




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
			cout << "ring_theory_global::write_code_for_division da != FX.degree(poly_A)" << endl;
			exit(1);
		}
		if (db != FX.degree(poly_B)) {
			cout << "ring_theory_global::write_code_for_division db != FX.degree(poly_B)" << endl;
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
		ost << "\tfor (i = 0; i <= " << db << "; i++) {" << endl;
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

		ost << "\tfor (i = " << db << "; i >= 0; i--) {" << endl;
		ost << "\t\tout[i] = R[i];" << endl;
		ost << "\t}" << endl;

		ost << "}" << endl;

	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_code << " of size " << Fio.file_size(fname_code) << endl;

	if (f_v) {
		cout << "ring_theory_global::write_code_for_division done" << endl;
	}
}


void ring_theory_global::polynomial_division(
		field_theory::finite_field *F,
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
		field_theory::finite_field *F,
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


	number_theory::number_theory_domain NT;




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
		field_theory::finite_field *F,
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

	number_theory::number_theory_domain NT;




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
		cout << "ring_theory_global::polynomial_mult_mod before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(A, B, C, M, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_mult_mod after FX.mult_mod" << endl;
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
		field_theory::finite_field *F,
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
	data_structures::string_tools ST;


	Get_int_vector_from_label(A_coeffs, data_A, sz_A, verbose_level);


	n = ST.strtolint(power_text);

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod n = " << n << endl;
	}


	Get_int_vector_from_label(M_coeffs, data_M, sz_M, verbose_level);

	number_theory::number_theory_domain NT;




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
		cout << "ring_theory_global::polynomial_power_mod before FX.mult_mod" << endl;
	}

	{
		FX.power_mod(A, M, n, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_power_mod after FX.mult_mod" << endl;
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
		field_theory::finite_field *F,
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

	number_theory::number_theory_domain NT;




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
		cout << "ring_theory_global::polynomial_find_roots before FX.mult_mod" << endl;
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
		cout << "ring_theory_global::polynomial_find_roots after FX.mult_mod" << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_find_roots done" << endl;
	}
}

void ring_theory_global::sift_polynomials(
		field_theory::finite_field *F,
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
				__FILE__, __LINE__, 0 /* verbose_level*/);
		if (f_v) {
			cout << "rk=" << rk << " poly=";
			D.print_object(p, cout);
			cout << endl;
		}


		int nb_primes;
		longinteger_object *primes;
		int *exponents;


		f = D.degree(p);
		Q.create(F->q, __FILE__, __LINE__);
		m1.create(-1, __FILE__, __LINE__);
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
				__FILE__, __LINE__, 0 /* verbose_level*/);
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
		field_theory::finite_field *F,
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


		char str[1000];

		snprintf(str, 1000, "polynomial_mult_%ld_%ld.tex", rk0, rk1);
		fname.assign(str);
		snprintf(str, 1000, "Polynomial Mult");
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
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
		orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::mult_polynomials written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::mult_polynomials done" << endl;
	}

}


void ring_theory_global::polynomial_division_with_report(
		field_theory::finite_field *F,
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


		char str[1000];

		snprintf(str, 1000, "polynomial_division_%ld_%ld.tex", rk0, rk1);
		fname.assign(str);
		snprintf(str, 1000, "Polynomial Division");
		title.assign(str);




		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk2, rk3;
			D.division_with_remainder_numerically_with_report(rk0, rk1, rk2, rk3, ost, verbose_level);
			ost << "$" << rk0 << " / " << rk1 << " = " << rk2 << "$ Remainder $" << rk3 << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::polynomial_division_with_report done" << endl;
	}

}

void ring_theory_global::polynomial_division_from_file_with_report(
		field_theory::finite_field *F,
		std::string &input_file, long int rk1, int verbose_level)
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


		char str[1000];

		snprintf(str, 1000, "polynomial_division_file_%ld.tex", rk1);
		fname.assign(str);
		snprintf(str, 1000, "Polynomial Division");
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int rk_q, rk_r;

			D.division_with_remainder_from_file_with_report(input_file, rk1,
					rk_q, rk_r, ost, verbose_level);

			ost << "$ / " << rk1 << " = " << rk_q << "$ Remainder $"
					<< rk_r << "$\\\\" << endl;


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_with_report "
						"after division_with_remainder_numerically_with_report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_from_file_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_with_report done" << endl;
	}

}

void ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report(
		field_theory::finite_field *F,
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


		char str[1000];

		snprintf(str, 1000, "polynomial_division_file_all_%d_error_patterns_%ld.tex", k, rk1);
		fname.assign(str);
		snprintf(str, 1000, "Polynomial Division");
		title.assign(str);



		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report "
						"before division_with_remainder_numerically_with_report" << endl;
			}
			//report(ost, verbose_level);

			long int *rk_q, *rk_r;
			int N, n, h;
			combinatorics::combinatorics_domain Combi;

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
		orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::polynomial_division_from_file_all_k_error_patterns_with_report written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "ring_theory_global::polynomial_division_from_file_with_report done" << endl;
	}

}

void ring_theory_global::create_irreducible_polynomial(
		field_theory::finite_field *F,
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
		cout << "ring_theory_global::create_irreducible_polynomial before allocating min_poly etc" << endl;
	}

	int degree = cylotomic_set_size;

	coding_theory::cyclic_codes Cyclic_codes;

	min_poly = NEW_OBJECTS(unipoly_object, degree + 2);
	unipoly_object *tmp = NEW_OBJECTS(unipoly_object, degree + 1);
	unipoly_object *linear_factor = NEW_OBJECTS(unipoly_object, 2);
	unipoly_object Pc, Pd;

	int i, j, h, r;

	// create the polynomial linear_factor = X - a:
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial creating linear_factor = X-a" << endl;
	}
	for (i = 0; i < 2; i++) {
		if (i == 1) {
			Fq->create_object_by_rank(linear_factor[i], 1, __FILE__, __LINE__, verbose_level);
		}
		else {
			Fq->create_object_by_rank(linear_factor[i], 0, __FILE__, __LINE__, verbose_level);
		}
	}
	for (i = 0; i <= degree; i++) {
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial creating generator[" << i << "]" << endl;
		}
		Fq->create_object_by_rank(min_poly[i], 0, __FILE__, __LINE__, verbose_level);
		Fq->create_object_by_rank(tmp[i], 0, __FILE__, __LINE__, verbose_level);
	}
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial creating generator[0]" << endl;
	}
	Fq->create_object_by_rank(min_poly[0], 1, __FILE__, __LINE__, verbose_level);

	// now coeffs has degree 1
	// and generator has degree 0

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial coeffs:" << endl;
		Cyclic_codes.print_polynomial(*Fq, 1, linear_factor);
		cout << endl;
		cout << "ring_theory_global::create_irreducible_polynomial generator:" << endl;
		Cyclic_codes.print_polynomial(*Fq, 0, min_poly);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial creating Pc" << endl;
	}
	Fq->create_object_by_rank(Pc, 0, __FILE__, __LINE__, verbose_level);
	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial creating Pd" << endl;
	}
	Fq->create_object_by_rank(Pd, 0, __FILE__, __LINE__, verbose_level);

	r = 0;
	for (h = 0; h < cylotomic_set_size; h++) {
		i = cyclotomic_set[h];
		if (f_v) {
			cout << "h=" << h << ", i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial before Fq.assign beta" << endl;
		}
		Fq->assign(Beta[i], linear_factor[0], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial before Fq.negate" << endl;
		}
		Fq->negate(linear_factor[0]);
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial root: " << i << " : ";
			Fq->print_object(linear_factor[0], cout);
			//cout << " : ";
			//print_polynomial(Fq, 2, coeffs);
			cout << endl;
		}


		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial before Fq.assign(min_poly[j], tmp[j])" << endl;
		}
		for (j = 0; j <= r; j++) {
			Fq->assign(min_poly[j], tmp[j], verbose_level);
		}

		//cout << "tmp:" << endl;
		//print_polynomial(Fq, r, tmp);
		//cout << endl;

		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial before Fq.assign(tmp[j], min_poly[j + 1])" << endl;
		}

		for (j = 0; j <= r; j++) {
			Fq->assign(tmp[j], min_poly[j + 1], verbose_level);
		}

		Fq->delete_object(min_poly[0]);
		Fq->create_object_by_rank(min_poly[0], 0, __FILE__, __LINE__, verbose_level);

		//cout << "generator after shifting up:" << endl;
		//print_polynomial(Fq, r + 1, generator);
		//cout << endl;

		for (j = 0; j <= r; j++) {
			if (f_v) {
				cout << "ring_theory_global::create_irreducible_polynomial j=" << j << endl;
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
				cout << "ring_theory_global::create_irreducible_polynomial before "
						"Fq.assign" << endl;
			}
			Fq->assign(Pd, min_poly[j], verbose_level);
		}
		r++;
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial r=" << r << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::create_irreducible_polynomial current polynomial: ";
			Cyclic_codes.print_polynomial(*Fq, r, min_poly);
			cout << endl;
		}

	}

	if (r != degree) {
		cout << "ring_theory_global::create_irreducible_polynomial r != degree" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial The minimum polynomial is: ";
		Cyclic_codes.print_polynomial(*Fq, r, min_poly);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::create_irreducible_polynomial done" << endl;
	}

}

void ring_theory_global::compute_nth_roots_as_polynomials(
		field_theory::finite_field *F,
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
	Fp.finite_field_init(p, FALSE /* f_without_tables */, verbose_level - 1);

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
	number_theory::number_theory_domain NT;

	m = NT.order_mod_p(F->q, n1);
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code order of q mod n is m=" << m << endl;
	}
	D.create_qnm1(Qm1, F->q, m);

	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(Qm1, n2, Index, r);
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code Index = " << Index << endl;
	}

	int subgroup_index;

	subgroup_index = Index.as_int();
	if (f_v) {
		cout << "ring_theory_global::make_cyclic_code subgroup_index = " << subgroup_index << endl;
	}

	//b = (q - 1) / n;
	if (r != 0) {
		cout << "ring_theory_global::make_cyclic_code n does not divide q^m-1" << endl;
		exit(1);
	}

#if 0
	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials choosing the following irreducible "
				"and primitive polynomial:" << endl;
		FpX.print_object(M, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials creating unipoly_domain Fq modulo M" << endl;
	}
	//unipoly_domain Fq(this, M, verbose_level);  // Fq = Fp[X] modulo factor polynomial M
	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials extension field created" << endl;
	}
#endif

	Beta = new unipoly_object[n2];

	for (i = 0; i < n2; i++) {
		Fq->create_object_by_rank(Beta[i], 0, __FILE__, __LINE__, verbose_level);

	}

	//Fq->create_object_by_rank(c, 0, __FILE__, __LINE__, verbose_level);
	Fq->create_object_by_rank(beta, F->p, __FILE__, __LINE__, verbose_level); // the element alpha
	//Fq->create_object_by_rank(beta_i, 1, __FILE__, __LINE__, verbose_level);
	if (subgroup_index != 1) {
		//Fq.power_int(beta, b);
		if (f_v) {
			cout << "\\alpha = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials before Fq->power_int" << endl;
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
			cout << "ring_theory_global::compute_nth_roots_as_polynomials subgroup_index is one" << endl;
		}
	}


	for (i = 0; i < n2; i++) {
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials before Fq.assign beta" << endl;
		}
		Fq->assign(beta, Beta[i], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::compute_nth_roots_as_polynomials before Fq.power_int" << endl;
		}
		Fq->power_int(Beta[i], i, verbose_level);
	}


	if (f_v) {
		cout << "ring_theory_global::compute_nth_roots_as_polynomials done" << endl;
	}

}

void ring_theory_global::compute_powers(
		field_theory::finite_field *F,
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
		Fq->create_object_by_rank(Beta[i], 0, __FILE__, __LINE__, verbose_level);

	}

	Fq->create_object_by_rank(beta, F->p, __FILE__, __LINE__, verbose_level); // the element alpha

	if (start_idx != 1) {

		if (f_v) {
			cout << "\\alpha = ";
			Fq->print_object(beta, cout);
			cout << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers before Fq->power_int" << endl;
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
			cout << "ring_theory_global::compute_powers subgroup_index is one" << endl;
		}
	}


	for (i = 0; i < n; i++) {
		if (f_v) {
			cout << "ring_theory_global::compute_powers i=" << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers working on root " << i << endl;
		}
		if (f_v) {
			cout << "ring_theory_global::compute_powers before Fq.assign beta" << endl;
		}
		Fq->assign(beta, Beta[i], verbose_level);
		if (f_v) {
			cout << "ring_theory_global::compute_powers before Fq.power_int" << endl;
		}
		Fq->power_int(Beta[i], i, verbose_level);
	}

	if (f_v) {
		cout << "ring_theory_global::compute_powers done" << endl;
	}

}


void ring_theory_global::make_all_irreducible_polynomials_of_degree_d(
		field_theory::finite_field *F,
		int d,
		std::vector<std::vector<int> > &Table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory::number_theory_domain NT;

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
	combinatorics::combinatorics_domain Combi;


	FX.create_object_by_rank_string(m, poly, 0 /* verbose_level */);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"chosen irreducible polynomial m = ";
		FX.print_object(m, cout);
		cout << endl;
	}

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, 0 /* verbose_level */);

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
	FX.Frobenius_matrix(Frobenius, m, verbose_level - 2);
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
	FX.compute_normal_basis(d, Normal_basis, Frobenius, verbose_level - 1);

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"Normal_basis = " << endl;
		Int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	cnt = 0;

	Combi.int_vec_first_regular_word(v, d, F->q);
	while (TRUE) {
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, v, d);
			cout << endl;
		}

		F->Linear_algebra->mult_vector_from_the_right(Normal_basis, v, w, d, d);
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : w = ";
			Int_vec_print(cout, w, d);
			cout << endl;
		}

		FX.delete_object(g);
		FX.create_object_of_degree(g, d - 1);
		for (i = 0; i < d; i++) {
			((int *) g)[1 + i] = w[i];
		}

		FX.minimum_polynomial_extension_field(g, m, minpol, d, Frobenius,
				verbose_level - 3);
		if (f_vv) {
			cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
					"regular word " << cnt << " : v = ";
			Int_vec_print(cout, v, d);
			cout << " irreducible polynomial = ";
			FX.print_object(minpol, cout);
			cout << endl;
		}



		std::vector<int> T;

		for (i = 0; i <= d; i++) {
			T.push_back(((int *)minpol)[1 + i]);
		}
		Table.push_back(T);


		cnt++;


		if (!Combi.int_vec_next_regular_word(v, d, F->q)) {
			break;
		}

	}

	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"there are " << cnt
				<< " irreducible polynomials "
				"of degree " << d << " over " << "F_" << F->q << endl;
	}

	FREE_int(Frobenius);
	FREE_int(Normal_basis);
	FREE_int(v);
	FREE_int(w);
	FX.delete_object(m);
	FX.delete_object(g);
	FX.delete_object(minpol);


	if (f_v) {
		cout << "ring_theory_global::make_all_irreducible_polynomials_of_degree_d "
				"d=" << d << " q=" << F->q << " done" << endl;
	}
}

int ring_theory_global::count_all_irreducible_polynomials_of_degree_d(
		field_theory::finite_field *F, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int cnt;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;

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
	knowledge_base::knowledge_base K;

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

	FX.create_object_by_rank(g, 0, __FILE__, __LINE__, 0 /* verbose_level */);
	FX.create_object_by_rank(minpol, 0, __FILE__, __LINE__, 0 /* verbose_level */);

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
	while (TRUE) {
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

		FX.minimum_polynomial_extension_field(g, m, minpol, d,
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
		field_theory::finite_field *F,
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

	make_all_irreducible_polynomials_of_degree_d(F, deg,
			Table, verbose_level);

	nb = Table.size();

	cout << "The " << nb << " irreducible polynomials of "
			"degree " << deg << " over F_" << F->q << " are:" << endl;

	orbiter_kernel_system::Orbiter->Int_vec->vec_print(Table);


	int *T;
	int i, j;

	T = NEW_int(Table.size() * (deg + 1));
	for (i = 0; i < Table.size(); i++) {
		for (j = 0; j < deg + 1; j++) {
			T[i * (deg + 1) + j] = Table[i][j];
		}
	}



	{

		string fname;
		string author;
		string title;
		string extra_praeamble;


		char str[1000];

		snprintf(str, 1000, "Irred_q%d_d%d.tex", F->q, deg);
		fname.assign(str);
		snprintf(str, 1000, "Irreducible Polynomials of Degree %d over F%d", deg, F->q);
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;
			geometry::geometry_global GG;
			long int rk;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "ring_theory_global::do_make_table_of_irreducible_polynomials before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "There are " << Table.size() << " irreducible polynomials of "
					"degree " << deg << " over the field F" << F->q << ":\\\\" << endl;
			ost << "The coefficients in increasing order are:\\\\" << endl;
			ost << endl;
			ost << "\\bigskip" << endl;
			ost << endl;
			//ost << "\\begin{multicols}{2}" << endl;
			ost << "\\noindent" << endl;
			for (i = 0; i < Table.size(); i++) {
				ost << i << " : $";
				for (j = 0; j <= deg; j++) {
					ost << T[i * (deg + 1) + j];
				}
				ost << " : ";
				rk = GG.AG_element_rank(F->q, T + i * (deg + 1), 1, deg + 1);
				ost << rk;
				ost << "$\\\\" << endl;
			}
			//ost << "\\end{multicols}" << endl;


			if (f_v) {
				cout << "ring_theory_global::do_make_table_of_irreducible_polynomials after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "ring_theory_global::do_make_table_of_irreducible_polynomials written file " << fname << " of size "
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
	field_theory::finite_field Fp;

	if (f_v) {
		cout << "ring_theory_global::search_for_primitive_polynomial_of_given_degree" << endl;
	}
	Fp.finite_field_init_small_order(p,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
			0 /*verbose_level*/);
	unipoly_domain FX(&Fp);

	unipoly_object m;
	longinteger_object rk;

	FX.create_object_by_rank(m, 0, __FILE__, __LINE__, verbose_level);

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
	number_theory::number_theory_domain NT;


	orbiter_kernel_system::Orbiter->longinteger_f_print_scientific = FALSE;


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
			field_theory::finite_field Fq;
			Fq.finite_field_init_small_order(q,
					FALSE /* f_without_tables */,
					FALSE /* f_compute_related_fields */,
					0 /*verbose_level*/);
			unipoly_domain FX(&Fq);

			unipoly_object m;
			longinteger_object rk;

			FX.create_object_by_rank(m, 0, __FILE__, __LINE__, verbose_level);

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
	field_theory::finite_field FQ;
	field_theory::finite_field Fq;
	number_theory::number_theory_domain NT;

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
		Fq.init_override_polynomial_small_order(q, poly,
				FALSE /* f_without_tables */,
				FALSE /* f_compute_related_fields */,
				verbose_level - 1);
	}
	else {
		Fq.finite_field_init_small_order(q,
				FALSE /* f_without_tables */,
				FALSE /* f_compute_related_fields */,
				verbose_level - 2);
	}
	FQ.finite_field_init_small_order(Q,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
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
	data_structures::sorting Sorting;

	for (c = 0; c < n; c++) {
		if (NT.gcd_lint(c, n) != 1)
			continue;
		C.create(c, __FILE__, __LINE__);
		N.create(n, __FILE__, __LINE__);
		D.extended_gcd(C, N, G, U, V, FALSE);
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
		field_theory::finite_field *F,
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
		FALSE /* f_init_incidence_structure */,
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
		Orbiter->Lint_vec.print(cout, Pts, nb_pts);
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




}}}


