/*
 * finit_field_io.cpp
 *
 *  Created on: Jan 5, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

void finite_field::report(std::ostream &ost, int verbose_level)
{
	ost << "\\small" << endl;
	ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "$q = " << q << "$\\\\" << endl;
	ost << "$p = " << p << "$\\\\" << endl;
	ost << "$e = " << e << "$\\\\" << endl;

	ost << "\\clearpage" << endl << endl;
	ost << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	cheat_sheet(ost, verbose_level);


}

void finite_field::print_minimum_polynomial(int p, const char *polynomial)
{
	finite_field GFp;

	GFp.finite_field_init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m, n;

	FX.create_object_by_rank_string(m, polynomial, 0);
	FX.create_object_by_rank_string(n, polynomial, 0);
	{
		unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);

		Fq.print_object(n, cout);
	}
	//cout << "finite_field::print_minimum_polynomial "
	//"before delete_object" << endl;
	FX.delete_object(m);
	FX.delete_object(n);
}

void finite_field::print()
{
	cout << "Finite field of order " << q << endl;
}

void finite_field::print_detailed(int f_add_mult_table)
{
	if (f_is_prime_field) {
		print_tables();
	}
	else {
		//char *poly;

		//poly = get_primitive_polynomial(p, e, 0 /* verbose_level */);

		cout << "polynomial = ";
		print_minimum_polynomial(p, polynomial);
		cout << endl;
		//cout << " = " << poly << endl;
		print_tables_extension_field(polynomial);
	}
	if (f_add_mult_table) {
		print_add_mult_tables(cout);
		print_add_mult_tables_in_C(label);
	}
}


void finite_field::print_add_mult_tables(std::ostream &ost)
{
	ost << "addition table:" << endl;
	Orbiter->Int_vec.print_integer_matrix_width(ost, add_table, q, q, q, log10_of_q + 1);
	ost << endl;


	ost << "multiplication table:" << endl;
	Orbiter->Int_vec.print_integer_matrix_width(ost, mult_table, q, q, q, log10_of_q + 1);
	ost << endl;
}

void finite_field::print_add_mult_tables_in_C(std::string &fname_base)
{

	string fname;

	fname.assign(fname_base);
	fname.append(".cpp");

	{
		ofstream ost(fname);

		ost << "//addition, multiplication, inversion and negation table:" << endl;
		ost << "int add_table[] = ";
		Orbiter->Int_vec.print_integer_matrix_in_C_source(ost, add_table, q, q);
		ost << endl;


		ost << "int mult_table[] = ";
		Orbiter->Int_vec.print_integer_matrix_in_C_source(ost, mult_table, q, q);
		ost << endl;

		ost << "int inv_table[] = ";
		Orbiter->Int_vec.print_integer_matrix_in_C_source(ost, inv_table, 1, q);
		ost << endl;

		ost << "int neg_table[] = ";
		Orbiter->Int_vec.print_integer_matrix_in_C_source(ost, negate_table, 1, q);
		ost << endl;
	}

}


void finite_field::print_tables()
{
	int i, a, b, c, l;



	cout << "i : inverse(i) : frobenius_power(i, 1) : alpha_power(i) : "
			"log_alpha(i)" << endl;
	for (i = 0; i < q; i++) {
		if (i)
			a = inverse(i);
		else
			a = -1;
		if (i)
			l = log_alpha(i);
		else
			l = -1;
		b = frobenius_power(i, 1);
		c = alpha_power(i);
		cout << setw(4) << i << " : "
			<< setw(4) << a << " : "
			<< setw(4) << b << " : "
			<< setw(4) << c << " : "
			<< setw(4) << l << endl;

	}
}

void finite_field::print_tables_extension_field(const char *poly)
{
	int i, a, b, c, l;
	int verbose_level = 0;

	finite_field GFp;
	GFp.finite_field_init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m;



	FX.create_object_by_rank_string(m, poly, verbose_level);

	unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);
	unipoly_object elt;



	cout << "i : inverse(i) : frobenius_power(i, 1) : alpha_power(i) : "
			"log_alpha(i) : elt[i]" << endl;
	for (i = 0; i < q; i++) {
		if (i)
			a = inverse(i);
		else
			a = -1;
		if (i)
			l = log_alpha(i);
		else
			l = -1;
		b = frobenius_power(i, 1);
		c = alpha_power(i);
		cout << setw(4) << i << " : "
			<< setw(4) << a << " : "
			<< setw(4) << b << " : "
			<< setw(4) << c << " : "
			<< setw(4) << l << " : ";
		Fq.create_object_by_rank(elt, i, __FILE__, __LINE__, verbose_level);
		Fq.print_object(elt, cout);
		cout << endl;
		Fq.delete_object(elt);

		}
	// FX.delete_object(m);  // this had to go, Anton Betten, Oct 30, 2011

	//cout << "print_tables finished" << endl;
#if 0
	cout << "inverse table:" << endl;
	cout << "{";
	for (i = 1; i < q; i++) {
		cout << inverse(i);
		if (i < q - 1)
			cout << ", ";
		}
	cout << "};" << endl;
	cout << "frobenius_table:" << endl;
	//print_integer_matrix(cout, frobenius_table, 1, q);
	cout << "i : i^p" << endl;
	for (i = 0; i < q; i++) {
		cout << i << " : " << frobenius_table[i] << endl;
		}


	cout << "primitive element alpha = " << alpha << endl;
	cout << "i : alpha^i" << endl;
	for (i = 0; i < q; i++) {
		//j = power(p, i);
		cout << i << " : " << alpha_power_table[i] << endl;
		}
	cout << "i : log_alpha(i)" << endl;
	for (i = 0; i < q; i++) {
		cout << i << " : " << log_alpha_table[i] << endl;
		}
#endif

	//cout << "alpha_power_table:" << endl;
	//print_integer_matrix(cout, alpha_power_table, 1, q);
	//cout << "log_alpha_table:" << endl;
	//print_integer_matrix(cout, log_alpha_table, 1, q);
}

void finite_field::display_T2(ostream &ost)
{
	int i;

	ost << "i & T2(i)" << endl;
	for (i = 0; i < q; i++) {
		ost << setw((int) log10_of_q) << i << " & "
				<< setw((int) log10_of_q) << T2(i) << endl;
		}
}

void finite_field::display_T3(ostream &ost)
{
	int i;

	ost << "i & T3(i)" << endl;
	for (i = 0; i < q; i++) {
		ost << setw((int) log10_of_q) << i << " & "
				<< setw((int) log10_of_q) << T3(i) << endl;
		}
}

void finite_field::display_N2(ostream &ost)
{
	int i;

	ost << "i & N2(i)" << endl;
	for (i = 0; i < q; i++) {
		ost << setw((int) log10_of_q) << i << " & "
				<< setw((int) log10_of_q) << N2(i) << endl;
		}
}

void finite_field::display_N3(ostream &ost)
{
	int i;

	ost << "i & N3(i)" << endl;
	for (i = 0; i < q; i++) {
		ost << setw((int) log10_of_q) << i << " & "
				<< setw((int) log10_of_q) << N3(i) << endl;
		}
}

void finite_field::print_integer_matrix_zech(ostream &ost,
		int *p, int m, int n)
{
	int i, j, a, h;
    int w;
	number_theory_domain NT;

	w = (int) NT.int_log10(q);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a == 0) {
				for (h = 0; h < w - 1; h++)
					ost << " ";
				ost << ". ";
				}
			else {
				a = log_alpha(a);
				ost << setw(w) << a << " ";
				}
			}
		ost << endl;
		}
}



void finite_field::print_embedding(finite_field &subfield,
	int *components, int *embedding, int *pair_embedding)
{
	int Q, q, i, j;

	Q = finite_field::q;
	q = subfield.q;
	cout << "embedding:" << endl;
	for (i = 0; i < q; i++) {
		cout << setw(4) << i << " : " << setw(4) << embedding[i] << endl;
		}
	cout << "components:" << endl;
	for (i = 0; i < Q; i++) {
		cout << setw(4) << i << setw(4) << components[i * 2 + 0]
			<< setw(4) << components[i * 2 + 1] << endl;
		}
	cout << "pair_embeddings:" << endl;
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			cout << setw(4) << i << setw(4) << j << setw(4)
				<< pair_embedding[i * q + j] << endl;
			}
		}
}

void finite_field::print_embedding_tex(finite_field &subfield,
	int *components, int *embedding, int *pair_embedding)
{
	int q, i, j, a, b, aa, bb, c;

	//Q = finite_field::q;
	q = subfield.q;

	for (j = 0; j < q; j++) {
		cout << " & ";
		subfield.print_element(cout, j);
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	for (i = 0; i < q; i++) {
		subfield.print_element(cout, i);
		if (i == 0) {
			a = 0;
		}
		else {
			a = subfield.alpha_power(i - 1);
		}
		aa = embedding[a];
		for (j = 0; j < q; j++) {
			if (j == 0) {
				b = 0;
			}
			else {
				b = subfield.alpha_power(j - 1);
			}
			bb = embedding[b];
			c = add(aa, mult(bb, p));
			cout << " & ";
			print_element(cout, c);
		}
		cout << "\\\\" << endl;
	}
}

void finite_field::print_indicator_square_nonsquare(int a)
{
	int l;

	if (p == 2) {
		cout << "finite_field::print_indicator_square_nonsquare "
				"the characteristic is two" << endl;
		exit(1);
	}
	if (a == 0) {
		cout << "0";
	}
	else {
		l = log_alpha(a);
		if (EVEN(l)) {
			cout << "+";
		}
		else {
			cout << "-";
		}
	}
}

void finite_field::print_element(ostream &ost, int a)
{
	int width;


	if (e == 1) {
		ost << a;
	}
	else {
		if (f_print_as_exponentials) {
			width = 10;
		}
		else {
			width = log10_of_q;
		}
		print_element_with_symbol(ost, a, f_print_as_exponentials,
				width, symbol_for_print);
	}
}

void finite_field::print_element_str(stringstream &ost, int a)
{
	int width;


	if (e == 1) {
		ost << a;
	}
	else {
		if (f_print_as_exponentials) {
			width = 10;
		}
		else {
			width = log10_of_q;
		}
		print_element_with_symbol_str(ost, a, f_print_as_exponentials,
				width, symbol_for_print);
	}
}

void finite_field::print_element_with_symbol(ostream &ost,
		int a, int f_exponential, int width, std::string &symbol)
{
	int b;

	if (f_exponential) {
#if 0
		if (symbol == NULL) {
			cout << "finite_field::print_element_with_symbol "
					"symbol == NULL" << endl;
			return;
		}
#endif
		if (a == 0) {
			//print_repeated_character(ost, ' ', width - 1);
			ost << "0";
		}
		else if (a == 1) {
			//print_repeated_character(ost, ' ', width - 1);
			ost << "1";
		}
		else {
			b = log_alpha(a);
			if (b == q - 1) {
				b = 0;
			}
			ost << symbol;
			if (b > 1) {
				ost << "^{" << b << "}";
			}
			else {
				ost << " ";
			}
		}
	}
	else {
		ost << setw((int) width) << a;
	}
}

void finite_field::print_element_with_symbol_str(stringstream &ost,
		int a, int f_exponential, int width, std::string &symbol)
{
	int b;

	if (f_exponential) {
#if 0
		if (symbol == NULL) {
			cout << "finite_field::print_element_with_symbol_str "
					"symbol == NULL" << endl;
			return;
		}
#endif
		if (a == 0) {
			//print_repeated_character(ost, ' ', width - 1);
			ost << "0";
		}
		else if (a == 1) {
			//print_repeated_character(ost, ' ', width - 1);
			ost << "1";
		}
		else {
			b = log_alpha(a);
			if (b == q - 1) {
				b = 0;
			}
			ost << symbol;
			if (b > 1) {
				ost << "^{" << b << "}";
			}
			else {
				ost << " ";
			}
		}
	}
	else {
		ost << setw((int) width) << a;
	}
}

void finite_field::int_vec_print_field_elements(ostream &ost, int *v, int len)
{
	int i;
	ost << "(";
	for (i = 0; i < len; i++) {
		print_element(ost, v[i]);
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void finite_field::int_vec_print_elements_exponential(ostream &ost,
		int *v, int len, std::string &symbol_for_print)
{
	int i;
	ost << "(";
	for (i = 0; i < len; i++) {
		if (v[i] >= q) {
			cout << "finite_field::int_vec_print_elements_exponential v[i] >= q" << endl;
			cout << "v[i]=" << v[i] << endl;
			exit(1);
		}
		print_element_with_symbol(ost, v[i],
			TRUE /*f_print_as_exponentials*/,
			10 /*width*/, symbol_for_print);
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void finite_field::make_fname_addition_table_csv(std::string &fname)
{
	char str[1000];

	sprintf(str, "GF_q%d", q);
	fname.assign(str);
	fname.append("_addition_table.csv");
}

void finite_field::make_fname_multiplication_table_csv(std::string &fname)
{
	char str[1000];

	sprintf(str, "GF_q%d", q);
	fname.assign(str);
	fname.append("_multiplication_table.csv");
}

void finite_field::make_fname_addition_table_reordered_csv(std::string &fname)
{
	char str[1000];

	sprintf(str, "GF_q%d", q);
	fname.assign(str);
	fname.append("_addition_table_reordered.csv");
}

void finite_field::make_fname_multiplication_table_reordered_csv(std::string &fname)
{
	char str[1000];

	sprintf(str, "GF_q%d", q);
	fname.assign(str);
	fname.append("_multiplication_table_reordered.csv");
}

void finite_field::addition_table_save_csv()
{
	int i, j, k;
	int *T;
	file_io Fio;

	T = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			k = add(i, j);
			T[i * q + j] = k;
		}
	}
	std::string fname;

	make_fname_addition_table_csv(fname);
	Fio.int_matrix_write_csv(fname, T, q, q);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	FREE_int(T);
}

void finite_field::multiplication_table_save_csv()
{
	int i, j, k;
	int *T;
	file_io Fio;

	T = NEW_int((q - 1) * (q - 1));
	for (i = 0; i < q - 1; i++) {
		for (j = 0; j < q - 1; j++) {
			k = mult(1 + i, 1 + j);
			T[i * (q - 1) + j] = k;
		}
	}
	std::string fname;

	make_fname_multiplication_table_csv(fname);
	Fio.int_matrix_write_csv(fname, T, q - 1, q - 1);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	FREE_int(T);
}

void finite_field::addition_table_reordered_save_csv()
{
	int i, j, a, b, c;
	int *T;
	file_io Fio;

	T = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		a = reordered_list_of_elements[i];
		for (j = 0; j < q; j++) {
			b = reordered_list_of_elements[j];
			c = add(a, b);
			//k = reordered_list_of_elements_inv[c];
			T[i * q + j] = c;
		}
	}
	std::string fname;

	make_fname_addition_table_reordered_csv(fname);
	Fio.int_matrix_write_csv(fname, T, q, q);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	FREE_int(T);
}


void finite_field::multiplication_table_reordered_save_csv()
{
	int i, j, a, b, c;
	int *T;
	file_io Fio;

	T = NEW_int(q * q);
	for (i = 1; i < q; i++) {
		a = reordered_list_of_elements[i];
		for (j = 1; j < q; j++) {
			b = reordered_list_of_elements[j];
			c = mult(a, b);
			//k = reordered_list_of_elements_inv[c];
			if (c == 0) {
				cout << "finite_field::multiplication_table_reordered_save_csv c == 0" << endl;
				exit(1);
			}
			T[(i - 1) * (q - 1) + j - 1] = c;
		}
	}
	std::string fname;

	make_fname_multiplication_table_reordered_csv(fname);
	Fio.int_matrix_write_csv(fname, T, q - 1, q - 1);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	FREE_int(T);
}


void finite_field::latex_addition_table(ostream &f,
		int f_elements_exponential, std::string &symbol_for_print)
{
	int i, j, k;

	//f << "\\arraycolsep=1pt" << endl;
	f << "\\begin{array}{|r|*{" << q << "}{r}|}" << endl;
	f << "\\hline" << endl;
	f << "+ ";
	for (i = 0; i < q; i++) {
		f << " &";
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < q; i++) {
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
		for (j = 0; j < q; j++) {
			k = add(i, j);
			f << "&";
			print_element_with_symbol(f, k, f_elements_exponential,
					10 /* width */,
				symbol_for_print);
		}
		f << "\\\\" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
}

void finite_field::latex_multiplication_table(ostream &f,
		int f_elements_exponential, std::string &symbol_for_print)
{
	int i, j, k;

	//f << "\\arraycolsep=1pt" << endl;
	f << "\\begin{array}{|r|*{" << q - 1 << "}{r}|}" << endl;
	f << "\\hline" << endl;
	f << "\\cdot ";
	for (i = 1; i < q; i++) {
		f << " &";
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 1; i < q; i++) {
		f << setw(3);
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
		for (j = 1; j < q; j++) {
			k = mult(i, j);
			f << "&" << setw(3);
			print_element_with_symbol(f, k, f_elements_exponential,
					10 /* width */,
				symbol_for_print);
		}
		f << "\\\\" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
}

void finite_field::latex_matrix(ostream &f, int f_elements_exponential,
		std::string &symbol_for_print, int *M, int m, int n)
{
	int i, j;

	f << "\\begin{array}{*{" << n << "}{r}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f << setw(3);
			print_element_with_symbol(f, M[i * n + j],
				f_elements_exponential, 10 /* width */,
				symbol_for_print);
			if (j < n - 1) {
				f << " & ";
			}
		}
		f << "\\\\" << endl;
	}
	f << "\\end{array}" << endl;
}


void finite_field::power_table(int t,
		int *power_table, int len)
{
	int i;

	power_table[0] = 1;
	for (i = 1; i < len; i++) {
		power_table[i] = mult(power_table[i - 1], t);
	}
}


void finite_field::cheat_sheet(ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::cheat_sheet" << endl;
	}

	int f_add_mult_table = TRUE;

	if (f_add_mult_table) {
		if (f_v) {
			print_add_mult_tables(cout);
		}
		print_add_mult_tables_in_C(label);
	}

	cheat_sheet_subfields(f, verbose_level);

	cheat_sheet_table_of_elements(f, verbose_level);

	cheat_sheet_main_table(f, verbose_level);

	cheat_sheet_addition_table(f, verbose_level);

	cheat_sheet_multiplication_table(f, verbose_level);

	cheat_sheet_power_table(f, TRUE /* f_with_polynomials */, verbose_level);

	cheat_sheet_power_table(f, FALSE /* f_with_polynomials */, verbose_level);

	if (f_v) {
		cout << "finite_field::cheat_sheet done" << endl;
	}

}

void finite_field::cheat_sheet_subfields(ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//const char *symbol_for_print = "\\alpha";
	number_theory_domain NT;


	if (f_v) {
		cout << "finite_field::cheat_sheet_subfields" << endl;
	}

	//f << "\\small" << endl;
	if (!f_is_prime_field) {
		f << "The polynomial used to define the field is : ";
		finite_field GFp;
		GFp.finite_field_init(p, 0);

		unipoly_domain FX(&GFp);
		unipoly_object m;


		FX.create_object_by_rank_string(m, polynomial, verbose_level - 2);
		f << "$";
		FX.print_object(m, f);
		f << "$ = " << polynomial << "\\\\" << endl;
	}

	f << "$Z_i = \\log_\\alpha (1 + \\alpha^i)$\\\\" << endl;

	if (!f_is_prime_field && !NT.is_prime(e)) {
		report_subfields(f, verbose_level);
	}
	if (!f_is_prime_field) {
		report_subfields_detailed(f, verbose_level);
	}

	if (f_v) {
		cout << "finite_field::cheat_sheet_subfields done" << endl;
	}
}

void finite_field::report_subfields(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;
	int h;

	if (f_v) {
		cout << "finite_field::report_subfields" << endl;
	}
	ost << "\\subsection*{Subfields}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Subfield} & \\mbox{Polynomial} & \\mbox{Numerical Rank} \\\\"
			<< endl;
	ost << "\\hline" << endl;
	for (h = 2; h < e; h++) {
		if ((e % h) == 0) {


			ost << "\\hline" << endl;
			long int poly;

			poly = compute_subfield_polynomial(
					NT.i_power_j(p, h),
					FALSE, cout,
					verbose_level);
			{
				finite_field GFp;
				GFp.finite_field_init(p, 0);

				unipoly_domain FX(&GFp);
				unipoly_object m;

				FX.create_object_by_rank_string(m, polynomial,
						0/*verbose_level*/);
				unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);
				unipoly_object elt;

				FX.create_object_by_rank(elt, poly, __FILE__, __LINE__, verbose_level);
				ost << "\\bbF_{" << NT.i_power_j(p, h) << "} & ";
				Fq.print_object(elt, ost);
				ost << " & " << poly;
				ost << "\\\\" << endl;
				Fq.delete_object(elt);
			}

		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	if (f_v) {
		cout << "finite_field::report_subfields done" << endl;
	}
}

void finite_field::report_subfields_detailed(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;
	int h;

	if (f_v) {
		cout << "finite_field::report_subfields_detailed" << endl;
	}
	ost << "\\subsection*{Subfields in Detail}" << endl;
	for (h = 1; h < e; h++) {
		if (e % h) {
			continue;
		}

		long int poly_numeric, q0;
		finite_field *Fq;

		Fq = NEW_OBJECT(finite_field);

		q0 = NT.i_power_j(p, h);

		poly_numeric = compute_subfield_polynomial(q0, TRUE, ost, verbose_level);


		char str[1000];

		sprintf(str, "%ld", poly_numeric);
		string poly_text;

		poly_text.assign(str);
		Fq->init_override_polynomial(q0, poly_text, verbose_level);

		subfield_structure *Sub;

		Sub = NEW_OBJECT(subfield_structure);

		Sub->init(this /* FQ */, Fq, verbose_level);


		if (f_v) {
			ost << "Subfield ${\\mathbb F}_{" << q0 << "}$ generated by polynomial " << poly_numeric << ":\\\\" << endl;
			Sub->report(ost);
		}

		FREE_OBJECT(Sub);
		FREE_OBJECT(Fq);
	}
	if (f_v) {
		cout << "finite_field::report_subfields_detailed done" << endl;
	}
}


void finite_field::cheat_sheet_addition_table(ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "finite_field::cheat_sheet_addition_table" << endl;
	}

	if (q <= 64) {
		f << "$$" << endl;
		latex_addition_table(f, FALSE /* f_elements_exponential */,
				symbol_for_print);
#if 0
		const char *symbol_for_print = "\\alpha";
		if (q >= 10) {
			f << "$$" << endl;
			f << "$$" << endl;
		}
		else {
			f << "\\qquad" << endl;
		}
		latex_addition_table(f, TRUE /* f_elements_exponential */,
				symbol_for_print);
#endif
		f << "$$" << endl;
	}
	else {
		f << "Addition table omitted" << endl;
	}



	if (f_v) {
		cout << "finite_field::cheat_sheet_addition_table done" << endl;
	}
}

void finite_field::cheat_sheet_multiplication_table(ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "finite_field::cheat_sheet_multiplication_table" << endl;
	}

	if (q <= 64) {
		f << "$$" << endl;
		latex_multiplication_table(f, FALSE /* f_elements_exponential */,
				symbol_for_print);
#if 0
		const char *symbol_for_print = "\\alpha";
		if (q >= 10) {
			f << "$$" << endl;
			f << "$$" << endl;
		}
		else {
			f << "\\qquad" << endl;
		}
		latex_multiplication_table(f, TRUE /* f_elements_exponential */,
				symbol_for_print);
#endif
		f << "$$" << endl;
	}
	else {
		f << "Multiplication table omitted" << endl;
	}



	if (f_v) {
		cout << "finite_field::cheat_sheet_multiplication_table done" << endl;
	}
}

void finite_field::cheat_sheet_power_table(std::ostream &ost, int f_with_polynomials, int verbose_level)
{
	int *Powers;
	int i, j, t;
	int len = q;
	int *v;
	geometry_global Gg;

	t = primitive_root();

	v = NEW_int(e);
	Powers = NEW_int(len);
	power_table(t, Powers, len);

	ost << "\\subsection*{Cyclic structure}" << endl;


	ost << "$$" << endl;
	cheat_sheet_power_table_top(ost, f_with_polynomials, verbose_level);

	for (i = 0; i < len; i++) {

		if (i && (i % 32) == 0) {
			cheat_sheet_power_table_bottom(ost, f_with_polynomials, verbose_level);
			ost << "$$" << endl;
			ost << "$$" << endl;
			cheat_sheet_power_table_top(ost, f_with_polynomials, verbose_level);
		}

		Gg.AG_element_unrank(p, v, 1, e, Powers[i]);

		ost << i << " & " << t << "^{" << i << "} & " << Powers[i] << " & ";
		for (j = e - 1; j >= 0; j--) {
			ost << v[j];
		}

		if (f_with_polynomials) {
			ost << " & ";

			print_element_as_polynomial(ost, v, verbose_level);
		}


		ost << "\\\\" << endl;
	}
	cheat_sheet_power_table_bottom(ost, f_with_polynomials, verbose_level);
	ost << "$$" << endl;

	FREE_int(v);
	FREE_int(Powers);

}

void finite_field::cheat_sheet_power_table_top(std::ostream &ost, int f_with_polynomials, int verbose_level)
{
	ost << "\\begin{array}{|r|r|r|r|";
	if (f_with_polynomials) {
		ost << "r|";
	}
	ost << "}" << endl;
	ost << "\\hline" << endl;


	ost << "i & \\alpha^i & \\alpha^i & \\mbox{vector}";
	if (f_with_polynomials) {
		ost << "& \\mbox{reduced rep.}";
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
}

void finite_field::cheat_sheet_power_table_bottom(std::ostream &ost, int f_with_polynomials, int verbose_level)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
}

void finite_field::cheat_sheet_table_of_elements(std::ostream &ost, int verbose_level)
{
	int *v;
	int i, j;
	int f_first;
	//std::string my_symbol;

	v = NEW_int(e);

	//my_symbol.assign("\alpha");

	ost << "\\subsection*{Table of Elements of ${\\mathbb F}_{" << q << "}$}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;

	geometry_global Gg;

	for (i = 0; i < q; i++) {
		Gg.AG_element_unrank(p, v, 1, e, i);
		ost << setw(3) << i;
		ost << " & ";
		f_first = TRUE;


		for (j = e - 1; j >= 0; j--) {
			ost << v[j];
		}
		ost << " & ";

		print_element_as_polynomial(ost, v, verbose_level);

		ost << "\\\\" << endl;

#if 0
		ost << " & ";
		print_element_with_symbol(ost, i,
			TRUE /*f_print_as_exponentials*/,
			10 /*width*/, my_symbol);
#endif
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	FREE_int(v);

}

void finite_field::print_element_as_polynomial(std::ostream &ost, int *v, int verbose_level)
{
	int j;
	int f_first = TRUE;

	for (j = e - 1; j >= 0; j--) {
		if (v[j] == 0) {
			continue;
		}

		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}

		if (j == 0 || v[j] > 1) {
			ost << setw(3) << v[j];
		}
		if (j) {
			ost << "\\alpha";
		}
		if (j > 1) {
			ost << "^{" << j << "}";
		}
	}
	if (f_first) {
		ost << "0";
	}
}

void finite_field::cheat_sheet_main_table(std::ostream &f, int verbose_level)
{
	int *v;
	int i, j, a;
	int f_first;

	v = NEW_int(e);


	int nb_cols = 7;

	if (e > 1) {
		nb_cols += 3;
	}
	if ((e % 2) == 0 && e > 2) {
		nb_cols += 2;
	}
	if ((e % 3) == 0 && e > 3) {
		nb_cols += 2;
	}


	cheat_sheet_main_table_top(f, nb_cols);

	geometry_global Gg;

	for (i = 0; i < q; i++) {
		Gg.AG_element_unrank(p, v, 1, e, i);
		f << setw(3) << i << " & ";
		f_first = TRUE;
		for (j = e - 1; j >= 0; j--) {
			if (v[j] == 0) {
				continue;
			}

			if (f_first) {
				f_first = FALSE;
			}
			else {
				f << " + ";
			}

			if (j == 0 || v[j] > 1) {
				f << setw(3) << v[j];
			}
			if (j) {
				f << "\\alpha";
			}
			if (j > 1) {
				f << "^{" << j << "}";
			}
		}
		if (f_first) {
			f << "0";
		}

		f << " = ";
		print_element_with_symbol(f, i,
			TRUE /*f_print_as_exponentials*/,
			10 /*width*/, symbol_for_print);



		// - gamma_i:
		f << " &" << negate(i);
		// gamma_i^{-1}:
		if (i == 0) {
			f << " & \\mbox{DNE}";
		}
		else {
			f << " &" << inverse(i);
		}



		// log_alpha:
		if (i == 0) {
			f << " & \\mbox{DNE}";
		}
		else {
			f << " &" << log_alpha(i);
		}
		// alpha_power:
		f << " &" << alpha_power(i);


		// Z_i:
		a = add(1, alpha_power(i));
		if (a == 0) {
			f << " & \\mbox{DNE}";
		}
		else {
			f << " &" << log_alpha(a);
		}




		// additional columns for extension fields:
		if (e > 1) {
			f << " &" << frobenius_power(i, 1);
			f << " &" << absolute_trace(i);
			f << " &" << absolute_norm(i);
		}

		if ((e % 2) == 0 && e > 2) {
			f << " &" << T2(i);
			f << " &" << N2(i);
		}
		if ((e % 3) == 0 && e > 3) {
			f << " &" << T3(i);
			f << " &" << N3(i);
		}


		f << "\\\\" << endl;

		if ((i % 25) == 0 && i) {
			cheat_sheet_main_table_bottom(f);
			cheat_sheet_main_table_top(f, nb_cols);
		}
	}

	cheat_sheet_main_table_bottom(f);

	FREE_int(v);

}

void finite_field::cheat_sheet_main_table_top(std::ostream &f, int nb_cols)
{
	f << "$$" << endl;
	f << "\\begin{array}{|*{" << nb_cols << "}{r|}}" << endl;
	f << "\\hline" << endl;
	f << "i & \\gamma_i ";
	f << "& -\\gamma_i";
	f << "& \\gamma_i^{-1}";
	f << "& \\log_\\alpha(\\gamma_i)";
	f << "& \\alpha^i";
	f << "& Z_i";
	if (e > 1) {
		f << "& \\phi(\\gamma_i) ";
		f << "& T(\\gamma_i) ";
		f << "& N(\\gamma_i) ";
	}
	if ((e % 2) == 0 && e > 2) {
		f << "& T_2(\\gamma_i) ";
		f << "& N_2(\\gamma_i) ";
	}
	if ((e % 3) == 0 && e > 3) {
		f << "& T_3(\\gamma_i) ";
		f << "& N_3(\\gamma_i) ";
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
}

void finite_field::cheat_sheet_main_table_bottom(std::ostream &f)
{
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
}


void finite_field::display_table_of_projective_points(
	std::ostream &ost, long int *Pts, int nb_pts, int len)
{

	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;

	display_table_of_projective_points2(ost, Pts, nb_pts, len);

	ost << "$$}%" << endl;
}

void finite_field::display_table_of_projective_points2(
	std::ostream &ost, long int *Pts, int nb_pts, int len)
{
	int i;
	int *coords;

	coords = NEW_int(len);
	ost << "\\begin{array}{|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & a_i & P_{a_i}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified_lint(coords, 1, len, Pts[i]);
		ost << i << " & " << Pts[i] << " & ";
		Orbiter->Int_vec.print(ost, coords, len);
		ost << "\\\\" << endl;
		if (((i + 1) % 30) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$}%" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & a_i & P_{a_i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	FREE_int(coords);
}

void finite_field::display_table_of_projective_points_easy(
	std::ostream &ost, long int *Pts, int nb_pts, int len)
{
	int i;
	int *coords;

	coords = NEW_int(len);
	ost << "\\begin{array}{|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "P_i\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified_lint(coords, 1, len, Pts[i]);
		Orbiter->Int_vec.print(ost, coords, len);
		ost << "\\\\" << endl;
		if (((i + 1) % 30) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$}%" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "P_i\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	FREE_int(coords);
}


void finite_field::export_magma(int d, long int *Pts, int nb_pts, std::string &fname)
{
	string fname2;
	int *v;
	int h, i, a, b;
	string_tools ST;

	v = NEW_int(d);
	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".magma");

	{
		ofstream fp(fname2);

		fp << "G,I:=PGammaL(" << d << "," << q
				<< ");F:=GF(" << q << ");" << endl;
		fp << "S:={};" << endl;
		fp << "a := F.1;" << endl;
		for (h = 0; h < nb_pts; h++) {
			PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

			PG_element_normalize_from_front(v, 1, d);

			fp << "Include(~S,Index(I,[";
			for (i = 0; i < d; i++) {
				a = v[i];
				if (a == 0) {
					fp << "0";
				}
				else if (a == 1) {
					fp << "1";
				}
				else {
					b = log_alpha(a);
					fp << "a^" << b;
				}
				if (i < d - 1) {
					fp << ",";
				}
			}
			fp << "]));" << endl;
		}
		fp << "Stab := Stabilizer(G,S);" << endl;
		fp << "Size(Stab);" << endl;
		fp << endl;
	}
	file_io Fio;

	cout << "Written file " << fname2 << " of size "
			<< Fio.file_size(fname2) << endl;

	FREE_int(v);
}

void finite_field::export_gap(int d, long int *Pts, int nb_pts, std::string &fname)
{
	string fname2;
	int *v;
	int h, i, a, b;
	string_tools ST;

	v = NEW_int(d);
	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".gap");

	{
		ofstream fp(fname2);

		fp << "LoadPackage(\"fining\");" << endl;
		fp << "pg := ProjectiveSpace(" << d - 1 << "," << q << ");" << endl;
		fp << "S:=[" << endl;
		for (h = 0; h < nb_pts; h++) {
			PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

			PG_element_normalize_from_front(v, 1, d);

			fp << "[";
			for (i = 0; i < d; i++) {
				a = v[i];
				if (a == 0) {
					fp << "0*Z(" << q << ")";
				}
				else if (a == 1) {
					fp << "Z(" << q << ")^0";
				}
				else {
					b = log_alpha(a);
					fp << "Z(" << q << ")^" << b;
				}
				if (i < d - 1) {
					fp << ",";
				}
			}
			fp << "]";
			if (h < nb_pts - 1) {
				fp << ",";
			}
			fp << endl;
		}
		fp << "];" << endl;
		fp << "S := List(S,x -> VectorSpaceToElement(pg,x));" << endl;
		fp << "g := CollineationGroup(pg);" << endl;
		fp << "stab := Stabilizer(g,Set(S),OnSets);" << endl;
		fp << "Size(stab);" << endl;
	}
	file_io Fio;

	cout << "Written file " << fname2 << " of size "
			<< Fio.file_size(fname2) << endl;

#if 0
LoadPackage("fining");
pg := ProjectiveSpace(2,4);
#points := Points(pg);
#pointslist := AsList(points);
#Display(pointslist[1]);
frame := [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]*Z(2)^0;
frame := List(frame,x -> VectorSpaceToElement(pg,x));
pairs := Combinations(frame,2);
secants := List(pairs,p -> Span(p[1],p[2]));
leftover := Filtered(pointslist,t->not ForAny(secants,s->t in s));
hyperoval := Union(frame,leftover);
g := CollineationGroup(pg);
stab := Stabilizer(g,Set(hyperoval),OnSets);
StructureDescription(stab);
#endif


	FREE_int(v);
}

void finite_field::print_matrix_latex(std::ostream &ost, int *A, int m, int n)
{
	int i, j, a;

	ost << "\\left[" << endl;
	ost << "\\begin{array}{*{" << n << "}{r}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = A[i * n + j];

#if 0
			if (is_prime(GFq->q)) {
				ost << setw(w) << a << " ";
			}
			else {
				ost << a;
				// GFq->print_element(ost, a);
			}
#else
			print_element(ost, a);
#endif

			if (j < n - 1)
				ost << " & ";
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;

}

void finite_field::print_matrix_numerical_latex(std::ostream &ost, int *A, int m, int n)
{
	int i, j, a;

	ost << "\\left[" << endl;
	ost << "\\begin{array}{*{" << n << "}{r}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = A[i * n + j];


			ost << a;
			if (j < n - 1)
				ost << " & ";
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;

}



}}


