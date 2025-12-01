/*
 * finite_field_io.cpp
 *
 *  Created on: Jan 31, 2023
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace field_theory {



finite_field_io::finite_field_io()
{
	Record_birth();
	F = NULL;
}

finite_field_io::~finite_field_io()
{
	Record_death();
}

void finite_field_io::init(
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_io::init" << endl;
	}

	finite_field_io::F = F;

	if (f_v) {
		cout << "finite_field_io::init done" << endl;
	}
}

void finite_field_io::report(
		std::ostream &ost, int verbose_level)
{
	//ost << "\\small" << endl;
	//ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "$q = " << F->q << "$\\\\" << endl;
	ost << "$p = " << F->p << "$\\\\" << endl;
	ost << "$e = " << F->e << "$\\\\" << endl;

	//ost << "\\clearpage" << endl << endl;
	ost << "\\section{The Finite Field with $" << F->q << "$ Elements}" << endl;
	cheat_sheet(ost, verbose_level);


}

void finite_field_io::print_minimum_polynomial_to_str(
		int p,
		std::string &polynomial, std::stringstream &s)
// this function creates a finite_field object
{

#if 1
	finite_field GFp;

	GFp.finite_field_init_small_order(p,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m, n;

	FX.create_object_by_rank_string(m, F->my_poly, 0);
	FX.create_object_by_rank_string(n, polynomial, 0);
	{
		ring_theory::unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);

		//Fq.print_object(n, cout);


		Fq.print_object_sstr(n, s);

	}
	//cout << "finite_field::print_minimum_polynomial "
	//"before delete_object" << endl;
	FX.delete_object(m);
	FX.delete_object(n);
#else
	ring_theory::longinteger_object rank;

	rank.create_from_base_10_string(polynomial);
	long int rk;
	int *v;
	geometry::geometry_global GG;

	v = NEW_int(e + 1);

	rk = rank.as_lint();
	GG.AG_element_unrank(p, v, 1, e + 1, rk);
	Int_vec_print(cout, v, e + 1);
	FREE_int(v);
#endif

}

void finite_field_io::print()
{
	cout << "Finite field of order " << F->q << endl;
}

void finite_field_io::print_detailed(
		int f_add_mult_table)
{
	if (F->f_is_prime_field) {
		print_tables();
	}
	else {
		//char *poly;

		//poly = get_primitive_polynomial(p, e, 0 /* verbose_level */);

		std::stringstream s;
		cout << "polynomial = ";
		print_minimum_polynomial_to_str(F->p, F->my_poly, s);
		cout << s.str() << endl;
		//cout << " = " << poly << endl;

		if (!F->f_has_table) {
			cout << "finite_field_io::print_detailed !f_has_table" << endl;
			exit(1);
		}
		F->get_T()->print_tables_extension_field(F->my_poly);
	}
	if (f_add_mult_table) {
		if (!F->f_has_table) {
			cout << "finite_field_io::print_detailed !f_has_table" << endl;
			exit(1);
		}
		F->get_T()->print_add_mult_tables(cout);
		F->get_T()->print_add_mult_tables_in_C(F->label);
	}
}




void finite_field_io::print_tables()
{
	int i, a, b, c, l;



	cout << "i : inverse(i) : frobenius_power(i, 1) : alpha_power(i) : "
			"log_alpha(i)" << endl;
	for (i = 0; i < F->q; i++) {
		if (i) {
			a = F->inverse(i);
		}
		else {
			a = -1;
		}
		if (i) {
			l = F->log_alpha(i);
		}
		else {
			l = -1;
		}
		b = F->frobenius_power(i, 1, 0 /* verbose_level */);
		c = F->alpha_power(i);
		cout << setw(4) << i << " : "
			<< setw(4) << a << " : "
			<< setw(4) << b << " : "
			<< setw(4) << c << " : "
			<< setw(4) << l << endl;

	}
}


void finite_field_io::display_T2(
		std::ostream &ost)
{
	int i;

	ost << "i & T2(i)" << endl;
	for (i = 0; i < F->q; i++) {
		ost << setw((int) F->log10_of_q) << i << " & "
				<< setw((int) F->log10_of_q) << F->T2(i) << endl;
	}
}

void finite_field_io::display_T3(
		std::ostream &ost)
{
	int i;

	ost << "i & T3(i)" << endl;
	for (i = 0; i < F->q; i++) {
		ost << setw((int) F->log10_of_q) << i << " & "
				<< setw((int) F->log10_of_q) << F->T3(i) << endl;
	}
}

void finite_field_io::display_N2(
		std::ostream &ost)
{
	int i;

	ost << "i & N2(i)" << endl;
	for (i = 0; i < F->q; i++) {
		ost << setw((int) F->log10_of_q) << i << " & "
				<< setw((int) F->log10_of_q) << F->N2(i) << endl;
	}
}

void finite_field_io::display_N3(
		std::ostream &ost)
{
	int i;

	ost << "i & N3(i)" << endl;
	for (i = 0; i < F->q; i++) {
		ost << setw((int) F->log10_of_q) << i << " & "
				<< setw((int) F->log10_of_q) << F->N3(i) << endl;
	}
}

void finite_field_io::print_integer_matrix_zech(
		std::ostream &ost,
		int *p, int m, int n)
{
	int i, j, a, h;
    int w;
    number_theory::number_theory_domain NT;

	w = (int) NT.int_log10(F->q);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a == 0) {
				for (h = 0; h < w - 1; h++) {
					ost << " ";
				}
				ost << ". ";
			}
			else {
				a = F->log_alpha(a);
				ost << setw(w) << a << " ";
			}
		}
		ost << endl;
	}
}




void finite_field_io::print_indicator_square_nonsquare(
		int a)
{
	int l;

	if (F->p == 2) {
		cout << "finite_field_io::print_indicator_square_nonsquare "
				"the characteristic is two" << endl;
		exit(1);
	}
	if (a == 0) {
		cout << "0";
	}
	else {
		l = F->log_alpha(a);
		if (EVEN(l)) {
			cout << "+";
		}
		else {
			cout << "-";
		}
	}
}

void finite_field_io::print_element(
		std::ostream &ost, int a)
{
	int width;
	int f_print_as_exponentials = true;

	if (F->Descr) {
		f_print_as_exponentials = F->Descr->f_print_as_exponentials;
	}

	if (F == NULL) {
		cout << "finite_field_io::print_element F == NULL" << endl;
		exit(1);
	}
	if (F->e == 1) {
		ost << a;
	}
	else {
		if (f_print_as_exponentials) {
			width = 10;
		}
		else {
			width = F->log10_of_q;
		}

		print_element_with_symbol(
				ost, a, f_print_as_exponentials,
				width, F->get_symbol_for_print());
	}
}

std::string finite_field_io::stringify_element(
		int a)
{

	string s;
	int width;
	int f_print_as_exponentials = true;

	if (F->Descr) {
		f_print_as_exponentials = F->Descr->f_print_as_exponentials;
	}

	if (F == NULL) {
		cout << "finite_field_io::print_element F == NULL" << endl;
		exit(1);
	}
	if (F->e == 1) {
		s += std::to_string(a);
	}
	else {
		if (f_print_as_exponentials) {
			width = 10;
		}
		else {
			width = F->log10_of_q;
		}

		s = stringify_element_with_symbol(
				a, f_print_as_exponentials,
				width, F->get_symbol_for_print());
	}
	return s;
}


void finite_field_io::print_element_str(
		std::stringstream &ost, int a)
{
	int width;


	if (F->e == 1) {
		ost << a;
	}
	else {
		if (F->Descr->f_print_as_exponentials) {
			width = 10;
		}
		else {
			width = F->log10_of_q;
		}
		print_element_with_symbol_str(
				ost, a, F->Descr->f_print_as_exponentials,
				width, F->get_symbol_for_print());
	}
}

void finite_field_io::print_element_with_symbol(
		std::ostream &ost,
		int a, int f_exponential,
		int width, std::string &symbol)
{
	int b;

	if (f_exponential) {
#if 0
		if (symbol == NULL) {
			cout << "finite_field_io::print_element_with_symbol "
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
			b = F->log_alpha(a);
			if (b == F->q - 1) {
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

std::string finite_field_io::stringify_element_with_symbol(
		int a, int f_exponential,
		int width, std::string &symbol)
{
	string s;
	int b;

	if (f_exponential) {
#if 0
		if (symbol == NULL) {
			cout << "finite_field_io::print_element_with_symbol "
					"symbol == NULL" << endl;
			return;
		}
#endif
		if (a == 0) {
			//print_repeated_character(ost, ' ', width - 1);
			s += "0";
		}
		else if (a == 1) {
			//print_repeated_character(ost, ' ', width - 1);
			s += "1";
		}
		else {
			b = F->log_alpha(a);
			if (b == F->q - 1) {
				b = 0;
			}
			s += symbol;
			if (b > 1) {
				s += "^{" + std::to_string(b) + "}";
			}
			else {
				s += " ";
			}
		}
	}
	else {
		string s1;

		s1 = std::to_string(a);
		if (s1.length() < width) {
			int nb_needed;
			int i;

			nb_needed = s1.length() - width;
			for (i = 0; i < nb_needed; i++) {
				string s2;

				s2 = " " + s1;
				s1 = s2;
			}
		}
		s += s1;
		//ost << setw((int) width) << a;
	}
	return s;
}


void finite_field_io::print_element_with_symbol_str(
		std::stringstream &ost,
		int a, int f_exponential,
		int width, std::string &symbol)
{
	int b;

	if (f_exponential) {
#if 0
		if (symbol == NULL) {
			cout << "finite_field_io::print_element_with_symbol_str "
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
			b = F->log_alpha(a);
			if (b == F->q - 1) {
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

void finite_field_io::int_vec_print_field_elements(
		std::ostream &ost, int *v, int len)
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

void finite_field_io::int_vec_print_elements_exponential(
		std::ostream &ost,
		int *v, int len, std::string &symbol_for_print)
{
	int i;
	ost << "(";
	for (i = 0; i < len; i++) {
		if (v[i] >= F->q) {
			cout << "finite_field_io::int_vec_print_elements_exponential "
					"v[i] >= q" << endl;
			cout << "v[i]=" << v[i] << endl;
			exit(1);
		}
		print_element_with_symbol(ost, v[i],
			true /*f_print_as_exponentials*/,
			10 /*width*/, symbol_for_print);
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void finite_field_io::make_fname_addition_table_csv(
		std::string &fname)
{
	fname = "GF_q" + std::to_string(F->q) + "_table_add.csv";
}

void finite_field_io::make_fname_multiplication_table_csv(
		std::string &fname)
{
	fname = "GF_q" + std::to_string(F->q) + "_table_mul.csv";
}

void finite_field_io::make_fname_addition_table_reordered_csv(
		std::string &fname)
{
	fname = "GF_q" + std::to_string(F->q) + "_table_add_r.csv";
}

void finite_field_io::make_fname_multiplication_table_reordered_csv(
		std::string &fname)
{
	fname = "GF_q" + std::to_string(F->q) + "_table_mul_r.csv";
}

void finite_field_io::addition_table_save_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int *M;
	other::orbiter_kernel_system::file_io Fio;

	M = NEW_int(F->q * F->q);
	for (i = 0; i < F->q; i++) {
		for (j = 0; j < F->q; j++) {
			k = F->add(i, j);
			M[i * F->q + j] = k;
		}
	}
	std::string fname;

	make_fname_addition_table_csv(fname);
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, F->q, F->q);
	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	FREE_int(M);
}

void finite_field_io::multiplication_table_save_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int *M;
	other::orbiter_kernel_system::file_io Fio;

	M = NEW_int(F->q * F->q);
	for (i = 0; i < F->q; i++) {
		for (j = 0; j < F->q; j++) {
			k = F->mult(i, j);
			M[i * F->q + j] = k;
		}
	}
	std::string fname;

	make_fname_multiplication_table_csv(fname);
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, F->q, F->q);
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	FREE_int(M);
}

void finite_field_io::addition_table_reordered_save_csv(
		int verbose_level)
{

	if (F->f_has_table) {
		std::string fname;

		make_fname_addition_table_reordered_csv(fname);

		F->get_T()->addition_table_reordered_save_csv(fname, verbose_level);
	}
	else {

		cout << "finite_field_io::addition_table_reordered_save_csv "
				"!f_has_table, skipping" << endl;
	}


}


void finite_field_io::multiplication_table_reordered_save_csv(
		int verbose_level)
{

	if (F->f_has_table) {
		std::string fname;

		make_fname_multiplication_table_reordered_csv(fname);

		F->get_T()->multiplication_table_reordered_save_csv(fname, verbose_level);
	}
	else {

		cout << "finite_field_io::multiplication_table_reordered_save_csv "
				"!f_has_table, skipping" << endl;

	}


}


void finite_field_io::latex_addition_table(
		std::ostream &f,
		int f_elements_exponential, std::string &symbol_for_print)
{
	int i, j, k;

	//f << "\\arraycolsep=1pt" << endl;
	f << "\\begin{array}{|r|*{" << F->q << "}{r}|}" << endl;
	f << "\\hline" << endl;
	f << "+ ";
	for (i = 0; i < F->q; i++) {
		f << " &";
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 0; i < F->q; i++) {
		print_element_with_symbol(f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
		for (j = 0; j < F->q; j++) {
			k = F->add(i, j);
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

void finite_field_io::latex_multiplication_table(
		std::ostream &f,
		int f_elements_exponential, std::string &symbol_for_print)
{
	int i, j, k;

	//f << "\\arraycolsep=1pt" << endl;
	f << "\\begin{array}{|r|*{" << F->q - 1 << "}{r}|}" << endl;
	f << "\\hline" << endl;
	f << "\\cdot ";
	for (i = 1; i < F->q; i++) {
		f << " &";
		print_element_with_symbol(
				f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
	for (i = 1; i < F->q; i++) {
		f << setw(3);
		print_element_with_symbol(
				f, i, f_elements_exponential,
				10 /* width */,
			symbol_for_print);
		for (j = 1; j < F->q; j++) {
			k = F->mult(i, j);
			f << "&" << setw(3);
			print_element_with_symbol(
					f, k, f_elements_exponential,
					10 /* width */,
				symbol_for_print);
		}
		f << "\\\\" << endl;
	}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
}

void finite_field_io::latex_matrix(
		std::ostream &f,
		int f_elements_exponential,
		std::string &symbol_for_print,
		int *M, int m, int n)
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


void finite_field_io::power_table(
		int t,
		int *power_table, int len)
{
	int i;

	power_table[0] = 1;
	for (i = 1; i < len; i++) {
		power_table[i] = F->mult(power_table[i - 1], t);
	}
}


void finite_field_io::cheat_sheet(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_io::cheat_sheet" << endl;
	}

	if (F->e > 1) {
		f << "Extension field generated by the polynomial" << endl;
		f << "$$" << endl;
		f << F->my_poly_tex << endl;
		f << "$$" << endl;
		f << "The numerical value of the polynomial "
				"is " << F->my_poly << "\\\\" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before F->Finite_field_properties->report_latex" << endl;
	}
	F->Finite_field_properties->report_latex(
			f, verbose_level);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after F->Finite_field_properties->report_latex" << endl;
	}


	int f_add_mult_table = true;

	if (f_add_mult_table) {

		if (F->f_has_table) {
			if (f_v) {
				F->get_T()->print_add_mult_tables(cout);
			}
			F->get_T()->print_add_mult_tables_in_C(F->label);
		}
		else {
			cout << "finite_field_io::cheat_sheet "
					"!f_has_table, skipping" << endl;
		}
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_subfields" << endl;
	}
	cheat_sheet_subfields(f, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_subfields" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_table_of_elements" << endl;
	}
	cheat_sheet_table_of_elements(f, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_table_of_elements" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_main_table" << endl;
	}
	cheat_sheet_main_table(f, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_main_table" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_addition_table" << endl;
	}
	cheat_sheet_addition_table(f, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_addition_table" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_multiplication_table" << endl;
	}
	cheat_sheet_multiplication_table(f, verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_multiplication_table" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_power_table true" << endl;
	}
	cheat_sheet_power_table(
			f, true /* f_with_polynomials */,
			verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_power_table true" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"before cheat_sheet_power_table false" << endl;
	}
	cheat_sheet_power_table(
			f, false /* f_with_polynomials */,
			verbose_level - 2);
	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"after cheat_sheet_power_table false" << endl;
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet "
				"done" << endl;
	}

}

void finite_field_io::cheat_sheet_subfields(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//const char *symbol_for_print = "\\alpha";
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "finite_field_io::cheat_sheet_subfields q=" << F->q << endl;
	}

	//f << "\\small" << endl;
	if (!F->f_is_prime_field) {
		f << "The polynomial used to define the field is : ";
		finite_field GFp;
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					"before GFp.finite_field_init_small_order" << endl;
		}
		GFp.finite_field_init_small_order(F->p,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					"after GFp.finite_field_init_small_order" << endl;
		}

		ring_theory::unipoly_domain FX(&GFp);
		ring_theory::unipoly_object m;


		FX.create_object_by_rank_string(m, F->my_poly, verbose_level - 2);
		f << "$";
		FX.print_object(m, f);
		f << "$ = " << F->my_poly << "\\\\" << endl;
	}

	f << "$Z_i = \\log_\\alpha (1 + \\alpha^i)$\\\\" << endl;

	if (!F->f_is_prime_field && !NT.is_prime(F->e)) {
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					" before report_subfields" << endl;
		}
		report_subfields(f, verbose_level - 2);
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					" after report_subfields" << endl;
		}
	}
	if (!F->f_is_prime_field) {
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					" before report_subfields_detailed" << endl;
		}
		report_subfields_detailed(f, verbose_level - 2);
		if (f_v) {
			cout << "finite_field_io::cheat_sheet_subfields q=" << F->q <<
					" after report_subfields_detailed" << endl;
		}
	}

	if (f_v) {
		cout << "finite_field_io::cheat_sheet_subfields q=" << F->q << " done" << endl;
	}
}

void finite_field_io::report_subfields(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	int h;

	if (f_v) {
		cout << "finite_field_io::report_subfields" << endl;
	}
	ost << "\\subsection*{Subfields}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "h & \\mbox{Subfield} & \\mbox{Polynomial} & \\mbox{Encoding} & d & \\alpha^d \\\\"
			<< endl;
	ost << "\\hline" << endl;
	for (h = 1; h <= F->e; h++) {

		if ((F->e % h) == 0) {

			if (f_v) {
				cout << "finite_field_io::report_subfields "
						"h=" << h << " / " << F->e << endl;
			}

			ost << "\\hline" << endl;

			long int q0;
			long int index_in_multiplicative_group;
			long int generator;

			q0 = NT.i_power_j(F->p, h);

			index_in_multiplicative_group = (F->q - 1) / (q0 - 1);

			generator = F->alpha_power(index_in_multiplicative_group);

			if (h == 1) {

				ost << h << " & ";
				ost << "\\bbF_{" << q0 << "} & ";
				ost << " & ";
				ost << " & " << index_in_multiplicative_group;
				ost << " & " << generator;
				ost << "\\\\" << endl;

			}
			else if (h == F->e) {

				finite_field GFp;
				GFp.finite_field_init_small_order(F->p,
						false /* f_without_tables */,
						false /* f_compute_related_fields */,
						0);

				ring_theory::unipoly_domain FX(&GFp);
				ring_theory::unipoly_object m;

				FX.create_object_by_rank_string(m, F->my_poly,
						0/*verbose_level*/);



				ost << h << " & ";
				ost << "\\bbF_{" << q0 << "} & ";
				FX.print_object(m, ost);
				ost << " & " << F->my_poly;
				ost << " & " << index_in_multiplicative_group;
				ost << " & " << generator;
				ost << "\\\\" << endl;

			}
			else {

				long int poly;

				if (f_v) {
					cout << "finite_field_io::report_subfields "
							"before F->compute_subfield_polynomial" << endl;
				}

				poly = F->compute_subfield_polynomial(
						q0,
						verbose_level - 3);

				if (f_v) {
					cout << "finite_field_io::report_subfields "
							"after F->compute_subfield_polynomial" << endl;
				}

				{
					finite_field GFp;
					GFp.finite_field_init_small_order(F->p,
							false /* f_without_tables */,
							false /* f_compute_related_fields */,
							0);

					ring_theory::unipoly_domain FX(&GFp);
					ring_theory::unipoly_object m;

					FX.create_object_by_rank_string(m, F->my_poly,
							0/*verbose_level*/);
					ring_theory::unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);
					ring_theory::unipoly_object elt;

					FX.create_object_by_rank(elt, poly, verbose_level - 2);



					ost << h << " & ";
					ost << "\\bbF_{" << q0 << "} & ";
					Fq.print_object(elt, ost);
					ost << " & " << poly;
					ost << " & " << index_in_multiplicative_group;
					ost << " & " << generator;
					ost << "\\\\" << endl;
					Fq.delete_object(elt);
				}

			}


		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	if (f_v) {
		cout << "finite_field_io::report_subfields done" << endl;
	}
}

void finite_field_io::report_subfields_detailed(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	int h;

	if (f_v) {
		cout << "finite_field_io::report_subfields_detailed" << endl;
	}
	ost << "\\subsection*{Subfields in Detail}" << endl;
	for (h = 1; h < F->e; h++) {
		if (F->e % h) {
			continue;
		}

		long int poly_numeric, q0;
		finite_field *Fq;

		Fq = NEW_OBJECT(finite_field);

		q0 = NT.i_power_j(F->p, h);


		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"before F->compute_subfield_polynomial" << endl;
		}
		poly_numeric = F->compute_subfield_polynomial(
				q0,
				verbose_level - 2);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"after F->compute_subfield_polynomial" << endl;
		}

		minimum_polynomial *M;

		M = NEW_OBJECT(minimum_polynomial);

		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"before M->compute_subfield_polynomial" << endl;
		}
		M->compute_subfield_polynomial(
				F,
				q0 /* order_subfield */,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"after M->compute_subfield_polynomial" << endl;
		}




		poly_numeric = M->min_poly_rank;

		string poly_text;

		poly_text = std::to_string(poly_numeric);

		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					" before Fq->init_override_polynomial_small_order" << endl;
		}
		Fq->init_override_polynomial_small_order(
				q0, poly_text,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				verbose_level - 2);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					" after Fq->init_override_polynomial_small_order" << endl;
		}

		subfield_structure *Sub;

		Sub = NEW_OBJECT(subfield_structure);

		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"before Sub->init" << endl;
		}
		Sub->init(F /* FQ */, Fq, verbose_level - 2);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"after Sub->init" << endl;
		}


		ost << "The subfield ${\\mathbb F}_{" << q0 << "}$, "
				"divisor $h=" << h << "$, is "
				"generated by polynomial " << poly_numeric << ":\\\\" << endl;

		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"before Sub->report" << endl;
		}
		Sub->report(ost, verbose_level);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"after Sub->report" << endl;
		}

		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"before M->report_table" << endl;
		}
		M->report_table(ost);
		if (f_v) {
			cout << "finite_field_io::report_subfields_detailed "
					"after M->report_table" << endl;
		}

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		FREE_OBJECT(M);
		FREE_OBJECT(Sub);
		FREE_OBJECT(Fq);
	}
	if (f_v) {
		cout << "finite_field_io::report_subfields_detailed done" << endl;
	}
}


void finite_field_io::cheat_sheet_addition_table(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "finite_field_io::cheat_sheet_addition_table" << endl;
	}

	if (F->q <= 64) {
		f << "$$" << endl;
		latex_addition_table(f, false /* f_elements_exponential */,
				F->get_symbol_for_print());
		f << "$$" << endl;
	}
	else {
		f << "Addition table omitted" << endl;
	}



	if (f_v) {
		cout << "finite_field_io::cheat_sheet_addition_table done" << endl;
	}
}

void finite_field_io::cheat_sheet_multiplication_table(
		std::ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "finite_field_io::cheat_sheet_multiplication_table" << endl;
	}

	if (F->q <= 64) {
		f << "$$" << endl;
		latex_multiplication_table(f, false /* f_elements_exponential */,
				F->get_symbol_for_print());
		f << "$$" << endl;
	}
	else {
		f << "Multiplication table omitted" << endl;
	}



	if (f_v) {
		cout << "finite_field_io::cheat_sheet_multiplication_table done" << endl;
	}
}

void finite_field_io::cheat_sheet_power_table(
		std::ostream &ost, int f_with_polynomials,
		int verbose_level)
{
	int *Powers;
	int i, j, t;
	int len = F->q;
	int *v;
	geometry::other_geometry::geometry_global Gg;

	t = F->primitive_root();

	v = NEW_int(F->e);
	Powers = NEW_int(len);
	power_table(t, Powers, len);

	ost << "\\subsection*{Cyclic structure}" << endl;


	if (F->q > 1024) {
		ost << "The field is too large. For the sake of space, "
				"the cyclic list cannot be generated\\\\" << endl;
	}
	else {
		ost << "$$" << endl;
		cheat_sheet_power_table_top(
				ost, f_with_polynomials, verbose_level);

		for (i = 0; i < len; i++) {

			if (i && (i % 32) == 0) {
				cheat_sheet_power_table_bottom(
						ost, f_with_polynomials, verbose_level);
				ost << "$$" << endl;
				ost << "$$" << endl;
				cheat_sheet_power_table_top(
						ost, f_with_polynomials, verbose_level);
			}

			Gg.AG_element_unrank(F->p, v, 1, F->e, Powers[i]);

			ost << i << " & " << t << "^{" << i << "} & " << Powers[i] << " & ";
			for (j = F->e - 1; j >= 0; j--) {
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
	}

	FREE_int(v);
	FREE_int(Powers);

}

void finite_field_io::cheat_sheet_power_table_top(
		std::ostream &ost, int f_with_polynomials, int verbose_level)
{
	ost << "\\begin{array}{|r|r|r|r|";
	if (f_with_polynomials) {
		ost << "r|";
	}
	ost << "}" << endl;
	ost << "\\hline" << endl;


	ost << "i & " << F->get_symbol_for_print() << "^i & "
			<< F->get_symbol_for_print() << "^i & \\mbox{vector}";
	if (f_with_polynomials) {
		ost << "& \\mbox{reduced rep.}";
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
}

void finite_field_io::cheat_sheet_power_table_bottom(
		std::ostream &ost, int f_with_polynomials, int verbose_level)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
}

void finite_field_io::cheat_sheet_table_of_elements(
		std::ostream &ost, int verbose_level)
{
	int *v;
	int i, j;
	//int f_first;
	//std::string my_symbol;

	v = NEW_int(F->e);

	//my_symbol.assign("\alpha");

	ost << "\\subsection*{Table of Elements of ${\\mathbb F}_{" << F->q << "}$}" << endl;

	int I, Imax;

	Imax = (F->q + 24) / 25;

	for (I = 0; I < Imax; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;

		geometry::other_geometry::geometry_global Gg;

		for (j = 0; j < 25; j++) {

			i = I * 25 + j;
			if (i >= F->q) {
				continue;
			}

			Gg.AG_element_unrank(F->p, v, 1, F->e, i);
			ost << setw(3) << i;
			ost << " & ";
			//f_first = true;

			int h;
			for (h = F->e - 1; h >= 0; h--) {
				ost << v[h];
			}
			ost << " & ";

			print_element_as_polynomial(ost, v, verbose_level);

			ost << "\\\\" << endl;

	#if 0
			ost << " & ";
			print_element_with_symbol(ost, i,
				true /*f_print_as_exponentials*/,
				10 /*width*/, my_symbol);
	#endif
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}

	FREE_int(v);

}

void finite_field_io::print_element_as_polynomial(
		std::ostream &ost, int *v, int verbose_level)
{
	int j;
	int f_first = true;

	for (j = F->e - 1; j >= 0; j--) {
		if (v[j] == 0) {
			continue;
		}

		if (f_first) {
			f_first = false;
		}
		else {
			ost << " + ";
		}

		if (j == 0 || v[j] > 1) {
			ost << setw(3) << v[j];
		}
		if (j) {
			ost << F->get_symbol_for_print();
		}
		if (j > 1) {
			ost << "^{" << j << "}";
		}
	}
	if (f_first) {
		ost << "0";
	}
}

void finite_field_io::cheat_sheet_main_table(
		std::ostream &f, int verbose_level)
{
	int *v;
	int i, j, a;
	int f_first;


	if (!F->f_has_table) {
		cout << "finite_field_io::cheat_sheet_main_table "
				"!f_has_table, skipping" << endl;
		return;
	}

	v = NEW_int(F->e);


	int nb_cols = 7;

	if (F->e > 1) {
		nb_cols += 3;
	}
	if ((F->e % 2) == 0 && F->e > 2) {
		nb_cols += 2;
	}
	if ((F->e % 3) == 0 && F->e > 3) {
		nb_cols += 2;
	}



	if (F->q > 1024) {
		f << "The field is too large to print all elements. \\\\" << endl;
	}
	else {
		cheat_sheet_main_table_top(f, nb_cols);

		geometry::other_geometry::geometry_global Gg;

		for (i = 0; i < F->q; i++) {
			Gg.AG_element_unrank(F->p, v, 1, F->e, i);
			f << setw(3) << i << " & ";
			f_first = true;
			for (j = F->e - 1; j >= 0; j--) {
				if (v[j] == 0) {
					continue;
				}

				if (f_first) {
					f_first = false;
				}
				else {
					f << " + ";
				}

				if (j == 0 || v[j] > 1) {
					f << setw(3) << v[j];
				}
				if (j) {
					f << F->get_symbol_for_print();
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
				true /*f_print_as_exponentials*/,
				10 /*width*/, F->get_symbol_for_print());



			// - gamma_i:
			f << " &" << F->negate(i);
			// gamma_i^{-1}:
			if (i == 0) {
				f << " & \\mbox{DNE}";
			}
			else {
				f << " &" << F->inverse(i);
			}



			// log_alpha:
			if (i == 0) {
				f << " & \\mbox{DNE}";
			}
			else {
				f << " &" << F->log_alpha(i);
			}
			// alpha_power:
			f << " &" << F->alpha_power(i);


			// Z_i:
			a = F->add(1, F->alpha_power(i));
			if (a == 0) {
				f << " & \\mbox{DNE}";
			}
			else {
				f << " &" << F->log_alpha(a);
			}




			// additional columns for extension fields:
			if (F->e > 1) {
				f << " &" << F->frobenius_power(i, 1, 0 /* verbose_level */);
				f << " &" << F->absolute_trace(i);
				f << " &" << F->absolute_norm(i);
			}

			if ((F->e % 2) == 0 && F->e > 2) {
				f << " &" << F->T2(i);
				f << " &" << F->N2(i);
			}
			if ((F->e % 3) == 0 && F->e > 3) {
				f << " &" << F->T3(i);
				f << " &" << F->N3(i);
			}


			f << "\\\\" << endl;

			if ((i % 25) == 0 && i) {
				cheat_sheet_main_table_bottom(f);
				cheat_sheet_main_table_top(f, nb_cols);
			}
		}
		cheat_sheet_main_table_bottom(f);
	}


	FREE_int(v);

}

void finite_field_io::cheat_sheet_main_table_top(
		std::ostream &f, int nb_cols)
{
	f << "$$" << endl;
	f << "\\begin{array}{|*{" << nb_cols << "}{r|}}" << endl;
	f << "\\hline" << endl;
	f << "i & \\gamma_i ";
	f << "& -\\gamma_i";
	f << "& \\gamma_i^{-1}";
	f << "& \\log_" << F->get_symbol_for_print() << "(\\gamma_i)";
	f << "& " << F->get_symbol_for_print() << "^i";
	f << "& Z_i";
	if (F->e > 1) {
		f << "& \\phi(\\gamma_i) ";
		f << "& T(\\gamma_i) ";
		f << "& N(\\gamma_i) ";
	}
	if ((F->e % 2) == 0 && F->e > 2) {
		f << "& T_2(\\gamma_i) ";
		f << "& N_2(\\gamma_i) ";
	}
	if ((F->e % 3) == 0 && F->e > 3) {
		f << "& T_3(\\gamma_i) ";
		f << "& N_3(\\gamma_i) ";
	}
	f << "\\\\" << endl;
	f << "\\hline" << endl;
}

void finite_field_io::cheat_sheet_main_table_bottom(
		std::ostream &f)
{
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
}


void finite_field_io::display_table_of_projective_points(
	std::ostream &ost, long int *Pts, int nb_pts, int len)
{

	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;

	display_table_of_projective_points2(ost, Pts, nb_pts, len);

	ost << "$$}%" << endl;
}

void finite_field_io::display_table_of_projective_points2(
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
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				coords, 1, len, Pts[i]);
		ost << i << " & " << Pts[i] << " & ";
		Int_vec_print(ost, coords, len);
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

void finite_field_io::display_table_of_projective_points_easy(
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
		F->Projective_space_basic->PG_element_unrank_modified_lint(
				coords, 1, len, Pts[i]);
		Int_vec_print(ost, coords, len);
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




void finite_field_io::print_matrix_latex(
		std::ostream &ost, int *A, int m, int n)
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

std::string finite_field_io::stringify_matrix_latex(
		int *A, int m, int n)
{
	string s;
	int i, j, a;

	s += "\\left[\n";
	s += "\\begin{array}{*{" + std::to_string(n) + "}{r}}\n";
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
			s += stringify_element(a);
#endif

			if (j < n - 1)
				s += " & ";
		}
		s += "\\\\\n";
	}
	s += "\\end{array}\n";
	s += "\\right]\n";
	return  s;
}


void finite_field_io::print_matrix_numerical_latex(
		std::ostream &ost, int *A, int m, int n)
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

void finite_field_io::read_from_string_coefficient_vector(
		std::string &str,
		int *&coeff, int &len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_io::read_from_string_coefficient_vector" << endl;
	}

	number_theory::number_theory_domain NT;


	{
		int *coeffs;
		int a, i;

		Int_vec_scan(str, coeffs, len);

		coeff = NEW_int(len);

		Int_vec_zero(coeff, len);


		for (i = 0; i < len; i++) {
			a = coeffs[i];
			if (a < 0 || a >= F->q) {
				if (F->e > 1) {
					cout << "finite_field_io::read_from_string_coefficient_vector "
							"In an extension field of order "
							<< F->q << ", what do you mean by " << a << endl;
					exit(1);
				}
				a = NT.mod(a, F->q);
			}
			coeff[i] = a;

		}
		FREE_int(coeffs);
	}
	if (f_v) {
		cout << "finite_field_io::read_from_string_coefficient_vector done" << endl;
	}
}



}}}}


