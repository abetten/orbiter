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

void finite_field::cheat_sheet_PG(int n,
		int f_surface, int verbose_level)
{
	//const char *override_poly;
	char fname[1000];
	char title[1000];
	char author[1000];
	//int f_with_group = FALSE;
	//int f_semilinear = FALSE;
	//int f_basis = TRUE;
	//int q = F->q;

	sprintf(fname, "PG_%d_%d.tex", n, q);
	sprintf(title, "Cheat Sheet PG($%d,%d$)", n, q);
	//sprintf(author, "");
	author[0] = 0;
	projective_space *P;

	P = NEW_OBJECT(projective_space);
	cout << "before P->init" << endl;
	P->init(n, this,
		TRUE /* f_init_incidence_structure */,
		verbose_level/*MINIMUM(2, verbose_level)*/);


	{
	ofstream f(fname);
	latex_interface L;

	L.head(f,
			FALSE /* f_book*/,
			TRUE /* f_title */,
			title, author,
			FALSE /* f_toc */,
			FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);

	f << "\\small" << endl;
	f << "\\arraycolsep=2pt" << endl;
	f << "\\parindent=0pt" << endl;
	f << "$q = " << q << "$\\\\" << endl;
	f << "$p = " << p << "$\\\\" << endl;
	f << "$e = " << e << "$\\\\" << endl;
	f << "$n = " << n << "$\\\\" << endl;
	f << "Number of points = " << P->N_points << "\\\\" << endl;
	f << "Number of lines = " << P->N_lines << "\\\\" << endl;
	f << "Number of lines on a point = " << P->r << "\\\\" << endl;
	f << "Number of points on a line = " << P->k << "\\\\" << endl;

	f << "\\clearpage" << endl << endl;
	f << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	cheat_sheet(f, verbose_level);

	if (n == 2) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{The Plane}" << endl;
		char fname_base[1000];
		long int *set;
		int i;
		int rad = 17000;

		set = NEW_lint(P->N_points);
		for (i = 0; i < P->N_points; i++) {
			set[i] = i;
			}
		sprintf(fname_base, "plane_of_order_%d", q);
		P->draw_point_set_in_plane(fname_base,
				set, P->N_points,
				TRUE /*f_with_points*/,
				TRUE /*f_point_labels*/,
				FALSE /*f_embedded*/,
				FALSE /*f_sideways*/,
				rad,
				0 /* verbose_level */);
		FREE_lint(set);
		f << "{\\scriptsize" << endl;
		f << "$$" << endl;
		f << "\\input " << fname_base << ".tex" << endl;
		f << "$$" << endl;
		f << "}%%" << endl;
		}

	f << "\\clearpage" << endl << endl;
	f << "\\section{Points and Lines}" << endl;
	P->cheat_sheet_points(f, verbose_level);

	P->cheat_sheet_point_table(f, verbose_level);



	f << "\\clearpage" << endl << endl;
	P->cheat_sheet_points_on_lines(f, verbose_level);

	f << "\\clearpage" << endl << endl;
	P->cheat_sheet_lines_on_points(f, verbose_level);


	// report subspaces:
	int k;

	for (k = 1; k < n; k++) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Subspaces of dimension " << k << "}" << endl;
		P->cheat_sheet_subspaces(f, k, verbose_level);
		}



	if (n >= 2 && P->N_lines < 25) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Line intersections}" << endl;
		P->cheat_sheet_line_intersection(f, verbose_level);
		}


	if (n >= 2 && P->N_points < 25) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Line through point-pairs}" << endl;
		P->cheat_sheet_line_through_pairs_of_points(f, verbose_level);
		}

	if (f_surface) {
		surface_domain *S;

		S = NEW_OBJECT(surface_domain);
		S->init(this, verbose_level + 2);

		f << "\\clearpage" << endl << endl;
		f << "\\section{Surface}" << endl;
		f << "\\subsection{Steiner Trihedral Pairs}" << endl;
		S->latex_table_of_trihedral_pairs(f);

		f << "\\clearpage" << endl << endl;
		f << "\\subsection{Eckardt Points}" << endl;
		S->latex_table_of_Eckardt_points(f);

#if 1
		long int *Lines;

		cout << "creating S_{3,1}:" << endl;
		Lines = NEW_lint(27);
		S->create_special_double_six(Lines,
				3 /*a*/, 1 /*b*/, 0 /* verbose_level */);
		S->create_remaining_fifteen_lines(Lines,
				Lines + 12, 0 /* verbose_level */);
		P->Grass_lines->print_set(Lines, 27);

		FREE_lint(Lines);
#endif
		FREE_OBJECT(S);
		}

	L.foot(f);
	}
	file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}


void finite_field::print_minimum_polynomial(int p, const char *polynomial)
{
	finite_field GFp;

	GFp.init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m, n;

	FX.create_object_by_rank_string(m, polynomial, 0);
	FX.create_object_by_rank_string(n, polynomial, 0);
	{
	unipoly_domain Fq(&GFp, m);

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
	if (e > 1) {
		//char *poly;

		//poly = get_primitive_polynomial(p, e, 0 /* verbose_level */);

		cout << "polynomial = ";
		print_minimum_polynomial(p, polynomial);
		cout << endl;
		//cout << " = " << poly << endl;
		print_tables_extension_field(polynomial);
		}
	else {
		print_tables();
		}
	if (f_add_mult_table) {
		print_add_mult_tables();
		}
}

void finite_field::print_add_mult_tables()
{
	cout << "addition table:" << endl;
	print_integer_matrix_width(cout, add_table, q, q, q, log10_of_q + 1);
	cout << endl;


	cout << "multiplication table:" << endl;
	print_integer_matrix_width(cout, mult_table, q, q, q, log10_of_q + 1);
	cout << endl;
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
	GFp.init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m;



	FX.create_object_by_rank_string(m, poly, verbose_level);

	unipoly_domain Fq(&GFp, m);
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
		Fq.create_object_by_rank(elt, i);
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
		if (EVEN(l))
			cout << "+";
		else
			cout << "-";
		}
}

void finite_field::print_element(ostream &ost, int a)
{
	int width;


	if (e == 1) {
		ost << a;
	} else {
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
	} else {
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
		int a, int f_exponential, int width, const char *symbol)
{
	int b;

	if (f_exponential) {
		if (symbol == NULL) {
			cout << "finite_field::print_element_with_symbol "
					"symbol == NULL" << endl;
			return;
			}
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
			if (b == q - 1)
				b = 0;
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
		int a, int f_exponential, int width, const char *symbol)
{
	int b;

	if (f_exponential) {
		if (symbol == NULL) {
			cout << "finite_field::print_element_with_symbol_str "
					"symbol == NULL" << endl;
			return;
			}
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
			if (b == q - 1)
				b = 0;
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

void finite_field::int_vec_print(ostream &ost, int *v, int len)
{
	int i;
	ost << "(";
	for (i = 0; i < len; i++) {
		print_element(ost, v[i]);
		if (i < len - 1)
			ost << ", ";
		}
	ost << ")";
}

void finite_field::int_vec_print_elements_exponential(ostream &ost,
		int *v, int len, const char *symbol_for_print)
{
	int i;
	ost << "(";
	for (i = 0; i < len; i++) {
		print_element_with_symbol(ost, v[i],
			TRUE /*f_print_as_exponentials*/,
			10 /*width*/, symbol_for_print);
		if (i < len - 1)
			ost << ", ";
		}
	ost << ")";
}

void finite_field::latex_addition_table(ostream &f,
		int f_elements_exponential, const char *symbol_for_print)
{
	int i, j, k;

	//f << "$$" << endl;
	f << "\\arraycolsep=1pt" << endl;
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
	//f << "$$" << endl;
}

void finite_field::latex_multiplication_table(ostream &f,
		int f_elements_exponential, const char *symbol_for_print)
{
	int i, j, k;

	f << "\\arraycolsep=1pt" << endl;
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
		const char *symbol_for_print, int *M, int m, int n)
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


void finite_field::power_table(int t, int *power_table, int len)
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
	int i, j;
	int *v;
	int f_first;
	int a, h;
	const char *symbol_for_print = "\\alpha";
	number_theory_domain NT;


	if (f_v) {
		cout << "finite_field::cheat_sheet" << endl;
		}
	v = NEW_int(e);

	f << "\\small" << endl;
	if (e > 1) {
		f << "polynomial: ";
		finite_field GFp;
		GFp.init(p, 0);

		unipoly_domain FX(&GFp);
		unipoly_object m;

		FX.create_object_by_rank_string(m, polynomial, verbose_level - 2);
		f << "$";
		FX.print_object(m, f);
		f << "$ = " << polynomial << "\\\\" << endl;
		}

	f << "$Z_i = \\log_\\alpha (1 + \\alpha^i)$\\\\" << endl;

	if (e > 1 && !NT.is_prime(e)) {
	f << "Subfields:" << endl;
	f << "$$" << endl;
	f << "\\begin{array}{|r|r|r|}" << endl;
	f << "\\hline" << endl;
	f << "\\mbox{order} & \\mbox{polynomial} & \\mbox{polynomial} \\\\"
			<< endl;
	f << "\\hline" << endl;
	for (h = 2; h < e; h++) {
		if ((e % h) == 0) {


			f << "\\hline" << endl;
			int poly;

			poly = compute_subfield_polynomial(
					NT.i_power_j(p, h), verbose_level);
			{
				finite_field GFp;
				GFp.init(p, 0);

				unipoly_domain FX(&GFp);
				unipoly_object m;

				FX.create_object_by_rank_string(m, polynomial,
						0/*verbose_level*/);
				unipoly_domain Fq(&GFp, m);
				unipoly_object elt;

				FX.create_object_by_rank(elt, poly);
				f << NT.i_power_j(p, h) << " & " << poly << " & ";
				Fq.print_object(elt, f);
				f << "\\\\" << endl;
				Fq.delete_object(elt);
			}

			}
		}
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
	}


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

	cheat_sheet_top(f, nb_cols);

	geometry_global Gg;

	for (i = 0; i < q; i++) {
		Gg.AG_element_unrank(p, v, 1, e, i);
		f << setw(3) << i << " & ";
		f_first = TRUE;
		for (j = e - 1; j >= 0; j--) {
			if (v[j] == 0)
				continue;

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
			cheat_sheet_bottom(f);
			cheat_sheet_top(f, nb_cols);
			}
		}

	cheat_sheet_bottom(f);


	FREE_int(v);
	if (f_v) {
		cout << "finite_field::cheat_sheet done" << endl;
		}
}

void finite_field::cheat_sheet_tables(ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j;
	//int *v;
	//int f_first;
	//int a, h;
	const char *symbol_for_print = "\\alpha";


	if (f_v) {
		cout << "finite_field::cheat_sheet_tables" << endl;
		}
	//v = NEW_int(e);

	if (q <= 64) {
		f << "$$" << endl;
		latex_addition_table(f, FALSE /* f_elements_exponential */,
				symbol_for_print);
		if (q >= 10) {
			f << "$$" << endl;
			f << "$$" << endl;
			}
		else {
			f << "\\qquad" << endl;
			}
		latex_addition_table(f, TRUE /* f_elements_exponential */,
				symbol_for_print);
		f << "$$" << endl;

		f << "$$" << endl;
		latex_multiplication_table(f, FALSE /* f_elements_exponential */,
				symbol_for_print);
		if (q >= 10) {
			f << "$$" << endl;
			f << "$$" << endl;
			}
		else {
			f << "\\qquad" << endl;
			}
		latex_multiplication_table(f, TRUE /* f_elements_exponential */,
				symbol_for_print);
		f << "$$" << endl;
		}
	else {
		f << "Addition and multiplication tables omitted" << endl;
		}


	//FREE_int(v);
	if (f_v) {
		cout << "finite_field::cheat_sheet_tables done" << endl;
		}
}

void finite_field::cheat_sheet_top(ostream &f, int nb_cols)
{
	f << "$$";
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

void finite_field::cheat_sheet_bottom(ostream &f)
{
	f << "\\hline" << endl;
	f << "\\end{array}" << endl;
	f << "$$" << endl;
}


void finite_field::display_table_of_projective_points(
	ostream &ost, long int *Pts, int nb_pts, int len)
{
	int i;
	int *coords;

	coords = NEW_int(len);
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & a_i & P_{a_i}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified_lint(coords, 1, len, Pts[i]);
		ost << i << " & " << Pts[i] << " & ";
		int_vec_print(ost, coords, len);
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
	ost << "$$}%" << endl;
	FREE_int(coords);
}


void finite_field::export_magma(int d, long int *Pts, int nb_pts, char *fname)
{
	char fname2[1000];
	int *v;
	int h, i, a, b;

	v = NEW_int(d);
	strcpy(fname2, fname);
	replace_extension_with(fname2, ".magma");

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

void finite_field::export_gap(int d, long int *Pts, int nb_pts, char *fname)
{
	char fname2[1000];
	int *v;
	int h, i, a, b;

	v = NEW_int(d);
	strcpy(fname2, fname);
	replace_extension_with(fname2, ".gap");

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




}}


