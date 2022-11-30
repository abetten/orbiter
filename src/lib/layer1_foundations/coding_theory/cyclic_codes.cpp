/*
 * cyclic_codes.cpp
 *
 *  Created on: Oct 9, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {

cyclic_codes::cyclic_codes()
{

}

cyclic_codes::~cyclic_codes()
{

}



void cyclic_codes::make_BCH_code(int n,
		field_theory::finite_field *F, int d,
		field_theory::nth_roots *&Nth, ring_theory::unipoly_object &P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cyclic_codes::make_BCH_code q=" << F->q << " n=" << n
				<< " d=" << d << endl;
	}


	Nth = NEW_OBJECT(field_theory::nth_roots);

	Nth->init(F, n, verbose_level);

	int *Selection;
	int *Sel;
	int nb_sel;
	int i, j;


	Selection = NEW_int(Nth->Cyc->S->nb_sets);
	Sel = NEW_int(Nth->Cyc->S->nb_sets);

	for (i = 0; i < Nth->Cyc->S->nb_sets; i++) {
		Selection[i] = FALSE;
	}

	for (i = 0; i < d - 1; i++) {
		j = Nth->Cyc->Index[(1 + i) % n];
		Selection[j] = TRUE;
	}

	nb_sel = 0;
	for (i = 0; i < Nth->Cyc->S->nb_sets; i++) {
		if (Selection[i]) {
			Sel[nb_sel++] = i;
		}
	}

	if (f_v) {
		cout << "cyclic_codes::make_BCH_code Sel=";
		Int_vec_print(cout, Sel, nb_sel);
		cout << endl;
	}

	ring_theory::unipoly_object Q;

	Nth->FX->create_object_by_rank(P, 1, __FILE__, __LINE__, 0 /*verbose_level*/);
	Nth->FX->create_object_by_rank(Q, 1, __FILE__, __LINE__, 0 /*verbose_level*/);

	for (i = 0; i < nb_sel; i++) {

		j = Sel[i];

		if (f_v) {
			cout << "cyclic_codes::make_BCH_code P=";
			Nth->FX->print_object(P, cout);
			cout << endl;
			cout << "j=" << j << endl;
			Nth->FX->print_object(Nth->min_poly_beta_Fq[j], cout);
			cout << endl;
		}
		Nth->FX->mult(P, Nth->min_poly_beta_Fq[j], Q, verbose_level);
		if (f_v) {
			cout << "cyclic_codes::make_BCH_code Q=";
			Nth->FX->print_object(Q, cout);
			cout << endl;
		}
		Nth->FX->assign(Q, P, 0 /* verbose_level */);
	}



	if (f_v) {
		cout << "cyclic_codes::make_BCH_code q=" << F->q << " n=" << n
				<< " d=" << d << " done" << endl;
	}
}



void cyclic_codes::make_cyclic_code(int n, int q, int t,
		int *roots, int nb_roots, int f_poly, std::string &poly,
		int f_dual, std::string &fname_txt, std::string &fname_csv,
		int verbose_level)
// this function creates a finite field, using the given polynomial if necessary
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, e, m, r, i;
	ring_theory::longinteger_object Qm1, Index;
	ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;
	orbiter_kernel_system::file_io Fio;

	NT.factor_prime_power(q, p, e);
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code q=" << q << " p=" << q
				<< " e=" << e << " n=" << n << endl;
		for (i = 0; i < nb_roots; i++) {
			cout << roots[i] << " ";
		}
		cout << endl;
		if (f_dual) {
			cout << "cyclic_codes::make_cyclic_code dual code" << endl;
		}
	}
	m = NT.order_mod_p(q, n);
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code order mod q is m=" << m << endl;
	}
	D.create_qnm1(Qm1, q, m);

	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(Qm1, n, Index, r);
	//b = (q - 1) / n;
	if (r != 0) {
		cout << "cyclic_codes::make_cyclic_code n does not divide q^m-1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code GF(" << q << "^" << m << ") has "
				<< n << "-th roots of unity" << endl;
		if (Index.is_one()) {
			cout << "cyclic_codes::make_cyclic_code this is a primitive code" << endl;
		}
		else {
			cout << "cyclic_codes::make_cyclic_code we take as " << n << "-th root \\beta = \\alpha^"
			<< Index << ", where \\alpha is a primitive element of "
					"the field" << endl;
		}
	}

	int j, degree, field_degree;
	int *taken;
	int *transversal, tl = 0;

	field_degree = m * e;
	degree = 0;
	taken = NEW_int(n);
	transversal = NEW_int(n);
	for (i = 0; i < n; i++) {
		taken[i] = FALSE;
	}


	for (i = 0; i < nb_roots; i++) {
		j = roots[i];
		if (taken[j]) {
			cout << q << "-cyclotomic coset of "
					<< j << " already taken" << endl;
			continue;
		}
		if (!taken[j]) {
			transversal[tl++] = j;
		}
		taken[j] = TRUE;
		degree++;
		if (f_v) {
			cout << q << "-cyclotomic coset of "
					<< j << " : " << j;
		}
		while (TRUE) {
			j = (q * j) % n;
			if (taken[j]) {
				break;
			}
			taken[j] = TRUE;
			degree++;
			if (f_v) {
				cout << "," << j;
			}
		}
		if (f_v) {
			cout << " degree=" << degree << endl;
		}
	}

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code transversal: ";
		for (i = 0; i < tl; i++) {
			cout << transversal[i] << " ";
		}
		cout << endl;
		cout << "cyclic_codes::make_cyclic_code exponents:";
		for (i = 0; i < n; i++) {
			if (!taken[i]) {
				continue;
			}
			cout << i << ", ";
		}
		cout << endl;
		cout << "cyclic_codes::make_cyclic_code degree=" << degree << endl;
	}

	if (f_dual) {
		for (i = 0; i < n; i++) {
			taken[i] = !taken[i];
		}
		degree = n - degree;
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code dually, exponents:";
			for (i = 0; i < n; i++) {
				if (!taken[i])
					continue;
				cout << i << ", ";
			}
			cout << endl;
			cout << "cyclic_codes::make_cyclic_code degree=" << degree << endl;
		}
	}

	field_theory::finite_field Fp;
	ring_theory::unipoly_object M;
	ring_theory::unipoly_object beta, beta_i, c;

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating the finite field of order " << p << endl;
	}
	Fp.finite_field_init(p, FALSE /* f_without_tables */, verbose_level - 1);

	ring_theory::unipoly_domain FpX(&Fp);
	string field_poly;
	knowledge_base K;

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code before K.get_primitive_polynomial" << endl;
	}
	K.get_primitive_polynomial(field_poly, p, field_degree, 0);

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code before FpX.create_object_by_rank_string" << endl;
	}
	FpX.create_object_by_rank_string(M, field_poly, verbose_level - 1);

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code choosing the following irreducible "
				"and primitive polynomial:" << endl;
		FpX.print_object(M, cout); cout << endl;
	}

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating unipoly_domain Fq modulo M" << endl;
	}
	ring_theory::unipoly_domain Fq(&Fp, M, verbose_level);  // Fq = Fp[X] modulo factor polynomial M
	if (f_vv) {
		cout << "cyclic_codes::make_cyclic_code extension field created" << endl;
	}

	Fq.create_object_by_rank(c, 0, __FILE__, __LINE__, verbose_level);
	Fq.create_object_by_rank(beta, p, __FILE__, __LINE__, verbose_level); // the element alpha
	Fq.create_object_by_rank(beta_i, 1, __FILE__, __LINE__, verbose_level);
	if (!Index.is_one()) {
		//Fq.power_int(beta, b);
		if (f_v) {
			cout << "\\alpha = ";
			Fq.print_object(beta, cout);
			cout << endl;
		}
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.power_longinteger" << endl;
		}
		Fq.power_longinteger(beta, Index, verbose_level - 1);
		if (f_v) {
			cout << "\\beta = \\alpha^" << Index << " = ";
			Fq.print_object(beta, cout);
			cout << endl;
		}
	}
	else {
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code this is a primitive BCH code" << endl;
		}
	}

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code before allocating generator etc" << endl;
	}

	ring_theory::unipoly_object *generator = NEW_OBJECTS(ring_theory::unipoly_object, degree + 2);
	ring_theory::unipoly_object *tmp = NEW_OBJECTS(ring_theory::unipoly_object, degree + 1);
	ring_theory::unipoly_object *coeffs = NEW_OBJECTS(ring_theory::unipoly_object, 2);
	ring_theory::unipoly_object Pc, Pd;


	// create the polynomial X - a:
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating X-a" << endl;
	}
	for (i = 0; i < 2; i++) {
		if (i == 1) {
			Fq.create_object_by_rank(coeffs[i], 1, __FILE__, __LINE__, verbose_level);
		}
		else {
			Fq.create_object_by_rank(coeffs[i], 0, __FILE__, __LINE__, verbose_level);
		}
	}
	for (i = 0; i <= degree; i++) {
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code creating generator[" << i << "]" << endl;
		}
		Fq.create_object_by_rank(generator[i], 0, __FILE__, __LINE__, verbose_level);
		Fq.create_object_by_rank(tmp[i], 0, __FILE__, __LINE__, verbose_level);
	}
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating generator[0]" << endl;
	}
	Fq.create_object_by_rank(generator[0], 1, __FILE__, __LINE__, verbose_level);

	// now coeffs has degree 1
	// and generator has degree 0

	if (f_vv) {
		cout << "cyclic_codes::make_cyclic_code coeffs:" << endl;
		print_polynomial(Fq, 1, coeffs);
		cout << endl;
		cout << "cyclic_codes::make_cyclic_code generator:" << endl;
		print_polynomial(Fq, 0, generator);
		cout << endl;
	}

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating Pc" << endl;
	}
	Fq.create_object_by_rank(Pc, 0, __FILE__, __LINE__, verbose_level);
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code creating Pd" << endl;
	}
	Fq.create_object_by_rank(Pd, 0, __FILE__, __LINE__, verbose_level);

	r = 0;
	for (i = 0; i < n; i++) {
		if (f_v) {
			cout << "i=" << i << ", r=" << r << endl;
		}
		if (!taken[i]) {
			continue;
		}
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code working on root " << i << endl;
		}
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.assign beta" << endl;
		}
		Fq.assign(beta, beta_i, verbose_level);
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.power_int" << endl;
		}
		Fq.power_int(beta_i, i, verbose_level);
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.negate" << endl;
		}
		Fq.negate(beta_i);
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.assign beta_i" << endl;
		}
		Fq.assign(beta_i, coeffs[0], verbose_level);
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code root: " << i << " : ";
			Fq.print_object(beta_i, cout);
			//cout << " : ";
			//print_polynomial(Fq, 2, coeffs);
			cout << endl;
		}


		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.assign(generator[j], tmp[j])" << endl;
		}
		for (j = 0; j <= r; j++) {
			Fq.assign(generator[j], tmp[j], verbose_level);
		}

		//cout << "tmp:" << endl;
		//print_polynomial(Fq, r, tmp);
		//cout << endl;

		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code before Fq.assign(tmp[j], generator[j + 1])" << endl;
		}
		for (j = 0; j <= r; j++) {
			Fq.assign(tmp[j], generator[j + 1], verbose_level);
			}
		Fq.delete_object(generator[0]);
		Fq.create_object_by_rank(generator[0], 0, __FILE__, __LINE__, verbose_level);

		//cout << "generator after shifting up:" << endl;
		//print_polynomial(Fq, r + 1, generator);
		//cout << endl;

		for (j = 0; j <= r; j++) {
			if (f_v) {
				cout << "cyclic_codes::make_cyclic_code j=" << j << endl;
			}
			if (f_v) {
				cout << "cyclic_codes::make_cyclic_code before Fq.mult(tmp[j], coeffs[0], Pc)" << endl;
			}
			Fq.mult(tmp[j], coeffs[0], Pc, verbose_level - 1);
			if (f_v) {
				cout << "cyclic_codes::make_cyclic_code before Fq.add()" << endl;
			}
			Fq.add(Pc, generator[j], Pd);
			if (f_v) {
				cout << "cyclic_codes::make_cyclic_code before Fq.assign()" << endl;
			}
			Fq.assign(Pd, generator[j], verbose_level);
		}
		r++;
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code r=" << r << endl;
		}
		if (f_v) {
			cout << "cyclic_codes::make_cyclic_code current polynomial: ";
			print_polynomial(Fq, r, generator);
			cout << endl;
		}

	}
	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code The generator polynomial is: ";
		print_polynomial(Fq, r, generator);
		cout << endl;
	}

	Fq.delete_object(c);
	Fq.delete_object(beta);
	Fq.delete_object(beta_i);


	int *generator_subfield;
	int *Genma;

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code before field_reduction" << endl;
	}
	field_reduction(n, q, p, e, m, Fp, Fq, r,
		generator, generator_subfield, f_poly, poly, verbose_level);
	cout << "cyclic_codes::make_cyclic_code generator polynomial:" << endl;
	for (j = 0; j <= degree; j++) {
		cout << generator_subfield[j] << " ";
	}
	cout << endl;

	if (f_v) {
		cout << "cyclic_codes::make_cyclic_code before generator_matrix_cyclic_code" << endl;
	}
	generator_matrix_cyclic_code(n, degree, generator_subfield, Genma);
	cout << "cyclic_codes::make_cyclic_code generator matrix: " << endl;
	Int_vec_print_integer_matrix_width(cout, Genma, n - degree, n, n, 3);


	{
		ofstream fp(fname_txt);
		int k = n - degree;


		fp << n << " " << k << " " << t << " " << q << endl;
		for (i = 0; i < k; i++) {
			for (j = 0; j < n; j++) {
				fp << Genma[i * n + j] << " ";
				}
			fp << endl;
			}
		fp << endl;
	}
	cout << "cyclic_codes::make_cyclic_code Written file " << fname_txt << " of size "
			<< Fio.file_size(fname_txt) << endl;


	{
	int k = n - degree;


	Fio.int_matrix_write_csv(fname_csv, Genma, k, n);
	cout << "cyclic_codes::make_cyclic_code Written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;
	}


	orbiter_kernel_system::latex_interface L;

	int k = n - degree;

	cout << "$$" << endl;
	cout << "\\left[" << endl;
	L.int_matrix_print_tex(cout, Genma, k, n);
	cout << "\\right]" << endl;
	cout << "$$" << endl;

	//cout << "before FREE_int(taken)" << endl;
	FREE_int(taken);
	//cout << "before FREE_int(transversal)" << endl;
	FREE_int(transversal);
	//cout << "before FREE_int(generator_subfield)" << endl;
	FREE_int(generator_subfield);
	//cout << "before FREE_int(Genma)" << endl;
	FREE_int(Genma);
	//cout << "before FREE_OBJECTS(generator)" << endl;
	for (i = 0; i <= degree; i++) {
		Fq.delete_object(generator[i]);
	}
	//FREE_OBJECTS(generator);
	cout << "before FREE_OBJECTS(tmp)" << endl;
	for (i = 0; i <= degree; i++) {
		Fq.delete_object(tmp[i]);
	}
	FREE_OBJECTS(tmp);
	//cout << "before FREE_OBJECTS(coeffs)" << endl;
	for (i = 0; i < 2; i++) {
		Fq.delete_object(coeffs[i]);
	}
	FREE_OBJECTS(coeffs);

}

void cyclic_codes::generator_matrix_cyclic_code(int n,
		int degree, int *generator_polynomial, int *&M)
{
	int k = n - degree;
	int i, j;

	M = NEW_int(k * n);
	Int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j <= degree; j++) {
			M[i * n + j + i] = generator_polynomial[j];
		}
	}
}

void cyclic_codes::print_polynomial(ring_theory::unipoly_domain &Fq,
		int degree, ring_theory::unipoly_object *coeffs)
{
	int i, f_first = TRUE;

	for (i = 0; i <= degree; i++) {
		if (Fq.is_zero(coeffs[i])) {
			continue;
		}
		if (!f_first) {
			cout << " + ";
		}
		f_first = FALSE;
		if (!Fq.is_one(coeffs[i])) {
			cout << "(";
			Fq.print_object(coeffs[i], cout);
			cout << ") * ";
		}
		cout << " Z^" << i;
	}
}

void cyclic_codes::print_polynomial_tight(std::ostream &ost,
		ring_theory::unipoly_domain &Fq,
		int degree, ring_theory::unipoly_object *coeffs)
{
	int i, f_first = TRUE;

	for (i = 0; i <= degree; i++) {
		if (!f_first) {
			ost << " + ";
		}
		f_first = FALSE;

		ost << "(";
		Fq.print_object_tight(coeffs[i], ost);
		ost << ") ";
		ost << " X^{" << i << "}";
	}
}


void cyclic_codes::field_reduction(int n, int q, int p, int e, int m,
		field_theory::finite_field &Fp,
		ring_theory::unipoly_domain &Fq,
	int degree, ring_theory::unipoly_object *generator, int *&generator_subfield,
	int f_poly, std::string &poly,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int r;
	ring_theory::longinteger_object Qm1, Index;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "cyclic_codes::field_reduction" << endl;
	}
	D.create_qnm1(Qm1, q, m);

	D.integral_division_by_int(Qm1, q - 1, Index, r);

	if (r != 0) {
		cout << "cyclic_codes::field_reduction q - 1 "
				"does not divide q^m - 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "considering the subfield GF(" << q
				<< ") of GF(" << q << "^" << m << ")" << endl;
		cout << "subgroup index = " << Index << endl;
	}

	ring_theory::unipoly_object c, beta, beta_i;
	ring_theory::longinteger_object *beta_rk_table, rk;
	int i, j;


	beta_rk_table = NEW_OBJECTS(ring_theory::longinteger_object, q);

	Fq.create_object_by_rank(c, 0, __FILE__, __LINE__, verbose_level);
	Fq.create_object_by_rank(beta, p, __FILE__, __LINE__, verbose_level); // the element alpha
	Fq.create_object_by_rank(beta_i, 1, __FILE__, __LINE__, verbose_level);
	if (f_v) {
		cout << "\\alpha = ";
		Fq.print_object(beta, cout);
		cout << endl;
	}
	Fq.power_longinteger(beta, Index, verbose_level - 1);
	if (f_v) {
		cout << "\\beta = \\alpha^" << Index << " = ";
		Fq.print_object(beta, cout);
		cout << endl;
	}
	for (i = 1; i <= q - 1; i++) {
		Fq.assign(beta, beta_i, verbose_level);
		Fq.power_int(beta_i, i, 0);
		Fq.rank_longinteger(beta_i, beta_rk_table[i]);
		if (f_v) {
			cout << i << " : ";
			Fq.print_object(beta_i, cout);
			cout << " : " << beta_rk_table[i] << endl;
		}
	}

	generator_subfield = NEW_int(degree + 1);

	for (i = 0; i <= degree; i++) {
		Fq.rank_longinteger(generator[i], rk);
		if (f_v) {
			cout << "coefficient " << i << " has rk " << rk << endl;
		}
		if (rk.is_zero()) {
			generator_subfield[i] = 0;
			continue;
		}
		for (j = 1; j <= q - 1; j++) {
			if (D.compare(rk, beta_rk_table[j]) == 0) {
				generator_subfield[i] = j;
				break;
			}
		}
		if (j == q) {
			cout << "error, coefficient "
					"does not lie in the subfield" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "over the subfield, exponential notation:" << endl;
		for (i = 0; i <= degree; i++) {
			cout << " " << generator_subfield[i] << "Z^" << i;
		}
		cout << endl;
	}
	if (f_v) {
		cout << "i : beta^i" << endl;
	}
	for (i = 0; i <= e; i++) {
		Fq.assign(beta, beta_i, verbose_level);
		Fq.power_int(beta_i, i, 0);
		if (f_v) {
			cout << i << " : ";
			Fq.print_object(beta_i, cout);
			cout << endl;
		}
	}

	if (!f_poly) {
		goto the_end;
	}

	{
		field_theory::finite_field fq;

		fq.init_override_polynomial(q, poly, FALSE /* f_without_tables */, verbose_level);
		cout << "q = " << q << " override polynomial = " << poly << endl;

		for (i = 0; i <= degree; i++) {
			j = generator_subfield[i];
			if (j == 0) {
				continue;
			}
			generator_subfield[i] = fq.alpha_power(j);
		}
		if (f_v) {
			cout << "over the subfield:" << endl;
			for (i = 0; i <= degree; i++) {
				j = generator_subfield[i];
				if (j == 0) {
					continue;
				}
				cout << " + " << j << " x^" << i;
			}
			cout << endl;
		}
	}

the_end:

	Fq.delete_object(c);
	Fq.delete_object(beta);
	Fq.delete_object(beta_i);
	FREE_OBJECTS(beta_rk_table);

	if (f_v) {
		cout << "cyclic_codes::field_reduction done" << endl;
	}

}

void cyclic_codes::BCH_generator_polynomial(
		field_theory::finite_field *F,
	ring_theory::unipoly_object &g, int n,
	int designed_distance, int &bose_distance,
	int &transversal_length, int *&transversal,
	ring_theory::longinteger_object *&rank_of_irreducibles,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int p = F->q;
	int e, i, j, r;
	ring_theory::longinteger_object q, b, m1, qm1;
	ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "cyclic_codes::BCH_generator_polynomial "
				"n=" << n << " designed_distance="
				<< designed_distance << " p=" << p << endl;
	}

	ring_theory::unipoly_domain FX(F);


	e = NT.order_mod_p(p, n);
	q.create(p, __FILE__, __LINE__);
	m1.create(-1, __FILE__, __LINE__);
	D.power_int(q, e);
	D.add(q, m1, qm1);
	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(qm1, n, b, r);
	//b = (q - 1) / n;
	if (r != 0) {
		cout << "cyclic_codes::BCH_generator_polynomial r != 0" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "GF(" << q << ") "
				"= GF(" << p << "^" << e << ") "
				"has " << n << "-th roots of unity" << endl;
		if (b.is_one()) {
			cout << "this is a primitive BCH code" << endl;
		}
		else {
			cout << "we take as " << n << "-th root \\beta = \\alpha^" << b << ", where "
				"\\alpha is a primitive element of the field" << endl;
		}
	}

	string field_poly;
	ring_theory::unipoly_object m, M, h1, h2;
	knowledge_base K;

	K.get_primitive_polynomial(field_poly, p, e, 0);
	FX.create_object_by_rank_string(m, field_poly, verbose_level - 2);
	FX.create_object_by_rank_string(M, field_poly, verbose_level - 2);

	FX.create_object_by_rank(g, 1, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(h1, 0, __FILE__, __LINE__, verbose_level);
	FX.create_object_by_rank(h2, 0, __FILE__, __LINE__, verbose_level);

	if (f_vv) {
		cout << "choosing the following irreducible "
				"and primitive polynomial:" << endl;
		FX.print_object(m, cout); cout << endl;
	}

	ring_theory::unipoly_domain Fq(F, M, verbose_level - 1);
	ring_theory::unipoly_object beta, beta_i, c;
	if (f_vvv) {
		cout << "extension field created" << endl;
	}
	Fq.create_object_by_rank(c, 0, __FILE__, __LINE__, verbose_level);
	Fq.create_object_by_rank(beta, p, __FILE__, __LINE__, verbose_level); // the primitive element alpha
	Fq.create_object_by_rank(beta_i, 1, __FILE__, __LINE__, verbose_level);
	if (!b.is_one()) {
		//Fq.power_int(beta, b, 0 /* verbose_level */);
		if (f_vvv) {
			cout << "\\alpha = ";
			Fq.print_object(beta, cout);
			cout << endl;
		}
		Fq.power_longinteger(beta, b, verbose_level - 1);
#if 0
		if (b.as_int() == 11) {
			for (i = 1; i <= b.as_int(); i++) {
				Fq.create_object_by_rank(beta, p); // the element alpha
				Fq.power_int(beta, i, 0 /* verbose_level */);
				cout << "\\alpha^" << i << " = ";
				Fq.print_object(beta, cout);
				cout << endl;
			}
		}
#endif
		if (f_vvv) {
			cout << "\\beta = \\alpha^" << b << " = ";
			Fq.print_object(beta, cout);
			cout << endl;
		}
	}
	else {
		if (f_vvv) {
			cout << "this is a primitive BCH code" << endl;
		}
	}

	// now beta is a primitive n-th root of unity

#if 0
	if (1 + designed_distance - 2 >= q - 1) {
		cout << "cyclic_codes::BCH_generator_polynomial "
				"1 + designed_distance - 2 >= q - 1" << endl;
		exit(1);
	}
#endif

	ring_theory::longinteger_object *beta_rk_table = NEW_OBJECTS(ring_theory::longinteger_object, n);
	ring_theory::longinteger_object ai, bi;


	for (i = 0; i < n; i++) {
		Fq.rank_longinteger(beta_i, beta_rk_table[i]);

		if (f_vvv) {
			cout << "\\beta^" << i << " = ";
			Fq.print_object(beta_i, cout);
			cout << " = " << beta_rk_table[i] << endl;
		}
		Fq.mult(beta, beta_i, c, verbose_level - 1);
		Fq.assign(c, beta_i, verbose_level);
	}
	if (f_vvv) {
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " = ";
			//Fq.print_object(beta_i, cout);
			cout << " = " << beta_rk_table[i] << endl;
		}
	}


	int *chosen = NEW_int(n);
	//int *transversal = NEW_int(n);
	//int transversal_length = 0, i0;
	int i0;

	transversal = NEW_int(n);
	transversal_length = 0;

	for (i = 0; i < n; i++) {
		chosen[i] = FALSE;
	}

	for (i = 1; i <= 1 + designed_distance - 2; i++) {
		Fq.mult(beta, beta_i, c, verbose_level - 1);
		Fq.assign(c, beta_i, verbose_level);

		Fq.rank_longinteger(beta_i, ai);
		if (f_vvv) {
			cout << "\\beta^" << i << " = ";
			Fq.print_object(beta_i, cout);
			cout << " = " << ai << endl;
		}
		if (chosen[i]) {
			continue;
		}

		transversal[transversal_length++] = i;
		if (f_v) {
			cout << "orbit of conjugate elements "
					"(in powers of \\beta):" << endl;
			cout << "{ ";
		}
		ai.assign_to(bi);
		i0 = i;
		do {
			chosen[i] = TRUE;
			Fq.create_object_by_rank_longinteger(c, bi, __FILE__, __LINE__, verbose_level);
			if (f_vvv) {
				cout << bi << " = ";
				Fq.print_object(c, cout);
			}
			else if (f_v) {
				cout << i << " ";
			}
			//power_coefficients(c, p);
			Fq.power_int(c, p, 0 /* verbose_level */);
			Fq.rank_longinteger(c, bi);
			for (j = 0; j < n; j++) {
				if (D.compare(bi, beta_rk_table[j]) == 0) {
					break;
				}
			}
			if (j == n) {
				cout << "couldn't find rank in the table (A)" << endl;
				exit(1);
			}
			if (f_vv) {
				cout << " is \\beta^" << j << endl;
			}
			i = j;
		} while (j != i0);
		if (f_vv || f_v) {
			cout << "}" << endl;
		}
	}

	// compute the bose_distance:
	Fq.create_object_by_rank(beta_i, 1, __FILE__, __LINE__, verbose_level);
	for (i = 1; ; i++) {
		Fq.mult(beta, beta_i, c, verbose_level - 1);
		FX.assign(c, beta_i, verbose_level);
		Fq.rank_longinteger(beta_i, ai);
		for (j = 0; j < n; j++) {
			if (D.compare(ai, beta_rk_table[j]) == 0) {
				break;
			}
		}
		if (j == n) {
			cout << "couldn't find rank in the table (B)" << endl;
			exit(1);
		}
		if (!chosen[j]) {
			break;
		}
	}
	bose_distance = i;

	ring_theory::longinteger_object rk;

	if (f_v) {
		cout << "taking the minimum polynomials of { ";
		for (i = 0; i < transversal_length; i++) {
			cout << transversal[i] << " ";
		}
		cout << "}" << endl;
	}

	rank_of_irreducibles = NEW_OBJECTS(
			ring_theory::longinteger_object, transversal_length);

	for (i = 0; i < transversal_length; i++) {

		// minimum_polynomial(h1, ai, p, f_vv);
		Fq.minimum_polynomial_factorring_longinteger(
				beta_rk_table[transversal[i]], rk, p, f_vv);
		FX.create_object_by_rank_longinteger(h1, rk, __FILE__, __LINE__, verbose_level - 2);
		if (f_vv) {
			cout << "minimal polynomial of \\beta^" << transversal[i] << " is ";
			FX.print_object(h1, cout);
			cout << " of rank " << rk << endl;
		}
		rk.assign_to(rank_of_irreducibles[i]);
		FX.mult(g, h1, h2, verbose_level - 1);
		FX.assign(h2, g, verbose_level);
	}

	Fq.delete_object(c);
	Fq.delete_object(beta);
	Fq.delete_object(beta_i);
	FX.delete_object(h1);
	FX.delete_object(h2);
	FX.delete_object(m);
	FREE_OBJECTS(beta_rk_table);
	FREE_int(chosen);
	//delete [] transversal;
	if (f_v) {
		cout << "BCH(" << n << "," << p << ","
				<< designed_distance << ") = ";
		FX.print_object(g, cout);
		cout << " bose_distance = " << bose_distance << endl;
	}
}

void cyclic_codes::compute_generator_matrix(
		ring_theory::unipoly_object a, int *&genma, int n, int &k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *r = (int *) a;
	int d = r[0];
	int *A = r + 1;

	int i, j, x;

	k = n - d;
	if (k < 0) {
		cout << "cyclic_codes::compute_generator_matrix k < 0" << endl;
		exit(1);
	}
	genma = NEW_int(k * n);
	for (i = 0; i < k * n; i++) {
		genma[i] = 0;
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j <= d; j++) {
			x = A[j];
			genma[i * n + i + j] = x;
		}
	}
	if (f_v) {
		cout << "cyclic_codes::compute_generator_matrix generator matrix:" << endl;
		Int_vec_print_integer_matrix(cout, genma, k, n);
	}
}


#if 0
void cyclic_codes::make_BCH_codes(int n, int q, int t, int b, int f_dual, int verbose_level)
// this function creates a finite field
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cyclic_codes::make_BCH_codes" << endl;
	}

	char fname[1000];
	std::string fname_txt;
	std::string fname_csv;
	number_theory::number_theory_domain NT;
	int *roots;
	int nb_roots;
	int i, j;

	roots = NEW_int(t - 1);
	nb_roots = t - 1;
	for (i = 0; i < t - 1; i++) {
		j = NT.mod(b + i, n);
		roots[i] = j;
	}
	snprintf(fname, 1000, "BCH_%d_%d", n, t);

	fname_txt.assign(fname);
	fname_txt.append(".txt");
	fname_csv.assign(fname);
	fname_csv.append(".csv");

	cout << "roots: ";
	Int_vec_print(cout, roots, nb_roots);
	cout << endl;

	string dummy;

	dummy.assign("");

	if (f_v) {
		cout << "cyclic_codes::make_BCH_codes before make_cyclic_code" << endl;
	}

	// this function creates a finite field:
	make_cyclic_code(n, q, t, roots, nb_roots,
			FALSE /*f_poly*/, dummy /*poly*/, f_dual,
			fname_txt, fname_csv, verbose_level);

	if (f_v) {
		cout << "cyclic_codes::make_BCH_codes after make_cyclic_code" << endl;
	}

	FREE_int(roots);

	if (f_v) {
		cout << "cyclic_codes::make_BCH_codes done" << endl;
	}
}
#endif

void cyclic_codes::generator_matrix_cyclic_code(
		field_theory::finite_field *F,
		int n,
		std::string &poly_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cyclic_codes::generator_matrix_cyclic_code" << endl;
	}

	int *coeffs;
	int sz;
	int k;
	int d;
	int *M;
	int *v;
	int i, j;
	long int rk;
	long int *Rk;

	Int_vec_scan(poly_coeffs, coeffs, sz);
	d = sz - 1;
	k = n - d;

	M = NEW_int(k * n);
	Int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		Int_vec_copy(coeffs, M + i * n + i, d + 1);
	}

	cout << "generator matrix:" << endl;
	Int_matrix_print(M, k, n);


	cout << "generator matrix:" << endl;
	Int_vec_print_fully(cout, M, k * n);
	cout << endl;

	v = NEW_int(k);
	Rk = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < k; i++) {
			v[i] = M[i * n + j];
		}
		F->PG_element_rank_modified_lint(
				v, 1, k, rk);
		Rk[j] = rk;
	}

	cout << "generator matrix in projective points:" << endl;
	Lint_vec_print_fully(cout, Rk, n);
	cout << endl;

	if (f_v) {
		cout << "cyclic_codes::generator_matrix_cyclic_code" << endl;
	}
}



}}}



