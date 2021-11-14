/*
 * nth_roots.cpp
 *
 *  Created on: Oct 2, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {




nth_roots::nth_roots()
{
	n = 0;
	F = NULL;
	Beta = NULL;
	Fq_Elements = NULL;
	M = NULL;
	Fp = NULL;
	FpX = NULL;
	Fq = NULL;
	FX = NULL;
	m = 0;
	r = 0;
	field_degree = 0;
	Qm1 = NULL;
	Index = NULL;
	Subfield_Index = NULL;
	Cyc = NULL;
	generator = NULL;
	generator_Fq = NULL;
	subfield_degree = 0;
	subfield_basis = NULL;
}

nth_roots::~nth_roots()
{
	if (Qm1) {
		FREE_OBJECT(Qm1);
	}
	if (Index) {
		FREE_OBJECT(Index);
	}
	if (Subfield_Index) {
		FREE_OBJECT(Subfield_Index);
	}
}


void nth_roots::init(finite_field *F, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nth_roots::init n=" << n << endl;
	}
	longinteger_domain D;
	number_theory_domain NT;
	algebra_global Algebra;
	coding_theory_domain Codes;
	int i;


	nth_roots::n = n;
	nth_roots::F = F;


	m = NT.order_mod_p(F->q, n);
	if (f_v) {
		cout << "nth_roots::init order of q mod n is m=" << m << endl;
	}
	Qm1 = NEW_OBJECT(longinteger_object);
	Index = NEW_OBJECT(longinteger_object);
	Subfield_Index = NEW_OBJECT(longinteger_object);
	D.create_qnm1(*Qm1, F->q, m);

	field_degree = F->e * m;

	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(*Qm1, n, *Index, r);
	if (f_v) {
		cout << "nth_roots::init Index = " << Index << endl;
	}
	D.integral_division_by_int(*Qm1, F->q - 1, *Subfield_Index, r);

	Fp = NEW_OBJECT(finite_field);

	Fp->finite_field_init(F->p, FALSE /* f_without_tables */, verbose_level - 1);

	FX = NEW_OBJECT(unipoly_domain);

	FX->init_basic(F, verbose_level);


	//unipoly_domain FpX(Fp);

	FpX = NEW_OBJECT(unipoly_domain);

	FpX->init_basic(Fp, verbose_level);

	FpX->create_object_by_rank_string(M,
			Algebra.get_primitive_polynomial(F->p, field_degree, 0),
			verbose_level - 2);


	if (f_v) {
		cout << "nth_roots::init creating unipoly_domain Fq modulo M" << endl;
	}

	Fq = NEW_OBJECT(unipoly_domain);

	Fq->init_factorring(Fp, M, verbose_level);
	//unipoly_domain Fq(Fp, M, verbose_level);
		// Fq = Fp[X] modulo factor polynomial M
	if (f_v) {
		cout << "nth_roots::init extension field created" << endl;
	}


	F->compute_nth_roots_as_polynomials(FpX, Fq, Beta, n, n, 0 /*verbose_level*/);






	if (f_v) {
		cout << "nth_roots::init the n-th roots are:" << endl;
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " &=& ";
			Fq->print_object_tight(Beta[i], cout);
			cout << " = ";
			Fq->print_object(Beta[i], cout);
			cout << "\\\\" << endl;
		}
	}


	F->compute_nth_roots_as_polynomials(FpX, Fq, Fq_Elements, n, F->q - 1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "nth_roots::init the (q-1)-th roots are:" << endl;
		for (i = 0; i < F->q - 1; i++) {
			cout << "\\gamma^" << i << " &=& ";
			Fq->print_object_tight(Fq_Elements[i], cout);
			cout << " = ";
			Fq->print_object(Fq_Elements[i], cout);
			cout << "\\\\" << endl;
		}
	}


	Cyc = NEW_OBJECT(cyclotomic_sets);

	if (f_v) {
		cout << "nth_roots::init before Cyc->init" << endl;
	}
	Cyc->init(F, n, 0 /*verbose_level*/);
	if (f_v) {
		cout << "nth_roots::init after Cyc->init" << endl;
	}

	if (f_v) {
		cout << "nth_roots::init q-cyclotomic sets mod n:" << endl;
		Cyc->print();
	}



	generator = (unipoly_object **) NEW_pvoid(Cyc->S->nb_sets);

	for (i = 0; i < Cyc->S->nb_sets; i++) {

		if (f_v) {
			cout << "nth_roots::init creating polynomial "
					<< i << " / " << Cyc->S->nb_sets << endl;
		}

		F->create_irreducible_polynomial(Fq,
			Beta, n,
			Cyc->S->Sets[i], Cyc->S->Set_size[i],
			generator[i],
			0 /*verbose_level*/);

		//Codes.print_polynomial(Fq, Cyc->S->Set_size[i], generator[i]);
		//cout << endl;

	}

	if (f_v) {
		cout << "nth_roots::init irreducible polynomials" << endl;

		for (i = 0; i < Cyc->S->nb_sets; i++) {
			cout << i << " : ";

			Orbiter->Lint_vec.print(cout, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

			cout << " : ";

			Codes.print_polynomial_tight(cout, *Fq, Cyc->S->Set_size[i], generator[i]);


			cout << endl;
		}
	}

	if (f_v) {
		cout << "nth_roots::init Index" << endl;
		Orbiter->Int_vec.print(cout, Cyc->Index, n);
		cout << endl;
	}


	if (f_v) {
		cout << "nth_roots::init before compute_subfield" << endl;
	}

	subfield_degree = F->e;

	compute_subfield(subfield_degree, subfield_basis, verbose_level);

	if (f_v) {
		cout << "nth_roots::init after compute_subfield" << endl;
	}


	if (f_v) {
		cout << "nth_roots::init subfield_basis=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, subfield_basis,
				subfield_degree, field_degree, field_degree, Fp->log10_of_q);
	}

	int *input_vector;
	int *coefficients;
	int j, h, a;
	geometry_global GG;

	input_vector = NEW_int(field_degree);
	coefficients = NEW_int(subfield_degree);

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		cout << i << " : ";

		Orbiter->Lint_vec.print(cout, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

		cout << " : ";

		//Codes.print_polynomial_tight(*Fq, Cyc->S->Set_size[i], generator[i]);

		int f_first = TRUE;
		int degree = Cyc->S->Set_size[i];

		for (j = 0; j <= degree; j++) {
			if (!f_first) {
				cout << " + ";
			}
			f_first = FALSE;

			for (h = 0; h < field_degree; h++) {
				input_vector[h] = FpX->s_i(generator[i][j], h);
			}

			Fp->get_coefficients_in_linear_combination(
				subfield_degree, field_degree, subfield_basis,
				input_vector, coefficients, 0 /*verbose_level*/);

			//Orbiter->Int_vec.print(cout, input_vector, field_degree);
			//cout << "=";
			//Orbiter->Int_vec.print(cout, coefficients, subfield_degree);
			//cout << "=";

			a = GG.AG_element_rank(F->p, coefficients, 1, subfield_degree);

			cout << a << " * Z^" << j;
		}

		cout << endl;
	}


	generator_Fq = (unipoly_object *) NEW_pvoid(Cyc->S->nb_sets);

	for (i = 0; i < Cyc->S->nb_sets; i++) {

		int degree = Cyc->S->Set_size[i];
		int *coeffs;

		coeffs = NEW_int(degree + 1);

		for (j = 0; j <= degree; j++) {

			for (h = 0; h < field_degree; h++) {
				input_vector[h] = FpX->s_i(generator[i][j], h);
			}

			Fp->get_coefficients_in_linear_combination(
				subfield_degree, field_degree, subfield_basis,
				input_vector, coefficients, 0 /*verbose_level*/);

			//Orbiter->Int_vec.print(cout, input_vector, field_degree);
			//cout << "=";
			//Orbiter->Int_vec.print(cout, coefficients, subfield_degree);
			//cout << "=";

			a = GG.AG_element_rank(F->p, coefficients, 1, subfield_degree);
			coeffs[j] = a;
		}

		FX->create_object_of_degree_with_coefficients(
				generator_Fq[i], degree, coeffs);
	}

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		cout << i << " : ";
		FX->print_object(generator_Fq[i], cout);
		cout << endl;
	}

	if (f_v) {
		cout << "nth_roots::init done" << endl;
	}
}

void nth_roots::compute_subfield(int subfield_degree, int *&field_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nth_roots::compute_subfield subfield_degree=" << subfield_degree << endl;
	}
	int *M;
	int *K;
	int *base_cols;
	int rk, kernel_m, kernel_n;
	long int a;
	int p, e, q1, subgroup_index;
	int i, j;
	geometry_global Gg;
	number_theory_domain NT;

	p = F->p;
	e = field_degree;
	M = NEW_int(e * (subfield_degree + 1));
	Orbiter->Int_vec.zero(M, e * (subfield_degree + 1));

	K = NEW_int(e);
	base_cols = NEW_int(e);
	q1 = NT.i_power_j(p, subfield_degree);
	subgroup_index = (Cyc->qm - 1) / (q1 - 1);
	if (f_v) {
		cout << "nth_roots::compute_subfield "
				"subfield " << p << "^" << subfield_degree << " : subgroup_index = "
			<< subgroup_index << endl;
	}

	unipoly_object *Beta;

	if (f_v) {
		cout << "nth_roots::compute_subfield before F->compute_powers" << endl;
	}
	F->compute_powers(Fq,
			n, subgroup_index,
			Beta, 0/*verbose_level*/);


	if (f_v) {
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " = ";
			Fq->print_object(Beta[i], cout);
			cout << endl;
		}
	}

	for (j = 0; j <= subfield_degree; j++) {
		for (i = 0; i < e; i++) {
			a = Fq->s_i(Beta[j], i);
			M[i * (subfield_degree + 1) + j] = a;
		}
#if 0
		j = i * subgroup_index;
		jj = alpha_power(j);
		Gg.AG_element_unrank(p, M + i, subfield_degree + 1, e, jj);

		{
			unipoly_object elt;

			Fq.create_object_by_rank(elt, jj, __FILE__, __LINE__, 0 /*verbose_level*/);
			if (f_v) {
				cout << i << " : " << j << " : " << jj << " : ";
				Fq.print_object(elt, cout);
				cout << endl;
			}
			Fq.delete_object(elt);
		}
#endif

	}

	field_basis = NEW_int(subfield_degree * e);

	for (j = 0; j < subfield_degree; j++) {
		for (i = 0; i < e; i++) {
			a = M[i * (subfield_degree + 1) + j];
			field_basis[j * e + i] = a;
		}
	}
	if (f_v) {
		cout << "nth_roots::compute_subfield field_basis=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, field_basis,
				subfield_degree, e, e, Fp->log10_of_q);
	}



#if 1
	if (f_v) {
		cout << "nth_roots::compute_subfield M=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M,
			e, subfield_degree + 1, subfield_degree + 1, Fp->log10_of_q);
	}
	rk = Fp->Gauss_simple(M, e, subfield_degree + 1, base_cols, 0/*verbose_level*/);
	if (f_v) {
		cout << "nth_roots::compute_subfield after Gauss=" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, M,
			e, subfield_degree + 1, subfield_degree + 1, Fp->log10_of_q);
		cout << "rk=" << rk << endl;
	}
	if (rk != subfield_degree) {
		cout << "nth_roots::compute_subfield fatal: rk != subfield_degree" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}

	Fp->matrix_get_kernel(M, e, subfield_degree + 1, base_cols, rk,
		kernel_m, kernel_n, K, 0 /* verbose_level */);

	if (f_v) {
		cout << "nth_roots::compute_subfield kernel_m=" << kernel_m << endl;
		cout << "nth_roots::compute_subfield kernel_n=" << kernel_n << endl;
	}
	if (kernel_n != 1) {
		cout << "nth_roots::compute_subfield kernel_n != 1" << endl;
		exit(1);
	}
	if (K[subfield_degree] == 0) {
		cout << "nth_roots::compute_subfield K[e1] == 0" << endl;
		exit(1);
	}
	if (K[subfield_degree] != 1) {
		a = Fp->inverse(K[subfield_degree]);
		for (i = 0; i < subfield_degree + 1; i++) {
			K[i] = Fp->mult(a, K[i]);
		}
	}

	if (f_v) {
		cout << "nth_roots::compute_subfield the relation is " << endl;
		Orbiter->Int_vec.print(cout, K, subfield_degree + 1);
		cout << endl;
	}

	a = Gg.AG_element_rank(p, K, 1, subfield_degree + 1);

	if (f_v) {
		unipoly_object elt;

		FpX->create_object_by_rank(elt, a, __FILE__, __LINE__, verbose_level);
		cout << "nth_roots::compute_subfield "
				"subfield of order "
				<< NT.i_power_j(p, subfield_degree)
				<< " : " << a << " = ";
		Fq->print_object(elt, cout);
		cout << endl;
		Fq->delete_object(elt);
	}

	FREE_int(M);
	FREE_int(K);
	FREE_int(base_cols);
#endif



	if (f_v) {
		cout << "nth_roots::compute_subfield done" << endl;
	}
}


void nth_roots::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string label;
	coding_theory_domain Codes;

	if (f_v) {
		cout << "nth_roots::report" << endl;
	}

	label.assign("\\alpha");

	Fq->init_variable_name(label);

	ost << "\\begin{align*}" << endl;
	for (i = 0; i < n; i++) {
		ost << "\\beta^{" << i << "} &= ";
		Fq->print_object_tight(Beta[i], ost);
		ost << " =  ";
		Fq->print_object(Beta[i], ost);
		ost << "\\\\" << endl;
	}
	ost << "\\end{align*}" << endl;
	ost << "\\clearpage" << endl;


	//cout << "nth_roots::init the (q-1)-th roots are:" << endl;
	ost << "\\begin{align*}" << endl;
	for (i = 0; i < F->q - 1; i++) {
		ost << "\\gamma^{" << i << "} &= ";
		Fq->print_object_tight(Fq_Elements[i], ost);
		ost << " = ";
		Fq->print_object(Fq_Elements[i], ost);
		ost << "\\\\" << endl;
	}
	ost << "\\end{align*}" << endl;
	ost << "\\clearpage" << endl;

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|l|l|l|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & r_i & {\\rm Cyc}(r_i) & m_{\\beta^{r_i}}(X) & m_{\\beta^{r_i}}(X) \\\\" << endl;
	ost << "\\hline" << endl;

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		ost << i << " & ";

		ost << Cyc->S->Sets[i][0];

		ost << " & ";

		Orbiter->Lint_vec.print(ost, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

		ost << " & ";

		Codes.print_polynomial_tight(ost, *Fq, Cyc->S->Set_size[i], generator[i]);

		ost << " & ";

		FX->print_object(generator_Fq[i], ost);


		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	ost << "\\clearpage" << endl;


}



}}
