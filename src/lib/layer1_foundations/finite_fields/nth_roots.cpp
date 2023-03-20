/*
 * nth_roots.cpp
 *
 *  Created on: Oct 2, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {




nth_roots::nth_roots()
{
	n = 0;
	F = NULL;
	Beta = NULL;
	Fq_Elements = NULL;
	Min_poly = NULL;
	Fp = NULL;
	FpX = NULL;
	FQ = NULL;
	FX = NULL;
	m = 0;
	r = 0;
	field_degree = 0;
	Qm = NULL;
	Qm1 = NULL;
	Index = NULL;
	Subfield_Index = NULL;
	Cyc = NULL;
	min_poly_beta_FQ = NULL;
	min_poly_beta_Fq = NULL;
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


void nth_roots::init(
		finite_field *F, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nth_roots::init n=" << n << endl;
	}
	ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;
	coding_theory::cyclic_codes Cyclic_codes;
	int i;


	nth_roots::n = n;
	nth_roots::F = F;


	m = NT.order_mod_p(F->q, n);
	if (f_v) {
		cout << "nth_roots::init order of q mod n is m=" << m << endl;
	}
	Qm = NEW_OBJECT(ring_theory::longinteger_object);
	Qm1 = NEW_OBJECT(ring_theory::longinteger_object);
	Index = NEW_OBJECT(ring_theory::longinteger_object);
	Subfield_Index = NEW_OBJECT(ring_theory::longinteger_object);
	D.create_q_to_the_n(*Qm, F->q, m);
	D.create_qnm1(*Qm1, F->q, m);

	field_degree = F->e * m;

	// q = i_power_j(p, e);
	// GF(q)=GF(p^e) has n-th roots of unity
	D.integral_division_by_int(*Qm1, n, *Index, r);
	if (f_v) {
		cout << "nth_roots::init n = " << n << endl;
		cout << "nth_roots::init m = " << m << endl;
		cout << "nth_roots::init field_degree = " << field_degree << endl;
		cout << "nth_roots::init Qm1 = " << *Qm1 << endl;
	}

	if (r) {
		cout << "nth_roots::init n does not divide Qm1" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "nth_roots::init Index = " << *Index << endl;
	}

	D.integral_division_by_int(*Qm1, F->q - 1, *Subfield_Index, r);
	if (r) {
		cout << "nth_roots::init q - 1 does not divide Qm1" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "nth_roots::init Subfield_Index = " << *Subfield_Index << endl;
	}


	Fp = NEW_OBJECT(finite_field);

	Fp->finite_field_init_small_order(F->p,
			FALSE /* f_without_tables */,
			FALSE /* f_compute_related_fields */,
			verbose_level - 1);

	FX = NEW_OBJECT(ring_theory::unipoly_domain);

	FX->init_basic(F, verbose_level);


	//unipoly_domain FpX(Fp);

	FpX = NEW_OBJECT(ring_theory::unipoly_domain);

	FpX->init_basic(Fp, verbose_level);

	knowledge_base::knowledge_base K;
	string field_poly;

	K.get_primitive_polynomial(field_poly,
			F->p, field_degree, 0);

	FpX->create_object_by_rank_string(Min_poly,
			field_poly,
			verbose_level - 2);


	if (f_v) {
		cout << "nth_roots::init creating unipoly_domain FQ modulo M" << endl;
	}

	FQ = NEW_OBJECT(ring_theory::unipoly_domain);

	FQ->init_factorring(Fp, Min_poly, verbose_level);
	//unipoly_domain Fq(Fp, Min_poly, verbose_level);
		// Fq = Fp[X] modulo factor polynomial M

	if (f_v) {
		cout << "nth_roots::init extension field created" << endl;
	}

	ring_theory::ring_theory_global R;

	if (f_v) {
		cout << "nth_roots::init before R.compute_nth_roots_as_polynomials" << endl;
	}
	R.compute_nth_roots_as_polynomials(F, FpX, FQ, Beta, n, n, 0 /*verbose_level*/);
	if (f_v) {
		cout << "nth_roots::init after R.compute_nth_roots_as_polynomials" << endl;
	}






	if (f_v) {
		cout << "nth_roots::init the n-th roots are:" << endl;
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " &=& ";
			FQ->print_object_tight(Beta[i], cout);
			cout << " = ";
			FQ->print_object(Beta[i], cout);
			cout << "\\\\" << endl;
		}
	}


	R.compute_nth_roots_as_polynomials(F, FpX,
			FQ, Fq_Elements, n, F->q - 1,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "nth_roots::init the (q-1)-th roots are:" << endl;
		for (i = 0; i < F->q - 1; i++) {
			cout << "\\gamma^" << i << " &=& ";
			FQ->print_object_tight(Fq_Elements[i], cout);
			cout << " = ";
			FQ->print_object(Fq_Elements[i], cout);
			cout << "\\\\" << endl;
		}
	}


	Cyc = NEW_OBJECT(number_theory::cyclotomic_sets);

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



	min_poly_beta_FQ = (ring_theory::unipoly_object **) NEW_pvoid(Cyc->S->nb_sets);

	for (i = 0; i < Cyc->S->nb_sets; i++) {

		if (f_v) {
			cout << "nth_roots::init creating polynomial "
					<< i << " / " << Cyc->S->nb_sets << endl;
		}

		R.create_irreducible_polynomial(F,
				FQ,
			Beta, n,
			Cyc->S->Sets[i], Cyc->S->Set_size[i],
			min_poly_beta_FQ[i],
			0 /*verbose_level*/);

		//Codes.print_polynomial(Fq, Cyc->S->Set_size[i], generator[i]);
		//cout << endl;

	}

	if (f_v) {
		cout << "nth_roots::init irreducible polynomials" << endl;

		for (i = 0; i < Cyc->S->nb_sets; i++) {
			cout << i << " : ";

			Lint_vec_print(cout, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

			cout << " : ";

			Cyclic_codes.print_polynomial_tight(cout, *FQ, Cyc->S->Set_size[i], min_poly_beta_FQ[i]);


			cout << endl;
		}
	}

	if (f_v) {
		cout << "nth_roots::init Index" << endl;
		Int_vec_print(cout, Cyc->Index, n);
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
		Int_vec_print_integer_matrix_width(cout, subfield_basis,
				subfield_degree, field_degree, field_degree, Fp->log10_of_q);
	}

	int *input_vector;
	int *coefficients;
	int j, h, a;
	geometry::geometry_global GG;

	input_vector = NEW_int(field_degree);
	coefficients = NEW_int(subfield_degree);

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		cout << i << " : ";

		Lint_vec_print(cout, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

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
				input_vector[h] = FpX->s_i(min_poly_beta_FQ[i][j], h);
			}

			Fp->Linear_algebra->get_coefficients_in_linear_combination(
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


	min_poly_beta_Fq = (ring_theory::unipoly_object *) NEW_pvoid(Cyc->S->nb_sets);

	for (i = 0; i < Cyc->S->nb_sets; i++) {

		int degree = Cyc->S->Set_size[i];
		int *coeffs;

		coeffs = NEW_int(degree + 1);

		for (j = 0; j <= degree; j++) {

			for (h = 0; h < field_degree; h++) {
				input_vector[h] = FpX->s_i(min_poly_beta_FQ[i][j], h);
			}

			Fp->Linear_algebra->get_coefficients_in_linear_combination(
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
				min_poly_beta_Fq[i], degree, coeffs);
	}

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		cout << i << " : ";
		FX->print_object(min_poly_beta_Fq[i], cout);
		cout << endl;
	}

	if (f_v) {
		cout << "nth_roots::init done" << endl;
	}
}

void nth_roots::compute_subfield(
		int subfield_degree, int *&field_basis, int verbose_level)
// field_basis[subfield_degree * field_degree]
// assumes that subfield_degree divides the field_degree
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nth_roots::compute_subfield subfield_degree=" << subfield_degree << endl;
	}
	int *M; // [e * (subfield_degree + 1)]
	int *K; // [e]
	int *base_cols; // [e]
	int rk, kernel_m, kernel_n;
	long int a;
	int p, e, q1, subgroup_index;
	int i, j;
	geometry::geometry_global Gg;
	number_theory::number_theory_domain NT;
	ring_theory::ring_theory_global R;

	p = F->p;
	e = field_degree;
	if (f_v) {
		cout << "nth_roots::compute_subfield p=" << p << endl;
		cout << "nth_roots::compute_subfield e=" << e << endl;
		cout << "nth_roots::compute_subfield subfield_degree=" << subfield_degree << endl;
	}
	M = NEW_int(e * (subfield_degree + 1));
	// The j-th column of M is the coefficients of beta[j] (a vector of length e)
	// j=0,..,subfield_degree
	// The columns are linearly dependent.
	// A linear dependency between the column vectors will be used to create
	// the irreducible polynomial defining the field.

	Int_vec_zero(M, e * (subfield_degree + 1));

	K = NEW_int(e);
	base_cols = NEW_int(e);
	q1 = NT.i_power_j(p, subfield_degree);
	subgroup_index = (Cyc->qm - 1) / (q1 - 1);
	if (f_v) {
		cout << "nth_roots::compute_subfield "
				"subfield " << p << "^" << subfield_degree << " : subgroup_index = "
			<< subgroup_index << endl;
	}

	ring_theory::unipoly_object *Beta;

	if (f_v) {
		cout << "nth_roots::compute_subfield before F->compute_powers" << endl;
	}
	R.compute_powers(F, FQ,
			n, subgroup_index,
			Beta, 0/*verbose_level*/);


	if (f_v) {
		for (i = 0; i < n; i++) {
			cout << "\\beta^" << i << " = ";
			FQ->print_object(Beta[i], cout);
			cout << endl;
		}
	}

	for (j = 0; j <= subfield_degree; j++) {
		for (i = 0; i < e; i++) {
			a = FQ->s_i(Beta[j], i);
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
	// field_basis is like M transposed, with the last row removed.
	// the rows of field_basis are the beta[j], j=0,...,subfield_degree - 1.

	for (j = 0; j < subfield_degree; j++) {
		for (i = 0; i < e; i++) {
			a = M[i * (subfield_degree + 1) + j];
			field_basis[j * e + i] = a;
		}
	}
	if (f_v) {
		cout << "nth_roots::compute_subfield field_basis=" << endl;
		Int_vec_print_integer_matrix_width(cout, field_basis,
				subfield_degree, e, e, Fp->log10_of_q);
	}



#if 1
	if (f_v) {
		cout << "nth_roots::compute_subfield M=" << endl;
		Int_vec_print_integer_matrix_width(cout, M,
			e, subfield_degree + 1, subfield_degree + 1, Fp->log10_of_q);
	}
	if (f_v) {
		cout << "nth_roots::compute_subfield before Fp->Linear_algebra->Gauss_simple" << endl;
	}
	rk = Fp->Linear_algebra->Gauss_simple(
			M,
			e,
			subfield_degree + 1,
			base_cols,
			verbose_level);
	if (f_v) {
		cout << "nth_roots::compute_subfield after Gauss=" << endl;
		Int_vec_print_integer_matrix_width(cout, M,
			e, subfield_degree + 1, subfield_degree + 1, Fp->log10_of_q);
		cout << "rk=" << rk << endl;
	}
	if (rk != subfield_degree) {
		cout << "nth_roots::compute_subfield fatal: rk != subfield_degree" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}

	if (f_v) {
		cout << "nth_roots::compute_subfield before Fp->Linear_algebra->matrix_get_kernel" << endl;
	}
	Fp->Linear_algebra->matrix_get_kernel(M,
			e, subfield_degree + 1, base_cols, rk,
			kernel_m, kernel_n, K,
			0 /* verbose_level */);

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
		Int_vec_print(cout, K, subfield_degree + 1);
		cout << endl;
	}

	a = Gg.AG_element_rank(p, K, 1, subfield_degree + 1);

	if (f_v) {
		ring_theory::unipoly_object elt;

		FpX->create_object_by_rank(elt, a, __FILE__, __LINE__, verbose_level);
		cout << "nth_roots::compute_subfield "
				"subfield of order "
				<< NT.i_power_j(p, subfield_degree)
				<< " : " << a << " = ";
		FQ->print_object(elt, cout);
		cout << endl;
		FQ->delete_object(elt);
	}

	FREE_int(M);
	FREE_int(K);
	FREE_int(base_cols);
#endif



	if (f_v) {
		cout << "nth_roots::compute_subfield done" << endl;
	}
}


void nth_roots::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	string label;
	coding_theory::cyclic_codes Cyclic_codes;
	orbiter_kernel_system::latex_interface Li;

	if (f_v) {
		cout << "nth_roots::report" << endl;
	}

	label.assign("\\alpha");

	FQ->init_variable_name(label);



	ost << "\\noindent Let $\\alpha$ be a primitive element of GF$(" << *Qm << ")$. " << endl;
	ost << "Let $\\beta$ be a primitive $" << n << "$-th root in GF$(" << *Qm << ")$, " << endl;
	ost << "so $\\beta=\\alpha^{" << *Index << "}.$ \\\\" << endl;
	//ost << "\\begin{align*}" << endl;
	for (i = 0; i < n; i++) {
		ost << "$\\beta^{" << i << "} = ";
		FQ->print_object_tight(Beta[i], ost);
		ost << " =  ";
		FQ->print_object(Beta[i], ost);
		ost << "$\\\\" << endl;
	}
	//ost << "\\end{align*}" << endl;
	//ost << "\\clearpage" << endl;
	ost << "\\bigskip" << endl << endl;


	//cout << "nth_roots::init the (q-1)-th roots are:" << endl;
	//ost << "\\begin{align*}" << endl;
	ost << "\\noindent Let $\\gamma$ be a primitive $" << F->q - 1 << "$-th root in GF$(" << *Qm << ")$, " << endl;
	ost << "so $\\gamma=\\alpha^{" << *Subfield_Index << "}.$ \\\\" << endl;
	for (i = 0; i < F->q - 1; i++) {
		ost << "$\\gamma^{" << i << "} = ";
		FQ->print_object_tight(Fq_Elements[i], ost);
		ost << " = ";
		FQ->print_object(Fq_Elements[i], ost);
		ost << "$\\\\" << endl;
	}
	//ost << "\\end{align*}" << endl;
	//ost << "\\clearpage" << endl;
	ost << "\\bigskip" << endl << endl;

	ost << "\\noindent The $q$-cyclotomic set for $q=" << F->q << "$ are:" << endl << endl;


	Cyc->print_latex(ost);

	ost << "\\bigskip" << endl << endl;

	ost << "Subfield basis, a basis for GF$(" << F->q << ")$ inside GF$(" << *Qm << ")$:" << endl;

	ost << "$$" << endl;
	ost << "\\left[" << endl;
	Li.int_matrix_print_tex(ost, subfield_basis, subfield_degree, field_degree);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	ost << "\\bigskip" << endl << endl;

	ost << "The irreducible polynomials associated with the $" << n << "$-th roots over GF$(" << F->q << ")$ are:" << endl << endl;

	ost << "\\clearpage" << endl << endl;

	//ost << "\\begin{figure}" << endl;
	//ost << "\\begin{sideways}" << endl;
	//ost << "\\begin{minipage}{\\textwidth}" << endl;
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "\\tiny" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|l|p{5cm}|l|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & r_i & {\\rm Cyc}(r_i) & $m_{\\beta^{r_i}}(X)$ & m_{\\beta^{r_i}}(X) & \\mbox{rank} \\\\" << endl;
	ost << "\\hline" << endl;

	int rk;

	for (i = 0; i < Cyc->S->nb_sets; i++) {
		ost << i << " & ";

		ost << Cyc->S->Sets[i][0];

		ost << " & ";

		Lint_vec_print(ost, Cyc->S->Sets[i], Cyc->S->Set_size[i]);

		ost << " & ";

		ost << "$";
		Cyclic_codes.print_polynomial_tight(ost, *FQ, Cyc->S->Set_size[i], min_poly_beta_FQ[i]);
		ost << "$";

		ost << " & ";

		FX->print_object(min_poly_beta_Fq[i], ost);

		rk = FX->rank(min_poly_beta_Fq[i]);

		ost << " & ";

		ost << rk;


		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	//ost << "\\end{minipage}" << endl;
	//ost << "\\end{sideways}" << endl;
	//ost << "\\end{figure}" << endl;

	ost << "\\clearpage" << endl;


}



}}}

