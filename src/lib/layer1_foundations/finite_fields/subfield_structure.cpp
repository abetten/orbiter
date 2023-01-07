// subfield_structure.cpp
//
// Anton Betten
//
// started:  November 14, 2011




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace field_theory {


subfield_structure::subfield_structure()
{
	FQ = NULL;
	Fq = NULL;
	Q = q = s = 0;
	Basis = NULL;
	embedding = NULL;
	embedding_inv = NULL;
	components = NULL;
	FQ_embedding = NULL;
	Fq_element = NULL;
	v = NULL;
	//null();
}



subfield_structure::~subfield_structure()
{
	if (Basis) {
		FREE_int(Basis);
	}
	if (embedding) {
		FREE_int(embedding);
	}
	if (embedding_inv) {
		FREE_int(embedding_inv);
	}
	if (components) {
		FREE_int(components);
	}
	if (FQ_embedding) {
		FREE_int(FQ_embedding);
	}
	if (Fq_element) {
		FREE_int(Fq_element);
	}
	if (v) {
		FREE_int(v);
	}
}

void subfield_structure::init(finite_field *FQ,
		finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, omega, i;
	int *my_basis;

	if (f_v) {
		cout << "subfield_structure::init" << endl;
	}
	subfield_structure::FQ = FQ;
	subfield_structure::Fq = Fq;
	Q = FQ->q;
	q = Fq->q;
	if (FQ->p != Fq->p) {
		cout << "subfield_structure::init "
				"different characteristics" << endl;
		exit(1);
	}
	s = FQ->e / Fq->e;
	if (Fq->e * s != FQ->e) {
		cout << "Fq is not a subfield of FQ" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "index = " << s << endl;
	}


	my_basis = NEW_int(s);
	alpha = FQ->p; // the primitive element
	omega = FQ->power(alpha, s);
	for (i = 0; i < s; i++) {
		my_basis[i] = FQ->power(omega, i);
	}
	init_with_given_basis(FQ, Fq, my_basis, verbose_level);

	FREE_int(my_basis);

	if (f_v) {
		cout << "subfield_structure::init done" << endl;
	}
}

void subfield_structure::init_with_given_basis(
		finite_field *FQ, finite_field *Fq, int *given_basis,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int /*alpha,*/ /*omega,*/ i, j, h;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "subfield_structure::init_with_given_basis" << endl;
	}
	subfield_structure::FQ = FQ;
	subfield_structure::Fq = Fq;
	Q = FQ->q;
	q = Fq->q;
	if (FQ->p != Fq->p) {
		cout << "subfield_structure::init_with_given_basis "
				"different characteristics" << endl;
		exit(1);
	}
	s = FQ->e / Fq->e;
	if (Fq->e * s != FQ->e) {
		cout << "Fq is not a subfield of FQ" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "index = " << s << endl;
	}
	Basis = NEW_int(s);
	embedding = NEW_int(Q);
	embedding_inv = NEW_int(Q);
	components = NEW_int(Q * s);
	FQ_embedding = NEW_int(q);
	Fq_element = NEW_int(Q);
	v = NEW_int(s);

	//alpha = FQ->p; // the primitive element
	//omega = FQ->power(alpha, s);
	for (i = 0; i < s; i++) {
		Basis[i] = given_basis[i];
	}
	if (f_v) {
		cout << "Field basis: ";
		Int_vec_print(cout, Basis, s);
		cout << endl;
	}
	for (i = 0; i < Q; i++) {
		Fq_element[i] = -1;
	}
	for (i = 0; i < q; i++) {
		j = FQ->embed(*Fq, s, i, 0 /* verbose_level */);
		FQ_embedding[i] = j;
		Fq_element[j] = i;
	}

	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(q, v, 1, s, i);
		j = evaluate_over_Fq(v);
		embedding[i] = j;
		embedding_inv[j] = i;
		for (h = 0; h < s; h++) {
			components[j * s + h] = v[h];
		}
	}
	
	
	if (f_v) {
		cout << "subfield_structure::init_with_given_basis done" << endl;
	}
}

void subfield_structure::print_embedding()
{
	long int i, j;
	geometry::geometry_global Gg;
	
	cout << "subfield_structure::print_embedding:" << endl;
	cout << "i : vector over F_q : embedding" << endl;
	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(q, v, 1, s, i);
		j = evaluate_over_Fq(v);
		cout << setw(4) << i << " : ";
		Int_vec_print(cout, v, s);
		cout << " : " << j << endl;
	}
	cout << "subfield_structure::print_embedding in reverse:" << endl;
	cout << "element i in F_Q : vector over F_q : vector AG_rank" << endl;
	for (i = 0; i < Q; i++) {
		j = Gg.AG_element_rank(q, components + i * s, 1, s);
		cout << setw(4) << i << " : ";
		Int_vec_print(cout, components + i * s, s);
		cout << " : " << j << endl;
	}
	
}

void subfield_structure::report(std::ostream &ost)
{
	int i, j;
	geometry::geometry_global Gg;


	ost << "\\subsection*{The Subfield of Order $" << q << "$}" << endl;
	ost << "Field basis:\\\\" << endl;
	ost << "$$" << endl;
	Int_vec_print(ost, Basis, s);
	//cout << endl;
	ost << "$$" << endl;
	ost << "Embedding:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < Q; i++) {
		Gg.AG_element_unrank(q, v, 1, s, i);
		j = evaluate_over_Fq(v);
		ost << setw(4) << i << " & ";
		Int_vec_print(ost, v, s);
		ost << " & " << j << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;


	ost << "In reverse:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "i \\in {\\mathbb F}_Q & \\mbox{vector} & \\mbox{rank}\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < Q; i++) {
		j = Gg.AG_element_rank(q, components + i * s, 1, s);
		ost << setw(4) << i << " & ";
		Int_vec_print(ost, components + i * s, s);
		ost << " & " << j << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

}

int subfield_structure::evaluate_over_FQ(int *v)
{
	int i, a, b;

	a = 0;
	for (i = 0; i < s; i++) {
		b = FQ->mult(v[i], Basis[i]);
		a = FQ->add(a, b);
	}
	return a;
}

int subfield_structure::evaluate_over_Fq(int *v)
{
	int i, a, b, c;

	a = 0;
	for (i = 0; i < s; i++) {
		c = FQ_embedding[v[i]];
		b = FQ->mult(c, Basis[i]);
		a = FQ->add(a, b);
	}
	return a;
}

void subfield_structure::lift_matrix(int *MQ,
		int m, int *Mq, int verbose_level)
// input is MQ[m * m] over the field FQ.
// output is Mq[n * n] over the field Fq,
{
	int f_v = (verbose_level >= 1);
	int i, j, I, J, a, b, c, d, u, v, n;

	if (f_v) {
		cout << "subfield_structure::lift_matrix" << endl;
	}
	n = m * s;
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			a = MQ[i * m + j];
			I = s * i;
			J = s * j;
			for (u = 0; u < s; u++) {
				b = Basis[u];
				c = FQ->mult(b, a);
				for (v = 0; v < s; v++) {
					d = components[c * s + v];
					Mq[(I + u) * n + J + v] = d;
				}
			}
		}
	}

	if (f_v) {
		cout << "subfield_structure::lift_matrix done" << endl;
	}
}

void subfield_structure::retract_matrix(int *Mq,
		int n, int *MQ, int m, int verbose_level)
// input is Mq[n * n] over the field Fq,
// output is MQ[m * m] over the field FQ.
{
	int f_v = (verbose_level >= 1);
	int *vec;
	long int i, j, I, J, u, v, d, b, bv, a, rk;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "subfield_structure::retract_matrix" << endl;
	}
	if (m * s != n) {
		cout << "subfield_structure::retract_matrix m * s != n" << endl;
		exit(1);
	}
	vec = NEW_int(s);
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			I = s * i;
			J = s * j;
			for (u = 0; u < s; u++) {
				for (v = 0; v < s; v++) {
					vec[v] = Mq[(I + u) * n + J + v];
				}
				rk = Gg.AG_element_rank(q, vec, 1, s);
				d = embedding[rk];
				b = Basis[u];
				bv = FQ->inverse(b);
				a = FQ->mult(d, bv);
				if (u == 0) {
					MQ[i * m + j] = a;
				}
				else {
					if (a != MQ[i * m + j]) {
						cout << "subfield_structure::retract_matrix "
								"a != MQ[i * m + j]" << endl;
						exit(1);
					}
				}
			}
		}
	}
	FREE_int(vec);
	if (f_v) {
		cout << "subfield_structure::retract_matrix done" << endl;
	}
}



//Date: Tue, 30 Dec 2014 21:08:19 -0700
//From: Tim Penttila

//To: "betten@math.colostate.edu" <betten@math.colostate.edu>
//Subject: RE: Oops
//Parts/Attachments:
//   1   OK    ~3 KB     Text
//   2 Shown   ~4 KB     Text
//----------------------------------------
//
//Hi Anton,
//
//Friday is predicted to be 42 Celsius, here in Adelaide. So you are
//right! (And I do like that!)
//
//Let b be an element of GF(q^2) of relative norm 1 over GF(q),i.e, b is
//different from 1 but b^{q+1} = 1 . Consider the polynomial
//
//f(t) = (tr(b))^{−1}tr(b^{(q-1)/3})(t + 1) + (tr(b))^{−1}tr((bt +
//b^q)^{(q-1)/3})(t + tr(b)t^{1/2}+ 1)^{1-(q-1)/3} + t^{1/2},
//where tr(x) =x + x^q is the relative trace. When q = 2^h, with h even,
//f(t) is an o-polynomial for the Adelaide hyperoval.
//
//Best,Tim


void subfield_structure::Adelaide_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int q = Fq->q;
	int e = Fq->e;
	int N = q + 2;

	int i, t, b, bq, bk, tr_b, tr_bk, tr_b_down, tr_bk_down, tr_b_down_inv;
	int a, tr_a, tr_a_down, t_lift, alpha, k;
	int sqrt_t, c, cv, d, f;
	int top1, top2, u, v, w, r;
	int *Mtx;

	if (f_v) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"q=" << q << endl;
	}

	if (ODD(e)) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"need e even" << endl;
		exit(1);
	}
	nb_pts = N;

	k = (q - 1) / 3;
	if (k * 3 != q - 1) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"k * 3 != q - 1" << endl;
		exit(1);
	}

	alpha = FQ->alpha;
	b = FQ->power(alpha, q - 1);
	if (FQ->power(b, q + 1) != 1) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"FQ->power(b, q + 1) != 1" << endl;
		exit(1);
	}
	bk = FQ->power(b, k);
	bq = FQ->frobenius_power(b, e);
	tr_b = FQ->add(b, bq);
	tr_bk = FQ->add(bk, FQ->frobenius_power(bk, e));
	tr_b_down = Fq_element[tr_b];
	if (tr_b_down == -1) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"tr_b_down == -1" << endl;
		exit(1);
	}
	tr_bk_down = Fq_element[tr_bk];
	if (tr_bk_down == -1) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"tr_bk_down == -1" << endl;
		exit(1);
	}

	tr_b_down_inv = Fq->inverse(tr_b_down);


	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	Int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {

		sqrt_t = Fq->frobenius_power(t, e - 1);
		if (Fq->mult(sqrt_t, sqrt_t) != t) {
			cout << "subfield_structure::Adelaide_hyperoval "
					"Fq->mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
		}


		t_lift = FQ_embedding[t];
		a = FQ->power(FQ->add(FQ->mult(b, t_lift), bq), k);
		tr_a = FQ->add(a, FQ->frobenius_power(a, e));
		tr_a_down = Fq_element[tr_a];
		if (tr_a_down == -1) {
			cout << "subfield_structure::Adelaide_hyperoval "
					"tr_a_down == -1" << endl;
			exit(1);
		}

		c = Fq->add3(t, Fq->mult(tr_b_down, sqrt_t), 1);
		cv = Fq->inverse(c);
		d = Fq->power(cv, k);
		f = Fq->mult(c, d);

		top1 = Fq->mult(tr_bk_down, Fq->add(t, 1));
		u = Fq->mult(top1, tr_b_down_inv);

		top2 = Fq->mult(tr_a_down, f);
		v = Fq->mult(top2, tr_b_down_inv);


		w = Fq->add3(u, v, sqrt_t);


		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = w;
	}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		Fq->PG_element_rank_modified(Mtx + i * 3, 1, 3, r);
		Pts[i] = r;
	}

	FREE_int(Mtx);

	if (f_v) {
		cout << "subfield_structure::Adelaide_hyperoval "
				"q=" << q << " done" << endl;
	}

}

void subfield_structure::create_adelaide_hyperoval(
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F = Fq;
	int q = F->q;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "subfield_structure::create_adelaide_hyperoval" << endl;
	}

	Adelaide_hyperoval(Pts, nb_pts, verbose_level);

	char str[1000];
	snprintf(str, sizeof(str), "adelaide_hyperoval_q%d.txt", q);
	fname.assign(str);


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;
		geometry::projective_space *P;

		v = NEW_int(d);
		P = NEW_OBJECT(geometry::projective_space);


		P->projective_space_init(n, F,
			FALSE /* f_init_incidence_structure */,
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				Int_vec_print(cout, v, d);
				cout << endl;
			}
		}
		FREE_int(v);
		FREE_OBJECT(P);
	}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "subfield_structure::create_adelaide_hyperoval "
				"the set is not a set, "
				"something is wrong" << endl;
		exit(1);
	}

}


void subfield_structure::field_reduction(int *input, int sz, int *output,
		int verbose_level)
// input[sz], output[(s * sz) * (s * sz * s)],
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, t, J;
	int n;
	int *w;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "subfield_structure::field_reduction" << endl;
	}
	n = sz * s;
	w = NEW_int(sz);

	if (f_v) {
		cout << "input:" << endl;
		Int_vec_print(cout, input, sz);
		cout << endl;
	}

	Int_vec_zero(output, s * n);

	for (i = 0; i < s; i++) {

		if (f_v) {
			cout << "i=" << i << " / " << s << endl;
		}
		// multiply by the i-th basis element,
		// put into the vector w[m]
		a = Basis[i];
		for (j = 0; j < sz; j++) {
			b = input[j];
			if (FALSE) {
				cout << "j=" << j << " / " << sz
						<< " a=" << a << " b=" << b << endl;
			}
			c = FQ->mult(b, a);
			w[j] = c;
		}
		if (f_v) {
			cout << "i=" << i << " / " << s << " w=";
			Int_vec_print(cout, w, sz);
			cout << endl;
		}

		for (j = 0; j < sz; j++) {
			J = j * s;
			b = w[j];
			for (t = 0; t < s; t++) {
				c = components[b * s + t];
				output[i * n + J + t] = c;
			}
		}
		if (f_v) {
			cout << "output:" << endl;
			Int_matrix_print(output, s, n);
		}
	}
	FREE_int(w);

	if (f_v) {
		cout << "subfield_structure::field_reduction done" << endl;
	}

}


}}}




