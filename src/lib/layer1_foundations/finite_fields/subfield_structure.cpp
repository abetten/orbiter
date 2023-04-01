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
	index_in_multiplicative_group = 0;
	Basis = NULL;
	embedding = NULL;
	embedding_inv = NULL;
	components = NULL;
	FQ_embedding = NULL;
	Fq_element = NULL;
	v = NULL;
	f_has_2D = FALSE;
	components_2D = NULL;
	embedding_2D = NULL;
	pair_embedding_2D = NULL;
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
	if (f_has_2D) {
		FREE_int(components_2D);
		FREE_int(embedding_2D);
		FREE_int(pair_embedding_2D);
	}
}

void subfield_structure::init(
		finite_field *FQ,
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
	if (f_v) {
		cout << "subfield_structure::init "
				"Q=" << Q << " q=" << q << endl;
	}
	if (FQ->p != Fq->p) {
		cout << "subfield_structure::init "
				"different characteristics" << endl;
		exit(1);
	}
	s = FQ->e / Fq->e;
	if (Fq->e * s != FQ->e) {
		cout << "Fq is not a subfield of FQ" << endl;
		cout << "subfield_structure::init "
				"FQ->e=" << FQ->e
				<< " Fq->e=" << Fq->e << endl;
		exit(1);
	}
	if (f_v) {
		cout << "index = " << s << endl;
	}

	index_in_multiplicative_group = (Q - 1) / (q - 1);
	if (f_v) {
		cout << "index of multiplicative groups = "
				<< index_in_multiplicative_group << endl;
	}


	my_basis = NEW_int(s);
	alpha = FQ->p; // the primitive element
	omega = FQ->power(alpha, s);
	for (i = 0; i < s; i++) {
		my_basis[i] = FQ->power(omega, i);
	}
	init_with_given_basis(FQ, Fq, my_basis, verbose_level);
	FREE_int(my_basis);


	if (s == 2) {
		if (f_v) {
			cout << "subfield_structure::init "
					"before embedding_2dimensional" << endl;
		}
		embedding_2dimensional(verbose_level);
		if (f_v) {
			cout << "subfield_structure::init "
					"after embedding_2dimensional" << endl;
		}
		f_has_2D = TRUE;
	}


	if (f_v) {
		cout << "subfield_structure::init done" << endl;
	}
}

void subfield_structure::init_with_given_basis(
		finite_field *FQ,
		finite_field *Fq, int *given_basis,
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
		//j = FQ->embed(*Fq, s, i, 0 /* verbose_level */);
		j = embed(i, 0 /* verbose_level */);
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

int subfield_structure::embed(int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;

	if (f_v) {
		cout << "subfield_structure::embed b=" << b << endl;
	}

	if (b == 0) {
		a = 0;
	}
	else {
		j = Fq->log_alpha(b);
		i = j * index_in_multiplicative_group;
		a = FQ->alpha_power(i);
	}

	if (f_v) {
		cout << "subfield_structure::embed b=" << b << " a=" << a << endl;
	}
	if (f_v) {
		cout << "subfield_structure::embed done" << endl;
	}
	return a;
}

int subfield_structure::retract(int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "subfield_structure::retract "
				"b=" << b << endl;
	}

	if (b == 0) {
		a = 0;
	}
	else {
		j = FQ->log_alpha(b);
		if ((j % index_in_multiplicative_group)) {
			cout << "subfield_structure::retract "
					"the element does not belong to the subfield" << endl;
			exit(1);
		}
		i = j / index_in_multiplicative_group;
		a = Fq->alpha_power(i);
	}

	if (f_v) {
		cout << "subfield_structure::retract "
				"b=" << b << " a=" << a << endl;
	}
	if (f_v) {
		cout << "subfield_structure::retract done" << endl;
	}
	return a;
}

void subfield_structure::embed_int_vec(
			int *v_in, int *v_out, int len,
			int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b;

	if (f_v) {
		cout << "subfield_structure::embed_int_vec" << endl;
	}
	for (i = 0; i < len; i++) {
		a = v_in[i];
		b = embed(a, verbose_level - 1);
		v_out[i] = b;
	}

	if (f_v) {
		cout << "subfield_structure::embed_int_vec done" << endl;
	}
}

void subfield_structure::retract_int_vec(
			int *v_in, int *v_out, int len,
			int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b;

	if (f_v) {
		cout << "subfield_structure::retract_int_vec" << endl;
	}
	for (i = 0; i < len; i++) {
		a = v_in[i];
		b = retract(a, verbose_level - 1);
		v_out[i] = b;
	}

	if (f_v) {
		cout << "subfield_structure::retract_int_vec done" << endl;
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
	//int i, j;
	//geometry::geometry_global Gg;


	ost << "Subfield of Order $" << q << "$:\\\\" << endl;
	ost << "polynomial: " << Fq->my_poly << "\\\\" << endl;
	ost << "Field basis:\\\\" << endl;
	ost << "$$" << endl;
	Int_vec_print(ost, Basis, s);
	ost << endl;
	ost << "$$" << endl;

	//report_embedding(ost);
	//report_embedding_reverse(ost);

	if (f_has_2D) {
		ost << "$$" << endl;
		print_embedding_2D_table_tex();
		ost << "$$" << endl;
	}
}

void subfield_structure::report_embedding(
		std::ostream &ost)
{
	int i, j;
	geometry::geometry_global Gg;

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
}

void subfield_structure::report_embedding_reverse(
		std::ostream &ost)
{
	int i, j;
	geometry::geometry_global Gg;

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

void subfield_structure::lift_matrix(
		int *MQ,
		int m, int *Mq, int verbose_level)
// input is MQ[m * m] over the field FQ.
// output is Mq[n * n] over the field Fq,
{
	int f_v = (verbose_level >= 1);
	//int i, j, I, J, a, b, c, d, u, v, n;

	if (f_v) {
		cout << "subfield_structure::lift_matrix" << endl;
	}

	lift_matrix_semilinear(
			MQ, 0 /* frob */,
			m, Mq, verbose_level - 1);

#if 0
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
#endif

	if (f_v) {
		cout << "subfield_structure::lift_matrix done" << endl;
	}
}

void subfield_structure::lift_matrix_semilinear(
		int *MQ, int frob,
		int m, int *Mq, int verbose_level)
// input is MQ[m * m] over the field FQ.
// output is Mq[n * n] over the field Fq,
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subfield_structure::lift_matrix" << endl;
#if 0
		cout << "subfield_structure::lift_matrix basis = ";
		Int_vec_print(cout, Basis, s);
		cout << endl;
		cout << "subfield_structure::lift_matrix components = " << endl;
		Int_matrix_print(components, Q, s);
		cout << endl;
#endif
	}

	int i, j, I, J, a, b, c, d, u, v, n;

	n = m * s;
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			a = MQ[i * m + j];
			I = s * i;
			J = s * j;
			for (u = 0; u < s; u++) {
				b = Basis[u];
				c = FQ->mult(b, a);
				c = FQ->frobenius_power(c, frob); // apply the Frobenius
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



void subfield_structure::retract_matrix(
		int *Mq,
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
	int top1, top2, u, v, w;
	long int r;
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
		Fq->Projective_space_basic->PG_element_rank_modified(
				Mtx + i * 3, 1, 3, r);
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


void subfield_structure::field_reduction(
		int *input, int sz, int *output,
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



void subfield_structure::embedding_2dimensional(int verbose_level)
	// we think of FQ as two dimensional vector space
	// over Fq with basis (1,alpha)
	// for i,j \in Fq, with x = i + j * alpha \in FQ, we have
	// pair_embedding_2D[i * q + j] = x;
	// also,
	// components_2D[x * 2 + 0] = i;
	// components_2D[x * 2 + 1] = j;
	// also, for i \in Fq, embedding[i] is the element
	// in FQ that corresponds to i

	// components_2D[Q * 2]
	// embedding_2D[q]
	// pair_embedding_2D[q * q]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int alpha, i, j, I, J, x;

	if (f_v) {
		cout << "subfield_structure::embedding_2dimensional" << endl;
	}
	components_2D = NEW_int(Q * 2);
	embedding_2D = NEW_int(q);
	pair_embedding_2D = NEW_int(q * q);
	alpha = Fq->p;
	embedding_2D[0] = 0;
	for (i = 0; i < q * q; i++) {
		pair_embedding_2D[i] = -1;
	}
	for (i = 0; i < Q * 2; i++) {
		components_2D[i] = -1;
	}
	for (i = 1; i < q; i++) {
		j = embed(i, verbose_level - 2);
		embedding_2D[i] = j;
	}
	for (i = 0; i < q; i++) {
		I = embed(i, verbose_level - 4);
		if (f_vv) {
			cout << "i=" << i << " I=" << I << endl;
		}
		for (j = 0; j < q; j++) {
			J = embed(j, verbose_level - 4);
			x = FQ->add(I, FQ->mult(alpha, J));
			if (pair_embedding_2D[i * q + j] != -1) {
				cout << "error" << endl;
				cout << "element (" << i << "," << j << ") embeds "
						"as (" << I << "," << J << ") = " << x << endl;
				exit(1);
			}
			pair_embedding_2D[i * q + j] = x;
			components_2D[x * 2 + 0] = i;
			components_2D[x * 2 + 1] = j;
			if (f_vv) {
				cout << "element (" << i << "," << j << ") embeds "
						"as (" << I << "," << J << ") = " << x << endl;
			}
		}
	}
	if (f_vv) {
		cout << "subfield_structure::embedding_2dimensional "
				"the two dimensional embedding is" << endl;
		print_embedding();
	}
	if (f_v) {
		cout << "subfield_structure::embedding_2dimensional "
				"done" << endl;
	}

}

void subfield_structure::print_embedding_2D()
{
	int i, j;

	cout << "embedding_2D:" << endl;
	for (i = 0; i < q; i++) {
		cout << setw(4) << i << " : " << setw(4) << embedding_2D[i] << endl;
	}
	cout << "components_2D:" << endl;
	for (i = 0; i < Q; i++) {
		cout << setw(4) << i << setw(4) << components_2D[i * 2 + 0]
			<< setw(4) << components_2D[i * 2 + 1] << endl;
	}
	cout << "pair_embedding_2D:" << endl;
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			cout << setw(4) << i << setw(4) << j << setw(4)
				<< pair_embedding_2D[i * q + j] << endl;
		}
	}
}

void subfield_structure::print_embedding_2D_table_tex()
{
	int i; //, j, a, b, aa, bb, c;


	cout << "\\begin{array}{|c|c|c|}" << endl;
	cout << "\\\\" << endl;
	cout << "i & \\mbox{elt in Fq} & \\mbox{elt in FQ} \\\\" << endl;
	cout << "\\\\" << endl;
	cout << "\\\\" << endl;
	for (i = 0; i < q; i++) {
		cout << i;
		cout << " & ";
		Fq->Io->print_element(cout, i);
		cout << " & ";
		FQ->Io->print_element(cout, embedding_2D[i]);
		cout << "\\\\" << endl;
	}
	cout << "\\end{array}" << endl;
#if 0
	cout << "\\hline" << endl;
	for (i = 0; i < q; i++) {
		Fq->print_element(cout, i);
		if (i == 0) {
			a = 0;
		}
		else {
			a = Fq->alpha_power(i - 1);
		}
		aa = embedding[a];
		for (j = 0; j < q; j++) {
			if (j == 0) {
				b = 0;
			}
			else {
				b = Fq->alpha_power(j - 1);
			}
			bb = embedding_2D[b];
			c = FQ->add(aa, FQ->mult(bb, Fq->p));
			cout << " & ";
			FQ->print_element(cout, c);
		}
		cout << "\\\\" << endl;
	}
#endif
}


}}}




