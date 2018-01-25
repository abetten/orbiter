// subfield_structure.C
//
// Anton Betten
//
// started:  November 14, 2011




#include "galois.h"


subfield_structure::subfield_structure()
{
	 null();
}

subfield_structure::~subfield_structure()
{
	freeself();
}

void subfield_structure::null()
{
	Basis = NULL;
	embedding = NULL;
	embedding_inv = NULL;
	components = NULL;
	FQ_embedding = NULL;
	Fq_element = NULL;
	v = NULL;
}

void subfield_structure::freeself()
{
	if (Basis) {
		FREE_INT(Basis);
		}
	if (embedding) {
		FREE_INT(embedding);
		}
	if (embedding_inv) {
		FREE_INT(embedding_inv);
		}
	if (components) {
		FREE_INT(components);
		}
	if (FQ_embedding) {
		FREE_INT(FQ_embedding);
		}
	if (Fq_element) {
		FREE_INT(Fq_element);
		}
	if (v) {
		FREE_INT(v);
		}
	null();
}

void subfield_structure::init(finite_field *FQ, finite_field *Fq, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT alpha, omega, i;
	INT *my_basis;

	if (f_v) {
		cout << "subfield_structure::init" << endl;
		}
	subfield_structure::FQ = FQ;
	subfield_structure::Fq = Fq;
	Q = FQ->q;
	q = Fq->q;
	if (FQ->p != Fq->p) {
		cout << "subfield_structure::init different characteristics" << endl;
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


	my_basis = NEW_INT(s);
	alpha = FQ->p; // the primitive element
	omega = FQ->power(alpha, s);
	for (i = 0; i < s; i++) {
		my_basis[i] = FQ->power(omega, i);
		}
	init_with_given_basis(FQ, Fq, my_basis, verbose_level);

	FREE_INT(my_basis);

	if (f_v) {
		cout << "subfield_structure::init done" << endl;
		}
}

void subfield_structure::init_with_given_basis(finite_field *FQ, finite_field *Fq, INT *given_basis, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT alpha, omega, i, j, h;

	if (f_v) {
		cout << "subfield_structure::init_with_given_basis" << endl;
		}
	subfield_structure::FQ = FQ;
	subfield_structure::Fq = Fq;
	Q = FQ->q;
	q = Fq->q;
	if (FQ->p != Fq->p) {
		cout << "subfield_structure::init_with_given_basis different characteristics" << endl;
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
	Basis = NEW_INT(s);
	embedding = NEW_INT(Q);
	embedding_inv = NEW_INT(Q);
	components = NEW_INT(Q * s);
	FQ_embedding = NEW_INT(q);
	Fq_element = NEW_INT(Q);
	v = NEW_INT(s);

	alpha = FQ->p; // the primitive element
	omega = FQ->power(alpha, s);
	for (i = 0; i < s; i++) {
		Basis[i] = given_basis[i];
		}
	if (f_v) {
		cout << "Field basis: ";
		INT_vec_print(cout, Basis, s);
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
		AG_element_unrank(q, v, 1, s, i);
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
	INT i, j;
	
	cout << "subfield_structure::print_embedding:" << endl;
	cout << "i : vector over F_q : embedding" << endl;
	for (i = 0; i < Q; i++) {
		AG_element_unrank(q, v, 1, s, i);
		j = evaluate_over_Fq(v);
		cout << setw(4) << i << " : ";
		INT_vec_print(cout, v, s);
		cout << " : " << j << endl;
		}
	cout << "subfield_structure::print_embedding in reverse:" << endl;
	cout << "element i in F_Q : vector over F_q : vector AG_rank" << endl;
	for (i = 0; i < Q; i++) {
		AG_element_rank(q, components + i * s, 1, s, j);
		cout << setw(4) << i << " : ";
		INT_vec_print(cout, components + i * s, s);
		cout << " : " << j << endl;
		}
	
}

INT subfield_structure::evaluate_over_FQ(INT *v)
{
	INT i, a, b;

	a = 0;
	for (i = 0; i < s; i++) {
		b = FQ->mult(v[i], Basis[i]);
		a = FQ->add(a, b);
		}
	return a;
}

INT subfield_structure::evaluate_over_Fq(INT *v)
{
	INT i, a, b, c;

	a = 0;
	for (i = 0; i < s; i++) {
		c = FQ_embedding[v[i]];
		b = FQ->mult(c, Basis[i]);
		a = FQ->add(a, b);
		}
	return a;
}

void subfield_structure::lift_matrix(INT *MQ, INT m, INT *Mq, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, I, J, a, b, c, d, u, v, n;

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

void subfield_structure::retract_matrix(INT *Mq, INT n, INT *MQ, INT m, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *vec;
	INT i, j, I, J, u, v, d, b, bv, a, rk;

	if (f_v) {
		cout << "subfield_structure::retract_matrix" << endl;
		}
	if (m * s != n) {
		cout << "subfield_structure::retract_matrix m * s != n" << endl;
		exit(1);
		}
	vec = NEW_INT(s);
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			I = s * i;
			J = s * j;
			for (u = 0; u < s; u++) {
				for (v = 0; v < s; v++) {
					vec[v] = Mq[(I + u) * n + J + v];
					}
				AG_element_rank(q, vec, 1, s, rk);
				d = embedding[rk];
				b = Basis[u];
				bv = FQ->inverse(b);
				a = FQ->mult(d, bv);
				if (u == 0) {
					MQ[i * m + j] = a;
					}
				else {
					if (a != MQ[i * m + j]) {
						cout << "subfield_structure::retract_matrix a != MQ[i * m + j]" << endl;
						exit(1);
						}
					}
				}
			}
		}
	FREE_INT(vec);
	if (f_v) {
		cout << "subfield_structure::retract_matrix done" << endl;
		}
}





