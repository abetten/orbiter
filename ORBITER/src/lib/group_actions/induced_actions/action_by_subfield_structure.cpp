// action_by_subfield_structure.cpp
//
// Anton Betten
// December 6, 2011

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

action_by_subfield_structure::action_by_subfield_structure()
{
	null();
}

action_by_subfield_structure::~action_by_subfield_structure()
{
	free();
}

void action_by_subfield_structure::null()
{
	MQ = NULL;
	FQ = NULL;
	Mq = NULL;
	Fq = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	Eltq = NULL;
	Mtx = NULL;
	S = NULL;
	Aq = NULL;
	low_level_point_size = 0;
}

void action_by_subfield_structure::free()
{
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (v3) {
		FREE_int(v3);
		}
	if (Eltq) {
		FREE_int(Eltq);
		}
	if (Mtx) {
		FREE_int(Mtx);
		}
	if (S) {
		FREE_OBJECT(S);
		}
	if (Aq) {
		FREE_OBJECT(Aq);
		}
	null();
}

void action_by_subfield_structure::init(action &A,
		finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p1, h1;
	int p, h;
	int q;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "action_by_subfield_structure::init" << endl;
		cout << "starting with action " << A.label << endl;
		}
	action_by_subfield_structure::Fq = Fq;
	q = Fq->q;
	if (A.type_G != matrix_group_t) {
		cout << "action_by_subfield_structure::init "
				"fatal: A.type_G != matrix_group_t" << endl;
		exit(1);
		}
	AQ = &A;
	MQ = AQ->G.matrix_grp;
	FQ = MQ->GFq;
	n = MQ->n;
	Q = FQ->q;
	action_by_subfield_structure::q = q;

	NT.is_prime_power(q, p1, h1);
	NT.is_prime_power(Q, p, h);
	if (p1 != p) {
		cout << "action_by_subfield_structure::init "
				"different characteristics of the fields" << endl;
		exit(1);
		}

	s = h / h1;
	if (h1 * s != h) {
		cout << "action_by_subfield_structure::init "
				"not a subfield" << endl;
		exit(1);
		}

	m = n * s;
	if (f_v) {
		cout << "action_by_subfield_structure::init" << endl;
		cout << "index=s=" << s << endl;
		cout << "m=s*n=" << m << endl;
		}


	degree = Gg.nb_PG_elements(m - 1, q);
	low_level_point_size = m;
	v1 = NEW_int(m);
	v2 = NEW_int(m);
	v3 = NEW_int(m);


	Aq = NEW_OBJECT(action);

	int f_basis = TRUE;
	int f_semilinear = FALSE;


	if (f_v) {
		cout << "action_by_subfield_structure::init "
				"before Aq->init_matrix_group" << endl;
		}

	vector_ge *nice_gens;

	Aq->init_projective_group(m, Fq,
			f_semilinear, f_basis,
			nice_gens,
			verbose_level - 2);
	Mq = Aq->G.matrix_grp;
	FREE_OBJECT(nice_gens);


	cout << "action_by_subfield_structure::init "
			"after Aq->init_matrix_group" << endl;
	
	cout << "action_by_subfield_structure::init "
			"creating subfield structure" << endl;

	S = NEW_OBJECT(subfield_structure);

	S->init(FQ, Fq, verbose_level);
	cout << "action_by_subfield_structure::init "
			"creating subfield structure done" << endl;
		
	Eltq = NEW_int(Aq->elt_size_in_int);
	Mtx = NEW_int(m * m);

}

long int action_by_subfield_structure::compute_image_int(
	action &A, int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int b;
	
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_int" << endl;
		}
	Fq->PG_element_unrank_modified_lint(v1, 1, m, a);
	if (f_vv) {
		cout << "action_by_subfield_structure::compute_image_int "
				"a = " << a << " v1 = ";
		int_vec_print(cout, v1, m);
		cout << endl;
		}
	
	compute_image_int_low_level(A, Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2=v1 * A=";
		int_vec_print(cout, v2, m);
		cout << endl;
		}

	Fq->PG_element_rank_modified_lint(v2, 1, m, b);
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_int "
				"done " << a << "->" << b << endl;
		}
	return b;
}

void action_by_subfield_structure::compute_image_int_low_level(
	action &A, int *Elt, int *input, int *output,
	int verbose_level)
{
	int *x = input;
	int *xA = output;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b, c, d, I, J, u, v;
	
	if (f_v) {
		cout << "action_by_subfield_structure::compute_"
				"image_int_low_level" << endl;
		}
	if (f_vv) {
		cout << "subfield structure action: x=";
		int_vec_print(cout, x, m);
		cout << endl;
		}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = Elt[i * n + j];
			I = s * i;
			J = s * j;
			for (u = 0; u < s; u++) {
				b = S->Basis[u];
				c = FQ->mult(b, a);
				for (v = 0; v < s; v++) {
					d = S->components[c * s + v];
					Mtx[(I + u) * m + J + v] = d;
					}
				}
			}
		}

	Fq->mult_vector_from_the_left(x, Mtx, xA, m, m);


	if (f_vv) {
		cout << "xA=";
		int_vec_print(cout, xA, m);
		cout << endl;
		}
	if (MQ->f_semilinear) {
		cout << "action_by_subfield_structure::compute_"
				"image_int_low_level "
				"cannot handle semilinear elements" << endl;
		exit(1);
#if 0
		for (i = 0; i < m; i++) {
			xA[i] = F->frobenius_power(xA[i], f);
			}
		if (f_vv) {
			cout << "after " << f << " field automorphisms: xA=";
			int_vec_print(cout, xA, m);
			cout << endl;
			}
#endif
		}
	if (f_v) {
		cout << "action_by_subfield_structure::compute_"
				"image_int_low_level "
				"done" << endl;
		}
}

}}


