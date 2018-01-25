// action_by_subfield_structure.C
//
// Anton Betten
// December 6, 2011

#include "galois.h"
#include "action.h"

INT action_by_subfield_structure::cntr_new = 0;
INT action_by_subfield_structure::cntr_objects = 0;
INT action_by_subfield_structure::f_debug_memory = FALSE;

void *action_by_subfield_structure::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "action_by_subfield_structure::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *action_by_subfield_structure::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_by_subfield_structure);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "action_by_subfield_structure::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void action_by_subfield_structure::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "action_by_subfield_structure::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void action_by_subfield_structure::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_by_subfield_structure);
	if (f_debug_memory) {
		cout << "action_by_subfield_structure::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

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
		FREE_INT(v1);
		}
	if (v2) {
		FREE_INT(v2);
		}
	if (v3) {
		FREE_INT(v3);
		}
	if (Eltq) {
		FREE_INT(Eltq);
		}
	if (Mtx) {
		FREE_INT(Mtx);
		}
	if (S) {
		FREE_OBJECT(S);
		}
	if (Aq) {
		FREE_OBJECT(Aq);
		}
	null();
}

void action_by_subfield_structure::init(action &A, finite_field *Fq, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT p1, h1;
	INT p, h;
	INT q;

	if (f_v) {
		cout << "action_by_subfield_structure::init" << endl;
		cout << "starting with action " << A.label << endl;
		}
	action_by_subfield_structure::Fq = Fq;
	q = Fq->q;
	if (A.type_G != matrix_group_t) {
		cout << "action_by_subfield_structure::init fatal: A.type_G != matrix_group_t" << endl;
		exit(1);
		}
	AQ = &A;
	MQ = AQ->G.matrix_grp;
	FQ = MQ->GFq;
	n = MQ->n;
	Q = FQ->q;
	action_by_subfield_structure::q = q;

	is_prime_power(q, p1, h1);
	is_prime_power(Q, p, h);
	if (p1 != p) {
		cout << "action_by_subfield_structure::init different characteristics of the fields" << endl;
		exit(1);
		}

	s = h / h1;
	if (h1 * s != h) {
		cout << "action_by_subfield_structure::init not a subfield" << endl;
		exit(1);
		}

	m = n * s;
	if (f_v) {
		cout << "action_by_subfield_structure::init" << endl;
		cout << "index=s=" << s << endl;
		cout << "m=s*n=" << m << endl;
		}


	degree = nb_PG_elements(m - 1, q);
	low_level_point_size = m;
	v1 = NEW_INT(m);
	v2 = NEW_INT(m);
	v3 = NEW_INT(m);


	Aq = NEW_OBJECT(action);

	INT f_basis = TRUE;
	INT f_semilinear = FALSE;


	if (f_v) {
		cout << "action_by_subfield_structure::init before Aq->init_matrix_group" << endl;
		}


	Aq->init_projective_group(m, Fq, f_semilinear, f_basis, verbose_level - 2);
	Mq = Aq->G.matrix_grp;


	cout << "action_by_subfield_structure::init after Aq->init_matrix_group" << endl;
	
	cout << "action_by_subfield_structure::init creating subfield structure" << endl;

	S = NEW_OBJECT(subfield_structure);

	S->init(FQ, Fq, verbose_level);
	cout << "action_by_subfield_structure::init creating subfield structure done" << endl;
		
	Eltq = NEW_INT(Aq->elt_size_in_INT);
	Mtx = NEW_INT(m * m);

}

INT action_by_subfield_structure::compute_image_INT(
	action &A, INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT b;
	
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_INT" << endl;
		}
	PG_element_unrank_modified(*Fq, v1, 1, m, a);
	if (f_vv) {
		cout << "action_by_subfield_structure::compute_image_INT a = " << a << " v1 = ";
		INT_vec_print(cout, v1, m);
		cout << endl;
		}
	
	compute_image_INT_low_level(A, Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2=v1 * A=";
		INT_vec_print(cout, v2, m);
		cout << endl;
		}

	PG_element_rank_modified(*Fq, v2, 1, m, b);
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_INT done " << a << "->" << b << endl;
		}
	return b;
}

void action_by_subfield_structure::compute_image_INT_low_level(
	action &A, INT *Elt, INT *input, INT *output, INT verbose_level)
{
	INT *x = input;
	INT *xA = output;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, a, b, c, d, I, J, u, v;
	
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_INT_low_level" << endl;
		}
	if (f_vv) {
		cout << "subfield structure action: x=";
		INT_vec_print(cout, x, m);
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
		INT_vec_print(cout, xA, m);
		cout << endl;
		}
	if (MQ->f_semilinear) {
		cout << "action_by_subfield_structure::compute_image_INT_low_level cannot handle semilinear elements" << endl;
		exit(1);
#if 0
		for (i = 0; i < m; i++) {
			xA[i] = F->frobenius_power(xA[i], f);
			}
		if (f_vv) {
			cout << "after " << f << " field automorphisms: xA=";
			INT_vec_print(cout, xA, m);
			cout << endl;
			}
#endif
		}
	if (f_v) {
		cout << "action_by_subfield_structure::compute_image_INT_low_level done" << endl;
		}
}


