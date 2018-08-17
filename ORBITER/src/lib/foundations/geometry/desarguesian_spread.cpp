// desarguesian_spread.C
//
// Anton Betten
// July 5, 2014

#include "foundations.h"



desarguesian_spread::desarguesian_spread()
{
	null();
};

desarguesian_spread::~desarguesian_spread()
{
	freeself();
}

void desarguesian_spread::null()
{
	Spread_elements = NULL;
}

void desarguesian_spread::freeself()
{
	if (Spread_elements) {
		FREE_INT(Spread_elements);
		}
}

void desarguesian_spread::init(INT n, INT m, INT s, 
	subfield_structure *SubS, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "desarguesian_spread::init m=" << m << " n=" << n << " s=" << s << endl;
		}
	desarguesian_spread::n = n;
	desarguesian_spread::m = m;
	desarguesian_spread::s = s;
	desarguesian_spread::SubS = SubS;
	FQ = SubS->FQ;
	Fq = SubS->Fq;
	q = Fq->q;
	Q = FQ->q;
	if (f_v) {
		cout << "desarguesian_spread::init q=" << q << endl;
		cout << "desarguesian_spread::init Q=" << Q << endl;
		}
	if (i_power_j(q, s) != Q) {
		cout << "desarguesian_spread::init i_power_j(q, s) != Q" << endl;
		exit(1);
		}
	if (s != SubS->s) {
		cout << "desarguesian_spread::init s != SubS->s" << endl;
		exit(1);
		}
	nb_points = nb_PG_elements(n - 1, q);
	if (f_v) {
		cout << "desarguesian_spread::init nb_points = " << nb_points << endl;
		}

	N = nb_PG_elements(m - 1, Q);
	if (f_v) {
		cout << "desarguesian_spread::init N = " << N << endl;
		}

	nb_points_per_spread_element = nb_PG_elements(s - 1, q);
	if (f_v) {
		cout << "desarguesian_spread::init nb_points_per_spread_element = " << nb_points_per_spread_element << endl;
		}

	if (f_v) {
		cout << "desarguesian_spread::init before calculate_spread_elements" << endl;
		}
	calculate_spread_elements(verbose_level - 2);
	if (f_v) {
		cout << "desarguesian_spread::init after calculate_spread_elements" << endl;
		}

	if (f_v) {
		cout << "desarguesian_spread::init done" << endl;
		}
}

void desarguesian_spread::calculate_spread_elements(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *v;
	INT *w;
	INT *z;
	INT h, i, j, a, b, c, J, t;

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements" << endl;
		}
	spread_element_size = s * n;
	Spread_elements = NEW_INT(N * spread_element_size);

	v = NEW_INT(m);
	w = NEW_INT(m);
	z = NEW_INT(s * n);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "h=" << h << " / " << N << endl;
			}
		PG_element_unrank_modified(*FQ, v, 1, m, h);
		if (f_vv) {
			INT_vec_print(cout, v, m);
			cout << endl;
			}
		for (i = 0; i < s; i++) {

			if (FALSE) {
				cout << "i=" << i << " / " << s << endl;
				}
			// multiply by the i-th basis element, put into the vector w[m]
			a = SubS->Basis[i];
			for (j = 0; j < m; j++) {
				b = v[j];
				if (FALSE) {
					cout << "j=" << j << " / " << m << " a=" << a << " b=" << b << endl;
					}
				c = FQ->mult(b, a);
				w[j] = c;
				}

			for (j = 0; j < m; j++) {
				J = j * s;
				b = w[j];
				for (t = 0; t < s; t++) {
					c = SubS->components[b * s + t];
					z[i * n + J + t] = c;
					}
				}
			}
		if (f_vv) {
			cout << "basis element " << h << " / " << N << ":" << endl;
			INT_vec_print(cout, v, m);
			cout << endl;
			INT_matrix_print(z, s, n);
			}
		INT_vec_copy(z, Spread_elements + h * spread_element_size, spread_element_size);
		}
	FREE_INT(v);
	FREE_INT(w);
	FREE_INT(z);

	
	INT *Spread_elt_basis;
	INT rk;

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements computing List_of_points" << endl;
		}
	v = NEW_INT(s);
	w = NEW_INT(n);
	List_of_points = NEW_INT(N * nb_points_per_spread_element);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "h=" << h << " / " << N << endl;
			}
		Spread_elt_basis = Spread_elements + h * spread_element_size;
		for (i = 0; i < nb_points_per_spread_element; i++) {
			PG_element_unrank_modified(*Fq, v, 1, s, i);
			Fq->mult_vector_from_the_left(v, Spread_elt_basis, w, s, n);
			PG_element_rank_modified(*Fq, w, 1, n, rk);
			List_of_points[h * nb_points_per_spread_element + i] = rk;
			}
		if (f_vv) {
			cout << "basis element " << h << " / " << N << ":" << endl;
			INT_matrix_print(Spread_elt_basis, s, n);
			cout << "Consists of the following points:" << endl;
			INT_vec_print(cout, List_of_points + h * nb_points_per_spread_element, nb_points_per_spread_element);
			cout << endl;
			}
		}
	FREE_INT(v);
	FREE_INT(w);

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements done" << endl;
		}
}


void desarguesian_spread::compute_intersection_type(INT k, INT *subspace, 
	INT *intersection_dimensions, INT verbose_level)
// intersection_dimensions[N]
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT h, k3;
	INT *intersection;

	if (f_v) {
		cout << "desarguesian_spread::compute_intersection_type" << endl;
		}
	
	intersection = NEW_INT(n * n);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "desarguesian_spread::compute_intersection_type " << h << " / " << N << endl;
			}
		Fq->intersect_subspaces(n, s, Spread_elements + h * spread_element_size, 
			k, subspace, 
			k3, intersection, 
			0 /*verbose_level - 2*/);

		intersection_dimensions[h] = k3;
		}
	FREE_INT(intersection);
	if (f_v) {
		cout << "desarguesian_spread::compute_intersection_type done" << endl;
		}
}

void desarguesian_spread::compute_shadow(INT *Basis, INT basis_sz, 
	INT *is_in_shadow, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Intersection_dimensions;
	INT i, j, rk;


	if (f_v) {
		cout << "desarguesian_spread::compute_shadow" << endl;
		}

	Intersection_dimensions = NEW_INT(N);
	compute_intersection_type(basis_sz, Basis, 
		Intersection_dimensions, 0 /*verbose_level - 1*/);

	if (f_vv) {
		cout << "Intersection_dimensions:";
		INT_vec_print(cout, Intersection_dimensions, N);
		cout << endl;
		}
	
	for (i = 0; i < nb_points; i++) {
		is_in_shadow[i] = FALSE;
		}
	for (i = 0; i < N; i++) {
		if (Intersection_dimensions[i]) {
			for (j = 0; j < nb_points_per_spread_element; j++) {
				rk = List_of_points[i * nb_points_per_spread_element + j];
				if (is_in_shadow[rk]) {
					cout << "is_in_shadow[rk] is TRUE, something is wrong with the spread" << endl;
					exit(1);
					}
				is_in_shadow[rk] = TRUE;
				}
			}
		}
	
	FREE_INT(Intersection_dimensions);
	if (f_v) {
		cout << "desarguesian_spread::compute_shadow done" << endl;
		}
}

void desarguesian_spread::compute_linear_set(INT *Basis, INT basis_sz, 
	INT *&the_linear_set, INT &the_linear_set_sz, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Intersection_dimensions;
	INT i, j;

	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set" << endl;
		}
	Intersection_dimensions = NEW_INT(N);

	compute_intersection_type(basis_sz, Basis, 
		Intersection_dimensions, 0 /*verbose_level - 1*/);
	
	the_linear_set_sz = 0;
	for (i = 0; i < N; i++) {
		if (Intersection_dimensions[i]) {
			the_linear_set_sz++;
			}
		}
	the_linear_set = NEW_INT(the_linear_set_sz);
	j = 0;
	for (i = 0; i < N; i++) {
		if (Intersection_dimensions[i]) {
			the_linear_set[j++] = i;
			}
		}
	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set The linear set is: ";
		INT_vec_print(cout, the_linear_set, the_linear_set_sz);
		cout << endl;
		}

	FREE_INT(Intersection_dimensions);
	
	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set done" << endl;
		}
}

void desarguesian_spread::print_spread_element_table_tex()
{
	INT a, b, i, j;
	INT *v;

	v = NEW_INT(m);
	for (a = 0; a < N; a++) {
		PG_element_unrank_modified(*FQ, v, 1, m, a);
		cout << "$";
		INT_vec_print(cout, v, m);
		cout << "$";
		cout << " & ";
		cout << "$";
		cout << "\\left[" << endl;
		cout << "\\begin{array}{*{" << n << "}{c}}" << endl;
		for (i = 0; i < s; i++) {
			for (j = 0; j < n; j++) {
				b = Spread_elements[a * spread_element_size + i * n + j];
				cout << b << " ";
				if (j < n - 1) {
					cout << "& ";
					}
				}
			cout << "\\\\" << endl;
			}
		cout << "\\end{array}" << endl;
		cout << "\\right]" << endl;
		cout << "$";
		cout << "\\\\" << endl;
		cout << "\\hline" << endl;
		}
	FREE_INT(v);
}

void desarguesian_spread::print_linear_set_tex(INT *set, INT sz)
{
	INT i;

	for (i = 0; i < sz; i++) {
		print_linear_set_element_tex(set[i], sz);
		if (i < sz - 1) {
			cout << ", ";
			}
		}
}

void desarguesian_spread::print_linear_set_element_tex(INT a, INT sz)
{
	INT *v;

	v = NEW_INT(m);
	PG_element_unrank_modified(*FQ, v, 1, m, a);
	cout << "D_{";
	INT_vec_print(cout, v, m);
	cout << "}";

	FREE_INT(v);
}

