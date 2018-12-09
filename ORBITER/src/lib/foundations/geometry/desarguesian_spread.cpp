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
	SubS = NULL;
	Spread_elements = NULL;
	List_of_points = NULL;
}

void desarguesian_spread::freeself()
{
	if (SubS) {
		FREE_OBJECT(SubS);
	}
	if (Spread_elements) {
		FREE_int(Spread_elements);
		}
	if (List_of_points) {
		FREE_int(List_of_points);
	}
	null();
}

void desarguesian_spread::init(int n, int m, int s, 
	subfield_structure *SubS, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "desarguesian_spread::init m=" << m
				<< " n=" << n << " s=" << s << endl;
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
		cout << "desarguesian_spread::init "
				"i_power_j(q, s) != Q" << endl;
		exit(1);
		}
	if (s != SubS->s) {
		cout << "desarguesian_spread::init "
				"s != SubS->s" << endl;
		exit(1);
		}
	nb_points = nb_PG_elements(n - 1, q);
	if (f_v) {
		cout << "desarguesian_spread::init "
				"nb_points = " << nb_points << endl;
		}

	N = nb_PG_elements(m - 1, Q);
	if (f_v) {
		cout << "desarguesian_spread::init N = " << N << endl;
		}

	nb_points_per_spread_element = nb_PG_elements(s - 1, q);
	if (f_v) {
		cout << "desarguesian_spread::init "
				"nb_points_per_spread_element = "
				<< nb_points_per_spread_element << endl;
		}

	if (f_v) {
		cout << "desarguesian_spread::init "
				"before calculate_spread_elements" << endl;
		}
	calculate_spread_elements(verbose_level - 2);
	if (f_v) {
		cout << "desarguesian_spread::init "
				"after calculate_spread_elements" << endl;
		}

	if (f_v) {
		cout << "desarguesian_spread::init done" << endl;
		}
}

void desarguesian_spread::calculate_spread_elements(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v;
	int *w;
	int *z;
	int h, i, j, a, b, c, J, t;

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements" << endl;
		}
	spread_element_size = s * n;
	Spread_elements = NEW_int(N * spread_element_size);

	v = NEW_int(m);
	w = NEW_int(m);
	z = NEW_int(s * n);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "h=" << h << " / " << N << endl;
			}
		FQ->PG_element_unrank_modified(v, 1, m, h);
		if (f_vv) {
			int_vec_print(cout, v, m);
			cout << endl;
			}
		for (i = 0; i < s; i++) {

			if (FALSE) {
				cout << "i=" << i << " / " << s << endl;
				}
			// multiply by the i-th basis element,
			// put into the vector w[m]
			a = SubS->Basis[i];
			for (j = 0; j < m; j++) {
				b = v[j];
				if (FALSE) {
					cout << "j=" << j << " / " << m
							<< " a=" << a << " b=" << b << endl;
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
			int_vec_print(cout, v, m);
			cout << endl;
			int_matrix_print(z, s, n);
			}
		int_vec_copy(z,
			Spread_elements + h * spread_element_size,
			spread_element_size);
		}
	FREE_int(v);
	FREE_int(w);
	FREE_int(z);

	
	int *Spread_elt_basis;
	int rk;

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements "
				"computing List_of_points" << endl;
		}
	v = NEW_int(s);
	w = NEW_int(n);
	List_of_points = NEW_int(N * nb_points_per_spread_element);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "h=" << h << " / " << N << endl;
			}
		Spread_elt_basis = Spread_elements + h * spread_element_size;
		for (i = 0; i < nb_points_per_spread_element; i++) {
			Fq->PG_element_unrank_modified(v, 1, s, i);
			Fq->mult_vector_from_the_left(v, Spread_elt_basis, w, s, n);
			Fq->PG_element_rank_modified(w, 1, n, rk);
			List_of_points[h * nb_points_per_spread_element + i] = rk;
			}
		if (f_vv) {
			cout << "basis element " << h << " / " << N << ":" << endl;
			int_matrix_print(Spread_elt_basis, s, n);
			cout << "Consists of the following points:" << endl;
			int_vec_print(cout,
				List_of_points + h * nb_points_per_spread_element,
				nb_points_per_spread_element);
			cout << endl;
			}
		}
	FREE_int(v);
	FREE_int(w);

	if (f_v) {
		cout << "desarguesian_spread::calculate_spread_elements done" << endl;
		}
}


void desarguesian_spread::compute_intersection_type(
	int k, int *subspace,
	int *intersection_dimensions, int verbose_level)
// intersection_dimensions[N]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, k3;
	int *intersection;

	if (f_v) {
		cout << "desarguesian_spread::compute_intersection_type" << endl;
		}
	
	intersection = NEW_int(n * n);
	for (h = 0; h < N; h++) {
		if (f_vv) {
			cout << "desarguesian_spread::compute_intersection_type "
					<< h << " / " << N << endl;
			}
		Fq->intersect_subspaces(n, s,
			Spread_elements + h * spread_element_size,
			k, subspace, 
			k3, intersection, 
			0 /*verbose_level - 2*/);

		intersection_dimensions[h] = k3;
		}
	FREE_int(intersection);
	if (f_v) {
		cout << "desarguesian_spread::compute_intersection_type "
				"done" << endl;
		}
}

void desarguesian_spread::compute_shadow(
	int *Basis, int basis_sz,
	int *is_in_shadow, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Intersection_dimensions;
	int i, j, rk;


	if (f_v) {
		cout << "desarguesian_spread::compute_shadow" << endl;
		}

	Intersection_dimensions = NEW_int(N);
	compute_intersection_type(basis_sz, Basis, 
		Intersection_dimensions, 0 /*verbose_level - 1*/);

	if (f_vv) {
		cout << "Intersection_dimensions:";
		int_vec_print(cout, Intersection_dimensions, N);
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
					cout << "is_in_shadow[rk] is TRUE, something is "
							"wrong with the spread" << endl;
					exit(1);
					}
				is_in_shadow[rk] = TRUE;
				}
			}
		}
	
	FREE_int(Intersection_dimensions);
	if (f_v) {
		cout << "desarguesian_spread::compute_shadow done" << endl;
		}
}

void desarguesian_spread::compute_linear_set(int *Basis, int basis_sz, 
	int *&the_linear_set, int &the_linear_set_sz, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Intersection_dimensions;
	int i, j;

	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set" << endl;
		}
	Intersection_dimensions = NEW_int(N);

	compute_intersection_type(basis_sz, Basis, 
		Intersection_dimensions, 0 /*verbose_level - 1*/);
	
	the_linear_set_sz = 0;
	for (i = 0; i < N; i++) {
		if (Intersection_dimensions[i]) {
			the_linear_set_sz++;
			}
		}
	the_linear_set = NEW_int(the_linear_set_sz);
	j = 0;
	for (i = 0; i < N; i++) {
		if (Intersection_dimensions[i]) {
			the_linear_set[j++] = i;
			}
		}
	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set "
				"The linear set is: ";
		int_vec_print(cout, the_linear_set, the_linear_set_sz);
		cout << endl;
		}

	FREE_int(Intersection_dimensions);
	
	if (f_v) {
		cout << "desarguesian_spread::compute_linear_set done" << endl;
		}
}

void desarguesian_spread::print_spread_element_table_tex()
{
	int a, b, i, j;
	int *v;

	v = NEW_int(m);
	for (a = 0; a < N; a++) {
		FQ->PG_element_unrank_modified(v, 1, m, a);
		cout << "$";
		int_vec_print(cout, v, m);
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
	FREE_int(v);
}

void desarguesian_spread::print_linear_set_tex(int *set, int sz)
{
	int i;

	for (i = 0; i < sz; i++) {
		print_linear_set_element_tex(set[i], sz);
		if (i < sz - 1) {
			cout << ", ";
			}
		}
}

void desarguesian_spread::print_linear_set_element_tex(int a, int sz)
{
	int *v;

	v = NEW_int(m);
	FQ->PG_element_unrank_modified(v, 1, m, a);
	cout << "D_{";
	int_vec_print(cout, v, m);
	cout << "}";

	FREE_int(v);
}

