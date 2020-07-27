/*
 * finite_field_linear_groups.cpp
 *
 *  Created on: Mar 24, 2019
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



void finite_field::diagonal_orbit_perm(int n,
		long int *orbit, long int *orbit_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n + 1);
	geometry_global Gg;
	long int l, ll;
	long int a, b, c;
	long int i, j;

	if (f_v) {
		cout << "finite_field::diagonal_orbit_perm" << endl;
	}
	l = Gg.nb_PG_elements(n - 1, q);
	ll = Gg.nb_AG_elements(n - 1, q - 1);

	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
		}
	for (i = 0; i < ll; i++) {
		v[0] = 1;
		Gg.AG_element_unrank(q - 1, v + 1, 1, n - 1, i);
		for (j = 1; j < n; j++) {
			v[j]++;
			}
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
				}
			}
		PG_element_rank_modified_lint(v, 1, n, a);
		if (f_v) {
			cout << " : " << a << endl;
			}
		b = orbit_inv[a];
		c = orbit[i];
		orbit[i] = a;
		orbit[b] = c;
		orbit_inv[a] = i;
		orbit_inv[c] = b;
		}
	FREE_int(v);
	if (f_v) {
		cout << "finite_field::diagonal_orbit_perm done" << endl;
	}
}

void finite_field::frobenius_orbit_perm(int n,
	long int *orbit, long int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n);
	geometry_global Gg;
	long int l;
	long int ll;
	long int a, b, c;
	long int i, j;

	if (f_v) {
		cout << "finite_field::frobenius_orbit_perm n=" << n
				<< " (vector space dimension)" << endl;
	}
	l = Gg.nb_PG_elements(n - 1, q);
	ll = e;
	if (f_v) {
		cout << "finite_field::frobenius_orbit_perm l=" << l << endl;
		}
	if (e == 1) {
		cout << "finite_field::frobenius_orbit_perm GFq.e == 1" << endl;
		exit(1);
		}
	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
		}
	if (f_v) {
		cout << "before PG_element_unrank_modified("
				<< n + p << ")" << endl;
		}
	PG_element_unrank_modified(v, 1, n, n + p);
	if (f_v) {
		cout << "after PG_element_unrank_modified("
				<< n + p << ")" << endl;
		}
	for (i = 0; i < ll; i++) {
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
				}
			}
		PG_element_rank_modified_lint(v, 1, n, a);
		if (f_v) {
			cout << " : " << a << endl;
			}
		b = orbit_inv[a];
		c = orbit[i];
		orbit[i] = a;
		orbit[b] = c;
		orbit_inv[a] = i;
		orbit_inv[c] = b;
		PG_element_apply_frobenius(n, v, 1);
		}
	FREE_int(v);
	if (f_v) {
		cout << "finite_field::frobenius_orbit_perm done" << endl;
		}
}

void finite_field::projective_matrix_group_base_and_orbits(int n,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	long int **orbit, long int **orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	geometry_global Gg;


	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_orbits "
				"base_len=" << base_len << endl;
	}
	for (i = 0; i < base_len; i++) {
		base[i] = i;
	}
	for (i = 0; i < base_len; i++) {
		transversal_length[i] = i;
	}
	if (f_v) {
		cout << "projective_matrix_group_base_and_orbits "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_semilinear) {
		base[base_len - 1] = n + p;
			// here was an error: the -1 was missing
			// A.B. 11/11/05
			// no that -1 needs to go
			// A.B. 3/9/2006
	}
	//transversal_length[0] = nb_PG_elements(n - 1, q);
	for (i = 0; i < n; i++) {
		transversal_length[i] =
				Gg.nb_PG_elements_not_in_subspace(n - 1, i - 1, q);
		if (f_v) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_v) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"before PG_element_modified_not_in_subspace_perm" << endl;
			}
		PG_element_modified_not_in_subspace_perm(n - 1, i - 1,
			orbit[i], orbit_inv[i], 0);

		if (f_v) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"after PG_element_modified_not_in_subspace_perm" << endl;
			}

		if (FALSE) {
			print_set_lint(cout, degree, orbit[i]);
			cout << endl;
			print_set_lint(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		}
	if (q > 2) {
		transversal_length[i] = Gg.nb_AG_elements(n - 1, q - 1);
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"before diagonal_orbit_perm" << endl;
			}
		diagonal_orbit_perm(n, orbit[i], orbit_inv[i], 0);

		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"after diagonal_orbit_perm" << endl;
			}

		if (FALSE) {
			print_set_lint(cout, degree, orbit[i]);
			cout << endl;
			print_set_lint(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		i++;
		}
	if (f_semilinear) {
		transversal_length[i] = e;
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"before frobenius_orbit_perm" << endl;
			}
		frobenius_orbit_perm(n,
				orbit[i], orbit_inv[i], verbose_level - 2);

		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"after frobenius_orbit_perm" << endl;
			}

		if (FALSE) {
			print_set_lint(cout, degree, orbit[i]);
			cout << endl;
			print_set_lint(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		i++;
		}
	if (i != base_len) {
		cout << "finite_field::projective_matrix_group_base_and_orbits i != base_len" << endl;
		cout << "i=" << i << endl;
		cout << "base_len=" << base_len << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_orbits base: ";
		lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "projective_matrix_group_base_and_orbits "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_orbits done" << endl;
		}
}

void finite_field::projective_matrix_group_base_and_transversal_length(int n,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	geometry_global Gg;


	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_transversal_length" << endl;
		}
	for (i = 0; i < base_len; i++) {
		base[i] = i;
		}
	if (f_semilinear) {
		base[base_len - 1] = n + p;
			// here was an error: the -1 was missing
			// A.B. 11/11/05
			// no that -1 needs to go
			// A.B. 3/9/2006
		}
	//transversal_length[0] = nb_PG_elements(n - 1, q);
	for (i = 0; i < n; i++) {
		transversal_length[i] =
				Gg.nb_PG_elements_not_in_subspace(n - 1, i - 1, q);
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_transversal_length "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		}
	if (q > 2) {
		transversal_length[i] = Gg.nb_AG_elements(n - 1, q - 1);
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_transversal_length: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		i++;
		}
	if (f_semilinear) {
		transversal_length[i] = e;
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_transversal_length: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		i++;
		}
	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_transversal_length base: ";
		lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "finite_field::projective_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "finite_field::projective_matrix_group_base_and_transversal_length done" << endl;
		}
}

void finite_field::affine_matrix_group_base_and_transversal_length(int n,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, c;
	number_theory_domain NT;


	if (f_v) {
		cout << "finite_field::affine_matrix_group_base_and_transversal_length" << endl;
		}
	c = 0;
	base[c] = 0;
	transversal_length[c] = NT.i_power_j(q, n);
	c++;
	for (i = 0; i < n; i++) {
		base[c] = NT.i_power_j_lint(q, i);
		transversal_length[c] = NT.i_power_j_lint(q, n) - NT.i_power_j_lint(q, i);
		c++;
		}
	if (f_semilinear) {
		base[c] = q + p;
		transversal_length[c] = e;
		c++;
		}
	if (c != base_len) {
		cout << "finite_field::affine_matrix_group_base_and_transversal_length "
				"c != base_len" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "finite_field::affine_matrix_group_base_and_transversal_length base: ";
		lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "finite_field::affine_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "finite_field::affine_matrix_group_base_and_transversal_length done" << endl;
		}
}


void finite_field::general_linear_matrix_group_base_and_transversal_length(int n,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, c;
	number_theory_domain NT;


	if (f_v) {
		cout << "finite_field::general_linear_matrix_group_base_and_"
				"transversal_length" << endl;
		}
	c = 0;
	for (i = 0; i < n; i++) {
		base[c] = NT.i_power_j_lint(q, i);
		transversal_length[c] = NT.i_power_j_lint(q, n) - NT.i_power_j_lint(q, i);
		c++;
		}
	if (f_semilinear) {
		base[c] = q + p;
		transversal_length[c] = e;
		c++;
		}
	if (c != base_len) {
		cout << "finite_field::general_linear_matrix_group_base_and_"
				"transversal_length c != base_len" << endl;
		cout << "c=" << c << endl;
		cout << "base_len=" << base_len << endl;
		exit(1);
		}
	if (f_v) {
		cout << "finite_field::general_linear_matrix_group_base_and_"
				"transversal_length base: ";
		lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "finite_field::general_linear_matrix_group_base_and_"
				"transversal_length transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "finite_field::general_linear_matrix_group_base_and_"
				"transversal_length done" << endl;
		}
}


void finite_field::strong_generators_for_projective_linear_group(
	int n,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	int *M;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::strong_generators_for_projective_linear_group" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++;
		}
	if (q > 2) {
		nb_gens += n - 1;
		}
	nb_gens += (n - 1) * e;
	nb_gens += n - 1;
	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}


	// the primitive elements on the diagonal:
	if (q > 2) {
		for (h = 0; h < n - 1; h++) {
			if (f_vv) {
				cout << "generators for primitive elements "
						"on the diagonal:" << endl;
				}
			identity_matrix(M, n);
			M[h * n + h] = primitive_root();
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << e << "):" << endl;
			}
		for (u = 0; u < e; u++) {
			identity_matrix(M, n);
			M[(n - 1) * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// the swaps along the diagonal:
	for (h = n - 2; h >= 0; h--) {
		if (f_vv) {
			cout << "generators for swaps along the diagonal:" << endl;
			}
		identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
			}
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	if (cur != nb_gens) {
		cout << "finite_field::strong_generators_for_projective_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}

	FREE_int(M);
	if (f_v) {
		cout << "finite_field::strong_generators_for_projective_linear_group "
				"done" << endl;
		}
}


void finite_field::strong_generators_for_affine_linear_group(
	int n,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::strong_generators_for_affine_linear_group" << endl;
		}
	size = n * n + n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++; // the field automorphism
		}
	nb_gens += (n - 1) * e; // the bottom layer

	if (q > 2) {
		nb_gens++;
		}

	nb_gens += n - 1; // the transpositions

	nb_gens += n * e; // the translations

	data = NEW_int(size * nb_gens);

	cur = 0;
	if (f_semilinear) {
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);
		data[cur * size + n * n + n] = 1;
		cur++;
		}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << e << "):" << endl;
			}
		for (u = 0; u < e; u++) {
			int_vec_zero(data + cur * size, size);
			identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				data[cur * size + n * n + n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (q > 2) {
		// the primitive element on the last diagonal:
		h = n - 1;
		if (f_vv) {
			cout << "generators for primitive element "
					"on the last diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);

		data[cur * size + h * n + h] = primitive_root();
		if (f_semilinear) {
			data[cur * size + n * n + n] = 0;
			}
		cur++;
		} // if


	// the swaps along the diagonal:
	for (h = n - 2; h >= 0; h--) {
		if (f_vv) {
			cout << "generators for swaps along the diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);
		data[cur * size + h * n + h] = 0;
		data[cur * size + h * n + h + 1] = 1;
		data[cur * size + (h + 1) * n + h] = 1;
		data[cur * size + (h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			data[cur * size + n * n + n] = 0;
			}
		cur++;
		} // next h

	// the translations:
	for (h = 0; h < n; h++) {
		for (u = 0; u < e; u++) {
			int_vec_zero(data + cur * size, size);
			identity_matrix(data + cur * size, n);

			data[cur * size + n * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				data[cur * size + n * n + n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (cur != nb_gens) {
		cout << "finite_field::strong_generators_for_affine_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "finite_field::strong_generators_for_affine_linear_group done" << endl;
		}
}

void finite_field::strong_generators_for_general_linear_group(
	int n,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::strong_generators_for_general_linear_group" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++; // the field automorphism
		}
	nb_gens += (n - 1) * e; // the bottom layer

	if (q > 2) {
		nb_gens++;
		}

	nb_gens += n - 1; // the transpositions


	data = NEW_int(size * nb_gens);

	cur = 0;
	if (f_semilinear) {
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);
		data[cur * size + n * n] = 1;
		cur++;
		}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << e << "):" << endl;
			}
		for (u = 0; u < e; u++) {
			int_vec_zero(data + cur * size, size);
			identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				data[cur * size + n * n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (q > 2) {
		// the primitive element on the last diagonal:
		h = n - 1;
		if (f_vv) {
			cout << "generators for primitive element "
					"on the last diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);

		data[cur * size + h * n + h] = primitive_root();
		if (f_semilinear) {
			data[cur * size + n * n] = 0;
			}
		cur++;
		} // if


	// the swaps along the diagonal:
	for (h = n - 2; h >= 0; h--) {
		if (f_vv) {
			cout << "generators for swaps along the diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		identity_matrix(data + cur * size, n);
		data[cur * size + h * n + h] = 0;
		data[cur * size + h * n + h + 1] = 1;
		data[cur * size + (h + 1) * n + h] = 1;
		data[cur * size + (h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			data[cur * size + n * n] = 0;
			}
		cur++;
		} // next h


	if (cur != nb_gens) {
		cout << "finite_field::strong_generators_for_general_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "finite_field::strong_generators_for_general_linear_group done" << endl;
		}
}

void finite_field::generators_for_parabolic_subgroup(
	int n,
	int f_semilinear, int k,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, g, u, cur;
	int *M;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::generators_for_parabolic_subgroup" << endl;
		}
	size = n * n;

	if (f_semilinear) {
		size++;
		}

	nb_gens = 0;

	// count the Frobenius generator
	if (f_semilinear) {
		nb_gens++;
		}

	// count the generators with primitive elements on the diagonal:
	if (q > 2) {
		nb_gens += n - 1;
		}


	// count the generators with entries in row k:
	for (h = 0; h < k - 1; h++) {
		for (u = 0; u < e; u++) {
			nb_gens++;
		}
	}
	// count the generators with entries in row n:
	for (h = k; h < n - 1; h++) {
		for (u = 0; u < e; u++) {
			nb_gens++;
		}
	}

	// count the generators with entries in the lower left block:
	nb_gens += k * (n - k) * e;

	// count the swaps:
	for (h = n - 2; h >= k; h--) {
		nb_gens++;
	}
	for (h = k - 2; h >= 0; h--) {
		nb_gens++;
	}
#if 0
	if (k > 1 && k < n - 1) {
		nb_gens += n - 2; // swaps
	}
	else {
		nb_gens += n - 1; // swaps
	}
#endif


	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}


	// the primitive elements on the diagonal:
	if (f_vv) {
		cout << "generators for primitive elements "
				"on the diagonal, cur=" << cur << endl;
		}
	if (q > 2) {
		for (h = 0; h < n - 1; h++) {
			identity_matrix(M, n);
			M[h * n + h] = primitive_root();
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// the entries in the row k:
	if (f_vv) {
		cout << "generators for the entries in the last row "
				"of a diagonal block, cur=" << cur << endl;
		}
	for (h = 0; h < k - 1; h++) {
		for (u = 0; u < e; u++) {
			identity_matrix(M, n);
			M[(k - 1) * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// the entries in the row n:
	for (h = k; h < n - 1; h++) {
		for (u = 0; u < e; u++) {
			identity_matrix(M, n);
			M[(n - 1) * n + h] = NT.i_power_j(p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// entries in the lower left block:
	if (f_vv) {
		cout << "generators for the entries in the lower left block, "
				"cur=" << cur << endl;
		}
	for (g = k; g < n; g++) {
		for (h = 0; h < k; h++) {
			for (u = 0; u < e; u++) {
				identity_matrix(M, n);
				M[g * n + h] = NT.i_power_j(p, u);
				if (f_semilinear) {
					M[n * n] = 0;
					}
				int_vec_copy(M, data + cur * size, size);
				cur++;
				}
			}
		}

	// the swaps along the diagonal:
	if (f_vv) {
		cout << "generators for swaps along the diagonal, "
				"cur=" << cur << endl;
		}
	for (h = n - 2; h >= k; h--) {
		identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
			}
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}
	for (h = k - 2; h >= 0; h--) {
		identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
			}
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	if (cur != nb_gens) {
		cout << "finite_field::generators_for_parabolic_subgroup "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}

	FREE_int(M);
	if (f_v) {
		cout << "finite_field::generators_for_parabolic_subgroup done" << endl;
		}
}

void finite_field::generators_for_stabilizer_of_three_collinear_points_in_PGL4(
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, i, j;
	int *M;
	int n = 4;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::generators_for_stabilizer_of_three_collinear_"
				"points_in_PGL4" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}

	nb_gens = 0;

	// automorphic
	if (f_semilinear) {
		nb_gens++;
		}
	nb_gens += 3; // Sym_3 in top left block plus scalars
	nb_gens++; // scalars bottom right



	nb_gens += 4 * e; // lower left block

	nb_gens++; // swaps lower right
	nb_gens += e; // PGL2 in lower right, bottom row

	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
		}
	identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = negate(1);
	M[1 * 4 + 1] = negate(1);
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	identity_matrix(M, n);
	M[0 * 4 + 0] = primitive_root();
	M[0 * 4 + 1] = 0;
	M[1 * 4 + 0] = 0;
	M[1 * 4 + 1] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars in bottom right:
	identity_matrix(M, n);
	M[3 * 4 + 3] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block:
	for (i = 2; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (u = 0; u < e; u++) {
				identity_matrix(M, n);
				M[i * n + j] = NT.i_power_j(p, u);
				if (f_semilinear) {
					M[n * n] = 0;
					}
				int_vec_copy(M, data + cur * size, size);
				cur++;
				}
			}
		}

	// swaps lower right:
	identity_matrix(M, n);
	M[2 * 4 + 2] = 0;
	M[2 * 4 + 3] = 1;
	M[3 * 4 + 2] = 1;
	M[3 * 4 + 3] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// PGL2 in lower right, bottom row
	for (u = 0; u < e; u++) {
		identity_matrix(M, n);
		M[3 * n + 2] = NT.i_power_j(p, u);
		if (f_semilinear) {
			M[n * n] = 0;
			}
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}




	if (cur != nb_gens) {
		cout << "finite_field::generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}

	FREE_int(M);
	if (f_v) {
		cout << "finite_field::generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 done" << endl;
		}
}


void finite_field::generators_for_stabilizer_of_triangle_in_PGL4(
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, j;
	int *M;
	int n = 4;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::generators_for_stabilizer_of_triangle_in_PGL4" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}

	nb_gens = 0;

	// automorphic
	if (f_semilinear) {
		nb_gens++;
		}
	nb_gens += 2; // Sym_3 in top left block
	nb_gens += 3; // scalars in top left block
	nb_gens++; // scalars bottom right

	nb_gens += 3 * e; // lower left block

	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
		}
	identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[1 * 4 + 1] = 0;
	M[2 * 4 + 2] = 0;
	M[0 * 4 + 2] = 1;
	M[1 * 4 + 0] = 1;
	M[2 * 4 + 1] = 1;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars in top left block:
	identity_matrix(M, n);
	M[0 * 4 + 0] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	identity_matrix(M, n);
	M[1 * 4 + 1] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	identity_matrix(M, n);
	M[2 * 4 + 2] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars bottom right
	identity_matrix(M, n);
	M[3 * 4 + 3] = primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block
	for (j = 0; j < 3; j++) {
		for (u = 0; u < e; u++) {
			identity_matrix(M, n);
			M[3 * n + j] = NT.i_power_j(p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}


	if (cur != nb_gens) {
		cout << "finite_field::generators_for_stabilizer_of_triangle_in_PGL4 "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}

	FREE_int(M);
	if (f_v) {
		cout << "finite_field::generators_for_stabilizer_of_triangle_in_PGL4 done" << endl;
		}
}







}}


