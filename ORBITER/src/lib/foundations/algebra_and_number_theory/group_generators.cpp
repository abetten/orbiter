// group_generators.C
//
// Anton Betten
//
// moved here from projective.C: September 4, 2016




#include "foundations.h"


void diagonal_orbit_perm(int n, finite_field &GFq,
		int *orbit, int *orbit_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n + 1);
	int l = nb_PG_elements(n - 1, GFq.q);
	int ll = nb_AG_elements(n - 1, GFq.q - 1);
	int a, b, c;
	int i, j;
	
	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
		}
	for (i = 0; i < ll; i++) {
		v[0] = 1;
		AG_element_unrank(GFq.q - 1, v + 1, 1, n - 1, i);
		for (j = 1; j < n; j++) {
			v[j]++;
			}
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
				}
			}
		GFq.PG_element_rank_modified(v, 1, n, a);
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
}

void frobenius_orbit_perm(int n, finite_field &GFq,
	int *orbit, int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n);
	int l = nb_PG_elements(n - 1, GFq.q);
	int ll = GFq.e;
	int a, b, c;
	int i, j;
	
	if (f_v) {
		cout << "frobenius_orbit_perm n=" << n
				<< " (vector space dimension)" << endl;
		cout << "l=" << l << endl;
		}
	if (GFq.e == 1) {
		cout << "frobenius_orbit_perm GFq.e == 1" << endl;
		exit(1);
		}
	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
		}
	if (f_v) {
		cout << "before PG_element_unrank_modified("
				<< n + GFq.p << ")" << endl;
		}
	GFq.PG_element_unrank_modified(v, 1, n, n + GFq.p);
	if (f_v) {
		cout << "after PG_element_unrank_modified("
				<< n + GFq.p << ")" << endl;
		}
	for (i = 0; i < ll; i++) {
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
				}
			}
		GFq.PG_element_rank_modified(v, 1, n, a);
		if (f_v) {
			cout << " : " << a << endl;
			}
		b = orbit_inv[a];
		c = orbit[i];
		orbit[i] = a;
		orbit[b] = c;
		orbit_inv[a] = i;
		orbit_inv[c] = b;
		PG_element_apply_frobenius(n, GFq, v, 1);
		}
	FREE_int(v);
}

void generators_symmetric_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	nb_perms = deg - 1;
	perms = NEW_int(nb_perms * deg);
	for (i = 0; i < nb_perms; i++) {
		perm_identity(perms + i * deg, deg);
		perms[i * deg + i] = i + 1;
		perms[i * deg + i + 1] = i;
		}
	if (f_v) {
		cout << "generators for symmetric group of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_cyclic_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	nb_perms = 1;
	perms = NEW_int(nb_perms * deg);
	for (j = 0; j < deg; j++) {
		perms[i * deg + j] = j + 1;
		}
	perms[i * deg + i + deg - 1] = 0;
	if (f_v) {
		cout << "generators for cyclic group of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_dihedral_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	d2 = deg >> 1;
	nb_perms = 2;
	perms = NEW_int(nb_perms * deg);
	for (j = 0; j < deg; j++) {
		perms[i * deg + j] = j + 1;
		}
	perms[i * deg + i + deg - 1] = 0;
	i++;
	for (j = 0; j <= d2; j++) {
		perms[i * deg + j] = deg - 1 - j;
		perms[i * deg + deg - 1 - j] = j;
		}
	if (f_v) {
		cout << "generators for dihedral group of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_dihedral_involution(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	d2 = deg >> 1;
	nb_perms = 1;
	perms = NEW_int(nb_perms * deg);
	i = 0;
	for (j = 0; j <= d2; j++) {
		perms[i * deg + j] = deg - 1 - j;
		perms[i * deg + deg - 1 - j] = j;
		}
	if (f_v) {
		cout << "generators for dihedral involution of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_identity_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	nb_perms = 1;
	perms = NEW_int(nb_perms * deg);
	for (j = 0; j < deg; j++) {
		perms[j] = j;
		}
	if (f_v) {
		cout << "generators for identity group of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_Hall_reflection_normalizer_group(int nb_pairs,
		int &nb_perms, int *&perms, int &degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h;

	if (f_v) {
		cout << "generators_Hall_reflection_normalizer_group" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = nb_pairs + (nb_pairs - 1);
	perms = NEW_int(nb_perms * degree);
	h = 0;
	for (i = 0; i < nb_pairs; i++, h++) {
		perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * i + 1;
		perms[h * degree + 2 * i + 1] = 2 * i;
	}
	for (i = 0; i < nb_pairs - 1; i++, h++) {
		perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * (i + 1);
		perms[h * degree + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[h * degree + 2 * (i + 1)] = 2 * i;
		perms[h * degree + 2 * (i + 1) + 1] = 2 * i + 1;
		}
	if (h != nb_perms) {
		cout << "generators_Hall_reflection_normalizer_group "
				"h != nb_perms" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "generators_Hall_reflection_normalizer_group "
				"generators for normalizer of the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * degree, degree);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "generators_Hall_reflection_normalizer_group done" << endl;
	}
}

void order_Hall_reflection_normalizer_factorized(int nb_pairs,
		int *&factors, int &nb_factors)
{
	int i, j, nb_perms;

	nb_perms = nb_pairs + nb_pairs - 1;
	nb_factors = nb_perms;
	factors = NEW_int(nb_perms);
	j = 0;
	for (i = 0; i < nb_pairs; i++, j++) {
		factors[j] = 2;
		}
	for (i = 0; i < nb_pairs - 1; i++, j++) {
		factors[j] = nb_pairs - i;
		}
	if (j != nb_factors) {
		cout << "order_Hall_reflection_normalizer_factorized "
				"j != nb_perms" << endl;
		exit(1);
		}
}

void order_Bn_group_factorized(int n, int *&factors, int &nb_factors)
{
	int i, j;

	nb_factors = n + n - 1;
	factors = NEW_int(nb_factors);
	j = 0;
	for (i = 0; i < n - 1; i++, j++) {
		factors[j] = n - i;
		}
	for (i = 0; i < n; i++, j++) {
		factors[j] = 2;
		}
	if (j != nb_factors) {
		cout << "order_Bn_group_factorized j != nb_factors" << endl;
		exit(1);
		}
}

void generators_Bn_group(int n, int &deg, int &nb_perms, int *&perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	
	if (f_v) {
		cout << "generators_Bn_group" << endl;
		}
	deg = 2 * n;
	nb_perms = n - 1 + n;
	perms = NEW_int(nb_perms * deg);
	j = 0;
	for (i = 0; i < n - 1; i++, j++) {
		perm_identity(perms + j * deg, deg);
		perms[j * deg + 2 * i] = 2 * (i + 1);
		perms[j * deg + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[j * deg + 2 * (i + 1)] = 2 * i;
		perms[j * deg + 2 * (i + 1) + 1] = 2 * i + 1;
		}
	for (i = 0; i < n; i++, j++) {
		perm_identity(perms + j * deg, deg);
		perms[j * deg + 2 * i] = 2 * i + 1;
		perms[j * deg + 2 * i + 1] = 2 * i;
		}
	if (f_v) {
		cout << "generators for Bn group of order n = " << n
				<< " and degree " << deg << " created" << endl;
		}
	if (j != nb_perms) {
		cout << "generators_Bn_group j != nb_perms" << endl;
		exit(1);
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "generators_Bn_group done" << endl;
		}
}

void generators_direct_product(int deg1, int nb_perms1, int *perms1, 
	int deg2, int nb_perms2, int *perms2, 
	int &deg3, int &nb_perms3, int *&perms3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k = 0;
	int *id1, *id2;
	
	deg3 = deg1 * deg2;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	id1 = NEW_int(deg1);
	id2 = NEW_int(deg2);
	perm_identity(id1, deg1);
	perm_identity(id2, deg2);
	
	for (i = 0; i < nb_perms1; i++) {
		perm_direct_product(deg1, deg2,
				perms1 + i * deg1, id2, perms3 + k * deg3);
		k++;
		}
	for (i = 0; i < nb_perms2; i++) {
		perm_direct_product(deg1, deg2, id1,
				perms2 + i * deg2, perms3 + k * deg3);
		k++;
		}
	FREE_int(id1);
	FREE_int(id2);
	if (f_v) {
		cout << "generators for direct product created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms3; i++) {
			perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
			}
		}
}

void generators_concatenate(int deg1, int nb_perms1, int *perms1, 
	int deg2, int nb_perms2, int *perms2, 
	int &deg3, int &nb_perms3, int *&perms3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k = 0;
	
	if (deg1 != deg2) {
		cout << "generators_concatenate: deg1 != deg2" << endl;
		exit(1);
		}
	deg3 = deg1;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	
	k = 0;
	for (i = 0; i < nb_perms1; i++) {
		perm_move(perms1 + i * deg1, perms3 + k * deg3, deg3);
		k++;
		}
	for (i = 0; i < nb_perms2; i++) {
		perm_move(perms2 + i * deg1, perms3 + k * deg3, deg3);
		k++;
		}
	if (f_v) {
		cout << "generators concatenated" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms3; i++) {
			perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
			}
		}
}


int matrix_group_base_len_projective_group(int n, int q,
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int base_len;

	base_len = n;
	if (q > 2) {
		base_len++;
		}
	if (f_semilinear) {
		base_len++;
		}
	if (f_v) {
		cout << "matrix_group_base_len_projective_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
		}
	return base_len;
}

int matrix_group_base_len_affine_group(int n, int q,
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int base_len;

	base_len = 1; // the point 0 takes care of killing the translations
	base_len += n;
	if (f_semilinear) {
		base_len++;
		}
	if (f_v) {
		cout << "matrix_group_base_len_affine_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
		}
	return base_len;
}

int matrix_group_base_len_general_linear_group(int n, int q,
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int base_len;

	base_len = 0; // no need to kill translations
	base_len += n;
	if (f_semilinear) {
		base_len++;
		}
	if (f_v) {
		cout << "matrix_group_base_len_general_linear_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
		}
	return base_len;
}



void projective_matrix_group_base_and_orbits(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int **orbit, int **orbit_inv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int q;


	if (f_v) {
		cout << "projective_matrix_group_base_and_orbits" << endl;
		}
	q = F->q;
	for (i = 0; i < base_len; i++) {
		base[i] = i;
		}
	if (f_semilinear) {
		base[base_len - 1] = n + F->p;
			// here was an error: the -1 was missing 
			// A.B. 11/11/05
			// no that -1 needs to go 
			// A.B. 3/9/2006
		}
	//transversal_length[0] = nb_PG_elements(n - 1, q);
	for (i = 0; i < n; i++) {
		transversal_length[i] =
				nb_PG_elements_not_in_subspace(n - 1, i - 1, q);
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"before PG_element_modified_not_in_subspace_perm" << endl;
			}
		PG_element_modified_not_in_subspace_perm(n - 1, i - 1, 
			*F, orbit[i], orbit_inv[i], 0);
			// global function in GALOIS/projective.C

		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"after PG_element_modified_not_in_subspace_perm" << endl;
			}
		
		if (FALSE) {
			print_set(cout, degree, orbit[i]);
			cout << endl;
			print_set(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		}
	if (q > 2) {
		transversal_length[i] = nb_AG_elements(n - 1, q - 1);
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"before diagonal_orbit_perm" << endl;
			}
		diagonal_orbit_perm(n, *F, orbit[i], orbit_inv[i], 0);
			// global function in GALOIS/projective.C
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"after diagonal_orbit_perm" << endl;
			}

		if (FALSE) {
			print_set(cout, degree, orbit[i]);
			cout << endl;
			print_set(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		i++;
		}
	if (f_semilinear) {
		transversal_length[i] = F->e;
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"before frobenius_orbit_perm" << endl;
			}
		frobenius_orbit_perm(n, *F,
				orbit[i], orbit_inv[i], verbose_level - 2);
			// global function in GALOIS/projective.C
		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"after frobenius_orbit_perm" << endl;
			}

		if (FALSE) {
			print_set(cout, degree, orbit[i]);
			cout << endl;
			print_set(cout, degree, orbit_inv[i]);
			cout << endl;
			}
		i++;
		}
	if (f_v) {
		cout << "projective_matrix_group_base_and_orbits base: ";
		int_vec_print(cout, base, base_len);
		cout << endl;
		cout << "projective_matrix_group_base_and_orbits "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "projective_matrix_group_base_and_orbits done" << endl;
		}
}

void projective_matrix_group_base_and_transversal_length(int n,
	finite_field *F, int f_semilinear,
	int base_len, int degree,
	int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int q;


	if (f_v) {
		cout << "projective_matrix_group_base_and_transversal_length" << endl;
		}
	q = F->q;
	for (i = 0; i < base_len; i++) {
		base[i] = i;
		}
	if (f_semilinear) {
		base[base_len - 1] = n + F->p;
			// here was an error: the -1 was missing
			// A.B. 11/11/05
			// no that -1 needs to go
			// A.B. 3/9/2006
		}
	//transversal_length[0] = nb_PG_elements(n - 1, q);
	for (i = 0; i < n; i++) {
		transversal_length[i] =
				nb_PG_elements_not_in_subspace(n - 1, i - 1, q);
		if (f_vv) {
			cout << "projective_matrix_group_base_and_transversal_length "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		}
	if (q > 2) {
		transversal_length[i] = nb_AG_elements(n - 1, q - 1);
		if (f_vv) {
			cout << "projective_matrix_group_base_and_transversal_length: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		i++;
		}
	if (f_semilinear) {
		transversal_length[i] = F->e;
		if (f_vv) {
			cout << "projective_matrix_group_base_and_transversal_length: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
			}
		i++;
		}
	if (f_v) {
		cout << "projective_matrix_group_base_and_transversal_length base: ";
		int_vec_print(cout, base, base_len);
		cout << endl;
		cout << "projective_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "projective_matrix_group_base_and_transversal_length done" << endl;
		}
}

void affine_matrix_group_base_and_transversal_length(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, c;
	int q;


	if (f_v) {
		cout << "affine_matrix_group_base_and_transversal_length" << endl;
		}
	q = F->q;
	c = 0;
	base[c] = 0;
	transversal_length[c] = i_power_j(q, n);
	c++;
	for (i = 0; i < n; i++) {
		base[c] = i_power_j(q, i);
		transversal_length[c] = i_power_j(q, n) - i_power_j(q, i);
		c++;
		}
	if (f_semilinear) {
		base[c] = F->q + F->p;
		transversal_length[c] = F->e;
		c++;
		}
	if (c != base_len) {
		cout << "affine_matrix_group_base_and_transversal_length "
				"c != base_len" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "affine_matrix_group_base_and_transversal_length base: ";
		int_vec_print(cout, base, base_len);
		cout << endl;
		cout << "affine_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "affine_matrix_group_base_and_transversal_length done" << endl;
		}
}


void general_linear_matrix_group_base_and_transversal_length(int n, 
	finite_field *F, int f_semilinear, 
	int base_len, int degree, 
	int *base, int *transversal_length, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, c;
	int q;


	if (f_v) {
		cout << "general_linear_matrix_group_base_and_"
				"transversal_length" << endl;
		}
	q = F->q;
	c = 0;
	for (i = 0; i < n; i++) {
		base[c] = i_power_j(q, i);
		transversal_length[c] = i_power_j(q, n) - i_power_j(q, i);
		c++;
		}
	if (f_semilinear) {
		base[c] = F->q + F->p;
		transversal_length[c] = F->e;
		c++;
		}
	if (c != base_len) {
		cout << "general_linear_matrix_group_base_and_"
				"transversal_length c != base_len" << endl;
		cout << "c=" << c << endl;
		cout << "base_len=" << base_len << endl;
		exit(1);
		}
	if (f_v) {
		cout << "general_linear_matrix_group_base_and_"
				"transversal_length base: ";
		int_vec_print(cout, base, base_len);
		cout << endl;
		cout << "general_linear_matrix_group_base_and_"
				"transversal_length transversal_length: ";
		int_vec_print(cout, transversal_length, base_len);
		cout << endl;
		}
	if (f_v) {
		cout << "general_linear_matrix_group_base_and_"
				"transversal_length done" << endl;
		}
}


void strong_generators_for_projective_linear_group(
	int n, finite_field *F,
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	int *M;
	
	if (f_v) {
		cout << "strong_generators_for_projective_linear_group" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++;
		}
	if (F->q > 2) {
		nb_gens += n - 1;
		}
	nb_gens += (n - 1) * F->e;
	nb_gens += n - 1;
	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		F->identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}


	// the primitive elements on the diagonal:
	if (F->q > 2) {
		for (h = 0; h < n - 1; h++) {
			if (f_vv) {
				cout << "generators for primitive elements "
						"on the diagonal:" << endl;
				}
			F->identity_matrix(M, n);
			M[h * n + h] = F->primitive_root();
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
					"(e=" << F->e << "):" << endl;
			}
		for (u = 0; u < F->e; u++) {
			F->identity_matrix(M, n);
			M[(n - 1) * n + h] = i_power_j(F->p, u);
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
		F->identity_matrix(M, n);
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
		cout << "strong_generators_for_projective_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}
	
	FREE_int(M);
	if (f_v) {
		cout << "strong_generators_for_projective_linear_group "
				"done" << endl;
		}
}


void strong_generators_for_affine_linear_group(
	int n, finite_field *F,
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	
	if (f_v) {
		cout << "strong_generators_for_affine_linear_group" << endl;
		}
	size = n * n + n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++; // the field automorphism
		}
	nb_gens += (n - 1) * F->e; // the bottom layer

	if (F->q > 2) {
		nb_gens++;
		}

	nb_gens += n - 1; // the transpositions

	nb_gens += n * F->e; // the translations

	data = NEW_int(size * nb_gens);

	cur = 0;
	if (f_semilinear) {
		int_vec_zero(data + cur * size, size);
		F->identity_matrix(data + cur * size, n);
		data[cur * size + n * n + n] = 1;
		cur++;
		}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << F->e << "):" << endl;
			}
		for (u = 0; u < F->e; u++) {
			int_vec_zero(data + cur * size, size);
			F->identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = i_power_j(F->p, u);
			if (f_semilinear) {
				data[cur * size + n * n + n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (F->q > 2) {
		// the primitive element on the last diagonal:
		h = n - 1;
		if (f_vv) {
			cout << "generators for primitive element "
					"on the last diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		F->identity_matrix(data + cur * size, n);

		data[cur * size + h * n + h] = F->primitive_root();
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
		F->identity_matrix(data + cur * size, n);
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
		for (u = 0; u < F->e; u++) {
			int_vec_zero(data + cur * size, size);
			F->identity_matrix(data + cur * size, n);

			data[cur * size + n * n + h] = i_power_j(F->p, u);
			if (f_semilinear) {
				data[cur * size + n * n + n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (cur != nb_gens) {
		cout << "strong_generators_for_affine_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "strong_generators_for_affine_linear_group done" << endl;
		}
}

void strong_generators_for_general_linear_group(
	int n, finite_field *F,
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	
	if (f_v) {
		cout << "strong_generators_for_general_linear_group" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++; // the field automorphism
		}
	nb_gens += (n - 1) * F->e; // the bottom layer

	if (F->q > 2) {
		nb_gens++;
		}

	nb_gens += n - 1; // the transpositions


	data = NEW_int(size * nb_gens);

	cur = 0;
	if (f_semilinear) {
		int_vec_zero(data + cur * size, size);
		F->identity_matrix(data + cur * size, n);
		data[cur * size + n * n] = 1;
		cur++;
		}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << F->e << "):" << endl;
			}
		for (u = 0; u < F->e; u++) {
			int_vec_zero(data + cur * size, size);
			F->identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = i_power_j(F->p, u);
			if (f_semilinear) {
				data[cur * size + n * n] = 0;
				}
			cur++;
			} // next u
		} // next h

	if (F->q > 2) {
		// the primitive element on the last diagonal:
		h = n - 1;
		if (f_vv) {
			cout << "generators for primitive element "
					"on the last diagonal:" << endl;
			}
		int_vec_zero(data + cur * size, size);
		F->identity_matrix(data + cur * size, n);

		data[cur * size + h * n + h] = F->primitive_root();
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
		F->identity_matrix(data + cur * size, n);
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
		cout << "strong_generators_for_general_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "strong_generators_for_general_linear_group done" << endl;
		}
}

void generators_for_parabolic_subgroup(
	int n, finite_field *F,
	int f_semilinear, int k, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, g, u, cur;
	int *M;
	
	if (f_v) {
		cout << "generators_for_parabolic_subgroup" << endl;
		}
	size = n * n;
	if (f_semilinear) {
		size++;
		}
	nb_gens = 0;
	if (f_semilinear) {
		nb_gens++;
		}
	if (F->q > 2) {
		nb_gens += n - 1;
		}
	nb_gens += (n - 1) * F->e;
	nb_gens += k * (n - k) * F->e;
	nb_gens += n - 2; // swaps
	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		F->identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}


	// the primitive elements on the diagonal:
	if (f_vv) {
		cout << "generators for primitive elements "
				"on the diagonal, cur=" << cur << endl;
		}
	if (F->q > 2) {
		for (h = 0; h < n - 1; h++) {
			F->identity_matrix(M, n);
			M[h * n + h] = F->primitive_root();
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}

	// the entries in the last row:
	if (f_vv) {
		cout << "generators for the entries in the last row "
				"of a diagonal block, cur=" << cur << endl;
		}
	for (h = 0; h < k - 1; h++) {
		for (u = 0; u < F->e; u++) {
			F->identity_matrix(M, n);
			M[(k - 1) * n + h] = i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}
	for (h = k - 1; h < n - 1; h++) {
		for (u = 0; u < F->e; u++) {
			F->identity_matrix(M, n);
			M[(n - 1) * n + h] = i_power_j(F->p, u);
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
			for (u = 0; u < F->e; u++) {
				F->identity_matrix(M, n);
				M[g * n + h] = i_power_j(F->p, u);
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
	for (h = n - 2; h >= n - k; h--) {
		F->identity_matrix(M, n);
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
		F->identity_matrix(M, n);
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
		cout << "generators_for_parabolic_subgroup "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}
	
	FREE_int(M);
	if (f_v) {
		cout << "generators_for_parabolic_subgroup done" << endl;
		}
}

void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
	finite_field *F,
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, i, j;
	int *M;
	int n = 4;
	
	if (f_v) {
		cout << "generators_for_stabilizer_of_three_collinear_"
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



	nb_gens += 4 * F->e; // lower left block

	nb_gens++; // swaps lower right
	nb_gens += F->e; // PGL2 in lower right, bottom row

	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		F->identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
		}
	F->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	F->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = F->negate(1);
	M[1 * 4 + 1] = F->negate(1);
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	F->identity_matrix(M, n);
	M[0 * 4 + 0] = F->primitive_root();
	M[0 * 4 + 1] = 0;
	M[1 * 4 + 0] = 0;
	M[1 * 4 + 1] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars in bottom right:
	F->identity_matrix(M, n);
	M[3 * 4 + 3] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block:
	for (i = 2; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (u = 0; u < F->e; u++) {
				F->identity_matrix(M, n);
				M[i * n + j] = i_power_j(F->p, u);
				if (f_semilinear) {
					M[n * n] = 0;
					}
				int_vec_copy(M, data + cur * size, size);
				cur++;
				}
			}
		}

	// swaps lower right:
	F->identity_matrix(M, n);
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
	for (u = 0; u < F->e; u++) {
		F->identity_matrix(M, n);
		M[3 * n + 2] = i_power_j(F->p, u);
		if (f_semilinear) {
			M[n * n] = 0;
			}
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}




	if (cur != nb_gens) {
		cout << "generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}
	
	FREE_int(M);
	if (f_v) {
		cout << "generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 done" << endl;
		}
}


void generators_for_stabilizer_of_triangle_in_PGL4(
	finite_field *F,
	int f_semilinear, 
	int *&data, int &size, int &nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, j;
	int *M;
	int n = 4;
	
	if (f_v) {
		cout << "generators_for_stabilizer_of_triangle_in_PGL4" << endl;
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

	nb_gens += 3 * F->e; // lower left block

	data = NEW_int(size * nb_gens);
	M = NEW_int(size);

	cur = 0;

	// the automorphic collineation:
	if (f_semilinear) {
		F->identity_matrix(M, n);
		M[n * n] = 1;
		int_vec_copy(M, data + cur * size, size);
		cur++;
		}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
		}
	F->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	F->identity_matrix(M, n);
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
	F->identity_matrix(M, n);
	M[0 * 4 + 0] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	F->identity_matrix(M, n);
	M[1 * 4 + 1] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;
	F->identity_matrix(M, n);
	M[2 * 4 + 2] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars bottom right
	F->identity_matrix(M, n);
	M[3 * 4 + 3] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
		}
	int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block
	for (j = 0; j < 3; j++) {
		for (u = 0; u < F->e; u++) {
			F->identity_matrix(M, n);
			M[3 * n + j] = i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
				}
			int_vec_copy(M, data + cur * size, size);
			cur++;
			}
		}


	if (cur != nb_gens) {
		cout << "generators_for_stabilizer_of_triangle_in_PGL4 "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
		}
	
	FREE_int(M);
	if (f_v) {
		cout << "generators_for_stabilizer_of_triangle_in_PGL4 done" << endl;
		}
}


