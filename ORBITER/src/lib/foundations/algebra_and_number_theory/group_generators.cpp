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

#if 0
void translation_in_AG(finite_field &GFq,
		int n, int i, int a, int *perm, int *v,
		int verbose_level)
// v[n] needs to be allocated 
// p[q^n] needs to be allocated
{
	int f_v = (verbose_level >= 1);
	int ii, j, l, q;
	
	q = GFq.q;
	l = nb_AG_elements(n, q);
	for (ii = 0; ii < l; ii++) {
		AG_element_unrank(q, v, 1 /* stride */, n, ii);
		// cout << "ii=" << ii << " v=" << v;
		v[i] = GFq.add(v[i], a);
		
		AG_element_rank(q, v, 1 /* stride */, n, j);
		perm[ii] = j;
		// cout << " j=" << j << endl;
		}
	if (f_v) {
		cout << "translation_in_AG() i=" << i << " a=" << a << " : ";
		perm_print(cout, perm, l);
		cout << endl;
		}
}

void frobenius_in_AG(finite_field &GFq,
		int n, int *perm, int *v,
		int verbose_level)
// v[n] needs to be allocated 
// p[q^n] needs to be allocated
{
	int f_v = (verbose_level >= 1);
	int i, j, l, q, p;
	
	q = GFq.q;
	p = GFq.p;
	l = nb_AG_elements(n, q);
	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1 /* stride */, n, i);
		for (j = 0; j < n; j++) {
			v[j] = GFq.power(v[j], p);
			}
		AG_element_rank(q, v, 1 /* stride */, n, j);
		perm[i] = j;
		}
	if (f_v) {
		cout << "frobenius_in_AG() : ";
		perm_print(cout, perm, l);
		cout << endl;
		}
}

void frobenius_in_PG(finite_field &GFq,
		int n, int *perm, int *v,
		int verbose_level)
// v[n + 1] needs to be allocated 
// p[q^n+...+q+1] needs to be allocated
{
	int f_v = (verbose_level >= 1);
	int i, j, l, q, p;
	
	q = GFq.q;
	p = GFq.p;
	l = nb_PG_elements(n, q);
	for (i = 0; i < l; i++) {
		GFq.PG_element_unrank_modified(v, 1 /* stride */, n + 1, i);
		for (j = 0; j <= n; j++) {
			v[j] = GFq.power(v[j], p);
			}
		GFq.PG_element_unrank_modified(v, 1 /* stride */, n + 1, j);
		perm[i] = j;
		}
	if (f_v) {
		cout << "frobenius_in_PG() : ";
		perm_print(cout, perm, l);
		cout << endl;
		}
}

void AG_representation_of_matrix(finite_field &GFq,
	int n, int f_from_the_right,
	int *M, int *v, int *w, int *perm,
	int verbose_level)
// perm[q^n] needs to be already allocated
{
	int f_v = (verbose_level >= 1);
	int i, j, l, q;
	
	q = GFq.q;
	l = nb_AG_elements(n, q);
	for (i = 0; i < l; i++) {
		AG_element_unrank(q, v, 1 /* stride */, n, i);
		if (f_from_the_right) {
			GFq.mult_matrix_matrix(v, M, w, 1, n, n);
			}
		else {
			GFq.mult_matrix_matrix(M, v, w, n, n, 1);
			}
		AG_element_rank(q, w, 1 /* stride */, n, j);
		perm[i] = j;
		}
	if (f_v) {
		cout << "AG_representation_of_matrix() : ";
		perm_print(cout, perm, l);
		cout << endl;
		}
	
}

void AG_representation_one_dimensional(finite_field &GFq, 
	int a, int *perm, int verbose_level)
// perm[q] needs to be already allocated
{
	int f_v = (verbose_level >= 1);
	int i, j, l, q, v, w;
	
	q = GFq.q;
	l = q;
	if (f_v) {
		cout << "AG_representation_one_dimensional() : "
				"q = " << q << " a=" << a << endl;
		}
	for (i = 0; i < q; i++) {
		AG_element_unrank(q, &v, 1 /* stride */, 1, i);
		w = GFq.mult(a, v);
		AG_element_rank(q, &w, 1 /* stride */, 1, j);
		perm[i] = j;
		}
	if (f_v) {
		cout << "AG_representation_one_dimensional() : ";
		perm_print(cout, perm, l);
		cout << endl;
		}
	
}

int nb_generators_affine_translations(finite_field &GFq, int n)
{
	return n * GFq.e;
}

void generators_affine_translations(finite_field &GFq,
		int n, int *perms, int verbose_level)
// primes[n * d] needs to be allocated, where d = q^n
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v, i, j, l, k = 0, a = 1;
	
	l = nb_AG_elements(n, GFq.q);
	
	if (f_v) {
		cout << "computing generators for affine translations, "
				"q=" << GFq.q << " n = " << n << endl;
		}
	v = NEW_int(n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < GFq.e; j++) {
			translation_in_AG(GFq, n, i, a, perms + k * l, v, f_vv);
			k++;
			a *= GFq.p;
			}
		}
	FREE_int(v);
}

void generators_AGL1xAGL1_subdirect1(
	finite_field &GFq1, finite_field &GFq2,
	int u, int v, int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *perms1;
	int *perms2;
	int nb1, nb2, q1, q2, q12, i, k = 0;
	
	q1 = GFq1.q;
	q2 = GFq2.q;
	q12 = q1 * q2;
	nb1 = nb_generators_affine_translations(GFq1, 1);
	nb2 = nb_generators_affine_translations(GFq2, 1);
	perms1 = NEW_int((nb1 + 3) * q1);
	perms2 = NEW_int((nb2 + 3) * q2);
	nb_perms = nb1 + nb2 + 1;
	perms = NEW_int(nb_perms * q12);

	perm_identity(perms1, q1);
	perm_identity(perms2, q2);

	generators_affine_translations(GFq1, 1, perms1 + q1, verbose_level - 2);
	generators_affine_translations(GFq2, 1, perms2 + q2, verbose_level - 2);
	if (f_v) {
		cout << "affine translations created" << endl;
		}
	
	AG_representation_one_dimensional(GFq1, GFq1.alpha,
			perms1 + (nb1 + 1) * q1,
		verbose_level - 2);
	AG_representation_one_dimensional(GFq2, GFq2.alpha,
			perms2 + (nb2 + 1) * q2,
		verbose_level - 2);
	if (f_v) {
		cout << "AG_representation_one_dimensional created" << endl;
		if (f_vv) {
			perm_print(cout, perms1 + (nb1 + 1) * q1, q1); cout << endl;
			perm_print(cout, perms2 + (nb2 + 1) * q2, q2); cout << endl;
			}
		}
	
	perm_raise(perms1 + (nb1 + 1) * q1, perms1 + (nb1 + 2) * q1, u, q1);
	perm_raise(perms2 + (nb2 + 1) * q2, perms2 + (nb2 + 2) * q2, v, q2);
	if (f_v) {
		cout << "raised to the powers u and v" << endl;
		if (f_vv) {
			perm_print(cout, perms1 + (nb1 + 2) * q1, q1); cout << endl;
			perm_print(cout, perms2 + (nb2 + 2) * q2, q2); cout << endl;
			}
		}
	
	for (i = 0; i < nb1; i++) {
		perm_direct_product(q1, q2,
				perms1 + (i + 1) * q1, perms2, perms + k * q12);
		k++;
		}
	for (i = 0; i < nb2; i++) {
		perm_direct_product(q1, q2,
				perms1, perms2 + (i + 1) * q2, perms + k * q12);
		k++;
		}
	perm_direct_product(q1, q2, 
		perms1 + (nb1 + 2) * q1, 
		perms2 + (nb2 + 2) * q2, 
		perms + k * q12);
	k++;
	if (f_v) {
		cout << "generators for subdirect product "
				"AGL(1," << q1 << ") x AGL(1," << q2 << ") created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * q12, q12);
			cout << endl;
			}
		}
	FREE_int(perms1);
	FREE_int(perms2);
}

void generators_AGL1q(finite_field &GFq,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb, q, i;
	
	q = GFq.q;
	nb = nb_generators_affine_translations(GFq, 1);
	nb_perms = nb + 1;
	perms = NEW_int(nb_perms * q);

	generators_affine_translations(GFq, 1, perms, verbose_level - 2);
	if (f_v) {
		cout << "affine translations created" << endl;
		}
	
	AG_representation_one_dimensional(GFq, GFq.alpha,
			perms + nb * q, verbose_level - 2);
	if (f_v) {
		cout << "AG_representation_one_dimensional created" << endl;
		}
	
	if (f_v) {
		cout << "generators for AGL(1," << q << ") created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * q, q);
			cout << endl;
			}
		}
}

void generators_AGL1q_subgroup(finite_field &GFq,
	int index_in_multiplicative_group,
	int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb, q, i, a, b;
	
	q = GFq.q;
	nb = nb_generators_affine_translations(GFq, 1);
	nb_perms = nb + 1;
	perms = NEW_int(nb_perms * q);

	generators_affine_translations(GFq, 1, perms, verbose_level - 2);
	if (f_v) {
		cout << "affine translations created" << endl;
		}
	
	a = GFq.alpha;
	b = GFq.power(a, index_in_multiplicative_group);
	AG_representation_one_dimensional(GFq, b,
			perms + nb * q, verbose_level - 2);
	if (f_v) {
		cout << "AG_representation_one_dimensional created" << endl;
		}
	
	if (f_v) {
		cout << "generators for AGL(1," << q << ") created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * q, q);
			cout << endl;
			}
		}
}

void generators_AGL1_x_AGL1(
	finite_field &GFq1, finite_field &GFq2, int &deg,
	int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int deg1, nb_perms1, *perms1;
	int deg2, nb_perms2, *perms2;
	int i;
	
	deg1 = GFq1.q;
	deg2 = GFq2.q;
	
	generators_AGL1q(GFq1, nb_perms1, perms1, verbose_level - 1);
	generators_AGL1q(GFq2, nb_perms2, perms2, verbose_level - 1);
	
	generators_direct_product(deg1, nb_perms1, perms1, deg2, 
		nb_perms2, perms2, deg, nb_perms, perms, verbose_level - 1);
	
	FREE_int(perms1);
	FREE_int(perms2);
	if (f_v) {
		cout << "generators for AGL(1," << deg1
				<< ") x AGL(1," << deg2 << ") created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_AGL1_x_AGL1_extension(
	finite_field &GFq1, finite_field &GFq2, int u, int v,
	int &deg, int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *perms1, *perms2;
	int q1, q2, i;
	
	q1 = GFq1.q;
	q2 = GFq2.q;
	
	perms1 = NEW_int(2 * q1);
	perms2 = NEW_int(2 * q2);
	
	deg = q1 * q2;
	nb_perms = 1;
	perms = NEW_int(nb_perms * deg);

	AG_representation_one_dimensional(GFq1, GFq1.alpha, perms1, f_vv);
	AG_representation_one_dimensional(GFq2, GFq2.alpha, perms2, f_vv);
	if (f_v) {
		cout << "AG_representation_one_dimensional created" << endl;
		}
	
	perm_raise(perms1, perms1 + q1, u, q1);
	perm_raise(perms2, perms2 + q2, v, q2);
	if (f_v) {
		cout << "raised to the powers u and v" << endl;
		}
	
	perm_direct_product(q1, q2, perms1 + q1, perms2 + q2, perms);
	FREE_int(perms1);
	FREE_int(perms2);
	
	if (f_v) {
		cout << "generators for a^" << u << "b^" << v << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_AGL1_x_AGL1_extended_once(
	finite_field &F1, finite_field &F2, int u, int v,
	int &deg, int &nb_perms, int *&perms,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int deg1, nb_perms1, *perms1;
	int deg2, nb_perms2, *perms2;
	int q1, q2, i;
	
	q1 = F1.q;
	q2 = F2.q;
	generators_AGL1_x_AGL1(F1, F2,
			deg1, nb_perms1, perms1, verbose_level - 1);
	generators_AGL1_x_AGL1_extension(F1, F2, u, v, deg2,
			nb_perms2, perms2, verbose_level - 1);
	
	generators_concatenate(deg1, nb_perms1, perms1, 
		deg2, nb_perms2, perms2, 
		deg, nb_perms, perms, verbose_level - 1);
	
	FREE_int(perms1);
	FREE_int(perms2);
	
	if (f_v) {
		cout << "generators for AGL(1," << q1 << ") x AGL(1," << q2 << ") "
				"extended by a^" << u << "b^" << v << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void generators_AGL1_x_AGL1_extended_twice(
		finite_field &F1, finite_field &F2,
		int u1, int v1, int u2, int v2, int &deg, int &nb_perms, int *&perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int deg1, nb_perms1, *perms1;
	int deg2, nb_perms2, *perms2;
	int q1, q2, i;
	
	q1 = F1.q;
	q2 = F2.q;
	generators_AGL1_x_AGL1_extended_once(F1, F2, u1, v1,
			deg1, nb_perms1, perms1, verbose_level - 1);
	generators_AGL1_x_AGL1_extension(F1, F2, u2, v2,
			deg2, nb_perms2, perms2, verbose_level - 1);
	
	generators_concatenate(deg1, nb_perms1, perms1, deg2,
			nb_perms2, perms2, deg, nb_perms, perms,
			verbose_level - 1);
	
	FREE_int(perms1);
	FREE_int(perms2);
	
	if (f_v) {
		cout << "generators for AGL(1," << q1 << ") x AGL(1," << q2 << ") "
				"extended by a^" << u1 << "b^" << v1
				<< " and by a^" << u2 << "b^" << v2 << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}
#endif

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


