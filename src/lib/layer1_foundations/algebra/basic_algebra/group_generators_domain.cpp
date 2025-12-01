// group_generators_domain.cpp
//
// Anton Betten
//
// moved here from projective.cpp: September 4, 2016




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {



group_generators_domain::group_generators_domain()
{
	Record_birth();

}

group_generators_domain::~group_generators_domain()
{
	Record_death();

}


void group_generators_domain::generators_symmetric_group(
		int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_symmetric_group" << endl;
	}
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
	}
	nb_perms = deg - 1;
	perms = NEW_int(nb_perms * deg);
	for (i = 0; i < nb_perms; i++) {
		Combi.Permutations->perm_identity(perms + i * deg, deg);
		perms[i * deg + i] = i + 1;
		perms[i * deg + i + 1] = i;
	}
	if (f_v) {
		cout << "group_generators_domain::generators_symmetric_group "
				"generators are:" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_symmetric_group "
				"done" << endl;
	}
}

void group_generators_domain::generators_cyclic_group(
		int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_cyclic_group" << endl;
	}
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
		cout << "group_generators_domain::generators_cyclic_group "
				"generators are:" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_cyclic_group "
				"done" << endl;
	}
}

void group_generators_domain::generators_dihedral_group(
		int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_dihedral_group" << endl;
	}
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
		cout << "group_generators_domain::generators_dihedral_group "
				"generators for dihedral group of degree "
				<< deg << " created" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_dihedral_group one" << endl;
	}
}

void group_generators_domain::generators_dihedral_involution(
		int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_dihedral_involution" << endl;
	}
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
		cout << "group_generators_domain::generators_dihedral_involution "
				"generators for dihedral involution of degree "
				<< deg << " created" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_dihedral_involution done" << endl;
	}
}

void group_generators_domain::generators_identity_group(
		int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_identity_group" << endl;
	}
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
		cout << "group_generators_domain::generators_identity_group "
				"generators for identity group of degree "
				<< deg << " created" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_identity_group done" << endl;
	}
}

void group_generators_domain::generators_Hall_reflection(
		int nb_pairs,
		int &nb_perms, int *&perms, int &degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = 1;
	perms = NEW_int(nb_perms * degree);
	Combi.Permutations->perm_identity(perms, degree);
	for (i = 0; i < nb_pairs; i++) {
		perms[2 * i] = 2 * i + 1;
		perms[2 * i + 1] = 2 * i;
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection "
				"generators for the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
	}
	if (f_vv) {
		for (i = 0; i < 1; i++) {
			Combi.Permutations->perm_print(cout, perms + i * degree, degree);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection done" << endl;
	}
}

void group_generators_domain::generators_Hall_reflection_normalizer_group(
		int nb_pairs,
		int &nb_perms, int *&perms, int &degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection_normalizer_group" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = nb_pairs + (nb_pairs - 1);
	perms = NEW_int(nb_perms * degree);
	h = 0;
	for (i = 0; i < nb_pairs; i++, h++) {
		Combi.Permutations->perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * i + 1;
		perms[h * degree + 2 * i + 1] = 2 * i;
	}
	for (i = 0; i < nb_pairs - 1; i++, h++) {
		Combi.Permutations->perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * (i + 1);
		perms[h * degree + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[h * degree + 2 * (i + 1)] = 2 * i;
		perms[h * degree + 2 * (i + 1) + 1] = 2 * i + 1;
	}
	if (h != nb_perms) {
		cout << "group_generators_domain::generators_Hall_reflection_normalizer_group "
				"h != nb_perms" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection_normalizer_group "
				"generators for normalizer of the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.Permutations->perm_print(cout, perms + i * degree, degree);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_reflection_normalizer_group done" << endl;
	}
}

void group_generators_domain::order_Hall_reflection_normalizer_factorized(
		int nb_pairs,
		int *&factors, int &nb_factors)
{
	int i, j, nb_perms;
	int f_v = false;

	if (f_v) {
		cout << "group_generators_domain::order_Hall_reflection_normalizer_factorized" << endl;
	}
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
		cout << "group_generators_domain:order_Hall_reflection_normalizer_factorized "
				"j != nb_perms" << endl;
		exit(1);
	}
}

void group_generators_domain::order_Bn_group_factorized(
		int n, int *&factors, int &nb_factors)
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
		cout << "group_generators_domain::order_Bn_group_factorized "
				"j != nb_factors" << endl;
		exit(1);
	}
}

void group_generators_domain::generators_Bn_group(
		int n, int &deg, int &nb_perms, int *&perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_Bn_group" << endl;
	}
	deg = 2 * n;
	nb_perms = n - 1 + n;
	perms = NEW_int(nb_perms * deg);
	j = 0;
	for (i = 0; i < n - 1; i++, j++) {
		Combi.Permutations->perm_identity(perms + j * deg, deg);
		perms[j * deg + 2 * i] = 2 * (i + 1);
		perms[j * deg + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[j * deg + 2 * (i + 1)] = 2 * i;
		perms[j * deg + 2 * (i + 1) + 1] = 2 * i + 1;
	}
	for (i = 0; i < n; i++, j++) {
		Combi.Permutations->perm_identity(perms + j * deg, deg);
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
			Combi.Permutations->perm_print(cout, perms + i * deg, deg);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Bn_group done" << endl;
	}
}

void group_generators_domain::generators_direct_product(
		int deg1, int nb_perms1, int *perms1,
		int deg2, int nb_perms2, int *perms2,
		int &deg3, int &nb_perms3, int *&perms3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k = 0;
	int *id1, *id2;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_direct_product" << endl;
	}
	deg3 = deg1 * deg2;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	id1 = NEW_int(deg1);
	id2 = NEW_int(deg2);
	Combi.Permutations->perm_identity(id1, deg1);
	Combi.Permutations->perm_identity(id2, deg2);
	
	for (i = 0; i < nb_perms1; i++) {
		Combi.Permutations->perm_direct_product(deg1, deg2,
				perms1 + i * deg1, id2, perms3 + k * deg3);
		k++;
	}
	for (i = 0; i < nb_perms2; i++) {
		Combi.Permutations->perm_direct_product(deg1, deg2, id1,
				perms2 + i * deg2, perms3 + k * deg3);
		k++;
	}
	FREE_int(id1);
	FREE_int(id2);
	if (f_vv) {
		for (i = 0; i < nb_perms3; i++) {
			Combi.Permutations->perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_direct_product done" << endl;
	}
}

void group_generators_domain::generators_concatenate(
	int deg1, int nb_perms1, int *perms1,
	int deg2, int nb_perms2, int *perms2, 
	int &deg3, int &nb_perms3, int *&perms3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k = 0;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_concatenate" << endl;
	}
	if (deg1 != deg2) {
		cout << "group_generators_domain::generators_concatenate"
				" deg1 != deg2" << endl;
		exit(1);
	}
	deg3 = deg1;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	
	k = 0;
	for (i = 0; i < nb_perms1; i++) {
		Combi.Permutations->perm_move(perms1 + i * deg1, perms3 + k * deg3, deg3);
		k++;
	}
	for (i = 0; i < nb_perms2; i++) {
		Combi.Permutations->perm_move(perms2 + i * deg1, perms3 + k * deg3, deg3);
		k++;
	}
	if (f_v) {
		cout << "generators concatenated" << endl;
	}
	if (f_vv) {
		for (i = 0; i < nb_perms3; i++) {
			Combi.Permutations->perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_generators_domain::generators_concatenate done" << endl;
	}
}


int group_generators_domain::matrix_group_base_len_projective_group(
		int n, int q,
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int base_len;

	if (f_v) {
		cout << "group_generators_domain::matrix_group_base_len_projective_group" << endl;
	}
	base_len = n;
	if (q > 2) {
		base_len++;
	}
	if (f_semilinear) {
		base_len++;
	}
	if (f_v) {
		cout << "group_generators_domain::matrix_group_base_len_projective_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
	}
	if (f_v) {
		cout << "group_generators_domain::matrix_group_base_len_projective_group done" << endl;
	}
	return base_len;
}

int group_generators_domain::matrix_group_base_len_affine_group(
		int n, int q,
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
		cout << "group_generators_domain::matrix_group_base_len_affine_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
	}
	return base_len;
}

int group_generators_domain::matrix_group_base_len_general_linear_group(
		int n, int q,
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
		cout << "group_generators_domain::matrix_group_base_len_general_linear_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
	}
	return base_len;
}

void group_generators_domain::order_POmega_epsilon(
		int epsilon, int k, int q,
		algebra::ring_theory::longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int f_v = (verbose_level >= 1);
	int w, m;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::order_POmega_epsilon" << endl;
	}
	w = Gg.Witt_index(epsilon, k);
	if (epsilon == -1) {
		m = w + 1;
	}
	else {
		m = w;
	}
	order_Pomega(epsilon, m, q, go, verbose_level);
	if (f_v) {
		cout << "group_generators_domain::order_POmega_epsilon "
				"done  epsilon=" << epsilon
				<< " k=" << k << " q=" << q << " order=" << go << endl;
	}

#if 0
	int f_v = (verbose_level >= 1);
	int n;

	n = Witt_index(epsilon, k);
	if (f_v) {
		cout << "Witt index is " << n << endl;
		}
	if (epsilon == 0) {
		order_Pomega(0, n, q, go, verbose_level);
		}
	else if (epsilon == 1) {
		order_Pomega_plusminus(1, n, q, go, verbose_level);
		}
	else if (epsilon == -1) {
		order_Pomega_plusminus(-1, n, q, go, verbose_level);
		}
#endif
}


void group_generators_domain::order_PO_epsilon(
		int f_semilinear,
		int epsilon, int k, int q,
		algebra::ring_theory::longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int f_v = (verbose_level >= 1);
	int m;
	number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::order_PO_epsilon" << endl;
	}
	m = Gg.Witt_index(epsilon, k);
	if (f_v) {
		cout << "Witt index = " << m << endl;
	}
	order_PO(epsilon, m, q, go, verbose_level);
	if (f_semilinear) {
		int p, e;
		ring_theory::longinteger_domain D;

		NT.factor_prime_power(q, p, e);
		D.mult_integer_in_place(go, e);
	}
	if (f_v) {
		cout << "order_Pgroup_generators_domain::order_PO_epsilon done "
				"f_semilinear=" << f_semilinear
				<< " epsilon=" << epsilon << " k=" << k
				<< " q=" << q << " order=" << go << endl;
	}
}

void group_generators_domain::order_PO(
		int epsilon, int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_generators_domain::order_PO epsilon = " << epsilon
				<< " m=" << m << " q=" << q << endl;
	}

	if (epsilon == 0) {
		order_PO_parabolic(m, q, o, verbose_level);
	}
	else if (epsilon == 1) {
		order_PO_plus(m, q, o, verbose_level);
	}
	else if (epsilon == -1) {
		order_PO_minus(m, q, o, verbose_level);
	}
	else {
		cout << "group_generators_domain::order_PO fatal: epsilon = " << epsilon << endl;
		exit(1);
	}
}

void group_generators_domain::order_Pomega(
		int epsilon, int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
{
	if (epsilon == 0) {
		order_Pomega_parabolic(m, q, o, verbose_level);
	}
	else if (epsilon == 1) {
		order_Pomega_plus(m, q, o, verbose_level);
	}
	else if (epsilon == -1) {
		order_Pomega_minus(m, q, o, verbose_level);
	}
	else {
		cout << "group_generators_domain::order_Pomega "
				"fatal: epsilon = " << epsilon << endl;
		exit(1);
	}
}

void group_generators_domain::order_PO_plus(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, T, Two, minusone;
	int i;
	geometry::other_geometry::geometry_global Gg;


	Two.create(2);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << "group_generators_domain::order_PO_plus " << q << "^(" << m << "*"
				<< m - 1 << ") = " << Q << endl;
	}
	// now Q = q^{m(m-1)}

	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "group_generators_domain::order_PO_plus " << q << "^"
					<< 2 * i << " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}
	// now O = \prod_{i=1}^{m-1} (q^{2i}-1)

	R.create(q);
	D.power_int(R, m);
	D.add(R, minusone, S);
	if (f_v) {
		cout << "group_generators_domain::order_PO_plus " << q << "^" << m << " - 1 = " << S << endl;
	}
	// now S = q^m-1

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	if (true /*EVEN(q)*/) {
		D.mult(T, Two, o);
	}
	else {
		T.assign_to(o);
	}


	if (f_v) {
		cout << "group_generators_domain::order_PO_plus the order of PO" << "("
				<< Gg.dimension_given_Witt_index(1, m) << ","
				<< q << ") is " << o << endl;
	}
}

void group_generators_domain::order_PO_minus(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+2
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, T, Two, plusone, minusone;
	int i;
	geometry::other_geometry::geometry_global Gg;


	Two.create(2);
	plusone.create(1);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m + 1));
	if (f_v) {
		cout << "group_generators_domain::order_PO_minus " << q << "^(" << m << "*"
				<< m + 1 << ") = " << Q << endl;
	}
	// now Q = q^{m(m+1)}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "group_generators_domain::order_PO_minus " << q << "^" << 2 * i
					<< " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}
	// now O = \prod_{i=1}^{m} (q^{2i}-1)

	R.create(q);
	D.power_int(R, m + 1);
	D.add(R, plusone, S);
	if (f_v) {
		cout << "group_generators_domain::order_PO_minus " << q << "^" << m + 1
				<< " + 1 = " << S << endl;
	}
	// now S = q^{m+1}-1

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	if (EVEN(q)) {
		D.mult(T, Two, o);
	}
	else {
		T.assign_to(o);
	}


	if (f_v) {
		cout << "group_generators_domain::order_PO_minus the order of PO^-" << "("
			<< Gg.dimension_given_Witt_index(-1, m) << ","
			<< q << ") is " << o << endl;
	}
}

void group_generators_domain::order_PO_parabolic(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+1
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, T, minusone;
	int i;
	geometry::other_geometry::geometry_global Gg;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * m);
	if (f_v) {
		cout << "group_generators_domain::order_PO_parabolic " << q << "^(" << m
				<< "^2" << ") = " << Q << endl;
	}
	// now Q = q^{m^2}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "group_generators_domain::order_PO_parabolic " << q << "^"
					<< 2 * i << " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}
	// now O = \prod_{i=1}^{m} (q^{2i}-1)


	D.mult(O, Q, o);


	if (f_v) {
		cout << "group_generators_domain::order_PO_parabolic the order of PO" << "("
			<< Gg.dimension_given_Witt_index(0, m) << ","
			<< q << ") is " << o << endl;
	}
}


void group_generators_domain::order_Pomega_plus(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, S1, T, minusone;
	int i, r;
	geometry::other_geometry::geometry_global Gg;


	if (f_v) {
		cout << "group_generators_domain::order_Pomega_plus" << endl;
	}
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << q << "^(" << m << "*" << m - 1 << ") = " << Q << endl;
	}
	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}

	R.create(q);
	D.power_int(R, m);
	D.add(R, minusone, S);
	if (f_v) {
		cout << q << "^" << m << " - 1 = " << S << endl;
	}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		S1.assign_to(S);
	}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		S1.assign_to(S);
	}

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	T.assign_to(o);


	if (f_v) {
		cout << "group_generators_domain::order_Pomega_plus "
				"the order of P\\Omega^1" << "("
			<< Gg.dimension_given_Witt_index(1, m) << ","
			<< q << ") is " << o << endl;
	}
}

void group_generators_domain::order_Pomega_minus(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = half the dimension,
// the dimension is n = 2m, the Witt index is m - 1
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, S1, T, minusone, plusone;
	int i, r;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::order_Pomega_minus m=" << m << " q=" << q << endl;
	}
	minusone.create(-1);
	plusone.create(1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << q << "^(" << m << "*" << m - 1 << ") = " << Q << endl;
	}
	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}

	R.create(q);
	D.power_int(R, m);
	D.add(R, plusone, S);
	if (f_v) {
		cout << q << "^" << m << " + 1 = " << S << endl;
	}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		if (f_v) {
			cout << "divide by 2" << endl;
		}
		S1.assign_to(S);
	}
	D.integral_division_by_int(S, 2, S1, r);
	if (r == 0) {
		if (f_v) {
			cout << "divide by 2" << endl;
		}
		S1.assign_to(S);
	}

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	T.assign_to(o);


	if (f_v) {
		cout << "group_generators_domain::order_Pomega_minus "
				"the order of P\\Omega^-1" << "("
			<< Gg.dimension_given_Witt_index(-1, m - 1) << ","
			<< q << ") is " << o << endl;
	}
}

void group_generators_domain::order_Pomega_parabolic(
		int m, int q,
		algebra::ring_theory::longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m + 1
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object O, Q, R, S, T, minusone;
	int i, r;
	geometry::other_geometry::geometry_global Gg;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * m);
	if (f_v) {
		cout << q << "^(" << m << "^2) = " << Q << endl;
	}
	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << q << "^" << 2 * i << " - 1 = " << S << endl;
		}
		D.mult(O, S, T);
		T.assign_to(O);
	}
	D.mult(O, Q, T);
	if (EVEN(q)) {
		T.assign_to(o);
	}
	else {
		D.integral_division_by_int(T, 2, o, r);
	}
	if (f_v) {
		cout << "group_generators_domain::order_Pomega_parabolic the order of P\\Omega" << "("
			<< Gg.dimension_given_Witt_index(0, m) << ","
			<< q << ") is " << o << endl;
	}
}

int group_generators_domain::index_POmega_in_PO(
		int epsilon, int m, int q, int verbose_level)
{
	if (epsilon == 0) {
		if (EVEN(q)) {
			return 1;
			}
		else {
			return 2;
			}
		}
	if (epsilon == 1) {
		if (EVEN(q)) {
			return 2;
			}
		else {
			if (DOUBLYEVEN(q - 1)) {
				return 4;
				}
			else {
				if (EVEN(m)) {
					return 4;
					}
				else {
					return 2;
					}
				}
			}
		}
	if (epsilon == -1) {
		if (EVEN(q)) {
			return 2;
			}
		else {
			if (DOUBLYEVEN(q - 1)) {
				return 2;
				}
			else {
				if (EVEN(m + 1)) {
					return 2;
					}
				else {
					return 4;
					}
				}
			}
		}
#if 0
	if (epsilon == -1) {
		cout << "index_POmega_in_PO epsilon = -1 not "
				"yet implemented, returning 1" << endl;
		return 1;
		exit(1);
		}
#endif
	cout << "index_POmega_in_PO epsilon not recognized, "
			"epsilon=" << epsilon << endl;
	exit(1);
}

void group_generators_domain::diagonal_orbit_perm(
		int n, field_theory::finite_field *F,
		long int *orbit, long int *orbit_inv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "group_generators_domain::diagonal_orbit_perm" << endl;
	}
	geometry::other_geometry::geometry_global Gg;
	long int l, ll;
	long int a, b, c;
	long int i, j;
	int *v;

	l = Gg.nb_PG_elements(n - 1, F->q);
	ll = Gg.nb_AG_elements(n - 1, F->q - 1);

	v = NEW_int(n + 1);
	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
	}
	for (i = 0; i < ll; i++) {
		v[0] = 1;
		Gg.AG_element_unrank(F->q - 1, v + 1, 1, n - 1, i);
		for (j = 1; j < n; j++) {
			v[j]++;
		}
		if (f_vv) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
			}
		}
		if (f_v) {
			cout << "group_generators_domain::diagonal_orbit_perm before PG_element_rank_modified_lint i=" << i << " v=";
			Int_vec_print(cout, v, n);
			cout << endl;
		}
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, n, a, verbose_level - 2);
		if (f_vv) {
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
		cout << "group_generators_domain::diagonal_orbit_perm done" << endl;
	}
}

void group_generators_domain::frobenius_orbit_perm(
		int n, field_theory::finite_field *F,
	long int *orbit, long int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;// (verbose_level >= 1);
	int *v = NEW_int(n);
	geometry::other_geometry::geometry_global Gg;
	long int l;
	long int ll;
	long int a, b, c;
	long int i, j;

	if (f_v) {
		cout << "group_generators_domain::frobenius_orbit_perm n=" << n
				<< " (vector space dimension)" << endl;
	}
	l = Gg.nb_PG_elements(n - 1, F->q);
	ll = F->e;
	if (f_v) {
		cout << "group_generators_domain::frobenius_orbit_perm l=" << l << endl;
	}
	if (F->e == 1) {
		cout << "group_generators_domain::frobenius_orbit_perm GFq.e == 1" << endl;
		exit(1);
	}
	//cout << "l = " << l << endl;
	for (i = 0; i < l; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
	}
	if (f_v) {
		cout << "before PG_element_unrank_modified("
				<< n + F->p << ")" << endl;
	}
	F->Projective_space_basic->PG_element_unrank_modified(v, 1, n, n + F->p);
	if (f_v) {
		cout << "after PG_element_unrank_modified("
				<< n + F->p << ")" << endl;
	}
	for (i = 0; i < ll; i++) {
		if (f_vv) {
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				cout << v[j] << " ";
			}
		}
		F->Projective_space_basic->PG_element_rank_modified_lint(v, 1, n, a, verbose_level - 2);
		if (f_vv) {
			cout << " : " << a << endl;
		}
		b = orbit_inv[a];
		c = orbit[i];
		orbit[i] = a;
		orbit[b] = c;
		orbit_inv[a] = i;
		orbit_inv[c] = b;
		F->Projective_space_basic->PG_element_apply_frobenius(n, v, 1, verbose_level - 2);
	}
	FREE_int(v);
	if (f_v) {
		cout << "group_generators_domain::frobenius_orbit_perm done" << endl;
	}
}

void group_generators_domain::projective_matrix_group_base_and_orbits(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	long int **orbit, long int **orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	geometry::other_geometry::geometry_global Gg;


	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
				"base_len=" << base_len << endl;
	}
	for (i = 0; i < base_len; i++) {
		base[i] = i;
	}
	for (i = 0; i < base_len; i++) {
		transversal_length[i] = i;
	}
	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
				"transversal_length: ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
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
				Gg.nb_PG_elements_not_in_subspace(n - 1, i - 1, F->q);
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
					"before PG_element_modified_not_in_subspace_perm" << endl;
		}
		PG_element_modified_not_in_subspace_perm(F, n - 1, i - 1,
			orbit[i], orbit_inv[i], 0);

		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
					"after PG_element_modified_not_in_subspace_perm" << endl;
		}

		if (false) {
			Lint_vec_print(cout, orbit[i], degree);
			cout << endl;
			Lint_vec_print(cout, orbit_inv[i], degree);
			cout << endl;
		}
	}
	if (F->q > 2) {
		transversal_length[i] = Gg.nb_AG_elements(n - 1, F->q - 1);
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"before diagonal_orbit_perm" << endl;
		}
		diagonal_orbit_perm(n, F, orbit[i], orbit_inv[i], 0 /* verbose_level - 2*/);

		if (f_vv) {
			cout << "projective_matrix_group_base_and_orbits "
					"after diagonal_orbit_perm" << endl;
		}

		if (false) {
			Lint_vec_print(cout, orbit[i], degree);
			cout << endl;
			Lint_vec_print(cout, orbit_inv[i], degree);
			cout << endl;
		}
		i++;
	}
	if (f_semilinear) {
		transversal_length[i] = F->e;
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
		if (f_vv) {
			cout << "finite_field::projective_matrix_group_base_and_orbits "
					"before frobenius_orbit_perm" << endl;
		}
		frobenius_orbit_perm(n, F,
				orbit[i], orbit_inv[i], 0 /*verbose_level - 2*/);

		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
					"after frobenius_orbit_perm" << endl;
		}

		if (false) {
			Lint_vec_print(cout, orbit[i], degree);
			cout << endl;
			Lint_vec_print(cout, orbit_inv[i], degree);
			cout << endl;
		}
		i++;
	}
	if (i != base_len) {
		cout << "group_generators_domain::projective_matrix_group_base_and_orbits "
				"i != base_len" << endl;
		cout << "i=" << i << endl;
		cout << "base_len=" << base_len << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "group_generators_domain::projective_matrix_group_base_and_orbits base: ";
		Lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "projective_matrix_group_base_and_orbits "
				"transversal_length: ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_orbits done" << endl;
	}
}

void group_generators_domain::projective_matrix_group_base_and_transversal_length(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	geometry::other_geometry::geometry_global Gg;


	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length" << endl;
	}
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
				Gg.nb_PG_elements_not_in_subspace(n - 1, i - 1, F->q);
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length "
					"transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
	}
	if (F->q > 2) {
		transversal_length[i] = Gg.nb_AG_elements(n - 1, F->q - 1);
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length: "
					"diagonal transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
		i++;
	}
	if (f_semilinear) {
		transversal_length[i] = F->e;
		if (f_vv) {
			cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length: "
					"frobenius transversal " << i << " of length "
					<< transversal_length[i] << endl;
		}
		i++;
	}
	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length base: ";
		Lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "finite_field::projective_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group_generators_domain::projective_matrix_group_base_and_transversal_length done" << endl;
	}
}

void group_generators_domain::affine_matrix_group_base_and_transversal_length(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, c;
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "group_generators_domain::affine_matrix_group_base_and_transversal_length" << endl;
	}
	c = 0;
	base[c] = 0;
	transversal_length[c] = NT.i_power_j(F->q, n);
	c++;
	for (i = 0; i < n; i++) {
		base[c] = NT.i_power_j_lint(F->q, i);
		transversal_length[c] = NT.i_power_j_lint(F->q, n) - NT.i_power_j_lint(F->q, i);
		c++;
	}
	if (f_semilinear) {
		if (n > 1) {
			base[c] = F->q + F->p; // ToDo: this does not work when n = 1
		}
		else {
			base[c] = F->p;
		}
		transversal_length[c] = F->e;
		c++;
	}
	if (c != base_len) {
		cout << "group_generators_domain::affine_matrix_group_base_and_transversal_length "
				"c != base_len" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "group_generators_domain::affine_matrix_group_base_and_transversal_length base: ";
		Lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "finite_field::affine_matrix_group_base_and_transversal_length "
				"transversal_length: ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group_generators_domain::affine_matrix_group_base_and_transversal_length done" << endl;
	}
}


void group_generators_domain::general_linear_matrix_group_base_and_transversal_length(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int base_len, int degree,
	long int *base, int *transversal_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, c;
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "group_generators_domain::general_linear_matrix_group_base_and_transversal_length" << endl;
	}
	c = 0;
	for (i = 0; i < n; i++) {
		base[c] = NT.i_power_j_lint(F->q, i);
		transversal_length[c] = NT.i_power_j_lint(F->q, n) - NT.i_power_j_lint(F->q, i);
		c++;
	}
	if (f_semilinear) {
		base[c] = F->q + F->p;
		transversal_length[c] = F->e;
		c++;
	}
	if (c != base_len) {
		cout << "group_generators_domain::general_linear_matrix_group_base_and_"
				"transversal_length c != base_len" << endl;
		cout << "c=" << c << endl;
		cout << "base_len=" << base_len << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "group_generators_domain::general_linear_matrix_group_base_and_"
				"transversal_length base: ";
		Lint_vec_print(cout, base, base_len);
		cout << endl;
		cout << "group_generators_domain::general_linear_matrix_group_base_and_"
				"transversal_length transversal_length: ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group_generators_domain::general_linear_matrix_group_base_and_transversal_length done" << endl;
	}
}


void group_generators_domain::strong_generators_for_projective_linear_group(
	int n, field_theory::finite_field *F,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	int *M;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_projective_linear_group" << endl;
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
		F->Linear_algebra->identity_matrix(M, n);
		M[n * n] = 1;
		Int_vec_copy(M, data + cur * size, size);
		if (f_v) {
			cout << "group_generators_domain::strong_generators_for_projective_linear_group generator " << cur << endl;
			Int_matrix_print(data + cur * n * n, 1, n * n);
		}
		cur++;
	}


	// the primitive elements on the diagonal:
	if (F->q > 2) {
		for (h = 0; h < n - 1; h++) {
			if (f_vv) {
				cout << "generators for primitive elements "
						"on the diagonal:" << endl;
			}
			F->Linear_algebra->identity_matrix(M, n);
			M[h * n + h] = F->primitive_root();
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
			if (f_v) {
				cout << "group_generators_domain::strong_generators_for_projective_linear_group generator " << cur << endl;
				Int_matrix_print(data + cur * n * n, 1, n * n);
			}
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
			F->Linear_algebra->identity_matrix(M, n);
			M[(n - 1) * n + h] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
			if (f_v) {
				cout << "group_generators_domain::strong_generators_for_projective_linear_group generator " << cur << endl;
				Int_matrix_print(data + cur * n * n, 1, n * n);
			}
			cur++;
		}
	}

	// the swaps along the diagonal:
	for (h = n - 2; h >= 0; h--) {
		if (f_vv) {
			cout << "generators for swaps along the diagonal:" << endl;
		}
		F->Linear_algebra->identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
		}
		Int_vec_copy(M, data + cur * size, size);
		if (f_v) {
			cout << "group_generators_domain::strong_generators_for_projective_linear_group generator " << cur << endl;
			Int_matrix_print(data + cur * n * n, 1, n * n);
		}
		cur++;
	}

	if (cur != nb_gens) {
		cout << "group_generators_domain::strong_generators_for_projective_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_projective_linear_group strong generators are:" << endl;
		Int_matrix_print(data, cur, size);
	}

	FREE_int(M);
	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_projective_linear_group "
				"done" << endl;
	}
}


void group_generators_domain::strong_generators_for_affine_linear_group(
	int n, field_theory::finite_field *F,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_affine_linear_group" << endl;
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
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);
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
			Int_vec_zero(data + cur * size, size);
			F->Linear_algebra->identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = NT.i_power_j(F->p, u);
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
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);

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
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);
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
			Int_vec_zero(data + cur * size, size);
			F->Linear_algebra->identity_matrix(data + cur * size, n);

			data[cur * size + n * n + h] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				data[cur * size + n * n + n] = 0;
			}
			cur++;
		} // next u
	} // next h

	if (cur != nb_gens) {
		cout << "group_generators_domain::strong_generators_for_affine_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_affine_linear_group strong generators are:" << endl;
		Int_matrix_print(data, cur, size);
	}

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_affine_linear_group done" << endl;
	}
}

void group_generators_domain::strong_generators_for_general_linear_group(
	int n, field_theory::finite_field *F,
	int f_semilinear,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, cur;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_general_linear_group" << endl;
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
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);
		data[cur * size + n * n] = 1;
		if (f_v) {
			cout << "group_generators_domain::strong_generators_for_general_linear_group generator " << cur << endl;
			Int_matrix_print(data + cur * n * n, 1, n * n);
		}
		cur++;
	}

	// the entries in the last row:
	for (h = 0; h < n - 1; h++) {
		if (f_vv) {
			cout << "generators for entries in the last row "
					"(e=" << F->e << "):" << endl;
		}
		for (u = 0; u < F->e; u++) {
			Int_vec_zero(data + cur * size, size);
			F->Linear_algebra->identity_matrix(data + cur * size, n);

			data[cur * size + (n - 1) * n + h] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				data[cur * size + n * n] = 0;
			}
			if (f_v) {
				cout << "group_generators_domain::strong_generators_for_general_linear_group generator " << cur << endl;
				Int_matrix_print(data + cur * n * n, 1, n * n);
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
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);

		data[cur * size + h * n + h] = F->primitive_root();
		if (f_semilinear) {
			data[cur * size + n * n] = 0;
		}
		if (f_v) {
			cout << "group_generators_domain::strong_generators_for_general_linear_group generator " << cur << endl;
			Int_matrix_print(data + cur * n * n, 1, n * n);
		}
		cur++;
	} // if


	// the swaps along the diagonal:
	for (h = n - 2; h >= 0; h--) {
		if (f_vv) {
			cout << "generators for swaps along the diagonal:" << endl;
		}
		Int_vec_zero(data + cur * size, size);
		F->Linear_algebra->identity_matrix(data + cur * size, n);
		data[cur * size + h * n + h] = 0;
		data[cur * size + h * n + h + 1] = 1;
		data[cur * size + (h + 1) * n + h] = 1;
		data[cur * size + (h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			data[cur * size + n * n] = 0;
		}
		if (f_v) {
			cout << "group_generators_domain::strong_generators_for_general_linear_group generator " << cur << endl;
			Int_matrix_print(data + cur * n * n, 1, n * n);
		}
		cur++;
	} // next h

	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_general_linear_group strong generators are:" << endl;
		Int_matrix_print(data, cur, size);
	}

	if (cur != nb_gens) {
		cout << "group_generators_domain::strong_generators_for_general_linear_group "
				"cur != nb_gens" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "group_generators_domain::strong_generators_for_general_linear_group done" << endl;
	}
}

void group_generators_domain::generators_for_parabolic_subgroup(
	int n, field_theory::finite_field *F,
	int f_semilinear, int k,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, g, u, cur;
	int *M;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::generators_for_parabolic_subgroup" << endl;
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
	if (F->q > 2) {
		nb_gens += n - 1;
	}


	// count the generators with entries in row k:
	for (h = 0; h < k - 1; h++) {
		for (u = 0; u < F->e; u++) {
			nb_gens++;
		}
	}
	// count the generators with entries in row n:
	for (h = k; h < n - 1; h++) {
		for (u = 0; u < F->e; u++) {
			nb_gens++;
		}
	}

	// count the generators with entries in the lower left block:
	nb_gens += k * (n - k) * F->e;

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
		F->Linear_algebra->identity_matrix(M, n);
		M[n * n] = 1;
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}


	// the primitive elements on the diagonal:
	if (f_vv) {
		cout << "generators for primitive elements "
				"on the diagonal, cur=" << cur << endl;
	}
	if (F->q > 2) {
		for (h = 0; h < n - 1; h++) {
			F->Linear_algebra->identity_matrix(M, n);
			M[h * n + h] = F->primitive_root();
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
			cur++;
		}
	}

	// the entries in the row k:
	if (f_vv) {
		cout << "generators for the entries in the last row "
				"of a diagonal block, cur=" << cur << endl;
	}
	for (h = 0; h < k - 1; h++) {
		for (u = 0; u < F->e; u++) {
			F->Linear_algebra->identity_matrix(M, n);
			M[(k - 1) * n + h] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
			cur++;
		}
	}

	// the entries in the row n:
	for (h = k; h < n - 1; h++) {
		for (u = 0; u < F->e; u++) {
			F->Linear_algebra->identity_matrix(M, n);
			M[(n - 1) * n + h] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
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
				F->Linear_algebra->identity_matrix(M, n);
				M[g * n + h] = NT.i_power_j(F->p, u);
				if (f_semilinear) {
					M[n * n] = 0;
				}
				Int_vec_copy(M, data + cur * size, size);
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
		F->Linear_algebra->identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
		}
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}
	for (h = k - 2; h >= 0; h--) {
		F->Linear_algebra->identity_matrix(M, n);
		M[h * n + h] = 0;
		M[h * n + h + 1] = 1;
		M[(h + 1) * n + h] = 1;
		M[(h + 1) * n + h + 1] = 0;
		if (f_semilinear) {
			M[n * n] = 0;
		}
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}

	if (cur != nb_gens) {
		cout << "group_generators_domain::generators_for_parabolic_subgroup "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
	}

	FREE_int(M);
	if (f_v) {
		cout << "group_generators_domain::generators_for_parabolic_subgroup done" << endl;
	}
}

void group_generators_domain::generators_for_stabilizer_of_three_collinear_points_in_PGL4(
	int f_semilinear, field_theory::finite_field *F,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, i, j;
	int *M;
	int n = 4;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::generators_for_stabilizer_of_three_collinear_"
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
		F->Linear_algebra->identity_matrix(M, n);
		M[n * n] = 1;
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
	}
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = F->negate(1);
	M[1 * 4 + 1] = F->negate(1);
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = F->primitive_root();
	M[0 * 4 + 1] = 0;
	M[1 * 4 + 0] = 0;
	M[1 * 4 + 1] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars in bottom right:
	F->Linear_algebra->identity_matrix(M, n);
	M[3 * 4 + 3] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block:
	for (i = 2; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (u = 0; u < F->e; u++) {
				F->Linear_algebra->identity_matrix(M, n);
				M[i * n + j] = NT.i_power_j(F->p, u);
				if (f_semilinear) {
					M[n * n] = 0;
				}
				Int_vec_copy(M, data + cur * size, size);
				cur++;
			}
		}
	}

	// swaps lower right:
	F->Linear_algebra->identity_matrix(M, n);
	M[2 * 4 + 2] = 0;
	M[2 * 4 + 3] = 1;
	M[3 * 4 + 2] = 1;
	M[3 * 4 + 3] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// PGL2 in lower right, bottom row
	for (u = 0; u < F->e; u++) {
		F->Linear_algebra->identity_matrix(M, n);
		M[3 * n + 2] = NT.i_power_j(F->p, u);
		if (f_semilinear) {
			M[n * n] = 0;
		}
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}




	if (cur != nb_gens) {
		cout << "group_generators_domain::generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
	}

	FREE_int(M);
	if (f_v) {
		cout << "group_generators_domain::generators_for_stabilizer_of_three_"
				"collinear_points_in_PGL4 done" << endl;
	}
}


void group_generators_domain::generators_for_stabilizer_of_triangle_in_PGL4(
	int f_semilinear, field_theory::finite_field *F,
	int *&data, int &size, int &nb_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int u, cur, j;
	int *M;
	int n = 4;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_generators_domain::generators_for_stabilizer_of_triangle_in_PGL4" << endl;
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
		F->Linear_algebra->identity_matrix(M, n);
		M[n * n] = 1;
		Int_vec_copy(M, data + cur * size, size);
		cur++;
	}

	// Sym_3 in top left block:
	if (f_vv) {
		cout << "generators for Sym_3 in top left block, "
				"cur=" << cur << endl;
	}
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[0 * 4 + 1] = 1;
	M[1 * 4 + 0] = 1;
	M[1 * 4 + 1] = 0;
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = 0;
	M[1 * 4 + 1] = 0;
	M[2 * 4 + 2] = 0;
	M[0 * 4 + 2] = 1;
	M[1 * 4 + 0] = 1;
	M[2 * 4 + 1] = 1;
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars in top left block:
	F->Linear_algebra->identity_matrix(M, n);
	M[0 * 4 + 0] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;
	F->Linear_algebra->identity_matrix(M, n);
	M[1 * 4 + 1] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;
	F->Linear_algebra->identity_matrix(M, n);
	M[2 * 4 + 2] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// scalars bottom right
	F->Linear_algebra->identity_matrix(M, n);
	M[3 * 4 + 3] = F->primitive_root();
	if (f_semilinear) {
		M[n * n] = 0;
	}
	Int_vec_copy(M, data + cur * size, size);
	cur++;

	// lower left block
	for (j = 0; j < 3; j++) {
		for (u = 0; u < F->e; u++) {
			F->Linear_algebra->identity_matrix(M, n);
			M[3 * n + j] = NT.i_power_j(F->p, u);
			if (f_semilinear) {
				M[n * n] = 0;
			}
			Int_vec_copy(M, data + cur * size, size);
			cur++;
		}
	}


	if (cur != nb_gens) {
		cout << "group_generators_domain::generators_for_stabilizer_of_triangle_in_PGL4 "
				"cur != nb_gens" << endl;
		cout << "cur = " << cur << endl;
		cout << "nb_gens = " << nb_gens << endl;
		exit(1);
	}

	FREE_int(M);
	if (f_v) {
		cout << "group_generators_domain::generators_for_stabilizer_of_triangle_in_PGL4 done" << endl;
	}
}

void group_generators_domain::builtin_transversal_rep_GLnq(
		int *A,
		int n,
		field_theory::finite_field *F,
		int f_semilinear, int i, int j,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int transversal_length;
	int ii, jj, i0, a;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::builtin_transversal_rep_GLnq  "
				"GL(" << n << "," << F->q << ") i = " << i
				<< " j = " << j << endl;
	}

	// make the n x n identity matrix:
	for (ii = 0; ii < n * n; ii++) {
		A[ii] = 0;
	}
	for (ii = 0; ii < i; ii++) {
		A[ii * n + ii] = 1;
	}
	if (f_semilinear) {
		A[n * n] = 0;
	}

	if ((i == n + 1 && F->q > 2) || (i == n && F->q == 2)) {
		if (!f_semilinear) {
			cout << "group_generators_domain::builtin_transversal_rep_GLnq "
					"must be semilinear to access transversal " << n << endl;
			exit(1);
		}
		A[n * n] = j;
	}
	else if (i == n && F->q > 2) {
		transversal_length = Gg.nb_AG_elements(n - 1, F->q - 1);
		if (j >= transversal_length) {
			cout << "group_generators_domain::builtin_transversal_rep_GLnq "
					"j = " << j << " >= transversal_length = "
					<< transversal_length << endl;
			exit(1);
		}
		int *v = NEW_int(n);
		Gg.AG_element_unrank(F->q - 1, v, 1, n - 1, j);
		A[0] = 1;
		for (jj = 0; jj < n - 1; jj++) {
			A[(jj + 1) * n + (jj + 1)] = v[jj] + 1;
		}
		FREE_int(v);
	}
	else {
		if (i == 0) {
			F->Projective_space_basic->PG_element_unrank_modified(A + i, n, n, j);
		}
		else {
			F->Projective_space_basic->PG_element_unrank_modified_not_in_subspace(
					A + i, n, n, i - 1, j);
		}
		i0 = -1;
		for (ii = 0; ii < n; ii++) {
			a = A[ii * n + i];
			if (ii >= i && i0 == -1 && a != 0) {
				i0 = ii;
			}
		}
		if (f_vv) {
			cout << "i0 = " << i0 << endl;
		}
		for (jj = i; jj < i0; jj++) {
			A[jj * n + jj + 1] = 1;
		}
		for (jj = i0 + 1; jj < n; jj++) {
			A[jj * n + jj] = 1;
		}
		//int_matrix_transpose(n, A);
		F->Linear_algebra->transpose_matrix_in_place(A, n);
	}

	if (f_vv) {
		cout << "group_generators_domain::transversal_rep_GLnq[" << i << "][" << j << "] = \n";
		Int_vec_print_integer_matrix(cout, A, n, n);
	}
}

void group_generators_domain::affine_translation(
		int n, field_theory::finite_field *F,
		int coordinate_idx, int field_base_idx, int *perm,
		int verbose_level)
// perm points to q^n ints
// field_base_idx is the base element whose translation
// we compute, 0 \le field_base_idx < e
// coordinate_idx is the coordinate in which we shift,
// 0 \le coordinate_idx < n
{
	int f_v = (verbose_level >= 1);
	long int i, j, l, a;
	int *v;
	number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::affine_translation "
				"coordinate_idx=" << coordinate_idx
				<< " field_base_idx=" << field_base_idx << endl;
	}
	v = NEW_int(n);
	l = Gg.nb_AG_elements(n, F->q);
	a = NT.i_power_j(F->p, field_base_idx);
	for (i = 0; i < l; i++) {
		Gg.AG_element_unrank(F->q, v, 1, l, i);
		v[coordinate_idx] = F->add(v[coordinate_idx], a);
		j = Gg.AG_element_rank(F->q, v, 1, l);
		perm[i] = j;
	}
	FREE_int(v);
}

void group_generators_domain::affine_multiplication(
		int n, field_theory::finite_field *F,
		int multiplication_order, int *perm,
		int verbose_level)
// perm points to q^n ints
// compute the diagonal multiplication by alpha, i.e.
// the multiplication by alpha of each component
{
	int f_v = (verbose_level >= 1);
	long int i, j, l, k;
	int alpha_power, a;
	int *v;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::affine_multiplication" << endl;
	}
	v = NEW_int(n);
	alpha_power = (F->q - 1) / multiplication_order;
	if (alpha_power * multiplication_order != F->q - 1) {
		cout << "group_generators_domain::affine_multiplication: "
				"multiplication_order does not divide F->q - 1" << endl;
		exit(1);
	}
	a = F->power(F->alpha, alpha_power);
	l = Gg.nb_AG_elements(n, F->q);
	for (i = 0; i < l; i++) {
		Gg.AG_element_unrank(F->q, v, 1, l, i);
		for (k = 0; k < n; k++) {
			v[k] = F->mult(v[k], a);
		}
		j = Gg.AG_element_rank(F->q, v, 1, l);
		perm[i] = j;
	}
	FREE_int(v);
}

void group_generators_domain::affine_frobenius(
		int n, field_theory::finite_field *F,
		int k, int *perm,
		int verbose_level)
// perm points to q^n ints
// compute the diagonal action of the Frobenius automorphism
// to the power k, i.e.,
// raises each component to the p^k-th power
{
	int f_v = (verbose_level >= 1);
	long int i, j, l, u;
	int *v;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::affine_frobenius" << endl;
	}
	v = NEW_int(n);
	l = Gg.nb_AG_elements(n, F->q);
	for (i = 0; i < l; i++) {
		Gg.AG_element_unrank(F->q, v, 1, l, i);
		for (u = 0; u < n; u++) {
			v[u] = F->frobenius_power(v[u], k, verbose_level - 1);
		}
		j = Gg.AG_element_rank(F->q, v, 1, l);
		perm[i] = j;
	}
	FREE_int(v);
}


int group_generators_domain::all_affine_translations_nb_gens(
		int n, field_theory::finite_field *F)
{
	int nb_gens;

	nb_gens = F->e * n;
	return nb_gens;
}

void group_generators_domain::all_affine_translations(
		int n, field_theory::finite_field *F, int *gens)
{
	int i, j, k = 0;
	int degree;
	geometry::other_geometry::geometry_global Gg;

	degree = Gg.nb_AG_elements(n, F->q);

	for (i = 0; i < n; i++) {
		for (j = 0; j < F->e; j++, k++) {
			affine_translation(n, F, i, j, gens + k * degree,
					0 /* verbose_level */);
		}
	}
}

void group_generators_domain::affine_generators(
		int n, field_theory::finite_field *F,
	int f_translations,
	int f_semilinear, int frobenius_power,
	int f_multiplication, int multiplication_order,
	int &nb_gens, int &degree, int *&gens,
	int &base_len, long int *&the_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k, h;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "group_generators_domain::affine_generators" << endl;
	}
	degree = Gg.nb_AG_elements(n, F->q);
	nb_gens = 0;
	base_len = 0;
	if (f_translations) {
		nb_gens += all_affine_translations_nb_gens(n, F);
		base_len++;
	}
	if (f_multiplication) {
		nb_gens++;
		base_len++;
	}
	if (f_semilinear) {
		nb_gens++;
		base_len++;
	}

	gens = NEW_int(nb_gens * degree);
	the_base = NEW_lint(base_len);
	k = 0;
	h = 0;
	if (f_translations) {
		all_affine_translations(n, F, gens);
		k += all_affine_translations_nb_gens(n, F);
		the_base[h++] = 0;
	}
	if (f_multiplication) {
		affine_multiplication(n, F, multiplication_order,
				gens + k * degree, 0 /* verbose_level */);
		k++;
		the_base[h++] = 1;
	}
	if (f_semilinear) {
		affine_frobenius(n, F, frobenius_power, gens + k * degree,
				0 /* verbose_level */);
		k++;
		the_base[h++] = F->p;
	}
	if (f_v) {
		cout << "group_generators_domain::affine_generators done" << endl;
	}
}

void group_generators_domain::PG_element_modified_not_in_subspace_perm(
		field_theory::finite_field *F,
		int n, int m,
	long int *orbit, long int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_generators_domain::PG_element_modified_not_in_subspace_perm" << endl;
	}

	int *v = NEW_int(n + 1);
	geometry::other_geometry::geometry_global Gg;
	long int l = Gg.nb_PG_elements(n, F->q);
	long int ll = Gg.nb_PG_elements_not_in_subspace(n, m, F->q);
	long int i, j1 = 0, j2 = ll, f_in, j;

	for (i = 0; i < l; i++) {
		F->Projective_space_basic->PG_element_unrank_modified_lint(v, 1, n + 1, i);
		f_in = Gg.PG_element_modified_is_in_subspace(n, m, v);
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n + 1; j++) {
				cout << v[j] << " ";
			}
		}
		if (f_in) {
			if (f_v) {
				cout << " is in the subspace" << endl;
			}
			orbit[j2] = i;
			orbit_inv[i] = j2;
			j2++;
		}
		else {
			if (f_v) {
				cout << " is not in the subspace" << endl;
			}
			orbit[j1] = i;
			orbit_inv[i] = j1;
			j1++;
		}
	}
	if (j1 != ll) {
		cout << "j1 != ll" << endl;
		exit(1);
	}
	if (j2 != l) {
		cout << "j2 != l" << endl;
		exit(1);
	}
	FREE_int(v);
	if (f_v) {
		cout << "group_generators_domain::PG_element_modified_not_in_subspace_perm done" << endl;
	}
}




}}}}



