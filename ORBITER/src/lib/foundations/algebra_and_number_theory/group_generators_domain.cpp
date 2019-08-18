// group_generators_domain.cpp
//
// Anton Betten
//
// moved here from projective.cpp: September 4, 2016




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



group_generators_domain::group_generators_domain()
{

}

group_generators_domain::~group_generators_domain()
{

}


void group_generators_domain::generators_symmetric_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	combinatorics_domain Combi;
	
	if (deg <= 1) {
		nb_perms = 0;
		perms = NULL;
		return;
		}
	nb_perms = deg - 1;
	perms = NEW_int(nb_perms * deg);
	for (i = 0; i < nb_perms; i++) {
		Combi.perm_identity(perms + i * deg, deg);
		perms[i * deg + i] = i + 1;
		perms[i * deg + i + 1] = i;
		}
	if (f_v) {
		cout << "generators for symmetric group of degree "
				<< deg << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void group_generators_domain::generators_cyclic_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	combinatorics_domain Combi;
	
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
			Combi.perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void group_generators_domain::generators_dihedral_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	combinatorics_domain Combi;
	
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
			Combi.perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void group_generators_domain::generators_dihedral_involution(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j, d2;
	combinatorics_domain Combi;
	
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
			Combi.perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
		}
}

void group_generators_domain::generators_identity_group(int deg,
		int &nb_perms, int *&perms, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i = 0, j;
	combinatorics_domain Combi;
	
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
			Combi.perm_print(cout, perms + i * deg, deg);
			cout << endl;
			}
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
	combinatorics_domain Combi;

	if (f_v) {
		cout << "group_generators_domain::generators_"
				"Hall_reflection" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = 1;
	perms = NEW_int(nb_perms * degree);
	Combi.perm_identity(perms, degree);
	for (i = 0; i < nb_pairs; i++) {
		perms[2 * i] = 2 * i + 1;
		perms[2 * i + 1] = 2 * i;
	}
	if (f_v) {
		cout << "group_generators_domain::generators_"
				"Hall_reflection "
				"generators for the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < 1; i++) {
			Combi.perm_print(cout, perms + i * degree, degree);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "group_generators_domain::generators_"
				"Hall_reflection done" << endl;
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
	combinatorics_domain Combi;

	if (f_v) {
		cout << "group_generators_domain::generators_Hall_"
				"reflection_normalizer_group" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = nb_pairs + (nb_pairs - 1);
	perms = NEW_int(nb_perms * degree);
	h = 0;
	for (i = 0; i < nb_pairs; i++, h++) {
		Combi.perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * i + 1;
		perms[h * degree + 2 * i + 1] = 2 * i;
	}
	for (i = 0; i < nb_pairs - 1; i++, h++) {
		Combi.perm_identity(perms + h * degree, degree);
		perms[h * degree + 2 * i] = 2 * (i + 1);
		perms[h * degree + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[h * degree + 2 * (i + 1)] = 2 * i;
		perms[h * degree + 2 * (i + 1) + 1] = 2 * i + 1;
		}
	if (h != nb_perms) {
		cout << "group_generators_domain::generators_Hall_"
				"reflection_normalizer_group "
				"h != nb_perms" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_"
				"reflection_normalizer_group "
				"generators for normalizer of the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms; i++) {
			Combi.perm_print(cout, perms + i * degree, degree);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "group_generators_domain::generators_Hall_"
				"reflection_normalizer_group done" << endl;
	}
}

void group_generators_domain::order_Hall_reflection_normalizer_factorized(
		int nb_pairs,
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
		cout << "group_generators_domain:order_Hall_"
				"reflection_normalizer_factorized "
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
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "group_generators_domain::generators_Bn_group" << endl;
		}
	deg = 2 * n;
	nb_perms = n - 1 + n;
	perms = NEW_int(nb_perms * deg);
	j = 0;
	for (i = 0; i < n - 1; i++, j++) {
		Combi.perm_identity(perms + j * deg, deg);
		perms[j * deg + 2 * i] = 2 * (i + 1);
		perms[j * deg + 2 * i + 1] = 2 * (i + 1) + 1;
		perms[j * deg + 2 * (i + 1)] = 2 * i;
		perms[j * deg + 2 * (i + 1) + 1] = 2 * i + 1;
		}
	for (i = 0; i < n; i++, j++) {
		Combi.perm_identity(perms + j * deg, deg);
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
			Combi.perm_print(cout, perms + i * deg, deg);
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
	combinatorics_domain Combi;
	
	deg3 = deg1 * deg2;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	id1 = NEW_int(deg1);
	id2 = NEW_int(deg2);
	Combi.perm_identity(id1, deg1);
	Combi.perm_identity(id2, deg2);
	
	for (i = 0; i < nb_perms1; i++) {
		Combi.perm_direct_product(deg1, deg2,
				perms1 + i * deg1, id2, perms3 + k * deg3);
		k++;
		}
	for (i = 0; i < nb_perms2; i++) {
		Combi.perm_direct_product(deg1, deg2, id1,
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
			Combi.perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
			}
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
	combinatorics_domain Combi;
	
	if (deg1 != deg2) {
		cout << "group_generators_domain::generators_concatenate:"
				" deg1 != deg2" << endl;
		exit(1);
		}
	deg3 = deg1;
	nb_perms3 = nb_perms1 + nb_perms2;
	perms3 = NEW_int(nb_perms3 * deg3);
	
	k = 0;
	for (i = 0; i < nb_perms1; i++) {
		Combi.perm_move(perms1 + i * deg1, perms3 + k * deg3, deg3);
		k++;
		}
	for (i = 0; i < nb_perms2; i++) {
		Combi.perm_move(perms2 + i * deg1, perms3 + k * deg3, deg3);
		k++;
		}
	if (f_v) {
		cout << "generators concatenated" << endl;
		}
	if (f_vv) {
		for (i = 0; i < nb_perms3; i++) {
			Combi.perm_print(cout, perms3 + i * deg3, deg3);
			cout << endl;
			}
		}
}


int group_generators_domain::matrix_group_base_len_projective_group(
		int n, int q,
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
		cout << "group_generators_domain::matrix_group_base_len_projective_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
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
		cout << "group_generators_domain::matrix_group_"
				"base_len_affine_group: "
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
		cout << "group_generators_domain::matrix_group_"
				"base_len_general_linear_group: "
				"n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear
				<< " base_len = " << base_len << endl;
		}
	return base_len;
}

void group_generators_domain::order_POmega_epsilon(
		int epsilon, int k, int q,
		longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int w, m;
	geometry_global Gg;

	w = Gg.Witt_index(epsilon, k);
	if (epsilon == -1) {
		m = w + 1;
		}
	else {
		m = w;
		}
	order_Pomega(epsilon, m, q, go, verbose_level);
	cout << "order_POmega_epsilon  epsilon=" << epsilon
			<< " k=" << k << " q=" << q << " order=" << go << endl;

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
		longinteger_object &go, int verbose_level)
// k is projective dimension
{
	int f_v = (verbose_level >= 1);
	int m;
	number_theory_domain NT;
	geometry_global Gg;

	if (f_v) {
		cout << "order_PO_epsilon" << endl;
		}
	m = Gg.Witt_index(epsilon, k);
	if (f_v) {
		cout << "Witt index = " << m << endl;
		}
	order_PO(epsilon, m, q, go, verbose_level);
	if (f_semilinear) {
		int p, e;
		longinteger_domain D;

		NT.factor_prime_power(q, p, e);
		D.mult_integer_in_place(go, e);
		}
	if (f_v) {
		cout << "order_PO_epsilon  f_semilinear=" << f_semilinear
				<< " epsilon=" << epsilon << " k=" << k
				<< " q=" << q << " order=" << go << endl;
		}
}

void group_generators_domain::order_PO(
		int epsilon, int m, int q,
		longinteger_object &o, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "order_PO epsilon = " << epsilon
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
		cout << "order_PO fatal: epsilon = " << epsilon << endl;
		exit(1);
		}
}

void group_generators_domain::order_Pomega(
		int epsilon, int m, int q,
		longinteger_object &o, int verbose_level)
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
		cout << "order_Pomega fatal: epsilon = " << epsilon << endl;
		exit(1);
		}
}

void group_generators_domain::order_PO_plus(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, Two, minusone;
	int i;
	geometry_global Gg;


	Two.create(2);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m - 1));
	if (f_v) {
		cout << "order_PO_plus " << q << "^(" << m << "*"
				<< m - 1 << ") = " << Q << endl;
		}
	// now Q = q^{m(m-1)}

	O.create(1);
	for (i = 1; i <= m - 1; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_plus " << q << "^"
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
		cout << "order_PO_plus " << q << "^" << m << " - 1 = " << S << endl;
		}
	// now S = q^m-1

	D.mult(O, S, T);
	T.assign_to(O);

	D.mult(O, Q, T);
	if (TRUE /*EVEN(q)*/) {
		D.mult(T, Two, o);
		}
	else {
		T.assign_to(o);
		}


	if (f_v) {
		cout << "order_PO_plus the order of PO" << "("
				<< Gg.dimension_given_Witt_index(1, m) << ","
				<< q << ") is " << o << endl;
		}
}

void group_generators_domain::order_PO_minus(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+2
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, Two, plusone, minusone;
	int i;
	geometry_global Gg;


	Two.create(2);
	plusone.create(1);
	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * (m + 1));
	if (f_v) {
		cout << "order_PO_minus " << q << "^(" << m << "*"
				<< m + 1 << ") = " << Q << endl;
		}
	// now Q = q^{m(m+1)}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_minus " << q << "^" << 2 * i
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
		cout << "order_PO_minus " << q << "^" << m + 1
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
		cout << "order_PO_minus the order of PO^-" << "("
			<< Gg.dimension_given_Witt_index(-1, m) << ","
			<< q << ") is " << o << endl;
		}
}

void group_generators_domain::order_PO_parabolic(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m+1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, minusone;
	int i;
	geometry_global Gg;


	minusone.create(-1);
	Q.create(q);
	D.power_int(Q, m * m);
	if (f_v) {
		cout << "order_PO_parabolic " << q << "^(" << m
				<< "^2" << ") = " << Q << endl;
		}
	// now Q = q^{m^2}

	O.create(1);
	for (i = 1; i <= m; i++) {
		R.create(q);
		D.power_int(R, 2 * i);
		D.add(R, minusone, S);
		if (f_v) {
			cout << "order_PO_parabolic " << q << "^"
					<< 2 * i << " - 1 = " << S << endl;
			}
		D.mult(O, S, T);
		T.assign_to(O);
		}
	// now O = \prod_{i=1}^{m} (q^{2i}-1)


	D.mult(O, Q, o);


	if (f_v) {
		cout << "order_PO_parabolic the order of PO" << "("
			<< Gg.dimension_given_Witt_index(0, m) << ","
			<< q << ") is " << o << endl;
		}
}


void group_generators_domain::order_Pomega_plus(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, S1, T, minusone;
	int i, r;
	geometry_global Gg;


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
		cout << "the order of P\\Omega^1" << "("
			<< Gg.dimension_given_Witt_index(1, m) << ","
			<< q << ") is " << o << endl;
		}
}

void group_generators_domain::order_Pomega_minus(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = half the dimension,
// the dimension is n = 2m, the Witt index is m - 1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, S1, T, minusone, plusone;
	int i, r;
	geometry_global Gg;

	if (f_v) {
		cout << "order_Pomega_minus m=" << m << " q=" << q << endl;
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
		cout << "the order of P\\Omega^-1" << "("
			<< Gg.dimension_given_Witt_index(-1, m - 1) << ","
			<< q << ") is " << o << endl;
		}
}

void group_generators_domain::order_Pomega_parabolic(
		int m, int q,
		longinteger_object &o, int verbose_level)
// m = Witt index, the dimension is n = 2m + 1
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object O, Q, R, S, T, minusone;
	int i, r;
	geometry_global Gg;


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
		cout << "the order of P\\Omega" << "("
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




}
}

