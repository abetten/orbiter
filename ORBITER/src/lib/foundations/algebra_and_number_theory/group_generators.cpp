// group_generators.C
//
// Anton Betten
//
// moved here from projective.C: September 4, 2016




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



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

void generators_Hall_reflection(int nb_pairs,
		int &nb_perms, int *&perms, int &degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "generators_Hall_reflection" << endl;
	}
	degree = nb_pairs * 2;
	nb_perms = 1;
	perms = NEW_int(nb_perms * degree);
	perm_identity(perms, degree);
	for (i = 0; i < nb_pairs; i++) {
		perms[2 * i] = 2 * i + 1;
		perms[2 * i + 1] = 2 * i;
	}
	if (f_v) {
		cout << "generators_Hall_reflection "
				"generators for the Hall reflection group "
				"of degree "
				<< degree << " created" << endl;
		}
	if (f_vv) {
		for (i = 0; i < 1; i++) {
			perm_print(cout, perms + i * degree, degree);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "generators_Hall_reflection done" << endl;
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



}
}

