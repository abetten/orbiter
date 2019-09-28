// get_poly.cpp
//
// Anton Betten
// October 20, 2005

#include "orbiter.h"

using namespace std;


using namespace orbiter;

int main(int argc, char **argv)
{
	os_interface Os;
	int t0 = Os.os_ticks();
	int verbose_level = 0;
	int i;
	int f_primitive_range = FALSE;
	int p_min, p_max, n_min, n_max;
	int f_irred = FALSE;
	int q, d;
	int f_primitive = FALSE;
	int deg;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-primitive_range") == 0) {
			f_primitive_range = TRUE;
			p_min = atoi(argv[++i]);
			p_max = atoi(argv[++i]);
			n_min = atoi(argv[++i]);
			n_max = atoi(argv[++i]);
			cout << "-primitive_range " << p_min
					<< " " << p_max << " "
					<< n_min << " "
					<< n_max << endl;
			}
		else if (strcmp(argv[i], "-primitive") == 0) {
			f_primitive = TRUE;
			q = atoi(argv[++i]);
			deg = atoi(argv[++i]);
			cout << "-primitive " << q << " " << deg << endl;
			}
		else if (strcmp(argv[i], "-irred") == 0) {
			f_irred = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			}
		}
	
	if (f_primitive_range) {
		search_for_primitive_polynomials(p_min, p_max,
				n_min, n_max, verbose_level);
			// in GALOIS/galois_global.cpp
		}
	else if (f_primitive) {
		char *poly;


		poly = search_for_primitive_polynomial_of_given_degree(
				q, deg, verbose_level);
			// in GALOIS/galois_global.cpp
		cout << "poly = " << poly << endl;
		}
	else if (f_irred) {
		int nb;
		int *Table;
		finite_field F;
		
		F.init(q, 0);

		F.make_all_irreducible_polynomials_of_degree_d(d,
				nb, Table, verbose_level);

		cout << "The " << nb << " irreducible polynomials of "
				"degree " << d << " over F_" << q << " are:" << endl;
		int_matrix_print(Table, nb, d + 1);

		FREE_int(Table);
		}
	
	Os.time_check(cout, t0);
	cout << endl;
}

