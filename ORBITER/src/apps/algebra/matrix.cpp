// matrix.C
//
// Anton Betten
// August 8, 2006
//
// searches for an element of order o in 
// PGL(n,q)
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;

void print_usage()
{
	cout << "usage: matrix.out [options] n q o" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <verbose_level>" << endl;
	cout << "  verbose level <verbose_level>" << endl;
}

int main(int argc, const char **argv)
{
	int t0 = os_ticks();
	int verbose_level = 0;
	int i;
	int n, q, o;
	geometry_global Gg;

	if (argc <= 3) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 3; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v" << verbose_level << endl;
			}
		}
	n = atoi(argv[argc - 3]);
	q = atoi(argv[argc - 2]);
	o = atoi(argv[argc - 1]);
	
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	finite_field GFq;
	GFq.init(q, 0);
	
	action *A;
	//matrix_group *M;
	int f_semilinear = TRUE;
	int order, nb, N;
	
	int *Elt1;
	
	A = NEW_OBJECT(action);
	N = Gg.nb_PG_elements(n - 1, q);

	if (f_v && N < 100) {
		GFq.display_all_PG_elements(n - 1);
		}
	int f_basis = TRUE;
	vector_ge *nice_gens;

	A->init_projective_group(n, &GFq, 
		f_semilinear, f_basis,
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "a permutation group of degree "
				<< A->degree << endl;
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	
	nb = 1;
	while (TRUE) {
		if (!A->f_has_sims) {
			cout << "fatal: !A->f_has_sims" << endl;
			exit(1);
			}
		A->Sims->random_element(Elt1, 0);
		order = A->element_order_if_divisor_of(Elt1, o);
		if (order == o) {
			break;
			}
		nb++;
		}
	
	cout << "found an element of order " << o
			<< " in PGL(" << n << "," << q << ") after "
			<< nb << " trials" << endl;
	A->element_print(Elt1, cout);

	time_check(cout, t0);
	cout << endl;
	
	{
	ofstream fp("tmp");
	fp << "success" << endl;
	}
	cout << "written file tmp indicating success" << endl;

	
	FREE_OBJECT(A);
}

