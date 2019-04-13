// orthogonal_points.C
//
// Anton Betten
//
// started:  February 3, 2005
// last change:  August 13, 2005




#include "orbiter.h"

using namespace std;



using namespace orbiter;


void print_usage();



void print_usage()
{
	cout << "usage: orthogonal_points.out [options] epsilon d q" << endl;
	cout << "computes the set of points on O^{epsilon}(d,q)" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <k>" << endl;
	cout << "   : verbose level k" << endl;
}

int main(int argc, char **argv)
{
	int t0 = os_ticks();
	int verbose_level = 0;
	int i;
	int d, n, q, epsilon;

	if (argc <= 3) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 3; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}

		}
	epsilon = atoi(argv[argc - 3]);
	d = atoi(argv[argc - 2]);
	q = atoi(argv[argc - 1]);

	int f_v = (verbose_level >= 1);

	n = d - 1; // projective dimension
		
	int m, m1, m2, nb, nb2, nb3, Sbar, N1;
	
	nb3 = nb_pts_Qepsilon(epsilon, n, q);
	
	nb = 0;
	nb2 = 0;
	if (epsilon == 0) {
		nb = (i_power_j(q, n) - 1) / (q - 1);
		m = n >> 1;
		Sbar = nb_pts_Sbar(m, q);
		N1 = nb_pts_N1(m, q);
		nb2 = Sbar + N1;
		cout << "nb_pts_Sbar = " << nb_pts_Sbar(m, q) << endl;
		cout << "nb_pts_N1 = " << nb_pts_N1(m, q) << endl;
		}
	else if (epsilon == -1) {
		m1 = (n + 1) >> 1;
		m2 = (n - 1) >> 1;
		nb = ((i_power_j(q, m1) + 1) *  (i_power_j(q, m2) - 1)) / (q - 1);
		m = n >> 1;
		Sbar = nb_pts_Sbar(m, q);
		N1 = nb_pts_N1(m, q);
		nb2 = Sbar + (q + 1) * N1;
		cout << "nb_pts_Sbar = " << Sbar << endl;
		cout << "nb_pts_N1 = " << N1 << endl;
		cout << "(q + 1) * nb_pts_N1 = " << (q + 1) * N1 << endl;
		}
	else if (epsilon == 1) {
		m1 = (n + 1) >> 1;
		m2 = (n - 1) >> 1;
		m = (n + 1) >> 1;
		nb = ((i_power_j(q, m1) - 1) *  (i_power_j(q, m2) + 1)) / (q - 1);
		nb2 = nb_pts_Sbar(m, q);
		//test_orthogonal(m, q);
		}
	cout << "nb=" << nb << endl;
	cout << "nb2=" << nb2 << endl;
	cout << "nb3=" << nb3 << endl;
	test_Orthogonal(epsilon, n, q);
	
	int c1 = 1, c2 = 0, c3 = 0;
	int N, j;
	finite_field GFq;
	int *v, *L;

	GFq.init(q, verbose_level);
	N = nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_int(d);
	L = NEW_int(N);

	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
		}
	if (epsilon == -1) {
		GFq.choose_anisotropic_form(c1, c2, c3, verbose_level);
		}
	else {
		c1 = 0;
		c2 = 0;
		c3 = 0;
		}
	for (i = 0; i < N; i++) {
		GFq.Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i);
		GFq.PG_element_rank_modified(v, 1, d, j);
		L[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

	cout << "list of points:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;

	char fname[1000];
	sprintf(fname, "Q%s_%d_%d.txt", plus_minus_letter(epsilon), n, q);
	write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	FREE_int(L);
	time_check(cout, t0);
	cout << endl;
}


