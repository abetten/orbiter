// counting_flags.C
//
// Anton Betten
// May 26, 2016


#include "orbiter.h"

void count_nb_liftings(longinteger_object &N, int m, int n, int q);
void count_nb_grass(longinteger_object &N, int m, int n, int q);
void count_nb_flags(longinteger_object &N, int m, int n, int q);



int main()
{
	int n = 2; 
	int n_max = 50;
	int q1 = 2;
	int q2 = 5;
	int m = 100;
	//int m_max = 50;
	//int k;
	//int n2;
	//double a, b, c, d;
	longinteger_object A, B, C, E;
	longinteger_object nb_liftings;
	longinteger_object nb_grass;
	longinteger_object nb_flags;
	longinteger_domain D;
	int nb_L, nb_G, nb_F;

	//n2 = int_n_choose_k(n, 2);

	cout << "n,m," << q1 << ",nb_liftings_" << q1 << ",nb_grass_"
			<< q1 << ",nb_flags_" << q1 << "," << q2
			<< ",nb_liftings_" << q2 << ",nb_grass_"
			<< q2 << ",nb_flags_" << q2  << endl;
	for (n = 1; n <= n_max; n++) {

		//cout << "n=" << n << endl;

		//cout << "nb_liftings=" << nb_liftings << endl;
		//cout << "nb_grass   =" << nb_grass << endl;
		//cout << "nb_flags   =" << nb_flags << endl;
		//cout << m << "," << n << "," << q << "," << nb_liftings
		//<< "," << nb_grass << "," << nb_flags << endl;
		cout << n << "," << m;

		count_nb_liftings(nb_liftings, m, n, q1);
		count_nb_grass(nb_grass, m, n, q1);
		count_nb_flags(nb_flags, m, n, q1);
		nb_L = D.logarithm_base_b(nb_liftings, 10);
		nb_G = D.logarithm_base_b(nb_grass, 10);
		nb_F = D.logarithm_base_b(nb_flags, 10);
		//cout << endl;

		cout << "," << q1 << "," << nb_L  << "," << nb_G  << "," << nb_F;

		count_nb_liftings(nb_liftings, m, n, q2);
		count_nb_grass(nb_grass, m, n, q2);
		count_nb_flags(nb_flags, m, n, q2);
		nb_L = D.logarithm_base_b(nb_liftings, 10);
		nb_G = D.logarithm_base_b(nb_grass, 10);
		nb_F = D.logarithm_base_b(nb_flags, 10);
		//cout << endl;

		cout << "," << q2 << "," << nb_L  << "," << nb_G  << "," << nb_F;

		cout  << endl;

		}
	//cout << "END" << endl;
}

void count_nb_liftings(longinteger_object &N, int m, int n, int q)
{
	N.create_power(q, n * (m - n));
}

void count_nb_grass(longinteger_object &N, int m, int n, int q)
{
	int k;
	longinteger_object A;
	longinteger_domain D;

	N.create(0);
	for (k = 0; k <= n; k++) {
		D.q_binomial_no_table(A, m, k, q, 0 /* verbose_level */);
		//cout << "[" << m << ", " << k << "]_" << q << " = " << A << endl;
		D.add_in_place(N, A);
		}
}

void count_nb_flags(longinteger_object &N, int m, int n, int q)
{
	int k;
	longinteger_object B, C, E;
	longinteger_domain D;

	B.create(1);
	for (k = 0; k < n; k++) {
		C.create_power_minus_one(q, m - k);
		D.mult_in_place(B, C);
		}
	E.create_power(q - 1, n);
	D.integral_division_exact(B, E, N);
}



