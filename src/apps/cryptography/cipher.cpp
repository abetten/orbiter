// cipher.cpp
//
// by Anton Betten
//
// Colorado State University
//
// for M360 mathematics of information security, Fall 03
//
// last change September 1 2005

#include "orbiter.h"

using namespace std;
using namespace orbiter;


enum cipher_type { no_cipher_type, substitution, vigenere, affine };

typedef enum cipher_type cipher_type;

void print_usage();

void quadratic_sieve(int n, int factorbase, int x0, int verbose_level);
void calc_log2(vector<int> &primes, vector<int> &primes_log2, int verbose_level);
void square_root(const char *square_root_number, int verbose_level);
void square_root_mod(const char *square_root_number, const char *mod_number, int verbose_level);
void reduce_primes(vector<int> &primes,
		longinteger_object &M,
		int &f_found_small_factor, int &small_factor,
		int verbose_level);
void do_sift_smooth(int sift_smooth_from,
		int sift_smooth_len,
		const char *sift_smooth_factor_base, int verbose_level);
void do_discrete_log(long int y, long int a, long int p, int verbose_level);
void do_inverse_mod(long int a, long int n, int verbose_level);
void do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
		int RSA_block_size, const char * RSA_encrypt_text, int verbose_level);
void do_RSA(long int RSA_d, long int RSA_m, const char *RSA_text, int verbose_level);
void affine_cipher(char *ptext, char *ctext, int a, int b);
void affine_decipher(char *ctext, char *ptext, char *guess);
void vigenere_cipher(char *ptext, char *ctext, char *key);
void vigenere_decipher(char *ctext, char *ptext, char *key);
void vigenere_analysis(char *ctext);
void vigenere_analysis2(char *ctext, int key_length);
int kasiski_test(char *ctext, int threshold);
void print_candidates(char *ctext,
		int i, int h, int nb_candidates, int *candidates);
void print_set(int l, int *s);
void print_on_top(char *text1, char *text2);
void decipher(char *ctext, char *ptext, char *guess);
void analyze(char *text);
double friedman_index(int *mult, int n);
double friedman_index_shifted(int *mult, int n, int shift);
void print_frequencies(int *mult);
void single_frequencies(char *text, int *mult);
void single_frequencies2(char *text, int stride, int n, int *mult);
void double_frequencies(char *text, int *mult);
void substition_cipher(char *ptext, char *ctext, char *key);
char lower_case(char c);
char upper_case(char c);
char is_alnum(char c);
void get_random_permutation(char *p);

double letter_probability[] = {
	0.082, // a
	0.015, // b
	0.028, // c
	0.043, // d
	0.127, // e
	0.022, // f
	0.020, // g
	0.061, // h
	0.070, // i
	0.002, // j
	0.008, // k
	0.040, // l
	0.024, // m
	0.067, // n
	0.075, // o
	0.019, // p
	0.001, // q
	0.060, // r
	0.063, // s
	0.091, // t
	0.028, // u
	0.010, // v
	0.001, // x
	0.020, // y
	0.001 // z
};


void print_usage()
{
	cout << "usage: cipher [options] \n";
	cout << "where options can be:\n";
	cout << "-cs plaintext          : cipher using random substitution\n";
	cout << "-cv plaintext key      : cipher using vigenere cipher with given key\n";
	cout << "-ca plaintext a b      : cipher using vigenere cipher with given key (a,b): x -> ax+b\n";
	cout << "-as text               : frequency analysis for substitution cypher\n";
	cout << "-av text               : analysis for vigenere cypher\n";
	cout << "-avk text keylength    : analysis for vigenere cypher: determine the key given the keylength\n";
	cout << "-kasiski               : kasiski test for vigenere cypher\n";
	cout << "-ds ciphertext guess   : guess a substitution cypher,\n";
	cout << "                       : guess is a string of pairs of letters,\n";
	cout << "                       : the first letter of a pair\n";
	cout << "                       : is replaced by the second one\n";
	cout << "-dv ciphertext key     : decipher vigenere ciphertext with given key\n";
	cout << "-da ciphertext xayb    : decipher affine ciphertext by guessing\n";
	cout << "                       : that x decodes to a, y decodes to b\n";
	cout << "-seed n                : seeds the random number generator with integer n\n";
	cout << "                       : (you need to seed the random number generator with\n";
	cout << "                       : different values in order to get different random\n";
	cout << "                       : substitutions for -cs)\n";
}

int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_cipher = FALSE;
	cipher_type t = no_cipher_type;
	int f_decipher = FALSE;
	int f_analyze = FALSE;
	int f_seed = FALSE;
	int f_kasiski = FALSE;
	int f_avk = FALSE;
	int key_length, threshold;
	int the_seed;
	int affine_a;
	int affine_b;
	char ptext[10000];
	char ctext[10000];
	char guess[10000];
	char key[1000];
	int f_RSA = FALSE;
	long int RSA_d;
	long int RSA_m;
	const char *RSA_text;
	int f_inverse_mod = FALSE;
	int inverse_mod_a = 0;
	int inverse_mod_n = 0;
	int f_discrete_log = FALSE;
	long int discrete_log_y = 0;
	long int discrete_log_a = 0;
	long int discrete_log_m = 0;
	int f_RSA_setup = FALSE;
	int RSA_setup_nb_bits;
	int RSA_setup_nb_tests_solovay_strassen;
	int RSA_setup_f_miller_rabin_test;
	int f_RSA_encrypt_text = FALSE;
	int RSA_block_size = 0;
	const char *RSA_encrypt_text = NULL;
	int f_sift_smooth = FALSE;
	int sift_smooth_from = 0;
	int sift_smooth_len = 0;
	const char *sift_smooth_factor_base = NULL;
	int f_square_root = FALSE;
	const char *square_root_number = NULL;
	int f_square_root_mod = FALSE;
	const char *square_root_mod_a = NULL;
	const char *square_root_mod_m = NULL;
	int f_quadratic_sieve = FALSE;
	int quadratic_sieve_n = 0;
	int quadratic_sieve_factorbase = 0;
	int quadratic_sieve_x0 = 0;

	
	cout << "this is cipher" << endl;
	//return 0;

	if (argc <= 1) {
		print_usage();
		exit(1);
	}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-cs") == 0) {
			f_cipher = TRUE;
			t = substitution;
			i++;
			strcpy(ptext, argv[i]);
			//cout << "-c " << ptext << endl;
		}
		else if (strcmp(argv[i], "-cv") == 0) {
			f_cipher = TRUE;
			t = vigenere;
			i++;
			strcpy(ptext, argv[i]);
			i++;
			strcpy(key, argv[i]);
			//cout << "-c " << ptext << endl;
		}
		else if (strcmp(argv[i], "-ca") == 0) {
			f_cipher = TRUE;
			t = affine;
			i++;
			strcpy(ptext, argv[i]);
			i++;
			sscanf(argv[i], "%d", &affine_a);
			i++;
			sscanf(argv[i], "%d", &affine_b);
			//cout << "-c " << ptext << endl;
		}
		else if (strcmp(argv[i], "-as") == 0) {
			f_analyze = TRUE;
			t = substitution;
			i++;
			strcpy(ctext, argv[i]);
			//cout << "-a " << ctext << endl;
		}
		else if (strcmp(argv[i], "-av") == 0) {
			f_analyze = TRUE;
			t = vigenere;
			i++;
			strcpy(ctext, argv[i]);
			//cout << "-a " << ctext << endl;
		}
		else if (strcmp(argv[i], "-avk") == 0) {
			f_avk = TRUE;
			i++;
			strcpy(ctext, argv[i]);
			i++;
			sscanf(argv[i], "%d", &key_length);
			//cout << "-a " << ctext << endl;
		}
		else if (strcmp(argv[i], "-kasiski") == 0) {
			f_kasiski = TRUE;
			i++;
			strcpy(ctext, argv[i]);
			i++;
			sscanf(argv[i], "%d", &threshold);
			//cout << "-a " << ctext << endl;
		}
		else if (strcmp(argv[i], "-ds") == 0) {
			f_decipher = TRUE;
			t = substitution;
			i++;
			strcpy(ctext, argv[i]);
			i++;
			strcpy(guess, argv[i]);
			//cout << "-d " << ctext << " " << guess << endl;
		}
		else if (strcmp(argv[i], "-dv") == 0) {
			f_decipher = TRUE;
			t = vigenere;
			i++;
			strcpy(ctext, argv[i]);
			i++;
			strcpy(key, argv[i]);
		}
		else if (strcmp(argv[i], "-da") == 0) {
			f_decipher = TRUE;
			t = affine;
			i++;
			strcpy(ctext, argv[i]);
			i++;
			strcpy(guess, argv[i]);
		}
		else if (strcmp(argv[i], "-seed") == 0) {
			f_seed = TRUE;
			i++;
			sscanf(argv[i], "%d", &the_seed);
			//cout << "-seed " << the_seed << endl;
		}
		else if (strcmp(argv[i], "-RSA") == 0) {
			f_RSA = TRUE;
			RSA_d = atol(argv[++i]);
			RSA_m = atol(argv[++i]);
			RSA_text = argv[++i];
			cout << "-RSA " << RSA_d << " " << RSA_m << " " << RSA_text << endl;
		}
		else if (strcmp(argv[i], "-RSA_encrypt_text") == 0) {
			f_RSA_encrypt_text = TRUE;
			RSA_d = atol(argv[++i]);
			RSA_m = atol(argv[++i]);
			RSA_block_size = atoi(argv[++i]);
			RSA_encrypt_text = argv[++i];
			cout << "-RSA_encrypt_text " << RSA_d << " " << RSA_m << " " << RSA_encrypt_text << endl;
		}
		else if (strcmp(argv[i], "-RSA_setup") == 0) {
			f_RSA_setup = TRUE;
			RSA_setup_nb_bits = atoi(argv[++i]);
			RSA_setup_nb_tests_solovay_strassen = atoi(argv[++i]);
			RSA_setup_f_miller_rabin_test = atoi(argv[++i]);
			cout << "-RSA_setup " << RSA_setup_nb_bits << " " << RSA_setup_nb_tests_solovay_strassen << " " << RSA_setup_f_miller_rabin_test << endl;
		}
		else if (strcmp(argv[i], "-inverse_mod") == 0) {
			f_inverse_mod = TRUE;
			inverse_mod_a = atol(argv[++i]);
			inverse_mod_n = atol(argv[++i]);
			cout << "-inverse_mod " << inverse_mod_a << " " << inverse_mod_n << endl;
		}
		else if (strcmp(argv[i], "-discrete_log") == 0) {
			f_discrete_log = TRUE;
			discrete_log_y = atol(argv[++i]);
			discrete_log_a = atol(argv[++i]);
			discrete_log_m = atol(argv[++i]);
			cout << "-discrete_log " << discrete_log_y << " " << discrete_log_a << " " << discrete_log_m << endl;
		}
		else if (strcmp(argv[i], "-sift_smooth") == 0) {
			f_sift_smooth = TRUE;
			sift_smooth_from = atol(argv[++i]);
			sift_smooth_len = atol(argv[++i]);
			sift_smooth_factor_base = argv[++i];
			cout << "-sift_smooth " << sift_smooth_from << " " << sift_smooth_len << " " << sift_smooth_factor_base << endl;
		}
		else if (strcmp(argv[i], "-square_root") == 0) {
			f_square_root = TRUE;
			square_root_number = argv[++i];
			cout << "-square_root " << square_root_number << endl;
		}
		else if (strcmp(argv[i], "-square_root_mod") == 0) {
			f_square_root_mod = TRUE;
			square_root_mod_a = argv[++i];
			square_root_mod_m = argv[++i];
			cout << "-square_root_mod " << square_root_mod_a << " " << square_root_mod_m << endl;
		}
		else if (strcmp(argv[i], "-quadratic_sieve") == 0) {
			f_quadratic_sieve = TRUE;
			quadratic_sieve_n = atoi(argv[++i]);
			quadratic_sieve_factorbase = atoi(argv[++i]);
			quadratic_sieve_x0 = atoi(argv[++i]);
			cout << "-quadratic_sieve " << quadratic_sieve_n << " "
					<< quadratic_sieve_factorbase << " " << quadratic_sieve_x0 << endl;
		}
	}




	if (f_seed) {
		srand(the_seed);
	}
	if (f_cipher) {
		if (t == substitution) {
			get_random_permutation(key);
			cout << "ptext: " << ptext << endl;
			substition_cipher(ptext, ctext, key);
			cout << "ctext: " << ctext << endl;
		}
		else if (t == vigenere) {
			cout << "vigenere cipher with key " << key << endl;
			cout << "ptext: " << ptext << endl;
			vigenere_cipher(ptext, ctext, key);
			cout << "ctext: " << ctext << endl;
		}
		else if (t == affine) {
			cout << "affine cipher with key (" << affine_a << "," << affine_b << ")" << endl;
			cout << "ptext: " << ptext << endl;
			affine_cipher(ptext, ctext, affine_a, affine_b);
			cout << "ctext: " << ctext << endl;
		}
	}
	else if (f_analyze) {
		if (t == substitution) {
			cout << "ctext: \n" << ctext << endl;
			analyze(ctext);
		}
		if (t == vigenere) {
			cout << "ctext: \n" << ctext << endl;
			vigenere_analysis(ctext);
		}
	}
	else if (f_avk) {
		vigenere_analysis2(ctext, key_length);
	}
	else if (f_kasiski) {
		int m;
		
		m = kasiski_test(ctext, threshold);
		cout << "kasiski test for threshold " << threshold
				<< " yields key length " << m << endl;
	}
	else if (f_decipher) {
		if (t == substitution) {
			cout << "ctext: " << ctext << endl;
			cout << "guess: " << guess << endl;
			decipher(ctext, ptext, guess);
			cout << " " << endl;
			print_on_top(ptext, ctext);
			//cout << "ptext: " << ptext << endl;
		}
		else if (t == vigenere) {
			cout << "ctext: " << ctext << endl;
			cout << "key  : " << key << endl;
			vigenere_decipher(ctext, ptext, key);
			cout << " " << endl;
			print_on_top(ptext, ctext);
		}
		else if (t == affine) {
			//cout << "affine cipher with key (" << affine_a << "," << affine_b << ")" << endl;
			cout << "affine cipher" << endl;
			cout << "ctext: " << ctext << endl;
			cout << "guess: " << guess << endl;
			affine_decipher(ctext, ptext, guess);
			//cout << " " << endl;
			//print_on_top(ptext, ctext);
		}
	}
	else if (f_discrete_log) {
		do_discrete_log(discrete_log_y, discrete_log_a, discrete_log_m, verbose_level);
	}
	else if (f_inverse_mod) {
		do_inverse_mod(inverse_mod_a, inverse_mod_n, verbose_level);
	}
	else if (f_RSA) {
		do_RSA(RSA_d, RSA_m, RSA_text, verbose_level);
	}
	else if (f_RSA_encrypt_text) {
		do_RSA_encrypt_text(RSA_d, RSA_m, RSA_block_size, RSA_encrypt_text, verbose_level);
	}
	else if (f_RSA_setup) {
		longinteger_domain D;
		longinteger_object n, p, q, a, b;

		D.RSA_setup(n, p, q, a, b,
			RSA_setup_nb_bits,
			RSA_setup_nb_tests_solovay_strassen,
			RSA_setup_f_miller_rabin_test,
			1 /*verbose_level */);
	}
	else if (f_sift_smooth) {
		do_sift_smooth(sift_smooth_from,
				sift_smooth_len,
				sift_smooth_factor_base, verbose_level);
	}
	else if (f_square_root) {
		square_root(square_root_number, verbose_level);
	}
	else if (f_square_root_mod) {
		square_root_mod(square_root_mod_a, square_root_mod_m, verbose_level);
	}
	else if (f_quadratic_sieve) {
		quadratic_sieve(quadratic_sieve_n,
				quadratic_sieve_factorbase,
				quadratic_sieve_x0,
				verbose_level);
	}
}

void quadratic_sieve(int n,
		int factorbase, int x0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector<int> small_factors, primes, primes_log2, R1, R2;
	longinteger_object M, sqrtM;
	longinteger_domain D;
	number_theory_domain NT;
	int f_found_small_factor = FALSE;
	int small_factor;
	int i;

	if (f_v) {
		cout << "quadratic_sieve" << endl;
	}
	if (f_v) {
		cout << "quadratic_sieve before sieve" << endl;
	}
	NT.sieve(primes, factorbase, verbose_level - 1);
	if (f_v) {
		cout << "quadratic_sieve after sieve" << endl;
		cout << "list of primes has length " << primes.size() << endl;
	}
	D.create_Mersenne(M, n);
	cout << "Mersenne number M_" << n << "=" << M << " log10=" << M.len() << endl;
	D.square_root(M, sqrtM, 0 /*verbose_level - 1*/);
	cout << "sqrtM=" << sqrtM << " log10=" << sqrtM.len() << endl;

	// N1.mult(sqrtM, sqrtM);
	// sqrtM.inc();
	// N2.mult(sqrtM, sqrtM);
	// sqrtM.dec();
	// cout << "sqrtM^2=" << N1 << endl;
	// cout << "M=" << M << endl;
	// cout << "(sqrtM+1)^2=" << N2 << endl;
	// longinteger_compare_verbose(N1, M);
	// longinteger_compare_verbose(M, N2);

	cout << "calling reduce_primes" << endl;
	//small_primes.m_l(0);
	while (TRUE) {
		//int p;
		reduce_primes(primes, M,
				f_found_small_factor, small_factor,
				verbose_level - 1);
		if (!f_found_small_factor) {
			break;
		}
		longinteger_object P, Q;

		cout << "dividing out small factor " << small_factor << endl;
		small_factors.push_back(small_factor);
		P.create(small_factor, __FILE__, __LINE__);
		D.integral_division_exact(M, P, Q);
		Q.assign_to(M);
		cout << "reduced M=" << M << " log10=" << M.len() << endl;
		D.square_root(M, sqrtM, 0 /*verbose_level - 1*/);
		cout << "sqrtM=" << sqrtM << endl << endl;
		}
	cout << "list of small factors has length " << small_factors.size() << endl;
	for (i = 0; i < small_factors.size(); i++) {
		cout << i << " : " << small_factors[i] << endl;
	}

	if (M.is_one()) {
		cout << "the number has been completely factored" << endl;
		exit(0);
	}


	D.calc_roots(M, sqrtM, primes, R1, R2, 0/*verbose_level - 1*/);
	calc_log2(primes, primes_log2, 0 /*verbose_level - 1*/);


	int f_x_file = FALSE;
	vector<int> X;

	if (f_x_file) {
#if 0
		int l = primes.size();
		int ll = l + 10;

		read_x_file(X, x_file, ll);
#endif
		}
	else {
		D.Quadratic_Sieve(factorbase, FALSE /* f_mod */, 0 /* mod_n */, 0 /* mod_r */, x0,
			n, M, sqrtM,
			primes, primes_log2, R1, R2, X, verbose_level - 1);
		if (FALSE /*f_mod*/) {
			exit(1);
			}
		}


	if (f_v) {
		cout << "quadratic_sieve done" << endl;
	}

}

void calc_log2(vector<int> &primes, vector<int> &primes_log2, int verbose_level)
{
	int i, l, k;

	l = primes.size();
	for (i = 0; i < l; i++) {
		k = log2(primes[i]);
		primes_log2.push_back(k);
		}
}

void square_root(const char *square_root_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object a, b;

	if (f_v) {
		cout << "square_root" << endl;
	}
	a.create_from_base_10_string(square_root_number);
	cout << "computing square root of " << a << endl;
	D.square_root(a, b, verbose_level - 4);
	cout << "square root of " << a << " is " << b << endl;

	if (f_v) {
		cout << "square_root done" << endl;
	}
}

void square_root_mod(const char *square_root_number, const char *mod_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	longinteger_object a, m;
	int b;

	if (f_v) {
		cout << "square_root" << endl;
	}
	a.create_from_base_10_string(square_root_number);
	cout << "computing square root of " << a << endl;
	m.create_from_base_10_string(mod_number);
	cout << "modulo " << m << endl;
	b = D.square_root_mod(a.as_int(), m.as_int(), verbose_level -1);
	cout << "square root of " << a << " mod " << m << " is " << b << endl;

	if (f_v) {
		cout << "square_root done" << endl;
	}
}

void reduce_primes(vector<int> &primes,
		longinteger_object &M,
		int &f_found_small_factor, int &small_factor,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l, r, s, p;
	longinteger_domain D;
	longinteger_object Q, R, P;

	if (f_v) {
		cout << "reduce_primes" << endl;
	}
	f_found_small_factor = FALSE;
	small_factor = 0;
	l = primes.size();
	for (i = 0; i < l; i++) {
		p = primes[i];
		// cout << "i=" << i << " prime=" << p << endl;
		D.integral_division_by_int(M,
				p, Q, r);

		R.create(r, __FILE__, __LINE__);
		P.create(p, __FILE__, __LINE__);

		s = D.jacobi(R, P, 0 /* verbose_level */);
		//s = Legendre(r, p, 0);
		// cout << "i=" << i << " p=" << p << " Mmodp=" << r
		//<< " Legendre(r,p)=" << s << endl;
		if (s == 0) {
			cout << "M is divisible by " << p << endl;
			//exit(1);
			f_found_small_factor = TRUE;
			small_factor = p;
			return;
			}
		if (s == -1) {
			primes.erase(primes.begin()+i);
			l--;
			i--;
			}
		}
	cout << "number of primes remaining = " << primes.size() << endl;
}


void do_sift_smooth(int sift_smooth_from,
		int sift_smooth_len,
		const char *sift_smooth_factor_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int sz;
	int a, i, j, nb, p, idx, cnt;
	number_theory_domain NT;
	sorting Sorting;

	if (f_v) {
		cout << "do_sift_smooth" << endl;
	}

	int_vec_scan(sift_smooth_factor_base, B, sz);
	Sorting.int_vec_heapsort(B, sz);

	cnt = 0;
	for (i = 0; i < sift_smooth_len; i++) {
		a = sift_smooth_from + i;

		int *primes;
		int *exponents;

		nb = NT.factor_int(a, primes, exponents);
		for (j = 0; j < nb; j++) {
			p = primes[j];
			if (!Sorting.int_vec_search(B, sz, p, idx)) {
				break;
			}
		}
		if (j == nb) {
			// the number is smooth:

			cout << cnt << " : " << a << " : ";
			NT.print_factorization(nb, primes, exponents);
			cout << endl;

			cnt++;
		}

		FREE_int(primes);
		FREE_int(exponents);

	}
}

void do_discrete_log(long int y, long int a, long int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "do_discrete_log" << endl;
		cout << "y=" << y << endl;
		cout << "a=" << a << endl;
		cout << "p=" << p << endl;
	}

	//finite_field F;
	long int n, b;

	//F.init(p, 0);
	for (n = 0; n < p - 1; n++) {
		//b = F.power(a, n);
		b = NT.power_mod(a, n, p);
		if (b == y) {
			cout << y << " = " << a << "^" << n << " mod " << p << endl;
			break;
		}
	}
}

void do_inverse_mod(long int a, long int n, int verbose_level)
{
	number_theory_domain NT;
	long int b;

	b = NT.inverse_mod(a, n);
	cout << "the inverse of " << a << " mod " << n << " is " << b << endl;
}

void do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
		int RSA_block_size, const char * RSA_encrypt_text, int verbose_level)
{
	int i, j, l, nb_blocks;
	long int a;
	char c;
	long int *Data;

	l = strlen(RSA_encrypt_text);
	nb_blocks = (l + RSA_block_size - 1) /  RSA_block_size;
	Data = NEW_lint(nb_blocks);
	for (i = 0; i < nb_blocks; i++) {
		a = 0;
		for (j = 0; j < RSA_block_size; j++) {
			c = RSA_encrypt_text[i * RSA_block_size + j];
			if (c >= 'a' && c <= 'z') {
				a *= 100;
				a += (int) (c - 'a') + 1;
			}
		Data[i] = a;
		}
	}

	longinteger_domain D;
	longinteger_object A, M;

	M.create(RSA_m, __FILE__, __LINE__);

	for (i = 0; i < nb_blocks; i++) {
		A.create(Data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << A;
		if (i < nb_blocks - 1) {
			cout << ",";
		}
	}
	cout << endl;
}

void do_RSA(long int RSA_d, long int RSA_m, const char *RSA_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data;
	int data_sz;
	int i;

	if (f_v) {
		cout << "do_RSA RSA_d=" << RSA_d << " RSA_m=" << RSA_m << endl;
	}
	lint_vec_scan(RSA_text, data, data_sz);
	if (f_v) {
		cout << "text: ";
		lint_vec_print(cout, data, data_sz);
		cout << endl;
	}

	longinteger_domain D;
	longinteger_object A, M;

	M.create(RSA_m, __FILE__, __LINE__);
	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << i << " : " << data[i] << " : " << A << endl;
	}
	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		cout << A;
		if (i < data_sz - 1) {
			cout << ",";
		}
	}
	cout << endl;

	long int a;
	int b, j, h;
	char str[1000];

	for (i = 0; i < data_sz; i++) {
		A.create(data[i], __FILE__, __LINE__);
		D.power_int_mod(
				A, RSA_d, M);
		//cout << A;
		a = A.as_lint();
		j = 0;
		while (a) {
			b = a % 100;
			if (b > 26 || b == 0) {
				cout << "out of range" << endl;
				exit(1);
			}
			str[j] = 'a' + b - 1;
			j++;
			str[j] = 0;
			a -= b;
			a /= 100;
		}
		for (h = j - 1; h >= 0; h--) {
			cout << str[h];
		}
	}
	cout << endl;
}

void affine_cipher(char *ptext, char *ctext, int a, int b)
{
	int i, l, x, y;
	
	cout << "applying key (" << a << "," << b << ")" << endl;
	l = strlen(ptext);
	for (i = 0; i < l; i++) {
		x = (int)(upper_case(ptext[i]) - 'A');
		y = a * x + b;
		y = y % 26;
		ctext[i] = 'A' + y;
	}
	ctext[l] = 0;
}

void affine_decipher(char *ctext, char *ptext, char *guess)
// we have ax_1 + b = y_1
// and     ax_2 + b = y_2
// or equivalently
//         matrix(x_1,1,x_2,1) vector(a,b) = vector(y_1,y_2)
// and hence
//         vector(a,b) = matrix(1,-1,-x_2,x_1) vector(y_1,y_2) * 1/(x_1 - x_2)
{
	int x1, x2, y1, y2, dy, dx, i;
	int a, b, a0, av, c, g, dxv, n, gg;
	number_theory_domain NT;

	if (strlen(guess) != 4) {
		cout << "guess must be 4 characters long!" << endl;
		exit(1);
	}
	cout << "guess=" << guess << endl;
	y1 = lower_case(guess[0]) - 'a';
	x1 = lower_case(guess[1]) - 'a';
	y2 = lower_case(guess[2]) - 'a';
	x2 = lower_case(guess[3]) - 'a';
	
	cout << "y1=" << y1 << endl;
	cout << "x1=" << x1 << endl;
	cout << "y2=" << y2 << endl;
	cout << "x2=" << x2 << endl;
	dy = remainder(y2 - y1, 26);
	dx = remainder(x2 - x1, 26);
	//cout << "dx = x2 - x1 = " << dx << endl;
	//cout << "dy = y2 - y1 = " << dy << endl;
	cout << "solving:  a * (x2-x1) = y2 - y1 mod 26" << endl;
	cout << "here:     a * " << dx << " = " << dy << " mod 26" << endl;
	n = 26;
	//g = gcd_int(dx, n);
	g = NT.gcd_lint(dx, n);
	if (remainder(dy, g) != 0) {
		cout << "gcd(x2-x1,26) does not divide y2-y1, hence no solution! try again" << endl;
		exit(1);
	}
	if (g != 1) {
		dy /= g;
		dx /= g;
		n /= g;
		cout << "reducing to:     a * " << dx << " = " << dy << " mod " << n << endl;
	}
	dxv = NT.inverse_mod(dx, n);
	cout << dx << "^-1 mod " << n << " = " << dxv << endl;
	a0 = remainder(dy * dxv, n);
	cout << "solution a = " << a0 << " mod " << n << endl;
	cout << "(ie " << g << " solutions for a mod 26)" << endl;
	for (i = 0; i < g; i++) {
		cout << "trying solution " << i + 1 << ":" << endl;
		a = a0 + i * n;
		b = remainder(y1 - a * x1, 26);
		cout << "yields key (" << a << "," << b << ")" << endl;
		//gg = gcd_int(a, 26);
		gg = NT.gcd_lint(a, 26);
		if (gg != 1) {
			cout << a << " not prime to 26, no solution" << endl;
			continue;
		}
		av = NT.inverse_mod(a, 26);
		cout << "a^-1 mod 26 = " << av << endl;
		c = av * (-b);
		c = remainder(c, 26);
		cout << "a^-1 (-b) = " << c << " mod 26" << endl;
		affine_cipher(ctext, ptext, av, c);
		cout << " " << endl;
		print_on_top(ptext, ctext);
	}
}

void vigenere_cipher(char *ptext, char *ctext, char *key)
{
	int i, j, l, key_len, a, b, c;
	
	key_len = strlen(key);
	l = strlen(ptext);
	for (i = 0, j = 0; i < l; i++, j++) {
		if (j == key_len) {
			j = 0;
		}
		if (is_alnum(ptext[i]) && is_alnum(key[j])) {
			a = (int)(lower_case(ptext[i]) - 'a');
			b = (int)(lower_case(key[j]) - 'a');
			c = a + b;
			c = c % 26;
			ctext[i] = 'a' + c;
		}
		else {
			ctext[i] = ptext[i];
		}
	}
	ctext[l] = 0;
}

void vigenere_decipher(char *ctext, char *ptext, char *key)
{
	int i, j, l, key_len, a, b, c;
	
	key_len = strlen(key);
	l = strlen(ctext);
	for (i = 0, j = 0; i < l; i++, j++) {
		if (j == key_len)
			j = 0;
		if (is_alnum(ctext[i]) && is_alnum(key[j])) {
			a = (int)(lower_case(ctext[i]) - 'a');
			b = (int)(lower_case(key[j]) - 'a');
			c = a - b;
			if (c < 0)
				c += 26;
			ptext[i] = 'a' + c;
		}
		else {
			ptext[i] = ctext[i];
		}
	}
	ptext[l] = 0;
}

void vigenere_analysis(char *ctext)
{
	int stride, l, n, m, j;
	int mult[100];
	double I, II;
	
	cout.precision(3);
	l = strlen(ctext);
	cout << "key length : Friedman indices in 1/1000-th : average" << endl;
	for (stride = 1; stride <= MINIMUM(l, 20); stride++) {
		m = l / stride;
		if (m < 2)
			continue;
		II = 0;
		cout << stride << " : ";
		for (j = 0; j < stride; j++) {
			if (j == stride - 1) {
				n = l - (stride - 1) * m;
			}
			else {
				n = m;
			}
			single_frequencies2(ctext + j, stride, n, mult);
			//print_frequencies(mult);
			I = friedman_index(mult, n);
			if (stride < 20) {
				if (j) {
					cout << ", ";
				}
				cout << (int)(1000. * I);
			}
			II += I;
		}
		II *= 1. / (double) stride;
		cout << " : " << (int)(1000. * II) << endl;
	}
}

void vigenere_analysis2(char *ctext, int key_length)
{
	int i, j, shift, n, m, l, a, h;
	int mult[100];
	int index[100];
	int shift0[100];
	double I;
	char c;
	
	l = strlen(ctext);
	m = l / key_length;
	cout << "    :";
	for (shift = 0; shift < 26; shift++) {
		cout << "  ";
		c = 'a' + shift;
		cout << c;
	}
	cout << endl;
	for (j = 0; j < key_length; j++) {
		cout.width(3);
		cout << j << " :";
		if (j == key_length - 1) {
			n = l - (key_length - 1) * m;
		}
		else {
			n = m;
		}
		single_frequencies2(ctext + j, key_length, n, mult);
		//print_frequencies(mult);
		for (shift = 0; shift < 26; shift++) {
			I = friedman_index_shifted(mult, n, shift);
			cout.width(3);
			a = (int)(1000. * I);
			cout << a;
			if (shift) {
				for (i = 0; i < shift; i++) {
					if (index[i] < a) {
						for (h = shift; h > i; h--) {
							index[h] = index[h - 1];
							shift0[h] = shift0[h - 1];
							}
						index[i] = a;
						shift0[i] = shift;
						break;
					}
				}
			}
			else {
				index[0] = a;
				shift0[0] = shift;
			}
		}
		cout << " : ";
		for (i = 0; i < 5; i++) {
			c = 'a' + shift0[i];
			if (i) {
				cout << ", ";
			}
			cout << c << " " << index[i];
		}
		cout << endl;
	}
}

int kasiski_test(char *ctext, int threshold)
{
	int l, i, j, k, u, h, offset;
	int *candidates, nb_candidates, *Nb_candidates;
	int *f_taken;
	int g = 0, g1;
	number_theory_domain NT;
	
	l = strlen(ctext);
	candidates = new int[l];
	f_taken = new int[l];
	Nb_candidates = new int[l];
	for (i = 0; i < l; i++) {
		f_taken[i] = FALSE;
	}

	for (i = 0; i < l; i++) {
		//cout << "position " << i << endl;
		nb_candidates = 0;
		for (j = i + 1; j < l; j++) {
			if (ctext[j] == ctext[i]) {
				candidates[nb_candidates++] = j;
			}
		}
		h = 1;
		//print_candidates(ctext, i, h, nb_candidates, candidates);
		
		//cout << "at position " << i << ", found " << nb_candidates << " matches of length 1" << endl;
		Nb_candidates[h - 1] += nb_candidates;
		while (nb_candidates) {
			for (k = 0; k < nb_candidates; k++) {
				j = candidates[k];
				if (ctext[i + h] != ctext[j + h]) {
					for (u = k + 1; u < nb_candidates; u++) {
						candidates[u - 1] = candidates[u];
					}
					nb_candidates--;
					k--;
				}
			}
			h++;
			Nb_candidates[h - 1] += nb_candidates;
			if (h >= threshold && nb_candidates) {
				print_candidates(ctext, i, h, nb_candidates, candidates);
				g1 = 0;
				for (k = 0; k < nb_candidates; k++) {
					offset = candidates[k] - i;
					if (g1 == 0) {
						g1 = offset;
					}
					else {
						//g1 = gcd_int(g1, offset);
						g1 = NT.gcd_lint(g1, offset);
					}
				}
				//cout << "g1 = " << g1 << endl;
				if (g == 0) {
					g = g1;
				}
				else {
					//g = gcd_int(g, g1);
					g = NT.gcd_lint(g, g1);
				}
				//cout << "g = " << g << endl;
				//break;
			}
		}
	}
	for (i = 0; Nb_candidates[i]; i++) {
		cout << "matches of length " << i + 1 << " : " <<  Nb_candidates[i] << endl;
	}
	delete [] candidates;
	delete [] f_taken;
	delete [] Nb_candidates;
	return g;
}

void print_candidates(char *ctext, int i, int h, int nb_candidates, int *candidates)
{
	int k, j, u;
	
	if (nb_candidates == 0)
		return;
	cout << "there are " << nb_candidates << " level " << h << " coincidences with position " << i << endl;
	for (k = 0; k < nb_candidates; k++) {
		j = candidates[k];
		cout << "at " << j << " : ";
		for (u = 0; u < h; u++) {
			cout << ctext[i + u];
		}
		cout << " : ";
		for (u = 0; u < h; u++) {
			cout << ctext[j + u];
		}
		cout << " offset " << j - i;
		cout << endl;
	}
}

void print_set(int l, int *s)
{
	int i;
	
	for (i = 0; i < l; i++) {
		cout << s[i] << " ";
	}
	cout << endl;
}

void print_on_top(char *text1, char *text2)
{
	int i, j, l, l2, lines, line_length;
	
	l = strlen(text1);
	l2 = strlen(text1);
	if (l2 != l) {
		cout << "text lengths do not match" << endl;
		exit(1);
	}
	lines = l / 80;
	for (i = 0; i <= lines; i++) {
		if (i == lines) {
			line_length = l % 80;
		}
		else {
			line_length = 80;
		}
		for (j = 0; j < line_length; j++) {
			cout << text1[i * 80 + j];
		}
		cout << endl;
		for (j = 0; j < line_length; j++) {
			cout << text2[i * 80 + j];
		}
		cout << endl;
		cout << " " << endl;
	}
}

void decipher(char *ctext, char *ptext, char *guess)
{
	int i, j, l;
	char key[1000], c1, c2;
	
	l = strlen(guess) / 2;
	for (i = 0; i < 26; i++) {
		key[i] = '-';
	}
	key[26] = 0;
	for (i = 0; i < l; i++) {
		c1 = guess[2 * i + 0];
		c2 = guess[2 * i + 1];
		c1 = lower_case(c1);
		c2 = lower_case(c2);
		cout << c1 << " -> " << c2 << endl;
		j = c1 - 'a';
		//cout << "j=" << j << endl;
		key[j] = c2;
	}
	substition_cipher(ctext, ptext, key);
}

void analyze(char *text)
{
	int mult[100];
	int mult2[100];
	
	single_frequencies(text, mult);
	cout << "single frequencies:" << endl;
	print_frequencies(mult);
	cout << endl;

	double_frequencies(text, mult2);
	cout << "double frequencies:" << endl;
	print_frequencies(mult2);
	cout << endl;
}

double friedman_index(int *mult, int n)
{
	int i, a = 0, b = n * (n - 1);
	double d;

	for (i = 0; i < 26; i++) {
		if (mult[i] > 1) {
			a += mult[i] * (mult[i] - 1);
		}
	}
	d = (double) a / (double) b;
	return d;
}

double friedman_index_shifted(int *mult, int n, int shift)
{
	int i, ii;
	double a = 0., d;

	for (i = 0; i < 26; i++) {
		ii = i + shift;
		ii = ii % 26;
		if (mult[ii]) {
			a += letter_probability[i] * (double) mult[ii];
		}
	}
	d = (double) a / (double) n;
	return d;
}

void print_frequencies(int *mult)
{
	int i, j = 0, k = 0, h, l = 0, f_first = TRUE;
	char c;
	
	for (i = 0; i < 26; i++) {
		if (mult[i]) {
			l++;
		}
	}
	int *mult_val = new int[2 * l];
	for (i = 0; i < 26; i++) {
		if (mult[i]) {
			for (j = 0; j < k; j++) {
				if (mult_val[2 * j + 1] < mult[i]) {
					// insert here:
					for (h = k; h > j; h--) {
						mult_val[2 * h + 1] = mult_val[2 * (h - 1) + 1];
						mult_val[2 * h + 0] = mult_val[2 * (h - 1) + 0];
					}
					mult_val[2 * j + 0] = i;
					mult_val[2 * j + 1] = mult[i];
					break;
				}
			}
			if (j == k) {
				mult_val[2 * k + 0] = i;
				mult_val[2 * k + 1] = mult[i];
			}
			k++;
		}
	}
	
	for (i = 0; i < l; i++) {
		c = 'a' + mult_val[2 * i + 0];
		j = mult_val[2 * i + 1];
		if (!f_first) {
			cout << ", ";
		}
		cout << c;
		if (j > 1) {
			cout << "^" << j;
		}
		f_first = FALSE;
	}
}

void single_frequencies(char *text, int *mult)
{
	int i, l;
	
	l = strlen(text);
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < l; i++) {
		mult[text[i] - 'a']++;
	}
}

void single_frequencies2(char *text, int stride, int n, int *mult)
{
	int i;
	
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < n; i++) {
		mult[text[i * stride] - 'a']++;
	}
}

void double_frequencies(char *text, int *mult)
{
	int i, l;
	
	l = strlen(text);
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < l - 1; i++) {
		if (text[i] == text[i + 1]) {
			mult[text[i] - 'a']++;
		}
	}
}

void substition_cipher(char *ptext, char *ctext, char *key)
{
	int i, l;
	char c;
	
	cout << "applying key:" << endl;
	for (i = 0; i < 26; i++) {
		c = 'a' + i;
		cout << c;
	}
	cout << endl;
	cout << key << endl;
	l = strlen(ptext);
	for (i = 0; i < l; i++) {
		if (is_alnum(ptext[i])) {
			ctext[i] = key[lower_case(ptext[i]) - 'a'];
		}
		else {
			ctext[i] = ptext[i];
		}
	}
	ctext[l] = 0;
}

char lower_case(char c)
{
	if (c >= 'A' && c <= 'Z') {
		c = c - ('A' - 'a');
		return c;
	}
	else if (c >= 'a' && c <= 'z') {
		return c;
	}
	else {
		//cout << "illegal character " << c << endl;
		//exit(1);
		return c;
	}
}

char upper_case(char c)
{
	if (c >= 'a' && c <= 'z') {
		c = c + ('A' - 'a');
		return c;
	}
	else if (c >= 'A' && c <= 'Z') {
		return c;
	}
	else {
		//cout << "illegal character " << c << endl;
		//exit(1);
		return c;
	}
}

char is_alnum(char c)
{
	if (c >= 'A' && c <= 'Z') {
		return TRUE;
	}
	if (c >= 'a' && c <= 'z') {
		return TRUE;
	}
	return FALSE;
}

void get_random_permutation(char *p)
{
	char digits[100];
	int i, j, k, l;
	os_interface OS;
	
	for (i = 0; i < 26; i++) {
		digits[i] = 'a' + i;
	}
	digits[26] = 0;
	for (i = 0; i < 26; i++) {
		l = strlen(digits);
		j = OS.random_integer(l);
		p[i] = digits[j];
		for (k = j + 1; k <= l; k++) {
			digits[k - 1] = digits[k];
		}
		l--;
	}
}

