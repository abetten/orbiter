/*
 * cryptography.h
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_
#define SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace cryptography {



// #############################################################################
// cryptography_domain.cpp
// #############################################################################

//! a collection of functions related to cryptography

class cryptography_domain {

public:

	cryptography_domain();
	~cryptography_domain();
	void affine_cipher(
			std::string &ptext,
			std::string &ctext, int a, int b);
	void affine_decipher(
			std::string &ctext,
			std::string &ptext, std::string &guess);
	void vigenere_cipher(
			std::string &ptext,
			std::string &ctext, std::string &key);
	void vigenere_decipher(
			std::string &ctext,
			std::string &ptext, std::string &key);
	void vigenere_analysis(
			std::string &ctext);
	void vigenere_analysis2(
			std::string &ctext, int key_length);
	int kasiski_test(
			std::string &ctext, int threshold);
	void print_candidates(
			std::string &ctext,
			int i, int h, int nb_candidates, int *candidates);
	void print_set(
			int l, int *s);
	void print_on_top(
			std::string &text1,
			std::string &text2);
	void decipher(
			std::string &ctext,
			std::string &ptext, std::string &guess);
	void analyze(
			std::string &text);
	double friedman_index(
			int *mult, int n);
	double friedman_index_shifted(
			int *mult, int n, int shift);
	void print_frequencies(
			int *mult);
	void single_frequencies(
			std::string &text, int *mult);
	void single_frequencies2(
			std::string &text, int stride, int n, int *mult);
	void double_frequencies(
			std::string &text, int *mult);
	void substition_cipher(
			std::string &ptext,
			std::string &ctext, std::string &key);
	char lower_case(
			char c);
	char upper_case(
			char c);
	char is_alnum(
			char c);
	void get_random_permutation(
			std::string &p);

	void make_affine_sequence(
			int a, int c, int m, int verbose_level);
	void make_2D_plot(
			int *orbit, int orbit_len, int cnt,
			int m, int a, int c, int verbose_level);
	void do_random_last(
			int random_nb, int verbose_level);
	void do_random(
			int random_nb,
			std::string &fname_csv, int verbose_level);

	void do_EC_Koblitz_encoding(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c, int EC_s,
			std::string &pt_text, std::string &EC_message,
			int verbose_level);
	void do_EC_points(
			algebra::field_theory::finite_field *F, std::string &label,
			int EC_b, int EC_c, int verbose_level);
	int EC_evaluate_RHS(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c, int x);
	// evaluates x^3 + bx + c
	void do_EC_add(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &pt1_text,
			std::string &pt2_text, int verbose_level);
	void do_EC_cyclic_subgroup(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &pt_text, int verbose_level);
	void do_EC_multiple_of(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &pt_text, int n, int verbose_level);
	void do_EC_discrete_log(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &base_pt_text,
			std::string &pt_text, int verbose_level);
	void do_EC_baby_step_giant_step(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &EC_bsgs_G, int EC_bsgs_N,
			std::string &EC_bsgs_cipher_text,
			int verbose_level);
	void do_EC_baby_step_giant_step_decode(
			algebra::field_theory::finite_field *F,
			int EC_b, int EC_c,
			std::string &EC_bsgs_A, int EC_bsgs_N,
			std::string &EC_bsgs_cipher_text_T,
			std::string &EC_bsgs_keys,
			int verbose_level);
	void do_RSA_encrypt_text(
			long int RSA_d, long int RSA_m,
			int RSA_block_size,
			std::string &RSA_encrypt_text,
			int verbose_level);
	void do_RSA(
			long int RSA_d,
			long int RSA_m, int RSA_block_size,
			std::string &RSA_text,
			int verbose_level);

	void NTRU_encrypt(
			int N, int p,
			algebra::field_theory::finite_field *Fq,
			std::string &H_coeffs,
			std::string &R_coeffs,
			std::string &Msg_coeffs,
			int verbose_level);
	void polynomial_center_lift(
			std::string &A_coeffs,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	void polynomial_reduce_mod_p(
			std::string &A_coeffs,
			algebra::field_theory::finite_field *F,
			int verbose_level);

	void do_solovay_strassen(
			int p, int a, int verbose_level);
	void do_miller_rabin(
			int p, int nb_times, int verbose_level);
	void do_fermat_test(
			int p, int nb_times, int verbose_level);
	void do_find_pseudoprime(
			int nb_digits, int nb_fermat,
			int nb_miller_rabin,
			int nb_solovay_strassen,
			int verbose_level);
	void do_find_strong_pseudoprime(
			int nb_digits,
			int nb_fermat, int nb_miller_rabin,
			int verbose_level);
	void do_miller_rabin_text(
			std::string &number_text,
			int nb_miller_rabin, int verbose_level);
	void quadratic_sieve(
			int n, int factorbase,
			int x0, int verbose_level);
	void calc_log2(
			std::vector<int> &primes,
			std::vector<int> &primes_log2,
			int verbose_level);
	void reduce_primes(
			std::vector<int> &primes,
			algebra::ring_theory::longinteger_object &M,
			int &f_found_small_factor, int &small_factor,
			int verbose_level);
	void do_sift_smooth(
			int sift_smooth_from,
			int sift_smooth_len,
			std::string &sift_smooth_factor_base,
			int verbose_level);


	void calc_roots(
			algebra::ring_theory::longinteger_object &M,
			algebra::ring_theory::longinteger_object &sqrtM,
		std::vector<int> &primes,
		std::vector<int> &R1, std::vector<int> &R2,
		int verbose_level);
	void Quadratic_Sieve(
		int factorbase,
		int f_mod, int mod_n, int mod_r, int x0,
		int n, algebra::ring_theory::longinteger_object &M,
		algebra::ring_theory::longinteger_object &sqrtM,
		std::vector<int> &primes,
		std::vector<int> &primes_log2,
		std::vector<int> &R1, std::vector<int> &R2,
		std::vector<int> &X,
		int verbose_level);
	int quadratic_sieve(
			algebra::ring_theory::longinteger_object& M,
			algebra::ring_theory::longinteger_object& sqrtM,
		std::vector<int> &primes,
		std::vector<int> &primes_log2,
		std::vector<int> &R1, std::vector<int> &R2,
		int from, int to,
		int ll, std::vector<int> &X, int verbose_level);
	int factor_over_factor_base(
			algebra::ring_theory::longinteger_object &x,
			std::vector<int> &primes,
			std::vector<int> &factor_idx,
			std::vector<int> &factor_exp,
			int verbose_level);
	int factor_over_factor_base2(
			algebra::ring_theory::longinteger_object &x,
			std::vector<int> &primes,
			std::vector<int> &exponents,
			int verbose_level);

	void find_probable_prime_above(
			algebra::ring_theory::longinteger_object &a,
		int nb_solovay_strassen_tests, int f_miller_rabin_test,
		int verbose_level);
	int solovay_strassen_is_prime(
			algebra::ring_theory::longinteger_object &n,
			int nb_tests, int verbose_level);
	int solovay_strassen_is_prime_single_test(
			algebra::ring_theory::longinteger_object &n,
			int verbose_level);
	int fermat_test_iterated_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &P,
			int nb_times,
			int verbose_level);
	int fermat_test_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &n,
			algebra::ring_theory::longinteger_object &a,
		int verbose_level);
	int solovay_strassen_test(
			algebra::ring_theory::longinteger_object &n,
			algebra::ring_theory::longinteger_object &a,
		int verbose_level);
	int solovay_strassen_test_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &n,
			algebra::ring_theory::longinteger_object &a,
		int verbose_level);
	int solovay_strassen_test_iterated_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &P,
			int nb_times,
			int verbose_level);
	// returns true is the test is conclusive,
	// i.e. if the number is not prime.
	int miller_rabin_test(
			algebra::ring_theory::longinteger_object &n,
			int verbose_level);
	int miller_rabin_test_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &n,
			int iteration, int verbose_level);
	int miller_rabin_test_iterated_with_latex_key(
			std::ostream &ost,
			algebra::ring_theory::longinteger_object &P, int nb_times,
			int verbose_level);
	// returns true is the test is conclusive,
	// i.e. if the number is not prime.
	void get_k_bit_random_pseudoprime(
			algebra::ring_theory::longinteger_object &n, int k,
		int nb_tests_solovay_strassen,
		int f_miller_rabin_test, int verbose_level);
	void RSA_setup(
			algebra::ring_theory::longinteger_object &n,
			algebra::ring_theory::longinteger_object &p,
			algebra::ring_theory::longinteger_object &q,
			algebra::ring_theory::longinteger_object &a,
			algebra::ring_theory::longinteger_object &b,
		int nb_bits,
		int nb_tests_solovay_strassen,
		int f_miller_rabin_test,
		int verbose_level);
	void do_babystep_giantstep(
			long int p, long int g, long int h,
			int f_latex, std::ostream &ost,
			int verbose_level);

};

}}}}




#endif /* SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_ */
