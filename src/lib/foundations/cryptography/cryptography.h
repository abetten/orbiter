/*
 * cryptography.h
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_
#define SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_



namespace orbiter {
namespace foundations {



// #############################################################################
// cryptography_domain.cpp
// #############################################################################

//! a collection of functions related to cryptography

class cryptography_domain {

public:

	cryptography_domain();
	~cryptography_domain();
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

	void make_affine_sequence(int a, int c, int m, int verbose_level);
	void make_2D_plot(int *orbit, int orbit_len, int cnt,
			int m, int a, int c, int verbose_level);
	void do_random_last(int random_nb, int verbose_level);
	void do_random(int random_nb, std::string &fname_csv, int verbose_level);

	void do_EC_Koblitz_encoding(int q,
			int EC_b, int EC_c, int EC_s,
			std::string &pt_text, std::string &EC_message,
			int verbose_level);
	void do_EC_points(int q, int EC_b, int EC_c, int verbose_level);
	int EC_evaluate_RHS(finite_field *F, int EC_b, int EC_c, int x);
	// evaluates x^3 + bx + c
	void do_EC_add(int q, int EC_b, int EC_c,
			std::string &pt1_text, std::string &pt2_text, int verbose_level);
	void do_EC_cyclic_subgroup(int q, int EC_b, int EC_c,
			std::string &pt_text, int verbose_level);
	void do_EC_multiple_of(int q, int EC_b, int EC_c,
			std::string &pt_text, int n, int verbose_level);
	void do_EC_discrete_log(int q, int EC_b, int EC_c,
			std::string &base_pt_text, std::string &pt_text, int verbose_level);
	void do_EC_baby_step_giant_step(int EC_q, int EC_b, int EC_c,
			std::string &EC_bsgs_G, int EC_bsgs_N,
			std::string &EC_bsgs_cipher_text,
			int verbose_level);
	void do_EC_baby_step_giant_step_decode(int EC_q, int EC_b, int EC_c,
			std::string &EC_bsgs_A, int EC_bsgs_N,
			std::string &EC_bsgs_cipher_text_T, std::string &EC_bsgs_keys,
			int verbose_level);
	void do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
			int RSA_block_size, std::string &RSA_encrypt_text, int verbose_level);
	void do_RSA(long int RSA_d, long int RSA_m,
			std::string &RSA_text, int verbose_level);

	void NTRU_encrypt(int N, int p, int q,
			std::string &H_coeffs, std::string &R_coeffs, std::string &Msg_coeffs,
			int verbose_level);
	void polynomial_center_lift(std::string &A_coeffs, int q,
			int verbose_level);
	void polynomial_reduce_mod_p(std::string &A_coeffs, int p,
			int verbose_level);

	void do_jacobi(int jacobi_top, int jacobi_bottom, int verbose_level);
	void do_solovay_strassen(int p, int a, int verbose_level);
	void do_miller_rabin(int p, int nb_times, int verbose_level);
	void do_fermat_test(int p, int nb_times, int verbose_level);
	void do_find_pseudoprime(int nb_digits, int nb_fermat,
			int nb_miller_rabin, int nb_solovay_strassen, int verbose_level);
	void do_find_strong_pseudoprime(int nb_digits,
			int nb_fermat, int nb_miller_rabin, int verbose_level);
	void do_miller_rabin_text(std::string &number_text,
			int nb_miller_rabin, int verbose_level);
	void quadratic_sieve(int n, int factorbase,
			int x0, int verbose_level);
	void calc_log2(std::vector<int> &primes,
			std::vector<int> &primes_log2, int verbose_level);
	void square_root(std::string &square_root_number, int verbose_level);
	void square_root_mod(std::string &square_root_number,
			std::string &mod_number, int verbose_level);
	void reduce_primes(std::vector<int> &primes,
			longinteger_object &M,
			int &f_found_small_factor, int &small_factor,
			int verbose_level);
	void do_sift_smooth(int sift_smooth_from,
			int sift_smooth_len,
			std::string &sift_smooth_factor_base, int verbose_level);
	void do_discrete_log(long int y, long int a, long int p, int verbose_level);
	void do_primitive_root(long int p, int verbose_level);
	void do_inverse_mod(long int a, long int n, int verbose_level);
	void do_extended_gcd(int a, int b, int verbose_level);
	void do_power_mod(long int a, long int k, long int n, int verbose_level);

};

}}


#endif /* SRC_LIB_FOUNDATIONS_CRYPTOGRAPHY_CRYPTOGRAPHY_H_ */
