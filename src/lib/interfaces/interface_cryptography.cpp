// interface_cryptography.cpp
//
// by Anton Betten
//
// Colorado State University
//
// for M360 mathematics of information security, Fall 03
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace interfaces {


interface_cryptography::interface_cryptography()
{
	//cout << "interface_cryptography::interface_cryptography" << endl;
	//cout << "sizeof(interface_cryptography)=" << sizeof(interface_cryptography) << endl;
	f_cipher = FALSE;
	t = no_cipher_type;
	//cout << "interface_cryptography::interface_cryptography 0" << endl;
	f_decipher = FALSE;
	//cout << "interface_cryptography::interface_cryptography 00a" << endl;
	f_analyze = FALSE;
	//cout << "interface_cryptography::interface_cryptography 00b" << endl;
	f_kasiski = FALSE;
	f_avk = FALSE;
	key_length = 0;
	threshold = 0;
	//cout << "interface_cryptography::interface_cryptography 00c" << endl;
	affine_a = 0;
	//cout << "interface_cryptography::interface_cryptography 00d" << endl;
	affine_b = 0;
	//char ptext[10000];
	//char ctext[10000];
	//char guess[10000];
	//char key[1000];
	//cout << "interface_cryptography::interface_cryptography 00e" << endl;
	f_RSA = FALSE;
	RSA_d = 0;
	RSA_m = 0;
	RSA_text = NULL;
	//cout << "interface_cryptography::interface_cryptography 00f" << endl;
	f_primitive_root = FALSE;
	primitive_root_p = 0;
	f_inverse_mod = FALSE;
	inverse_mod_a = 0;
	inverse_mod_n = 0;
	//cout << "interface_cryptography::interface_cryptography 00g" << endl;
	f_extended_gcd = FALSE;
	extended_gcd_a = 0;
	extended_gcd_b = 0;
	f_power_mod = FALSE;
	power_mod_a = 0;
	power_mod_k = 0;
	power_mod_n = 0;
	//cout << "interface_cryptography::interface_cryptography 00h" << endl;
	f_discrete_log = FALSE;
	//cout << "interface_cryptography::interface_cryptography 0b" << endl;
	discrete_log_y = 0;
	discrete_log_a = 0;
	discrete_log_m = 0;
	f_RSA_setup = FALSE;
	RSA_setup_nb_bits = 0;
	RSA_setup_nb_tests_solovay_strassen = 0;
	RSA_setup_f_miller_rabin_test = 0;
	f_RSA_encrypt_text = FALSE;
	RSA_block_size = 0;
	RSA_encrypt_text = NULL;
	f_sift_smooth = FALSE;
	sift_smooth_from = 0;
	sift_smooth_len = 0;
	//cout << "interface_cryptography::interface_cryptography 1" << endl;
	sift_smooth_factor_base = NULL;
	f_square_root = FALSE;
	square_root_number = NULL;
	f_square_root_mod = FALSE;
	square_root_mod_a = NULL;
	square_root_mod_m = NULL;
	//cout << "interface_cryptography::interface_cryptography 1a" << endl;
	f_quadratic_sieve = FALSE;
	quadratic_sieve_n = 0;
	quadratic_sieve_factorbase = 0;
	quadratic_sieve_x0 = 0;
	//cout << "interface_cryptography::interface_cryptography 1b" << endl;
	f_jacobi = FALSE;
	jacobi_top = 0;
	jacobi_bottom = 0;
	//cout << "interface_cryptography::interface_cryptography 1c" << endl;
	f_solovay_strassen = FALSE;
	solovay_strassen_p = 0;
	solovay_strassen_a = 0;
	//cout << "interface_cryptography::interface_cryptography 1d" << endl;
	f_miller_rabin = FALSE;
	miller_rabin_p = 0;
	miller_rabin_nb_times = 0;
	//cout << "interface_cryptography::interface_cryptography 1e" << endl;
	f_fermat_test = FALSE;
	fermat_test_p = 0;
	fermat_test_nb_times = 0;
	//cout << "interface_cryptography::interface_cryptography 1f" << endl;
	f_find_pseudoprime = FALSE;
	find_pseudoprime_nb_digits = 0;
	find_pseudoprime_nb_fermat = 0;
	find_pseudoprime_nb_miller_rabin = 0;
	find_pseudoprime_nb_solovay_strassen = 0;
	//cout << "interface_cryptography::interface_cryptography 1g" << endl;
	f_find_strong_pseudoprime = FALSE;
	f_miller_rabin_text = FALSE;
	miller_rabin_text_nb_times = 0;
	miller_rabin_number_text = NULL;
	//cout << "interface_cryptography::interface_cryptography 1h" << endl;
	f_random = FALSE;
	random_nb = 0;
	//random_fname_csv = NULL;
	f_random_last = FALSE;
	random_last_nb = 0;
	//cout << "interface_cryptography::interface_cryptography 1i" << endl;
	f_affine_sequence = FALSE;
	affine_sequence_a = 0;
	//cout << "interface_cryptography::interface_cryptography 2" << endl;
	affine_sequence_c = 0;
	affine_sequence_m = 0;
	f_EC_Koblitz_encoding = FALSE;
	EC_s = 0;
	EC_message = NULL;
	f_EC_points = FALSE;
	f_EC_add = FALSE;
	EC_pt1_text = NULL;
	EC_pt2_text = NULL;
	f_EC_cyclic_subgroup = FALSE;
	EC_q = 0;
	EC_b = 0;
	EC_c = 0;
	EC_pt_text = NULL;
	f_EC_multiple_of = FALSE;
	EC_multiple_of_n = 0;
	f_EC_discrete_log = FALSE;
	EC_discrete_log_pt_text = NULL;
	f_EC_baby_step_giant_step = FALSE;
	EC_bsgs_G = NULL;
	EC_bsgs_N = 0;
	EC_bsgs_cipher_text = NULL;
	f_EC_baby_step_giant_step_decode = FALSE;
	EC_bsgs_A = NULL;
	EC_bsgs_keys = NULL;


	//cout << "interface_cryptography::interface_cryptography done" << endl;
	f_NTRU_encrypt = FALSE;
	NTRU_encrypt_N = 0;
	NTRU_encrypt_p = 0;
	NTRU_encrypt_q = 0;
	//NTRU_encrypt_H, NTRU_encrypt_R, NTRU_encrypt_Msg
	f_polynomial_center_lift = FALSE;
	polynomial_center_lift_q = 0;
	//polynomial_center_lift_A
	f_polynomial_reduce_mod_p = FALSE;
	polynomial_reduce_mod_p = 0;
	//polynomial_reduce_mod_p_A;

}




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




void interface_cryptography::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-cs") == 0) {
		cout << "-cs <ptext>" << endl;
	}
	else if (strcmp(argv[i], "-cv") == 0) {
		cout << "-cv <ptext> <key>" << endl;
	}
	else if (strcmp(argv[i], "-ca") == 0) {
		cout << "-ca <ptext> <int : a> <int : b>" << endl;
	}
	else if (strcmp(argv[i], "-as") == 0) {
		cout << "-as <ctext>" << endl;
	}
	else if (strcmp(argv[i], "-av") == 0) {
		cout << "-av <ctext>" << endl;
	}
	else if (strcmp(argv[i], "-avk") == 0) {
		cout << "-avk <ctext> <int : key_length>" << endl;
	}
	else if (strcmp(argv[i], "-kasiski") == 0) {
		cout << "-kasiski <ctext> <int : threshold>" << endl;
	}
	else if (strcmp(argv[i], "-ds") == 0) {
		cout << "-ds <ctext> <guess>" << endl;
	}
	else if (strcmp(argv[i], "-dv") == 0) {
		cout << "-dv <ctext> <key>" << endl;
	}
	else if (strcmp(argv[i], "-da") == 0) {
		cout << "-da <ctext> <guess>" << endl;
	}
	else if (strcmp(argv[i], "-RSA") == 0) {
		cout << "-RSA <int : d> <int : m> <text>" << endl;
	}
	else if (strcmp(argv[i], "-RSA_encrypt_text") == 0) {
		cout << "-RSA_encrypt_text <int : d> <int : m> <int : block_size> <text>" << endl;
	}
	else if (strcmp(argv[i], "-RSA_setup") == 0) {
		cout << "-RSA_setup <int : nb_bits> <int : nb_tests_solovay_strassen> <int : f_miller_rabin_test>" << endl;
	}
	else if (strcmp(argv[i], "-primitive_root") == 0) {
		cout << "-primitive_root <int : p>" << endl;
	}
	else if (strcmp(argv[i], "-inverse_mod") == 0) {
		cout << "-primitive_root <int : a> <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-extended_gcd") == 0) {
		cout << "-extended_gcd <int : a> <int : b>" << endl;
	}
	else if (strcmp(argv[i], "-power_mod") == 0) {
		cout << "-power_mod <int : a> <int : k> <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-discrete_log") == 0) {
		cout << "-discrete_log <int : y> <int : a> <int : m>" << endl;
	}
	else if (strcmp(argv[i], "-sift_smooth") == 0) {
		cout << "-sift_smooth <int : from> <int : ken> <string : factor_base>" << endl;
	}
	else if (strcmp(argv[i], "-square_root") == 0) {
		cout << "-square_root <int : number>" << endl;
	}
	else if (strcmp(argv[i], "-square_root_mod") == 0) {
		cout << "-square_root_mod <int : a> <int : m>" << endl;
	}
	else if (strcmp(argv[i], "-quadratic_sieve") == 0) {
		cout << "-quadratic_sieve <int : n> <string : factor_base> <int : x0>" << endl;
	}
	else if (strcmp(argv[i], "-jacobi") == 0) {
		cout << "-jacobi <int : top> <int : bottom>" << endl;
	}
	else if (strcmp(argv[i], "-solovay_strassen") == 0) {
		cout << "-solovay_strassen <int : a> <int : p>" << endl;
	}
	else if (strcmp(argv[i], "-miller_rabin") == 0) {
		cout << "-miller_rabin <int : p> <int : nb_times>" << endl;
	}
	else if (strcmp(argv[i], "-fermat_test") == 0) {
		cout << "-fermat_test <int : p> <int : nb_times>" << endl;
	}
	else if (strcmp(argv[i], "-find_pseudoprime") == 0) {
		cout << "-find_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin> <int : nb_solovay_strassen>" << endl;
	}
	else if (strcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		cout << "-find_strong_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin>" << endl;
	}
	else if (strcmp(argv[i], "-miller_rabin_text") == 0) {
		cout << "-fermat_test <int : nb_times> <string : number>" << endl;
	}
	else if (strcmp(argv[i], "-random") == 0) {
		cout << "-random <int : nb_times> <string : fname_csv>" << endl;
	}
	else if (strcmp(argv[i], "-random_last") == 0) {
		cout << "-random_last <int : nb_times>" << endl;
	}
	else if (strcmp(argv[i], "-affine_sequence") == 0) {
		cout << "-affine_sequence <int : a> <int : c> <int : m>" << endl;
	}
	else if (strcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
		cout << "-EC_points <int : q> <int : b> <int : c> <int : s> <string : pt_G> <string : message> " << endl;
	}
	else if (strcmp(argv[i], "-EC_points") == 0) {
		cout << "-EC_points <int : q> <int : b> <int : c>" << endl;
	}
	else if (strcmp(argv[i], "-EC_add") == 0) {
		cout << "-EC_add <int : q> <int : b> <int : c> <string : pt1> <string : pt2>" << endl;
	}
	else if (strcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
		cout << "-EC_cyclic_subgroup <int : q> <int : b> <int : c> <string : pt>" << endl;
	}
	else if (strcmp(argv[i], "-EC_multiple_of") == 0) {
		cout << "-EC_multiple_of <int : q> <int : b> <int : c> <string : pt> <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-EC_discrete_log") == 0) {
		cout << "-EC_discrete_log <int : q> <int : b> <int : c> <string : base_pt> <int : n> <string : pt>" << endl;
	}
	else if (strcmp(argv[i], "-EC_bsgs") == 0) {
		cout << "-EC_bsgs <int : q> <int : b> <int : c> <string : G> <int : N> <string : cipher_text_R>" << endl;
	}
	else if (strcmp(argv[i], "-EC_bsgs_decode") == 0) {
		cout << "-EC_bsgs_decode <int : q> <int : b> <int : c> <string : A> <int : N> <string : cipher_text_T> <string : keys> " << endl;
	}
	else if (strcmp(argv[i], "-NTRU_encrypt") == 0) {
		cout << "-NTRU_encrypt <int : N> <int : p> <int : q> <string : H> <string : R> <string : Msg>" << endl;
	}
	else if (strcmp(argv[i], "-polynomial_center_lift") == 0) {
		cout << "-polynomial_center_lift <int : q> <string : A>" << endl;
	}
	else if (strcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
		cout << "-polynomial_reduce_mod_p <int : p> <string : A>" << endl;
	}
#if 0
	else if (strcmp(argv[i], "-ntt") == 0) {
		cout << "-ntt <int : t> <int : q>" << endl;
	}
#endif
}

int interface_cryptography::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_cryptography::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (strcmp(argv[i], "-cs") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-cv") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-ca") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-as") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-av") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-avk") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-kasiski") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-ds") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-dv") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-da") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-RSA") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-RSA_encrypt_text") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-RSA_setup") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-primitive_root") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-inverse_mod") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-extended_gcd") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-power_mod") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-discrete_log") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-sift_smooth") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-square_root") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-square_root_mod") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-quadratic_sieve") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-jacobi") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-solovay_strassen") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-miller_rabin") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-fermat_test") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-find_pseudoprime") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-miller_rabin_text") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-random") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-random_last") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-affine_sequence") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_points") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_add") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_multiple_of") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_discrete_log") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_bsgs") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-EC_bsgs_decode") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-NTRU_encrypt") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-polynomial_center_lift") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
		return true;
	}
#if 0
	else if (strcmp(argv[i], "-ntt") == 0) {
		return true;
	}
#endif
	return false;
}

void interface_cryptography::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_cryptography::read_arguments" << endl;
	}
	//return 0;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-cs") == 0) {
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
		else if (strcmp(argv[i], "-primitive_root") == 0) {
			f_primitive_root = TRUE;
			primitive_root_p = atol(argv[++i]);
			cout << "-primitive_root " << primitive_root_p << endl;
		}
		else if (strcmp(argv[i], "-inverse_mod") == 0) {
			f_inverse_mod = TRUE;
			inverse_mod_a = atol(argv[++i]);
			inverse_mod_n = atol(argv[++i]);
			cout << "-inverse_mod " << inverse_mod_a << " " << inverse_mod_n << endl;
		}
		else if (strcmp(argv[i], "-extended_gcd") == 0) {
			f_extended_gcd = TRUE;
			extended_gcd_a = atol(argv[++i]);
			extended_gcd_b = atol(argv[++i]);
			cout << "-extended_gcd " << extended_gcd_a << " " << extended_gcd_b << endl;
		}
		else if (strcmp(argv[i], "-power_mod") == 0) {
			f_power_mod = TRUE;
			power_mod_a = atol(argv[++i]);
			power_mod_k = atol(argv[++i]);
			power_mod_n = atol(argv[++i]);
			cout << "-power_mod " << power_mod_a << " " << power_mod_k << " " << power_mod_n << endl;
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
		else if (strcmp(argv[i], "-jacobi") == 0) {
			f_jacobi = TRUE;
			jacobi_top = atoi(argv[++i]);
			jacobi_bottom = atoi(argv[++i]);
			cout << "-jacobi " << jacobi_top << " "
					<< jacobi_bottom << endl;
		}
		else if (strcmp(argv[i], "-solovay_strassen") == 0) {
			f_solovay_strassen = TRUE;
			solovay_strassen_p = atoi(argv[++i]);
			solovay_strassen_a = atoi(argv[++i]);
			cout << "-solovay_strassen " << solovay_strassen_p << " "
					<< solovay_strassen_a << endl;
		}
		else if (strcmp(argv[i], "-miller_rabin") == 0) {
			f_miller_rabin = TRUE;
			miller_rabin_p = atoi(argv[++i]);
			miller_rabin_nb_times = atoi(argv[++i]);
			cout << "-miller_rabin " << miller_rabin_p << " " << miller_rabin_nb_times << endl;
		}
		else if (strcmp(argv[i], "-fermat_test") == 0) {
			f_fermat_test = TRUE;
			fermat_test_p = atoi(argv[++i]);
			fermat_test_nb_times = atoi(argv[++i]);
			cout << "-fermat_test " << fermat_test_p << " " << fermat_test_nb_times << endl;
		}
		else if (strcmp(argv[i], "-find_pseudoprime") == 0) {
			f_find_pseudoprime = TRUE;
			find_pseudoprime_nb_digits = atoi(argv[++i]);
			find_pseudoprime_nb_fermat = atoi(argv[++i]);
			find_pseudoprime_nb_miller_rabin = atoi(argv[++i]);
			find_pseudoprime_nb_solovay_strassen = atoi(argv[++i]);
			cout << "-find_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< " " << find_pseudoprime_nb_solovay_strassen << endl;
		}
		else if (strcmp(argv[i], "-find_strong_pseudoprime") == 0) {
			f_find_strong_pseudoprime = TRUE;
			find_pseudoprime_nb_digits = atoi(argv[++i]);
			find_pseudoprime_nb_fermat = atoi(argv[++i]);
			find_pseudoprime_nb_miller_rabin = atoi(argv[++i]);
			cout << "-find_strong_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< endl;
		}
		else if (strcmp(argv[i], "-miller_rabin_text") == 0) {
			f_miller_rabin_text = TRUE;
			miller_rabin_text_nb_times = atoi(argv[++i]);
			miller_rabin_number_text = argv[++i];
			cout << "-miller_rabin " << miller_rabin_text_nb_times
					<< " " << miller_rabin_number_text
					<< endl;
		}
		else if (strcmp(argv[i], "-random") == 0) {
			f_random = TRUE;
			random_nb = atoi(argv[++i]);
			random_fname_csv.assign(argv[++i]);
			cout << "-random " << random_nb << " " << random_fname_csv << endl;
		}
		else if (strcmp(argv[i], "-random_last") == 0) {
			f_random_last = TRUE;
			random_last_nb = atoi(argv[++i]);
			cout << "-random_last " << random_last_nb << endl;
		}
		else if (strcmp(argv[i], "-affine_sequence") == 0) {
			f_affine_sequence = TRUE;
			affine_sequence_a = atoi(argv[++i]);
			affine_sequence_c = atoi(argv[++i]);
			affine_sequence_m = atoi(argv[++i]);
			cout << "-affine_sequence " << affine_sequence_a
					<< " " << affine_sequence_c << " " << affine_sequence_m << endl;
		}
		else if (strcmp(argv[i], "-EC_Koblitz_encoding") == 0) {
			f_EC_Koblitz_encoding = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_s = atoi(argv[++i]);
			EC_pt_text = argv[++i];
			EC_message = argv[++i];
			cout << "-EC_Koblitz_encoding " << EC_q
					<< " " << EC_b << " " << EC_c << " " << EC_s << " "
					<< EC_pt_text << " " << EC_message << endl;
		}
		else if (strcmp(argv[i], "-EC_points") == 0) {
			f_EC_points = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			cout << "-EC_points " << EC_q
					<< " " << EC_b << " " << EC_c << endl;
		}
		else if (strcmp(argv[i], "-EC_add") == 0) {
			f_EC_add = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_pt1_text = argv[++i];
			EC_pt2_text = argv[++i];
			cout << "-EC_add " << EC_q
					<< " " << EC_b << " " << EC_c << " " << EC_pt1_text << " " << EC_pt2_text << endl;
		}
		else if (strcmp(argv[i], "-EC_cyclic_subgroup") == 0) {
			f_EC_cyclic_subgroup = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_pt_text = argv[++i];
			cout << "-EC_cyclic_subgroup " << EC_q
					<< " " << EC_b << " " << EC_c << " " << EC_pt_text << endl;
		}
		else if (strcmp(argv[i], "-EC_multiple_of") == 0) {
			f_EC_multiple_of = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_pt_text = argv[++i];
			EC_multiple_of_n = atoi(argv[++i]);
			cout << "-EC_multiple_of " << EC_q
					<< " " << EC_b << " " << EC_c << " " << EC_pt_text
					<< " " << EC_multiple_of_n << endl;
		}
		else if (strcmp(argv[i], "-EC_discrete_log") == 0) {
			f_EC_discrete_log = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_pt_text = argv[++i];
			EC_discrete_log_pt_text = argv[++i];
			cout << "-EC_discrete_log " << EC_q
					<< " " << EC_b << " " << EC_c << " " << EC_pt_text << " "
					<< EC_discrete_log_pt_text << endl;
		}
		else if (strcmp(argv[i], "-EC_bsgs") == 0) {
			f_EC_baby_step_giant_step = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_bsgs_G = argv[++i];
			EC_bsgs_N = atoi(argv[++i]);
			EC_bsgs_cipher_text = argv[++i];
			cout << "-EC_baby_step_giant_step " << EC_q
					<< " " << EC_b << " " << EC_c << " "
					<< EC_bsgs_G << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << endl;
		}
		else if (strcmp(argv[i], "-EC_bsgs_decode") == 0) {
			f_EC_baby_step_giant_step_decode = TRUE;
			EC_q = atoi(argv[++i]);
			EC_b = atoi(argv[++i]);
			EC_c = atoi(argv[++i]);
			EC_bsgs_A = argv[++i];
			EC_bsgs_N = atoi(argv[++i]);
			EC_bsgs_cipher_text = argv[++i];
			EC_bsgs_keys = argv[++i];
			cout << "-EC_baby_step_giant_step_decode " << EC_q
					<< " " << EC_b << " " << EC_c << " "
					<< EC_bsgs_A << " "
					<< EC_bsgs_N << " "
					<< EC_bsgs_cipher_text << " "
					<< EC_bsgs_keys << " "
					<< endl;
		}
		else if (strcmp(argv[i], "-NTRU_encrypt") == 0) {
			f_NTRU_encrypt = TRUE;
			NTRU_encrypt_N = atoi(argv[++i]);
			NTRU_encrypt_p = atoi(argv[++i]);
			NTRU_encrypt_q = atoi(argv[++i]);
			NTRU_encrypt_H.assign(argv[++i]);
			NTRU_encrypt_R.assign(argv[++i]);
			NTRU_encrypt_Msg.assign(argv[++i]);
			cout << "-polynomial_mult_mod " << NTRU_encrypt_N
					<< " " << NTRU_encrypt_p
					<< " " << NTRU_encrypt_q
					<< " " << NTRU_encrypt_H
					<< " " << NTRU_encrypt_R
					<< " " << NTRU_encrypt_Msg << endl;
		}
		else if (strcmp(argv[i], "-polynomial_center_lift") == 0) {
			f_polynomial_center_lift = TRUE;
			polynomial_center_lift_q = atoi(argv[++i]);
			polynomial_center_lift_A.assign(argv[++i]);
			cout << "-polynomial_center_lift " << polynomial_center_lift_q
					<< " " << polynomial_center_lift_A << endl;
		}
		else if (strcmp(argv[i], "-polynomial_reduce_mod_p") == 0) {
			f_polynomial_reduce_mod_p = TRUE;
			polynomial_reduce_mod_p = atoi(argv[++i]);
			polynomial_reduce_mod_p_A.assign(argv[++i]);
			cout << "-polynomial_reduce_mod_p " << polynomial_reduce_mod_p
					<< " " << polynomial_reduce_mod_p_A << endl;
		}

#if 0
		else if (strcmp(argv[i], "-ntt") == 0) {
			f_ntt = TRUE;
			ntt_t = atoi(argv[++i]);
			ntt_q = atoi(argv[++i]);
			cout << "-ntt " << ntt_t
					<< " " << ntt_q
					<< endl;
		}
#endif
	}
}

void interface_cryptography::worker(int verbose_level)
{
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
	else if (f_primitive_root) {
		do_primitive_root(primitive_root_p, verbose_level);
	}
	else if (f_inverse_mod) {
		do_inverse_mod(inverse_mod_a, inverse_mod_n, verbose_level);
	}
	else if (f_extended_gcd) {
		do_extended_gcd(extended_gcd_a, extended_gcd_b, verbose_level);
	}
	else if (f_power_mod) {
		do_power_mod(power_mod_a, power_mod_k, power_mod_n, verbose_level);
	}
	else if (f_RSA) {
		algebra_global Algebra;

		Algebra.do_RSA(RSA_d, RSA_m, RSA_text, verbose_level);
	}
	else if (f_RSA_encrypt_text) {
		algebra_global Algebra;

		Algebra.do_RSA_encrypt_text(RSA_d, RSA_m, RSA_block_size, RSA_encrypt_text, verbose_level);
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
	else if (f_jacobi) {
		do_jacobi(jacobi_top, jacobi_bottom, verbose_level);
	}
	else if (f_solovay_strassen) {
		do_solovay_strassen(solovay_strassen_p, solovay_strassen_a, verbose_level);
	}
	else if (f_miller_rabin) {
		do_miller_rabin(miller_rabin_p, miller_rabin_nb_times, verbose_level);
	}
	else if (f_fermat_test) {
		do_fermat_test(fermat_test_p, fermat_test_nb_times, verbose_level);
	}
	else if (f_find_pseudoprime) {
		do_find_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				find_pseudoprime_nb_solovay_strassen,
				verbose_level);
	}
	else if (f_find_strong_pseudoprime) {
		do_find_strong_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				verbose_level);
	}
	else if (f_miller_rabin_text) {
		do_miller_rabin_text(
				miller_rabin_number_text, miller_rabin_text_nb_times,
				verbose_level);
	}
	else if (f_random) {
		do_random(
				random_nb, random_fname_csv,
				verbose_level);
	}
	else if (f_random_last) {
		do_random_last(
				random_last_nb,
				verbose_level);
	}
	else if (f_affine_sequence) {
		make_affine_sequence(affine_sequence_a,
				affine_sequence_c, affine_sequence_m, verbose_level);
	}
	else if (f_EC_Koblitz_encoding) {

		algebra_global Algebra;

		Algebra.do_EC_Koblitz_encoding(EC_q, EC_b, EC_c, EC_s, EC_pt_text, EC_message, verbose_level);
	}
	else if (f_EC_points) {
		algebra_global Algebra;

		Algebra.do_EC_points(EC_q, EC_b, EC_c, verbose_level);
	}
	else if (f_EC_add) {
		algebra_global Algebra;

		Algebra.do_EC_add(EC_q, EC_b, EC_c, EC_pt1_text, EC_pt2_text, verbose_level);
	}
	else if (f_EC_cyclic_subgroup) {
		algebra_global Algebra;

		Algebra.do_EC_cyclic_subgroup(EC_q, EC_b, EC_c, EC_pt_text, verbose_level);
	}
	else if (f_EC_multiple_of) {
		algebra_global Algebra;

		Algebra.do_EC_multiple_of(EC_q, EC_b, EC_c,
				EC_pt_text, EC_multiple_of_n, verbose_level);
	}
	else if (f_EC_discrete_log) {
		algebra_global Algebra;

		Algebra.do_EC_discrete_log(EC_q, EC_b, EC_c, EC_pt_text,
				EC_discrete_log_pt_text, verbose_level);
	}
	else if (f_EC_baby_step_giant_step) {
		algebra_global Algebra;

		Algebra.do_EC_baby_step_giant_step(EC_q, EC_b, EC_c,
				EC_bsgs_G, EC_bsgs_N, EC_bsgs_cipher_text,
				verbose_level);
	}
	else if (f_EC_baby_step_giant_step_decode) {
		algebra_global Algebra;

		Algebra.do_EC_baby_step_giant_step_decode(EC_q, EC_b, EC_c,
				EC_bsgs_A, EC_bsgs_N, EC_bsgs_cipher_text, EC_bsgs_keys,
				verbose_level);
	}
	else if (f_NTRU_encrypt) {
		algebra_global Algebra;

		Algebra.NTRU_encrypt(NTRU_encrypt_N, NTRU_encrypt_p, NTRU_encrypt_q,
				NTRU_encrypt_H, NTRU_encrypt_R, NTRU_encrypt_Msg,
				verbose_level);
	}
	else if (f_polynomial_center_lift) {
		algebra_global Algebra;

		Algebra.polynomial_center_lift(polynomial_center_lift_A, polynomial_center_lift_q, verbose_level);
	}
	else if (f_polynomial_reduce_mod_p) {
		algebra_global Algebra;

		Algebra.polynomial_reduce_mod_p(polynomial_reduce_mod_p_A, polynomial_reduce_mod_p, verbose_level);
	}
#if 0
	else if (f_ntt) {
		number_theoretic_transform NTT;
		NTT.init(ntt_t, ntt_q, verbose_level);
	}
#endif

}


void interface_cryptography::make_affine_sequence(int a, int c, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *f_reached;
	int *orbit;
	int x0, x, y, len, cnt;
	number_theory_domain NT;

	if (f_v) {
		cout << "make_affine_sequence a=" << a << " c=" << c << " m=" << m << endl;
		}
	f_reached = NEW_int(m);
	orbit = NEW_int(m);
	int_vec_zero(f_reached, m);
	cnt = 0;
	for (x0 = 0; x0 < m; x0++) {
		if (f_reached[x0]) {
			continue;
			}

		x = x0;
		orbit[0] = x0;
		len = 1;
		while (TRUE) {
			f_reached[x] = TRUE;
			y = NT.mult_mod(a, x, m);
			y = NT.add_mod(y, c, m);

			if (f_reached[y]) {
				break;
				}
			orbit[len++] = y;
			x = y;
			}
		cout << "orbit " << cnt << " of " << x0 << " has length " << len << " : ";
		int_vec_print(cout, orbit, len);
		cout << endl;

		make_2D_plot(orbit, len, cnt, m, a, c, verbose_level);
		//make_graph(orbit, len, m, verbose_level);
		//list_sequence_in_binary(orbit, len, m, verbose_level);
		cnt++;
		}
	FREE_int(orbit);
	FREE_int(f_reached);


}

void interface_cryptography::make_2D_plot(
		int *orbit, int orbit_len, int cnt, int m, int a, int c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "m=" << m << " orbit_len=" << orbit_len << endl;
		}
	int *M;
	int h, x, y;

	M = NEW_int(m * m);
	int_vec_zero(M, m * m);



	for (h = 0; h < orbit_len - 1; h++) {
		x = orbit[h];
		y = orbit[h + 1];
		M[x * m + y] = 1;
	}
	char str[1000];
	string fname;
	file_io Fio;

	snprintf(str, 1000, "orbit_cnt%d_m%d_a%d_c%d.csv", cnt, m, a, c);
	fname.assign(str);
	Fio.int_matrix_write_csv(fname, M, m, m);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(M);
}



void interface_cryptography::do_random_last(int random_nb, int verbose_level)
{
	int i, r = 0;


	cout << "RAND_MAX=" << RAND_MAX << endl;

	for (i = 0; i < random_nb; i++) {
		r = rand();
	}
	cout << r << endl;


}

void interface_cryptography::do_random(int random_nb, std::string &fname_csv, int verbose_level)
{
	int i;
	int *R;


	cout << "RAND_MAX=" << RAND_MAX << endl;

	R = NEW_int(random_nb);
	for (i = 0; i < random_nb; i++) {
		R[i] = rand();
	}

	file_io Fio;

	Fio.int_vec_write_csv(R, random_nb, fname_csv, "R");

	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;


}

void interface_cryptography::do_jacobi(int jacobi_top, int jacobi_bottom, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "jacobi_%d_%d.tex", jacobi_top, jacobi_bottom);
	snprintf(title, 1000, "Jacobi %d over %d", jacobi_top, jacobi_bottom);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	//algebra_global AG;

	//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	latex_interface L;

	//F.init(q), verbose_level - 2);

	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	number_theory_domain NT;
	longinteger_domain D;

	longinteger_object A, B;

	A.create(jacobi_top, __FILE__, __LINE__);

	B.create(jacobi_bottom, __FILE__, __LINE__);

	D.jacobi(A, B, verbose_level);

	NT.Jacobi_with_key_in_latex(f,
			jacobi_top, jacobi_bottom, verbose_level);
	//Computes the Jacobi symbol $\left( \frac{a}{m} \right)$.

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::do_solovay_strassen(int p, int a, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "solovay_strassen_%d_%d.tex", p, a);
	snprintf(title, 1000, "Solovay Strassen %d with base %d", p, a);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	//algebra_global AG;

	//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	latex_interface L;

	//F.init(q), verbose_level - 2);

	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	number_theory_domain NT;
	longinteger_domain D;

	longinteger_object P, A;

	P.create(p, __FILE__, __LINE__);

	A.create(a, __FILE__, __LINE__);

	//D.jacobi(A, B, verbose_level);

	D.solovay_strassen_test_with_latex_key(f,
			P, A,
			verbose_level);


	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::do_miller_rabin(int p, int nb_times, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "miller_rabin_%d.tex", p);
	snprintf(title, 1000, "Miller Rabin %d", p);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	//algebra_global AG;

	//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	latex_interface L;

	//F.init(q), verbose_level - 2);

	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	longinteger_domain D;

	longinteger_object P, A;

	P.create(p, __FILE__, __LINE__);

	int i;

	for (i = 0; i < nb_times; i++) {

		f << "Miller Rabin test no " << i << ":\\\\" << endl;
		if (!D.miller_rabin_test_with_latex_key(f,
			P, i,
			verbose_level)) {
			break;
		}

	}
	if (i == nb_times) {
		f << "Miller Rabin: The number is probably prime. Miller Rabin is inconclusive.\\\\" << endl;
	}
	else {
		f << "Miller Rabin: The number is not prime.\\\\" << endl;
	}

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::do_fermat_test(int p, int nb_times, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "fermat_%d.tex", p);
	snprintf(title, 1000, "Fermat test %d", p);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	//algebra_global AG;

	//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	latex_interface L;

	//F.init(q), verbose_level - 2);

	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	longinteger_domain D;
	longinteger_object P;


	P.create(p, __FILE__, __LINE__);

	if (D.fermat_test_iterated_with_latex_key(f,
			P, nb_times,
			verbose_level)) {
		f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
	}
	else {
		f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
	}

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::do_find_pseudoprime(int nb_digits, int nb_fermat, int nb_miller_rabin, int nb_solovay_strassen, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "pseudoprime_%d.tex", nb_digits);
	snprintf(title, 1000, "Pseudoprime %d", nb_digits);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	latex_interface L;


	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	longinteger_domain D;
	longinteger_object P;


	int cnt = -1;

	f << "\\begin{multicols}{2}" << endl;
	f << "\\begin{enumerate}[(1)]" << endl;
	while (TRUE) {

		cnt++;

		D.random_number_with_n_decimals(P, nb_digits, verbose_level);

		f << "\\item" << endl;
		f << "Trial " << cnt << ", testing random number " << P << endl;

		if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
			f << "the number is not prime by looking at the lowest digit" << endl;
			continue;
		}

		f << "\\begin{enumerate}[(a)]" << endl;
		f << "\\item" << endl;
		if (D.fermat_test_iterated_with_latex_key(f,
				P, nb_fermat,
				verbose_level)) {
			//f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			continue;
		}
		else {
			//f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
		}

		f << "\\item" << endl;
		if (D.miller_rabin_test_iterated_with_latex_key(f,
				P, nb_miller_rabin,
				verbose_level)) {
			f << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			continue;
		}
		else {
			//f << "Miller Rabin: The number $" << P << "$ is probably prime. Miller Rabin test is inconclusive.\\\\" << endl;
		}


		f << "\\item" << endl;
		if (D.solovay_strassen_test_iterated_with_latex_key(f,
				P, nb_solovay_strassen,
				verbose_level)) {
			//f << "Solovay-Strassen: The number $" << P << "$ is not prime.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			continue;
		}
		else {
			//f << "Solovay-Strassen: The number $" << P << "$ is probably prime. Solovay-Strassen test is inconclusive.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			break;
		}
		f << "\\end{enumerate}" << endl;

	}
	f << "\\end{enumerate}" << endl;
	f << "\\end{multicols}" << endl;

	f << "\\noindent" << endl;
	f << "The number $" << P << "$ is probably prime. \\\\" << endl;
	f << "Number of Fermat tests = " << nb_fermat << " \\\\" << endl;
	f << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;
	f << "Number of Solovay-Strassen tests = " << nb_solovay_strassen << " \\\\" << endl;

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::do_find_strong_pseudoprime(int nb_digits, int nb_fermat, int nb_miller_rabin, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "strong_pseudoprime_%d.tex", nb_digits);
	snprintf(title, 1000, "Strong Pseudoprime %d", nb_digits);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	latex_interface L;


	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	longinteger_domain D;
	longinteger_object P;


	int cnt = -1;

	f << "\\begin{multicols}{2}" << endl;
	f << "\\begin{enumerate}[(1)]" << endl;
	while (TRUE) {

		cnt++;

		D.random_number_with_n_decimals(P, nb_digits, verbose_level);

		f << "\\item" << endl;
		f << "Trial " << cnt << ", testing random number " << P << endl;

		if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
			f << "the number is not prime by looking at the lowest digit" << endl;
			continue;
		}

		f << "\\begin{enumerate}[(a)]" << endl;
		f << "\\item" << endl;
		if (D.fermat_test_iterated_with_latex_key(f,
				P, nb_fermat,
				verbose_level)) {
			//f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			continue;
		}
		else {
			//f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
		}

		f << "\\item" << endl;
		if (D.miller_rabin_test_iterated_with_latex_key(f,
				P, nb_miller_rabin,
				verbose_level)) {
			//f << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			continue;
		}
		else {
			//f << "Miller Rabin: The number $" << P << "$ is probably prime. Miller Rabin test is inconclusive.\\\\" << endl;
			f << "\\end{enumerate}" << endl;
			break;
		}


		f << "\\end{enumerate}" << endl;

	}
	f << "\\end{enumerate}" << endl;
	f << "\\end{multicols}" << endl;

	f << "\\noindent" << endl;
	f << "The number $" << P << "$ is probably prime. \\\\" << endl;
	f << "Number of Fermat tests = " << nb_fermat << " \\\\" << endl;
	f << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}


void interface_cryptography::do_miller_rabin_text(const char *number_text, int nb_miller_rabin, int verbose_level)
{
	char fname[1000];
	char title[1000];
	char author[1000];

	snprintf(fname, 1000, "miller_rabin_%s.tex", number_text);
	snprintf(title, 1000, "Miller Rabin %s", number_text);
	//sprintf(author, "");
	author[0] = 0;


	{
	ofstream f(fname);


	latex_interface L;


	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	longinteger_domain D;
	longinteger_object P;


	f << "\\begin{multicols}{2}" << endl;

	P.create_from_base_10_string(number_text);


	if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
		f << "the number is not prime by looking at the lowest digit" << endl;
	}
	else {

		if (D.miller_rabin_test_iterated_with_latex_key(f,
				P, nb_miller_rabin,
				verbose_level)) {
			f << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
		}
		else {
			f << "The number $" << P << "$ is probably prime. \\\\" << endl;
		}
	}


	f << "\\end{multicols}" << endl;

	f << "\\noindent" << endl;
	f << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;

	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;


}

void interface_cryptography::quadratic_sieve(int n,
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
	for (i = 0; i < (int) small_factors.size(); i++) {
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

void interface_cryptography::calc_log2(vector<int> &primes, vector<int> &primes_log2, int verbose_level)
{
	int i, l, k;

	l = primes.size();
	for (i = 0; i < l; i++) {
		k = log2(primes[i]);
		primes_log2.push_back(k);
		}
}

void interface_cryptography::square_root(const char *square_root_number, int verbose_level)
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

void interface_cryptography::square_root_mod(const char *square_root_number, const char *mod_number, int verbose_level)
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

void interface_cryptography::reduce_primes(vector<int> &primes,
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


void interface_cryptography::do_sift_smooth(int sift_smooth_from,
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

void interface_cryptography::do_discrete_log(long int y, long int a, long int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;
	int t0, t1, dt;
	os_interface Os;

	if (f_v) {
		cout << "do_discrete_log" << endl;
		cout << "y=" << y << endl;
		cout << "a=" << a << endl;
		cout << "p=" << p << endl;
	}

	t0 = Os.os_ticks();
	//finite_field F;
	long int n, b;

	//F.init(p, 0);
	for (n = 0; n < p - 1; n++) {
		//b = F.power(a, n);
		b = NT.power_mod(a, n, p);
		if (b == y) {
			break;
		}
	}

	if (n == p - 1) {
		cout << "could not solve the discrete log problem." << endl;
	}
	else {
		cout << "The discrete log is " << n << " since ";
		cout << y << " = " << a << "^" << n << " mod " << p << endl;
	}

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}

void interface_cryptography::do_primitive_root(long int p, int verbose_level)
{
	number_theory_domain NT;
	long int a;
	int t0, t1, dt;
	os_interface Os;

	t0 = Os.os_ticks();

	a = NT.primitive_root_randomized(p, verbose_level);
	cout << "a primitive root modulo " << p << " is " << a << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}


void interface_cryptography::do_inverse_mod(long int a, long int n, int verbose_level)
{
	number_theory_domain NT;
	long int b;
	int t0, t1, dt;
	os_interface Os;

	t0 = Os.os_ticks();

	b = NT.inverse_mod(a, n);
	cout << "the inverse of " << a << " mod " << n << " is " << b << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}

void interface_cryptography::do_extended_gcd(int a, int b, int verbose_level)
{
	{
	longinteger_domain D;

	longinteger_object A, B, G, U, V;

	A.create(a, __FILE__, __LINE__);
	B.create(b, __FILE__, __LINE__);

	cout << "before D.extended_gcd" << endl;
	D.extended_gcd(A, B,
			G, U, V, verbose_level);
	cout << "after D.extended_gcd" << endl;

	}

}


void interface_cryptography::do_power_mod(long int a, long int k, long int n, int verbose_level)
{
	number_theory_domain NT;
	long int b;
	int t0, t1, dt;
	os_interface Os;

	t0 = Os.os_ticks();

	b = NT.power_mod(a, k, n);
	cout << "the power of " << a << " to the " << k << " mod " << n << " is " << b << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;

}

void interface_cryptography::affine_cipher(char *ptext, char *ctext, int a, int b)
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

void interface_cryptography::affine_decipher(char *ctext, char *ptext, char *guess)
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

void interface_cryptography::vigenere_cipher(char *ptext, char *ctext, char *key)
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

void interface_cryptography::vigenere_decipher(char *ctext, char *ptext, char *key)
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

void interface_cryptography::vigenere_analysis(char *ctext)
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

void interface_cryptography::vigenere_analysis2(char *ctext, int key_length)
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

int interface_cryptography::kasiski_test(char *ctext, int threshold)
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

void interface_cryptography::print_candidates(char *ctext, int i, int h, int nb_candidates, int *candidates)
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

void interface_cryptography::print_set(int l, int *s)
{
	int i;
	
	for (i = 0; i < l; i++) {
		cout << s[i] << " ";
	}
	cout << endl;
}

void interface_cryptography::print_on_top(char *text1, char *text2)
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

void interface_cryptography::decipher(char *ctext, char *ptext, char *guess)
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

void interface_cryptography::analyze(char *text)
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

double interface_cryptography::friedman_index(int *mult, int n)
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

double interface_cryptography::friedman_index_shifted(int *mult, int n, int shift)
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

void interface_cryptography::print_frequencies(int *mult)
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

void interface_cryptography::single_frequencies(char *text, int *mult)
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

void interface_cryptography::single_frequencies2(char *text, int stride, int n, int *mult)
{
	int i;
	
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < n; i++) {
		mult[text[i * stride] - 'a']++;
	}
}

void interface_cryptography::double_frequencies(char *text, int *mult)
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

void interface_cryptography::substition_cipher(char *ptext, char *ctext, char *key)
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

char interface_cryptography::lower_case(char c)
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

char interface_cryptography::upper_case(char c)
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

char interface_cryptography::is_alnum(char c)
{
	if (c >= 'A' && c <= 'Z') {
		return TRUE;
	}
	if (c >= 'a' && c <= 'z') {
		return TRUE;
	}
	return FALSE;
}

void interface_cryptography::get_random_permutation(char *p)
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

}}

