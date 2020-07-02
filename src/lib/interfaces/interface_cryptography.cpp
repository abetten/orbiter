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
	random_fname_csv = NULL;
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
	f_nullspace = FALSE;
	nullspace_q = 0;
	nullspace_m = 0;
	nullspace_n = 0;
	nullspace_text = NULL;
	f_RREF = FALSE;
	RREF_q = 0;
	RREF_m = 0;
	RREF_n = 0;
	//cout << "interface_cryptography::interface_cryptography 3" << endl;
	RREF_text = NULL;
	f_weight_enumerator = FALSE;
	f_normalize_from_the_right = FALSE;
	f_normalize_from_the_left = FALSE;
	f_transversal = FALSE;
	transversal_q = 0;
	transversal_line_1_basis = NULL;
	transversal_line_2_basis = NULL;
	transversal_point = NULL;
	f_intersection_of_two_lines = FALSE;
	intersection_of_two_lines_q = 0;
	line_1_basis = NULL;
	line_2_basis = NULL;
	f_trace = FALSE;
	trace_q = 0;
	f_norm = FALSE;
	norm_q = 0;
	f_count_subprimitive = FALSE;
	count_subprimitive_Q_max = 0;
	count_subprimitive_H_max = 0;
#if 0
	f_ntt = FALSE;
	ntt_t = 0;
	ntt_q = 0;
#endif
	//cout << "interface_cryptography::interface_cryptography done" << endl;

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
	else if (strcmp(argv[i], "-nullspace") == 0) {
		cout << "-nullspace <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-RREF") == 0) {
		cout << "-RREF <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-weight_enumerator") == 0) {
		cout << "-weight_enumerator <int : q> <int : m> <int : n> <string : coeff_matrix>" << endl;
	}
	else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
		cout << "-normalize_from_the_right" << endl;
	}
	else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
		cout << "-normalize_from_the_left" << endl;
	}
	else if (strcmp(argv[i], "-transversal") == 0) {
		cout << "-transversal <int : q> <string : line_1> <string : line_2> <string : pt>" << endl;
	}
	else if (strcmp(argv[i], "-intersection_of_two_lines") == 0) {
		cout << "-intersection_of_two_lines <int : q> <string : line_1> <string : line_2>" << endl;
	}
	else if (strcmp(argv[i], "-trace") == 0) {
		cout << "-trace <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-norm") == 0) {
		cout << "-norm <int : q>" << endl;
	}
	else if (strcmp(argv[i], "-count_subprimitive") == 0) {
		cout << "-count_subprimitive <int : Q_max> <int : H_max>" << endl;
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
	else if (strcmp(argv[i], "-nullspace") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-RREF") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-weight_enumerator") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-transversal") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-intersection_of_two_lines") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-trace") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-norm") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-count_subprimitive") == 0) {
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
			random_fname_csv = argv[++i];
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
		else if (strcmp(argv[i], "-nullspace") == 0) {
			f_nullspace = TRUE;
			nullspace_q = atoi(argv[++i]);
			nullspace_m = atoi(argv[++i]);
			nullspace_n = atoi(argv[++i]);
			nullspace_text = argv[++i];
			cout << "-nullspace " << nullspace_q
					<< " " << nullspace_m << " " << nullspace_n << " " << nullspace_text << endl;
		}
		else if (strcmp(argv[i], "-RREF") == 0) {
			f_RREF = TRUE;
			RREF_q = atoi(argv[++i]);
			RREF_m = atoi(argv[++i]);
			RREF_n = atoi(argv[++i]);
			RREF_text = argv[++i];
			cout << "-RREF " << RREF_q
					<< " " << RREF_m << " " << RREF_n << " " << RREF_text << endl;
		}
		else if (strcmp(argv[i], "-weight_enumerator") == 0) {
			f_weight_enumerator = TRUE;
			RREF_q = atoi(argv[++i]);
			RREF_m = atoi(argv[++i]);
			RREF_n = atoi(argv[++i]);
			RREF_text = argv[++i];
			cout << "-weight_enumerator " << RREF_q
					<< " " << RREF_m << " " << RREF_n << " " << RREF_text << endl;
		}
		else if (strcmp(argv[i], "-normalize_from_the_right") == 0) {
			f_normalize_from_the_right = TRUE;
			cout << "-normalize_from_the_right " << endl;
		}
		else if (strcmp(argv[i], "-normalize_from_the_left") == 0) {
			f_normalize_from_the_left = TRUE;
			cout << "-normalize_from_the_left " << endl;
		}
		else if (strcmp(argv[i], "-transversal") == 0) {
			f_transversal = TRUE;
			transversal_q = atoi(argv[++i]);
			transversal_line_1_basis = argv[++i];
			transversal_line_2_basis = argv[++i];
			transversal_point = argv[++i];
			cout << "-transversal " << transversal_q
					<< " " << transversal_line_1_basis
					<< " " << transversal_line_2_basis
					<< " " << transversal_point << endl;
		}
		else if (strcmp(argv[i], "-intersection_of_two_lines") == 0) {
			f_intersection_of_two_lines = TRUE;
			intersection_of_two_lines_q = atoi(argv[++i]);
			line_1_basis = argv[++i];
			line_2_basis = argv[++i];
			cout << "-intersection_of_two_lines " << intersection_of_two_lines_q
					<< " " << line_1_basis
					<< " " << line_2_basis
					<< endl;
		}
		else if (strcmp(argv[i], "-trace") == 0) {
			f_trace = TRUE;
			trace_q = atoi(argv[++i]);
			cout << "-trace " << trace_q
					<< endl;
		}
		else if (strcmp(argv[i], "-norm") == 0) {
			f_norm = TRUE;
			norm_q = atoi(argv[++i]);
			cout << "-norm " << norm_q
					<< endl;
		}
		else if (strcmp(argv[i], "-count_subprimitive") == 0) {
			f_count_subprimitive = TRUE;
			count_subprimitive_Q_max = atoi(argv[++i]);
			count_subprimitive_H_max = atoi(argv[++i]);
			cout << "-count_subprimitive " << count_subprimitive_Q_max
					<< " " << count_subprimitive_H_max
					<< endl;
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
		do_EC_Koblitz_encoding(EC_q, EC_b, EC_c, EC_s, EC_pt_text, EC_message, verbose_level);
	}
	else if (f_EC_points) {
		do_EC_points(EC_q, EC_b, EC_c, verbose_level);
	}
	else if (f_EC_add) {
		do_EC_add(EC_q, EC_b, EC_c, EC_pt1_text, EC_pt2_text, verbose_level);
	}
	else if (f_EC_cyclic_subgroup) {
		do_EC_cyclic_subgroup(EC_q, EC_b, EC_c, EC_pt_text, verbose_level);
	}
	else if (f_EC_multiple_of) {
		do_EC_multiple_of(EC_q, EC_b, EC_c,
				EC_pt_text, EC_multiple_of_n, verbose_level);
	}
	else if (f_EC_discrete_log) {
		do_EC_discrete_log(EC_q, EC_b, EC_c, EC_pt_text,
				EC_discrete_log_pt_text, verbose_level);
	}
	else if (f_EC_baby_step_giant_step) {
		do_EC_baby_step_giant_step(EC_q, EC_b, EC_c,
				EC_bsgs_G, EC_bsgs_N, EC_bsgs_cipher_text,
				verbose_level);
	}
	else if (f_EC_baby_step_giant_step_decode) {
		do_EC_baby_step_giant_step_decode(EC_q, EC_b, EC_c,
				EC_bsgs_A, EC_bsgs_N, EC_bsgs_cipher_text, EC_bsgs_keys,
				verbose_level);
	}
	else if (f_nullspace) {
		do_nullspace(nullspace_q, nullspace_m, nullspace_n,
				nullspace_text, f_normalize_from_the_left,
				f_normalize_from_the_right, verbose_level);
	}
	else if (f_RREF) {
		do_RREF(RREF_q, RREF_m, RREF_n, RREF_text,
				f_normalize_from_the_left, f_normalize_from_the_right, verbose_level);
	}
	else if (f_weight_enumerator) {
		do_weight_enumerator(RREF_q, RREF_m, RREF_n, RREF_text,
				f_normalize_from_the_left, f_normalize_from_the_right, verbose_level);
	}
	else if (f_transversal) {
		do_transversal(transversal_q,
				transversal_line_1_basis,
				transversal_line_2_basis,
				transversal_point,
				f_normalize_from_the_left, f_normalize_from_the_right,
				verbose_level);
	}
	else if (f_intersection_of_two_lines) {
		do_intersection_of_two_lines(intersection_of_two_lines_q,
				line_1_basis,
				line_2_basis,
				f_normalize_from_the_left, f_normalize_from_the_right,
				verbose_level);
	}
	else if (f_trace) {
		do_trace(trace_q, verbose_level);
	}
	else if (f_norm) {
		do_norm(norm_q, verbose_level);
	}
	else if (f_count_subprimitive) {
		algebra_global AG;
		AG.count_subprimitive(count_subprimitive_Q_max, count_subprimitive_H_max);
	}
#if 0
	else if (f_ntt) {
		number_theoretic_transform NTT;
		NTT.init(ntt_t, ntt_q, verbose_level);
	}
#endif

}

void interface_cryptography::do_trace(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "do_trace" << endl;
	}
	F = NEW_OBJECT(finite_field);

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	F->init(q, 0);
	for (s = 0; s < q; s++) {
		t = F->absolute_trace(s);
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Trace 0:" << endl;
	int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Trace 1:" << endl;
	int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;

	char fname_csv[1000];
	file_io Fio;

	snprintf(fname_csv, 1000, "F_q%d_trace_0.csv", q);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Trace_0");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;

	snprintf(fname_csv, 1000, "F_q%d_trace_1.csv", q);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Trace_1");
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	FREE_OBJECT(F);
	if (f_v) {
		cout << "do_trace done" << endl;
	}
}

void interface_cryptography::do_norm(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int s, t;
	int *T0 = NULL;
	int *T1 = NULL;
	int nb_T0 = 0;
	int nb_T1 = 0;

	if (f_v) {
		cout << "do_norm" << endl;
	}
	F = NEW_OBJECT(finite_field);

	T0 = NEW_int(q);
	T1 = NEW_int(q);

	F->init(q, 0);
	for (s = 0; s < q; s++) {
		t = F->absolute_norm(s);
		if (t == 1) {
			T1[nb_T1++] = s;
		}
		else {
			T0[nb_T0++] = s;
		}
	}

	cout << "Norm 0:" << endl;
	int_vec_print_fully(cout, T0, nb_T0);
	cout << endl;

	cout << "Norm 1:" << endl;
	int_vec_print_fully(cout, T1, nb_T1);
	cout << endl;


	char fname_csv[1000];
	file_io Fio;

	snprintf(fname_csv, 1000, "F_q%d_norm_0.csv", q);
	Fio.int_vec_write_csv(T0, nb_T0,
			fname_csv, "Norm_0");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;

	snprintf(fname_csv, 1000, "F_q%d_norm_1.csv", q);
	Fio.int_vec_write_csv(T1, nb_T1,
			fname_csv, "Norm_1");
	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;


	FREE_OBJECT(F);
	if (f_v) {
		cout << "do_norm done" << endl;
	}
}

void interface_cryptography::do_intersection_of_two_lines(int q,
		const char *line_1_basis,
		const char *line_2_basis,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *Line1;
	int *Line2;
	int *A;
	int *B;
	int *C;
	int len, rk, i;

	if (f_v) {
		cout << "do_intersecton_of_two_lines" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "do_intersecton_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "do_intersecton_of_two_lines len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);
	C = NEW_int(16);

	// Line 1
	int_vec_copy(Line1, A, 8);
	rk = F->perp_standard(4, 2, A, verbose_level);
	if (rk != 2) {
		cout << "do_intersecton_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}

	// Line 2
	int_vec_copy(Line2, B, 8);
	rk = F->perp_standard(4, 2, B, verbose_level);
	if (rk != 2) {
		cout << "do_intersecton_of_two_lines rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}


	int_vec_copy(A + 8, C, 8);
	int_vec_copy(B + 8, C + 8, 8);
	rk = F->perp_standard(4, 4, C, verbose_level);
	if (rk != 3) {
		cout << "do_intersecton_of_two_lines rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}

	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 3; i < 4; i++) {
			F->PG_element_normalize_from_front(
					C + i * 4, 1, 4);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 3; i < 4; i++) {
			F->PG_element_normalize(
					C + i * 4, 1, 4);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(C + 12, 1, 4);
		cout << "rk=" << rk << endl;

	}


	FREE_int(Line1);
	FREE_int(Line2);
	FREE_int(A);
	FREE_int(B);
	FREE_int(C);

	if (f_v) {
		cout << "do_intersecton_of_two_lines done" << endl;
	}

}

void interface_cryptography::do_transversal(int q,
		const char *line_1_basis,
		const char *line_2_basis,
		const char *point,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *Line1;
	int *Line2;
	int *Pt;
	int *A;
	int *B;
	int len, rk, i;

	if (f_v) {
		cout << "do_transversal" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(line_1_basis, Line1, len);
	if (len != 8) {
		cout << "do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(line_2_basis, Line2, len);
	if (len != 8) {
		cout << "do_transversal len != 8" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	int_vec_scan(point, Pt, len);
	if (len != 4) {
		cout << "do_transversal len != 4" << endl;
		cout << "received " << len << endl;
		exit(1);
	}
	A = NEW_int(16);
	B = NEW_int(16);

	// Line 1
	int_vec_copy(Line1, A, 8);
	int_vec_copy(Pt, A + 8, 4);
	rk = F->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	int_vec_copy(A + 12, B, 4);

	// Line 2
	int_vec_copy(Line2, A, 8);
	int_vec_copy(Pt, A + 8, 4);
	rk = F->perp_standard(4, 3, A, verbose_level);
	if (rk != 3) {
		cout << "do_transversal rk != 3" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	int_vec_copy(A + 12, B + 4, 4);

	// B
	rk = F->perp_standard(4, 2, B, verbose_level);
	if (rk != 2) {
		cout << "do_transversal rk != 2" << endl;
		cout << "rk= " << rk << endl;
		exit(1);
	}
	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 2; i < 4; i++) {
			F->PG_element_normalize_from_front(
					B + i * 4, 1, 4);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 2; i < 4; i++) {
			F->PG_element_normalize(
					B + i * 4, 1, 4);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(B + 8, 2, 4);
		cout << "rk=" << rk << endl;

	}


	FREE_int(Line1);
	FREE_int(Line2);
	FREE_int(Pt);
	FREE_int(A);
	FREE_int(B);

	if (f_v) {
		cout << "do_transversal done" << endl;
	}
}

void interface_cryptography::do_nullspace(int q,
		int m, int n, const char *text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i, rk1;
	latex_interface Li;

	if (f_v) {
		cout << "do_nullspace" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}

	if (m > n) {
		cout << "nullspace needs m < n" << endl;
		exit(1);
	}

	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	int_vec_copy(M, A, m * n);

	rk = F->perp_standard(n, m, A, verbose_level);


	cout << "after perp_standard:" << endl;
	int_matrix_print(A, n, n);
	cout << "rk=" << rk << endl;

	cout << "after RREF" << endl;
	rk1 = F->Gauss_int(A + rk * n,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, n - rk, n, n,
		0 /*verbose_level*/);


	cout << "after RREF" << endl;
	int_matrix_print(A + rk * n, rk1, n);
	cout << "rank of nullspace = " << rk1 << endl;

	cout << "coefficients:" << endl;
	int_vec_print(cout, A + rk * n, rk1 * n);
	cout << endl;

	cout << "$$" << endl;
	cout << "\\left[" << endl;
	Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
	cout << "\\right]" << endl;
	cout << "$$" << endl;

	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = rk; i < n; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = rk; i < n; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, n, n);
		cout << "rk=" << rk << endl;

		cout << "$$" << endl;
		cout << "\\left[" << endl;
		Li.int_matrix_print_tex(cout, A + rk * n, rk1, n);
		cout << "\\right]" << endl;
		cout << "$$" << endl;
	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_nullspace done" << endl;
	}
}

void interface_cryptography::do_RREF(int q,
		int m, int n, const char *text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int len, rk, i;
	latex_interface Li;

	if (f_v) {
		cout << "do_RREF" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	int_vec_copy(M, A, m * n);

	rk = F->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	cout << "after RREF:" << endl;
	int_matrix_print(A, rk, n);
	cout << "rk=" << rk << endl;

	cout << "coefficients:" << endl;
	int_vec_print(cout, A, rk * n);
	cout << endl;

	cout << "$$" << endl;
	cout << "\\left[" << endl;
	Li.int_matrix_print_tex(cout, A, rk, n);
	cout << "\\right]" << endl;
	cout << "$$" << endl;

	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);

	if (f_v) {
		cout << "do_RREF done" << endl;
	}
}

void interface_cryptography::do_weight_enumerator(int q,
		int m, int n, const char *text,
		int f_normalize_from_the_left, int f_normalize_from_the_right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int *M;
	int *A;
	int *base_cols;
	int *weight_enumerator;
	int len, rk, i;

	if (f_v) {
		cout << "do_weight_enumerator" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(text, M, len);
	if (len != m * n) {
		cout << "number of coordinates received differs from m * n" << endl;
		cout << "received " << len << endl;
		exit(1);
	}


	A = NEW_int(n * n);
	base_cols = NEW_int(n);
	weight_enumerator = NEW_int(n + 1);
	int_vec_copy(M, A, m * n);

	rk = F->Gauss_int(A,
		FALSE /* f_special */, TRUE /* f_complete */, base_cols,
		FALSE /* f_P */, NULL /*P*/, m, n, n,
		0 /*verbose_level*/);


	cout << "after RREF:" << endl;
	int_matrix_print(A, rk, n);
	cout << "rk=" << rk << endl;

	cout << "coefficients:" << endl;
	int_vec_print(cout, A, rk * n);
	cout << endl;


	F->code_weight_enumerator(n, rk,
		A /* code */, // [k * n]
		weight_enumerator, // [n + 1]
		verbose_level);

	cout << "The weight enumerator is:" << endl;
	for (i = 0; i <= n; i++) {
		cout << i << " : " << weight_enumerator[i] << endl;
	}

	int f_first = TRUE;

	for (i = 0; i <= n; i++) {
		if (weight_enumerator[i] == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			cout << " + ";
		}
		cout << weight_enumerator[i];
		if (i) {
			cout << "*";
			cout << "x";
			if (i > 1) {
				cout << "^";
				if (i < 10) {
					cout << i;
				}
				else {
					cout << "(" << i << ")";
				}
			}
		}
		if (n - i) {
			cout << "*";
			cout << "y";
			if (n - i > 1) {
				cout << "^";
				if (n - i < 10) {
					cout << n - i;
				}
				else {
					cout << "(" << n - i << ")";
				}
			}
		}

	}
	cout << endl;


	if (f_normalize_from_the_left) {
		cout << "normalizing from the left" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize_from_front(
					A + i * n, 1, n);
		}

		cout << "after normalize from the left:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}

	if (f_normalize_from_the_right) {
		cout << "normalizing from the right" << endl;
		for (i = 0; i < rk; i++) {
			F->PG_element_normalize(
					A + i * n, 1, n);
		}

		cout << "after normalize from the right:" << endl;
		int_matrix_print(A, rk, n);
		cout << "rk=" << rk << endl;

	}


	FREE_int(M);
	FREE_int(A);
	FREE_int(base_cols);
	FREE_int(weight_enumerator);

	if (f_v) {
		cout << "do_weight_enumerator done" << endl;
	}
}

void interface_cryptography::do_EC_Koblitz_encoding(int q,
		int EC_b, int EC_c, int EC_s,
		const char *pt_text, const char *EC_message,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x0, x, y;

	if (f_v) {
		cout << "do_EC_Koblitz_encoding" << endl;
	}
	if (f_v) {
		cout << "do_EC_Koblitz_encoding b = " << EC_b << endl;
		cout << "do_EC_Koblitz_encoding c = " << EC_c << endl;
		cout << "do_EC_Koblitz_encoding s = " << EC_s << endl;
	}

	vector<vector<int>> Encoding;
	vector<int> J;

	int u, i, j, r;

	u = q / 27;
	if (f_v) {
		cout << "do_EC_Koblitz_encoding u = " << u << endl;
	}


	F = NEW_OBJECT(finite_field);
	F->init(q, 0 /*verbose_level*/);
	for (i = 1; i <= 26; i++) {
		x0 = i * u;
		for (j = 0; j < u; j++) {
			x = x0 + j;
			r = EC_evaluate_RHS(F, EC_b, EC_c, x);
			if (F->square_root(r, y)) {
				break;
			}
		}
		if (j < u) {
			{
				vector<int> pt;

				J.push_back(j);
				pt.push_back(x);
				pt.push_back(y);
				pt.push_back(1);
				Encoding.push_back(pt);
			}
		}
		else {
			cout << "failure to encode letter " << i << endl;
			exit(1);
		}
	}
	for (i = 0; i < 26; i++) {


		x = (i + 1) * u + J[i];

		r = EC_evaluate_RHS(F, EC_b, EC_c, x);

		F->square_root(r, y);

		cout << (char)('A' + i) << " & " << i + 1 << " & " << J[i] << " & " << x
				<< " & " << r
				<< " & " << y
				<< " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ "
				<< "\\\\" << endl;

	}

	cout << "without j:" << endl;
	for (i = 0; i < 26; i++) {
		cout << (char)('A' + i) << " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ \\\\" << endl;

	}



	vector<vector<int>> Pts;
	int order;
	int *v;
	int len;
	int Gx, Gy, Gz;
	int Mx, My, Mz;
	int Rx, Ry, Rz;
	int Ax, Ay, Az;
	int Cx, Cy, Cz;
	int Tx, Ty, Tz;
	int Dx, Dy, Dz;
	int msRx, msRy, msRz;
	int m, k, plain;
	os_interface Os;

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);
	cout << "G = (" << Gx << "," << Gy << "," << Gz << ")" << endl;


	F->elliptic_curve_all_point_multiples(
			EC_b, EC_c, order,
			Gx, Gy, Gz,
			Pts,
			verbose_level);


	int minus_s;

	minus_s = order - EC_s;

	cout << "order = " << order << endl;
	cout << "minus_s = " << minus_s << endl;

	Ax = Pts[EC_s - 1][0];
	Ay = Pts[EC_s - 1][1];
	Az = 1;
	cout << "A = (" << Ax << "," << Ay << "," << Az << ")" << endl;

	len = strlen(EC_message);

	F->nb_calls_to_elliptic_curve_addition() = 0;

	vector<vector<int>> Ciphertext;

	for (i = 0; i < len; i++) {
		if (EC_message[i] < 'A' || EC_message[i] > 'Z') {
			continue;
		}
		m = EC_message[i] - 'A' + 1;
		k = 1 + Os.random_integer(order - 1);

		Mx = Encoding[m - 1][0];
		My = Encoding[m - 1][1];
		Mz = 1;

		// R := k * G
		//cout << "$R=" << k << "*G$\\\\" << endl;

		F->elliptic_curve_point_multiple /*_with_log*/(
					EC_b, EC_c, k,
					Gx, Gy, Gz,
					Rx, Ry, Rz,
					0 /*verbose_level*/);
		//cout << "$R=" << k << "*G=(" << Rx << "," << Ry << "," << Rz << ")$\\\\" << endl;

		// C := k * A
		//cout << "$C=" << k << "*A$\\\\" << endl;
		F->elliptic_curve_point_multiple /*_with_log*/(
					EC_b, EC_c, k,
					Ax, Ay, Az,
					Cx, Cy, Cz,
					0 /*verbose_level*/);
		//cout << "$C=" << k << "*A=(" << Cx << "," << Cy << "," << Cz << ")$\\\\" << endl;

		// T := C + M
		F->elliptic_curve_addition(EC_b, EC_c,
				Cx, Cy, Cz,
				Mx, My, Mz,
				Tx, Ty, Tz,
				0 /*verbose_level*/);
		//cout << "$T=C+M=(" << Tx << "," << Ty << "," << Tz << ")$\\\\" << endl;
		{
		vector<int> cipher;

		cipher.push_back(Rx);
		cipher.push_back(Ry);
		cipher.push_back(Tx);
		cipher.push_back(Ty);
		Ciphertext.push_back(cipher);
		}

		cout << setw(4) << i << " & " << EC_message[i] << " & " << setw(4) << m << " & " << setw(4) << k
				<< "& (" << setw(4) << Mx << "," << setw(4) << My << "," << setw(4) << Mz << ") "
				<< "& (" << setw(4) << Rx << "," << setw(4) << Ry << "," << setw(4) << Rz << ") "
				<< "& (" << setw(4) << Cx << "," << setw(4) << Cy << "," << setw(4) << Cz << ") "
				<< "& (" << setw(4) << Tx << "," << setw(4) << Ty << "," << setw(4) << Tz << ") "
				<< "\\\\"
				<< endl;

	}

	cout << "Ciphertext:\\\\" << endl;
	for (i = 0; i < (int) Ciphertext.size(); i++) {
		cout << Ciphertext[i][0] << ",";
		cout << Ciphertext[i][1] << ",";
		cout << Ciphertext[i][2] << ",";
		cout << Ciphertext[i][3] << "\\\\" << endl;
	}

	for (i = 0; i < (int) Ciphertext.size(); i++) {
		Rx = Ciphertext[i][0];
		Ry = Ciphertext[i][1];
		Tx = Ciphertext[i][2];
		Ty = Ciphertext[i][3];

		// msR := -s * R
		F->elliptic_curve_point_multiple(
					EC_b, EC_c, minus_s,
					Rx, Ry, Rz,
					msRx, msRy, msRz,
					0 /*verbose_level*/);

		// D := msR + T
		F->elliptic_curve_addition(EC_b, EC_c,
				msRx, msRy, msRz,
				Tx, Ty, Tz,
				Dx, Dy, Dz,
				0 /*verbose_level*/);

		plain = Dx / u;

		cout << setw(4) << i << " & (" << Rx << "," << Ry << "," << Tx << "," << Ty << ") "
				<< "& (" << setw(4) << msRx << "," << setw(4) << msRy << "," << setw(4) << msRz << ") "
				<< "& (" << setw(4) << Dx << "," << setw(4) << Dy << "," << setw(4) << Dz << ") "
				<< " & " << plain << " & " << (char)('A' - 1 + plain)
				<< "\\\\"
				<< endl;

	}

	cout << "nb_calls_to_elliptic_curve_addition="
			<< F->nb_calls_to_elliptic_curve_addition() << endl;


	if (f_v) {
		cout << "do_EC_Koblitz_encoding done" << endl;
	}
}

void interface_cryptography::do_EC_points(int q,
		int EC_b, int EC_c, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x, y, r, y1, y2;

	if (f_v) {
		cout << "do_EC_points" << endl;
	}
	vector<vector<int>> Pts;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	for (x = 0; x < q; x++) {
		r = EC_evaluate_RHS(F, EC_b, EC_c, x);
		if (r == 0) {

			{
				vector<int> pt;

				pt.push_back(x);
				pt.push_back(0);
				pt.push_back(1);
				Pts.push_back(pt);
			}
		}
		else {
			if (F->square_root(r, y)) {
				y1 = y;
				y2 = F->negate(y);
				if (y2 == y1) {
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
				else {
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y2);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
			}
			else {
				// no point for this x coordinate
			}

#if 0
			if (p != 2) {
				l = Legendre(r, q, 0);

				if (l == 1) {
					y = sqrt_mod_involved(r, q);
						// DISCRETA/global.cpp

					if (F->mult(y, y) != r) {
						cout << "There is a problem "
								"with the square root" << endl;
						exit(1);
					}
					y1 = y;
					y2 = F->negate(y);
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					add_point_to_table(x, y2, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					//cout << nb++ << " : (" << x << ","
					// << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << ","
					// << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				y = F->frobenius_power(r, e - 1);
				add_point_to_table(x, y, 1);
				if (nb == bound) {
					cout << "The number of points exceeds "
							"the bound" << endl;
					exit(1);
				}
				//cout << nb++ << " : (" << x << ","
				// << y << ",1)" << endl;
			}
#endif

		}
	}
	{
		vector<int> pt;

		pt.push_back(0);
		pt.push_back(1);
		pt.push_back(0);
		Pts.push_back(pt);
	}
	int i;
	cout << "We found " << Pts.size() << " points:" << endl;

	for (i = 0; i < (int) Pts.size(); i++) {
		if (i == (int) Pts.size()) {

			cout << i << " : {\\cal O} : 1\\\\" << endl;

		}
		else {
			{
			vector<vector<int>> Multiples;
			int order;


			F->elliptic_curve_all_point_multiples(
					EC_b, EC_c, order,
					Pts[i][0], Pts[i][1], 1,
					Multiples,
					0 /*verbose_level*/);

			//cout << "we found that the point has order " << order << endl;

			cout << i << " : $(" << Pts[i][0] << "," << Pts[i][1] << ")$ : " << order << "\\\\" << endl;
			}
		}
	}


	if (f_v) {
		cout << "do_EC_points done" << endl;
	}
}

int interface_cryptography::EC_evaluate_RHS(finite_field *F,
		int EC_b, int EC_c, int x)
// evaluates x^3 + bx + c
{
	int x2, x3, t;

	x2 = F->mult(x, x);
	x3 = F->mult(x2, x);
	t = F->add(x3, F->mult(EC_b, x));
	t = F->add(t, EC_c);
	return t;
}


void interface_cryptography::do_EC_add(int q,
		int EC_b, int EC_c,
		const char *pt1_text, const char *pt2_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int *v;
	int len;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	if (f_v) {
		cout << "do_EC_add" << endl;
	}
	vector<vector<int>> Pts;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt1_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);

	int_vec_scan(pt2_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x2 = v[0];
	y2 = v[1];
	z2 = 1;
	FREE_int(v);


	F->elliptic_curve_addition(EC_b, EC_c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3,
			verbose_level);
	cout << "(" << x1 << "," << y1 << "," << z1 << ")";
	cout << " + ";
	cout << "(" << x2 << "," << y2 << "," << z2 << ")";
	cout << " = ";
	cout << "(" << x3 << "," << y3 << "," << z3 << ")";
	cout << endl;


	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_add done" << endl;
	}
}

void interface_cryptography::do_EC_cyclic_subgroup(int q,
		int EC_b, int EC_c, const char *pt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int *v;
	int len, i;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	if (f_v) {
		cout << "do_EC_cyclic_subgroup" << endl;
	}
	vector<vector<int>> Pts;
	int order;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	F->elliptic_curve_all_point_multiples(
			EC_b, EC_c, order,
			x1, y1, z1,
			Pts,
			verbose_level);

	cout << "we found that the point has order " << order << endl;
	cout << "The multiples are:" << endl;
	cout << "i : (" << x1 << "," << y1 << ")" << endl;
	for (i = 0; i < (int) Pts.size(); i++) {

		vector<int> pts = Pts[i];

		if (i < (int) Pts.size() - 1) {
			cout << setw(3) << i + 1 << " : ";
			cout << "$(" << pts[0] << "," << pts[1] << ")$";
			cout << "\\\\" << endl;
		}
		else {
			cout << setw(3) << i + 1 << " : ";
			cout << "${\\cal O}$";
			cout << "\\\\" << endl;

		}
	}

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_cyclic_subgroup done" << endl;
	}
}

void interface_cryptography::do_EC_multiple_of(int q,
		int EC_b, int EC_c, const char *pt_text, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;

	if (f_v) {
		cout << "do_EC_multiple_of" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	F->elliptic_curve_point_multiple(
			EC_b, EC_c, n,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);

	cout << "The " << n << "-fold multiple of (" << x1 << "," << y1 << ") is ";
	if (z3 == 0) {

	}
	else {
		if (z3 != 1) {
			cout << "z1 != 1" << endl;
			exit(1);
		}
		cout << "(" << x3 << "," << y3 << ")" << endl;
	}

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_multiple_of done" << endl;
	}
}

void interface_cryptography::do_EC_discrete_log(int q,
		int EC_b, int EC_c,
		const char *base_pt_text, const char *pt_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;
	int n;

	if (f_v) {
		cout << "do_EC_multiple_of" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	int_vec_scan(base_pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two ccordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	int_vec_scan(pt_text, v, len);
	if (len == 2) {
		x3 = v[0];
		y3 = v[1];
		z3 = 1;
	}
	else if (len == 3) {
		x3 = v[0];
		y3 = v[1];
		z3 = v[2];
	}
	else {
		cout << "the point should have either two or three coordinates" << endl;
		exit(1);
	}
	FREE_int(v);


	n = F->elliptic_curve_discrete_log(
			EC_b, EC_c,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);


	cout << "The discrete log of (" << x3 << "," << y3 << "," << z3 << ") "
			"w.r.t. (" << x1 << "," << y1 << "," << z1 << ") "
			"is " << n << endl;

	FREE_OBJECT(F);

	if (f_v) {
		cout << "do_EC_multiple_of done" << endl;
	}
}

void interface_cryptography::do_EC_baby_step_giant_step(int EC_q, int EC_b, int EC_c,
		const char *EC_bsgs_G, int EC_bsgs_N, const char *EC_bsgs_cipher_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int Gx, Gy, Gz;
	int nGx, nGy, nGz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int Ax, Ay, Az;
	int *v;
	int len;
	int n;

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(EC_q, 0 /*verbose_level*/);


	int_vec_scan(EC_bsgs_G, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);

	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step N = " << EC_bsgs_N << endl;
		cout << "interface_cryptography::do_EC_baby_step_giant_step n = " << n << endl;
	}

	int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h, i;

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step "
				"cipher_text_length = " << cipher_text_length << endl;
	}

	F->elliptic_curve_point_multiple(
			EC_b, EC_c, n,
			Gx, Gy, Gz,
			nGx, nGy, nGz,
			0 /*verbose_level*/);

	cout << "$" << n << " * G = (" << nGx << "," << nGy << ")$\\\\" << endl;

	cout << " & ";
	for (h = 0; h < cipher_text_length; h++) {
		Cx = v[2 * h + 0];
		Cy = v[2 * h + 1];
		Cz = 1;
		cout << " & (" << Cx << "," << Cy << ")";
	}
	cout << endl;

	for (i = 1; i <= n + 1; i++) {

		F->elliptic_curve_point_multiple(
				EC_b, EC_c, i,
				Gx, Gy, Gz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << i << " & (" << Mx << "," << My << ")";

		for (h = 0; h < cipher_text_length; h++) {
			Cx = v[2 * h + 0];
			Cy = v[2 * h + 1];
			Cz = 1;

			F->elliptic_curve_point_multiple(
					EC_b, EC_c, i,
					nGx, nGy, nGz,
					Mx, My, Mz,
					0 /*verbose_level*/);

			My = F->negate(My);



			F->elliptic_curve_addition(EC_b, EC_c,
					Cx, Cy, Cz,
					Mx, My, Mz,
					Ax, Ay, Az,
					0 /*verbose_level*/);

			cout << " & (" << Ax << "," << Ay << ")";

		}
		cout << "\\\\" << endl;
	}



	FREE_int(v);

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step done" << endl;
	}
}

void interface_cryptography::do_EC_baby_step_giant_step_decode(
		int EC_q, int EC_b, int EC_c,
		const char *EC_bsgs_A, int EC_bsgs_N,
		const char *EC_bsgs_cipher_text_T, const char *EC_bsgs_keys,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	int Ax, Ay, Az;
	int Tx, Ty, Tz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int *v;
	int len;
	int n;
	int *keys;
	int nb_keys;
	int u, plain;

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init(EC_q, 0 /*verbose_level*/);

	u = EC_q / 27;
	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode u = " << u << endl;
	}


	int_vec_scan(EC_bsgs_A, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Ax = v[0];
	Ay = v[1];
	Az = 1;
	FREE_int(v);

	int_vec_scan(EC_bsgs_keys, keys, nb_keys);


	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode N = " << EC_bsgs_N << endl;
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode n = " << n << endl;
	}

	int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h;

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode "
				"cipher_text_length = " << cipher_text_length << endl;
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode "
				"nb_keys = " << nb_keys << endl;
	}
	if (nb_keys != cipher_text_length) {
		cout << "nb_keys != cipher_text_length" << endl;
		exit(1);
	}


	for (h = 0; h < cipher_text_length; h++) {
		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;
		cout << h << " & (" << Tx << "," << Ty << ")\\\\" << endl;;
	}
	cout << endl;


	for (h = 0; h < cipher_text_length; h++) {



		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;


		F->elliptic_curve_point_multiple(
				EC_b, EC_c, keys[h],
				Ax, Ay, Az,
				Cx, Cy, Cz,
				0 /*verbose_level*/);

		Cy = F->negate(Cy);


		cout << h << " & " << keys[h]
			<< " & (" << Tx << "," << Ty << ")"
			<< " & (" << Cx << "," << Cy << ")";


		F->elliptic_curve_addition(EC_b, EC_c,
				Tx, Ty, Tz,
				Cx, Cy, Cz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << " & (" << Mx << "," << My << ")";

		plain = Mx / u;
		cout << " & " << plain << " & " << (char)('A' - 1 + plain) << "\\\\" << endl;

	}


	FREE_int(v);
	FREE_int(keys);

	if (f_v) {
		cout << "interface_cryptography::do_EC_baby_step_giant_step_decode done" << endl;
	}
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
	char fname[1000];
	file_io Fio;

	snprintf(fname, 1000, "orbit_cnt%d_m%d_a%d_c%d.csv", cnt, m, a, c);
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

void interface_cryptography::do_random(int random_nb, const char *fname_csv, int verbose_level)
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

void interface_cryptography::do_RSA_encrypt_text(long int RSA_d, long int RSA_m,
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

void interface_cryptography::do_RSA(long int RSA_d, long int RSA_m, const char *RSA_text, int verbose_level)
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

