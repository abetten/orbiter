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
	//RSA_text = NULL;
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
	//RSA_encrypt_text = NULL;
	f_sift_smooth = FALSE;
	sift_smooth_from = 0;
	sift_smooth_len = 0;
	//cout << "interface_cryptography::interface_cryptography 1" << endl;
	//sift_smooth_factor_base = NULL;
	f_square_root = FALSE;
	//square_root_number = NULL;
	f_square_root_mod = FALSE;
	//square_root_mod_a = NULL;
	//square_root_mod_m = NULL;
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
	//miller_rabin_number_text = NULL;
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

}






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
#if 0
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
#endif

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
#if 0
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
#endif

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
			RSA_text.assign(argv[++i]);
			cout << "-RSA " << RSA_d << " " << RSA_m << " " << RSA_text << endl;
		}
		else if (strcmp(argv[i], "-RSA_encrypt_text") == 0) {
			f_RSA_encrypt_text = TRUE;
			RSA_d = atol(argv[++i]);
			RSA_m = atol(argv[++i]);
			RSA_block_size = atoi(argv[++i]);
			RSA_encrypt_text.assign(argv[++i]);
			cout << "-RSA_encrypt_text " << RSA_d << " "
					<< RSA_m << " " << RSA_encrypt_text << endl;
		}
		else if (strcmp(argv[i], "-RSA_setup") == 0) {
			f_RSA_setup = TRUE;
			RSA_setup_nb_bits = atoi(argv[++i]);
			RSA_setup_nb_tests_solovay_strassen = atoi(argv[++i]);
			RSA_setup_f_miller_rabin_test = atoi(argv[++i]);
			cout << "-RSA_setup " << RSA_setup_nb_bits << " "
					<< RSA_setup_nb_tests_solovay_strassen << " "
					<< RSA_setup_f_miller_rabin_test << endl;
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
			cout << "-discrete_log " << discrete_log_y << " "
					<< discrete_log_a << " " << discrete_log_m << endl;
		}
		else if (strcmp(argv[i], "-sift_smooth") == 0) {
			f_sift_smooth = TRUE;
			sift_smooth_from = atol(argv[++i]);
			sift_smooth_len = atol(argv[++i]);
			sift_smooth_factor_base = argv[++i];
			cout << "-sift_smooth " << sift_smooth_from << " "
					<< sift_smooth_len << " " << sift_smooth_factor_base << endl;
		}
		else if (strcmp(argv[i], "-square_root") == 0) {
			f_square_root = TRUE;
			square_root_number.assign(argv[++i]);
			cout << "-square_root " << square_root_number << endl;
		}
		else if (strcmp(argv[i], "-square_root_mod") == 0) {
			f_square_root_mod = TRUE;
			square_root_mod_a.assign(argv[++i]);
			square_root_mod_m.assign(argv[++i]);
			cout << "-square_root_mod " << square_root_mod_a << " "
					<< square_root_mod_m << endl;
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
			miller_rabin_number_text.assign(argv[++i]);
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

		cryptography_domain Crypto;

		if (t == substitution) {
			Crypto.get_random_permutation(key);
			cout << "ptext: " << ptext << endl;
			Crypto.substition_cipher(ptext, ctext, key);
			cout << "ctext: " << ctext << endl;
		}
		else if (t == vigenere) {
			cout << "vigenere cipher with key " << key << endl;
			cout << "ptext: " << ptext << endl;
			Crypto.vigenere_cipher(ptext, ctext, key);
			cout << "ctext: " << ctext << endl;
		}
		else if (t == affine) {
			cout << "affine cipher with key (" << affine_a << "," << affine_b << ")" << endl;
			cout << "ptext: " << ptext << endl;
			Crypto.affine_cipher(ptext, ctext, affine_a, affine_b);
			cout << "ctext: " << ctext << endl;
		}
	}
	else if (f_analyze) {

		cryptography_domain Crypto;

		if (t == substitution) {
			cout << "ctext: \n" << ctext << endl;
			Crypto.analyze(ctext);
		}
		if (t == vigenere) {
			cout << "ctext: \n" << ctext << endl;
			Crypto.vigenere_analysis(ctext);
		}
	}
	else if (f_avk) {

		cryptography_domain Crypto;

		Crypto.vigenere_analysis2(ctext, key_length);
	}
	else if (f_kasiski) {

		cryptography_domain Crypto;

		int m;

		m = Crypto.kasiski_test(ctext, threshold);
		cout << "kasiski test for threshold " << threshold
				<< " yields key length " << m << endl;
	}
	else if (f_decipher) {

		cryptography_domain Crypto;

		if (t == substitution) {
			cout << "ctext: " << ctext << endl;
			cout << "guess: " << guess << endl;
			Crypto.decipher(ctext, ptext, guess);
			cout << " " << endl;
			Crypto.print_on_top(ptext, ctext);
			//cout << "ptext: " << ptext << endl;
		}
		else if (t == vigenere) {
			cout << "ctext: " << ctext << endl;
			cout << "key  : " << key << endl;
			Crypto.vigenere_decipher(ctext, ptext, key);
			cout << " " << endl;
			Crypto.print_on_top(ptext, ctext);
		}
		else if (t == affine) {
			//cout << "affine cipher with key (" << affine_a << "," << affine_b << ")" << endl;
			cout << "affine cipher" << endl;
			cout << "ctext: " << ctext << endl;
			cout << "guess: " << guess << endl;
			Crypto.affine_decipher(ctext, ptext, guess);
			//cout << " " << endl;
			//print_on_top(ptext, ctext);
		}
	}
	else if (f_discrete_log) {

		cryptography_domain Crypto;


		Crypto.do_discrete_log(discrete_log_y, discrete_log_a, discrete_log_m, verbose_level);
	}
	else if (f_primitive_root) {

		cryptography_domain Crypto;

		Crypto.do_primitive_root(primitive_root_p, verbose_level);
	}
	else if (f_inverse_mod) {

		cryptography_domain Crypto;

		Crypto.do_inverse_mod(inverse_mod_a, inverse_mod_n, verbose_level);
	}
	else if (f_extended_gcd) {

		cryptography_domain Crypto;

		Crypto.do_extended_gcd(extended_gcd_a, extended_gcd_b, verbose_level);
	}
	else if (f_power_mod) {

		cryptography_domain Crypto;

		Crypto.do_power_mod(power_mod_a, power_mod_k, power_mod_n, verbose_level);
	}
	else if (f_RSA) {

		cryptography_domain Crypto;

		Crypto.do_RSA(RSA_d, RSA_m, RSA_text, verbose_level);
	}
	else if (f_RSA_encrypt_text) {

		cryptography_domain Crypto;

		Crypto.do_RSA_encrypt_text(RSA_d, RSA_m, RSA_block_size,
				RSA_encrypt_text, verbose_level);
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

		cryptography_domain Crypto;

		Crypto.do_sift_smooth(sift_smooth_from,
				sift_smooth_len,
				sift_smooth_factor_base, verbose_level);
	}
	else if (f_square_root) {

		cryptography_domain Crypto;

		Crypto.square_root(square_root_number, verbose_level);
	}
	else if (f_square_root_mod) {

		cryptography_domain Crypto;

		Crypto.square_root_mod(square_root_mod_a, square_root_mod_m, verbose_level);
	}
	else if (f_quadratic_sieve) {

		cryptography_domain Crypto;

		Crypto.quadratic_sieve(quadratic_sieve_n,
				quadratic_sieve_factorbase,
				quadratic_sieve_x0,
				verbose_level);
	}
	else if (f_jacobi) {

		cryptography_domain Crypto;

		Crypto.do_jacobi(jacobi_top, jacobi_bottom, verbose_level);
	}
	else if (f_solovay_strassen) {

		cryptography_domain Crypto;

		Crypto.do_solovay_strassen(solovay_strassen_p, solovay_strassen_a, verbose_level);
	}
	else if (f_miller_rabin) {

		cryptography_domain Crypto;

		Crypto.do_miller_rabin(miller_rabin_p, miller_rabin_nb_times, verbose_level);
	}
	else if (f_fermat_test) {

		cryptography_domain Crypto;

		Crypto.do_fermat_test(fermat_test_p, fermat_test_nb_times, verbose_level);
	}
	else if (f_find_pseudoprime) {

		cryptography_domain Crypto;

		Crypto.do_find_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				find_pseudoprime_nb_solovay_strassen,
				verbose_level);
	}
	else if (f_find_strong_pseudoprime) {

		cryptography_domain Crypto;

		Crypto.do_find_strong_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				verbose_level);
	}
	else if (f_miller_rabin_text) {

		cryptography_domain Crypto;

		Crypto.do_miller_rabin_text(
				miller_rabin_number_text, miller_rabin_text_nb_times,
				verbose_level);
	}
	else if (f_random) {

		cryptography_domain Crypto;

		Crypto.do_random(
				random_nb, random_fname_csv,
				verbose_level);
	}
	else if (f_random_last) {

		cryptography_domain Crypto;

		Crypto.do_random_last(
				random_last_nb,
				verbose_level);
	}
	else if (f_affine_sequence) {

		cryptography_domain Crypto;

		Crypto.make_affine_sequence(affine_sequence_a,
				affine_sequence_c, affine_sequence_m, verbose_level);
	}
#if 0
	else if (f_ntt) {
		number_theoretic_transform NTT;
		NTT.init(ntt_t, ntt_q, verbose_level);
	}
#endif

}




}}

