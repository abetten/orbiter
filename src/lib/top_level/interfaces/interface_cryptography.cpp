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
namespace top_level {


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
	//std::string ptext;
	//std::string ctext;
	//std::string guess;
	//std::string key;
	//cout << "interface_cryptography::interface_cryptography 00e" << endl;

	f_RSA = FALSE;
	RSA_d = 0;
	RSA_m = 0;
	//RSA_text = NULL;
	//cout << "interface_cryptography::interface_cryptography 00f" << endl;

	f_primitive_root = FALSE;
	//std::string primitive_root_p;

	f_smallest_primitive_root = FALSE;
	smallest_primitive_root_p = 0;

	f_smallest_primitive_root_interval = FALSE;
	smallest_primitive_root_interval_min = 0;
	smallest_primitive_root_interval_max = 0;

	f_number_of_primitive_roots_interval = FALSE;

	f_inverse_mod = FALSE;
	inverse_mod_a = 0;
	inverse_mod_n = 0;
	//cout << "interface_cryptography::interface_cryptography 00g" << endl;

	f_extended_gcd = FALSE;
	extended_gcd_a = 0;
	extended_gcd_b = 0;

	f_power_mod = FALSE;
	//std::string power_mod_a;
	//std::string power_mod_k;
	//std::string power_mod_n;
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

	f_all_square_roots_mod_n = FALSE;
	//std::string f_all_square_roots_mod_n_a;
	//std::string f_all_square_roots_mod_n_n;

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






void interface_cryptography::print_help(int argc, std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-cipher_substitution") == 0) {
		cout << "-cipher_substitution <ptext>" << endl;
	}
	else if (stringcmp(argv[i], "-cipher_vigenere") == 0) {
		cout << "-cipher_vigenere <ptext> <key>" << endl;
	}
	else if (stringcmp(argv[i], "-cipher_affine") == 0) {
		cout << "-cipher_affine <ptext> <int : a> <int : b>" << endl;
	}
	else if (stringcmp(argv[i], "-analyze_substitution") == 0) {
		cout << "-analyze_substitution <ctext>" << endl;
	}
	else if (stringcmp(argv[i], "-analyze_vigenere") == 0) {
		cout << "-analyze_vigenere <ctext>" << endl;
	}
	else if (stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		cout << "-analyze_vigenere_kasiski <ctext> <int : key_length>" << endl;
	}
	else if (stringcmp(argv[i], "-kasiski") == 0) {
		cout << "-kasiski <ctext> <int : threshold>" << endl;
	}
	else if (stringcmp(argv[i], "-decipher_substitution") == 0) {
		cout << "-decipher_substitution <ctext> <guess>" << endl;
	}
	else if (stringcmp(argv[i], "-decipher_vigenere") == 0) {
		cout << "-decipher_vigenere <ctext> <key>" << endl;
	}
	else if (stringcmp(argv[i], "-decipher_affine") == 0) {
		cout << "-decipher_affine <ctext> <guess>" << endl;
	}
	else if (stringcmp(argv[i], "-RSA") == 0) {
		cout << "-RSA <int : d> <int : m> <text>" << endl;
	}
	else if (stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		cout << "-RSA_encrypt_text <int : d> <int : m> <int : block_size> <text>" << endl;
	}
	else if (stringcmp(argv[i], "-RSA_setup") == 0) {
		cout << "-RSA_setup <int : nb_bits> <int : nb_tests_solovay_strassen> <int : f_miller_rabin_test>" << endl;
	}
	else if (stringcmp(argv[i], "-primitive_root") == 0) {
		cout << "-primitive_root <int : p>" << endl;
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		cout << "-smallest_primitive_root <int : p>" << endl;
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		cout << "-smallest_primitive_root_interval <int : p_min> <int : p_max>" << endl;
	}
	else if (stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		cout << "-number_of_primitive_roots_interval <int : p_min> <int : p_max>" << endl;
	}
	else if (stringcmp(argv[i], "-inverse_mod") == 0) {
		cout << "-primitive_root <int : a> <int : n>" << endl;
	}
	else if (stringcmp(argv[i], "-extended_gcd") == 0) {
		cout << "-extended_gcd <int : a> <int : b>" << endl;
	}
	else if (stringcmp(argv[i], "-power_mod") == 0) {
		cout << "-power_mod <int : a> <int : k> <int : n>" << endl;
	}
	else if (stringcmp(argv[i], "-discrete_log") == 0) {
		cout << "-discrete_log <int : y> <int : a> <int : m>" << endl;
	}
	else if (stringcmp(argv[i], "-sift_smooth") == 0) {
		cout << "-sift_smooth <int : from> <int : ken> <string : factor_base>" << endl;
	}
	else if (stringcmp(argv[i], "-square_root") == 0) {
		cout << "-square_root <int : number>" << endl;
	}
	else if (stringcmp(argv[i], "-square_root_mod") == 0) {
		cout << "-square_root_mod <int : a> <int : m>" << endl;
	}
	else if (stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		cout << "-all_square_roots_mod_n <int : a> <int : n>" << endl;
	}
	else if (stringcmp(argv[i], "-quadratic_sieve") == 0) {
		cout << "-quadratic_sieve <int : n> <string : factor_base> <int : x0>" << endl;
	}
	else if (stringcmp(argv[i], "-jacobi") == 0) {
		cout << "-jacobi <int : top> <int : bottom>" << endl;
	}
	else if (stringcmp(argv[i], "-solovay_strassen") == 0) {
		cout << "-solovay_strassen <int : a> <int : p>" << endl;
	}
	else if (stringcmp(argv[i], "-miller_rabin") == 0) {
		cout << "-miller_rabin <int : p> <int : nb_times>" << endl;
	}
	else if (stringcmp(argv[i], "-fermat_test") == 0) {
		cout << "-fermat_test <int : p> <int : nb_times>" << endl;
	}
	else if (stringcmp(argv[i], "-find_pseudoprime") == 0) {
		cout << "-find_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin> <int : nb_solovay_strassen>" << endl;
	}
	else if (stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		cout << "-find_strong_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin>" << endl;
	}
	else if (stringcmp(argv[i], "-miller_rabin_text") == 0) {
		cout << "-fermat_test <int : nb_times> <string : number>" << endl;
	}
	else if (stringcmp(argv[i], "-random") == 0) {
		cout << "-random <int : nb_times> <string : fname_csv>" << endl;
	}
	else if (stringcmp(argv[i], "-random_last") == 0) {
		cout << "-random_last <int : nb_times>" << endl;
	}
	else if (stringcmp(argv[i], "-affine_sequence") == 0) {
		cout << "-affine_sequence <int : a> <int : c> <int : m>" << endl;
	}
}

int interface_cryptography::recognize_keyword(int argc, std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_cryptography::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-cipher_substitution") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-cipher_vigenere") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-cipher_affine") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-analyze_substitution") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-analyze_vigenere") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-kasiski") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-decipher_substitution") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-decipher_vigenere") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-decipher_affine") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-RSA") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-RSA_setup") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-primitive_root") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-inverse_mod") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-extended_gcd") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-power_mod") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-discrete_log") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-sift_smooth") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-square_root") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-square_root_mod") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-quadratic_sieve") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-jacobi") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-solovay_strassen") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-miller_rabin") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-fermat_test") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-find_pseudoprime") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-miller_rabin_text") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-random") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-random_last") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-affine_sequence") == 0) {
		return true;
	}
	return false;
}

void interface_cryptography::read_arguments(int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_cryptography::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_cryptography::read_arguments the next argument is " << argv[i] << endl;
	}

	if (stringcmp(argv[i], "-cipher_substitution") == 0) {
		f_cipher = TRUE;
		t = substitution;
		ptext.assign(argv[++i]);
		if (f_v) {
			cout << "-cipher_substitution " << ptext << endl;
		}
	}
	else if (stringcmp(argv[i], "-cipher_vigenere") == 0) {
		f_cipher = TRUE;
		t = vigenere;
		ptext.assign(argv[++i]);
		key.assign(argv[++i]);
		if (f_v) {
			cout << "-cipher_vigenere " << ptext << endl;
		}
	}
	else if (stringcmp(argv[i], "-cipher_affine") == 0) {
		f_cipher = TRUE;
		t = affine;
		ptext.assign(argv[++i]);
		affine_a = strtoi(argv[++i]);
		affine_b = strtoi(argv[++i]);
		if (f_v) {
			cout << "-cipher_affine " << ptext << endl;
		}
	}
	else if (stringcmp(argv[i], "-analyze_substitution") == 0) {
		f_analyze = TRUE;
		t = substitution;
		ctext.assign(argv[++i]);
		if (f_v) {
			cout << "-analyze_substitution " << ctext << endl;
		}
	}
	else if (stringcmp(argv[i], "-analyze_vigenere") == 0) {
		f_analyze = TRUE;
		t = vigenere;
		ctext.assign(argv[++i]);
		if (f_v) {
			cout << "-analyze_vigenere " << ctext << endl;
		}
	}
	else if (stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		f_avk = TRUE;
		ctext.assign(argv[++i]);
		key_length = strtoi(argv[++i]);
		if (f_v) {
			cout << "-analyze_vigenere_kasiski " << ctext << endl;
		}
	}
	else if (stringcmp(argv[i], "-kasiski") == 0) {
		f_kasiski = TRUE;
		ctext.assign(argv[++i]);
		threshold = strtoi(argv[++i]);
		if (f_v) {
			cout << "-kasiski " << ctext << endl;
		}
	}
	else if (stringcmp(argv[i], "-decipher_substitution") == 0) {
		f_decipher = TRUE;
		t = substitution;
		ctext.assign(argv[++i]);
		guess.assign(argv[++i]);
		if (f_v) {
			cout << "-decipher_substitution " << ctext << " " << guess << endl;
		}
	}
	else if (stringcmp(argv[i], "-decipher_vigenere") == 0) {
		f_decipher = TRUE;
		t = vigenere;
		ctext.assign(argv[++i]);
		key.assign(argv[++i]);
	}
	else if (stringcmp(argv[i], "-decipher_affine") == 0) {
		f_decipher = TRUE;
		t = affine;
		ctext.assign(argv[++i]);
		guess.assign(argv[++i]);
	}
	else if (stringcmp(argv[i], "-RSA") == 0) {
		f_RSA = TRUE;
		RSA_d = strtoi(argv[++i]);
		RSA_m = strtoi(argv[++i]);
		RSA_block_size = strtoi(argv[++i]);
		RSA_text.assign(argv[++i]);
		if (f_v) {
			cout << "-RSA " << RSA_d << " " << RSA_m << " " << RSA_block_size << " " << RSA_text << endl;
		}
	}
	else if (stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		f_RSA_encrypt_text = TRUE;
		RSA_d = strtoi(argv[++i]);
		RSA_m = strtoi(argv[++i]);
		RSA_block_size = strtoi(argv[++i]);
		RSA_encrypt_text.assign(argv[++i]);
		if (f_v) {
			cout << "-RSA_encrypt_text " << RSA_d << " "
					<< RSA_m << " " << RSA_block_size << " " << RSA_encrypt_text << endl;
		}
	}
	else if (stringcmp(argv[i], "-RSA_setup") == 0) {
		f_RSA_setup = TRUE;
		RSA_setup_nb_bits = strtoi(argv[++i]);
		RSA_setup_nb_tests_solovay_strassen = strtoi(argv[++i]);
		RSA_setup_f_miller_rabin_test = strtoi(argv[++i]);
		if (f_v) {
			cout << "-RSA_setup " << RSA_setup_nb_bits << " "
					<< RSA_setup_nb_tests_solovay_strassen << " "
					<< RSA_setup_f_miller_rabin_test << endl;
		}
	}
	else if (stringcmp(argv[i], "-primitive_root") == 0) {
		f_primitive_root = TRUE;
		primitive_root_p.assign(argv[++i]);
		if (f_v) {
			cout << "-primitive_root " << primitive_root_p << endl;
		}
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		f_smallest_primitive_root = TRUE;
		smallest_primitive_root_p = strtoi(argv[++i]);
		if (f_v) {
			cout << "-smallest_primitive_root " << smallest_primitive_root_p << endl;
		}
	}
	else if (stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		f_smallest_primitive_root_interval = TRUE;
		smallest_primitive_root_interval_min = strtoi(argv[++i]);
		smallest_primitive_root_interval_max = strtoi(argv[++i]);
		if (f_v) {
			cout << "-smallest_primitive_root_interval " << smallest_primitive_root_interval_min
					<< " " << smallest_primitive_root_interval_max << endl;
		}
	}
	else if (stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		f_number_of_primitive_roots_interval = TRUE;
		smallest_primitive_root_interval_min = strtoi(argv[++i]);
		smallest_primitive_root_interval_max = strtoi(argv[++i]);
		if (f_v) {
			cout << "-number_of_primitive_roots_interval " << smallest_primitive_root_interval_min
					<< " " << smallest_primitive_root_interval_max << endl;
		}
	}
	else if (stringcmp(argv[i], "-inverse_mod") == 0) {
		f_inverse_mod = TRUE;
		inverse_mod_a = strtoi(argv[++i]);
		inverse_mod_n = strtoi(argv[++i]);
		if (f_v) {
			cout << "-inverse_mod " << inverse_mod_a << " " << inverse_mod_n << endl;
		}
	}
	else if (stringcmp(argv[i], "-extended_gcd") == 0) {
		f_extended_gcd = TRUE;
		extended_gcd_a = strtoi(argv[++i]);
		extended_gcd_b = strtoi(argv[++i]);
		if (f_v) {
			cout << "-extended_gcd " << extended_gcd_a << " " << extended_gcd_b << endl;
		}
	}
	else if (stringcmp(argv[i], "-power_mod") == 0) {
		f_power_mod = TRUE;
		power_mod_a.assign(argv[++i]);
		power_mod_k.assign(argv[++i]);
		power_mod_n.assign(argv[++i]);
		if (f_v) {
			cout << "-power_mod " << power_mod_a << " " << power_mod_k << " " << power_mod_n << endl;
		}
	}
	else if (stringcmp(argv[i], "-discrete_log") == 0) {
		f_discrete_log = TRUE;
		discrete_log_y = strtoi(argv[++i]);
		discrete_log_a = strtoi(argv[++i]);
		discrete_log_m = strtoi(argv[++i]);
		if (f_v) {
			cout << "-discrete_log " << discrete_log_y << " "
					<< discrete_log_a << " " << discrete_log_m << endl;
		}
	}
	else if (stringcmp(argv[i], "-sift_smooth") == 0) {
		f_sift_smooth = TRUE;
		sift_smooth_from = strtoi(argv[++i]);
		sift_smooth_len = strtoi(argv[++i]);
		sift_smooth_factor_base = argv[++i];
		if (f_v) {
			cout << "-sift_smooth " << sift_smooth_from << " "
					<< sift_smooth_len << " " << sift_smooth_factor_base << endl;
		}
	}
	else if (stringcmp(argv[i], "-square_root") == 0) {
		f_square_root = TRUE;
		square_root_number.assign(argv[++i]);
		if (f_v) {
			cout << "-square_root " << square_root_number << endl;
		}
	}
	else if (stringcmp(argv[i], "-square_root_mod") == 0) {
		f_square_root_mod = TRUE;
		square_root_mod_a.assign(argv[++i]);
		square_root_mod_m.assign(argv[++i]);
		if (f_v) {
			cout << "-square_root_mod " << square_root_mod_a << " "
					<< square_root_mod_m << endl;
		}
	}
	else if (stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		f_all_square_roots_mod_n = TRUE;
		all_square_roots_mod_n_a.assign(argv[++i]);
		all_square_roots_mod_n_n.assign(argv[++i]);
		if (f_v) {
			cout << "-all_square_roots_mod_n " << all_square_roots_mod_n_a << " "
					<< all_square_roots_mod_n_n << endl;
		}
	}
	else if (stringcmp(argv[i], "-quadratic_sieve") == 0) {
		f_quadratic_sieve = TRUE;
		quadratic_sieve_n = strtoi(argv[++i]);
		quadratic_sieve_factorbase = strtoi(argv[++i]);
		quadratic_sieve_x0 = strtoi(argv[++i]);
		if (f_v) {
			cout << "-quadratic_sieve " << quadratic_sieve_n << " "
					<< quadratic_sieve_factorbase << " " << quadratic_sieve_x0 << endl;
		}
	}
	else if (stringcmp(argv[i], "-jacobi") == 0) {
		f_jacobi = TRUE;
		jacobi_top = strtoi(argv[++i]);
		jacobi_bottom = strtoi(argv[++i]);
		if (f_v) {
			cout << "-jacobi " << jacobi_top << " "
					<< jacobi_bottom << endl;
		}
	}
	else if (stringcmp(argv[i], "-solovay_strassen") == 0) {
		f_solovay_strassen = TRUE;
		solovay_strassen_p = strtoi(argv[++i]);
		solovay_strassen_a = strtoi(argv[++i]);
		if (f_v) {
			cout << "-solovay_strassen " << solovay_strassen_p << " "
					<< solovay_strassen_a << endl;
		}
	}
	else if (stringcmp(argv[i], "-miller_rabin") == 0) {
		f_miller_rabin = TRUE;
		miller_rabin_p = strtoi(argv[++i]);
		miller_rabin_nb_times = strtoi(argv[++i]);
		if (f_v) {
			cout << "-miller_rabin " << miller_rabin_p << " " << miller_rabin_nb_times << endl;
		}
	}
	else if (stringcmp(argv[i], "-fermat_test") == 0) {
		f_fermat_test = TRUE;
		fermat_test_p = strtoi(argv[++i]);
		fermat_test_nb_times = strtoi(argv[++i]);
		if (f_v) {
			cout << "-fermat_test " << fermat_test_p << " " << fermat_test_nb_times << endl;
		}
	}
	else if (stringcmp(argv[i], "-find_pseudoprime") == 0) {
		f_find_pseudoprime = TRUE;
		find_pseudoprime_nb_digits = strtoi(argv[++i]);
		find_pseudoprime_nb_fermat = strtoi(argv[++i]);
		find_pseudoprime_nb_miller_rabin = strtoi(argv[++i]);
		find_pseudoprime_nb_solovay_strassen = strtoi(argv[++i]);
		if (f_v) {
			cout << "-find_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< " " << find_pseudoprime_nb_solovay_strassen << endl;
		}
	}
	else if (stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		f_find_strong_pseudoprime = TRUE;
		find_pseudoprime_nb_digits = strtoi(argv[++i]);
		find_pseudoprime_nb_fermat = strtoi(argv[++i]);
		find_pseudoprime_nb_miller_rabin = strtoi(argv[++i]);
		if (f_v) {
			cout << "-find_strong_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< endl;
		}
	}
	else if (stringcmp(argv[i], "-miller_rabin_text") == 0) {
		f_miller_rabin_text = TRUE;
		miller_rabin_text_nb_times = strtoi(argv[++i]);
		miller_rabin_number_text.assign(argv[++i]);
		if (f_v) {
			cout << "-miller_rabin " << miller_rabin_text_nb_times
					<< " " << miller_rabin_number_text
					<< endl;
		}
	}
	else if (stringcmp(argv[i], "-random") == 0) {
		f_random = TRUE;
		random_nb = strtoi(argv[++i]);
		random_fname_csv.assign(argv[++i]);
		if (f_v) {
			cout << "-random " << random_nb << " " << random_fname_csv << endl;
		}
	}
	else if (stringcmp(argv[i], "-random_last") == 0) {
		f_random_last = TRUE;
		random_last_nb = strtoi(argv[++i]);
		if (f_v) {
			cout << "-random_last " << random_last_nb << endl;
		}
	}
	else if (stringcmp(argv[i], "-affine_sequence") == 0) {
		f_affine_sequence = TRUE;
		affine_sequence_a = strtoi(argv[++i]);
		affine_sequence_c = strtoi(argv[++i]);
		affine_sequence_m = strtoi(argv[++i]);
		if (f_v) {
			cout << "-affine_sequence " << affine_sequence_a
					<< " " << affine_sequence_c << " " << affine_sequence_m << endl;
		}
	}
	if (f_v) {
		cout << "interface_cryptography::read_arguments done" << endl;
	}
}

void interface_cryptography::print()
{
	if (f_cipher && t == substitution) {
		cout << "-cipher_substitution " << ptext << endl;
	}
	if (f_cipher && t == vigenere) {
		cout << "cipher_vigenere" << ptext << " " << key << endl;
	}
	if (f_cipher && t == affine) {
		cout << "-cipher_affine " << ptext << " " << affine_a << " " << affine_b << endl;
	}
	if (f_analyze && t == substitution) {
		cout << "-analyze_substitution " << ctext << endl;
	}
	if (f_analyze && t == vigenere) {
		cout << "-analyze_vigenere " << ctext << endl;
	}
	if (f_avk) {
		cout << "-analyze_vigenere_kasiski " << ctext << " " << key_length << endl;
	}
	if (f_kasiski) {
		cout << "-kasiski " << ctext << " " << threshold << endl;
	}
	if (f_decipher && t == substitution) {
		cout << "-decipher_substitution " << ctext << " " << guess << endl;
	}
	if (f_decipher && t == vigenere) {
		cout << "-decipher_vigenere " << ctext << " " << key << endl;
	}
	if (f_decipher && t == affine) {
		cout << "-decipher_affine " << ctext << " " << guess << endl;
	}
	if (f_RSA) {
		cout << "-RSA " << RSA_d << " " << RSA_m << " " << RSA_block_size << " " << RSA_text << endl;
	}
	if (f_RSA_encrypt_text) {
		cout << "-RSA_encrypt_text " << RSA_d << " "
				<< RSA_m << " " << RSA_block_size << " " << RSA_encrypt_text << endl;
	}
	if (f_RSA_setup) {
		cout << "-RSA_setup " << RSA_setup_nb_bits << " "
				<< RSA_setup_nb_tests_solovay_strassen << " "
				<< RSA_setup_f_miller_rabin_test << endl;
	}
	if (f_primitive_root) {
		cout << "-primitive_root " << primitive_root_p << endl;
	}
	if (f_smallest_primitive_root) {
		cout << "-smallest_primitive_root " << smallest_primitive_root_p << endl;
	}
	if (f_smallest_primitive_root_interval) {
		cout << "-smallest_primitive_root_interval " << smallest_primitive_root_interval_min
				<< " " << smallest_primitive_root_interval_max << endl;
	}
	if (f_number_of_primitive_roots_interval) {
		cout << "-number_of_primitive_roots_interval " << smallest_primitive_root_interval_min
				<< " " << smallest_primitive_root_interval_max << endl;
	}
	if (f_inverse_mod) {
		cout << "-inverse_mod " << inverse_mod_a << " " << inverse_mod_n << endl;
	}
	if (f_extended_gcd) {
		cout << "-extended_gcd " << extended_gcd_a << " " << extended_gcd_b << endl;
	}
	if (f_power_mod) {
		cout << "-power_mod " << power_mod_a << " " << power_mod_k << " " << power_mod_n << endl;
	}
	if (f_discrete_log) {
		cout << "-discrete_log " << discrete_log_y << " "
				<< discrete_log_a << " " << discrete_log_m << endl;
	}
	if (f_sift_smooth) {
		cout << "-sift_smooth " << sift_smooth_from << " "
				<< sift_smooth_len << " " << sift_smooth_factor_base << endl;
	}
	if (f_square_root) {
		cout << "-square_root " << square_root_number << endl;
	}
	if (f_square_root_mod) {
		cout << "-square_root_mod " << square_root_mod_a << " "
				<< square_root_mod_m << endl;
	}
	if (f_all_square_roots_mod_n) {
		cout << "-all_square_roots_mod_n " << all_square_roots_mod_n_a << " "
				<< all_square_roots_mod_n_n << endl;
	}
	if (f_quadratic_sieve) {
		cout << "-quadratic_sieve " << quadratic_sieve_n << " "
				<< quadratic_sieve_factorbase << " " << quadratic_sieve_x0 << endl;
	}
	if (f_jacobi) {
		cout << "-jacobi " << jacobi_top << " "
				<< jacobi_bottom << endl;
	}
	if (f_solovay_strassen) {
		cout << "-solovay_strassen " << solovay_strassen_p << " "
				<< solovay_strassen_a << endl;
	}
	if (f_miller_rabin) {
		cout << "-miller_rabin " << miller_rabin_p << " " << miller_rabin_nb_times << endl;
	}
	if (f_fermat_test) {
		cout << "-fermat_test " << fermat_test_p << " " << fermat_test_nb_times << endl;
	}
	if (f_find_pseudoprime) {
		cout << "-find_pseudoprime " << find_pseudoprime_nb_digits
				<< " " << find_pseudoprime_nb_fermat
				<< " " << find_pseudoprime_nb_miller_rabin
				<< " " << find_pseudoprime_nb_solovay_strassen << endl;
	}
	if (f_find_strong_pseudoprime) {
		cout << "-find_strong_pseudoprime " << find_pseudoprime_nb_digits
				<< " " << find_pseudoprime_nb_fermat
				<< " " << find_pseudoprime_nb_miller_rabin
				<< endl;
	}
	if (f_miller_rabin_text) {
		cout << "-miller_rabin " << miller_rabin_text_nb_times
				<< " " << miller_rabin_number_text
				<< endl;
	}
	if (f_random) {
		cout << "-random " << random_nb << " " << random_fname_csv << endl;
	}
	if (f_random_last) {
		cout << "-random_last " << random_last_nb << endl;
	}
	if (f_affine_sequence) {
		cout << "-affine_sequence " << affine_sequence_a
				<< " " << affine_sequence_c << " " << affine_sequence_m << endl;
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
			cout << "ctext: " << endl << ctext << endl;
			Crypto.analyze(ctext);
		}
		if (t == vigenere) {
			cout << "ctext: " << endl << ctext << endl;
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

		//longinteger_domain D;
		longinteger_object p;

		p.create_from_base_10_string(primitive_root_p);
		Crypto.do_primitive_root_longinteger(p, verbose_level);
	}
	else if (f_smallest_primitive_root) {

		cryptography_domain Crypto;

		Crypto.do_smallest_primitive_root(smallest_primitive_root_p, verbose_level);
	}
	else if (f_smallest_primitive_root_interval) {

		cryptography_domain Crypto;

		Crypto.do_smallest_primitive_root_interval(smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
	}
	else if (f_number_of_primitive_roots_interval) {

		cryptography_domain Crypto;

		Crypto.do_number_of_primitive_roots_interval(smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
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

		longinteger_object a;
		longinteger_object k;
		longinteger_object n;

		a.create_from_base_10_string(power_mod_a);
		k.create_from_base_10_string(power_mod_k);
		n.create_from_base_10_string(power_mod_n);

		Crypto.do_power_mod(a, k, n, verbose_level);
	}
	else if (f_RSA) {

		cryptography_domain Crypto;

		Crypto.do_RSA(RSA_d, RSA_m, RSA_block_size, RSA_text, verbose_level);
	}
	else if (f_RSA_encrypt_text) {

		cryptography_domain Crypto;

		Crypto.do_RSA_encrypt_text(RSA_d, RSA_m, RSA_block_size,
				RSA_encrypt_text, verbose_level);
	}
	else if (f_RSA_setup) {
		cryptography_domain Crypto;
		longinteger_object n, p, q, a, b;

		Crypto.RSA_setup(n, p, q, a, b,
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

	else if (f_all_square_roots_mod_n) {

		cryptography_domain Crypto;
		vector<long int> S;
		int i;

		Crypto.all_square_roots_mod_n_by_exhaustive_search_lint(
				all_square_roots_mod_n_a, all_square_roots_mod_n_n, S, verbose_level);

		cout << "We found " << S.size() << " square roots of "
				<< all_square_roots_mod_n_a << " mod " << all_square_roots_mod_n_n << endl;
		cout << "They are:" << endl;
		for (i = 0; i < S.size(); i++) {
			cout << i << " : " << S[i] << endl;
		}
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
}




}}

