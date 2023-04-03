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
namespace layer5_applications {
namespace user_interface {


interface_cryptography::interface_cryptography()
{
	//cout << "interface_cryptography::interface_cryptography" << endl;
	//cout << "sizeof(interface_cryptography)=" << sizeof(interface_cryptography) << endl;

	f_cipher = false;
	t = no_cipher_type;
	//cout << "interface_cryptography::interface_cryptography 0" << endl;

	f_decipher = false;
	//cout << "interface_cryptography::interface_cryptography 00a" << endl;

	f_analyze = false;
	//cout << "interface_cryptography::interface_cryptography 00b" << endl;

	f_kasiski = false;
	f_avk = false;
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

	f_RSA = false;
	RSA_d = 0;
	RSA_m = 0;
	//RSA_text = NULL;
	//cout << "interface_cryptography::interface_cryptography 00f" << endl;

	f_primitive_root = false;
	//std::string primitive_root_p;

	f_smallest_primitive_root = false;
	smallest_primitive_root_p = 0;

	f_smallest_primitive_root_interval = false;
	smallest_primitive_root_interval_min = 0;
	smallest_primitive_root_interval_max = 0;

	f_number_of_primitive_roots_interval = false;

	f_inverse_mod = false;
	inverse_mod_a = 0;
	inverse_mod_n = 0;
	//cout << "interface_cryptography::interface_cryptography 00g" << endl;

	f_extended_gcd = false;
	extended_gcd_a = 0;
	extended_gcd_b = 0;

	f_power_mod = false;
	//std::string power_mod_a;
	//std::string power_mod_k;
	//std::string power_mod_n;
	//cout << "interface_cryptography::interface_cryptography 00h" << endl;

	f_discrete_log = false;
	//cout << "interface_cryptography::interface_cryptography 0b" << endl;
	discrete_log_y = 0;
	discrete_log_a = 0;
	discrete_log_m = 0;

	f_RSA_setup = false;
	RSA_setup_nb_bits = 0;
	RSA_setup_nb_tests_solovay_strassen = 0;
	RSA_setup_f_miller_rabin_test = 0;

	f_RSA_encrypt_text = false;
	RSA_block_size = 0;
	//RSA_encrypt_text = NULL;

	f_sift_smooth = false;
	sift_smooth_from = 0;
	sift_smooth_len = 0;
	//cout << "interface_cryptography::interface_cryptography 1" << endl;
	//sift_smooth_factor_base = NULL;

	f_square_root = false;
	//square_root_number = NULL;

	f_square_root_mod = false;
	//square_root_mod_a = NULL;
	//square_root_mod_m = NULL;
	//cout << "interface_cryptography::interface_cryptography 1a" << endl;

	f_all_square_roots_mod_n = false;
	//std::string f_all_square_roots_mod_n_a;
	//std::string f_all_square_roots_mod_n_n;

	f_quadratic_sieve = false;
	quadratic_sieve_n = 0;
	quadratic_sieve_factorbase = 0;
	quadratic_sieve_x0 = 0;
	//cout << "interface_cryptography::interface_cryptography 1b" << endl;

	f_jacobi = false;
	jacobi_top = 0;
	jacobi_bottom = 0;
	//cout << "interface_cryptography::interface_cryptography 1c" << endl;

	f_solovay_strassen = false;
	solovay_strassen_p = 0;
	solovay_strassen_a = 0;
	//cout << "interface_cryptography::interface_cryptography 1d" << endl;

	f_miller_rabin = false;
	miller_rabin_p = 0;
	miller_rabin_nb_times = 0;
	//cout << "interface_cryptography::interface_cryptography 1e" << endl;

	f_fermat_test = false;
	fermat_test_p = 0;
	fermat_test_nb_times = 0;
	//cout << "interface_cryptography::interface_cryptography 1f" << endl;

	f_find_pseudoprime = false;
	find_pseudoprime_nb_digits = 0;
	find_pseudoprime_nb_fermat = 0;
	find_pseudoprime_nb_miller_rabin = 0;
	find_pseudoprime_nb_solovay_strassen = 0;
	//cout << "interface_cryptography::interface_cryptography 1g" << endl;

	f_find_strong_pseudoprime = false;

	f_miller_rabin_text = false;
	miller_rabin_text_nb_times = 0;
	//miller_rabin_number_text = NULL;
	//cout << "interface_cryptography::interface_cryptography 1h" << endl;

	f_random = false;
	random_nb = 0;
	//random_fname_csv = NULL;

	f_random_last = false;
	random_last_nb = 0;
	//cout << "interface_cryptography::interface_cryptography 1i" << endl;

	f_affine_sequence = false;
	affine_sequence_a = 0;
	//cout << "interface_cryptography::interface_cryptography 2" << endl;
	affine_sequence_c = 0;
	affine_sequence_m = 0;

	f_Chinese_remainders = false;
	//std::string Chinese_remainders_R;
	//std::string Chinese_remainders_M;

}






void interface_cryptography::print_help(int argc, std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-cipher_substitution") == 0) {
		cout << "-cipher_substitution <ptext>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-cipher_vigenere") == 0) {
		cout << "-cipher_vigenere <ptext> <key>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-cipher_affine") == 0) {
		cout << "-cipher_affine <ptext> <int : a> <int : b>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-analyze_substitution") == 0) {
		cout << "-analyze_substitution <ctext>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere") == 0) {
		cout << "-analyze_vigenere <ctext>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		cout << "-analyze_vigenere_kasiski <ctext> <int : key_length>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-kasiski") == 0) {
		cout << "-kasiski <ctext> <int : threshold>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-decipher_substitution") == 0) {
		cout << "-decipher_substitution <ctext> <guess>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-decipher_vigenere") == 0) {
		cout << "-decipher_vigenere <ctext> <key>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-decipher_affine") == 0) {
		cout << "-decipher_affine <ctext> <guess>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-RSA") == 0) {
		cout << "-RSA <int : d> <int : m> <text>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		cout << "-RSA_encrypt_text <int : d> <int : m> <int : block_size> <text>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-RSA_setup") == 0) {
		cout << "-RSA_setup <int : nb_bits> <int : nb_tests_solovay_strassen> <int : f_miller_rabin_test>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
		cout << "-primitive_root <int : p>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		cout << "-smallest_primitive_root <int : p>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		cout << "-smallest_primitive_root_interval <int : p_min> <int : p_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		cout << "-number_of_primitive_roots_interval <int : p_min> <int : p_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-inverse_mod") == 0) {
		cout << "-primitive_root <int : a> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-extended_gcd") == 0) {
		cout << "-extended_gcd <int : a> <int : b>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-power_mod") == 0) {
		cout << "-power_mod <int : a> <int : k> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-discrete_log") == 0) {
		cout << "-discrete_log <int : y> <int : a> <int : m>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-sift_smooth") == 0) {
		cout << "-sift_smooth <int : from> <int : ken> <string : factor_base>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-square_root") == 0) {
		cout << "-square_root <int : number>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-square_root_mod") == 0) {
		cout << "-square_root_mod <int : a> <int : m>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		cout << "-all_square_roots_mod_n <int : a> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-quadratic_sieve") == 0) {
		cout << "-quadratic_sieve <int : n> <string : factor_base> <int : x0>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-jacobi") == 0) {
		cout << "-jacobi <int : top> <int : bottom>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-solovay_strassen") == 0) {
		cout << "-solovay_strassen <int : a> <int : p>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin") == 0) {
		cout << "-miller_rabin <int : p> <int : nb_times>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-fermat_test") == 0) {
		cout << "-fermat_test <int : p> <int : nb_times>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-find_pseudoprime") == 0) {
		cout << "-find_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin> <int : nb_solovay_strassen>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		cout << "-find_strong_pseudoprime <int : nb_digits> <int : nb_fermat> <int : nb_miller_rabin>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin_text") == 0) {
		cout << "-fermat_test <int : nb_times> <string : number>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-random") == 0) {
		cout << "-random <int : nb_times> <string : fname_csv>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-random_last") == 0) {
		cout << "-random_last <int : nb_times>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-affine_sequence") == 0) {
		cout << "-affine_sequence <int : a> <int : c> <int : m>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		cout << "-Chinese_remainders <string : Remainders> <string : Moduli>" << endl;
	}
}

int interface_cryptography::recognize_keyword(int argc, std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_cryptography::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-cipher_substitution") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-cipher_vigenere") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-cipher_affine") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-analyze_substitution") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-kasiski") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-decipher_substitution") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-decipher_vigenere") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-decipher_affine") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-RSA") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-RSA_setup") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-inverse_mod") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-extended_gcd") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-power_mod") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-discrete_log") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-sift_smooth") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-square_root") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-square_root_mod") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-quadratic_sieve") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-jacobi") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-solovay_strassen") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-fermat_test") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-find_pseudoprime") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin_text") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-random") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-random_last") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-affine_sequence") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		return true;
	}
	return false;
}

void interface_cryptography::read_arguments(int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_cryptography::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_cryptography::read_arguments the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-cipher_substitution") == 0) {
		f_cipher = true;
		t = substitution;
		ptext.assign(argv[++i]);
		if (f_v) {
			cout << "-cipher_substitution " << ptext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-cipher_vigenere") == 0) {
		f_cipher = true;
		t = vigenere;
		ptext.assign(argv[++i]);
		key.assign(argv[++i]);
		if (f_v) {
			cout << "-cipher_vigenere " << ptext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-cipher_affine") == 0) {
		f_cipher = true;
		t = affine;
		ptext.assign(argv[++i]);
		affine_a = ST.strtoi(argv[++i]);
		affine_b = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-cipher_affine " << ptext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-analyze_substitution") == 0) {
		f_analyze = true;
		t = substitution;
		ctext.assign(argv[++i]);
		if (f_v) {
			cout << "-analyze_substitution " << ctext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere") == 0) {
		f_analyze = true;
		t = vigenere;
		ctext.assign(argv[++i]);
		if (f_v) {
			cout << "-analyze_vigenere " << ctext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-analyze_vigenere_kasiski") == 0) {
		f_avk = true;
		ctext.assign(argv[++i]);
		key_length = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-analyze_vigenere_kasiski " << ctext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-kasiski") == 0) {
		f_kasiski = true;
		ctext.assign(argv[++i]);
		threshold = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-kasiski " << ctext << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-decipher_substitution") == 0) {
		f_decipher = true;
		t = substitution;
		ctext.assign(argv[++i]);
		guess.assign(argv[++i]);
		if (f_v) {
			cout << "-decipher_substitution " << ctext << " " << guess << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-decipher_vigenere") == 0) {
		f_decipher = true;
		t = vigenere;
		ctext.assign(argv[++i]);
		key.assign(argv[++i]);
	}
	else if (ST.stringcmp(argv[i], "-decipher_affine") == 0) {
		f_decipher = true;
		t = affine;
		ctext.assign(argv[++i]);
		guess.assign(argv[++i]);
	}
	else if (ST.stringcmp(argv[i], "-RSA") == 0) {
		f_RSA = true;
		RSA_d = ST.strtoi(argv[++i]);
		RSA_m = ST.strtoi(argv[++i]);
		RSA_block_size = ST.strtoi(argv[++i]);
		RSA_text.assign(argv[++i]);
		if (f_v) {
			cout << "-RSA " << RSA_d << " " << RSA_m << " " << RSA_block_size << " " << RSA_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-RSA_encrypt_text") == 0) {
		f_RSA_encrypt_text = true;
		RSA_d = ST.strtoi(argv[++i]);
		RSA_m = ST.strtoi(argv[++i]);
		RSA_block_size = ST.strtoi(argv[++i]);
		RSA_encrypt_text.assign(argv[++i]);
		if (f_v) {
			cout << "-RSA_encrypt_text " << RSA_d << " "
					<< RSA_m << " " << RSA_block_size << " " << RSA_encrypt_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-RSA_setup") == 0) {
		f_RSA_setup = true;
		RSA_setup_nb_bits = ST.strtoi(argv[++i]);
		RSA_setup_nb_tests_solovay_strassen = ST.strtoi(argv[++i]);
		RSA_setup_f_miller_rabin_test = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-RSA_setup " << RSA_setup_nb_bits << " "
					<< RSA_setup_nb_tests_solovay_strassen << " "
					<< RSA_setup_f_miller_rabin_test << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
		f_primitive_root = true;
		primitive_root_p.assign(argv[++i]);
		if (f_v) {
			cout << "-primitive_root " << primitive_root_p << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root") == 0) {
		f_smallest_primitive_root = true;
		smallest_primitive_root_p = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-smallest_primitive_root " << smallest_primitive_root_p << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-smallest_primitive_root_interval") == 0) {
		f_smallest_primitive_root_interval = true;
		smallest_primitive_root_interval_min = ST.strtoi(argv[++i]);
		smallest_primitive_root_interval_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-smallest_primitive_root_interval " << smallest_primitive_root_interval_min
					<< " " << smallest_primitive_root_interval_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-number_of_primitive_roots_interval") == 0) {
		f_number_of_primitive_roots_interval = true;
		smallest_primitive_root_interval_min = ST.strtoi(argv[++i]);
		smallest_primitive_root_interval_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-number_of_primitive_roots_interval " << smallest_primitive_root_interval_min
					<< " " << smallest_primitive_root_interval_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-inverse_mod") == 0) {
		f_inverse_mod = true;
		inverse_mod_a = ST.strtoi(argv[++i]);
		inverse_mod_n = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-inverse_mod " << inverse_mod_a << " " << inverse_mod_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-extended_gcd") == 0) {
		f_extended_gcd = true;
		extended_gcd_a = ST.strtoi(argv[++i]);
		extended_gcd_b = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-extended_gcd " << extended_gcd_a << " " << extended_gcd_b << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-power_mod") == 0) {
		f_power_mod = true;
		power_mod_a.assign(argv[++i]);
		power_mod_k.assign(argv[++i]);
		power_mod_n.assign(argv[++i]);
		if (f_v) {
			cout << "-power_mod " << power_mod_a << " " << power_mod_k << " " << power_mod_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-discrete_log") == 0) {
		f_discrete_log = true;
		discrete_log_y = ST.strtoi(argv[++i]);
		discrete_log_a = ST.strtoi(argv[++i]);
		discrete_log_m = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-discrete_log " << discrete_log_y << " "
					<< discrete_log_a << " " << discrete_log_m << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-sift_smooth") == 0) {
		f_sift_smooth = true;
		sift_smooth_from = ST.strtoi(argv[++i]);
		sift_smooth_len = ST.strtoi(argv[++i]);
		sift_smooth_factor_base = argv[++i];
		if (f_v) {
			cout << "-sift_smooth " << sift_smooth_from << " "
					<< sift_smooth_len << " " << sift_smooth_factor_base << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-square_root") == 0) {
		f_square_root = true;
		square_root_number.assign(argv[++i]);
		if (f_v) {
			cout << "-square_root " << square_root_number << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-square_root_mod") == 0) {
		f_square_root_mod = true;
		square_root_mod_a.assign(argv[++i]);
		square_root_mod_m.assign(argv[++i]);
		if (f_v) {
			cout << "-square_root_mod " << square_root_mod_a << " "
					<< square_root_mod_m << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		f_all_square_roots_mod_n = true;
		all_square_roots_mod_n_a.assign(argv[++i]);
		all_square_roots_mod_n_n.assign(argv[++i]);
		if (f_v) {
			cout << "-all_square_roots_mod_n " << all_square_roots_mod_n_a << " "
					<< all_square_roots_mod_n_n << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-quadratic_sieve") == 0) {
		f_quadratic_sieve = true;
		quadratic_sieve_n = ST.strtoi(argv[++i]);
		quadratic_sieve_factorbase = ST.strtoi(argv[++i]);
		quadratic_sieve_x0 = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-quadratic_sieve " << quadratic_sieve_n << " "
					<< quadratic_sieve_factorbase << " " << quadratic_sieve_x0 << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-jacobi") == 0) {
		f_jacobi = true;
		jacobi_top = ST.strtoi(argv[++i]);
		jacobi_bottom = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-jacobi " << jacobi_top << " "
					<< jacobi_bottom << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-solovay_strassen") == 0) {
		f_solovay_strassen = true;
		solovay_strassen_p = ST.strtoi(argv[++i]);
		solovay_strassen_a = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-solovay_strassen " << solovay_strassen_p << " "
					<< solovay_strassen_a << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin") == 0) {
		f_miller_rabin = true;
		miller_rabin_p = ST.strtoi(argv[++i]);
		miller_rabin_nb_times = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-miller_rabin " << miller_rabin_p << " " << miller_rabin_nb_times << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-fermat_test") == 0) {
		f_fermat_test = true;
		fermat_test_p = ST.strtoi(argv[++i]);
		fermat_test_nb_times = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-fermat_test " << fermat_test_p << " " << fermat_test_nb_times << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-find_pseudoprime") == 0) {
		f_find_pseudoprime = true;
		find_pseudoprime_nb_digits = ST.strtoi(argv[++i]);
		find_pseudoprime_nb_fermat = ST.strtoi(argv[++i]);
		find_pseudoprime_nb_miller_rabin = ST.strtoi(argv[++i]);
		find_pseudoprime_nb_solovay_strassen = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-find_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< " " << find_pseudoprime_nb_solovay_strassen << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-find_strong_pseudoprime") == 0) {
		f_find_strong_pseudoprime = true;
		find_pseudoprime_nb_digits = ST.strtoi(argv[++i]);
		find_pseudoprime_nb_fermat = ST.strtoi(argv[++i]);
		find_pseudoprime_nb_miller_rabin = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-find_strong_pseudoprime " << find_pseudoprime_nb_digits
					<< " " << find_pseudoprime_nb_fermat
					<< " " << find_pseudoprime_nb_miller_rabin
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-miller_rabin_text") == 0) {
		f_miller_rabin_text = true;
		miller_rabin_text_nb_times = ST.strtoi(argv[++i]);
		miller_rabin_number_text.assign(argv[++i]);
		if (f_v) {
			cout << "-miller_rabin " << miller_rabin_text_nb_times
					<< " " << miller_rabin_number_text
					<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-random") == 0) {
		f_random = true;
		random_nb = ST.strtoi(argv[++i]);
		random_fname_csv.assign(argv[++i]);
		if (f_v) {
			cout << "-random " << random_nb << " " << random_fname_csv << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-random_last") == 0) {
		f_random_last = true;
		random_last_nb = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-random_last " << random_last_nb << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-affine_sequence") == 0) {
		f_affine_sequence = true;
		affine_sequence_a = ST.strtoi(argv[++i]);
		affine_sequence_c = ST.strtoi(argv[++i]);
		affine_sequence_m = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-affine_sequence " << affine_sequence_a
					<< " " << affine_sequence_c << " " << affine_sequence_m << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		f_Chinese_remainders = true;
		Chinese_remainders_R.assign(argv[++i]);
		Chinese_remainders_M.assign(argv[++i]);
		if (f_v) {
			cout << "-Chinese_remainders " << Chinese_remainders_R
					<< " " << Chinese_remainders_M << endl;
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
	if (f_Chinese_remainders) {
		cout << "-Chinese_remainders " << Chinese_remainders_R
				<< " " << Chinese_remainders_M << endl;
	}
}

void interface_cryptography::worker(int verbose_level)
{
	if (f_cipher) {

		cryptography::cryptography_domain Crypto;

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

		cryptography::cryptography_domain Crypto;

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

		cryptography::cryptography_domain Crypto;

		Crypto.vigenere_analysis2(ctext, key_length);
	}
	else if (f_kasiski) {

		cryptography::cryptography_domain Crypto;

		int m;

		m = Crypto.kasiski_test(ctext, threshold);
		cout << "kasiski test for threshold " << threshold
				<< " yields key length " << m << endl;
	}
	else if (f_decipher) {

		cryptography::cryptography_domain Crypto;

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

		cryptography::cryptography_domain Crypto;


		Crypto.do_discrete_log(discrete_log_y, discrete_log_a, discrete_log_m, verbose_level);
	}
	else if (f_primitive_root) {

		cryptography::cryptography_domain Crypto;

		//longinteger_domain D;
		ring_theory::longinteger_object p;

		p.create_from_base_10_string(primitive_root_p);
		Crypto.do_primitive_root_longinteger(p, verbose_level);
	}
	else if (f_smallest_primitive_root) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_smallest_primitive_root(smallest_primitive_root_p, verbose_level);
	}
	else if (f_smallest_primitive_root_interval) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_smallest_primitive_root_interval(smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
	}
	else if (f_number_of_primitive_roots_interval) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_number_of_primitive_roots_interval(smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
	}
	else if (f_inverse_mod) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_inverse_mod(inverse_mod_a, inverse_mod_n, verbose_level);
	}
	else if (f_extended_gcd) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_extended_gcd(extended_gcd_a, extended_gcd_b, verbose_level);
	}
	else if (f_power_mod) {

		cryptography::cryptography_domain Crypto;

		ring_theory::longinteger_object a;
		ring_theory::longinteger_object k;
		ring_theory::longinteger_object n;

		a.create_from_base_10_string(power_mod_a);
		k.create_from_base_10_string(power_mod_k);
		n.create_from_base_10_string(power_mod_n);

		Crypto.do_power_mod(a, k, n, verbose_level);
	}
	else if (f_RSA) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_RSA(RSA_d, RSA_m, RSA_block_size, RSA_text, verbose_level);
	}
	else if (f_RSA_encrypt_text) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_RSA_encrypt_text(RSA_d, RSA_m, RSA_block_size,
				RSA_encrypt_text, verbose_level);
	}
	else if (f_RSA_setup) {
		cryptography::cryptography_domain Crypto;
		ring_theory::longinteger_object n, p, q, a, b;

		Crypto.RSA_setup(n, p, q, a, b,
			RSA_setup_nb_bits,
			RSA_setup_nb_tests_solovay_strassen,
			RSA_setup_f_miller_rabin_test,
			1 /*verbose_level */);
	}
	else if (f_sift_smooth) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_sift_smooth(sift_smooth_from,
				sift_smooth_len,
				sift_smooth_factor_base, verbose_level);
	}
	else if (f_square_root) {

		cryptography::cryptography_domain Crypto;

		Crypto.square_root(square_root_number, verbose_level);
	}
	else if (f_square_root_mod) {

		cryptography::cryptography_domain Crypto;

		Crypto.square_root_mod(square_root_mod_a, square_root_mod_m, verbose_level);
	}

	else if (f_all_square_roots_mod_n) {

		cryptography::cryptography_domain Crypto;
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

		cryptography::cryptography_domain Crypto;

		Crypto.quadratic_sieve(quadratic_sieve_n,
				quadratic_sieve_factorbase,
				quadratic_sieve_x0,
				verbose_level);
	}
	else if (f_jacobi) {

		number_theory::number_theory_domain NT;

		NT.do_jacobi(jacobi_top, jacobi_bottom, verbose_level);
	}
	else if (f_solovay_strassen) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_solovay_strassen(solovay_strassen_p, solovay_strassen_a, verbose_level);
	}
	else if (f_miller_rabin) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_miller_rabin(miller_rabin_p, miller_rabin_nb_times, verbose_level);
	}
	else if (f_fermat_test) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_fermat_test(fermat_test_p, fermat_test_nb_times, verbose_level);
	}
	else if (f_find_pseudoprime) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_find_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				find_pseudoprime_nb_solovay_strassen,
				verbose_level);
	}
	else if (f_find_strong_pseudoprime) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_find_strong_pseudoprime(
				find_pseudoprime_nb_digits,
				find_pseudoprime_nb_fermat,
				find_pseudoprime_nb_miller_rabin,
				verbose_level);
	}
	else if (f_miller_rabin_text) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_miller_rabin_text(
				miller_rabin_number_text, miller_rabin_text_nb_times,
				verbose_level);
	}
	else if (f_random) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_random(
				random_nb, random_fname_csv,
				verbose_level);
	}
	else if (f_random_last) {

		cryptography::cryptography_domain Crypto;

		Crypto.do_random_last(
				random_last_nb,
				verbose_level);
	}
	else if (f_affine_sequence) {

		cryptography::cryptography_domain Crypto;

		Crypto.make_affine_sequence(affine_sequence_a,
				affine_sequence_c, affine_sequence_m, verbose_level);
	}
	else if (f_Chinese_remainders) {

		long int *R;
		int sz1;
		long int *M;
		int sz2;

		Get_vector_or_set(Chinese_remainders_R, R, sz1);
		Get_vector_or_set(Chinese_remainders_M, M, sz2);

		number_theory::number_theory_domain NT;
		std::vector<long int> Remainders;
		std::vector<long int> Moduli;
		int i;
		long int x, Modulus;

		if (sz1 != sz2) {
			cout << "remainders and moduli must have the same length" << endl;
			exit(1);
		}

		for (i = 0; i < sz1; i++) {
			Remainders.push_back(R[i]);
			Moduli.push_back(M[i]);
		}

		x = NT.Chinese_Remainders(
				Remainders,
				Moduli, Modulus, verbose_level);


		cout << "The solution is " << x << " modulo " << Modulus << endl;

		ring_theory::longinteger_domain D;
		ring_theory::longinteger_object xl, Ml;

		D.Chinese_Remainders(
				Remainders,
				Moduli,
				xl, Ml, verbose_level);

		cout << "The solution is " << xl << " modulo " << Ml << " (computed in longinteger)" << endl;

		FREE_lint(R);
		FREE_lint(M);

	}




}




}}}


