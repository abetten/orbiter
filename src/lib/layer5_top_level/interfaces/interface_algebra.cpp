/*
 * interface_algebra.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace user_interface {


interface_algebra::interface_algebra()
{
	Record_birth();
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

	f_square_root = false;
	//square_root_number = NULL;

	f_square_root_mod = false;
	//square_root_mod_a = NULL;
	//square_root_mod_m = NULL;
	//cout << "interface_cryptography::interface_cryptography 1a" << endl;

	f_all_square_roots_mod_n = false;
	//std::string f_all_square_roots_mod_n_a;
	//std::string f_all_square_roots_mod_n_n;



	f_count_subprimitive = false;
	count_subprimitive_Q_max = 0;
	count_subprimitive_H_max = 0;


	f_character_table_symmetric_group = false;
	character_table_symmetric_group_n = 0;

	f_make_A5_in_PSL_2_q = false;
	make_A5_in_PSL_2_q_q = 0;

	f_order_of_q_mod_n = false;
	order_of_q_mod_n_q = 0;
	order_of_q_mod_n_n_min = 0;
	order_of_q_mod_n_n_max = 0;

	f_eulerfunction_interval = false;
	eulerfunction_interval_n_min = 0;
	eulerfunction_interval_n_max = 0;

	f_young_symmetrizer = false;
	young_symmetrizer_n = 0;

	f_young_symmetrizer_sym_4 = false;


	f_draw_mod_n = false;
	Draw_mod_n_description = NULL;

	f_power_function_mod_n = false;
	power_function_mod_n_k = 0;
	power_function_mod_n_n = 0;


	f_all_rational_normal_forms = false;
	//std::string all_rational_normal_forms_finite_field_label;
	all_rational_normal_forms_d = 0;

	f_eigenstuff = false;
	//std::string eigenstuff_finite_field_label;
	eigenstuff_n = 0;
	//eigenstuff_coeffs = NULL;
	//eigenstuff_fname = NULL;

	f_smith_normal_form = false;
	//std::string smith_normal_form_matrix

	f_jacobi = false;
	jacobi_top = 0;
	jacobi_bottom = 0;

	f_Chinese_remainders = false;
	//std::string Chinese_remainders_R;
	//std::string Chinese_remainders_M;


}

interface_algebra::~interface_algebra()
{
	Record_death();
}

void interface_algebra::print_help(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	other::data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
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
	else if (ST.stringcmp(argv[i], "-square_root") == 0) {
		cout << "-square_root <int : number>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-square_root_mod") == 0) {
		cout << "-square_root_mod <int : a> <int : m>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		cout << "-all_square_roots_mod_n <int : a> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		cout << "-count_subprimitive <int : Q_max> <int : H_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		cout << "-character_table_symmetric_group <int : deg> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		cout << "-make_A5_in_PSL_2_q <int : q> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		cout << "-order_of_q_mod_n <int : q> <int : n_min> <int : n_max>  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-eulerfunction_interval") == 0) {
		cout << "-eulerfunction_interval <int : n_min> <int : n_max>  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer") == 0) {
		cout << "-young_symmetrizer  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		cout << "-young_symmetrizer_sym_4  " << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_mod_n") == 0) {
		cout << "-draw_mod_n descr -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-power_function_mod_n") == 0) {
		cout << "-power_function_mod_n <int : a> <int : n>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		cout << "-all_rational_normal_forms <string : finite_field_label> <int : degree>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-eigenstuff") == 0) {
		cout << "-eigenstuff <string : finite_field_label> <int : n> <intvec : coeffs>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-smith_normal_form") == 0) {
		cout << "-smith_normal_form <string : matrix_label>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-jacobi") == 0) {
		cout << "-jacobi <int : top> <int : bottom>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		cout << "-Chinese_remainders <string : Remainders> <string : Moduli>" << endl;
	}
}

int interface_algebra::recognize_keyword(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_algebra::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
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
	else if (ST.stringcmp(argv[i], "-square_root") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-square_root_mod") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-all_square_roots_mod_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-eulerfunction_interval") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_mod_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-power_function_mod_n") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-eigenstuff") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-smith_normal_form") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-jacobi") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		return true;
	}
	if (f_v) {
		cout << "interface_algebra::recognize_keyword not recognizing" << endl;
	}
	return false;
}


void interface_algebra::read_arguments(
		int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_algebra::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_algebra::read_arguments the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-primitive_root") == 0) {
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
	else if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		f_count_subprimitive = true;
		count_subprimitive_Q_max = ST.strtoi(argv[++i]);
		count_subprimitive_H_max = ST.strtoi(argv[++i]);
		cout << "-count_subprimitive "
				<< count_subprimitive_Q_max
				<< " " << count_subprimitive_H_max
				<< endl;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		f_character_table_symmetric_group = true;
		character_table_symmetric_group_n = ST.strtoi(argv[++i]);
		cout << "-character_table_symmetric_group " << character_table_symmetric_group_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		f_make_A5_in_PSL_2_q = true;
		make_A5_in_PSL_2_q_q = ST.strtoi(argv[++i]);
		cout << "-make_A5_in_PSL_2_q " << make_A5_in_PSL_2_q_q << endl;
	}

	else if (ST.stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		f_order_of_q_mod_n = true;
		order_of_q_mod_n_q = ST.strtoi(argv[++i]);
		order_of_q_mod_n_n_min = ST.strtoi(argv[++i]);
		order_of_q_mod_n_n_max = ST.strtoi(argv[++i]);
		cout << "-order_of_q_mod_n " << order_of_q_mod_n_q
				<< " " << order_of_q_mod_n_n_min
				<< " " << order_of_q_mod_n_n_max << " " << endl;
	}

	else if (ST.stringcmp(argv[i], "-eulerfunction_interval") == 0) {
		f_eulerfunction_interval = true;
		eulerfunction_interval_n_min = ST.strtoi(argv[++i]);
		eulerfunction_interval_n_max = ST.strtoi(argv[++i]);
		cout << "-eulerfunction_interval "
				<< " " << eulerfunction_interval_n_min
				<< " " << eulerfunction_interval_n_max << " " << endl;
	}



	else if (ST.stringcmp(argv[i], "-young_symmetrizer") == 0) {
		f_young_symmetrizer = true;
		young_symmetrizer_n = ST.strtoi(argv[++i]);
		cout << "-young_symmetrizer " << " " << young_symmetrizer_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		f_young_symmetrizer_sym_4 = true;
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_mod_n") == 0) {
		f_draw_mod_n = true;
		cout << "-draw_mod_n " << endl;
		Draw_mod_n_description = NEW_OBJECT(other::graphics::draw_mod_n_description);
		i += Draw_mod_n_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		cout << "interface_algebra::read_arguments finished "
				"reading -draw_mod_n" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		cout << "-draw_mod_n " << endl;
		Draw_mod_n_description->print();
	}
	else if (ST.stringcmp(argv[i], "-power_function_mod_n") == 0) {
		f_power_function_mod_n = true;
		power_function_mod_n_k = ST.strtoi(argv[++i]);
		power_function_mod_n_n = ST.strtoi(argv[++i]);
		cout << "-power_mod_n " << " "
				<< power_function_mod_n_k << " "
				<< power_function_mod_n_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		f_all_rational_normal_forms = true;
		all_rational_normal_forms_finite_field_label.assign(argv[++i]);
		all_rational_normal_forms_d = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-f_all_rational_normal_forms "
				<< all_rational_normal_forms_finite_field_label
				<< " " << all_rational_normal_forms_d << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-eigenstuff") == 0) {
		f_eigenstuff = true;
		eigenstuff_finite_field_label.assign(argv[++i]);
		eigenstuff_n = ST.strtoi(argv[++i]);
		eigenstuff_coeffs.assign(argv[++i]);
		if (f_v) {
			cout << "-eigenstuff "
				<< eigenstuff_finite_field_label
				<< " " << eigenstuff_n
				<< " " << eigenstuff_coeffs << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-smith_normal_form") == 0) {
		f_smith_normal_form = true;
		smith_normal_form_matrix.assign(argv[++i]);
		if (f_v) {
			cout << "-smith_normal_form "
				<< smith_normal_form_matrix << endl;
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
	else if (ST.stringcmp(argv[i], "-Chinese_remainders") == 0) {
		f_Chinese_remainders = true;
		Chinese_remainders_R.assign(argv[++i]);
		Chinese_remainders_M.assign(argv[++i]);
		if (f_v) {
			cout << "-Chinese_remainders " << Chinese_remainders_R
					<< " " << Chinese_remainders_M << endl;
		}
	}

}

void interface_algebra::print()
{
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
	if (f_count_subprimitive) {
		cout << "-count_subprimitive "
				<< count_subprimitive_Q_max
				<< " " << count_subprimitive_H_max
				<< endl;
	}
	if (f_character_table_symmetric_group) {
		cout << "-character_table_symmetric_group " << character_table_symmetric_group_n << endl;
	}
	if (f_make_A5_in_PSL_2_q) {
		cout << "-make_A5_in_PSL_2_q " << make_A5_in_PSL_2_q_q << endl;
	}
	if (f_order_of_q_mod_n) {
		cout << "-order_of_q_mod_n " << order_of_q_mod_n_q
				<< " " << order_of_q_mod_n_n_min
				<< " " << order_of_q_mod_n_n_max << " " << endl;
	}

	if (f_eulerfunction_interval) {
		cout << "-eulerfunction_interval "
				<< " " << eulerfunction_interval_n_min
				<< " " << eulerfunction_interval_n_max << " " << endl;
	}

	if (f_young_symmetrizer) {
		cout << "-young_symmetrizer " << " " << young_symmetrizer_n << endl;
	}
	if (f_young_symmetrizer_sym_4) {
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	if (f_draw_mod_n) {
		cout << "-draw_mod_n " << endl;
		Draw_mod_n_description->print();
	}
	if (f_power_function_mod_n) {
		cout << "-power_function_mod_n " << " " << power_function_mod_n_k << " " << power_function_mod_n_n << endl;
	}

	if (f_all_rational_normal_forms) {
		cout << "-all_rational_normal_forms "
				<< all_rational_normal_forms_finite_field_label
				<< " " << all_rational_normal_forms_d << endl;
	}
	if (f_eigenstuff) {
		cout << "-eigenstuff "
			<< eigenstuff_finite_field_label << " "
			<< eigenstuff_n << " "
			<< eigenstuff_coeffs << endl;
	}
	if (f_smith_normal_form) {
		cout << "-smith_normal_form "
			<< smith_normal_form_matrix << endl;
	}
	if (f_jacobi) {
		cout << "-jacobi " << jacobi_top << " "
				<< jacobi_bottom << endl;
	}
	if (f_Chinese_remainders) {
		cout << "-Chinese_remainders " << Chinese_remainders_R
				<< " " << Chinese_remainders_M << endl;
	}

}


void interface_algebra::worker(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::worker" << endl;
	}

	if (f_discrete_log) {

		algebra::number_theory::number_theory_domain NT;


		NT.do_discrete_log(
				discrete_log_y, discrete_log_a, discrete_log_m,
				verbose_level);
	}
	else if (f_primitive_root) {

		algebra::number_theory::number_theory_domain NT;

		//longinteger_domain D;
		algebra::ring_theory::longinteger_object p;

		p.create_from_base_10_string(primitive_root_p);
		NT.do_primitive_root_longinteger(p, verbose_level);
	}
	else if (f_smallest_primitive_root) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_smallest_primitive_root(smallest_primitive_root_p, verbose_level);
	}
	else if (f_smallest_primitive_root_interval) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_smallest_primitive_root_interval(
				smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
	}
	else if (f_number_of_primitive_roots_interval) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_number_of_primitive_roots_interval(
				smallest_primitive_root_interval_min,
				smallest_primitive_root_interval_max, verbose_level);
	}
	else if (f_inverse_mod) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_inverse_mod(inverse_mod_a, inverse_mod_n, verbose_level);
	}
	else if (f_extended_gcd) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_extended_gcd(extended_gcd_a, extended_gcd_b, verbose_level);
	}
	else if (f_power_mod) {

		algebra::number_theory::number_theory_domain NT;

		algebra::ring_theory::longinteger_object a;
		algebra::ring_theory::longinteger_object k;
		algebra::ring_theory::longinteger_object n;

		a.create_from_base_10_string(power_mod_a);
		k.create_from_base_10_string(power_mod_k);
		n.create_from_base_10_string(power_mod_n);

		NT.do_power_mod(a, k, n, verbose_level);
	}
	else if (f_square_root) {

		algebra::number_theory::number_theory_domain NT;

		NT.square_root(square_root_number, verbose_level);
	}
	else if (f_square_root_mod) {

		algebra::number_theory::number_theory_domain NT;

		NT.square_root_mod(square_root_mod_a, square_root_mod_m, verbose_level);
	}

	else if (f_all_square_roots_mod_n) {

		algebra::number_theory::number_theory_domain NT;
		vector<long int> S;
		int i;

		NT.all_square_roots_mod_n_by_exhaustive_search_lint(
				all_square_roots_mod_n_a, all_square_roots_mod_n_n, S, verbose_level);

		cout << "We found " << S.size() << " square roots of "
				<< all_square_roots_mod_n_a << " mod " << all_square_roots_mod_n_n << endl;
		cout << "They are:" << endl;
		for (i = 0; i < S.size(); i++) {
			cout << i << " : " << S[i] << endl;
		}
	}
	else if (f_character_table_symmetric_group) {

		apps_algebra::algebra_global_with_action A;

		A.do_character_table_symmetric_group(
				character_table_symmetric_group_n,
				verbose_level);

	}

	else if (f_make_A5_in_PSL_2_q) {

		group_constructions::group_constructions_global Group_constructions_global;

		Group_constructions_global.A5_in_PSL_(
				make_A5_in_PSL_2_q_q, verbose_level);

	}

	else if (f_count_subprimitive) {

		algebra::basic_algebra::algebra_global Algebra;

		Algebra.count_subprimitive(
				count_subprimitive_Q_max,
				count_subprimitive_H_max);
	}
#if 0
	else if (f_search_for_primitive_polynomial_in_range) {

		ring_theory::ring_theory_global R;

		R.do_search_for_primitive_polynomial_in_range(
				p_min, p_max, deg_min, deg_max,
				verbose_level);
	}
#endif

	else if (f_order_of_q_mod_n) {

		algebra::basic_algebra::algebra_global Algebra;

		Algebra.order_of_q_mod_n(
				order_of_q_mod_n_q, order_of_q_mod_n_n_min, order_of_q_mod_n_n_max,
				verbose_level);

	}

	else if (f_eulerfunction_interval) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_eulerfunction_interval(
				eulerfunction_interval_n_min, eulerfunction_interval_n_max,
				verbose_level);

	}


	else if (f_young_symmetrizer) {
		apps_algebra::algebra_global_with_action Algebra;

		Algebra.young_symmetrizer(young_symmetrizer_n, verbose_level);
	}

	else if (f_young_symmetrizer_sym_4) {
		apps_algebra::algebra_global_with_action Algebra;

		Algebra.young_symmetrizer_sym_4(verbose_level);
	}

	else if (f_draw_mod_n) {
		other::graphics::plot_tools PT;


		PT.draw_mod_n(
				Draw_mod_n_description,
				verbose_level);
	}

	else if (f_power_function_mod_n) {

		algebra::basic_algebra::algebra_global Algebra;

		Algebra.power_function_mod_n(
				power_function_mod_n_k, power_function_mod_n_n,
				verbose_level);

	}



	else if (f_all_rational_normal_forms) {

		apps_algebra::algebra_global_with_action Algebra;

		algebra::field_theory::finite_field *F;

		F = Get_finite_field(all_rational_normal_forms_finite_field_label);

		Algebra.classes_GL(
				F, all_rational_normal_forms_d,
				false /* f_no_eigenvalue_one */, verbose_level);



	}

	else if (f_eigenstuff) {


		apps_algebra::algebra_global_with_action Algebra;
		int *data;
		int sz;
		algebra::field_theory::finite_field *F;


		F = Get_finite_field(eigenstuff_finite_field_label);

		Int_vec_scan(eigenstuff_coeffs, data, sz);

		if (sz != eigenstuff_n * eigenstuff_n) {
			cout << "sz != eigenstuff_n * eigenstuff_n" << endl;
			exit(1);
		}

		Algebra.do_eigenstuff(F, eigenstuff_n, data, verbose_level);

	}

	else if (f_smith_normal_form) {


		if (f_v) {
			cout << "interface_algebra::worker f_smith_normal_form" << endl;
		}

		algebra::basic_algebra::algebra_global Algebra;
		int *A;
		int m, n;

		Get_matrix(smith_normal_form_matrix, A, m, n);


		Algebra.smith_normal_form(
					A, m, n, smith_normal_form_matrix, verbose_level);


#if 0
		typed_objects::discreta_matrix M;
		typed_objects::discreta_matrix P, Pv, Q, Qv;
		int i, j, a;
		number_theory::number_theory_domain NT;

		M.m_mn(m, n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				a = A[i * n + j];
				M.m_iji(i, j, a);
			}
		}

		if (f_v) {
			cout << "M=" << endl;
			cout << M << endl;
		}

		M.smith_normal_form(
				P, Pv,
				Q, Qv,
				verbose_level);
		if (f_v) {
			cout << "interface_algebra::worker M = " << endl;
			cout << M << endl;
			cout << "interface_algebra::worker P = " << endl;
			cout << P << endl;
			cout << "interface_algebra::worker Pv = " << endl;
			cout << Pv << endl;
			cout << "interface_algebra::worker Q = " << endl;
			cout << Q << endl;
			cout << "interface_algebra::worker Qv = " << endl;
			cout << Qv << endl;
		}
#endif

	}
	else if (f_jacobi) {

		algebra::number_theory::number_theory_domain NT;

		NT.do_jacobi(jacobi_top, jacobi_bottom, verbose_level);
	}
	else if (f_Chinese_remainders) {

		long int *R;
		int sz1;
		long int *M;
		int sz2;

		Get_vector_or_set(Chinese_remainders_R, R, sz1);
		Get_vector_or_set(Chinese_remainders_M, M, sz2);

		algebra::number_theory::number_theory_domain NT;
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

		algebra::ring_theory::longinteger_domain D;
		algebra::ring_theory::longinteger_object xl, Ml;

		D.Chinese_Remainders(
				Remainders,
				Moduli,
				xl, Ml, verbose_level);

		cout << "The solution is " << xl << " modulo " << Ml << " (computed in longinteger)" << endl;

		FREE_lint(R);
		FREE_lint(M);

	}



	if (f_v) {
		cout << "interface_algebra::worker done" << endl;
	}

}







}}}


