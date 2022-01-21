/*
 * interface_algebra.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


interface_algebra::interface_algebra()
{
	f_count_subprimitive = FALSE;
	count_subprimitive_Q_max = 0;
	count_subprimitive_H_max = 0;
	f_equivalence_class_of_fractions = FALSE;
	equivalence_class_of_fractions_N = 0;


	f_character_table_symmetric_group = FALSE;
	character_table_symmetric_group_n = 0;

	f_make_A5_in_PSL_2_q = FALSE;
	make_A5_in_PSL_2_q_q = 0;


	f_search_for_primitive_polynomial_in_range = FALSE;
	p_min = 0;
	p_max = 0;
	deg_min = 0;
	deg_max = 0;

	f_order_of_q_mod_n = FALSE;
	order_of_q_mod_n_q = 0;
	order_of_q_mod_n_n_min = 0;
	order_of_q_mod_n_n_max = 0;

	f_eulerfunction_interval = FALSE;
	eulerfunction_interval_n_min = 0;
	eulerfunction_interval_n_max = 0;

	f_young_symmetrizer = FALSE;
	young_symmetrizer_n = 0;

	f_young_symmetrizer_sym_4 = FALSE;


	f_draw_mod_n = FALSE;
	Draw_mod_n_description = NULL;

	f_power_function_mod_n = FALSE;
	power_function_mod_n_k = 0;
	power_function_mod_n_n = 0;


	f_all_rational_normal_forms = FALSE;
	//std::string all_rational_normal_forms_finite_field_label;
	all_rational_normal_forms_d = 0;

	f_eigenstuff = FALSE;
	f_eigenstuff_from_file = FALSE;
	//std::string eigenstuff_finite_field_label;
	eigenstuff_n = 0;
	//eigenstuff_coeffs = NULL;
	//eigenstuff_fname = NULL;


}


void interface_algebra::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		cout << "-count_subprimitive <int : Q_max> <int : H_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		cout << "-equivalence_class_of_fractions <int : N> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		cout << "-character_table_symmetric_group <int : deg> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		cout << "-make_A5_in_PSL_2_q <int : q> " << endl;
	}
	else if (ST.stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		cout << "-search_for_primitive_polynomial_in_range  " << endl;
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
	else if (ST.stringcmp(argv[i], "-eigenstuff_from_file") == 0) {
		cout << "-eigenstuff_from_file <string : finite_field_label>  <int : n> <string : fname>" << endl;
	}
}


int interface_algebra::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_algebra::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
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
	else if (ST.stringcmp(argv[i], "-eigenstuff_from_file") == 0) {
		return true;
	}
	if (f_v) {
		cout << "interface_algebra::recognize_keyword not recognizing" << endl;
	}
	return false;
}


void interface_algebra::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_algebra::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_algebra::read_arguments the next argument is " << argv[i] << endl;
	}

	if (ST.stringcmp(argv[i], "-count_subprimitive") == 0) {
		f_count_subprimitive = TRUE;
		count_subprimitive_Q_max = ST.strtoi(argv[++i]);
		count_subprimitive_H_max = ST.strtoi(argv[++i]);
		cout << "-count_subprimitive "
				<< count_subprimitive_Q_max
				<< " " << count_subprimitive_H_max
				<< endl;
	}
	else if (ST.stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		f_equivalence_class_of_fractions = TRUE;
		equivalence_class_of_fractions_N = ST.strtoi(argv[++i]);
		cout << "-equivalence_class_of_fractions " << equivalence_class_of_fractions_N
				<< endl;
	}
	else if (ST.stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		f_character_table_symmetric_group = TRUE;
		character_table_symmetric_group_n = ST.strtoi(argv[++i]);
		cout << "-character_table_symmetric_group " << character_table_symmetric_group_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		f_make_A5_in_PSL_2_q = TRUE;
		make_A5_in_PSL_2_q_q = ST.strtoi(argv[++i]);
		cout << "-make_A5_in_PSL_2_q " << make_A5_in_PSL_2_q_q << endl;
	}
	else if (ST.stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		f_search_for_primitive_polynomial_in_range = TRUE;
		p_min = ST.strtoi(argv[++i]);
		p_max = ST.strtoi(argv[++i]);
		deg_min = ST.strtoi(argv[++i]);
		deg_max = ST.strtoi(argv[++i]);
		cout << "-search_for_primitive_polynomial_in_range " << p_min
				<< " " << p_max
				<< " " << deg_min
				<< " " << deg_max << " " << endl;
	}

	else if (ST.stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		f_order_of_q_mod_n = TRUE;
		order_of_q_mod_n_q = ST.strtoi(argv[++i]);
		order_of_q_mod_n_n_min = ST.strtoi(argv[++i]);
		order_of_q_mod_n_n_max = ST.strtoi(argv[++i]);
		cout << "-order_of_q_mod_n " << order_of_q_mod_n_q
				<< " " << order_of_q_mod_n_n_min
				<< " " << order_of_q_mod_n_n_max << " " << endl;
	}

	else if (ST.stringcmp(argv[i], "-eulerfunction_interval") == 0) {
		f_eulerfunction_interval = TRUE;
		eulerfunction_interval_n_min = ST.strtoi(argv[++i]);
		eulerfunction_interval_n_max = ST.strtoi(argv[++i]);
		cout << "-eulerfunction_interval "
				<< " " << eulerfunction_interval_n_min
				<< " " << eulerfunction_interval_n_max << " " << endl;
	}



	else if (ST.stringcmp(argv[i], "-young_symmetrizer") == 0) {
		f_young_symmetrizer = TRUE;
		young_symmetrizer_n = ST.strtoi(argv[++i]);
		cout << "-young_symmetrizer " << " " << young_symmetrizer_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		f_young_symmetrizer_sym_4 = TRUE;
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_mod_n") == 0) {
		f_draw_mod_n = TRUE;
		cout << "-draw_mod_n " << endl;
		Draw_mod_n_description = NEW_OBJECT(draw_mod_n_description);
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
	}
	else if (ST.stringcmp(argv[i], "-power_function_mod_n") == 0) {
		f_power_function_mod_n = TRUE;
		power_function_mod_n_k = ST.strtoi(argv[++i]);
		power_function_mod_n_n = ST.strtoi(argv[++i]);
		cout << "-power_mod_n " << " "
				<< power_function_mod_n_k << " "
				<< power_function_mod_n_n << endl;
	}
	else if (ST.stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		f_all_rational_normal_forms = TRUE;
		all_rational_normal_forms_finite_field_label.assign(argv[++i]);
		all_rational_normal_forms_d = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-f_all_rational_normal_forms "
				<< all_rational_normal_forms_finite_field_label
				<< " " << all_rational_normal_forms_d << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-eigenstuff") == 0) {
		f_eigenstuff = TRUE;
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
	else if (ST.stringcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
		f_eigenstuff_from_file = TRUE;
		eigenstuff_finite_field_label.assign(argv[++i]);
		eigenstuff_n = ST.strtoi(argv[++i]);
		eigenstuff_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-eigenstuff_from_file "
					<< eigenstuff_finite_field_label
					<< " " << eigenstuff_n
					<< " " << eigenstuff_fname << endl;
		}
	}

}


void interface_algebra::print()
{
	if (f_count_subprimitive) {
		cout << "-count_subprimitive "
				<< count_subprimitive_Q_max
				<< " " << count_subprimitive_H_max
				<< endl;
	}
	if (f_equivalence_class_of_fractions) {
		cout << "-equivalence_class_of_fractions "
				<< equivalence_class_of_fractions_N
				<< endl;
	}
	if (f_character_table_symmetric_group) {
		cout << "-character_table_symmetric_group " << character_table_symmetric_group_n << endl;
	}
	if (f_make_A5_in_PSL_2_q) {
		cout << "-make_A5_in_PSL_2_q " << make_A5_in_PSL_2_q_q << endl;
	}
	if (f_search_for_primitive_polynomial_in_range) {
		cout << "-search_for_primitive_polynomial_in_range " << p_min
				<< " " << p_max
				<< " " << deg_min
				<< " " << deg_max << " " << endl;
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
	if (f_eigenstuff_from_file) {
		cout << "-eigenstuff_from_file "
				<< eigenstuff_finite_field_label << " " << eigenstuff_n
			<< " " << eigenstuff_fname << endl;
	}

}


void interface_algebra::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::worker" << endl;
	}

	if (f_character_table_symmetric_group) {
		do_character_table_symmetric_group(character_table_symmetric_group_n, verbose_level);
	}

	else if (f_make_A5_in_PSL_2_q) {

		apps_algebra::algebra_global_with_action A;

		A.A5_in_PSL_(make_A5_in_PSL_2_q_q, verbose_level);

	}

	else if (f_count_subprimitive) {

		algebra::algebra_global Algebra;

		Algebra.count_subprimitive(count_subprimitive_Q_max,
				count_subprimitive_H_max);
	}

	else if (f_equivalence_class_of_fractions) {

		algebra::algebra_global Algebra;

		Algebra.do_equivalence_class_of_fractions(equivalence_class_of_fractions_N,
				verbose_level);
	}

	else if (f_search_for_primitive_polynomial_in_range) {

		ring_theory::ring_theory_global R;

		R.do_search_for_primitive_polynomial_in_range(
				p_min, p_max, deg_min, deg_max,
				verbose_level);
	}

	else if (f_order_of_q_mod_n) {

		algebra::algebra_global Algebra;

		Algebra.order_of_q_mod_n(
				order_of_q_mod_n_q, order_of_q_mod_n_n_min, order_of_q_mod_n_n_max,
				verbose_level);

	}

	else if (f_eulerfunction_interval) {

		number_theory::number_theory_domain NT;

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
		plot_tools PT;
		layered_graph_draw_options *O;


		if (!Orbiter->f_draw_options) {
			cout << "please use option -draw_options .. -end" << endl;
			exit(1);
		}
		O = Orbiter->draw_options;
		PT.draw_mod_n(Draw_mod_n_description,
				O,
				verbose_level);
	}

	else if (f_power_function_mod_n) {

		algebra::algebra_global Algebra;

		Algebra.power_function_mod_n(
				power_function_mod_n_k, power_function_mod_n_n,
				verbose_level);

	}

	else if (f_all_rational_normal_forms) {

		apps_algebra::algebra_global_with_action Algebra;

		field_theory::finite_field *F;
		int idx;


		idx = The_Orbiter_top_level_session->find_symbol(all_rational_normal_forms_finite_field_label);

		F = (field_theory::finite_field *) The_Orbiter_top_level_session->get_object(idx);

		Algebra.classes_GL(F, all_rational_normal_forms_d,
				FALSE /* f_no_eigenvalue_one */, verbose_level);



	}

	else if (f_eigenstuff) {


		apps_algebra::algebra_global_with_action Algebra;
		int *data;
		int sz;
		field_theory::finite_field *F;
		int idx;


		idx = The_Orbiter_top_level_session->find_symbol(eigenstuff_finite_field_label);

		F = (field_theory::finite_field *) The_Orbiter_top_level_session->get_object(idx);

		Orbiter->Int_vec->scan(eigenstuff_coeffs, data, sz);

		if (sz != eigenstuff_n * eigenstuff_n) {
			cout << "sz != eigenstuff_n * eigenstuff_n" << endl;
			exit(1);
		}

		Algebra.do_eigenstuff(F, eigenstuff_n, data, verbose_level);

	}

	else if (f_eigenstuff_from_file) {


#if 0
		eigenstuff_n = strtoi(argv[++i]);
		eigenstuff_fname.assign(argv[++i]);
		algebra_global Algebra;

		Algebra.power_mod_n(
				power_mod_n_a, power_mod_n_n,
				verbose_level);
#endif

	}



	if (f_v) {
		cout << "interface_algebra::worker done" << endl;
	}

}



void interface_algebra::do_character_table_symmetric_group(int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_character_table_symmetric_group" << endl;
		cout << "deg=" << deg << endl;
	}

	apps_algebra::character_table_burnside *CTB;

	CTB = NEW_OBJECT(apps_algebra::character_table_burnside);

	CTB->do_it(deg, verbose_level);

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "interface_algebra::do_character_table_symmetric_group done" << endl;
	}
}







}}

