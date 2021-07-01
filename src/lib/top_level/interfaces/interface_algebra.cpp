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
	deg = 0;

	f_make_A5_in_PSL_2_q = FALSE;
	q = 0;


	f_search_for_primitive_polynomial_in_range = FALSE;
	p_min = 0;
	p_max = 0;
	deg_min = 0;
	deg_max = 0;

	f_order_of_q_mod_n = FALSE;
	order_of_q_mod_n_q = 0;
	order_of_q_mod_n_n_min = 0;
	order_of_q_mod_n_n_max = 0;

	f_young_symmetrizer = FALSE;
	young_symmetrizer_n = 0;

	f_young_symmetrizer_sym_4 = FALSE;


	f_draw_mod_n = FALSE;
	Draw_mod_n_description = NULL;

	f_power_mod_n = FALSE;
	power_mod_n_a = 0;
	power_mod_n_n = 0;


	f_all_rational_normal_forms = FALSE;
	//std::string all_rational_normal_forms_finite_field_label;
	all_rational_normal_forms_d = 0;

	f_eigenstuff = FALSE;
	f_eigenstuff_from_file = FALSE;
	//std::string eigenstuff_finite_field_label;
	eigenstuff_n = 0;
	//eigenstuff_coeffs = NULL;
	//eigenstuff_fname = NULL;

#if 0
	//
	f_study_surface = FALSE;
	study_surface_nb = 0;

	f_decomposition_by_element = FALSE;
	decomposition_by_element_n = 0;
	decomposition_by_element_power = 1;
	//std::string decomposition_by_element_data
	//decomposition_by_element_fname_base
#endif


}


void interface_algebra::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-count_subprimitive") == 0) {
		cout << "-count_subprimitive <int : Q_max> <int : H_max>" << endl;
	}
	else if (stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		cout << "-equivalence_class_of_fractions <int : N> " << endl;
	}
	else if (stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		cout << "-character_table_symmetric_group <int : deg> " << endl;
	}
	else if (stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		cout << "-make_A5_in_PSL_2_q <int : q> " << endl;
	}
	else if (stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		cout << "-search_for_primitive_polynomial_in_range  " << endl;
	}
	else if (stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		cout << "-order_of_q_mod_n <int : q> <int : n_min> <int : n_max>  " << endl;
	}
	else if (stringcmp(argv[i], "-young_symmetrizer") == 0) {
		cout << "-young_symmetrizer  " << endl;
	}
	else if (stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		cout << "-young_symmetrizer_sym_4  " << endl;
	}
	else if (stringcmp(argv[i], "-draw_mod_n") == 0) {
		cout << "-draw_mod_n descr -end" << endl;
	}
	else if (stringcmp(argv[i], "-power_mod_n") == 0) {
		cout << "-power_mod_n <int : a> <int : n>" << endl;
	}
	else if (stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		cout << "-all_rational_normal_forms <string : finite_field_label> <int : degree>" << endl;
	}
	else if (stringcmp(argv[i], "-eigenstuff") == 0) {
		cout << "-eigenstuff <string : finite_field_label> <int : n> <intvec : coeffs>" << endl;
	}
	else if (stringcmp(argv[i], "-eigenstuff_from_file") == 0) {
		cout << "-eigenstuff_from_file <string : finite_field_label>  <int : n> <string : fname>" << endl;
	}
}


int interface_algebra::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::recognize_keyword" << endl;
	}
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-count_subprimitive") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-young_symmetrizer") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-draw_mod_n") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-power_mod_n") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-eigenstuff") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-eigenstuff_from_file") == 0) {
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

	if (f_v) {
		cout << "interface_algebra::read_arguments" << endl;
	}


	if (f_v) {
		cout << "interface_algebra::read_arguments the next argument is " << argv[i] << endl;
	}

	if (stringcmp(argv[i], "-count_subprimitive") == 0) {
		f_count_subprimitive = TRUE;
		count_subprimitive_Q_max = strtoi(argv[++i]);
		count_subprimitive_H_max = strtoi(argv[++i]);
		cout << "-count_subprimitive "
				<< count_subprimitive_Q_max
				<< " " << count_subprimitive_H_max
				<< endl;
	}
	else if (stringcmp(argv[i], "-equivalence_class_of_fractions") == 0) {
		f_equivalence_class_of_fractions = TRUE;
		equivalence_class_of_fractions_N = strtoi(argv[++i]);
		cout << "-equivalence_class_of_fractions " << equivalence_class_of_fractions_N
				<< endl;
	}
	else if (stringcmp(argv[i], "-character_table_symmetric_group") == 0) {
		f_character_table_symmetric_group = TRUE;
		deg = strtoi(argv[++i]);
		cout << "-character_table_symmetric_group " << deg << endl;
	}
	else if (stringcmp(argv[i], "-make_A5_in_PSL_2_q") == 0) {
		f_make_A5_in_PSL_2_q = TRUE;
		q = strtoi(argv[++i]);
		cout << "-make_A5_in_PSL_2_q " << q << endl;
	}
	else if (stringcmp(argv[i], "-search_for_primitive_polynomial_in_range") == 0) {
		f_search_for_primitive_polynomial_in_range = TRUE;
		p_min = strtoi(argv[++i]);
		p_max = strtoi(argv[++i]);
		deg_min = strtoi(argv[++i]);
		deg_max = strtoi(argv[++i]);
		cout << "-search_for_primitive_polynomial_in_range " << p_min
				<< " " << p_max
				<< " " << deg_min
				<< " " << deg_max << " " << endl;
	}

	else if (stringcmp(argv[i], "-order_of_q_mod_n") == 0) {
		f_order_of_q_mod_n = TRUE;
		order_of_q_mod_n_q = strtoi(argv[++i]);
		order_of_q_mod_n_n_min = strtoi(argv[++i]);
		order_of_q_mod_n_n_max = strtoi(argv[++i]);
		cout << "-order_of_q_mod_n " << order_of_q_mod_n_q
				<< " " << order_of_q_mod_n_n_min
				<< " " << order_of_q_mod_n_n_max << " " << endl;
	}



	else if (stringcmp(argv[i], "-young_symmetrizer") == 0) {
		f_young_symmetrizer = TRUE;
		young_symmetrizer_n = strtoi(argv[++i]);
		cout << "-young_symmetrizer " << " " << young_symmetrizer_n << endl;
	}
	else if (stringcmp(argv[i], "-young_symmetrizer_sym_4") == 0) {
		f_young_symmetrizer_sym_4 = TRUE;
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	else if (stringcmp(argv[i], "-draw_mod_n") == 0) {
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
	else if (stringcmp(argv[i], "-power_mod_n") == 0) {
		f_power_mod_n = TRUE;
		power_mod_n_a = strtoi(argv[++i]);
		power_mod_n_n = strtoi(argv[++i]);
		cout << "-power_mod_n " << " " << power_mod_n_a << " " << power_mod_n_n << endl;
	}
	else if (stringcmp(argv[i], "-all_rational_normal_forms") == 0) {
		f_all_rational_normal_forms = TRUE;
		all_rational_normal_forms_finite_field_label.assign(argv[++i]);
		all_rational_normal_forms_d = strtoi(argv[++i]);
		if (f_v) {
			cout << "-f_all_rational_normal_forms " << all_rational_normal_forms_finite_field_label << " " << all_rational_normal_forms_d << endl;
		}
	}
	else if (stringcmp(argv[i], "-eigenstuff") == 0) {
		f_eigenstuff = TRUE;
		eigenstuff_finite_field_label.assign(argv[++i]);
		eigenstuff_n = strtoi(argv[++i]);
		eigenstuff_coeffs.assign(argv[++i]);
		if (f_v) {
			cout << "-eigenstuff " << eigenstuff_finite_field_label << " " << eigenstuff_n
				<< " " << eigenstuff_coeffs << endl;
		}
	}
	else if (stringcmp(argv[i], "-eigenstuff_matrix_from_file") == 0) {
		f_eigenstuff_from_file = TRUE;
		eigenstuff_finite_field_label.assign(argv[++i]);
		eigenstuff_n = strtoi(argv[++i]);
		eigenstuff_fname.assign(argv[++i]);
		if (f_v) {
			cout << "-eigenstuff_from_file " << eigenstuff_finite_field_label << " " << eigenstuff_n
				<< " " << eigenstuff_fname << endl;
		}
	}
#if 0
	else if (stringcmp(argv[i], "-study_surface") == 0) {
		f_study_surface = TRUE;
		study_surface_nb = strtoi(argv[++i]);
		if (f_v) {
			cout << "-study_surface" << study_surface_nb << endl;
		}
	}
	else if (stringcmp(argv[i], "-decomposition_by_element") == 0) {
		f_decomposition_by_element = TRUE;
		decomposition_by_element_n = strtoi(argv[++i]);
		decomposition_by_element_power = strtoi(argv[++i]);
		decomposition_by_element_data.assign(argv[++i]);
		decomposition_by_element_fname_base.assign(argv[++i]);
		if (f_v) {
			cout << "-decomposition_by_element " <<  decomposition_by_element_power
				<< " " << decomposition_by_element_data
				<< " " << decomposition_by_element_fname_base << endl;
		}
	}
#endif


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
		cout << "-equivalence_class_of_fractions " << equivalence_class_of_fractions_N
				<< endl;
	}
	if (f_character_table_symmetric_group) {
		cout << "-character_table_symmetric_group " << deg << endl;
	}
	if (f_make_A5_in_PSL_2_q) {
		cout << "-make_A5_in_PSL_2_q " << q << endl;
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



	if (f_young_symmetrizer) {
		cout << "-young_symmetrizer " << " " << young_symmetrizer_n << endl;
	}
	if (f_young_symmetrizer_sym_4) {
		cout << "-young_symmetrizer_sym_4 " << endl;
	}
	if (f_draw_mod_n) {
		cout << "-draw_mod_n " << endl;
	}
	if (f_power_mod_n) {
		cout << "-power_mod_n " << " " << power_mod_n_a << " " << power_mod_n_n << endl;
	}

	if (f_all_rational_normal_forms) {
		cout << "-all_rational_normal_forms " << all_rational_normal_forms_finite_field_label
				<< " " << all_rational_normal_forms_d << endl;
	}
	if (f_eigenstuff) {
		cout << "-eigenstuff "
			<< eigenstuff_finite_field_label << " " << eigenstuff_n << " " << eigenstuff_coeffs << endl;
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
		do_character_table_symmetric_group(deg, verbose_level);
	}

	else if (f_make_A5_in_PSL_2_q) {
		do_make_A5_in_PSL_2_q(q, verbose_level);
	}

	else if (f_count_subprimitive) {

		algebra_global Algebra;

		Algebra.count_subprimitive(count_subprimitive_Q_max,
				count_subprimitive_H_max);
	}

	else if (f_equivalence_class_of_fractions) {

		algebra_global Algebra;

		Algebra.do_equivalence_class_of_fractions(equivalence_class_of_fractions_N, verbose_level);
	}

	else if (f_search_for_primitive_polynomial_in_range) {

		algebra_global Algebra;

		Algebra.do_search_for_primitive_polynomial_in_range(
				p_min, p_max, deg_min, deg_max,
				verbose_level);
	}

	else if (f_order_of_q_mod_n) {

		algebra_global Algebra;

		Algebra.order_of_q_mod_n(
				order_of_q_mod_n_q, order_of_q_mod_n_n_min, order_of_q_mod_n_n_max,
				verbose_level);

	}

	else if (f_young_symmetrizer) {
		algebra_global_with_action Algebra;

		Algebra.young_symmetrizer(young_symmetrizer_n, verbose_level);
	}

	else if (f_young_symmetrizer_sym_4) {
		algebra_global_with_action Algebra;

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

	else if (f_power_mod_n) {

		algebra_global Algebra;

		Algebra.power_mod_n(
				power_mod_n_a, power_mod_n_n,
				verbose_level);

	}

	else if (f_all_rational_normal_forms) {

		algebra_global_with_action Algebra;

		finite_field *F;
		int idx;


		idx = The_Orbiter_top_level_session->find_symbol(all_rational_normal_forms_finite_field_label);

		F = (finite_field *) The_Orbiter_top_level_session->get_object(idx);

		Algebra.classes_GL(F, all_rational_normal_forms_d,
				FALSE /* f_no_eigenvalue_one */, verbose_level);



	}

	else if (f_eigenstuff) {


		algebra_global_with_action Algebra;
		int *data;
		int sz;
		finite_field *F;
		int idx;


		idx = The_Orbiter_top_level_session->find_symbol(eigenstuff_finite_field_label);

		F = (finite_field *) The_Orbiter_top_level_session->get_object(idx);

		Orbiter->Int_vec.scan(eigenstuff_coeffs, data, sz);

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

	character_table_burnside *CTB;

	CTB = NEW_OBJECT(character_table_burnside);

	CTB->do_it(deg, verbose_level);

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "interface_algebra::do_character_table_symmetric_group done" << endl;
	}
}

void interface_algebra::do_make_A5_in_PSL_2_q(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_algebra::do_make_A5_in_PSL_2_q" << endl;
		cout << "q=" << q << endl;
	}

	algebra_global_with_action A;

	A.A5_in_PSL_(q, verbose_level);

	if (f_v) {
		cout << "interface_algebra::do_make_A5_in_PSL_2_q done" << endl;
	}
}







}}

