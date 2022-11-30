/*
 * create_code_description.cpp
 *
 *  Created on: Aug 10, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_coding_theory {


create_code_description::create_code_description()
{
	f_field = FALSE;
	//std::string field_label;

	f_linear_code_through_generator_matrix = FALSE;
	//std::string linear_code_through_generator_matrix_label_genma;

	f_linear_code_from_projective_set = FALSE;
	linear_code_from_projective_set_nmk = 0;
	//std::string linear_code_from_projective_set_set;

	f_linear_code_by_columns_of_parity_check = FALSE;
	linear_code_by_columns_of_parity_check_nmk = 0;
	//std::string linear_code_by_columns_of_parity_check_set;

	f_first_order_Reed_Muller = FALSE;
	first_order_Reed_Muller_m = 0;

	f_BCH = FALSE;
	BCH_n = 0;
	BCH_d = 0;

	f_Reed_Solomon = FALSE;
	Reed_Solomon_n = 0;
	Reed_Solomon_d = 0;

	f_Gilbert_Varshamov = FALSE;
	Gilbert_Varshamov_n = 0;
	Gilbert_Varshamov_k = 0;
	Gilbert_Varshamov_d = 0;

	std::vector<code_modification_description> Modifications;

}


int create_code_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "create_code_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		code_modification_description M;

		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = TRUE;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-linear_code_through_generator_matrix") == 0) {
			f_linear_code_through_generator_matrix = TRUE;
			linear_code_through_generator_matrix_label_genma.assign(argv[++i]);
			if (f_v) {
				cout << "-linear_code_through_generator_matrix " << linear_code_through_generator_matrix_label_genma << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-linear_code_from_projective_set") == 0) {
			f_linear_code_from_projective_set = TRUE;
			linear_code_from_projective_set_nmk = ST.strtoi(argv[++i]);
			linear_code_from_projective_set_set.assign(argv[++i]);
			if (f_v) {
				cout << "-linear_code_from_projective_set "
						<< linear_code_from_projective_set_nmk
						<< " " << linear_code_from_projective_set_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-linear_code_by_columns_of_parity_check") == 0) {
			f_linear_code_by_columns_of_parity_check = TRUE;
			linear_code_by_columns_of_parity_check_nmk = ST.strtoi(argv[++i]);
			linear_code_by_columns_of_parity_check_set.assign(argv[++i]);
			if (f_v) {
				cout << "-linear_code_by_columns_of_parity_check "
						<< linear_code_by_columns_of_parity_check_nmk
						<< " " << linear_code_by_columns_of_parity_check_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-first_order_Reed_Muller") == 0) {
			f_first_order_Reed_Muller = TRUE;
			first_order_Reed_Muller_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-first_order_Reed_Muller " << first_order_Reed_Muller_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-BCH") == 0) {
			f_BCH = TRUE;
			BCH_n = ST.strtoi(argv[++i]);
			BCH_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-BCH " << BCH_n << " " << BCH_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Reed_Solomon") == 0) {
			f_Reed_Solomon = TRUE;
			Reed_Solomon_n = ST.strtoi(argv[++i]);
			Reed_Solomon_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Reed_Solomon " << Reed_Solomon_n
						<< " " << Reed_Solomon_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Gilbert_Varshamov") == 0) {
			f_Gilbert_Varshamov = TRUE;
			Gilbert_Varshamov_n = ST.strtoi(argv[++i]);
			Gilbert_Varshamov_k = ST.strtoi(argv[++i]);
			Gilbert_Varshamov_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Gilbert_Varshamov " << Gilbert_Varshamov_n
						<< " " << Gilbert_Varshamov_k
						<< " " << Gilbert_Varshamov_d
						<< endl;
			}
		}


		else if (M.check_and_parse_argument(
				argc, i, argv,
				verbose_level)) {
			Modifications.push_back(M);
		}

		if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "create_code_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "create_code_description::read_arguments done" << endl;
	}
	return i + 1;
}

void create_code_description::print()
{
	if (f_field) {
		cout << "-field " << field_label << endl;
	}
	if (f_linear_code_through_generator_matrix) {
		cout << "-linear_code_through_generator_matrix "
				<< linear_code_through_generator_matrix_label_genma << endl;
	}
	if (f_linear_code_from_projective_set) {
		cout << "-linear_code_from_projective_set "
				<< linear_code_from_projective_set_nmk
				<< " " << linear_code_from_projective_set_set << endl;
	}
	if (f_linear_code_by_columns_of_parity_check) {
		cout << "-linear_code_by_columns_of_parity_check "
				<< linear_code_by_columns_of_parity_check_nmk
				<< " " << linear_code_by_columns_of_parity_check_set << endl;
	}
	if (f_first_order_Reed_Muller) {
		cout << "-first_order_Reed_Muller " << first_order_Reed_Muller_m << endl;
	}
	if (f_Reed_Solomon) {
		cout << "-Reed_Solomon " << Reed_Solomon_n << " " << Reed_Solomon_d << endl;
	}
	if (f_Gilbert_Varshamov) {
		cout << "-Gilbert_Varshamov " << Gilbert_Varshamov_n
				<< " " << Gilbert_Varshamov_k
				<< " " << Gilbert_Varshamov_d
				<< endl;
	}

	int i;

	for (i = 0; i < Modifications.size(); i++) {
		Modifications[i].print();
	}

}




}}}

