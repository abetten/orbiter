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
	Record_birth();
	f_field = false;
	//std::string field_label;

	f_generator_matrix = false;
	//std::string generator_matrix_label_genma;

	f_basis = false;
	basis_n = 0;
	//std::string basis_label;

	f_long_code = false;
	long_code_n = 0;
	//long_code_generators;

	f_projective_set = false;
	projective_set_nmk = 0;
	//std::string projective_set_set;

	f_columns_of_generator_matrix = false;
	columns_of_generator_matrix_k = 0;
	//std::string by_columns_of_generator_matrix_set;

	f_Reed_Muller = false;
	Reed_Muller_m = 0;

	f_BCH = false;
	BCH_n = 0;
	BCH_d = 0;

	f_Reed_Solomon = false;
	Reed_Solomon_n = 0;
	Reed_Solomon_d = 0;

	f_Gilbert_Varshamov = false;
	Gilbert_Varshamov_n = 0;
	Gilbert_Varshamov_k = 0;
	Gilbert_Varshamov_d = 0;

	f_ttpA = false;
	//std::string ttpA_field_label;

	f_ttpB = false;
	//std::string ttpB_field_label;

	std::vector<code_modification_description> Modifications;

}

create_code_description::~create_code_description()
{
	Record_death();
}

int create_code_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "create_code_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		code_modification_description M;

		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = true;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-generator_matrix") == 0) {
			f_generator_matrix = true;
			generator_matrix_label_genma.assign(argv[++i]);
			if (f_v) {
				cout << "-generator_matrix " << generator_matrix_label_genma << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-basis") == 0) {
			f_basis = true;
			basis_n = ST.strtoi(argv[++i]);
			basis_label.assign(argv[++i]);
			if (f_v) {
				cout << "-basis " << basis_n << " " << basis_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-long_code") == 0) {
			f_long_code = true;
			long_code_n = ST.strtoi(argv[++i]);

			int n, h;
			n = ST.strtoi(argv[++i]);
			for (h = 0; h < n; h++) {
				string s;

				s.assign(argv[++i]);
				long_code_generators.push_back(s);
			}
			if (f_v) {
				cout << "-long_code " << long_code_n << endl;
				for (int h = 0; h < n; h++) {
					cout << " " << long_code_generators[h] << endl;
				}
			}
		}
		else if (ST.stringcmp(argv[i], "-projective_set") == 0) {
			f_projective_set = true;
			projective_set_nmk = ST.strtoi(argv[++i]);
			projective_set_set.assign(argv[++i]);
			if (f_v) {
				cout << "-projective_set "
						<< projective_set_nmk
						<< " " << projective_set_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-columns_of_generator_matrix") == 0) {
			f_columns_of_generator_matrix = true;
			columns_of_generator_matrix_k = ST.strtoi(argv[++i]);
			columns_of_generator_matrix_set.assign(argv[++i]);
			if (f_v) {
				cout << "-columns_of_generator_matrix "
						<< columns_of_generator_matrix_k
						<< " " << columns_of_generator_matrix_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Reed_Muller") == 0) {
			f_Reed_Muller = true;
			Reed_Muller_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Reed_Muller " << Reed_Muller_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-BCH") == 0) {
			f_BCH = true;
			BCH_n = ST.strtoi(argv[++i]);
			BCH_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-BCH " << BCH_n << " " << BCH_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Reed_Solomon") == 0) {
			f_Reed_Solomon = true;
			Reed_Solomon_n = ST.strtoi(argv[++i]);
			Reed_Solomon_d = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Reed_Solomon " << Reed_Solomon_n
						<< " " << Reed_Solomon_d << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Gilbert_Varshamov") == 0) {
			f_Gilbert_Varshamov = true;
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
		else if (ST.stringcmp(argv[i], "-ttpA") == 0) {
			f_ttpA = true;
			ttpA_field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ttpA " << ttpA_field_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ttpB") == 0) {
			f_ttpB = true;
			ttpB_field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ttpB " << ttpB_field_label
						<< endl;
			}
		}

		else if (M.check_and_parse_argument(
				argc, i, argv,
				verbose_level)) {
			Modifications.push_back(M);
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
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
	if (f_generator_matrix) {
		cout << "-generator_matrix "
				<< generator_matrix_label_genma << endl;
	}
	if (f_basis) {
		cout << "-basis " << basis_n << " " << basis_label << endl;
	}
	if (f_long_code) {
		cout << "-long_code " << long_code_n << endl;
		for (int h = 0; h < long_code_n; h++) {
			cout << " " << long_code_generators[h] << endl;
		}
	}
	if (f_projective_set) {
		cout << "-projective_set "
				<< projective_set_nmk
				<< " " << projective_set_set << endl;
	}
	if (f_columns_of_generator_matrix) {
		cout << "-columns_of_generator_matrix "
				<< columns_of_generator_matrix_k
				<< " " << columns_of_generator_matrix_set << endl;
	}
	if (f_Reed_Muller) {
		cout << "-first_order_Reed_Muller " << Reed_Muller_m << endl;
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
	if (f_ttpA) {
		cout << "-ttpA " << ttpA_field_label
				<< endl;
	}
	if (f_ttpB) {
		cout << "-ttpB " << ttpB_field_label
				<< endl;
	}

	int i;

	for (i = 0; i < Modifications.size(); i++) {
		Modifications[i].print();
	}

}




}}}

