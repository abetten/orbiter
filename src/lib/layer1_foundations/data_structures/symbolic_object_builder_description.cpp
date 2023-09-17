/*
 * symbolic_object_builder_description.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


symbolic_object_builder_description::symbolic_object_builder_description()
{
	f_label_txt = false;
	//std::string label_txt;

	f_label_tex = false;
	//std::string label_tex;

	f_managed_variables = false;
	//std::string managed_variables;

	f_text = false;
	//std::string text_txt;

	f_field = false;
	//std::string field_label;

	f_field_pointer = false;
	field_pointer = NULL;

	f_ring = false;
	//std::string ring_label;

	f_file = false;
	//std::string file_name;

	f_matrix = false;
	nb_rows = 0;

	f_determinant = false;
	//std::string determinant_source;

	f_characteristic_polynomial = false;
	//std::string characteristic_polynomial_variable;
	//std::string characteristic_polynomial_source;

	f_substitute = false;
	//std::string substitute_variables;
	//std::string substitute_target;
	//std::string substitute_source;

	f_simplify = false;
	//std::string simplify_source;

	f_expand = false;
	//std::string expand_source;

	f_right_nullspace = false;
	//std::string right_nullspace_source;


	f_minor = false;
	//std::string minor_source;
	minor_i = 0;
	minor_j = 0;

	f_symbolic_nullspace = false;
	//std::string symbolic_nullspace_source;

	f_stack_matrices_vertically = false;
	f_stack_matrices_horizontally = false;
	f_stack_matrices_z_shape = false;
	//std::string stack_matrices_label;

	f_multiply_2x2_from_the_left = false;
	//std::string multiply_2x2_from_the_left_source;
	//std::string multiply_2x2_from_the_left_A2;
	multiply_2x2_from_the_left_i = 0;
	multiply_2x2_from_the_left_j = 0;

	f_matrix_entry = false;
	//std::string matrix_entry_source;
	matrix_entry_i = 0;
	matrix_entry_j = 0;

	f_vector_entry = false;
	//std::string vector_entry_source;
	vector_entry_i = 0;


	f_collect = false;
	//std::string collect_source;
	//std::string collect_by;

	f_encode_CRC = false;
	encode_CRC_block_length = 0;
	//std::string encode_CRC_data_polynomial;
	//std::string encode_CRC_check_polynomial;

	f_decode_CRC = false;
	decode_CRC_block_length = false;
	//std::string decode_CRC_data_polynomial;
	//std::string decode_CRC_check_polynomial;

	f_submatrix = false;
	//std::string submatrix_source;
	submatrix_row_first = 0;
	submatrix_nb_rows = 0;
	submatrix_col_first = 0;
	submatrix_nb_cols = 0;

	f_do_not_simplify = false;

	f_write_trees_during_expand = false;

}

symbolic_object_builder_description::~symbolic_object_builder_description()
{
}


int symbolic_object_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	string_tools ST;

	if (f_v) {
		cout << "symbolic_object_builder_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-label_txt " << label_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label_tex " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-managed_variables") == 0) {
			f_managed_variables = true;
			managed_variables.assign(argv[++i]);
			if (f_v) {
				cout << "-managed_variables " << managed_variables << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-text") == 0) {
			f_text = true;
			text_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-text " << text_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = true;
			field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ring") == 0) {
			f_ring = true;
			ring_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ring " << ring_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = true;
			file_name.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << file_name << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-matrix") == 0) {
			f_matrix = true;
			nb_rows = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-matrix " << nb_rows << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-determinant") == 0) {
			f_determinant = true;
			determinant_source.assign(argv[++i]);
			if (f_v) {
				cout << "-determinant "
						<< " " << determinant_source
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-characteristic_polynomial") == 0) {
			f_characteristic_polynomial = true;
			characteristic_polynomial_variable.assign(argv[++i]);
			characteristic_polynomial_source.assign(argv[++i]);
			if (f_v) {
				cout << "-f_characteristic_polynomial "
						<< " " << characteristic_polynomial_variable
						<< " " << characteristic_polynomial_source
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-substitute") == 0) {
			f_substitute = true;
			substitute_variables.assign(argv[++i]);
			substitute_target.assign(argv[++i]);
			substitute_source.assign(argv[++i]);
			if (f_v) {
				cout << "-substitute "
						<< " " << substitute_variables
						<< " " << substitute_target
						<< " " << substitute_source
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-simplify") == 0) {
			f_simplify = true;
			simplify_source.assign(argv[++i]);
			if (f_v) {
				cout << "-simplify "
						<< " " << simplify_source
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-expand") == 0) {
			f_expand = true;
			expand_source.assign(argv[++i]);
			if (f_v) {
				cout << "-expand "
						<< " " << expand_source
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-right_nullspace") == 0) {
			f_right_nullspace = true;
			right_nullspace_source.assign(argv[++i]);
			if (f_v) {
				cout << "-right_nullspace "
						<< " " << right_nullspace_source
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-minor") == 0) {
			f_minor = true;
			minor_source.assign(argv[++i]);
			minor_i = ST.strtoi(argv[++i]);
			minor_j = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-minor "
						<< " " << minor_source
						<< " " << minor_i
						<< " " << minor_j
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-symbolic_nullspace") == 0) {
			f_symbolic_nullspace = true;
			symbolic_nullspace_source.assign(argv[++i]);
			if (f_v) {
				cout << "-symbolic_nullspace "
						<< " " << symbolic_nullspace_source
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stack_matrices_vertically") == 0) {
			f_stack_matrices_vertically = true;
			stack_matrices_label.assign(argv[++i]);
			if (f_v) {
				cout << "-stack_matrices_vertically "
						<< " " << stack_matrices_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stack_matrices_horizontally") == 0) {
			f_stack_matrices_horizontally = true;
			stack_matrices_label.assign(argv[++i]);
			if (f_v) {
				cout << "-stack_matrices_horizontally "
						<< " " << stack_matrices_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-stack_matrices_z_shape") == 0) {
			f_stack_matrices_z_shape = true;
			stack_matrices_label.assign(argv[++i]);
			if (f_v) {
				cout << "-stack_matrices_z_shape "
						<< " " << stack_matrices_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-multiply_2x2_from_the_left") == 0) {
			f_multiply_2x2_from_the_left = true;
			multiply_2x2_from_the_left_source.assign(argv[++i]);
			multiply_2x2_from_the_left_A2.assign(argv[++i]);
			multiply_2x2_from_the_left_i = ST.strtoi(argv[++i]);
			multiply_2x2_from_the_left_j = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-multiply_2x2_from_the_left "
						<< " " << multiply_2x2_from_the_left_source
						<< " " << multiply_2x2_from_the_left_A2
						<< " " << multiply_2x2_from_the_left_i
						<< " " << multiply_2x2_from_the_left_j
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-matrix_entry") == 0) {
			f_matrix_entry = true;
			matrix_entry_source.assign(argv[++i]);
			matrix_entry_i = ST.strtoi(argv[++i]);
			matrix_entry_j = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-matrix_entry "
						<< " " << matrix_entry_source
						<< " " << matrix_entry_i
						<< " " << matrix_entry_j
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-vector_entry") == 0) {
			f_vector_entry = true;
			vector_entry_source.assign(argv[++i]);
			vector_entry_i = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-vector_entry "
						<< " " << vector_entry_source
						<< " " << vector_entry_i
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-collect") == 0) {
			f_collect = true;
			collect_source.assign(argv[++i]);
			collect_by.assign(argv[++i]);
			if (f_v) {
				cout << "-collect "
						<< " " << collect_source
						<< " " << collect_by
						<< endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-encode_CRC") == 0) {
			f_encode_CRC = true;
			encode_CRC_block_length = ST.strtoi(argv[++i]);
			encode_CRC_data_polynomial.assign(argv[++i]);
			encode_CRC_check_polynomial.assign(argv[++i]);
			if (f_v) {
				cout << "-encode_CRC "
						<< " " << encode_CRC_block_length
						<< " " << encode_CRC_data_polynomial
						<< " " << encode_CRC_check_polynomial
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-decode_CRC") == 0) {
			f_decode_CRC = true;
			decode_CRC_block_length = ST.strtoi(argv[++i]);
			decode_CRC_data_polynomial.assign(argv[++i]);
			decode_CRC_check_polynomial.assign(argv[++i]);
			if (f_v) {
				cout << "-decode_CRC "
						<< " " << decode_CRC_block_length
						<< " " << decode_CRC_data_polynomial
						<< " " << decode_CRC_check_polynomial
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-submatrix") == 0) {
			f_submatrix = true;
			submatrix_source.assign(argv[++i]);
			submatrix_row_first = ST.strtoi(argv[++i]);
			submatrix_nb_rows = ST.strtoi(argv[++i]);
			submatrix_col_first = ST.strtoi(argv[++i]);
			submatrix_nb_cols = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-submatrix "
						<< " " << submatrix_source
						<< " " << submatrix_row_first
						<< " " << submatrix_nb_rows
						<< " " << submatrix_col_first
						<< " " << submatrix_nb_cols
						<< endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-do_not_simplify") == 0) {
			f_do_not_simplify = true;
			if (f_v) {
				cout << "-do_not_simplify "
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-write_trees_during_expand") == 0) {
			f_write_trees_during_expand = true;
			if (f_v) {
				cout << "-write_trees_during_expand "
						<< endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "symbolic_object_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "symbolic_object_builder_description::read_arguments done" << endl;
	}
	return i + 1;
}

void symbolic_object_builder_description::print()
{
	if (f_label_txt) {
		cout << "-label_txt " << label_txt << endl;
	}
	if (f_label_tex) {
		cout << "-label_tex " << label_tex << endl;
	}
	if (f_managed_variables) {
		cout << "-managed_variables " << managed_variables << endl;
	}
	if (f_text) {
		cout << "-text " << text_txt << endl;
	}
	if (f_field) {
		cout << "-field " << field_label << endl;
	}
	if (f_field_pointer) {
		cout << "field_pointer is given" << endl;
	}
	if (f_ring) {
		cout << "-ring " << ring_label << endl;
	}
	if (f_file) {
		cout << "-file " << file_name << endl;
	}
	if (f_matrix) {
		cout << "-matrix " << nb_rows << endl;
	}
	if (f_determinant) {
		cout << "-determinant "
				<< " " << determinant_source
				<< endl;
	}
	if (f_characteristic_polynomial) {
		cout << "-characteristic_polynomial "
				<< " " << characteristic_polynomial_variable
				<< " " << characteristic_polynomial_source
				<< endl;
	}
	if (f_substitute) {
		cout << "-substitute "
				<< " " << substitute_variables
				<< " " << substitute_target
				<< " " << substitute_source
				<< endl;
	}
	if (f_simplify) {
		cout << "-simplify "
				<< " " << simplify_source
				<< endl;
	}
	if (f_expand) {
		cout << "-expand "
				<< " " << expand_source
				<< endl;
	}
	if (f_right_nullspace) {
		cout << "-right_nullspace "
				<< " " << right_nullspace_source
				<< endl;
	}
	if (f_minor) {
		cout << "-minor "
				<< " " << minor_source
				<< " " << minor_i
				<< " " << minor_j
				<< endl;
	}
	if (f_symbolic_nullspace) {
		cout << "-symbolic_nullspace "
				<< " " << symbolic_nullspace_source
				<< endl;
	}
	if (f_stack_matrices_vertically) {
			cout << "-stack_matrices_vertically "
					<< " " << stack_matrices_label
					<< endl;
	}
	if (f_stack_matrices_horizontally) {
			cout << "-stack_matrices_horizontally "
					<< " " << stack_matrices_label
					<< endl;
	}
	if (f_stack_matrices_z_shape) {
			cout << "-stack_matrices_z_shape "
					<< " " << stack_matrices_label
					<< endl;
	}
	if (f_multiply_2x2_from_the_left) {
		cout << "-multiply_2x2_from_the_left "
				<< " " << multiply_2x2_from_the_left_source
				<< " " << multiply_2x2_from_the_left_A2
				<< " " << multiply_2x2_from_the_left_i
				<< " " << multiply_2x2_from_the_left_j
				<< endl;
	}
	if (f_matrix_entry) {
		cout << "-matrix_entry "
				<< " " << matrix_entry_source
				<< " " << matrix_entry_i
				<< " " << matrix_entry_j
				<< endl;
	}
	if (f_vector_entry) {
		cout << "-vector_entry "
				<< " " << vector_entry_source
				<< " " << vector_entry_i
				<< endl;
	}
	if (f_collect) {
		cout << "-collect "
				<< " " << collect_source
				<< " " << collect_by
				<< endl;
	}
	if (f_encode_CRC) {
		cout << "-encode_CRC "
				<< " " << encode_CRC_block_length
				<< " " << encode_CRC_data_polynomial
				<< " " << encode_CRC_check_polynomial
				<< endl;
	}
	if (f_decode_CRC) {
		cout << "-decode_CRC "
				<< " " << decode_CRC_block_length
				<< " " << decode_CRC_data_polynomial
				<< " " << decode_CRC_check_polynomial
				<< endl;
	}
	if (f_submatrix) {
		cout << "-submatrix "
				<< " " << submatrix_source
				<< " " << submatrix_row_first
				<< " " << submatrix_nb_rows
				<< " " << submatrix_col_first
				<< " " << submatrix_nb_cols
				<< endl;
	}
	if (f_do_not_simplify) {
		cout << "-do_not_simplify "
				<< endl;
	}
	if (f_write_trees_during_expand) {
			cout << "-write_trees_during_expand "
					<< endl;
	}

}


}}}

