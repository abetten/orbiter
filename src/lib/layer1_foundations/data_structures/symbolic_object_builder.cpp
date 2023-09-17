/*
 * symbolic_object_builder.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


symbolic_object_builder::symbolic_object_builder()
{
	Descr = NULL;

	Fq = NULL;

	Ring = NULL;

	Formula_vector = NULL;
}

symbolic_object_builder::~symbolic_object_builder()
{
	if (Formula_vector) {
		FREE_OBJECT(Formula_vector);
		Formula_vector = NULL;
	}
}

void symbolic_object_builder::init(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::init label=" << label << endl;
	}

	symbolic_object_builder::Descr = Descr;
	symbolic_object_builder::label = label;
	//string managed_variables;


	if (Descr->f_field) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-field " << Descr->field_label << endl;
		}
		Fq = Get_finite_field(Descr->field_label);
	}
	else if (Descr->f_field_pointer) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"field_pointer" << endl;
		}
		Fq = Descr->field_pointer;
	}

	if (Descr->f_ring) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-ring " << Descr->ring_label << endl;
		}
		Ring = Get_ring(Descr->ring_label);
	}

	if (Descr->f_ring && !Descr->f_field && !Descr->f_field_pointer) {
		Fq = Ring->get_F();
	}

	if (!Descr->f_ring && !Descr->f_field && !Descr->f_field_pointer) {
		cout << "symbolic_object_builder::init please use either "
				"-ring <ring> or -field <field> "
				"to specify the domain of coefficients" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "symbolic_object_builder::init "
				"before process_arguments" << endl;
	}
	process_arguments(verbose_level - 2);
	if (f_v) {
		cout << "symbolic_object_builder::init "
				"after process_arguments" << endl;
	}

	if (f_v) {
		cout << "symbolic_object_builder::init "
				"symbolic object vector of length "
				<< Formula_vector->len << endl;
		//Lint_vec_print(cout, v, len);
		cout << endl;

		//cout << "ir tree:" << endl;
		//print_Sajeeb(cout);


		cout << "before Formula_vector->print_formula_size" << endl;
		Formula_vector->print_formula_size(cout, verbose_level);
		cout << "after Formula_vector->print_formula_size" << endl;

		cout << "before Formula_vector->print_formula" << endl;
		Formula_vector->print_formula(cout, verbose_level);
		cout << "after Formula_vector->print_formula" << endl;

		cout << "before Formula_vector->print_latex" << endl;
		Formula_vector->print_latex(cout, label);
		cout << "after Formula_vector->print_latex" << endl;


		string fname;

		fname = label + ".tex";

		{
			l1_interfaces::latex_interface L;
			ofstream ost(fname);

			L.head_easy_and_enlarged(ost);

			cout << "before Formula_vector->print_latex" << endl;
			Formula_vector->print_latex(ost, label);
			cout << "after Formula_vector->print_latex" << endl;

			L.foot(ost);
		}

		//latex_split(std::string &name, int split_level, int split_mod, int verbose_level)

		cout << "before Formula_vector->latex_tree" << endl;
		Formula_vector->latex_tree(verbose_level);
		cout << "after Formula_vector->latex_tree" << endl;

		cout << "before Formula_vector->export_tree" << endl;
		Formula_vector->export_tree(verbose_level);
		cout << "after Formula_vector->export_tree" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::init done" << endl;
	}
}


void symbolic_object_builder::process_arguments(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::process_arguments" << endl;
	}


	if (Descr->f_text) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"-text " << Descr->text_txt << endl;
			if (Descr->f_managed_variables) {
				cout << "symbolic_object_builder::process_arguments "
						"-managed_variables " << Descr->managed_variables << endl;
			}
		}


		Formula_vector = NEW_OBJECT(expression_parser::formula_vector);
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"before Formula_vector->init_from_text" << endl;
		}
		Formula_vector->init_from_text(
				label /*Descr->label_txt*/,
				Descr->label_tex,
				Descr->text_txt,
				Fq,
				Descr->f_managed_variables,
				Descr->managed_variables,
				Descr->f_matrix, Descr->nb_rows,
				verbose_level - 1);
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"after Formula_vector->init_from_text" << endl;
		}

	}
	else if (Descr->f_determinant) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -determinant"
					<< " " << Descr->determinant_source
					<< endl;
		}

		do_determinant(
				Descr,
				label,
				verbose_level - 1);


	}
	else if (Descr->f_characteristic_polynomial) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -characteristic_polynomial"
					<< " " << Descr->characteristic_polynomial_variable
					<< " " << Descr->characteristic_polynomial_source
					<< endl;
		}

		do_characteristic_polynomial(
				Descr,
				label,
				verbose_level - 1);


	}

	else if (Descr->f_substitute) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -substitute"
					<< " " << Descr->substitute_variables
					<< " " << Descr->substitute_target
					<< " " << Descr->substitute_source
					<< endl;
		}

		do_substitute(
					Descr,
					label,
					verbose_level - 1);


	}

	else if (Descr->f_simplify) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -simplify"
					<< " " << Descr->simplify_source
					<< endl;
		}


		do_simplify(
					Descr,
					label,
					verbose_level - 1);

		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -simplify finished" << endl;
		}

	}
	else if (Descr->f_expand) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -expand"
					<< " " << Descr->expand_source
					<< endl;
		}


		do_expand(
					Descr,
					label,
					verbose_level - 1);

		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -expand finished" << endl;
		}

	}
	else if (Descr->f_right_nullspace) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -right_nullspace"
					<< " " << Descr->right_nullspace_source
					<< endl;
		}

		do_right_nullspace(
					Descr,
					label,
					verbose_level - 1);

	}
	else if (Descr->f_minor) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -minor"
					<< " " << Descr->minor_source
					<< " " << Descr->minor_i
					<< " " << Descr->minor_j
					<< endl;
		}

		do_minor(
					Descr,
					Descr->minor_i, Descr->minor_j,
					label,
					verbose_level - 1);

	}

	else if (Descr->f_symbolic_nullspace) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -symbolic_nullspace"
					<< " " << Descr->symbolic_nullspace_source
					<< endl;
		}

		do_symbolic_nullspace(
					Descr,
					label,
					verbose_level - 1);

	}
	else if (Descr->f_stack_matrices_vertically) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -stack_matrices_vertically"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level - 1);

	}
	else if (Descr->f_stack_matrices_horizontally) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -stack_matrices_horizontally"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level - 1);
	}
	else if (Descr->f_stack_matrices_z_shape) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -stack_matrices_z_shape"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level - 1);
	}

	else if (Descr->f_multiply_2x2_from_the_left) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -multiply_2x2_from_the_left"
					<< " " << Descr->multiply_2x2_from_the_left_source
					<< " " << Descr->multiply_2x2_from_the_left_A2
					<< " " << Descr->multiply_2x2_from_the_left_i
					<< " " << Descr->multiply_2x2_from_the_left_j
					<< endl;
		}
		do_multiply_2x2_from_the_left(
					Descr,
					label,
					verbose_level - 1);
	}

	else if (Descr->f_matrix_entry) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -matrix_entry"
					<< " " << Descr->matrix_entry_source
					<< endl;
		}
		do_matrix_entry(
					Descr,
					label,
					verbose_level - 1);
	}

	else if (Descr->f_vector_entry) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -vector_entry"
					<< " " << Descr->vector_entry_source
					<< endl;
		}
		do_vector_entry(
					Descr,
					label,
					verbose_level - 1);
	}

	else if (Descr->f_collect) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -collect"
					<< " " << Descr->collect_source
					<< " " << Descr->collect_by
					<< endl;
		}
		do_collect(
					Descr,
					label,
					verbose_level - 1);
	}

	else if (Descr->f_encode_CRC) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -encode_CRC"
					<< " " << Descr->encode_CRC_block_length
					<< " " << Descr->encode_CRC_data_polynomial
					<< " " << Descr->encode_CRC_check_polynomial
					<< endl;
		}
		do_CRC_encode(
					Descr,
					label,
					verbose_level - 1);
	}

	else if (Descr->f_decode_CRC) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -decode_CRC"
					<< " " << Descr->decode_CRC_block_length
					<< " " << Descr->decode_CRC_data_polynomial
					<< " " << Descr->decode_CRC_check_polynomial
					<< endl;
		}
		do_CRC_decode(
					Descr,
					label,
					verbose_level - 1);
	}


	else if (Descr->f_submatrix) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments -submatrix"
					<< " " << Descr->submatrix_source
					<< " " << Descr->submatrix_row_first
					<< " " << Descr->submatrix_nb_rows
					<< " " << Descr->submatrix_col_first
					<< " " << Descr->submatrix_nb_cols
					<< endl;
		}
		do_submatrix(
					Descr,
					label,
					verbose_level - 1);
	}


	if (Descr->f_file) {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"-file" << Descr->file_name << endl;
		}


	}

	if (f_v) {
		cout << "symbolic_object_builder::process_arguments "
				"before simplifying the formula vector" << endl;
	}

	if (!Descr->f_do_not_simplify) {
		expression_parser::formula_vector *old;

		old = Formula_vector;

		Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"before Formula_vector->simplify label=" << label << endl;
		}

		Formula_vector->simplify(
				old,
				Fq,
				label,
				label,
				verbose_level - 2);

		FREE_OBJECT(old);
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"after simplifying the formula vector" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "symbolic_object_builder::process_arguments "
					"skipping simplification because of -do_not_simplify" << endl;

		}
	}

	if (f_v) {
		cout << "symbolic_object_builder::process_arguments done" << endl;
	}

}


void symbolic_object_builder::do_determinant(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_determinant" << endl;
	}

	data_structures::symbolic_object_builder *O1;

	O1 = Get_symbol(Descr->determinant_source);

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	if (f_v) {
		cout << "symbolic_object_builder::do_determinant "
				"before Formula_vector->determinant" << endl;
	}
	Formula_vector->determinant(
			O1->Formula_vector,
			Fq,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_determinant "
				"after Formula_vector->determinant" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_determinant done" << endl;
	}

}


void symbolic_object_builder::do_characteristic_polynomial(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_characteristic_polynomial" << endl;
	}

	string variable;

	variable = Descr->characteristic_polynomial_variable;

	data_structures::symbolic_object_builder *O1;

	O1 = Get_symbol(Descr->characteristic_polynomial_source);

	if (!O1->Formula_vector->f_matrix) {
		cout << "symbolic_object_builder::do_characteristic_polynomial "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int n;

	n = O1->Formula_vector->nb_rows;

	if (O1->Formula_vector->nb_rows != O1->Formula_vector->nb_cols) {
		cout << "symbolic_object_builder::do_characteristic_polynomial "
				"the matrix is not square" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "symbolic_object_builder::do_characteristic_polynomial "
				"we found a square matrix of size " << n << endl;
	}

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "symbolic_object_builder::do_characteristic_polynomial "
				"before Formula_vector->characteristic_polynomial" << endl;
	}
	Formula_vector->characteristic_polynomial(
			O1->Formula_vector,
			Fq,
			variable,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_characteristic_polynomial "
				"after Formula_vector->characteristic_polynomial" << endl;
	}

	if (f_v) {
		cout << "symbolic_object_builder::do_characteristic_polynomial done" << endl;
	}

}



void symbolic_object_builder::do_substitute(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_substitute" << endl;
	}

	data_structures::symbolic_object_builder *O_target;
	data_structures::symbolic_object_builder *O_source;

	O_target = Get_symbol(Descr->substitute_target);
	O_source = Get_symbol(Descr->substitute_source);


	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "symbolic_object_builder::do_substitute "
				"before Formula_vector->substitute" << endl;
	}
	Formula_vector->substitute(
			O_source->Formula_vector,
			O_target->Formula_vector,
			Descr->substitute_variables,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_substitute "
				"after Formula_vector->substitute" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_substitute done" << endl;
	}

}

void symbolic_object_builder::do_simplify(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_simplify" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->simplify_source);

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	if (f_v) {
		cout << "symbolic_object_builder::do_simplify "
				"before Formula_vector->simplify" << endl;
	}
	Formula_vector->simplify(
			O_source->Formula_vector,
			Fq,
			label, label,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_simplify "
				"after Formula_vector->simplify" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_simplify done" << endl;
	}

}

void symbolic_object_builder::do_expand(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_expand" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->expand_source);

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	if (f_v) {
		cout << "symbolic_object_builder::do_expand "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector->expand(
			O_source->Formula_vector,
			Fq,
			label, label,
			Descr->managed_variables,
			Descr->f_write_trees_during_expand,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_expand "
				"after Formula_vector->expand" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_expand done" << endl;
	}

}


void symbolic_object_builder::do_right_nullspace(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_right_nullspace" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->right_nullspace_source);



	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "symbolic_object_builder::do_right_nullspace "
				"before Formula_vector->right_nullspace" << endl;
	}
	Formula_vector->right_nullspace(
			O_source->Formula_vector,
			Fq,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_right_nullspace "
				"after Formula_vector->right_nullspace" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_right_nullspace done" << endl;
	}
}


void symbolic_object_builder::do_minor(
		symbolic_object_builder_description *Descr,
		int minor_i, int minor_j,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_minor" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->minor_source);



	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "symbolic_object_builder::do_minor "
				"before Formula_vector->matrix_minor" << endl;
	}
	Formula_vector->matrix_minor(
			O_source->Formula_vector,
			Fq,
			minor_i, minor_j,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_minor "
				"after Formula_vector->matrix_minor" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_minor done" << endl;
	}
}


void symbolic_object_builder::do_symbolic_nullspace(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_symbolic_nullspace" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->symbolic_nullspace_source);



	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "symbolic_object_builder::do_symbolic_nullspace "
				"before Formula_vector->symbolic_nullspace" << endl;
	}
	Formula_vector->symbolic_nullspace(
			O_source->Formula_vector,
			Fq,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_symbolic_nullspace "
				"after Formula_vector->symbolic_nullspace" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_symbolic_nullspace done" << endl;
	}
}





void symbolic_object_builder::do_stack(
		symbolic_object_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_stack" << endl;
	}


	data_structures::string_tools ST;
	std::vector<std::string> variables;

	ST.parse_comma_separated_list(
			Descr->stack_matrices_label, variables,
			verbose_level - 1);

	int i;
	int *Nb_rows;
	int *Nb_cols;

	Nb_rows = NEW_int(variables.size());
	Nb_cols = NEW_int(variables.size());

	if (variables.size() < 1) {
		cout << "symbolic_object_builder::do_stack "
				"we need at least one matrix" << endl;
		exit(1);
	}
	for (i = 0; i < variables.size(); i++) {

		data_structures::symbolic_object_builder *O_source;

		O_source = Get_symbol(variables[i]);

		if (!O_source->Formula_vector->f_matrix) {
			cout << "symbolic_object_builder::do_stack "
					"input " << variables[i] << " is not a matrix" << endl;
			exit(1);
		}
		Nb_rows[i] = O_source->Formula_vector->nb_rows;
		Nb_cols[i] = O_source->Formula_vector->nb_cols;

	}

	if (Descr->f_stack_matrices_vertically) {
		if (f_v) {
			cout << "symbolic_object_builder::do_stack -stack_matrices_vertically"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}

		if (!orbiter_kernel_system::Orbiter->Int_vec->is_constant(
				Nb_cols, variables.size())) {
			cout << "the matrices do not have the same number of columns" << endl;
			exit(1);
		}

		int nb_c;
		int N;

		nb_c = Nb_cols[0];

		N = 0;

		for (i = 0; i < variables.size(); i++) {
			 N += Nb_rows[i];
		}

		Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

		string label_txt;
		string label_tex;

		label_txt = "stack_v";
		label_tex = "stackv";

		int len, n, j;

		len = N * nb_c;

		Formula_vector->init_and_allocate(
					label_txt, label_tex,
					true,
					Descr->managed_variables,
					len, verbose_level - 1);

		n = 0;

		for (i = 0; i < variables.size(); i++) {

			data_structures::symbolic_object_builder *O_source;

			O_source = Get_symbol(variables[i]);

			if (O_source->Formula_vector->len != Nb_rows[i] * nb_c) {
				cout << "O_source->Formula_vector->len != Nb_rows[i] * nb_c" << endl;
				exit(1);
			}

			for (j = 0; j < O_source->Formula_vector->len; j++) {
				O_source->Formula_vector->V[j].copy_to(
						&Formula_vector->V[n * nb_c + j], verbose_level - 1);
			}

			n += Nb_rows[i];
		}

		Formula_vector->f_matrix = true;
		Formula_vector->nb_rows = N;
		Formula_vector->nb_cols = nb_c;

	}
	if (Descr->f_stack_matrices_horizontally) {
		if (f_v) {
			cout << "symbolic_object_builder::do_stack -stack_matrices_horizontally"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		if (!orbiter_kernel_system::Orbiter->Int_vec->is_constant(
				Nb_rows, variables.size())) {
			cout << "the matrices do not have the same number of rows" << endl;
			exit(1);
		}
	}
	if (Descr->f_stack_matrices_z_shape) {
		if (f_v) {
			cout << "symbolic_object_builder::do_stack -stack_matrices_z_shape"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_stack done" << endl;
	}
}

void symbolic_object_builder::do_multiply_2x2_from_the_left(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left" << endl;
	}


	data_structures::symbolic_object_builder *O_source;
	data_structures::symbolic_object_builder *O_A2;

	O_source = Get_symbol(Descr->multiply_2x2_from_the_left_source);
	O_A2 = Get_symbol(Descr->multiply_2x2_from_the_left_A2);

	int i, j;

	i = Descr->multiply_2x2_from_the_left_i;
	j = Descr->multiply_2x2_from_the_left_j;

	expression_parser::formula_vector *Formula_vector_tmp;

	Formula_vector_tmp = NEW_OBJECT(expression_parser::formula_vector);

	int len;

	if (!O_source->Formula_vector->f_matrix) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"input M must be a matrix" << endl;
		exit(1);
	}
	if (!O_A2->Formula_vector->f_matrix) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"input A2 must be a matrix" << endl;
		exit(1);
	}

	len = O_source->Formula_vector->nb_rows * O_source->Formula_vector->nb_cols;

	Formula_vector_tmp->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				len, verbose_level - 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"before Formula_vector_tmp->multiply_2by2_from_the_left" << endl;
	}
	Formula_vector_tmp->multiply_2by2_from_the_left(
			O_source->Formula_vector,
			O_A2->Formula_vector,
			i, j,
			O_source->Fq,
			label, label,
			Descr->managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"after Formula_vector_tmp->multiply_2by2_from_the_left" << endl;
	}


	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	if (f_v) {
		cout << "symbolic_object_builder::do_expand "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector->expand(
			Formula_vector_tmp,
			Fq,
			label, label,
			Descr->managed_variables,
			Descr->f_write_trees_during_expand,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::do_expand "
				"after Formula_vector->expand" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left done" << endl;
	}
}

void symbolic_object_builder::do_matrix_entry(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->matrix_entry_source);

	int i, j;

	i = Descr->matrix_entry_i;
	j = Descr->matrix_entry_j;

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	int len;

	if (!O_source->Formula_vector->f_matrix) {
		cout << "symbolic_object_builder::do_matrix_entry "
				"input M must be a matrix" << endl;
		exit(1);
	}
	int n;

	//m = O_source->Formula_vector->nb_rows;
	n = O_source->Formula_vector->nb_cols;

	len = 1;

	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				len, verbose_level - 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry "
				"before O_source->Formula_vector->V[i * n + j].copy_to" << endl;
	}
	O_source->Formula_vector->V[i * n + j].copy_to(
			&Formula_vector->V[0],
			verbose_level - 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry "
				"after O_source->Formula_vector->V[i * n + j].copy_to" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry done" << endl;
	}
}


void symbolic_object_builder::do_vector_entry(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_vector_entry" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->vector_entry_source);

	int i;

	i = Descr->vector_entry_i;

	if (i >= O_source->Formula_vector->len) {
		cout << "symbolic_object_builder::do_vector_entry out of range" << endl;
		cout << "i=" << i << endl;
		cout << "O_source->Formula_vector->len=" << O_source->Formula_vector->len << endl;
		exit(1);
	}

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

	int len;

	len = 1;

	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				len, verbose_level - 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_vector_entry "
				"before O_source->Formula_vector->V[i].copy_to" << endl;
	}
	O_source->Formula_vector->V[i].copy_to(
			&Formula_vector->V[0],
			verbose_level - 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_vector_entry "
				"after O_source->Formula_vector->V[i].copy_to" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_vector_entry done" << endl;
	}
}


void symbolic_object_builder::do_collect(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_collect" << endl;
	}

	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->collect_source);

	if (O_source->Formula_vector->len != 1) {
		cout << "symbolic_object_builder::do_collect "
				"input must be a singleton" << endl;
		exit(1);
	}



	int d;

	string variable;

	variable = Descr->collect_by;

	d = O_source->Formula_vector->V[0].highest_order_term(
			variable, 0 /*verbose_level*/);

	if (f_v) {
		cout << "symbolic_object_builder::do_collect "
				"highest_order_term = " << d << endl;
	}

	int len = d + 1;

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				len, 0 /*verbose_level*/);

	int i, j;//, j1;

	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "symbolic_object_builder::do_collect "
					"init_empty_plus_node " << i << endl;
		}
		Formula_vector->V[i].init_empty_plus_node(
				label, label /*label_tex*/,
				Descr->managed_variables,
				Fq, 0 /*verbose_level*/);
	}


	if (O_source->Formula_vector->V[0].tree->Root->type != operation_type_add) {
		cout << "symbolic_object_builder::do_collect "
				"root is not an addition node" << endl;
		//exit(1);

		j = O_source->Formula_vector->V[0].tree->Root->exponent_of_variable(
				variable, 0 /*verbose_level*/);

		expression_parser::syntax_tree_node *Output_node;

		Output_node = NEW_OBJECT(expression_parser::syntax_tree_node);

		O_source->Formula_vector->V[0].tree->Root->copy_to(
				Formula_vector->V[j].tree,
				Output_node,
				0 /*verbose_level*/);

		int j1;

		// destroy the appearances of the variable in the term:

		j1 = Output_node->exponent_of_variable_destructive(variable);

		if (j1 != j) {
			cout << "symbolic_object_builder::do_collect j1 != j" << endl;
			exit(1);
		}

		Formula_vector->V[j].tree->Root->append_node(Output_node, 0 /* verbose_level */);

	}
	else {
		for (i = 0; i < O_source->Formula_vector->V[0].tree->Root->nb_nodes; i++) {

			j = O_source->Formula_vector->V[0].tree->Root->Nodes[i]->exponent_of_variable(
					variable, 0 /*verbose_level*/);

			if (f_v) {
				cout << "symbolic_object_builder::do_collect "
						"node " << i << " / " << O_source->Formula_vector->V[0].tree->Root->nb_nodes
						<< " has degree " << j << " in " << variable
						<< endl;
			}


			expression_parser::syntax_tree_node *Output_node;

			Output_node = NEW_OBJECT(expression_parser::syntax_tree_node);

			O_source->Formula_vector->V[0].tree->Root->Nodes[i]->copy_to(
					Formula_vector->V[j].tree,
					Output_node,
					0 /*verbose_level*/);

			int j1;

			// destroy the appearances of the variable in the term:

			j1 = Output_node->exponent_of_variable_destructive(variable);

			if (j1 != j) {
				cout << "symbolic_object_builder::do_collect j1 != j" << endl;
				exit(1);
			}


			Formula_vector->V[j].tree->Root->append_node(Output_node, 0 /* verbose_level */);

		}
	}

	if (f_v) {
		for (i = 0; i < len; i++) {
			cout << "symbolic_object_builder::do_collect "
					"node " << i << " / " << len
					<< " has " << Formula_vector->V[i].tree->Root->nb_nodes << " terms"
					<< endl;

		}
	}

	if (f_v) {
		cout << "symbolic_object_builder::do_collect done" << endl;
	}
}




void symbolic_object_builder::do_CRC_encode(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_encode"
				<< " " << Descr->encode_CRC_block_length
				<< " " << Descr->encode_CRC_data_polynomial
				<< " " << Descr->encode_CRC_check_polynomial
				<< endl;
	}
	data_structures::symbolic_object_builder *O_data;
	data_structures::symbolic_object_builder *O_check;

	O_data = Get_symbol(Descr->encode_CRC_data_polynomial);

	O_check = Get_symbol(Descr->encode_CRC_check_polynomial);


	int nb_blocks;

	nb_blocks = O_data->Formula_vector->len;


	if (O_check->Formula_vector->len != 1) {
		cout << "symbolic_object_builder::do_CRC_encode "
				"check polynomial cannot be a vector" << endl;
		exit(1);
	}

	std::string variable1;

	if (O_check->Formula_vector->V[0].tree->variables.size() != 1) {
		cout << "symbolic_object_builder::do_CRC_encode "
				"check polynomial must have exactly one variable" << endl;
		exit(1);
	}

	variable1 = O_check->Formula_vector->V[0].tree->variables[0];
	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_encode "
				"variable from the CRC polynomial = " << variable1 << endl;
	}

	int *CRC_poly;
	int CRC_poly_nb_coeff;
	int CRC_poly_degree;

	O_check->Formula_vector->V[0].get_monopoly(
			variable1, CRC_poly, CRC_poly_nb_coeff, verbose_level - 1);

	if (CRC_poly_nb_coeff < 1) {
		cout << "symbolic_object_builder::do_CRC_encode "
				"CRC_poly_nb_coeff < 1" << endl;
		exit(1);
	}

	CRC_poly_degree = CRC_poly_nb_coeff - 1;

	if (CRC_poly[CRC_poly_degree] != 1) {
		cout << "symbolic_object_builder::do_CRC_encode "
				"CRC_poly is not monic" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_encode check poly: ";
		Int_vec_print(cout, CRC_poly, CRC_poly_nb_coeff);
		cout << endl;
	}


	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				nb_blocks, 0 /*verbose_level*/);

	int cnt;

	for (cnt = 0; cnt < nb_blocks; cnt++) {
		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode "
					"block " << cnt << " / " << nb_blocks << endl;
		}
		int *Data_poly;
		int Data_poly_nb_coeff;
		//int Data_poly_degree;
		std::string variable2;

		variable2 = O_data->Formula_vector->V[cnt].tree->variables[0];
		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode "
					"variable from the data polynomial is " << variable2 << endl;
		}

		O_data->Formula_vector->V[cnt].get_monopoly(
				variable2, Data_poly, Data_poly_nb_coeff, verbose_level - 1);


		if (Data_poly_nb_coeff < 1) {
			cout << "symbolic_object_builder::do_CRC_encode "
					"Data_poly_nb_coeff < 1" << endl;
			exit(1);
		}

		//Data_poly_degree = Data_poly_nb_coeff - 1;

		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " data polynomial read, "
							"nb_coeff = " << Data_poly_nb_coeff << endl;
		}
		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " data polynomial = ";
			Int_vec_print_fully(cout, Data_poly, Data_poly_nb_coeff);
			cout << endl;
			int h;
			for (h = 0; h < Data_poly_nb_coeff; h++) {
				cout << h << " : " << Data_poly[h] << endl;
			}
		}

		if (Data_poly_nb_coeff > Descr->encode_CRC_block_length) {
			cout << "symbolic_object_builder::do_CRC_encode "
					"data polynomial is too big" << endl;
			exit(1);
		}

		int N;
		int *Data;
		ring_theory::ring_theory_global RG;

		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " CRC_poly_degree = " << CRC_poly_degree << endl;
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
				<< nb_blocks << " encode_CRC_block_length = " << Descr->encode_CRC_block_length << endl;
		}

		N = Descr->encode_CRC_block_length;

		Data = NEW_int(N);

		Int_vec_zero(Data, N);

		Int_vec_copy(Data_poly, Data + CRC_poly_degree, Data_poly_nb_coeff);


		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " after shifting up, = ";
			Int_vec_print_fully(cout, Data, N);
			cout << endl;
			int h;
			for (h = 0; h < N; h++) {
				cout << h << " : " << Data[h] << endl;
			}
		}



		int *coeff_table_q;
		int coeff_table_q_len;
		int *coeff_table_r;
		int coeff_table_r_len;

		string prefix;

		prefix = "encoding_block_" + std::to_string(cnt);

		RG.polynomial_division_coefficient_table_with_report(
				prefix,
				Fq,
				Data, Data_poly_nb_coeff + CRC_poly_degree,
				CRC_poly, CRC_poly_nb_coeff,
				coeff_table_q, coeff_table_q_len,
				coeff_table_r, coeff_table_r_len,
				verbose_level - 1);


		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " remainder polynomial = ";
			Int_vec_print(cout, coeff_table_r, coeff_table_r_len);
			cout << endl;
			cout << "symbolic_object_builder::do_CRC_encode block " << cnt << " / "
					<< nb_blocks << " quotient polynomial = ";
			Int_vec_print(cout, coeff_table_q, coeff_table_q_len);
			cout << endl;
		}


		// copy the information into the upper part of Data:

		Int_vec_zero(Data, N);

		Int_vec_copy(Data_poly, Data + CRC_poly_degree, Data_poly_nb_coeff);

		// copy the negative of the remainder in the lower part to make a codeword:
		int h;

		for (h = 0; h < coeff_table_r_len; h++) {
			Data[h] = Fq->negate(coeff_table_r[h]);
		}


		string label;
		string managed_variables;


		label = "block" + std::to_string(cnt) + "_encoded";

		Formula_vector->V[cnt].init_formula_monopoly(
				label, label,
				Fq,
				managed_variables,
				variable2,
				Data, N,
				verbose_level - 1);

		FREE_int(coeff_table_q);
		FREE_int(coeff_table_r);
		FREE_int(Data_poly);
		FREE_int(Data);

	}
	FREE_int(CRC_poly);



	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_encode done" << endl;
	}
}



void symbolic_object_builder::do_CRC_decode(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_decode"
				<< " " << Descr->decode_CRC_block_length
				<< " " << Descr->decode_CRC_data_polynomial
				<< " " << Descr->decode_CRC_check_polynomial
				<< endl;
	}
	data_structures::symbolic_object_builder *O_data;
	data_structures::symbolic_object_builder *O_check;

	O_data = Get_symbol(Descr->decode_CRC_data_polynomial);

	O_check = Get_symbol(Descr->decode_CRC_check_polynomial);


	int nb_blocks;

	nb_blocks = O_data->Formula_vector->len;


	if (O_check->Formula_vector->len != 1) {
		cout << "symbolic_object_builder::do_CRC_decode "
				"check polynomial cannot be a vector" << endl;
		exit(1);
	}

	std::string variable1;

	if (O_check->Formula_vector->V[0].tree->variables.size() != 1) {
		cout << "symbolic_object_builder::do_CRC_decode "
				"check polynomial must have exactly one variable" << endl;
		exit(1);
	}

	variable1 = O_check->Formula_vector->V[0].tree->variables[0];
	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_decode "
				"variable from the CRC polynomial = " << variable1 << endl;
	}

	int *CRC_poly;
	int CRC_poly_nb_coeff;
	int CRC_poly_degree;

	O_check->Formula_vector->V[0].get_monopoly(
			variable1, CRC_poly, CRC_poly_nb_coeff, verbose_level);

	if (CRC_poly_nb_coeff < 1) {
		cout << "symbolic_object_builder::do_CRC_decode "
				"CRC_poly_nb_coeff < 1" << endl;
		exit(1);
	}

	CRC_poly_degree = CRC_poly_nb_coeff - 1;

	if (CRC_poly[CRC_poly_degree] != 1) {
		cout << "symbolic_object_builder::do_CRC_decode "
				"CRC_poly is not monic" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_decode check poly: ";
		Int_vec_print(cout, CRC_poly, CRC_poly_nb_coeff);
		cout << endl;
	}


	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				nb_blocks, 0 /*verbose_level*/);

	int cnt;

	for (cnt = 0; cnt < nb_blocks; cnt++) {
		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_decode "
					"block " << cnt << " / " << nb_blocks << endl;
		}
		int *Data_poly;
		int Data_poly_nb_coeff;
		//int Data_poly_degree;
		std::string variable2;

		variable2 = O_data->Formula_vector->V[cnt].tree->variables[0];
		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_decode "
					"variable from the data polynomial is " << variable2 << endl;
		}

		O_data->Formula_vector->V[cnt].get_monopoly(
				variable2, Data_poly, Data_poly_nb_coeff, verbose_level - 1);


		if (Data_poly_nb_coeff < 1) {
			cout << "symbolic_object_builder::do_CRC_decode "
					"Data_poly_nb_coeff < 1" << endl;
			exit(1);
		}

		//Data_poly_degree = Data_poly_nb_coeff - 1;

		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
					<< nb_blocks << " data polynomial read, "
							"nb_coeff = " << Data_poly_nb_coeff << endl;
		}
		if (false) {
			cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
					<< nb_blocks << " data polynomial = ";
			Int_vec_print(cout, Data_poly, Data_poly_nb_coeff);
			cout << endl;
			int h;
			for (h = 0; h < Data_poly_nb_coeff; h++) {
				cout << h << " : " << Data_poly[h] << endl;
			}
		}

		if (Data_poly_nb_coeff > Descr->decode_CRC_block_length) {
			cout << "symbolic_object_builder::do_CRC_decode "
					"data polynomial is too big" << endl;
			exit(1);
		}

		int N;
		int *Data;
		ring_theory::ring_theory_global RG;

		N = Descr->decode_CRC_block_length;
		//N = Descr->decode_CRC_block_length + CRC_poly_degree;

		Data = NEW_int(N);

		Int_vec_zero(Data, N);

		Int_vec_copy(Data_poly, Data, Data_poly_nb_coeff);


		int *coeff_table_q;
		int coeff_table_q_len;
		int *coeff_table_r;
		int coeff_table_r_len;

		string prefix;

		prefix = "decoding_block_" + std::to_string(cnt);

		RG.polynomial_division_coefficient_table_with_report(
				prefix,
				Fq,
				Data, Data_poly_nb_coeff,
				CRC_poly, CRC_poly_nb_coeff,
				coeff_table_q, coeff_table_q_len,
				coeff_table_r, coeff_table_r_len,
				verbose_level - 1);


		if (f_v) {
			cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
					<< nb_blocks << " remainder polynomial = ";
			Int_vec_print(cout, coeff_table_r, coeff_table_r_len);
			cout << endl;
			cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
					<< nb_blocks << " quotient polynomial = ";
			Int_vec_print(cout, coeff_table_q, coeff_table_q_len);
			cout << endl;
		}



		if (Int_vec_is_zero(coeff_table_r, coeff_table_r_len)) {
			if (f_v) {
				cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
						<< nb_blocks << " block is OK, because remainder is zero" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "symbolic_object_builder::do_CRC_decode block " << cnt << " / "
						<< nb_blocks << " block is not OK, because remainder is nonzero" << endl;
			}
		}
		// copy the remainder polynomial into data, padded with zeros:

		Int_vec_zero(Data, N);

		Int_vec_copy(coeff_table_r, Data, coeff_table_r_len);


		string label;
		string managed_variables;


		label = "block" + std::to_string(cnt) + "_decoded";

		Formula_vector->V[cnt].init_formula_monopoly(
				label, label,
				Fq,
				managed_variables,
				variable2,
				Data, coeff_table_r_len,
				verbose_level - 1);

		FREE_int(coeff_table_q);
		FREE_int(coeff_table_r);
		FREE_int(Data_poly);
		FREE_int(Data);

	}
	FREE_int(CRC_poly);



	if (f_v) {
		cout << "symbolic_object_builder::do_CRC_decode done" << endl;
	}
}


void symbolic_object_builder::do_submatrix(
		symbolic_object_builder_description *Descr,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::do_submatrix"
				<< " " << Descr->submatrix_source
				<< " " << Descr->submatrix_row_first
				<< " " << Descr->submatrix_nb_rows
				<< " " << Descr->submatrix_col_first
				<< " " << Descr->submatrix_nb_cols
				<< endl;
	}
	data_structures::symbolic_object_builder *O_source;

	O_source = Get_symbol(Descr->submatrix_source);

	if (!O_source->Formula_vector->f_matrix) {
		cout << "symbolic_object_builder::do_submatrix we expect the input object to be a matrix" << endl;
		exit(1);
	}
	if (O_source->Formula_vector->nb_rows < Descr->submatrix_row_first + Descr->submatrix_nb_rows) {
		cout << "symbolic_object_builder::do_submatrix the input matrix does not have sufficiently many rows" << endl;
		exit(1);
	}
	if (O_source->Formula_vector->nb_cols < Descr->submatrix_col_first + Descr->submatrix_nb_cols) {
		cout << "symbolic_object_builder::do_submatrix the input matrix does not have sufficiently many columns" << endl;
		exit(1);
	}

	int N, i, j, i0, j0;

	N = Descr->submatrix_nb_rows * Descr->submatrix_nb_cols;

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	Formula_vector->init_and_allocate(
				label, label,
				true,
				Descr->managed_variables,
				N, 0 /*verbose_level*/);


	Formula_vector->f_matrix = true;
	Formula_vector->nb_rows = Descr->submatrix_nb_rows;
	Formula_vector->nb_cols = Descr->submatrix_nb_cols;


	if (f_v) {
		cout << "symbolic_object_builder::do_submatrix "
				"extracting submatrix" << endl;
	}

	for (i = 0; i < Descr->submatrix_nb_rows; i++) {

		i0 = Descr->submatrix_row_first + i;

		for (j = 0; j < Descr->submatrix_nb_cols; j++) {

			j0 = Descr->submatrix_col_first + j;

			O_source->Formula_vector->V[i0 * O_source->Formula_vector->nb_cols + j0].copy_to(
					&Formula_vector->V[i * Descr->submatrix_nb_cols + j],
					verbose_level - 1);


		}
	}
	if (f_v) {
		cout << "symbolic_object_builder::do_submatrix "
				"extracting submatrix finished" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::do_submatrix done" << endl;
	}
}




void symbolic_object_builder::multiply_terms(
		expression_parser::formula **terms,
		int n,
		int &stage_counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms" << endl;
	}

	void **Terms;


	Terms = new pvoid [n];

	int i;

	for (i = 0; i < n; i++) {
		if (!terms[i]->f_Sajeeb) {
			cout << "The formula must be of type Sajeeb" << endl;
			exit(1);
		}
		Terms[i] = terms[i]->Expression_parser_sajeeb;
	}

	l1_interfaces::expression_parser_sajeeb E;

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"before E.multiply" << endl;
	}
	E.multiply(
			(l1_interfaces::expression_parser_sajeeb **) Terms,
			n,
			stage_counter,
			verbose_level - 1);
	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"after E.multiply" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"before delete [] Terms" << endl;
	}

	delete [] Terms;

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"after delete [] Terms" << endl;
	}

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms done" << endl;
	}
}



}}}



