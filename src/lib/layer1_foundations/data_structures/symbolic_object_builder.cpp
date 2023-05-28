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
	string managed_variables;


	if (Descr->f_field) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-field " << Descr->field_label << endl;
		}
		Fq = Get_finite_field(Descr->field_label);
	}

	if (Descr->f_ring) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-ring " << Descr->ring_label << endl;
		}
		Ring = Get_ring(Descr->ring_label);
		int i;
		for (i = 0; i < Ring->nb_variables; i++) {
			managed_variables += Ring->get_symbol(i);
			if (i < Ring->nb_variables - 1) {
				managed_variables += ",";
			}
		}
	}

	if (Descr->f_ring && !Descr->f_field) {
		Fq = Ring->get_F();
	}

	if (!Descr->f_ring && !Descr->f_field) {
		cout << "symbolic_object_builder::init please use either "
				"-ring <ring> or -field <field> "
				"to specify the domain of coefficients" << endl;
		exit(1);
	}

	if (Descr->f_text) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-text" << Descr->text_txt << endl;
		}


		Formula_vector = NEW_OBJECT(expression_parser::formula_vector);
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"before Formula_vector->init_from_text" << endl;
		}
		Formula_vector->init_from_text(
				label /*Descr->label_txt*/,
				Descr->label_tex,
				Descr->text_txt,
				Fq,
				managed_variables,
				Descr->f_matrix, Descr->nb_rows,
				verbose_level);
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"after Formula_vector->init_from_text" << endl;
		}

	}
	else if (Descr->f_determinant) {
		if (f_v) {
			cout << "symbolic_object_builder::init -determinant"
					<< " " << Descr->determinant_source
					<< endl;
		}

		do_determinant(
				Descr,
				label,
				verbose_level);


	}
	else if (Descr->f_characteristic_polynomial) {
		if (f_v) {
			cout << "symbolic_object_builder::init -characteristic_polynomial"
					<< " " << Descr->characteristic_polynomial_variable
					<< " " << Descr->characteristic_polynomial_source
					<< endl;
		}

		do_characteristic_polynomial(
				Descr,
				label,
				verbose_level);


	}

	else if (Descr->f_substitute) {
		if (f_v) {
			cout << "symbolic_object_builder::init -substitute"
					<< " " << Descr->substitute_variables
					<< " " << Descr->substitute_target
					<< " " << Descr->substitute_source
					<< endl;
		}

		do_substitute(
					Descr,
					label,
					verbose_level);


	}

	else if (Descr->f_simplify) {
		if (f_v) {
			cout << "symbolic_object_builder::init -simplify"
					<< " " << Descr->simplify_source
					<< endl;
		}


		do_simplify(
					Descr,
					label,
					verbose_level);

		if (f_v) {
			cout << "symbolic_object_builder::init -simplify finished" << endl;
		}

	}
	else if (Descr->f_expand) {
		if (f_v) {
			cout << "symbolic_object_builder::init -expand"
					<< " " << Descr->expand_source
					<< endl;
		}


		do_expand(
					Descr,
					label,
					verbose_level);

		if (f_v) {
			cout << "symbolic_object_builder::init -expand finished" << endl;
		}

	}
	else if (Descr->f_right_nullspace) {
		if (f_v) {
			cout << "symbolic_object_builder::init -right_nullspace"
					<< " " << Descr->right_nullspace_source
					<< endl;
		}

		do_right_nullspace(
					Descr,
					label,
					verbose_level);

	}
	else if (Descr->f_minor) {
		if (f_v) {
			cout << "symbolic_object_builder::init -minor"
					<< " " << Descr->minor_source
					<< " " << Descr->minor_i
					<< " " << Descr->minor_j
					<< endl;
		}

		do_minor(
					Descr,
					Descr->minor_i, Descr->minor_j,
					label,
					verbose_level);

	}

	else if (Descr->f_symbolic_nullspace) {
		if (f_v) {
			cout << "symbolic_object_builder::init -symbolic_nullspace"
					<< " " << Descr->symbolic_nullspace_source
					<< endl;
		}

		do_symbolic_nullspace(
					Descr,
					label,
					verbose_level);

	}
	else if (Descr->f_stack_matrices_vertically) {
		if (f_v) {
			cout << "symbolic_object_builder::init -stack_matrices_vertically"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level);

	}
	else if (Descr->f_stack_matrices_horizontally) {
		if (f_v) {
			cout << "symbolic_object_builder::init -stack_matrices_horizontally"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level);
	}
	else if (Descr->f_stack_matrices_z_shape) {
		if (f_v) {
			cout << "symbolic_object_builder::init -stack_matrices_z_shape"
					<< " " << Descr->stack_matrices_label
					<< endl;
		}
		do_stack(
					Descr,
					verbose_level);
	}

	else if (Descr->f_multiply_2x2_from_the_left) {
		if (f_v) {
			cout << "symbolic_object_builder::init -multiply_2x2_from_the_left"
					<< " " << Descr->multiply_2x2_from_the_left_source
					<< " " << Descr->multiply_2x2_from_the_left_A2
					<< " " << Descr->multiply_2x2_from_the_left_i
					<< " " << Descr->multiply_2x2_from_the_left_j
					<< endl;
		}
		do_multiply_2x2_from_the_left(
					Descr,
					label,
					verbose_level);
	}

	else if (Descr->f_matrix_entry) {
		if (f_v) {
			cout << "symbolic_object_builder::init -matrix_entry"
					<< " " << Descr->matrix_entry_source
					<< endl;
		}
		do_matrix_entry(
					Descr,
					label,
					verbose_level);
	}

	else if (Descr->f_collect) {
		if (f_v) {
			cout << "symbolic_object_builder::init -collect"
					<< " " << Descr->collect_source
					<< " " << Descr->collect_by
					<< endl;
		}
		do_collect(
					Descr,
					label,
					verbose_level);
	}


	if (Descr->f_file) {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"-file" << Descr->file_name << endl;
		}


	}

	if (f_v) {
		cout << "symbolic_object_builder::init "
				"before simplifying the formula vector" << endl;
	}

	if (!Descr->f_do_not_simplify) {
		expression_parser::formula_vector *old;

		old = Formula_vector;

		Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

		if (f_v) {
			cout << "symbolic_object_builder::init "
					"before Formula_vector->simplify label=" << label << endl;
		}

		Formula_vector->simplify(
				old,
				Fq,
				label,
				label,
				verbose_level);

		FREE_OBJECT(old);
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"after simplifying the formula vector" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "symbolic_object_builder::init "
					"skipping simplification because of -do_not_simplify" << endl;

		}
	}

	if (f_v) {
		cout << "symbolic_object_builder::init "
				"symbolic object vector of length "
				<< Formula_vector->len << endl;
		//Lint_vec_print(cout, v, len);
		cout << endl;

		//cout << "ir tree:" << endl;
		//print_Sajeeb(cout);


		cout << "final tree:" << endl;
		Formula_vector->print_formula(cout, verbose_level);

		Formula_vector->print_latex(cout);

		//latex_split(std::string &name, int split_level, int split_mod, int verbose_level)

		Formula_vector->latex_tree(verbose_level);
	}


	if (f_v) {
		cout << "symbolic_object_builder::init done" << endl;
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
			verbose_level);
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
			verbose_level);
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
			verbose_level);
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
			verbose_level);
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
			Descr->f_write_trees_during_expand,
			verbose_level);
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
			verbose_level);
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
				"before Formula_vector->minor" << endl;
	}
	Formula_vector->minor(
			O_source->Formula_vector,
			Fq,
			minor_i, minor_j,
			label, label,
			verbose_level);
	if (f_v) {
		cout << "symbolic_object_builder::do_minor "
				"after Formula_vector->minor" << endl;
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
			verbose_level);
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
			verbose_level);

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
					len, verbose_level);

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
						&Formula_vector->V[n * nb_c + j], verbose_level);
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

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);

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

	Formula_vector->init_and_allocate(
				label, label,
				len, verbose_level);

	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"before Formula_vector->multiply_2by2_from_the_left" << endl;
	}
	Formula_vector->multiply_2by2_from_the_left(
			O_source->Formula_vector,
			O_A2->Formula_vector,
			i, j,
			O_source->Fq,
			label, label,
			verbose_level);
	if (f_v) {
		cout << "symbolic_object_builder::do_multiply_2x2_from_the_left "
				"after Formula_vector->multiply_2by2_from_the_left" << endl;
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
	int m, n;

	m = O_source->Formula_vector->nb_rows;
	n = O_source->Formula_vector->nb_cols;

	len = 1;

	Formula_vector->init_and_allocate(
				label, label,
				len, verbose_level);

	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry "
				"before O_source->Formula_vector->V[i * n + j].copy_to" << endl;
	}
	O_source->Formula_vector->V[i * n + j].copy_to(
			&Formula_vector->V[0],
			verbose_level);

	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry "
				"after O_source->Formula_vector->V[i * n + j].copy_to" << endl;
	}



	if (f_v) {
		cout << "symbolic_object_builder::do_matrix_entry done" << endl;
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
			variable, verbose_level);

	if (f_v) {
		cout << "symbolic_object_builder::do_collect highest_order_term = " << d << endl;
	}

	int len = d + 1;

	Formula_vector = NEW_OBJECT(expression_parser::formula_vector);


	Formula_vector->init_and_allocate(
				label, label,
				len, verbose_level);

	int i, j, j1;

	for (i = 0; i < len; i++) {
		Formula_vector->V[i].init_empty_formula(
				label, label /*label_tex*/, Fq, verbose_level);
	}


	if (O_source->Formula_vector->V[0].tree->Root->type != operation_type_add) {
		cout << "symbolic_object_builder::do_collect "
				"root node must be addition node" << endl;
		exit(1);
	}
	for (i = 0; i < O_source->Formula_vector->V[0].tree->Root->nb_nodes; i++) {

		j = O_source->Formula_vector->V[0].tree->Root->Nodes[i]->exponent_of_variable(
				variable, verbose_level);


		expression_parser::syntax_tree_node *Output_node;

		Output_node = NEW_OBJECT(expression_parser::syntax_tree_node);

		O_source->Formula_vector->V[0].tree->Root->Nodes[i]->copy_to(
				Formula_vector->V[j].tree,
				Output_node,
				verbose_level);

		j1 = Output_node->exponent_of_variable_destructive(variable);


		Formula_vector->V[j].tree->Root->Nodes[Formula_vector->V[j].tree->Root->nb_nodes++] = Output_node;

	}

	if (f_v) {
		cout << "symbolic_object_builder::do_collect done" << endl;
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
			verbose_level);
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



