/*
 * l1_interfaces.h
 *
 *  Created on: Mar 18, 2023
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_




namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


// #############################################################################
// Eigen_interface.cpp:
// #############################################################################

void orbiter_eigenvalues(int *Mtx, int nb_points, double *E, int verbose_level);


// #############################################################################
// expression_parser_sajeeb.cpp:
// #############################################################################

//! interface to Sajeeb's expression parser

class expression_parser_sajeeb {
public:

	expression_parser::formula *Formula;

	void *private_data;

	expression_parser_sajeeb();
	~expression_parser_sajeeb();
	void init_formula(
			expression_parser::formula *Formula,
			int verbose_level);
	void get_subtrees(
			ring_theory::homogeneous_polynomial_domain *Poly,
			int verbose_level);
	void evaluate(
			ring_theory::homogeneous_polynomial_domain *Poly,
			std::map<std::string, std::string> &symbol_table, int *Values,
			int verbose_level);


};



// #############################################################################
// interface_gap_low.cpp:
// #############################################################################

//! interface to GAP at the foundation level

class interface_gap_low {
public:

	interface_gap_low();
	~interface_gap_low();
	void fining_set_stabilizer_in_collineation_group(
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void collineation_set_stabilizer(
			std::ostream &ost,
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			int verbose_level);
	void write_matrix(
			std::ostream &ost,
			field_theory::finite_field *F,
			int *Mtx, int d,
			int verbose_level);
	void write_element_of_finite_field(
			std::ostream &ost,
			field_theory::finite_field *F, int a);

};

// #############################################################################
// interface_magma_low.cpp:
// #############################################################################

//! interface to magma at the foundation level

class interface_magma_low {
public:

	interface_magma_low();
	~interface_magma_low();
	void magma_set_stabilizer_in_collineation_group(
			field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void export_colored_graph_to_magma(
			graph_theory::colored_graph *Gamma,
			std::string &fname, int verbose_level);

};


// #############################################################################
// latex_interface.cpp
// #############################################################################


//! interface to create latex output files



class latex_interface {
public:
	latex_interface();
	~latex_interface();
	void head_easy(std::ostream& ost);
	void head_easy_with_extras_in_the_praeamble(
			std::ostream& ost, std::string &extras);
	void head_easy_sideways(std::ostream& ost);
	void head(std::ostream& ost, int f_book, int f_title,
			std::string &title, std::string &author,
		int f_toc, int f_landscape, int f_12pt,
		int f_enlarged_page, int f_pagenumbers,
		std::string &extras_for_preamble);
	void foot(std::ostream& ost);

	// two functions from DISCRETA1:

	void incma_latex_with_text_labels(
			std::ostream &fp,
			graphics::draw_incidence_structure_description *Descr,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *incma,
		int f_labelling_points, std::string *point_labels,
		int f_labelling_blocks, std::string *block_labels,
		int verbose_level);
	void incma_latex(std::ostream &fp,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *incma,
		int verbose_level);
	void incma_latex_with_labels(std::ostream &fp,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *row_labels_int,
		int *col_labels_int,
		int *incma,
		int verbose_level);
	void print_01_matrix_tex(
			std::ostream &ost, int *p, int m, int n);
	void print_integer_matrix_tex(
			std::ostream &ost, int *p, int m, int n);
	void print_lint_matrix_tex(
			std::ostream &ost,
		long int *p, int m, int n);
	void print_longinteger_matrix_tex(
			std::ostream &ost,
			ring_theory::longinteger_object *p, int m, int n);
	void print_integer_matrix_with_labels(
			std::ostream &ost, int *p,
		int m, int n, int *row_labels, int *col_labels, int f_tex);
	void print_lint_matrix_with_labels(
			std::ostream &ost,
		long int *p, int m, int n,
		long int *row_labels, long int *col_labels,
		int f_tex);
	void print_integer_matrix_with_standard_labels(
			std::ostream &ost,
		int *p, int m, int n, int f_tex);
	void print_lint_matrix_with_standard_labels(
			std::ostream &ost,
		long int *p, int m, int n, int f_tex);
	void print_integer_matrix_with_standard_labels_and_offset(
			std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset, int f_tex);
	void print_lint_matrix_with_standard_labels_and_offset(
			std::ostream &ost,
		long int *p, int m, int n, int m_offset, int n_offset, int f_tex);
	void print_integer_matrix_tex_block_by_block(
			std::ostream &ost,
		int *p, int m, int n, int block_width);
	void print_integer_matrix_with_standard_labels_and_offset_text(
			std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset);
	void print_lint_matrix_with_standard_labels_and_offset_text(
		std::ostream &ost, long int *p, int m, int n,
		int m_offset, int n_offset);
	void print_integer_matrix_with_standard_labels_and_offset_tex(
			std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset);
	void print_lint_matrix_with_standard_labels_and_offset_tex(
		std::ostream &ost, long int *p, int m, int n,
		int m_offset, int n_offset);
	void print_big_integer_matrix_tex(
			std::ostream &ost, int *p, int m, int n);
	void int_vec_print_as_matrix(std::ostream &ost,
		int *v, int len, int width, int f_tex);
	void lint_vec_print_as_matrix(std::ostream &ost,
		long int *v, int len, int width, int f_tex);
	void int_matrix_print_with_labels_and_partition(
			std::ostream &ost,
		int *p, int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(int *p, int m, int n,
			int i, int j, int val, std::string &output, void *data),
		void *data,
		int f_tex);
	void lint_matrix_print_with_labels_and_partition(
			std::ostream &ost,
		long int *p, int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(long int *p, int m, int n,
			int i, int j, int val, std::string &output, void *data),
		void *data,
		int f_tex);
	void int_matrix_print_tex(
			std::ostream &ost, int *p, int m, int n);
	void lint_matrix_print_tex(
			std::ostream &ost, long int *p, int m, int n);
	void print_cycle_tex_with_special_point_labels(
			std::ostream &ost, int *pts, int nb_pts,
			void (*point_label)(std::stringstream &sstr,
					int pt, void *data),
			void *point_label_data);
	void int_set_print_tex(
			std::ostream &ost, int *v, int len);
	void lint_set_print_tex(
			std::ostream &ost, long int *v, int len);
	void lint_set_print_tex_text_mode(
			std::ostream &ost, long int *v, int len);
	void print_type_vector_tex(
			std::ostream &ost, int *v, int len);
	void int_set_print_masked_tex(
			std::ostream &ost,
		int *v, int len, const char *mask_begin, const char *mask_end);
	void lint_set_print_masked_tex(
			std::ostream &ost,
		long int *v, int len,
		const char *mask_begin,
		const char *mask_end);
	void int_set_print_tex_for_inline_text(
			std::ostream &ost,
			int *v, int len);
	void lint_set_print_tex_for_inline_text(
			std::ostream &ost,
			long int *v, int len);
	void latexable_string(
			std::stringstream &str,
			const char *p, int max_len, int line_skip);
	void print_row_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		long int *row_class_size, int nb_row_classes,
		long int *col_class_size, int nb_col_classes,
		long int *row_scheme);
	void print_column_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		long int *row_class_size, int nb_row_classes,
		long int *col_class_size, int nb_col_classes,
		long int *col_scheme);

};



// #############################################################################
// nauty_interface.cpp
// #############################################################################

//! low-level interface to the graph canonization software nauty

class nauty_interface {

public:

	void nauty_interface_graph_bitvec(int v,
			data_structures::bitvector *Bitvec,
		int *partition,
		data_structures::nauty_output *NO,
		int verbose_level);
	void nauty_interface_graph_int(int v, int *Adj,
		int *partition,
		data_structures::nauty_output *NO,
		int verbose_level);
	void nauty_interface_matrix_int(
		combinatorics::encoded_combinatorial_object *Enc,
		data_structures::nauty_output *NO,
		int verbose_level);


};



}}}




#endif /* SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_ */
