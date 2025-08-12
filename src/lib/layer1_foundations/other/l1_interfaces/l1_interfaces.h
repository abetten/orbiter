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
namespace other {
namespace l1_interfaces {

// #############################################################################
// dirk_kaempfer.cpp:
// #############################################################################

//! interface to the refinement program for TDO by Dirk Kaempfer



int Dirk_Kaempfer_main();





// #############################################################################
// easy_BMP_interface.cpp:
// #############################################################################

//! interface to the easy_BMP library for working with bmp files


class easy_BMP_interface {
public:

	easy_BMP_interface();
	~easy_BMP_interface();
	void draw_bitmap(
			graphics::draw_bitmap_control *C, int verbose_level);
	void random_noise_in_bitmap_file(
			std::string fname_input,
			std::string fname_output,
			int probability_numerator,
			int probability_denominator,
			int verbose_level);
	void random_noise_in_bitmap_file_burst(
			std::string fname_input,
			std::string fname_output,
			int probability_numerator,
			int probability_denominator,
			int burst_length_max,
			int verbose_level);

};

// #############################################################################
// eigen_interface.cpp:
// #############################################################################

//! interface to Eigen

class eigen_interface {
public:

	eigen_interface();
	~eigen_interface();
	void orbiter_eigenvalues(
			int *Mtx, int nb_points, double *E, int verbose_level);
};


// #############################################################################
// expression_parser_sajeeb.cpp:
// #############################################################################

//! interface to Sajeeb's expression parser

class expression_parser_sajeeb {
public:

	algebra::expression_parser::formula *Formula;

	void *private_data;

	expression_parser_sajeeb();
	~expression_parser_sajeeb();
	void init_formula(
			algebra::expression_parser::formula *Formula,
			int verbose_level);
	void print(
			std::ostream &ost,
			int verbose_level);
	std::string string_representation();
	void get_subtrees(
			algebra::ring_theory::homogeneous_polynomial_domain *Poly,
			int verbose_level);
	void evaluate(
			algebra::ring_theory::homogeneous_polynomial_domain *Poly,
			std::map<std::string, std::string> &symbol_table, int *Values,
			int verbose_level);
	void multiply(
			expression_parser_sajeeb **terms,
			int n,
			int &stage_counter,
			int verbose_level);
	void convert_to_orbiter(
			algebra::expression_parser::syntax_tree *&Tree,
			algebra::field_theory::finite_field *F,
			int f_has_managed_variables,
			std::string &managed_variables,
			int verbose_level);


};



// #############################################################################
// gnuplot_interface.cpp:
// #############################################################################

//! interface to gnuplot

class gnuplot_interface {
public:

	gnuplot_interface();
	~gnuplot_interface();
	void gnuplot(
			std::string &data_file_csv,
			std::string &title,
			std::string &label_x,
			std::string &label_y,
			int verbose_level);

};


// #############################################################################
// interface_gap_low.cpp:
// #############################################################################

//! interface to GAP

class interface_gap_low {
public:

	interface_gap_low();
	~interface_gap_low();
	void fining_set_stabilizer_in_collineation_group(
			algebra::field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void collineation_set_stabilizer(
			std::ostream &ost,
			algebra::field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			int verbose_level);
	void write_matrix(
			std::ostream &ost,
			algebra::field_theory::finite_field *F,
			int *Mtx, int d,
			int verbose_level);
	void write_element_of_finite_field(
			std::ostream &ost,
			algebra::field_theory::finite_field *F, int a);

};

// #############################################################################
// interface_magma_low.cpp:
// #############################################################################

//! interface to magma

class interface_magma_low {
public:

	interface_magma_low();
	~interface_magma_low();
	void magma_set_stabilizer_in_collineation_group(
			algebra::field_theory::finite_field *F,
			int d, long int *Pts, int nb_pts,
			std::string &fname,
			int verbose_level);
	void export_colored_graph_to_magma(
			combinatorics::graph_theory::colored_graph *Gamma,
			std::string &fname, int verbose_level);
	void export_linear_code(
			std::string &fname,
			algebra::field_theory::finite_field *F,
			int *genma, int n, int k,
			int verbose_level);
	void read_permutation_group(
			std::string &fname,
		int degree, int *&gens, int &nb_gens, int &go,
		int verbose_level);
	void run_magma_file(
			std::string &fname, int verbose_level);

};


// #############################################################################
// latex_interface.cpp
// #############################################################################


//! interface to latex to create latex source files



class latex_interface {
public:
	latex_interface();
	~latex_interface();
	void head_easy(
			std::ostream& ost);
	void head_easy_and_enlarged(
			std::ostream& ost);
	void head_easy_with_extras_in_the_praeamble(
			std::ostream& ost, std::string &extras);
	void head_easy_sideways(
			std::ostream& ost);
	void head(
			std::ostream& ost, int f_book, int f_title,
			std::string &title, std::string &author,
		int f_toc, int f_landscape, int f_12pt,
		int f_enlarged_page, int f_pagenumbers,
		std::string &extras_for_preamble);
	void foot(
			std::ostream& ost);

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
	void incma_latex(
			std::ostream &fp,
			graphics::draw_incidence_structure_description *Descr,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *incma,
		int verbose_level);
	void incma_latex_with_labels(
			std::ostream &fp,
			graphics::draw_incidence_structure_description *Descr,
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
			algebra::ring_theory::longinteger_object *p, int m, int n);
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
	void int_vec_print_as_matrix(
			std::ostream &ost,
		int *v, int len, int width, int f_tex);
	void lint_vec_print_as_matrix(
			std::ostream &ost,
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
	void print_table_of_strings_with_headers(
			std::ostream &ost, std::string *headers,
			std::string *Table, int m, int n);
	void print_table_of_strings_with_headers_rc(
			std::ostream &ost, std::string *headers_row,
			std::string *headers_col,
			std::string *Table, int m, int n);
	void print_table_of_strings(
			std::ostream &ost, std::string *Table, int m, int n);
	void print_tabular_of_strings(
			std::ostream &ost, std::string *Table, int m, int n);
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
		int *v, int len,
		const char *mask_begin,
		const char *mask_end);
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
	void report_matrix(
			std::string &fname,
			std::string &title,
			std::string &author,
			std::string &extra_praeamble,
		int *M, int nb_rows, int nb_cols);
	void report_matrix_longinteger(
			std::string &fname,
			std::string &title,
			std::string &author,
			std::string &extra_praeamble,
			algebra::ring_theory::longinteger_object *M,
			int nb_rows, int nb_cols,
			int verbose_level);
	void print_decomposition_matrix(
			std::ostream &ost,
			int m, int n,
			std::string &top_left_entry,
			std::string *cols_labels,
			std::string *row_labels,
			std::string *entries,
			int f_enter_math_mode);
	void print_vector_vertically_with_label(
			std::ostream &ost,
			std::string &label,
			std::vector<std::string> &v,
			int f_brackets,
			int f_enter_math_mode);
	void print_vector_horizontally_with_label(
			std::ostream &ost,
			std::string &label,
			std::vector<std::string> &v,
			int f_brackets,
			int f_enter_math_mode);

};


// #############################################################################
// nauty_interface_control.cpp:
// #############################################################################

//! Input options for Nauty


class nauty_interface_control {

public:

	// nauty_control.csv


	int f_save_nauty_input_graphs;
	std::string save_nauty_input_graphs_prefix;

	int f_save_orbit_of_equations;
	std::string save_orbit_of_equations_prefix;

	int f_partition;
	std::string partition_text;

	int f_show_canonical_form;

	int f_reduce_memory_footprint;
	int reduce_memory_footprint_level;


	nauty_interface_control();
	~nauty_interface_control();
	void init(
			int f_save_nauty_input_graphs,
			std::string &save_nauty_input_graphs_prefix,
			int verbose_level);
	int parse_arguments(
			int argc, std::string *argv,
			int verbose_level);
	void print();

};



// #############################################################################
// nauty_interface_for_combo.cpp
// #############################################################################


//! interface to nauty, assuming object_with_canonical_form is present



class nauty_interface_for_combo {
public:
	nauty_interface_for_combo();
	~nauty_interface_for_combo();
	void run_nauty_for_combo(
			combinatorics::canonical_form_classification::any_combinatorial_object *Any_combo,
			int f_compute_canonical_form,
			other::l1_interfaces::nauty_interface_control *Nauty_control,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			combinatorics::canonical_form_classification::encoded_combinatorial_object *&Enc,
			int verbose_level);
	// called from
	// object_with_canonical_form::run_nauty_basic
	// object_with_canonical_form::canonical_labeling
	// classification_of_objects::process_object
	// nauty_interface_with_group::set_stabilizer_of_object
	// classify_using_canonical_forms::find_object
	void run_nauty_for_combo_basic(
			combinatorics::canonical_form_classification::any_combinatorial_object *Any_combo,
			other::l1_interfaces::nauty_interface_control *Nauty_control,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);

};


// #############################################################################
// nauty_interface.cpp
// #############################################################################

//! interface to the graph canonization software nauty

class nauty_interface {

public:

	nauty_interface();
	~nauty_interface();

	void nauty_interface_graph_bitvec(
			int v,
			data_structures::bitvector *Bitvec,
		int *partition,
		l1_interfaces::nauty_output *NO,
		int verbose_level);
	void nauty_interface_graph_int(
			int v, int *Adj,
		int *partition,
		l1_interfaces::nauty_output *NO,
		int verbose_level);
	void Levi_graph(
			combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc,
		l1_interfaces::nauty_output *NO,
		int verbose_level);


};


// #############################################################################
// nauty_output.cpp:
// #############################################################################


//! output data created by a run of nauty

class nauty_output {
public:


	int N;
	int invariant_set_start;
	int invariant_set_size;

	int *Aut;  // [Aut_counter * N]
	int Aut_counter;

	int *Base; // [Base_length]
	int Base_length;

	long int *Base_lint; // [N]
	int *Transversal_length; // [N]


	algebra::ring_theory::longinteger_object *Ago;

	int *canonical_labeling; // [N]

	// nauty function call stats:
	long int nb_firstpathnode;
	long int nb_othernode;
	long int nb_processnode;
	long int nb_firstterminal;

	nauty_output();
	~nauty_output();
	void nauty_output_allocate(
			int N,
			int invariant_set_start, int invariant_set_size,
			int verbose_level);
	void print();
	void print_stats();
	int belong_to_the_same_orbit(
			int a, int b, int verbose_level);
	int get_output_size(
			int verbose_level);
	void stringify_as_vector(
			std::vector<std::string> &V,
			int verbose_level);
	void stringify(
			std::string &s_n, std::string &s_ago,
			std::string &s_base_length, std::string &s_aut_counter,
			std::string &s_base, std::string &s_tl,
			std::string &s_aut, std::string &s_cl,
			std::string &s_stats,
			int verbose_level);
	void nauty_output_init_from_string(
			int N,
			int invariant_set_start, int invariant_set_size,
			int idx_start,
			std::vector<std::string> &Carrying_through,
			int verbose_level);
	long int nauty_complexity();

};

// #############################################################################
// povray_interface.cpp
// #############################################################################

//! povray interface for 3D graphics



class povray_interface {
public:


	std::string color_white_simple;
	std::string color_white;
	std::string color_white_very_transparent;
	std::string color_black;
	std::string color_pink;
	std::string color_pink_transparent;
	std::string color_green;
	std::string color_gold;
	std::string color_red;
	std::string color_blue;
	std::string color_yellow;
	std::string color_yellow_transparent;
	std::string color_scarlet;
	std::string color_brown;
	std::string color_orange;
	std::string color_orange_transparent;
	std::string color_orange_no_phong;
	std::string color_chrome;
	std::string color_gold_dode;
	std::string color_gold_transparent;
	std::string color_red_wine_transparent;
	std::string color_yellow_lemon_transparent;

	double sky[3];
	double location[3];
	double look_at[3];


	povray_interface();
	~povray_interface();
	void beginning(
			std::ostream &ost,
			double angle,
			double *sky,
			double *location,
			double *look_at,
			int f_with_background);
	void animation_rotate_around_origin_and_1_1_1(
			std::ostream &ost);
	void animation_rotate_around_origin_and_given_vector(
			double *v,
			std::ostream &ost);
	void animation_rotate_xyz(
		double angle_x_deg, double angle_y_deg, double angle_z_deg,
		std::ostream &ost);
	void animation_rotate_around_origin_and_given_vector_by_a_given_angle(
		double *v, double angle_zero_one, std::ostream &ost);
	void union_start(
			std::ostream &ost);
	void union_end(
			std::ostream &ost,
			double scale_factor, double clipping_radius);
	void union_end_box_clipping(
			std::ostream &ost, double scale_factor,
			double box_x, double box_y, double box_z);
	void union_end_no_clipping(
			std::ostream &ost, double scale_factor);
	void bottom_plane(
			std::ostream &ost);
	void rotate_111(
			int h, int nb_frames, std::ostream &fp);
	void rotate_around_z_axis(
			int h, int nb_frames, std::ostream &fp);
	void ini(
			std::ostream &ost,
			const char *fname_pov, int first_frame,
		int last_frame);
};


// #############################################################################
// pugixml_interface.cpp
// #############################################################################

//! interface to pugixml for reading xml files



class pugixml_interface {
public:
	pugixml_interface();
	~pugixml_interface();
	void read_file(
			std::string &fname,
			std::vector<std::vector<std::string> > &Classes_parsed,
			int verbose_level);


};



}}}}




#endif /* SRC_LIB_LAYER1_FOUNDATIONS_L1_INTERFACES_L1_INTERFACES_H_ */
