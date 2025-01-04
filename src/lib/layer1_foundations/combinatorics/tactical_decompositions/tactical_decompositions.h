/*
 * tactical_decompositions.h
 *
 *  Created on: Dec 1, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_TACTICAL_DECOMPOSITIONS_TACTICAL_DECOMPOSITIONS_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_TACTICAL_DECOMPOSITIONS_TACTICAL_DECOMPOSITIONS_H_


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {



// #############################################################################
// decomposition_scheme.cpp
// #############################################################################


//! a decomposition scheme of an incidence structure


class decomposition_scheme {

public:

	decomposition *Decomposition;

	row_and_col_partition *RC;

	int f_has_row_scheme;
	int *row_scheme;
	int f_has_col_scheme;
	int *col_scheme;

	other::data_structures::set_of_sets *SoS_points;
	other::data_structures::set_of_sets *SoS_lines;

	decomposition_scheme();
	~decomposition_scheme();
	void init_row_and_col_schemes(
			decomposition *Decomposition,
			int verbose_level);
	// called from combinatorics_domain::compute_TDO_decomposition_of_projective_space
	void get_classes(
			int verbose_level);
	void init_row_scheme(
			decomposition *Decomposition,
			int verbose_level);
	void init_col_scheme(
			decomposition *Decomposition,
			int verbose_level);
	void get_row_scheme(
			int verbose_level);
	void get_col_scheme(
			int verbose_level);
	void print_row_decomposition_tex(
		std::ostream &ost,
		int f_enter_math, int f_print_subscripts,
		int verbose_level);
	void print_column_decomposition_tex(
		std::ostream &ost,
		int f_enter_math, int f_print_subscripts,
		int verbose_level);
	void print_decomposition_scheme_tex(
			std::ostream &ost,
		int *scheme);
	void print_tactical_decomposition_scheme_tex(
			std::ostream &ost,
		int f_print_subscripts);
	void print_tactical_decomposition_scheme_tex_internal(
		std::ostream &ost, int f_enter_math_mode,
		int f_print_subscripts);
	void print_row_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		int f_print_subscripts);
	void print_column_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		int f_print_subscripts);
	void print_non_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		int f_print_subscripts);
	void stringify_row_scheme(
			std::string *&Table, int f_print_subscripts);
		// Table[(nb_row_classes + 1) * (nb_col_classes + 1)]
	void stringify_col_scheme(
			std::string *&Table, int f_print_subscripts);
		// Table[(nb_row_classes + 1) * (nb_col_classes + 1)]
	void write_csv(
			std::string &fname_row, std::string &fname_col,
			std::string &fname_row_classes, std::string &fname_col_classes,
			int verbose_level);
	void report_latex_with_external_files(
			std::ostream &ost,
			std::string &label_scheme,
			std::string &label_txt,
			int upper_bound_on_size_for_printing,
			int verbose_level);
	void report_classes_with_external_files(
			std::ostream &ost,
			std::string &label_scheme,
			std::string &label_txt,
			int verbose_level);
	void export_csv(
			std::string &label_scheme,
			std::string &label_txt,
			int verbose_level);

};


// #############################################################################
// decomposition.cpp
// #############################################################################


//! decomposition of an incidence matrix


class decomposition {

public:

	int nb_points;
	int nb_blocks;
	int N;
	int *Incma;

	geometry::other_geometry::incidence_structure *Inc;
	other::data_structures::partitionstack *Stack;

	int f_has_decomposition;
	decomposition_scheme *Scheme;


	decomposition();
	~decomposition();
	void init_incidence_structure(
			geometry::other_geometry::incidence_structure *Inc,
			int verbose_level);
	// called from
	// decomposition::init_decomposition_of_projective_space
	// combinatorics_domain::refine_the_partition
	// combinatorics_domain::compute_TDO_decomposition_of_projective_space_old
	// set_of_sets::get_eckardt_points
	// combinatorics_with_action::report_TDO_and_TDA
	// combinatorics_with_action::report_TDA
	// group_action_on_combinatorial_object::init
	// design_activity::do_tactical_decomposition
	void init_inc_and_stack(
			geometry::other_geometry::incidence_structure *Inc,
			other::data_structures::partitionstack *Stack,
		int verbose_level);
	void init_decomposition_of_projective_space(
			geometry::projective_geometry::projective_space *P,
			long int *points, int nb_points,
			long int *lines, int nb_lines,
			int verbose_level);
	void init_incidence_matrix(
			int m, int n, int *M,
		int verbose_level);
		// copies the incidence matrix
	void compute_TDO_deep(
			int verbose_level);
	void compute_the_decomposition(
			int verbose_level);
	void setup_default_partition(
			int verbose_level);
	void compute_TDO_old(
			int max_depth, int verbose_level);
	void get_row_scheme(
			int verbose_level);
	void get_col_scheme(
			int verbose_level);
	void compute_TDO_safe(
		int depth, int verbose_level);
	void compute_TDO_safe_and_write_files(
		int depth,
		std::string &fname_base,
		std::vector<std::string> &file_names,
		int verbose_level);
	int refine_column_partition_safe(
			int verbose_level);
	int refine_row_partition_safe(
			int verbose_level);
	void get_and_print_row_decomposition_scheme(
		int f_list_incidences,
		int f_local_coordinates, int verbose_level);
	void get_and_print_col_decomposition_scheme(
		int f_list_incidences,
		int f_local_coordinates, int verbose_level);
	void get_permuted_incidence_matrix(
			row_and_col_partition *RC,
		int *&incma, int verbose_level);
	void latex(
			std::ostream &ost,
			other::graphics::draw_incidence_structure_description *Draw_options,
			row_and_col_partition *RC,
			int verbose_level);
	void get_row_decomposition_scheme(
			row_and_col_partition *RC,
		int *row_scheme, int verbose_level);
	void get_row_decomposition_scheme_if_possible(
			row_and_col_partition *RC,
		int *row_scheme, int verbose_level);
	void get_col_decomposition_scheme(
			row_and_col_partition *RC,
		int *col_scheme, int verbose_level);
	void row_scheme_to_col_scheme(
			row_and_col_partition *RC,
		int *row_scheme, int *col_scheme,
		int verbose_level);
	void print_row_tactical_decomposition_scheme_incidences_tex(
		std::ostream &ost, int f_enter_math_mode,
		row_and_col_partition *RC,
		int f_local_coordinates, int verbose_level);
	void print_col_tactical_decomposition_scheme_incidences_tex(
		std::ostream &ost, int f_enter_math_mode,
		row_and_col_partition *RC,
		int f_local_coordinates, int verbose_level);
	void get_incidences_by_row_scheme(
			row_and_col_partition *RC,
		int row_class_idx, int col_class_idx,
		int rij, int *&incidences, int verbose_level);
	void get_incidences_by_col_scheme(
			row_and_col_partition *RC,
		int row_class_idx, int col_class_idx,
		int kij, int *&incidences, int verbose_level);
	void get_and_print_decomposition_schemes();
	void get_and_print_row_tactical_decomposition_scheme_tex(
		std::ostream &ost,
		int f_enter_math, int f_print_subscripts);
	void get_and_print_column_tactical_decomposition_scheme_tex(
		std::ostream &ost,
		int f_enter_math, int f_print_subscripts);
	void print_partitioned(
		std::ostream &ost,
		int f_labeled);
	void print_column_labels(
		std::ostream &ost,
		int *col_classes, int nb_col_classes, int width);
	void print_hline(
		std::ostream &ost,
		row_and_col_partition *RC,
		int width, int f_labeled);
	void print_line(
		std::ostream &ost,
		row_and_col_partition *RC,
		int row_cell, int i,
		int width, int f_labeled);
	void stringify_decomposition(
			row_and_col_partition *RC,
			std::string *&T,
			int *the_scheme,
			int f_print_subscripts);
	void prepare_col_labels(
			row_and_col_partition *RC,
			std::vector<std::string> &col_labels, int f_print_subscripts);
	void prepare_row_labels(
			row_and_col_partition *RC,
			std::vector<std::string> &row_labels, int f_print_subscripts);
	void prepare_matrix(
			row_and_col_partition *RC,
			std::vector<std::string> &matrix_labels,
			int *the_scheme);
	void print_row_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		row_and_col_partition *RC,
		int *row_scheme, int f_print_subscripts);
	void print_column_tactical_decomposition_scheme_tex(
		std::ostream &ost, int f_enter_math_mode,
		row_and_col_partition *RC,
		int *col_scheme, int f_print_subscripts);
	void compute_TDO(
			int verbose_level);
	void get_and_report_classes(
			std::ostream &ost,
			int verbose_level);
	void print_schemes(
			std::ostream &ost,
			combinatorics::canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);

};



// #############################################################################
// geo_parameter.cpp
// #############################################################################


#define MODE_UNDEFINED 0
#define MODE_SINGLE 1
#define MODE_STACK 2

#define UNKNOWNTYPE 0
#define POINTTACTICAL 1
#define BLOCKTACTICAL 2
#define POINTANDBLOCKTACTICAL 3

#define FUSE_TYPE_NONE 0
#define FUSE_TYPE_SIMPLE 1
#define FUSE_TYPE_DOUBLE 2
//#define FUSE_TYPE_MULTI 3
//#define FUSE_TYPE_TDO 4

//! decomposition stack of a linear space or incidence geometry



class geo_parameter {
public:
	int decomposition_type;
	int fuse_type;
	int v, b;

	int mode;
	std::string label;

	// for MODE_SINGLE
	int nb_V, nb_B;
	int *V, *B;
	int *scheme;
	int *fuse;

	// for MODE_STACK
	int nb_parts, nb_entries;

	int *part;
	int *entries;
	int part_nb_alloc;
	int entries_nb_alloc;


	int lambda_level;
	int row_level, col_level;
	int extra_row_level, extra_col_level;

	geo_parameter();
	~geo_parameter();
	void append_to_part(
			int a);
	void append_to_entries(
			int a1, int a2, int a3, int a4);
	void write(
			std::ofstream &aStream, std::string &label);
	void write_mode_single(
			std::ofstream &aStream,
			std::string &label);
	void write_mode_stack(
			std::ofstream &aStream,
			std::string &label);
	void convert_single_to_stack(
			int verbose_level);
	int partition_number_row(
			int row_idx);
	int partition_number_col(
			int col_idx);
	int input(
			std::ifstream &aStream);
	int input_mode_single(
			std::ifstream &aStream);
	int input_mode_stack(
			std::ifstream &aStream, int verbose_level);
	void init_tdo_scheme(
			tdo_scheme_synthetic &G, int verbose_level);
	void print_schemes(
			tdo_scheme_synthetic &G);
	void print_schemes_tex(
			tdo_scheme_synthetic &G);
	void print_scheme_tex(
			std::ostream &ost,
			tdo_scheme_synthetic &G, int h);
	void print_C_source();
	void convert_single_to_stack_fuse_simple_pt(
			int verbose_level);
	void convert_single_to_stack_fuse_simple_bt(
			int verbose_level);
	void convert_single_to_stack_fuse_double_pt(
			int verbose_level);
	void cut_off_two_lines(
			geo_parameter &GP2,
		int *&part_relabel, int *&part_length,
		int verbose_level);
	void cut_off(
			geo_parameter &GP2, int w,
		int *&part_relabel, int *&part_length,
		int verbose_level);
	void copy(
			geo_parameter &GP2);
	void print_schemes();
	int tdo_scheme_get_row_class_length_fused(
			tdo_scheme_synthetic &G,
			int h, int class_first, int class_len);
	int tdo_scheme_get_col_class_length_fused(
			tdo_scheme_synthetic &G,
			int h, int class_first, int class_len);
};


// #############################################################################
// row_and_col_partition.cpp
// #############################################################################


//! the partition associated with a decomposition of an incidence matrix


class row_and_col_partition {

public:

	other::data_structures::partitionstack *Stack;

	int *row_classes;
	int *row_class_idx;
	int nb_row_classes;

	int *col_classes;
	int *col_class_idx;
	int nb_col_classes;

	row_and_col_partition();
	~row_and_col_partition();
	void init_from_partitionstack(
			other::data_structures::partitionstack *Stack,
			int verbose_level);
	void print_classes_of_decomposition_tex(
			std::ostream &ost);
	void print_decomposition_scheme(
			std::ostream &ost,
		int *scheme);
	void print_row_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math_mode,
		int *row_scheme, int f_print_subscripts);
	void print_column_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math_mode,
		int *col_scheme, int f_print_subscripts);


};




// #############################################################################
// tdo_data.cpp TDO parameter refinement
// #############################################################################

//! a utility class related to the class tdo_scheme

class tdo_data {
public:
	int *types_first;
	int *types_len;
	int *only_one_type;
	int nb_only_one_type;
	int *multiple_types;
	int nb_multiple_types;
	int *types_first2;
	solvers::diophant *D1;
	solvers::diophant *D2;

	tdo_data();
	~tdo_data();
	void free();
	void allocate(
			int R);
	int solve_first_system(
			int verbose_level,
		int *&line_types, int &nb_line_types,
		int &line_types_allocated);
	void solve_second_system_omit(
			int verbose_level,
		int *classes_len,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		int omit);
	void solve_second_system_with_help(
			int verbose_level,
		int f_use_mckay_solver, int f_once,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		int cnt_second_system, solution_file_data *Sol);
	void solve_second_system_from_file(
			int verbose_level,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions,
		std::string &solution_file_name);
	void solve_second_system(
			int verbose_level,
		int f_use_mckay_solver, int f_once,
		int *classes_len, int f_scale, int scaling,
		int *&line_types, int &nb_line_types,
		int *&distributions, int &nb_distributions);
};

// #############################################################################
// tdo_refinement_description.cpp
// #############################################################################



//! input data for the parameter refinement of a linear space

class tdo_refinement_description {
	public:

	int f_lambda3;
	int lambda3, block_size;
	int f_scale;
	int scaling;
	int f_range;
	int range_first, range_len;
	int f_select;
	std::string select_label;
	int f_omit1;
	int omit1;
	int f_omit2;
	int omit2;
	int f_D1_upper_bound_x0;
	int D1_upper_bound_x0;
	int f_reverse;
	int f_reverse_inverse;
	int f_use_packing_numbers;
	int f_dual_is_linear_space;
	int f_do_the_geometric_test;
	int f_once;
	int f_use_mckay_solver;
	int f_input_file;
	std::string fname_in;

	solution_file_data *Sol;

	tdo_refinement_description();
	~tdo_refinement_description();
	int read_arguments(
			int argc, std::string *argv, int verbose_level);
	void print();

};


// #############################################################################
// tdo_refinement.cpp
// #############################################################################



//! refinement of the parameters of a linear space

class tdo_refinement {
	public:

	int t0;
	int cnt;

	tdo_refinement_description *Descr;

	std::string fname;
	std::string fname_out;



	geo_parameter GP;

	geo_parameter GP2;



	int f_doit;
	int nb_written, nb_written_tactical, nb_tactical;
	int cnt_second_system;

	tdo_refinement();
	~tdo_refinement();
	void init(
			tdo_refinement_description *Descr,
			int verbose_level);
	void main_loop(
			int verbose_level);
	void do_it(
			std::ofstream &g, int verbose_level);
	void do_row_refinement(
			std::ofstream &g,
			tdo_scheme_synthetic &G,
			other::data_structures::partitionstack &P,
			int verbose_level);
	void do_col_refinement(
			std::ofstream &g,
			tdo_scheme_synthetic &G,
			other::data_structures::partitionstack &P,
			int verbose_level);
	void do_all_row_refinements(
			std::string &label_in, std::ofstream &g,
			tdo_scheme_synthetic &G,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions, int &nb_tactical,
		int verbose_level);
	void do_all_column_refinements(
			std::string &label_in, std::ofstream &g,
			tdo_scheme_synthetic &G,
		int *line_types, int nb_line_types, int line_type_len,
		int *distributions, int nb_distributions, int &nb_tactical,
		int verbose_level);
	int do_row_refinement(
			int t,
			std::string &label_in,
			std::ofstream &g,
			tdo_scheme_synthetic &G,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions,
		int verbose_level);
		// returns true or false depending on whether the
		// refinement gave a tactical decomposition
	int do_column_refinement(
			int t, std::string &label_in,
			std::ofstream &g,
			tdo_scheme_synthetic &G,
		int *line_types, int nb_line_types, int line_type_len,
		int *distributions, int nb_distributions,
		int verbose_level);
		// returns true or false depending on whether the
		// refinement gave a tactical decomposition
};



// #############################################################################
// tdo_scheme_compute.cpp
// #############################################################################

//! tactical decomposition of an incidence structure obtained by refinement

class tdo_scheme_compute {

public:

	canonical_form_classification::encoded_combinatorial_object *Enc;
	decomposition *Decomp;

	int f_TDA;
	int nb_orbits;
	int *orbit_first;
	int *orbit_len;
	int *orbit;

	tdo_scheme_compute();
	~tdo_scheme_compute();
	void init(
			canonical_form_classification::encoded_combinatorial_object *Enc,
			int max_depth,
			int verbose_level);
	// used by combinatorial_object_with_properties::compute_TDO
	void init_TDA(
			canonical_form_classification::encoded_combinatorial_object *Enc,
			int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
			int verbose_level);
	void print_schemes(
			std::ostream &ost,
			canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);

};


// #############################################################################
// tdo_scheme_synthetic.cpp
// #############################################################################



#define NUMBER_OF_SCHEMES 5
#define ROW_SCHEME 0
#define COL_SCHEME 1
#define LAMBDA_SCHEME 2
#define EXTRA_ROW_SCHEME 3
#define EXTRA_COL_SCHEME 4


//! internal class related to class tdo_data


struct solution_file_data {
	int nb_solution_files;
	std::vector<int> system_no;
	std::vector<std::string> solution_file;
};

//! canonical tactical decomposition of an incidence structure

class tdo_scheme_synthetic {

public:

	// the following is needed by the TDO process:
	// allocated in init_partition_stack
	// freed in exit_partition_stack

	//partition_backtrack PB;

	other::data_structures::partitionstack *P;

	int part_length;
	int *part;
	int nb_entries;
	int *entries;
	int row_level;
	int col_level;
	int lambda_level;
	int extra_row_level;
	int extra_col_level;

	int mn; // m + n
	int m; // # of rows
	int n; // # of columns

	int level[NUMBER_OF_SCHEMES];
	int *row_classes[NUMBER_OF_SCHEMES], nb_row_classes[NUMBER_OF_SCHEMES];
	int *col_classes[NUMBER_OF_SCHEMES], nb_col_classes[NUMBER_OF_SCHEMES];
	int *row_class_index[NUMBER_OF_SCHEMES];
	int *col_class_index[NUMBER_OF_SCHEMES];
	int *row_classes_first[NUMBER_OF_SCHEMES];
	int *row_classes_len[NUMBER_OF_SCHEMES];
	int *row_class_no[NUMBER_OF_SCHEMES];
	int *col_classes_first[NUMBER_OF_SCHEMES];
	int *col_classes_len[NUMBER_OF_SCHEMES];
	int *col_class_no[NUMBER_OF_SCHEMES];

	int *the_row_scheme;
	int *the_col_scheme;
	int *the_extra_row_scheme;
	int *the_extra_col_scheme;
	int *the_row_scheme_cur; // [m * nb_col_classes[ROW_SCHEME]]
	int *the_col_scheme_cur; // [n * nb_row_classes[COL_SCHEME]]
	int *the_extra_row_scheme_cur; // [m * nb_col_classes[EXTRA_ROW_SCHEME]]
	int *the_extra_col_scheme_cur; // [n * nb_row_classes[EXTRA_COL_SCHEME]]

	// end of TDO process data

	tdo_scheme_synthetic();
	~tdo_scheme_synthetic();

	void init_part_and_entries(
			int *part, int *entries, int verbose_level);
	void init_part_and_entries_int(
			int *part, int *entries, int verbose_level);
	void init_TDO(
			int *Part, int *Entries,
		int Row_level, int Col_level,
		int Extra_row_level, int Extra_col_level,
		int Lambda_level, int verbose_level);
	void exit_TDO();
	void init_partition_stack(
			int verbose_level);
	void exit_partition_stack();
	void get_partition(
			int h, int l, int verbose_level);
	void free_partition(
			int h);
	void complete_partition_info(
			int h, int verbose_level);
	void get_row_or_col_scheme(
			int h, int l, int verbose_level);
	void get_column_split_partition(
			int verbose_level,
			other::data_structures::partitionstack &P);
	void get_row_split_partition(
			int verbose_level,
			other::data_structures::partitionstack &P);
	void print_all_schemes();
	void print_scheme(
			int h, int verbose_level);
	void print_scheme_tex(
			std::ostream &ost, int h);
	void print_scheme_tex_fancy(
			std::ostream &ost,
			int h, int f_label, std::string &label);
	void compute_whether_first_inc_must_be_moved(
			int *f_first_inc_must_be_moved, int verbose_level);
	int count_nb_inc_from_row_scheme(
			int verbose_level);
	int count_nb_inc_from_extra_row_scheme(
			int verbose_level);


	int geometric_test_for_row_scheme(
			other::data_structures::partitionstack &P,
		int *point_types, int nb_point_types, int point_type_len,
		int *distributions, int nb_distributions,
		int f_omit1, int omit1, int verbose_level);
	int geometric_test_for_row_scheme_level_s(
			other::data_structures::partitionstack &P, int s,
		int *point_types, int nb_point_types, int point_type_len,
		int *distribution,
		int *non_zero_blocks, int nb_non_zero_blocks,
		int f_omit1, int omit1,
		int verbose_level);


	int refine_rows(
			int verbose_level,
		int f_use_mckay, int f_once,
		other::data_structures::partitionstack &P,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit2, int omit2,
		int f_use_packing_numbers,
		int f_dual_is_linear_space,
		int f_do_the_geometric_test);
	int refine_rows_easy(
			int verbose_level,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system);
	int refine_rows_hard(
			other::data_structures::partitionstack &P,
			int verbose_level,
		int f_use_mckay, int f_once,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_use_packing_numbers, int f_dual_is_linear_space);
	void row_refinement_L1_L2(
			other::data_structures::partitionstack &P,
			int f_omit, int omit,
		int &L1, int &L2, int verbose_level);
	int tdo_rows_setup_first_system(
			int verbose_level,
		tdo_data &T, int r,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *&point_types, int &nb_point_types);
	int tdo_rows_setup_second_system(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int f_use_packing_numbers,
		int f_dual_is_linear_space,
		int *&point_types, int &nb_point_types);
	int tdo_rows_setup_second_system_eqns_joining(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit, int f_dual_is_linear_space,
		int *point_types, int nb_point_types,
		int eqn_offset);
	int tdo_rows_setup_second_system_eqns_counting(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *point_types, int nb_point_types,
		int eqn_offset);
	int tdo_rows_setup_second_system_eqns_packing(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *point_types, int nb_point_types,
		int eqn_start, int &nb_eqns_used);

	int refine_columns(
			int verbose_level, int f_once,
			other::data_structures::partitionstack &P,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_D1_upper_bound_x0, int D1_upper_bound_x0,
		int f_use_mckay_solver,
		int f_use_packing_numbers);
	int refine_cols_hard(
			other::data_structures::partitionstack &P,
		int verbose_level, int f_once,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions,
		int &cnt_second_system, solution_file_data *Sol,
		int f_omit1, int omit1, int f_omit, int omit,
		int f_D1_upper_bound_x0, int D1_upper_bound_x0,
		int f_use_mckay_solver,
		int f_use_packing_numbers);
	void column_refinement_L1_L2(
			other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int &L1, int &L2, int verbose_level);
	int tdo_columns_setup_first_system(
			int verbose_level,
		tdo_data &T, int r,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *&line_types, int &nb_line_types);
	int tdo_columns_setup_second_system(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int f_use_packing_numbers,
		int *&line_types, int &nb_line_types);
	int tdo_columns_setup_second_system_eqns_joining(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start);
	void tdo_columns_setup_second_system_eqns_counting(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start);
	int tdo_columns_setup_second_system_eqns_upper_bound(
			int verbose_level,
		tdo_data &T,
		other::data_structures::partitionstack &P,
		int f_omit, int omit,
		int *line_types, int nb_line_types,
		int eqn_start, int &nb_eqns_used);


	int td3_refine_rows(
			int verbose_level, int f_once,
		int lambda3, int block_size,
		int *&point_types, int &nb_point_types, int &point_type_len,
		int *&distributions, int &nb_distributions);
	int td3_rows_setup_first_system(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T, int r,
		other::data_structures::partitionstack &P,
		int &nb_vars,int &nb_eqns,
		int *&point_types, int &nb_point_types);
	int td3_rows_setup_second_system(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int &Nb_vars, int &Nb_eqns,
		int *&point_types, int &nb_point_types);
	int td3_rows_counting_flags(
			int verbose_level,
		int lambda3, int block_size, int lambda2, int &S,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&point_types, int &nb_point_types, int eqn_offset);
	int td3_refine_columns(
			int verbose_level, int f_once,
		int lambda3, int block_size,
		int f_scale, int scaling,
		int *&line_types, int &nb_line_types, int &line_type_len,
		int *&distributions, int &nb_distributions);
	int td3_columns_setup_first_system(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T, int r,
		other::data_structures::partitionstack &P,
		int &nb_vars, int &nb_eqns,
		int *&line_types, int &nb_line_types);
	int td3_columns_setup_second_system(
			int verbose_level,
		int lambda3, int block_size, int lambda2, int f_scale, int scaling,
		tdo_data &T,
		int nb_vars, int &Nb_vars, int &Nb_eqns,
		int *&line_types, int &nb_line_types);
	int td3_columns_triples_same_class(
			int verbose_level,
		int lambda3, int block_size,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_pairs_same_class(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_counting_flags(
			int verbose_level,
		int lambda3, int block_size, int lambda2, int &S,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda2_joining_pairs_from_different_classes(
		int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda3_joining_triples_2_1(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);
	int td3_columns_lambda3_joining_triples_1_1_1(
			int verbose_level,
		int lambda3, int block_size, int lambda2,
		tdo_data &T,
		int nb_vars, int Nb_vars,
		int *&line_types, int &nb_line_types, int eqn_offset);


};







}}}}


#endif /* SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_TACTICAL_DECOMPOSITIONS_TACTICAL_DECOMPOSITIONS_H_ */
