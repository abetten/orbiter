// io_and_os.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_IO_AND_OS_IO_AND_OS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_IO_AND_OS_IO_AND_OS_H_





namespace orbiter {
namespace foundations {



// #############################################################################
// create_file_description.cpp
// #############################################################################


#define MAX_LINES 100

//! rules to create text files


class create_file_description {
public:
	int f_file_mask;
	std::string file_mask;
	int f_N;
	int N;
	int nb_lines;
	std::string lines[MAX_LINES];
	int f_line_numeric[MAX_LINES];
	int nb_final_lines;
	std::string final_lines[MAX_LINES];
	int f_command;
	std::string command;
	int f_repeat;
	int repeat_N;
	int repeat_start;
	int repeat_increment;
	std::string repeat_mask;
	int f_split;
	int split_m;
	int f_read_cases;
	std::string read_cases_fname;
	int f_read_cases_text;
	int read_cases_column_of_case;
	int read_cases_column_of_fname;
	int f_tasks;
	int nb_tasks;
	std::string tasks_line;

	create_file_description();
	~create_file_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};


// #############################################################################
// file_io.cpp
// #############################################################################

//! a collection of functions related to file io




class file_io {
public:
	file_io();
	~file_io();

	void concatenate_files(const char *fname_in_mask, int N,
		const char *fname_out, const char *EOF_marker, int f_title_line,
		int &cnt_total,
		std::vector<int> missing_idx,
		int verbose_level);
	void concatenate_files_into(const char *fname_in_mask, int N,
		std::ofstream &fp_out, const char *EOF_marker, int f_title_line,
		int &cnt_total,
		std::vector<int> &missing_idx,
		int verbose_level);
	void poset_classification_read_candidates_of_orbit(
			std::string &fname, int orbit_at_level,
		long int *&candidates, int &nb_candidates, int verbose_level);
	void read_candidates_for_one_orbit_from_file(std::string &prefix,
			int level, int orbit_at_level, int level_of_candidates_file,
			long int *S,
			void (*early_test_func_callback)(long int *S, int len,
				long int *candidates, int nb_candidates,
				long int *good_candidates, int &nb_good_candidates,
				void *data, int verbose_level),
			void *early_test_func_callback_data,
			long int *&candidates,
			int &nb_candidates,
			int verbose_level);
	int find_orbit_index_in_data_file(std::string &prefix,
			int level_of_candidates_file, long int *starter,
			int verbose_level);
	void write_exact_cover_problem_to_file(int *Inc, int nb_rows,
		int nb_cols, std::string &fname);
	void read_solution_file(std::string &fname,
		int *Inc, int nb_rows, int nb_cols,
		int *&Solutions, int &sol_length, int &nb_sol,
		int verbose_level);
	void count_number_of_solutions_in_file_and_get_solution_size(
			std::string &fname,
		int &nb_solutions, int &solution_size,
		int verbose_level);
	void count_number_of_solutions_in_file(std::string &fname,
		int &nb_solutions,
		int verbose_level);
	void count_number_of_solutions_in_file_by_case(std::string &fname,
		int *&nb_solutions, int *&case_nb, int &nb_cases,
		int verbose_level);
	void read_solutions_from_file_and_get_solution_size(std::string &fname,
		int &nb_solutions, int *&Solutions, int &solution_size,
		int verbose_level);
	void read_solutions_from_file(std::string &fname,
		int &nb_solutions, int *&Solutions, int solution_size,
		int verbose_level);
	void read_solutions_from_file_size_is_known(std::string &fname,
		std::vector<std::vector<int> > &Solutions, int solution_size,
		int verbose_level);
	void read_solutions_from_file_by_case(std::string &fname,
		int *nb_solutions, int *case_nb, int nb_cases,
		int **&Solutions, int solution_size,
		int verbose_level);
	void copy_file_to_ostream(std::ostream &ost, const char *fname);
	void int_vec_write_csv(int *v, int len,
			std::string &fname, const char *label);
	void lint_vec_write_csv(long int *v, int len,
			std::string &fname, const char *label);
	void int_vecs_write_csv(int *v1, int *v2, int len,
			std::string &fname,
		const char *label1, const char *label2);
	void int_vecs3_write_csv(int *v1, int *v2, int *v3, int len,
			std::string &fname,
		const char *label1, const char *label2, const char *label3);
	void int_vec_array_write_csv(int nb_vecs, int **Vec, int len,
			std::string &fname, const char **column_label);
	void lint_vec_array_write_csv(int nb_vecs, long int **Vec, int len,
			std::string &fname, const char **column_label);
	void int_matrix_write_csv(std::string &fname, int *M, int m, int n);
	void lint_matrix_write_csv(std::string &fname, long int *M, int m, int n);
	void lint_matrix_write_csv_override_headers(std::string &fname, std::string *headers, long int *M, int m, int n);
	void double_matrix_write_csv(std::string &fname,
		double *M, int m, int n);
	void int_matrix_write_csv_with_labels(std::string &fname,
		int *M, int m, int n, const char **column_label);
	void lint_matrix_write_csv_with_labels(std::string &fname,
		long int *M, int m, int n, const char **column_label);
	void int_matrix_read_csv(std::string &fname, int *&M,
		int &m, int &n, int verbose_level);
	void lint_matrix_read_csv(std::string &fname,
		long int *&M, int &m, int &n, int verbose_level);
	void double_matrix_read_csv(std::string &fname, double *&M,
		int &m, int &n, int verbose_level);
	void int_matrix_write_cas_friendly(std::string &fname, int *M, int m, int n);
	void int_matrix_write_text(std::string &fname,
		int *M, int m, int n);
	void lint_matrix_write_text(std::string &fname, long int *M, int m, int n);
	void int_matrix_read_text(std::string &fname,
		int *&M, int &m, int &n);
	void parse_sets(int nb_cases, char **data, int f_casenumbers,
		int *&Set_sizes, long int **&Sets, char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void parse_sets_and_check_sizes_easy(int len, int nb_cases,
		char **data, long int **&sets);
	void parse_line(char *line, int &len, long int *&set,
		char *ago_ascii, char *aut_ascii);
	int count_number_of_orbits_in_file(std::string &fname, int verbose_level);
	int count_number_of_lines_in_file(std::string &fname, int verbose_level);
	int try_to_read_file(std::string &fname, int &nb_cases,
		char **&data, int verbose_level);
	void read_and_parse_data_file(std::string &fname, int &nb_cases,
		char **&data, long int **&sets, int *&set_sizes, int verbose_level);
	void free_data_fancy(int nb_cases,
		int *Set_sizes, long int **Sets,
		char **Ago_ascii, char **Aut_ascii,
		int *Casenumbers);
	void read_and_parse_data_file_fancy(std::string &fname,
		int f_casenumbers,
		int &nb_cases,
		int *&Set_sizes, long int **&Sets, char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void read_set_from_file(std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file(std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_lint(std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_lint(std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_int4(std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void read_set_from_file_int8(std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_as_int4(std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void write_set_to_file_as_int8(std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_k_th_set_from_file(std::string &fname, int k,
		int *&the_set, int &set_size, int verbose_level);
	void write_incidence_matrix_to_file(std::string &fname,
		int *Inc, int m, int n, int verbose_level);
	void read_incidence_matrix_from_inc_file(int *&M, int &m, int &n,
			std::string &inc_file_name, int inc_file_idx, int verbose_level);
	int inc_file_get_number_of_geometries(
		char *inc_file_name, int verbose_level);
	long int file_size(std::string &fname);
	long int file_size(const char *name);
	void delete_file(const char *fname);

	void fwrite_int4(FILE *fp, int a);
	int_4 fread_int4(FILE *fp);
	void fwrite_uchars(FILE *fp, uchar *p, int len);
	void fread_uchars(FILE *fp, uchar *p, int len);

	void read_numbers_from_file(std::string &fname,
		int *&the_set, int &set_size, int verbose_level);
	void read_ascii_set_of_sets_constant_size(
			std::string &fname_ascii,
			int *&Sets, int &nb_sets, int &set_size, int verbose_level);
	void write_decomposition_stack(char *fname, int m, int n,
			int *v, int *b, int *aij, int verbose_level);
	void create_file(create_file_description *Descr, int verbose_level);
	void fix_escape_characters(char *str);
	void create_files(create_file_description *Descr,
		int verbose_level);
	void create_files_list_of_cases(spreadsheet *S,
			create_file_description *Descr, int verbose_level);
	int number_of_vertices_in_colored_graph(std::string &fname, int verbose_level);
	void do_csv_file_select_rows(std::string &fname,
			std::string &rows_text,
			int verbose_level);
	void do_csv_file_split_rows_modulo(std::string &fname,
			int split_modulo,
			int verbose_level);
	void do_csv_file_select_cols(std::string &fname,
			std::string &cols_text,
			int verbose_level);
	void do_csv_file_select_rows_and_cols(std::string &fname,
			std::string &rows_text, std::string &cols_text,
			int verbose_level);
	void do_csv_file_join(
			std::vector<std::string> &csv_file_join_fname,
			std::vector<std::string> &csv_file_join_identifier, int verbose_level);
	void do_csv_file_concatenate(
			std::vector<std::string> &fname, std::string &fname_out, int verbose_level);
	void do_csv_file_latex(std::string &fname,
			int f_produce_latex_header,
			int verbose_level);
	void read_solutions_and_tally(std::string &fname, int sz, int verbose_level);

};


// #############################################################################
// file_output.cpp
// #############################################################################


//! a wrapper class for an ofstream which allows to store extra data

class file_output {
public:
	char fname[1000];
	int f_file_is_open;
	std::ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void null();
	void freeself();
	void open(const char *fname, void *user_data,
			int verbose_level);
	void close();
	void write_line(int nb, int *data,
			int verbose_level);
	void write_EOF(int nb_sol, int verbose_level);
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
			std::ostream& ost, const char *extras);
	void head_easy_sideways(std::ostream& ost);
	void head(std::ostream& ost, int f_book, int f_title,
		const char *title, const char *author,
		int f_toc, int f_landscape, int f_12pt,
		int f_enlarged_page, int f_pagenumbers,
		const char *extras_for_preamble);
	void foot(std::ostream& ost);

	// two functions from DISCRETA1:

	void incma_latex_picture(std::ostream &fp,
		int width, int width_10,
		int f_outline_thin, const char *unit_length,
		const char *thick_lines, const char *thin_lines,
		const char *geo_line_width,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *R, int *X, int dim_X,
		int f_labelling_points, const char **point_labels,
		int f_labelling_blocks, const char **block_labels);
	// width for one box in 0.1mm
	// width_10 is 1 10th of width
	// example: width = 40, width_10 = 4 */
	void incma_latex(std::ostream &fp,
		int v, int b,
		int V, int B, int *Vi, int *Bj,
		int *R, int *X, int dim_X);
	void incma_latex_override_unit_length(const char *override_unit_length);
	void incma_latex_override_unit_length_drop();
	void print_01_matrix_tex(std::ostream &ost, int *p, int m, int n);
	void print_integer_matrix_tex(std::ostream &ost, int *p, int m, int n);
	void print_lint_matrix_tex(std::ostream &ost,
		long int *p, int m, int n);
	void print_longinteger_matrix_tex(std::ostream &ost,
		longinteger_object *p, int m, int n);
	void print_integer_matrix_with_labels(std::ostream &ost, int *p,
		int m, int n, int *row_labels, int *col_labels, int f_tex);
	void print_lint_matrix_with_labels(std::ostream &ost,
		long int *p, int m, int n, long int *row_labels, long int *col_labels,
		int f_tex);
	void print_integer_matrix_with_standard_labels(std::ostream &ost,
		int *p, int m, int n, int f_tex);
	void print_lint_matrix_with_standard_labels(std::ostream &ost,
		long int *p, int m, int n, int f_tex);
	void print_integer_matrix_with_standard_labels_and_offset(std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset, int f_tex);
	void print_lint_matrix_with_standard_labels_and_offset(std::ostream &ost,
		long int *p, int m, int n, int m_offset, int n_offset, int f_tex);
	void print_integer_matrix_tex_block_by_block(std::ostream &ost,
		int *p, int m, int n, int block_width);
	void print_integer_matrix_with_standard_labels_and_offset_text(std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset);
	void print_lint_matrix_with_standard_labels_and_offset_text(
		std::ostream &ost, long int *p, int m, int n, int m_offset, int n_offset);
	void print_integer_matrix_with_standard_labels_and_offset_tex(std::ostream &ost,
		int *p, int m, int n, int m_offset, int n_offset);
	void print_lint_matrix_with_standard_labels_and_offset_tex(
		std::ostream &ost, long int *p, int m, int n,
		int m_offset, int n_offset);
	void print_big_integer_matrix_tex(std::ostream &ost, int *p, int m, int n);
	void int_vec_print_as_matrix(std::ostream &ost,
		int *v, int len, int width, int f_tex);
	void lint_vec_print_as_matrix(std::ostream &ost,
		long int *v, int len, int width, int f_tex);
	void int_matrix_print_with_labels_and_partition(std::ostream &ost,
		int *p, int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(int *p, int m, int n,
			int i, int j, int val, std::string &output, void *data),
		void *data,
		int f_tex);
	void lint_matrix_print_with_labels_and_partition(std::ostream &ost,
		long int *p, int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(long int *p, int m, int n,
			int i, int j, int val, std::string &output, void *data),
		void *data,
		int f_tex);
	void int_matrix_print_tex(std::ostream &ost, int *p, int m, int n);
	void lint_matrix_print_tex(std::ostream &ost, long int *p, int m, int n);
	void print_cycle_tex_with_special_point_labels(
			std::ostream &ost, int *pts, int nb_pts,
			void (*point_label)(std::stringstream &sstr, int pt, void *data),
			void *point_label_data);
	void int_set_print_tex(std::ostream &ost, int *v, int len);
	void lint_set_print_tex(std::ostream &ost, long int *v, int len);
	void print_type_vector_tex(std::ostream &ost, int *v, int len);
	void int_set_print_masked_tex(std::ostream &ost,
		int *v, int len, const char *mask_begin, const char *mask_end);
	void lint_set_print_masked_tex(std::ostream &ost,
		long int *v, int len,
		const char *mask_begin,
		const char *mask_end);
	void int_set_print_tex_for_inline_text(std::ostream &ost,
			int *v, int len);
	void lint_set_print_tex_for_inline_text(std::ostream &ost,
			long int *v, int len);
	void latexable_string(std::stringstream &str,
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
// mem_object_registry_entry.cpp
// #############################################################################

//! a class related to mem_object_registry


class mem_object_registry_entry {
public:
	int time_stamp;
	void *pointer;
	int object_type;
	long int object_n;
	int object_size_of;
		// needed for objects of type class
	const char *extra_type_info;
	const char *source_file;
	int source_line;

	mem_object_registry_entry();
	~mem_object_registry_entry();
	void null();
	void set_type_from_string(char *str);
	void print_type(std::ostream &ost);
	int size_of();
	void print(int line);
	void print_csv(std::ostream &ost, int line);
};

extern int f_memory_debug;
extern int memory_debug_verbose_level;
extern mem_object_registry global_mem_object_registry;

void start_memory_debug();
void stop_memory_debug();


// #############################################################################
// mem_object_registry.cpp:
// #############################################################################




#define REGISTRY_SIZE 1000
#define POINTER_TYPE_INVALID 0
#define POINTER_TYPE_int 1
#define POINTER_TYPE_pint 2
#define POINTER_TYPE_lint 3
#define POINTER_TYPE_plint 4
#define POINTER_TYPE_ppint 5
#define POINTER_TYPE_pplint 6
#define POINTER_TYPE_char 7
#define POINTER_TYPE_uchar 8
#define POINTER_TYPE_pchar 9
#define POINTER_TYPE_puchar 10
#define POINTER_TYPE_PVOID 11
#define POINTER_TYPE_OBJECT 12
#define POINTER_TYPE_OBJECTS 13



//! maintains a registry of allocated memory



class mem_object_registry {
public:
	int f_automatic_dump;
	int automatic_dump_interval;
	char automatic_dump_fname_mask[1000];

	int nb_entries_allocated;
	int nb_entries_used;
	mem_object_registry_entry *entries;
		// entries are sorted by
		// the value of the pointer

	int nb_allocate_total;
		// total number of allocations
	int nb_delete_total;
		// total number of deletions
	int cur_time;
		// increments with every allocate and every delete

	int f_ignore_duplicates;
		// do not complain about duplicate entries
	int f_accumulate;
		// do not remove entries when deleting memory

	mem_object_registry();
	~mem_object_registry();
	void init(int verbose_level);
	void accumulate_and_ignore_duplicates(int verbose_level);
	void allocate(int N, int verbose_level);
	void set_automatic_dump(
			int automatic_dump_interval, const char *fname_mask,
			int verbose_level);
	void automatic_dump();
	void manual_dump();
	void manual_dump_with_file_name(const char *fname);
	void dump();
	void dump_to_csv_file(const char *fname);
	int *allocate_int(long int n, const char *file, int line);
	void free_int(int *p, const char *file, int line);
	int **allocate_pint(long int n, const char *file, int line);
	void free_pint(int **p, const char *file, int line);
	long int *allocate_lint(long int n, const char *file, int line);
	void free_lint(long int *p, const char *file, int line);
	long int **allocate_plint(long int n, const char *file, int line);
	void free_plint(long int **p, const char *file, int line);
	int ***allocate_ppint(long int n, const char *file, int line);
	void free_ppint(int ***p, const char *file, int line);
	long int ***allocate_pplint(long int n, const char *file, int line);
	void free_pplint(long int ***p, const char *file, int line);
	char *allocate_char(long int n, const char *file, int line);
	void free_char(char *p, const char *file, int line);
	uchar *allocate_uchar(long int n, const char *file, int line);
	void free_uchar(uchar *p, const char *file, int line);
	char **allocate_pchar(long int n, const char *file, int line);
	void free_pchar(char **p, const char *file, int line);
	uchar **allocate_puchar(long int n, const char *file, int line);
	void free_puchar(uchar **p, const char *file, int line);
	void **allocate_pvoid(long int n, const char *file, int line);
	void free_pvoid(void **p, const char *file, int line);
	void *allocate_OBJECTS(void *p, long int n, std::size_t size_of,
			const char *extra_type_info, const char *file, int line);
	void free_OBJECTS(void *p, const char *file, int line);
	void *allocate_OBJECT(void *p, std::size_t size_of,
			const char *extra_type_info, const char *file, int line);
	void free_OBJECT(void *p, const char *file, int line);
	int search(void *p, int &idx);
	void insert_at(int idx);
	void add_to_registry(void *pointer,
			int object_type, long int object_n, int object_size_of,
			const char *extra_type_info,
			const char *source_file, int source_line,
			int verbose_level);
	void delete_from_registry(void *pointer, int verbose_level);
	void sort_by_size(int verbose_level);
	void sort_by_location_and_get_frequency(int verbose_level);
	void sort_by_type(int verbose_level);
	void sort_by_location(int verbose_level);
};



// #############################################################################
// memory_object.cpp
// #############################################################################

//! for serialization of complex data types




class memory_object {
public:
	memory_object();
	~memory_object();
	void null();
	void freeself();

	char *data;
	long int alloc_length;
		// maintained by alloc()
	long int used_length;
	long int cur_pointer;


	char & s_i(int i) { return data[i]; };
	void init(long int length, char *initial_data, int verbose_level);
	void alloc(long int length, int verbose_level);
	void append(long int length, char *d, int verbose_level);
	void realloc(long int &new_length, int verbose_level);
	void write_char(char c);
	void read_char(char *c);
	void write_string(const char *p);
	void read_string(char *&p);
	void write_double(double f);
	void read_double(double *f);
	void write_lint(long int i);
	void read_lint(long int *i);
	void write_int(int i);
	void read_int(int *i);
	void read_file(const char *fname, int verbose_level);
	void write_file(const char *fname, int verbose_level);
	int multiplicity_of_character(char c);
};

// #############################################################################
// orbiter_data_file.cpp
// #############################################################################



//! read output files from the poset classification



class orbiter_data_file {
public:
	int nb_cases;
	long int **sets;
	int *set_sizes;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	
	orbiter_data_file();
	~orbiter_data_file();
	void null();
	void freeself();
	void load(std::string &fname, int verbose_level);
};

// #############################################################################
// os_interface.cpp
// #############################################################################


//! interface to system functions

class os_interface {
public:

	void runtime(long *l);
	int os_memory_usage();
	int os_ticks();
	int os_ticks_system();
	int os_ticks_per_second();
	void os_ticks_to_dhms(int ticks, int tps,
			int &d, int &h, int &m, int &s);
	void time_check_delta(std::ostream &ost, int dt);
	void print_elapsed_time(std::ostream &ost, int d, int h, int m, int s);
	void time_check(std::ostream &ost, int t0);
	int delta_time(int t0);
	void seed_random_generator_with_system_time();
	void seed_random_generator(int seed);
	int random_integer(int p);
	int os_seconds_past_1970();
	void get_string_from_command_line(std::string &p, int argc, std::string *argv,
			int &i, int verbose_level);
	void test_swap();
	void block_swap_chars(char *ptr, int size, int no);
	void code_int4(char *&p, int_4 i);
	int_4 decode_int4(char *&p);
	void code_uchar(char *&p, uchar a);
	void decode_uchar(char *&p, uchar &a);
	void get_date(std::string &str);
	void test_typedefs();

};


// #############################################################################
// override_double.cpp
// #############################################################################


//! to temporarily override a double variable with a new value

class override_double {
public:
	double *p;
	double original_value;
	double new_value;

	override_double(double *p, double value);
	~override_double();
};


// #############################################################################
// prepare_frames.cpp
// #############################################################################


//! to prepare the frames of a video

class prepare_frames {
public:

	int nb_inputs;
	int input_first[1000];
	int input_len[1000];
	std::string input_mask[1000];
	int f_o;
	std::string output_mask;
	int f_output_starts_at;
	int output_starts_at;
	int f_step;
	int step;

	prepare_frames();
	~prepare_frames();
	int parse_arguments(int argc, std::string *argv, int verbose_level);
	void print();
	void print_item(int i);
	void do_the_work(int verbose_level);
};



// #############################################################################
// util.cpp
// #############################################################################





const char *strip_directory(const char *p);
int is_all_whitespace(const char *str);
int is_all_digits(char *p);
int str2int(std::string &str);
void print_longinteger_after_multiplying(std::ostream &ost, int *factors, int len);
int my_atoi(char *str);
long int my_atol(char *str);
int compare_strings(void *a, void *b, void *data);
int strcmp_with_or_without(char *p, char *q);
uint32_t root_of_tree_uint32_t (uint32_t* S, uint32_t i);
int util_compare_func(void *a, void *b, void *data);
void text_to_three_double(const char *text, double *d);
void text_to_three_double(std::string &text, double *d);
int stringcmp(std::string &str, const char *p);
int strtoi(std::string &str);
long int strtolint(std::string &str);
double strtof(std::string &str);
int string_starts_with_a_number(std::string &str);
void dump_memory_chain(void *allocated_objects);
void print_vector(std::ostream &ost, int *v, int size);
void itoa(char *p, int len_of_p, int i);
void char_swap(char *p, char *q, int len);
void int_swap(int& x, int& y);
void uchar_print_bitwise(std::ostream &ost, uchar u);
void uchar_move(uchar *p, uchar *q, int len);


void print_set(std::ostream &ost, int size, long int *set);
void print_set_lint(std::ostream &ost, int size, long int *set);
void print_incidence_structure(std::ostream &ost,
		int m, int n, int len, int *S);


void print_line_of_number_signs();
void print_repeated_character(std::ostream &ost, char c, int n);
void print_pointer_hex(std::ostream &ost, void *p);
void print_hex_digit(std::ostream &ost, int digit);
int compare_sets(int *set1, int *set2, int sz1, int sz2);
int hashing(int hash0, int a);
int hashing_fixed_width(int hash0, int a, int bit_length);




}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_IO_AND_OS_IO_AND_OS_H_ */




