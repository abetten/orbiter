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

//! to create files

#define MAX_LINES 100


class create_file_description {
public:
	int f_file_mask;
	const char *file_mask;
	int f_N;
	int N;
	int nb_lines;
	const char *lines[MAX_LINES];
	int f_line_numeric[MAX_LINES];
	int nb_final_lines;
	const char *final_lines[MAX_LINES];
	int f_command;
	const char *command;
	int f_repeat;
	int repeat_N;
	int repeat_start;
	int repeat_increment;
	const char *repeat_mask;
	int f_split;
	int split_m;
	int f_read_cases;
	const char *read_cases_fname;
	int f_read_cases_text;
	int read_cases_column_of_case;
	int read_cases_column_of_fname;
	int f_tasks;
	int nb_tasks;
	const char *tasks_line;

	create_file_description();
	~create_file_description();
	int read_arguments(
		int argc, const char **argv,
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
		const char *fname, int orbit_at_level,
		long int *&candidates, int &nb_candidates, int verbose_level);
	void read_candidates_for_one_orbit_from_file(char *prefix,
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
	int find_orbit_index_in_data_file(const char *prefix,
			int level_of_candidates_file, long int *starter,
			int verbose_level);
	void write_exact_cover_problem_to_file(int *Inc, int nb_rows,
		int nb_cols, const char *fname);
	void read_solution_file(char *fname,
		int *Inc, int nb_rows, int nb_cols,
		int *&Solutions, int &sol_length, int &nb_sol,
		int verbose_level);
	void count_number_of_solutions_in_file_and_get_solution_size(
		const char *fname,
		int &nb_solutions, int &solution_size,
		int verbose_level);
	void count_number_of_solutions_in_file(const char *fname,
		int &nb_solutions,
		int verbose_level);
	void count_number_of_solutions_in_file_by_case(const char *fname,
		int *&nb_solutions, int *&case_nb, int &nb_cases,
		int verbose_level);
	void read_solutions_from_file_and_get_solution_size(const char *fname,
		int &nb_solutions, int *&Solutions, int &solution_size,
		int verbose_level);
	void read_solutions_from_file(const char *fname,
		int &nb_solutions, int *&Solutions, int solution_size,
		int verbose_level);
	void read_solutions_from_file_by_case(const char *fname,
		int *nb_solutions, int *case_nb, int nb_cases,
		int **&Solutions, int solution_size,
		int verbose_level);
	void copy_file_to_ostream(std::ostream &ost, char *fname);
	void int_vec_write_csv(int *v, int len,
		const char *fname, const char *label);
	void lint_vec_write_csv(long int *v, int len,
		const char *fname, const char *label);
	void int_vecs_write_csv(int *v1, int *v2, int len,
		const char *fname,
		const char *label1, const char *label2);
	void int_vecs3_write_csv(int *v1, int *v2, int *v3, int len,
		const char *fname,
		const char *label1, const char *label2, const char *label3);
	void int_vec_array_write_csv(int nb_vecs, int **Vec, int len,
		const char *fname, const char **column_label);
	void lint_vec_array_write_csv(int nb_vecs, long int **Vec, int len,
		const char *fname, const char **column_label);
	void int_matrix_write_csv(const char *fname, int *M, int m, int n);
	void lint_matrix_write_csv(const char *fname, long int *M, int m, int n);
	void double_matrix_write_csv(const char *fname,
		double *M, int m, int n);
	void int_matrix_write_csv_with_labels(const char *fname,
		int *M, int m, int n, const char **column_label);
	void lint_matrix_write_csv_with_labels(const char *fname,
		long int *M, int m, int n, const char **column_label);
	void int_matrix_read_csv(const char *fname, int *&M,
		int &m, int &n, int verbose_level);
	void lint_matrix_read_csv(const char *fname,
		long int *&M, int &m, int &n, int verbose_level);
	void double_matrix_read_csv(const char *fname, double *&M,
		int &m, int &n, int verbose_level);
	void int_matrix_write_cas_friendly(const char *fname, int *M, int m, int n);
	void int_matrix_write_text(const char *fname,
		int *M, int m, int n);
	void lint_matrix_write_text(const char *fname, long int *M, int m, int n);
	void int_matrix_read_text(const char *fname,
		int *&M, int &m, int &n);
	void parse_sets(int nb_cases, char **data, int f_casenumbers,
		int *&Set_sizes, long int **&Sets, char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void parse_sets_and_check_sizes_easy(int len, int nb_cases,
		char **data, long int **&sets);
	void parse_line(char *line, int &len, long int *&set,
		char *ago_ascii, char *aut_ascii);
	int count_number_of_orbits_in_file(const char *fname, int verbose_level);
	int count_number_of_lines_in_file(const char *fname, int verbose_level);
	int try_to_read_file(const char *fname, int &nb_cases,
		char **&data, int verbose_level);
	void read_and_parse_data_file(const char *fname, int &nb_cases,
		char **&data, long int **&sets, int *&set_sizes, int verbose_level);
	void free_data_fancy(int nb_cases,
		int *Set_sizes, long int **Sets,
		char **Ago_ascii, char **Aut_ascii,
		int *Casenumbers);
	void read_and_parse_data_file_fancy(const char *fname,
		int f_casenumbers,
		int &nb_cases,
		int *&Set_sizes, long int **&Sets, char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void read_set_from_file(const char *fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file(const char *fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_lint(const char *fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_lint(const char *fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_int4(const char *fname,
		long int *&the_set, int &set_size, int verbose_level);
	void read_set_from_file_int8(const char *fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_as_int4(const char *fname,
		long int *the_set, int set_size, int verbose_level);
	void write_set_to_file_as_int8(const char *fname,
		long int *the_set, int set_size, int verbose_level);
	void read_k_th_set_from_file(const char *fname, int k,
		int *&the_set, int &set_size, int verbose_level);
	void write_incidence_matrix_to_file(char *fname,
		int *Inc, int m, int n, int verbose_level);
	void read_incidence_matrix_from_inc_file(int *&M, int &m, int &n,
		char *inc_file_name, int inc_file_idx, int verbose_level);
	int inc_file_get_number_of_geometries(
		char *inc_file_name, int verbose_level);
	long int file_size(const char *name);
	void delete_file(const char *fname);
	void fwrite_int4(FILE *fp, int a);
	int_4 fread_int4(FILE *fp);
	void fwrite_uchars(FILE *fp, uchar *p, int len);
	void fread_uchars(FILE *fp, uchar *p, int len);
	void read_numbers_from_file(const char *fname,
		int *&the_set, int &set_size, int verbose_level);
	void read_ascii_set_of_sets_constant_size(
			const char *fname_ascii,
			int *&Sets, int &nb_sets, int &set_size, int verbose_level);
	void write_decomposition_stack(char *fname, int m, int n,
			int *v, int *b, int *aij, int verbose_level);
	void create_file(create_file_description *Descr, int verbose_level);
	void create_files(create_file_description *Descr,
		int verbose_level);
	void create_files_list_of_cases(spreadsheet *S,
			create_file_description *Descr, int verbose_level);
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
	void int_matrix_print_with_labels_and_partition(std::ostream &ost, int *p,
		int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(int *p, int m, int n, int i, int j,
			int val, char *output, void *data),
		void *data,
		int f_tex);
	void lint_matrix_print_with_labels_and_partition(std::ostream &ost,
		long int *p, int m, int n,
		int *row_labels, int *col_labels,
		int *row_part_first, int *row_part_len, int nb_row_parts,
		int *col_part_first, int *col_part_len, int nb_col_parts,
		void (*process_function_or_NULL)(long int *p, int m, int n,
			int i, int j, int val, char *output, void *data),
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
	void load(const char *fname, int verbose_level);
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
	void os_date_string(char *str, int sz);
	int os_seconds_past_1970();

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
	const char *input_mask[1000];
	int f_o;
	const char *output_mask;
	int f_output_starts_at;
	int output_starts_at;
	int f_step;
	int step;

	prepare_frames();
	~prepare_frames();
	int parse_arguments(int argc, const char **argv);
	void do_the_work(int verbose_level);
};



// #############################################################################
// util.cpp
// #############################################################################


void int_vec_add(int *v1, int *v2, int *w, int len);
void int_vec_add3(int *v1, int *v2, int *v3, int *w, int len);
void int_vec_apply(int *from, int *through, int *to, int len);
void int_vec_apply_lint(int *from, long int *through, long int *to, int len);
void lint_vec_apply(long int *from, long int *through, long int *to, int len);
int int_vec_is_constant_on_subset(int *v, int *subset, int sz, int &value);
void int_vec_take_away(int *v, int &len, int *take_away, int nb_take_away);
	// v must be sorted
void lint_vec_take_away(long int *v, int &len,
		long int *take_away, int nb_take_away);
int int_vec_count_number_of_nonzero_entries(int *v, int len);
int int_vec_find_first_nonzero_entry(int *v, int len);
void int_vec_zero(int *v, int len);
void lint_vec_zero(long int *v, int len);
void int_vec_mone(int *v, int len);
void lint_vec_mone(long int *v, int len);
void int_vec_copy(int *from, int *to, int len);
void lint_vec_copy(long int *from, long int *to, int len);
void int_vec_copy_to_lint(int *from, long int *to, int len);
void lint_vec_copy_to_int(long int *from, int *to, int len);
void int_vec_swap(int *v1, int *v2, int len);
void int_vec_delete_element_assume_sorted(int *v, int &len, int a);
uchar *bitvector_allocate(long int length);
uchar *bitvector_allocate_and_coded_length(long int length, int &coded_length);
void bitvector_m_ii(uchar *bitvec, long int i, int a);
void bitvector_set_bit(uchar *bitvec, long int i);
int bitvector_s_i(uchar *bitvec, long int i);
// returns 0 or 1
uint32_t int_vec_hash(int *data, int len);
uint32_t lint_vec_hash(long int *data, int len);
uint32_t char_vec_hash(char *data, int len);
int int_vec_hash_after_sorting(int *data, int len);
int lint_vec_hash_after_sorting(long int *data, int len);
void int_vec_complement(int *v, int n, int k);
// computes the complement to v + k (v must be allocated to n lements)
void int_vec_complement(int *v, int *w, int n, int k);
// computes the complement of v[k] w[n - k] 
void lint_vec_complement(long int *v, long int *w, int n, int k);
void int_vec_init5(int *v, int a0, int a1, int a2, int a3, int a4);
void dump_memory_chain(void *allocated_objects);
void print_vector(std::ostream &ost, int *v, int size);
int int_vec_minimum(int *v, int len);
long int lint_vec_minimum(long int *v, int len);
int int_vec_maximum(int *v, int len);
long int lint_vec_maximum(long int *v, int len);
int int_vec_first_difference(int *p, int *q, int len);
void itoa(char *p, int len_of_p, int i);
void char_swap(char *p, char *q, int len);
void int_vec_distribution_compute_and_print(std::ostream &ost, int *v, int v_len);
void int_vec_distribution(int *v, int len_v, int *&val, int *&mult, int &len);
void int_distribution_print(std::ostream &ost, int *val, int *mult, int len);
void int_swap(int& x, int& y);
void int_set_print(int *v, int len);
void lint_set_print(long int *v, int len);
void int_set_print(std::ostream &ost, int *v, int len);
void lint_set_print(std::ostream &ost, long int *v, int len);
void int_vec_print(std::ostream &ost, int *v, int len);
void lint_vec_print(std::ostream &ost, long int *v, int len);
void int_vec_print_str(std::stringstream &ost, int *v, int len);
void int_vec_print_as_table(std::ostream &ost, int *v, int len, int width);
void lint_vec_print_as_table(std::ostream &ost, long int *v, int len, int width);
void int_vec_print_fully(std::ostream &ost, int *v, int len);
void lint_vec_print_fully(std::ostream &ost, long int *v, int len);
void int_vec_print_Cpp(std::ostream &ost, int *v, int len);
void int_vec_print_GAP(std::ostream &ost, int *v, int len);
void int_vec_print_classified(int *v, int len);
void int_vec_print_classified_str(std::stringstream &sstr,
		int *v, int len, int f_backwards);
void integer_vec_print(std::ostream &ost, int *v, int len);
void print_integer_matrix(std::ostream &ost, int *p, int m, int n);
void print_integer_matrix_width(std::ostream &ost, int *p,
	int m, int n, int dim_n, int w);
void lint_matrix_print_width(std::ostream &ost,
	long int *p, int m, int n, int dim_n, int w);
void int_matrix_make_block_matrix_2x2(int *Mtx, int k, 
	int *A, int *B, int *C, int *D);
// makes the 2k x 2k block matrix 
// (A B)
// (C D)
void int_matrix_delete_column_in_place(int *Mtx, int k, int n, int pivot);
// afterwards, the matrix is k x (n - 1)
int int_matrix_max_log_of_entries(int *p, int m, int n);
int lint_matrix_max_log_of_entries(long int *p, int m, int n);
void int_matrix_print_ost(std::ostream &ost, int *p, int m, int n);
void int_matrix_print(int *p, int m, int n);
void lint_matrix_print(long int *p, int m, int n);
void int_matrix_print_tight(int *p, int m, int n);
void int_matrix_print_ost(std::ostream &ost, int *p, int m, int n, int w);
void int_matrix_print(int *p, int m, int n, int w);
void lint_matrix_print(long int *p, int m, int n, int w);
void int_matrix_print_bitwise(int *p, int m, int n);
void uchar_print_bitwise(std::ostream &ost, uchar u);
void uchar_move(uchar *p, uchar *q, int len);
void int_submatrix_all_rows(int *A, int m, int n, 
	int nb_cols, int *cols, int *B);
void int_submatrix_all_cols(int *A, int m, int n, 
	int nb_rows, int *rows, int *B);
void int_submatrix(int *A, int m, int n, int nb_rows, 
	int *rows, int nb_cols, int *cols, int *B);
void int_matrix_transpose(int n, int *A);
void int_matrix_transpose(int *M, int m, int n, int *Mt);
// Mt must point to the right amount of memory (n * m int's)
void int_matrix_shorten_rows(int *&p, int m, int n);
void pint_matrix_shorten_rows(pint *&p, int m, int n);



void print_set(std::ostream &ost, int size, long int *set);
void print_set_lint(std::ostream &ost, int size, long int *set);
void block_swap_chars(char *ptr, int size, int no);
void code_int4(char *&p, int_4 i);
int_4 decode_int4(char *&p);
void code_uchar(char *&p, uchar a);
void decode_uchar(char *&p, uchar &a);
void print_incidence_structure(std::ostream &ost,
		int m, int n, int len, int *S);
void int_vec_scan(const char *s, int *&v, int &len);
void lint_vec_scan(const char *s, long int *&v, int &len);
void int_vec_scan_from_stream(std::istream & is, int *&v, int &len);
void lint_vec_scan_from_stream(std::istream & is, long int *&v, int &len);
void scan_permutation_from_string(const char *s, 
	int *&perm, int &degree, int verbose_level);
void scan_permutation_from_stream(std::istream & is,
	int *&perm, int &degree, int verbose_level);
char get_character(std::istream & is, int verbose_level);
void replace_extension_with(char *p, const char *new_ext);
void chop_off_extension(char *p);
void chop_off_extension_if_present(char *p, const char *ext);
void get_fname_base(const char *p, char *fname_base);
void get_extension_if_present(const char *p, char *ext);
void get_extension_if_present_and_chop_off(char *p, char *ext);
int s_scan_int(char **s, int *i);
int s_scan_lint(char **s, long int *i);
int s_scan_double(char **s, double *d);
int s_scan_token(char **s, char *str);
int s_scan_token_arbitrary(char **s, char *str);
int s_scan_str(char **s, char *str);
int s_scan_token_comma_separated(char **s, char *str);
int hashing(int hash0, int a);
int hashing_fixed_width(int hash0, int a, int bit_length);
int int_vec_hash(int *v, int len, int bit_length);
void print_line_of_number_signs();
void print_repeated_character(std::ostream &ost, char c, int n);
void print_pointer_hex(std::ostream &ost, void *p);
void print_hex_digit(std::ostream &ost, int digit);
int compare_sets(int *set1, int *set2, int sz1, int sz2);
//int test_if_sets_are_disjoint(int *set1, int *set2, int sz1, int sz2);
int test_if_sets_are_disjoint_assuming_sorted(int *set1, int *set2, int sz1, int sz2);
int test_if_sets_are_disjoint_assuming_sorted_lint(long int *set1, long int *set2, int sz1, int sz2);
void int_vec_print_to_str(char *str, int *data, int len);
void lint_vec_print_to_str(char *str, long int *data, int len);
void int_vec_print_to_str_naked(char *str, int *data, int len);
void lint_vec_print_to_str_naked(char *str, long int *data, int len);
int is_csv_file(const char *fname);
int is_xml_file(const char *fname);

void test_typedefs();
void chop_string(const char *str, int &argc, char **&argv);
const char *strip_directory(const char *p);
int is_all_whitespace(const char *str);
int is_all_digits(char *p);
void int_vec_print(int *v, int len);
int str2int(std::string &str);
void print_longinteger_after_multiplying(std::ostream &ost, int *factors, int len);
int my_atoi(char *str);
long int my_atol(char *str);
int compare_strings(void *a, void *b, void *data);
int strcmp_with_or_without(char *p, char *q);
uint32_t root_of_tree_uint32_t (uint32_t* S, uint32_t i);
int util_compare_func(void *a, void *b, void *data);
void text_to_three_double(const char *text, double *d);




}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_IO_AND_OS_IO_AND_OS_H_ */




