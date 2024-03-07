// orbiter_kernel_system.h
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
namespace layer1_foundations {
namespace orbiter_kernel_system {



// #############################################################################
// create_file_description.cpp
// #############################################################################


#define MAX_LINES 100

//! instructions to create text files


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
	void print();

};



// #############################################################################
// csv_file_support.cpp
// #############################################################################

//! support for csv files




class csv_file_support {
public:

	file_io *Fio;

	csv_file_support();
	~csv_file_support();

	void init(
			file_io *Fio);
	void int_vec_write_csv(
			int *v, int len,
			std::string &fname, std::string &label);
	void lint_vec_write_csv(
			long int *v, int len,
			std::string &fname, std::string &label);
	void int_vecs_write_csv(
			int *v1, int *v2, int len,
			std::string &fname, std::string &label1, std::string &label2);
	void int_vecs3_write_csv(
			int *v1, int *v2, int *v3, int len,
			std::string &fname,
			std::string &label1, std::string &label2, std::string &label3);
	void int_vec_array_write_csv(
			int nb_vecs, int **Vec, int len,
			std::string &fname, std::string *column_label);
	void lint_vec_array_write_csv(
			int nb_vecs, long int **Vec, int len,
			std::string &fname, std::string *column_label);
	void int_matrix_write_csv(
			std::string &fname, int *M, int m, int n);
	void lint_matrix_write_csv(
			std::string &fname,
			long int *M, int m, int n);
	void lint_matrix_write_csv_override_headers(
			std::string &fname,
			std::string *headers, long int *M, int m, int n);
	void vector_write_csv(
			std::string &fname,
			std::vector<int > &V);
	void vector_matrix_write_csv(
			std::string &fname,
			std::vector<std::vector<int> > &V);
	void double_matrix_write_csv(
			std::string &fname,
		double *M, int m, int n);
	void int_matrix_write_csv_with_labels(
			std::string &fname,
		int *M, int m, int n, const char **column_label);
	void lint_matrix_write_csv_with_labels(
			std::string &fname,
		long int *M, int m, int n, const char **column_label);
	void int_matrix_read_csv(
			std::string &fname, int *&M,
		int &m, int &n, int verbose_level);
	void int_matrix_read_csv_no_border(
			std::string &fname,
		int *&M, int &m, int &n, int verbose_level);
	void lint_matrix_read_csv_no_border(
			std::string &fname,
		long int *&M, int &m, int &n, int verbose_level);
	void int_matrix_read_csv_data_column(
			std::string &fname,
		int *&M, int &m, int &n, int col_idx, int verbose_level);
	void lint_matrix_read_csv_data_column(
			std::string &fname,
		long int *&M, int &m, int &n, int col_idx, int verbose_level);
	void lint_matrix_read_csv(
			std::string &fname,
		long int *&M, int &m, int &n, int verbose_level);
	void double_matrix_read_csv(
			std::string &fname, double *&M,
		int &m, int &n, int verbose_level);

	void do_csv_file_select_rows(
			std::string &fname,
			std::string &rows_text,
			int verbose_level);
	void do_csv_file_select_rows_complement(
			std::string &fname,
			std::string &rows_text,
			int verbose_level);
	void do_csv_file_select_rows_worker(
			std::string &fname,
			std::string &rows_text,
			int f_complement,
			int verbose_level);
	void do_csv_file_split_rows_modulo(
			std::string &fname,
			int split_modulo,
			int verbose_level);
	void do_csv_file_select_cols(
			std::string &fname,
			std::string &fname_append,
			std::string &cols_text,
			int verbose_level);
	void do_csv_file_select_rows_and_cols(
			std::string &fname,
			std::string &rows_text, std::string &cols_text,
			int verbose_level);
	void do_csv_file_extract_column_to_txt(
			std::string &csv_fname, std::string &col_label,
			int verbose_level);
	void do_csv_file_sort_each_row(
			std::string &csv_fname, int verbose_level);
	void do_csv_file_join(
			std::vector<std::string> &csv_file_join_fname,
			std::vector<std::string> &csv_file_join_identifier,
			int verbose_level);
	void do_csv_file_concatenate(
			std::vector<std::string> &fname, std::string &fname_out,
			int verbose_level);
	void do_csv_file_concatenate_from_mask(
			std::string &fname_in_mask, int N, std::string &fname_out,
			int verbose_level);
	void do_csv_file_latex(std::string &fname,
			int f_produce_latex_header,
			int nb_lines_per_table,
			int verbose_level);
	void read_csv_file_and_tally(
			std::string &fname, int verbose_level);

	void grade_statistic_from_csv(
			std::string &fname_csv,
			int f_midterm1, std::string &midterm1_label,
			int f_midterm2, std::string &midterm2_label,
			int f_final, std::string &final_label,
			int f_oracle_grade, std::string &oracle_grade_label,
			int verbose_level);


	void csv_file_sort_rows(
			std::string &fname,
			int verbose_level);
	void csv_file_sort_rows_and_remove_duplicates(
			std::string &fname,
			int verbose_level);
	void write_table_of_strings(
			std::string &fname,
			int nb_rows, int nb_cols, std::string *Table,
			std::string &headings,
			int verbose_level);
	void write_table_of_strings_with_headings(
			std::string &fname,
			int nb_rows, int nb_cols, std::string *Table,
			std::string *Row_headings,
			std::string *Col_headings,
			int verbose_level);
	void write_table_of_strings_with_col_headings(
			std::string &fname,
			int nb_rows, int nb_cols, std::string *Table,
			std::string *Col_headings,
			int verbose_level);

	int read_column_and_count_nb_sets(
			std::string &fname, std::string &col_label,
			int verbose_level);
	void read_column_and_parse(
			std::string &fname, std::string &col_label,
			data_structures::set_of_sets *&SoS,
			int verbose_level);
#if 0
	void save_fibration(
			std::vector<std::vector<std::pair<int, int> > > &Fibration,
			std::string &fname, int verbose_level);
	void save_cumulative_canonical_labeling(
			std::vector<std::vector<int> > &Cumulative_canonical_labeling,
			std::string &fname, int verbose_level);
	void save_cumulative_ago(
			std::vector<long int> &Cumulative_Ago,
			std::string &fname, int verbose_level);
	void save_cumulative_data(
			std::vector<std::vector<int> > &Cumulative_data,
			std::string &fname, int verbose_level);
#endif
	void write_characteristic_matrix(
			std::string &fname,
			long int *data, int nb_rows, int data_sz, int nb_cols,
			int verbose_level);
	void split_by_values(
			std::string &fname_in, int verbose_level);
	void change_values(
			std::string &fname_in, std::string &fname_out,
			std::string &input_values,
			std::string &output_values,
			int verbose_level);
	void write_gedcom_file_as_csv(
			std::string &fname,
			std::vector<std::vector<std::string> > &Data,
		int verbose_level);
	void write_ancestry_indi(
			std::string &fname,
			std::vector<std::vector<std::string> > &Data,
			int nb_indi,
			data_structures::ancestry_indi **Individual,
		int verbose_level);
	void write_ancestry_family(
			std::string &fname,
			std::vector<std::vector<std::string> > &Data,
			int nb_indi,
			int nb_fam,
			data_structures::ancestry_indi **Individual,
			data_structures::ancestry_family **Family,
		int verbose_level);
	void read_table_of_strings(
			std::string &fname, std::string *&col_label,
			std::string *&Table, int &m, int &n,
			int verbose_level);


};




// #############################################################################
// file_io.cpp
// #############################################################################

//! a collection of functions related to file io




class file_io {
public:

	csv_file_support *Csv_file_support;


	file_io();
	~file_io();

	void concatenate_files(
			std::string &fname_in_mask, int N,
			std::string &fname_out, std::string &EOF_marker,
			int f_title_line,
		int &cnt_total,
		std::vector<int> missing_idx,
		int verbose_level);
	void concatenate_files_into(
			std::string &fname_in_mask, int N,
		std::ofstream &fp_out, std::string &EOF_marker,
		int f_title_line,
		int &cnt_total,
		std::vector<int> &missing_idx,
		int verbose_level);
	void poset_classification_read_candidates_of_orbit(
			std::string &fname, int orbit_at_level,
		long int *&candidates, int &nb_candidates, int verbose_level);
	void read_candidates_for_one_orbit_from_file(
			std::string &prefix,
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
	int find_orbit_index_in_data_file(
			std::string &prefix,
			int level_of_candidates_file, long int *starter,
			int verbose_level);
	void write_exact_cover_problem_to_file(
			int *Inc, int nb_rows,
		int nb_cols, std::string &fname);
	void read_solution_file(
			std::string &fname,
		int *Inc, int nb_rows, int nb_cols,
		int *&Solutions, int &sol_length, int &nb_sol,
		int verbose_level);
	void count_number_of_solutions_in_file_and_get_solution_size(
			std::string &fname,
		int &nb_solutions, int &solution_size,
		int verbose_level);
	void count_number_of_solutions_in_file(
			std::string &fname,
		int &nb_solutions,
		int verbose_level);
	void count_number_of_solutions_in_file_by_case(
			std::string &fname,
		int *&nb_solutions, int *&case_nb, int &nb_cases,
		int verbose_level);
	void read_solutions_from_file_and_get_solution_size(
			std::string &fname,
		int &nb_solutions, long int *&Solutions, int &solution_size,
		int verbose_level);
	void read_solutions_from_file(
			std::string &fname,
		int &nb_solutions, long int *&Solutions, int solution_size,
		int verbose_level);
	void read_solutions_from_file_size_is_known(
			std::string &fname,
		std::vector<std::vector<long int> > &Solutions,
		int solution_size,
		int verbose_level);
	void read_solutions_from_file_by_case(
			std::string &fname,
		int *nb_solutions, int *case_nb, int nb_cases,
		long int **&Solutions, int solution_size,
		int verbose_level);
	void copy_file_to_ostream(
			std::ostream &ost, std::string &fname);




	void int_matrix_write_cas_friendly(
			std::string &fname,
			int *M, int m, int n);
	void int_matrix_write_text(
			std::string &fname,
		int *M, int m, int n);
	void lint_matrix_write_text(
			std::string &fname,
			long int *M, int m, int n);
	void int_matrix_read_text(
			std::string &fname,
		int *&M, int &m, int &n);
	void read_dimacs_graph_format(
			std::string &fname,
			int &nb_V, std::vector<std::vector<int> > &Edges,
			int verbose_level);
	void parse_sets(
			int nb_cases, char **data, int f_casenumbers,
		int *&Set_sizes, long int **&Sets,
		char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void parse_sets_and_check_sizes_easy(
			int len, int nb_cases,
		char **data, long int **&sets);
	void parse_line(
			char *line, int &len, long int *&set,
		char *ago_ascii, char *aut_ascii);
	int count_number_of_orbits_in_file(
			std::string &fname, int verbose_level);
	int count_number_of_lines_in_file(
			std::string &fname, int verbose_level);
	int try_to_read_file(
			std::string &fname, int &nb_cases,
		char **&data,
		int verbose_level);
	void read_and_parse_data_file(
			std::string &fname, int &nb_cases,
		char **&data, long int **&sets, int *&set_sizes,
		int verbose_level);
	void free_data_fancy(
			int nb_cases,
		int *Set_sizes, long int **Sets,
		char **Ago_ascii, char **Aut_ascii,
		int *Casenumbers);
	void read_and_parse_data_file_fancy(
			std::string &fname,
		int f_casenumbers,
		int &nb_cases,
		int *&Set_sizes, long int **&Sets,
		char **&Ago_ascii, char **&Aut_ascii,
		int *&Casenumbers,
		int verbose_level);
	void read_set_from_file(
			std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file(
			std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_lint(
			std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_lint(
			std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_set_from_file_int4(
			std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void read_set_from_file_int8(
			std::string &fname,
		long int *&the_set, int &set_size, int verbose_level);
	void write_set_to_file_as_int4(
			std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void write_set_to_file_as_int8(
			std::string &fname,
		long int *the_set, int set_size, int verbose_level);
	void read_k_th_set_from_file(
			std::string &fname, int k,
		int *&the_set, int &set_size, int verbose_level);
	void write_incidence_matrix_to_file(
			std::string &fname,
		int *Inc, int m, int n, int verbose_level);
	void read_incidence_matrix_from_inc_file(
			int *&M, int &m, int &n,
			std::string &inc_file_name, int inc_file_idx,
			int verbose_level);
	void read_incidence_file(
			std::vector<std::vector<int> > &Geos,
			int &m, int &n, int &nb_flags,
			std::string &inc_file_name, int verbose_level);
	void read_incidence_by_row_ranks_file(
			std::vector<std::vector<int> > &Geos,
			int &m, int &n, int &r,
			std::string &inc_file_name, int verbose_level);
	int inc_file_get_number_of_geometries(
			std::string &inc_file_name, int verbose_level);
	long int file_size(
			std::string &fname);
	long int file_size(
			const char *name);
	void delete_file(
			std::string &fname);

	void fwrite_int4(
			FILE *fp, int a);
	int_4 fread_int4(
			FILE *fp);
	void fwrite_uchars(
			FILE *fp, unsigned char *p, int len);
	void fread_uchars(
			FILE *fp, unsigned char *p, int len);

	void read_ascii_set_of_sets_constant_size(
			std::string &fname_ascii,
			int *&Sets, int &nb_sets, int &set_size,
			int verbose_level);
	void write_decomposition_stack(
			std::string &fname, int m, int n,
			int *v, int *b, int *aij, int verbose_level);
	void create_file(
			create_file_description *Descr, int verbose_level);
	void create_files(
			create_file_description *Descr,
		int verbose_level);
	void create_files_list_of_cases(
			data_structures::spreadsheet *S,
			create_file_description *Descr,
			int verbose_level);
	int number_of_vertices_in_colored_graph(
			std::string &fname, int verbose_level);



	void read_solutions_and_tally(
			std::string &fname, int sz, int verbose_level);
	void extract_from_makefile(
			std::string &fname,
			std::string &label,
			int f_tail, std::string &tail,
			std::vector<std::string> &text,
			int verbose_level);




	void count_solutions_in_list_of_files(
			int nb_files, std::string *fname,
			int *List_of_cases, int *&Nb_sol_per_file,
			int solution_size,
			int f_has_final_test_function,
			int (*final_test_function)(long int *data, int sz,
					void *final_test_data, int verbose_level),
			void *final_test_data,
			int verbose_level);
	void read_file_as_array_of_strings(
			std::string &fname,
		std::string *&Lines,
		int &nb_lines,
		int verbose_level);
	void serialize_file_names(
		std::string &fname_list_of_file,
		std::string &output_mask,
		int &nb_files,
		int verbose_level);
	void read_error_pattern_from_output_file(
			std::string &fname,
			int nb_lines,
			std::vector<std::vector<int> > &Error1,
			std::vector<std::vector<int> > &Error2,
		int verbose_level);
	void read_gedcom_file(
			std::string &fname,
			std::vector<std::vector<std::string> > &Data,
		int verbose_level);
	void write_solutions_as_index_set(
			std::string &fname_solutions,
			int *Sol, int nb_sol, int width, int sum,
			int verbose_level);





};


// #############################################################################
// file_output.cpp
// #############################################################################


//! a wrapper class for an ofstream which allows to store extra data

class file_output {
public:
	std::string fname;
	int f_file_is_open;
	std::ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void open(
			std::string &fname,
			void *user_data,
			int verbose_level);
	void close();
	void write_line(
			int nb, int *data,
			int verbose_level);
	void write_EOF(
			int nb_sol, int verbose_level);
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
	void set_type_from_string(
			char *str);
	void print_type(
			std::ostream &ost);
	int size_of();
	void print(
			int line);
	void print_csv(
			std::ostream &ost, int line);
};


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
	std::string automatic_dump_fname_mask;

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
			int automatic_dump_interval, std::string &fname_mask,
			int verbose_level);
	void automatic_dump();
	void manual_dump();
	void manual_dump_with_file_name(std::string &fname);
	void dump();
	void dump_to_csv_file(std::string &fname);
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

	char *data;
	long int alloc_length;
		// maintained by alloc()
	long int used_length;
	long int cur_pointer;


	char & s_i(
			int i) { return data[i]; };
	void init(
			long int length, char *initial_data, int verbose_level);
	void alloc(
			long int length, int verbose_level);
	void append(
			long int length, char *d, int verbose_level);
	void realloc(
			long int &new_length, int verbose_level);
	void write_char(
			char c);
	void read_char(
			char *c);
	void write_string(
			const char *p);
	void write_string(
			std::string &p);
	void read_string(
			std::string &p);
	void write_double(
			double f);
	void read_double(
			double *f);
	void write_lint(
			long int i);
	void read_lint(
			long int *i);
	void write_int(
			int i);
	void read_int(
			int *i);
	void read_file(
			std::string &fname, int verbose_level);
	void write_file(
			std::string &fname, int verbose_level);
	int multiplicity_of_character(
			char c);
};


// #############################################################################
// numerics.cpp
// #############################################################################



//! numerical functions, mostly concerned with double

class numerics {
public:
	numerics();
	~numerics();
	void vec_print(
			double *a, int len);
	void vec_linear_combination1(
			double c1, double *v1,
			double *w, int len);
	void vec_linear_combination(
			double c1, double *v1,
			double c2, double *v2, double *v3, int len);
	void vec_linear_combination3(
			double c1, double *v1,
			double c2, double *v2,
			double c3, double *v3,
			double *w, int len);
	void vec_add(
			double *a, double *b, double *c, int len);
	void vec_subtract(
			double *a, double *b, double *c, int len);
	void vec_scalar_multiple(
			double *a, double lambda, int len);
	int Gauss_elimination(
			double *A, int m, int n,
		int *base_cols, int f_complete,
		int verbose_level);
	void print_system(
			double *A, int m, int n);
	void get_kernel(
			double *M, int m, int n,
		int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n,
		double *kernel);
	// kernel must point to the appropriate amount of memory!
	// (at least n * (n - nb_base_cols) doubles)
	// m is not used!
	int Null_space(
			double *M, int m, int n, double *K,
		int verbose_level);
	// K will be k x n
	// where k is the return value.
	void vec_normalize_from_back(
			double *v, int len);
	void vec_normalize_to_minus_one_from_back(
			double *v, int len);
	int triangular_prism(
			double *P1, double *P2, double *P3,
		double *abc3, double *angles3, double *T3,
		int verbose_level);
	int general_prism(
			double *Pts, int nb_pts, double *Pts_xy,
		double *abc3, double *angles3, double *T3,
		int verbose_level);
	void mult_matrix(
			double *v, double *R, double *vR);
	void mult_matrix_matrix(
			double *A, double *B, double *C,
			int m, int n, int o);
	// A is m x n, B is n x o, C is m x o
	void print_matrix_3x3(
			double *R);
	void print_matrix(
			double *R, int m, int n);
	void make_Rz(
			double *R, double phi);
	void make_Ry(
			double *R, double psi);
	void make_Rx(
			double *R, double chi);
	double atan_xy(
			double x, double y);
	double dot_product(
			double *u, double *v, int len);
	void cross_product(
			double *u, double *v, double *n);
	double distance_euclidean(
			double *x, double *y, int len);
	double distance_from_origin(
			double x1, double x2, double x3);
	double distance_from_origin(
			double *x, int len);
	void make_unit_vector(
			double *v, int len);
	void center_of_mass(
			double *Pts, int len,
			int *Pt_idx, int nb_pts, double *c);
	void plane_through_three_points(
			double *p1, double *p2, double *p3,
		double *n, double &d);
	void orthogonal_transformation_from_point_to_basis_vector(
		double *from,
		double *A, double *Av, int verbose_level);
	void output_double(
			double a, std::ostream &ost);
	void mult_matrix_4x4(
			double *v, double *R, double *vR);
	void transpose_matrix_4x4(
			double *A, double *At);
	void transpose_matrix_nxn(
			double *A, double *At, int n);
	void substitute_quadric_linear(
			double *coeff_in, double *coeff_out,
		double *A4_inv, int verbose_level);
	void substitute_cubic_linear_using_povray_ordering(
			double *coeff_in, double *coeff_out,
			double *A4_inv, int verbose_level);
	void substitute_quartic_linear_using_povray_ordering(
		double *coeff_in, double *coeff_out,
		double *A4_inv, int verbose_level);
	void make_transform_t_varphi_u_double(
			int n, double *varphi, double *u,
		double *A, double *Av, int verbose_level);
	// varphi are the dual coordinates of a plane.
	// u is a vector such that varphi(u) \neq -1.
	// A = I + varphi * u.
	void matrix_double_inverse(
			double *A, double *Av, int n,
			int verbose_level);
	int line_centered(
			double *pt1_in, double *pt2_in,
		double *pt1_out, double *pt2_out, double r,
		int verbose_level);
	int line_centered_tolerant(
			double *pt1_in, double *pt2_in,
		double *pt1_out, double *pt2_out, double r,
		int verbose_level);
	int sign_of(double a);
	void eigenvalues(
			double *A, int n, double *lambda,
			int verbose_level);
	void eigenvectors(
			double *A, double *Basis,
			int n, double *lambda, int verbose_level);
	double rad2deg(
			double phi);
	void vec_copy(
			double *from, double *to, int len);
	void vec_swap(
			double *from, double *to, int len);
	void vec_print(
			std::ostream &ost, double *v, int len);
	void vec_scan(
			const char *s, double *&v, int &len);
	void vec_scan(
			std::string &s, double *&v, int &len);
	void vec_scan_from_stream(
			std::istream & is, double *&v, int &len);


	double cos_grad(
			double phi);
	double sin_grad(
			double phi);
	double tan_grad(
			double phi);
	double atan_grad(
			double x);
	void adjust_coordinates_double(
			double *Px, double *Py, int *Qx, int *Qy,
		int N,
		double xmin, double ymin, double xmax, double ymax,
		int verbose_level);
	void Intersection_of_lines(
			double *X, double *Y,
		double *a, double *b, double *c,
		int l1, int l2, int pt);
	void intersection_of_lines(
			double a1, double b1, double c1,
		double a2, double b2, double c2,
		double &x, double &y);
	void Line_through_points(
			double *X, double *Y,
		double *a, double *b, double *c,
		int pt1, int pt2, int line_idx);
	void line_through_points(
			double pt1_x, double pt1_y,
		double pt2_x, double pt2_y,
		double &a, double &b, double &c);
	void intersect_circle_line_through(
			double rad, double x0, double y0,
		double pt1_x, double pt1_y,
		double pt2_x, double pt2_y,
		double &x1, double &y1, double &x2, double &y2);
	void intersect_circle_line(
			double rad, double x0, double y0,
		double a, double b, double c,
		double &x1, double &y1, double &x2, double &y2);
	void affine_combination(
			double *X, double *Y,
		int pt0, int pt1, int pt2, double alpha, int new_pt);
	void on_circle_double(
			double *Px, double *Py, int idx,
		double angle_in_degree, double rad);
	void affine_pt1(
			int *Px, int *Py, int p0, int p1, int p2,
		double f1, int p3);
	void affine_pt2(
			int *Px, int *Py, int p0, int p1, int p1b,
		double f1, int p2, int p2b, double f2, int p3);
	double norm_of_vector_2D(
			int x1, int x2, int y1, int y2);
	void transform_llur(
			int *in, int *out, int &x, int &y);
	void transform_dist(
			int *in, int *out, int &x, int &y);
	void transform_dist_x(
			int *in, int *out, int &x);
	void transform_dist_y(
			int *in, int *out, int &y);
	void transform_llur_double(
			double *in, double *out, double &x, double &y);
	void on_circle_int(
			int *Px, int *Py, int idx, int angle_in_degree, int rad);
	double power_of(
			double x, int n);
	double bernoulli(
			double p, int n, int k);
	void local_coordinates_wrt_triangle(
			double *pt,
			double *triangle_points, double &x, double &y,
			int verbose_level);
	int intersect_line_and_line(
			double *line1_pt1_coords, double *line1_pt2_coords,
			double *line2_pt1_coords, double *line2_pt2_coords,
			double &lambda,
			double *pt_coords,
			int verbose_level);
	void clebsch_map_up(
			double *line1_pt1_coords, double *line1_pt2_coords,
			double *line2_pt1_coords, double *line2_pt2_coords,
		double *pt_in, double *pt_out,
		double *Cubic_coords_povray_ordering,
		int line1_idx, int line2_idx,
		int verbose_level);
	void project_to_disc(
			int f_projection_on, int f_transition,
			int step, int nb_steps,
		double rad, double height, double x, double y,
		double &xp, double &yp);

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
	void load(
			std::string &fname, int verbose_level);
};


// #############################################################################
// orbiter_session.cpp
// #############################################################################


//! global Orbiter session
extern orbiter_kernel_system::orbiter_session *Orbiter;


//! The orbiter session is responsible for the command line interface and the program execution. There will only be one instance of this object


class orbiter_session {

public:

	int verbose_level;

	long int t0;

	int f_draw_options;
	graphics::layered_graph_draw_options *draw_options;


	int f_draw_incidence_structure_description;
	graphics::draw_incidence_structure_description
		*Draw_incidence_structure_description;

	int f_list_arguments;

	int f_seed;
	int the_seed;

	int f_memory_debug;
	int memory_debug_verbose_level;

	int f_override_polynomial;
	std::string override_polynomial;

	int f_orbiter_path;
	std::string orbiter_path;

	int f_magma_path;
	std::string magma_path;

	int f_fork;
	int fork_argument_idx;
	std::string fork_variable;
	std::string fork_logfile_mask;
	int fork_from;
	int fork_to;
	int fork_step;

	orbiter_kernel_system::orbiter_symbol_table *Orbiter_symbol_table;

	long int nb_times_finite_field_created;
	long int nb_times_projective_space_created;
	long int nb_times_action_created;
	long int nb_calls_to_densenauty;

	data_structures::int_vec *Int_vec;
	data_structures::lint_vec *Lint_vec;

	orbiter_kernel_system::mem_object_registry *global_mem_object_registry;

	int longinteger_f_print_scientific;
	int syntax_tree_node_index;

	int f_has_get_projective_space_low_level_function;
	geometry::projective_space *(* get_projective_space_low_level_function)(void *ptr);


	orbiter_session();
	~orbiter_session();
	void print_help(
			int argc,
			std::string *argv, int i, int verbose_level);
	int recognize_keyword(
			int argc,
			std::string *argv, int i, int verbose_level);
	int read_arguments(
			int argc,
			std::string *argv, int i0, int verbose_level);
	void fork(
			int argc, std::string *argv, int verbose_level);

	void *get_object(
			int idx);
	enum symbol_table_object_type get_object_type(
			int idx);
	int find_symbol(
			std::string &label);
	void get_vector_from_label(
			std::string &label, long int *&v, int &sz,
			int verbose_level);
	void get_int_vector_from_label(
			std::string &label, int *&v, int &sz,
			int verbose_level);
	void get_lint_vector_from_label(
			std::string &label, long int *&v, int &sz,
			int verbose_level);
	void get_matrix_from_label(
			std::string &label,
			int *&v, int &m, int &n);
	void find_symbols(
			std::vector<std::string> &Labels, int *&Idx);
	void print_symbol_table();
	void add_symbol_table_entry(
			std::string &label,
			orbiter_symbol_table_entry *Symb, int verbose_level);
	void get_lint_vec(
			std::string &label,
			long int *&the_set, int &set_size, int verbose_level);
	void print_type(
			symbol_table_object_type t);
	field_theory::finite_field
			*get_object_of_type_finite_field(
			std::string &label);
	ring_theory::homogeneous_polynomial_domain
			*get_object_of_type_polynomial_ring(
			std::string &label);
	data_structures::vector_builder *get_object_of_type_vector(
			std::string &label);
	expression_parser::symbolic_object_builder *get_object_of_type_symbolic_object(
			std::string &label);
	coding_theory::crc_object *get_object_of_type_crc_code(
			std::string &label);
	geometry::projective_space *get_projective_space_low_level(
			std::string &label);
	geometry_builder::geometry_builder *get_geometry_builder(
			std::string &label);
	int find_object_of_type_symbolic_object(
			std::string &label);
	void start_memory_debug();
	void stop_memory_debug();

};


// #############################################################################
// orbiter_symbol_table_entry.cpp
// #############################################################################





//! symbol table to store data entries for the orbiter run-time system


class orbiter_symbol_table_entry {
public:
	std::string label;
	enum symbol_table_entry_type type;
	enum symbol_table_object_type object_type;
	int *vec;
	int vec_len;
	std::string str;
	void *ptr;

	orbiter_symbol_table_entry();
	~orbiter_symbol_table_entry();
	void freeself();
	void init(
			std::string &str_label);
	void init_finite_field(
			std::string &label,
			field_theory::finite_field *F,
			int verbose_level);
	void init_polynomial_ring(
			std::string &label,
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void init_any_group(
			std::string &label,
			void *p, int verbose_level);
	void init_linear_group(
			std::string &label,
			void *p, int verbose_level);
	void init_permutation_group(
			std::string &label,
			void *p, int verbose_level);
	void init_modified_group(
			std::string &label,
			void *p, int verbose_level);
	void init_projective_space(
			std::string &label,
			void *p, int verbose_level);
	void init_orthogonal_space(
			std::string &label,
			void *p, int verbose_level);
	void init_BLT_set_classify(
			std::string &label,
			void *p, int verbose_level);
	void init_spread_classify(
			std::string &label,
			void *p, int verbose_level);
#if 0
	void init_formula(
			std::string &label,
			void *p, int verbose_level);
#endif
	void init_cubic_surface(
			std::string &label,
			void *p, int verbose_level);
	void init_quartic_curve(
			std::string &label,
			void *p, int verbose_level);
	void init_BLT_set(
			std::string &label,
			void *p, int verbose_level);
	void init_classification_of_cubic_surfaces_with_double_sixes(
			std::string &label,
			void *p, int verbose_level);
	void init_collection(
			std::string &label,
			std::string &list_of_objects, int verbose_level);
	void init_geometric_object(
			std::string &label,
			geometry::geometric_object_create *COC,
			int verbose_level);
	void init_graph(
			std::string &label,
			void *Gr, int verbose_level);
	void init_code(
			std::string &label,
			void *Code, int verbose_level);
	void init_spread(
			std::string &label,
			void *Spread, int verbose_level);
	void init_translation_plane(
			std::string &label,
			void *Tp, int verbose_level);
	void init_spread_table(
			std::string &label,
			void *Spread_table_with_selection,
			int verbose_level);
	void init_packing_classify(
			std::string &label,
			void *Packing_classify, int verbose_level);
	void init_packing_was(
			std::string &label,
			void *P, int verbose_level);
	void init_packing_was_choose_fixed_points(
			std::string &label,
			void *P, int verbose_level);
	void init_packing_long_orbits(
			std::string &label,
			void *PL, int verbose_level);
	void init_graph_classify(
			std::string &label,
			void *GC, int verbose_level);
	void init_diophant(
			std::string &label,
			void *Dio, int verbose_level);
	void init_design(
			std::string &label,
			void *DC, int verbose_level);
	void init_design_table(
			std::string &label,
			void *DT, int verbose_level);
	void init_large_set_was(
			std::string &label,
			void *LSW, int verbose_level);
	void init_set(
			std::string &label,
			void *SB, int verbose_level);
	void init_vector(
			std::string &label,
			void *VB, int verbose_level);
	void init_symbolic_object(
			std::string &label,
			void *SB, int verbose_level);
	void init_combinatorial_object(
			std::string &label,
			void *Combo, int verbose_level);
	void init_geometry_builder_object(
			std::string &label,
			geometry_builder::geometry_builder *GB,
			int verbose_level);
	void init_vector_ge(
			std::string &label,
			void *V, int verbose_level);
	void init_action_on_forms(
			std::string &label,
			void *AF, int verbose_level);
	void init_orbits(
			std::string &label,
			void *OC, int verbose_level);
	void init_poset_classification_control(
			std::string &label,
			void *PCC, int verbose_level);
	void init_poset_classification_report_options(
			std::string &label,
			void *PCRO, int verbose_level);
	void init_arc_generator_control(
			std::string &label,
			void *AGC, int verbose_level);
	void init_poset_classification_activity(
			std::string &label,
			void *PCA, int verbose_level);
	void init_crc_code(
			std::string &label,
			void *Code, int verbose_level);
	void init_mapping(
			std::string &label,
			void *Mapping, int verbose_level);
	void print();
};


// #############################################################################
// orbiter_symbol_table.cpp
// #############################################################################




//! symbol table to store Orbiter objects for the Orbiter run-time system


class orbiter_symbol_table {
public:
	std::vector<orbiter_symbol_table_entry> Table;

	int f_has_free_entry_callback;
	void (*free_entry_callback)(orbiter_symbol_table_entry *Symb, int verbose_level);

	orbiter_symbol_table();
	~orbiter_symbol_table();
	int find_symbol(
			std::string &str);
	void add_symbol_table_entry(
			std::string &str,
			orbiter_symbol_table_entry *Symb, int verbose_level);
	void free_table_entry(
			int idx, int verbose_level);
	void print_symbol_table();
	void *get_object(
			int idx);
	symbol_table_object_type get_object_type(
			int idx);
	void print_type(
			symbol_table_object_type t);
	std::string stringify_type(
			symbol_table_object_type t);

};




// #############################################################################
// os_interface.cpp
// #############################################################################


//! interface to system functions

class os_interface {
public:

	void runtime(long *l);
	int os_memory_usage();
	long int os_ticks();
	int os_ticks_system();
	int os_ticks_per_second();
	void os_ticks_to_dhms(
			long int ticks, int tps,
			int &d, int &h, int &m, int &s);
	void time_check_delta(
			std::ostream &ost, long int dt);
	std::string stringify_time_difference(
			long int t0);
	std::string stringify_delta_t(
			long int dt);
	void print_elapsed_time(
			std::ostream &ost, int d, int h, int m, int s);
	std::string stringify_elapsed_time(
			int d, int h, int m, int s);
	void time_check(
			std::ostream &ost, long int t0);
	long int delta_time(
			long int t0);
	void seed_random_generator_with_system_time();
	void seed_random_generator(
			int seed);
	int random_integer(
			int p);
	int os_seconds_past_1970();
	void get_string_from_command_line(
			std::string &p, int argc, std::string *argv,
			int &i, int verbose_level);
	void test_swap();
	void block_swap_chars(
			char *ptr, int size, int no);
	void code_int4(
			char *&p, int_4 i);
	int_4 decode_int4(
			const char *&p);
	void code_uchar(
			char *&p, unsigned char a);
	void decode_uchar(
			const char *&p, unsigned char &a);
	void get_date(
			std::string &str);
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

	override_double(
			double *p, double value);
	~override_double();
};


// #############################################################################
// prepare_frames.cpp
// #############################################################################


//! to prepare files using a unified file naming scheme

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
	int parse_arguments(
			int argc, std::string *argv, int verbose_level);
	void print();
	void print_item(
			int i);
	void do_the_work(
			int verbose_level);
};




}}}




#endif /* ORBITER_SRC_LIB_FOUNDATIONS_IO_AND_OS_IO_AND_OS_H_ */




