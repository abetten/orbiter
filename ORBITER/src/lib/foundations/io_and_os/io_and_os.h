// io_and_os.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005




// #############################################################################
// file_output.C:
// #############################################################################


//! a wrapper class for an ofstream which allows to store extra data

class file_output {
public:
	char fname[1000];
	int f_file_is_open;
	ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void null();
	void freeself();
	void open(const char *fname, void *user_data, int verbose_level);
	void close();
	void write_line(int nb, int *data, int verbose_level);
	void write_EOF(int nb_sol, int verbose_level);
};

// #############################################################################
// memory.C:
// #############################################################################



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
	int *allocate_int(int n, const char *file, int line);
	void free_int(int *p, const char *file, int line);
	int **allocate_pint(int n, const char *file, int line);
	void free_pint(int **p, const char *file, int line);
	int ***allocate_ppint(int n, const char *file, int line);
	void free_ppint(int ***p, const char *file, int line);
	char *allocate_char(int n, const char *file, int line);
	void free_char(char *p, const char *file, int line);
	uchar *allocate_uchar(int n, const char *file, int line);
	void free_uchar(uchar *p, const char *file, int line);
	char **allocate_pchar(int n, const char *file, int line);
	void free_pchar(char **p, const char *file, int line);
	uchar **allocate_puchar(int n, const char *file, int line);
	void free_puchar(uchar **p, const char *file, int line);
	void **allocate_pvoid(int n, const char *file, int line);
	void free_pvoid(void **p, const char *file, int line);
	void *allocate_OBJECTS(void *p, int n, int size_of,
			const char *extra_type_info, const char *file, int line);
	void free_OBJECTS(void *p, const char *file, int line);
	void *allocate_OBJECT(void *p, int size_of,
			const char *extra_type_info, const char *file, int line);
	void free_OBJECT(void *p, const char *file, int line);
	int search(void *p, int &idx);
	void insert_at(int idx);
	void add_to_registry(void *pointer,
			int object_type, int object_n, int object_size_of,
			const char *extra_type_info,
			const char *source_file, int source_line,
			int verbose_level);
	void delete_from_registry(void *pointer, int verbose_level);
	void sort_by_size(int verbose_level);
	void sort_by_location_and_get_frequency(int verbose_level);
	void sort_by_type(int verbose_level);
	void sort_by_location(int verbose_level);
};

//! a class related to mem_object_registry


class mem_object_registry_entry {
public:
	int time_stamp;
	void *pointer;
	int object_type;
	int object_n;
	int object_size_of;
		// needed for objects of type class
	const char *extra_type_info;
	const char *source_file;
	int source_line;

	mem_object_registry_entry();
	~mem_object_registry_entry();
	void null();
	void set_type_from_string(char *str);
	void print_type(ostream &ost);
	int size_of();
	void print(int line);
	void print_csv(ostream &ost, int line);
};

extern int f_memory_debug;
extern int memory_debug_verbose_level;
extern mem_object_registry global_mem_object_registry;

void start_memory_debug();
void stop_memory_debug();



// #############################################################################
// memory_object.C:
// #############################################################################

//! can be used for serialization




class memory_object {
public:
	memory_object();
	~memory_object();
	void null();
	void freeself();

	char *char_pointer;
	int alloc_length;
	int used_length;
	int cur_pointer;


	char & s_i(int i) { return char_pointer[i]; };
	void init(int length, char *d, int verbose_level);
	void alloc(int length, int verbose_level);
	void append(int length, char *d, int verbose_level);
	void realloc(int new_length, int verbose_level);
	void write_char(char c);
	void read_char(char *c);
	void write_string(const char *p);
	void read_string(char *&p);
	void write_double(double f);
	void read_double(double *f);
	void write_int64(int i);
	void read_int64(int *i);
	void write_int(int i);
	void read_int(int *i);
	void read_file(const char *fname, int verbose_level);
	void write_file(const char *fname, int verbose_level);
	int multiplicity_of_character(char c);
	void compress(int verbose_level);
	void decompress(int verbose_level);
};

// #############################################################################
// orbiter_data_file.C:
// #############################################################################



//! a class to read output files from orbiters poset classification



class orbiter_data_file {
public:
	int nb_cases;
	int **sets;
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
// util.C:
// #############################################################################


void int_vec_add(int *v1, int *v2, int *w, int len);
void int_vec_add3(int *v1, int *v2, int *v3, int *w, int len);
void int_vec_apply(int *from, int *through, int *to, int len);
int int_vec_is_constant_on_subset(int *v, int *subset, int sz, int &value);
void int_vec_take_away(int *v, int &len, int *take_away, int nb_take_away);
	// v must be sorted
int int_vec_count_number_of_nonzero_entries(int *v, int len);
int int_vec_find_first_nonzero_entry(int *v, int len);
void int_vec_zero(int *v, int len);
void int_vec_mone(int *v, int len);
void int_vec_copy(int *from, int *to, int len);
void double_vec_copy(double *from, double *to, int len);
void int_vec_swap(int *v1, int *v2, int len);
void int_vec_delete_element_assume_sorted(int *v, int &len, int a);
uchar *bitvector_allocate(int length);
uchar *bitvector_allocate_and_coded_length(int length, int &coded_length);
void bitvector_m_ii(uchar *bitvec, int i, int a);
void bitvector_set_bit(uchar *bitvec, int i);
int bitvector_s_i(uchar *bitvec, int i);
// returns 0 or 1
int int_vec_hash(int *data, int len);
int int_vec_hash_after_sorting(int *data, int len);
const char *plus_minus_string(int epsilon);
const char *plus_minus_letter(int epsilon);
void int_vec_complement(int *v, int n, int k);
// computes the complement to v + k (v must be allocated to n lements)
void int_vec_complement(int *v, int *w, int n, int k);
// computes the complement of v[k] w[n - k] 
void int_vec_init5(int *v, int a0, int a1, int a2, int a3, int a4);
void dump_memory_chain(void *allocated_objects);
void print_vector(ostream &ost, int *v, int size);
int int_vec_minimum(int *v, int len);
int int_vec_maximum(int *v, int len);
//void int_vec_copy(int len, int *from, int *to);
int int_vec_first_difference(int *p, int *q, int len);
void itoa(char *p, int len_of_p, int i);
void char_swap(char *p, char *q, int len);
void int_vec_distribution_compute_and_print(ostream &ost, int *v, int v_len);
void int_vec_distribution(int *v, int len_v, int *&val, int *&mult, int &len);
void int_distribution_print(ostream &ost, int *val, int *mult, int len);
void int_swap(int& x, int& y);
void int_set_print(int *v, int len);
void int_set_print(ostream &ost, int *v, int len);
void int_set_print_tex(ostream &ost, int *v, int len);
void int_set_print_masked_tex(ostream &ost, 
	int *v, int len, const char *mask_begin, const char *mask_end);
void int_set_print_tex_for_inline_text(ostream &ost, int *v, int len);
void int_vec_print(ostream &ost, int *v, int len);
void int_vec_print_as_matrix(ostream &ost, 
	int *v, int len, int width, int f_tex);
void int_vec_print_as_table(ostream &ost, int *v, int len, int width);
void int_vec_print_fully(ostream &ost, int *v, int len);
void int_vec_print_Cpp(ostream &ost, int *v, int len);
void int_vec_print_GAP(ostream &ost, int *v, int len);
void double_vec_print(ostream &ost, double *v, int len);
void integer_vec_print(ostream &ost, int *v, int len);
void print_integer_matrix(ostream &ost, int *p, int m, int n);
void print_integer_matrix_width(ostream &ost, int *p, 
	int m, int n, int dim_n, int w);
void print_01_matrix_tex(ostream &ost, int *p, int m, int n);
void print_integer_matrix_tex(ostream &ost, int *p, int m, int n);
void print_integer_matrix_with_labels(ostream &ost, int *p, 
	int m, int n, int *row_labels, int *col_labels, int f_tex);
void print_integer_matrix_with_standard_labels(ostream &ost, 
	int *p, int m, int n, int f_tex);
void print_integer_matrix_with_standard_labels_and_offset(ostream &ost, 
	int *p, int m, int n, int m_offset, int n_offset, int f_tex);
void print_integer_matrix_tex_block_by_block(ostream &ost, 
	int *p, int m, int n, int block_width);
void print_integer_matrix_with_standard_labels_and_offset_text(ostream &ost, 
	int *p, int m, int n, int m_offset, int n_offset);
void print_integer_matrix_with_standard_labels_and_offset_tex(ostream &ost, 
	int *p, int m, int n, int m_offset, int n_offset);
void print_big_integer_matrix_tex(ostream &ost, int *p, int m, int n);
void int_matrix_make_block_matrix_2x2(int *Mtx, int k, 
	int *A, int *B, int *C, int *D);
// makes the 2k x 2k block matrix 
// (A B)
// (C D)
void int_matrix_delete_column_in_place(int *Mtx, int k, int n, int pivot);
// afterwards, the matrix is k x (n - 1)
int int_matrix_max_log_of_entries(int *p, int m, int n);
void int_matrix_print_ost(ostream &ost, int *p, int m, int n);
void int_matrix_print(int *p, int m, int n);
void int_matrix_print_ost(ostream &ost, int *p, int m, int n, int w);
void int_matrix_print(int *p, int m, int n, int w);
void int_matrix_print_tex(ostream &ost, int *p, int m, int n);
void int_matrix_print_bitwise(int *p, int m, int n);
void uchar_print_bitwise(ostream &ost, uchar u);
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
void runtime(long *l);
int os_memory_usage();
int os_ticks();
int os_ticks_system();
int os_ticks_per_second();
void os_ticks_to_dhms(int ticks, int tps, int &d, int &h, int &m, int &s);
void time_check_delta(ostream &ost, int dt);
void print_elapsed_time(ostream &ost, int d, int h, int m, int s);
void time_check(ostream &ost, int t0);
int delta_time(int t0);
int file_size(const char *name);
void delete_file(const char *fname);
void fwrite_int4(FILE *fp, int a);
int4 fread_int4(FILE *fp);
void fwrite_uchars(FILE *fp, uchar *p, int len);
void fread_uchars(FILE *fp, uchar *p, int len);
void latex_head_easy(ostream& ost);
void latex_head_easy_with_extras_in_the_praeamble(ostream& ost, const char *extras);
void latex_head_easy_sideways(ostream& ost);
void latex_head(ostream& ost, int f_book, int f_title, 
	const char *title, const char *author, 
	int f_toc, int f_landscape, int f_12pt, 
	int f_enlarged_page, int f_pagenumbers, 
	const char *extras_for_preamble);
void latex_foot(ostream& ost);
void seed_random_generator_with_system_time();
void seed_random_generator(int seed);
int random_integer(int p);
void print_set(ostream &ost, int size, int *set);
void block_swap_chars(char *ptr, int size, int no);
void code_int4(char *&p, int4 i);
int4 decode_int4(char *&p);
void code_uchar(char *&p, uchar a);
void decode_uchar(char *&p, uchar &a);
void print_incidence_structure(ostream &ost, int m, int n, int len, int *S);
void int_vec_scan(const char *s, int *&v, int &len);
void int_vec_scan_from_stream(istream & is, int *&v, int &len);
void double_vec_scan(const char *s, double *&v, int &len);
void double_vec_scan_from_stream(istream & is, double *&v, int &len);
void scan_permutation_from_string(const char *s, 
	int *&perm, int &degree, int verbose_level);
void scan_permutation_from_stream(istream & is, 
	int *&perm, int &degree, int verbose_level);
char get_character(istream & is, int verbose_level);
void replace_extension_with(char *p, const char *new_ext);
void chop_off_extension(char *p);
void chop_off_extension_if_present(char *p, const char *ext);
void get_fname_base(const char *p, char *fname_base);
void get_extension_if_present(const char *p, char *ext);
void get_extension_if_present_and_chop_off(char *p, char *ext);
int s_scan_int(char **s, int *i);
int s_scan_token(char **s, char *str);
int s_scan_token_arbitrary(char **s, char *str);
int s_scan_str(char **s, char *str);
int s_scan_token_comma_separated(char **s, char *str);
int hashing(int hash0, int a);
int hashing_fixed_width(int hash0, int a, int bit_length);
int int_vec_hash(int *v, int len, int bit_length);
void parse_sets(int nb_cases, char **data, int f_casenumbers, 
	int *&Set_sizes, int **&Sets, char **&Ago_ascii, char **&Aut_ascii, 
	int *&Casenumbers, 
	int verbose_level);
void parse_sets_and_check_sizes_easy(int len, int nb_cases, 
	char **data, int **&sets);
void parse_line(char *line, int &len, int *&set, 
	char *ago_ascii, char *aut_ascii);
int count_number_of_orbits_in_file(const char *fname, int verbose_level);
int count_number_of_lines_in_file(const char *fname, int verbose_level);
int try_to_read_file(const char *fname, int &nb_cases, 
	char **&data, int verbose_level);
void read_and_parse_data_file(const char *fname, int &nb_cases, 
	char **&data, int **&sets, int *&set_sizes, int verbose_level);
void free_data_fancy(int nb_cases, 
	int *Set_sizes, int **Sets, 
	char **Ago_ascii, char **Aut_ascii, 
	int *Casenumbers);
void read_and_parse_data_file_fancy(const char *fname, 
	int f_casenumbers, 
	int &nb_cases, 
	int *&Set_sizes, int **&Sets, char **&Ago_ascii, char **&Aut_ascii, 
	int *&Casenumbers, 
	int verbose_level);
void read_set_from_file(const char *fname, 
	int *&the_set, int &set_size, int verbose_level);
void write_set_to_file(const char *fname, 
	int *the_set, int set_size, int verbose_level);
void read_set_from_file_int4(const char *fname, 
	int *&the_set, int &set_size, int verbose_level);
void write_set_to_file_as_int4(const char *fname, 
	int *the_set, int set_size, int verbose_level);
void write_set_to_file_as_int8(const char *fname, 
	int *the_set, int set_size, int verbose_level);
void read_k_th_set_from_file(const char *fname, int k, 
	int *&the_set, int &set_size, int verbose_level);
void write_incidence_matrix_to_file(char *fname, 
	int *Inc, int m, int n, int verbose_level);
void read_incidence_matrix_from_inc_file(int *&M, int &m, int &n, 
	char *inc_file_name, int inc_file_idx, int verbose_level);
int inc_file_get_number_of_geometries(
	char *inc_file_name, int verbose_level);

void print_line_of_number_signs();
void print_repeated_character(ostream &ost, char c, int n);
void print_pointer_hex(ostream &ost, void *p);
void print_hex_digit(ostream &ost, int digit);
void count_number_of_solutions_in_file(const char *fname, 
	int &nb_solutions, 
	int verbose_level);
void count_number_of_solutions_in_file_by_case(const char *fname, 
	int *&nb_solutions, int *&case_nb, int &nb_cases, 
	int verbose_level);
void read_solutions_from_file(const char *fname, 
	int &nb_solutions, int *&Solutions, int solution_size, 
	int verbose_level);
void read_solutions_from_file_by_case(const char *fname, 
	int *nb_solutions, int *case_nb, int nb_cases, 
	int **&Solutions, int solution_size, 
	int verbose_level);
void copy_file_to_ostream(ostream &ost, char *fname);
void int_vec_write_csv(int *v, int len, 
	const char *fname, const char *label);
void int_vecs_write_csv(int *v1, int *v2, int len, 
	const char *fname, 
	const char *label1, const char *label2);
void int_vec_array_write_csv(int nb_vecs, int **Vec, int len, 
	const char *fname, const char **column_label);
void int_matrix_write_csv(const char *fname, int *M, int m, int n);
void double_matrix_write_csv(const char *fname, 
	double *M, int m, int n);
void int_matrix_write_csv_with_labels(const char *fname, 
	int *M, int m, int n, const char **column_label);
void int_matrix_read_csv(const char *fname, int *&M, 
	int &m, int &n, int verbose_level);
void double_matrix_read_csv(const char *fname, double *&M, 
	int &m, int &n, int verbose_level);
void int_matrix_write_text(const char *fname, 
	int *M, int m, int n);
void int_matrix_read_text(const char *fname, 
	int *&M, int &m, int &n);
int compare_sets(int *set1, int *set2, int sz1, int sz2);
int test_if_sets_are_disjoint(int *set1, int *set2, int sz1, int sz2);
void make_graph_of_disjoint_sets_from_rows_of_matrix(
	int *M, int m, int n, 
	int *&Adj, int verbose_level);
void write_exact_cover_problem_to_file(int *Inc, int nb_rows, 
	int nb_cols, const char *fname);
void read_solution_file(char *fname, 
	int *Inc, int nb_rows, int nb_cols, 
	int *&Solutions, int &sol_length, int &nb_sol, 
	int verbose_level);
// sol_length must be constant
void int_vec_print_to_str(char *str, int *data, int len);
void int_vec_print_to_str_naked(char *str, int *data, int len);
void int_matrix_print_with_labels_and_partition(ostream &ost, int *p, 
	int m, int n, 
	int *row_labels, int *col_labels, 
	int *row_part_first, int *row_part_len, int nb_row_parts,  
	int *col_part_first, int *col_part_len, int nb_col_parts,  
	void (*process_function_or_NULL)(int *p, int m, int n, int i, int j, 
		int val, char *output, void *data), 
	void *data, 
	int f_tex);
int is_csv_file(const char *fname);
int is_xml_file(const char *fname);
void os_date_string(char *str, int sz);
int os_seconds_past_1970();
void povray_beginning(ostream &ost,
		double angle,
		const char *sky,
		const char *location,
		const char *look_at,
		int f_with_background);
void povray_animation_rotate_around_origin_and_1_1_1(ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector(double *v, 
	ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector_by_a_given_angle(
	double *v, double angle_zero_one, ostream &ost);
void povray_union_start(ostream &ost);
void povray_union_end(ostream &ost, double clipping_radius);
void povray_bottom_plane(ostream &ost);
void povray_rotate_111(int h, int nb_frames, ostream &fp);
void povray_ini(ostream &ost, const char *fname_pov, int first_frame, 
	int last_frame);
void test_typedefs();
void concatenate_files(const char *fname_in_mask, int N, 
	const char *fname_out, const char *EOF_marker, int f_title_line, 
	int verbose_level);
void chop_string(const char *str, int &argc, char **&argv);









