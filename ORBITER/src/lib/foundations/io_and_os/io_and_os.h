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
	BYTE fname[1000];
	INT f_file_is_open;
	ofstream *fp;
	void *user_data;
	
	file_output();
	~file_output();
	void null();
	void freeself();
	void open(const BYTE *fname, void *user_data, INT verbose_level);
	void close();
	void write_line(INT nb, INT *data, INT verbose_level);
	void write_EOF(INT nb_sol, INT verbose_level);
};

// #############################################################################
// memory.C:
// #############################################################################



//! maintains a registry of allocated memory



class mem_object_registry {
public:
	INT f_automatic_dump;
	INT automatic_dump_interval;
	BYTE automatic_dump_fname_mask[1000];

	INT nb_entries_allocated;
	INT nb_entries_used;
	mem_object_registry_entry *entries;
		// entries are sorted by
		// the value of the pointer

	INT nb_allocate_total;
		// total number of allocations
	INT nb_delete_total;
		// total number of deletions
	INT cur_time;
		// increments with every allocate and every delete

	mem_object_registry();
	~mem_object_registry();
	void init(INT verbose_level);
	void allocate(INT N, INT verbose_level);
	void set_automatic_dump(
			INT automatic_dump_interval, const BYTE *fname_mask,
			INT verbose_level);
	void automatic_dump();
	void manual_dump();
	void manual_dump_with_file_name(const BYTE *fname);
	void dump();
	void dump_to_csv_file(const BYTE *fname);
	int *allocate_int(INT n, const char *file, int line);
	void free_int(int *p, const char *file, int line);
	int **allocate_pint(INT n, const char *file, int line);
	void free_pint(int **p, const char *file, int line);
	INT *allocate_INT(INT n, const char *file, int line);
	void free_INT(INT *p, const char *file, int line);
	INT **allocate_PINT(INT n, const char *file, int line);
	void free_PINT(INT **p, const char *file, int line);
	INT ***allocate_PPINT(INT n, const char *file, int line);
	void free_PPINT(INT ***p, const char *file, int line);
	BYTE *allocate_BYTE(INT n, const char *file, int line);
	void free_BYTE(BYTE *p, const char *file, int line);
	UBYTE *allocate_UBYTE(INT n, const char *file, int line);
	void free_UBYTE(UBYTE *p, const char *file, int line);
	BYTE **allocate_PBYTE(INT n, const char *file, int line);
	void free_PBYTE(BYTE **p, const char *file, int line);
	UBYTE **allocate_PUBYTE(INT n, const char *file, int line);
	void free_PUBYTE(UBYTE **p, const char *file, int line);
	void **allocate_pvoid(INT n, const char *file, int line);
	void free_pvoid(void **p, const char *file, int line);
	void *allocate_OBJECTS(void *p, INT n, INT size_of,
			const char *extra_type_info, const char *file, int line);
	void free_OBJECTS(void *p, const char *file, int line);
	void *allocate_OBJECT(void *p, INT size_of,
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
	void set_type_from_string(BYTE *str);
	void print_type(ostream &ost);
	int size_of();
	void print(INT line);
	void print_csv(ostream &ost, INT line);
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

	BYTE *char_pointer;
	INT alloc_length;
	INT used_length;
	INT cur_pointer;


	char & s_i(INT i) { return char_pointer[i]; };
	void init(INT length, char *d, INT verbose_level);
	void alloc(INT length, INT verbose_level);
	void append(INT length, char *d, INT verbose_level);
	void realloc(INT new_length, INT verbose_level);
	void write_char(char c);
	void read_char(char *c);
	void write_string(const BYTE *p);
	void read_string(BYTE *&p);
	void write_double(double f);
	void read_double(double *f);
	void write_int64(INT i);
	void read_int64(INT *i);
	void write_int(INT i);
	void read_int(INT *i);
	void read_file(const BYTE *fname, INT verbose_level);
	void write_file(const BYTE *fname, INT verbose_level);
	INT multiplicity_of_character(BYTE c);
	void compress(INT verbose_level);
	void decompress(INT verbose_level);
};

// #############################################################################
// orbiter_data_file.C:
// #############################################################################



//! a class to read output files from orbiters poset classification



class orbiter_data_file {
public:
	INT nb_cases;
	INT **sets;
	INT *set_sizes;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii;
	INT *Casenumbers;
	
	orbiter_data_file();
	~orbiter_data_file();
	void null();
	void freeself();
	void load(const BYTE *fname, INT verbose_level);
};



// #############################################################################
// util.C:
// #############################################################################


void INT_vec_add(INT *v1, INT *v2, INT *w, INT len);
void INT_vec_add3(INT *v1, INT *v2, INT *v3, INT *w, INT len);
void INT_vec_apply(INT *from, INT *through, INT *to, INT len);
INT INT_vec_is_constant_on_subset(INT *v, INT *subset, INT sz, INT &value);
void INT_vec_take_away(INT *v, INT &len, INT *take_away, INT nb_take_away);
	// v must be sorted
INT INT_vec_count_number_of_nonzero_entries(INT *v, INT len);
INT INT_vec_find_first_nonzero_entry(INT *v, INT len);
void INT_vec_zero(INT *v, INT len);
void INT_vec_mone(INT *v, INT len);
void INT_vec_copy(INT *from, INT *to, INT len);
void double_vec_copy(double *from, double *to, INT len);
void INT_vec_swap(INT *v1, INT *v2, INT len);
void INT_vec_delete_element_assume_sorted(INT *v, INT &len, INT a);
UBYTE *bitvector_allocate(INT length);
UBYTE *bitvector_allocate_and_coded_length(INT length, INT &coded_length);
void bitvector_m_ii(UBYTE *bitvec, INT i, INT a);
void bitvector_set_bit(UBYTE *bitvec, INT i);
INT bitvector_s_i(UBYTE *bitvec, INT i);
// returns 0 or 1
INT INT_vec_hash(INT *data, INT len);
INT INT_vec_hash_after_sorting(INT *data, INT len);
const BYTE *plus_minus_string(INT epsilon);
const BYTE *plus_minus_letter(INT epsilon);
void INT_vec_complement(INT *v, INT n, INT k);
// computes the complement to v + k (v must be allocated to n lements)
void INT_vec_complement(INT *v, INT *w, INT n, INT k);
// computes the complement of v[k] w[n - k] 
void INT_vec_init5(INT *v, INT a0, INT a1, INT a2, INT a3, INT a4);
void dump_memory_chain(void *allocated_objects);
void print_vector(ostream &ost, INT *v, int size);
INT INT_vec_minimum(INT *v, INT len);
INT INT_vec_maximum(INT *v, INT len);
//void INT_vec_copy(INT len, INT *from, INT *to);
INT INT_vec_first_difference(INT *p, INT *q, INT len);
void itoa(char *p, INT len_of_p, INT i);
void BYTE_swap(BYTE *p, BYTE *q, INT len);
void INT_vec_distribution_compute_and_print(ostream &ost, INT *v, INT v_len);
void INT_vec_distribution(INT *v, INT len_v, INT *&val, INT *&mult, INT &len);
void INT_distribution_print(ostream &ost, INT *val, INT *mult, INT len);
void INT_swap(INT& x, INT& y);
void INT_set_print(INT *v, INT len);
void INT_set_print(ostream &ost, INT *v, INT len);
void INT_set_print_tex(ostream &ost, INT *v, INT len);
void INT_set_print_masked_tex(ostream &ost, 
	INT *v, INT len, const BYTE *mask_begin, const BYTE *mask_end);
void INT_set_print_tex_for_inline_text(ostream &ost, INT *v, INT len);
void INT_vec_print(ostream &ost, INT *v, INT len);
void INT_vec_print_as_matrix(ostream &ost, 
	INT *v, INT len, INT width, INT f_tex);
void INT_vec_print_as_table(ostream &ost, INT *v, INT len, INT width);
void INT_vec_print_fully(ostream &ost, INT *v, INT len);
void INT_vec_print_Cpp(ostream &ost, INT *v, INT len);
void INT_vec_print_GAP(ostream &ost, INT *v, INT len);
void double_vec_print(ostream &ost, double *v, INT len);
void integer_vec_print(ostream &ost, int *v, int len);
void print_integer_matrix(ostream &ost, INT *p, INT m, INT n);
void print_integer_matrix_width(ostream &ost, INT *p, 
	INT m, INT n, INT dim_n, INT w);
void print_01_matrix_tex(ostream &ost, INT *p, INT m, INT n);
void print_integer_matrix_tex(ostream &ost, INT *p, INT m, INT n);
void print_integer_matrix_with_labels(ostream &ost, INT *p, 
	INT m, INT n, INT *row_labels, INT *col_labels, INT f_tex);
void print_integer_matrix_with_standard_labels(ostream &ost, 
	INT *p, INT m, INT n, INT f_tex);
void print_integer_matrix_with_standard_labels_and_offset(ostream &ost, 
	INT *p, INT m, INT n, INT m_offset, INT n_offset, INT f_tex);
void print_integer_matrix_tex_block_by_block(ostream &ost, 
	INT *p, INT m, INT n, INT block_width);
void print_integer_matrix_with_standard_labels_and_offset_text(ostream &ost, 
	INT *p, INT m, INT n, INT m_offset, INT n_offset);
void print_integer_matrix_with_standard_labels_and_offset_tex(ostream &ost, 
	INT *p, INT m, INT n, INT m_offset, INT n_offset);
void print_big_integer_matrix_tex(ostream &ost, INT *p, INT m, INT n);
void INT_matrix_make_block_matrix_2x2(INT *Mtx, INT k, 
	INT *A, INT *B, INT *C, INT *D);
// makes the 2k x 2k block matrix 
// (A B)
// (C D)
void INT_matrix_delete_column_in_place(INT *Mtx, INT k, INT n, INT pivot);
// afterwards, the matrix is k x (n - 1)
INT INT_matrix_max_log_of_entries(INT *p, INT m, INT n);
void INT_matrix_print(INT *p, INT m, INT n);
void INT_matrix_print(INT *p, INT m, INT n, INT w);
void INT_matrix_print_tex(ostream &ost, INT *p, INT m, INT n);
void INT_matrix_print_bitwise(INT *p, INT m, INT n);
void UBYTE_print_bitwise(ostream &ost, UBYTE u);
void UBYTE_move(UBYTE *p, UBYTE *q, INT len);
void INT_submatrix_all_rows(INT *A, INT m, INT n, 
	INT nb_cols, INT *cols, INT *B);
void INT_submatrix_all_cols(INT *A, INT m, INT n, 
	INT nb_rows, INT *rows, INT *B);
void INT_submatrix(INT *A, INT m, INT n, INT nb_rows, 
	INT *rows, INT nb_cols, INT *cols, INT *B);
void INT_matrix_transpose(INT n, INT *A);
void INT_matrix_transpose(INT *M, INT m, INT n, INT *Mt);
// Mt must point to the right amount of memory (n * m INT's)
void INT_matrix_shorten_rows(INT *&p, INT m, INT n);
void PINT_matrix_shorten_rows(PINT *&p, INT m, INT n);
void runtime(long *l);
INT os_memory_usage();
INT os_ticks();
INT os_ticks_system();
INT os_ticks_per_second();
void os_ticks_to_dhms(INT ticks, INT tps, INT &d, INT &h, INT &m, INT &s);
void time_check_delta(ostream &ost, INT dt);
void print_elapsed_time(ostream &ost, INT d, INT h, INT m, INT s);
void time_check(ostream &ost, INT t0);
INT delta_time(INT t0);
INT file_size(const BYTE *name);
void delete_file(const BYTE *fname);
void fwrite_INT4(FILE *fp, INT a);
INT4 fread_INT4(FILE *fp);
void fwrite_UBYTEs(FILE *fp, UBYTE *p, INT len);
void fread_UBYTEs(FILE *fp, UBYTE *p, INT len);
void latex_head_easy(ostream& ost);
void latex_head_easy_with_extras_in_the_praeamble(ostream& ost, const BYTE *extras);
void latex_head_easy_sideways(ostream& ost);
void latex_head(ostream& ost, INT f_book, INT f_title, 
	const BYTE *title, const BYTE *author, 
	INT f_toc, INT f_landscape, INT f_12pt, 
	INT f_enlarged_page, INT f_pagenumbers, 
	const BYTE *extras_for_preamble);
void latex_foot(ostream& ost);
void seed_random_generator_with_system_time();
void seed_random_generator(INT seed);
INT random_integer(INT p);
void print_set(ostream &ost, INT size, INT *set);
void block_swap_bytes(SCHAR *ptr, INT size, INT no);
void code_INT4(char *&p, INT4 i);
INT4 decode_INT4(char *&p);
void code_UBYTE(char *&p, UBYTE a);
void decode_UBYTE(char *&p, UBYTE &a);
void print_incidence_structure(ostream &ost, INT m, INT n, INT len, INT *S);
void INT_vec_scan(const BYTE *s, INT *&v, INT &len);
void INT_vec_scan_from_stream(istream & is, INT *&v, INT &len);
void double_vec_scan(const BYTE *s, double *&v, INT &len);
void double_vec_scan_from_stream(istream & is, double *&v, INT &len);
void scan_permutation_from_string(const char *s, 
	INT *&perm, INT &degree, INT verbose_level);
void scan_permutation_from_stream(istream & is, 
	INT *&perm, INT &degree, INT verbose_level);
char get_character(istream & is, INT verbose_level);
void replace_extension_with(char *p, const char *new_ext);
void chop_off_extension(char *p);
void chop_off_extension_if_present(char *p, const char *ext);
void get_fname_base(const char *p, BYTE *fname_base);
void get_extension_if_present(const char *p, char *ext);
void get_extension_if_present_and_chop_off(char *p, char *ext);
INT s_scan_int(BYTE **s, INT *i);
INT s_scan_token(BYTE **s, BYTE *str);
INT s_scan_token_arbitrary(BYTE **s, BYTE *str);
INT s_scan_str(BYTE **s, BYTE *str);
INT s_scan_token_comma_separated(BYTE **s, BYTE *str);
INT hashing(INT hash0, INT a);
INT hashing_fixed_width(INT hash0, INT a, INT bit_length);
INT INT_vec_hash(INT *v, INT len, INT bit_length);
void parse_sets(INT nb_cases, BYTE **data, INT f_casenumbers, 
	INT *&Set_sizes, INT **&Sets, BYTE **&Ago_ascii, BYTE **&Aut_ascii, 
	INT *&Casenumbers, 
	INT verbose_level);
void parse_sets_and_check_sizes_easy(INT len, INT nb_cases, 
	BYTE **data, INT **&sets);
void parse_line(BYTE *line, INT &len, INT *&set, 
	BYTE *ago_ascii, BYTE *aut_ascii);
INT count_number_of_orbits_in_file(const BYTE *fname, INT verbose_level);
INT count_number_of_lines_in_file(const BYTE *fname, INT verbose_level);
INT try_to_read_file(const BYTE *fname, INT &nb_cases, 
	BYTE **&data, INT verbose_level);
void read_and_parse_data_file(const BYTE *fname, INT &nb_cases, 
	BYTE **&data, INT **&sets, INT *&set_sizes, INT verbose_level);
void free_data_fancy(INT nb_cases, 
	INT *Set_sizes, INT **Sets, 
	BYTE **Ago_ascii, BYTE **Aut_ascii, 
	INT *Casenumbers);
void read_and_parse_data_file_fancy(const BYTE *fname, 
	INT f_casenumbers, 
	INT &nb_cases, 
	INT *&Set_sizes, INT **&Sets, BYTE **&Ago_ascii, BYTE **&Aut_ascii, 
	INT *&Casenumbers, 
	INT verbose_level);
void read_set_from_file(const BYTE *fname, 
	INT *&the_set, INT &set_size, INT verbose_level);
void write_set_to_file(const BYTE *fname, 
	INT *the_set, INT set_size, INT verbose_level);
void read_set_from_file_INT4(const BYTE *fname, 
	INT *&the_set, INT &set_size, INT verbose_level);
void write_set_to_file_as_INT4(const BYTE *fname, 
	INT *the_set, INT set_size, INT verbose_level);
void write_set_to_file_as_INT8(const BYTE *fname, 
	INT *the_set, INT set_size, INT verbose_level);
void read_k_th_set_from_file(const BYTE *fname, INT k, 
	INT *&the_set, INT &set_size, INT verbose_level);
void write_incidence_matrix_to_file(BYTE *fname, 
	INT *Inc, INT m, INT n, INT verbose_level);
void read_incidence_matrix_from_inc_file(INT *&M, INT &m, INT &n, 
	BYTE *inc_file_name, INT inc_file_idx, INT verbose_level);
INT inc_file_get_number_of_geometries(
	BYTE *inc_file_name, INT verbose_level);

void print_line_of_number_signs();
void print_repeated_character(ostream &ost, BYTE c, INT n);
void print_pointer_hex(ostream &ost, void *p);
void print_hex_digit(ostream &ost, INT digit);
void count_number_of_solutions_in_file(const BYTE *fname, 
	INT &nb_solutions, 
	INT verbose_level);
void count_number_of_solutions_in_file_by_case(const BYTE *fname, 
	INT *&nb_solutions, INT *&case_nb, INT &nb_cases, 
	INT verbose_level);
void read_solutions_from_file(const BYTE *fname, 
	INT &nb_solutions, INT *&Solutions, INT solution_size, 
	INT verbose_level);
void read_solutions_from_file_by_case(const BYTE *fname, 
	INT *nb_solutions, INT *case_nb, INT nb_cases, 
	INT **&Solutions, INT solution_size, 
	INT verbose_level);
void copy_file_to_ostream(ostream &ost, BYTE *fname);
void INT_vec_write_csv(INT *v, INT len, 
	const BYTE *fname, const BYTE *label);
void INT_vecs_write_csv(INT *v1, INT *v2, INT len, 
	const BYTE *fname, 
	const BYTE *label1, const BYTE *label2);
void INT_vec_array_write_csv(INT nb_vecs, INT **Vec, INT len, 
	const BYTE *fname, const BYTE **column_label);
void INT_matrix_write_csv(const BYTE *fname, INT *M, INT m, INT n);
void double_matrix_write_csv(const BYTE *fname, 
	double *M, INT m, INT n);
void INT_matrix_write_csv_with_labels(const BYTE *fname, 
	INT *M, INT m, INT n, const BYTE **column_label);
void INT_matrix_read_csv(const BYTE *fname, INT *&M, 
	INT &m, INT &n, INT verbose_level);
void double_matrix_read_csv(const BYTE *fname, double *&M, 
	INT &m, INT &n, INT verbose_level);
void INT_matrix_write_text(const BYTE *fname, 
	INT *M, INT m, INT n);
void INT_matrix_read_text(const BYTE *fname, 
	INT *&M, INT &m, INT &n);
INT compare_sets(INT *set1, INT *set2, INT sz1, INT sz2);
INT test_if_sets_are_disjoint(INT *set1, INT *set2, INT sz1, INT sz2);
void make_graph_of_disjoint_sets_from_rows_of_matrix(
	INT *M, INT m, INT n, 
	INT *&Adj, INT verbose_level);
void write_exact_cover_problem_to_file(INT *Inc, INT nb_rows, 
	INT nb_cols, const BYTE *fname);
void read_solution_file(BYTE *fname, 
	INT *Inc, INT nb_rows, INT nb_cols, 
	INT *&Solutions, INT &sol_length, INT &nb_sol, 
	INT verbose_level);
// sol_length must be constant
void INT_vec_print_to_str(BYTE *str, INT *data, INT len);
void INT_matrix_print_with_labels_and_partition(ostream &ost, INT *p, 
	INT m, INT n, 
	INT *row_labels, INT *col_labels, 
	INT *row_part_first, INT *row_part_len, INT nb_row_parts,  
	INT *col_part_first, INT *col_part_len, INT nb_col_parts,  
	void (*process_function_or_NULL)(INT *p, INT m, INT n, INT i, INT j, 
		INT val, BYTE *output, void *data), 
	void *data, 
	INT f_tex);
INT is_csv_file(const BYTE *fname);
INT is_xml_file(const BYTE *fname);
void os_date_string(BYTE *str, INT sz);
INT os_seconds_past_1970();
void povray_beginning(ostream &ost, double angle);
void povray_animation_rotate_around_origin_and_1_1_1(ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector(double *v, 
	ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector_by_a_given_angle(
	double *v, double angle_zero_one, ostream &ost);
void povray_union_start(ostream &ost);
void povray_union_end(ostream &ost, double clipping_radius);
void povray_bottom_plane(ostream &ost);
void povray_rotate_111(INT h, INT nb_frames, ostream &fp);
void povray_ini(ostream &ost, const BYTE *fname_pov, INT first_frame, 
	INT last_frame);
void test_typedefs();
void concatenate_files(const BYTE *fname_in_mask, INT N, 
	const BYTE *fname_out, const BYTE *EOF_marker, INT f_title_line, 
	INT verbose_level);









