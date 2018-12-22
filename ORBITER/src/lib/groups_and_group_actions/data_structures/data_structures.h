// data_structures.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005


// #############################################################################
// group.C:
// #############################################################################


//! a container data structure for groups




class group {

public:
	action *A;

	int f_has_ascii_coding;
	char *ascii_coding;

	int f_has_strong_generators;
	vector_ge *SG;
	int *tl;
	
	int f_has_sims;
	sims *S;
	
	group();
	~group();
	void null();
	void freeself();
	group(action *A);
	group(action *A, const char *ascii_coding);
	group(action *A, vector_ge &SG, int *tl);
	void init(action *A);
	void init_ascii_coding_to_sims(const char *ascii_coding);
	void init_ascii_coding(const char *ascii_coding);
	void delete_ascii_coding();
	void delete_sims();
	void init_strong_generators_empty_set();
	void init_strong_generators(vector_ge &SG, int *tl);
	void init_strong_generators_by_hdl(int nb_gen, int *gen_hdl, 
		int *tl, int verbose_level);
	void delete_strong_generators();
	void require_ascii_coding();
	void require_strong_generators();
	void require_sims();
	void group_order(longinteger_object &go);
	void print_group_order(ostream &ost);
	void print_tl();
	void code_ascii(int verbose_level);
	void decode_ascii(int verbose_level);
	void schreier_sims(int verbose_level);
	void get_strong_generators(int verbose_level);
	void point_stabilizer(group &stab, int pt, int verbose_level);
	void point_stabilizer_with_action(action *A2, 
		group &stab, int pt, int verbose_level);
	void induced_action(action &induced_action, 
		group &H, group &K, int verbose_level);
	void extension(group &N, group &H, int verbose_level);
		// N needs to have strong generators, 
		// H needs to have sims
		// N and H may have different actions, 
		// the action of N is taken for the extension.
	void print_strong_generators(ostream &ost, 
		int f_print_as_permutation);
	void print_strong_generators_with_different_action(
		ostream &ost, action *A2);
	void print_strong_generators_with_different_action_verbose(
		ostream &ost, action *A2, int verbose_level);

};

// #############################################################################
// orbit_transversal.C:
// #############################################################################

//! a container data structure for a poset classification from Orbiter output


class orbit_transversal {

public:
	action *A;
	action *A2;
	
	int nb_orbits;
	set_and_stabilizer *Reps;

	orbit_transversal();
	~orbit_transversal();
	void null();
	void freeself();
	void read_from_file(action *A, action *A2, 
		const char *fname, int verbose_level);
};

// #############################################################################
// page_storage.C:
// #############################################################################

//! a data structure to store group elements in compressed form



class page_storage {

public:
	int overall_length;
	
	int entry_size; // in char
	int page_length_log; // number of bits
	int page_length; // entries per page
	int page_size; // size in char of one page
	int allocation_table_length;
		// size in char of one allocation table
	
	int page_ptr_used;
	int page_ptr_allocated;
	int page_ptr_oversize;
	
	uchar **pages;
	uchar **allocation_tables;
	
	int next_free_entry;
	int nb_free_entries;
	
	int f_elt_print_function;
	void (* elt_print)(void *p, void *data, ostream &ost);
	void *elt_print_data;


	void init(int entry_size, int page_length_log, 
		int verbose_level);
	void add_elt_print_function(
		void (* elt_print)(void *p, void *data, ostream &ost), 
		void *elt_print_data);
	void print();
	uchar *s_i_and_allocate(int i);
	uchar *s_i_and_deallocate(int i);
	uchar *s_i(int i);
	uchar *s_i_and_allocation_bit(int i, int &f_allocated);
	void check_allocation_table();
	int store(uchar *elt);
	void dispose(int hdl);
	void check_free_list();
	page_storage();
	~page_storage();
	void print_storage_used();
	
};

void test_page_storage(int f_v);



// #############################################################################
// projective_space_with_action.C:
// #############################################################################


//! projective space PG(n,q) with automorphism group PGGL(n+1,q)



class projective_space_with_action {

public:

	int n;
	int d; // n + 1
	int q;
	finite_field *F; // do not free
	int f_semilinear;
	int f_init_incidence_structure;

	projective_space *P;
	

	action *A; // linear group PGGL(d,q)
	action *A_on_lines; // linear group PGGL(d,q) acting on lines
	sims *S; // linear group PGGL(d,q)

	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, int n, int f_semilinear, 
		int f_init_incidence_structure, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	strong_generators *set_stabilizer(
		int *set, int set_size, int &canonical_pt, 
		int *canonical_set_or_NULL, 
		int f_save_incma_in_and_out, 
		const char *save_incma_in_and_out_prefix, 
		int f_compute_canonical_form, 
		uchar *&canonical_form, int &canonical_form_len, 
		int verbose_level);
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP, 
		int f_save_incma_in_and_out, 
		const char *save_incma_in_and_out_prefix, 
		int f_compute_canonical_form, 
		uchar *&canonical_form, 
		int &canonical_form_len, 
		int verbose_level);
	void report_fixed_objects_in_PG_3_tex(
		int *Elt, ostream &ost, 
		int verbose_level);
};

// #############################################################################
// schreier_vector_handler.cpp:
// #############################################################################

//! manages access to schreier vectors


class schreier_vector_handler {
public:
	action *A;
	action *A2;
	int *cosetrep;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int f_check_image;
	int f_allow_failure;
	int nb_calls_to_coset_rep_inv;
	int nb_calls_to_coset_rep_inv_recursion;

	schreier_vector_handler();
	~schreier_vector_handler();
	void null();
	void freeself();
	void init(action *A, action *A2,
			int f_allow_failure,
			int verbose_level);
	int coset_rep_inv(
			schreier_vector *S,
			int pt, int &pt0,
			int verbose_level);
	int coset_rep_inv_recursion(
		schreier_vector *S,
		int pt, int &pt0,
		int verbose_level);
	schreier_vector *sv_read_file(
			int gen_hdl_first, int nb_gen,
			FILE *fp, int verbose_level);
	void sv_write_file(schreier_vector *Sv,
			FILE *fp, int verbose_level);
	set_of_sets *get_orbits_as_set_of_sets(schreier_vector *Sv,
			int verbose_level);

};

// #############################################################################
// schreier_vector.cpp:
// #############################################################################

//! compact storage of schreier vectors


class schreier_vector {
public:
	int gen_hdl_first;
	int nb_gen;
	int number_of_orbits;
	int *sv;
		// the length of sv is n+1 if the group is trivial
		// and 3*n + 1 otherwise.
		//
		// sv[0] = n = number of points in the set on which we act
		// the next n entries are the points of the set
		// the next 2*n entries only exist if the group is non-trivial:
		// the next n entries are the previous pointers
		// the next n entries are the labels
	int f_has_local_generators;
	vector_ge *local_gens;

	schreier_vector();
	~schreier_vector();
	void null();
	void freeself();
	void init(int gen_hdl_first, int nb_gen, int *sv,
			int verbose_level);
	void set_sv(int *sv, int verbose_level);
	int *points();
	int *prev();
	int *label();
	int get_number_of_points();
	int get_number_of_orbits();
	int count_number_of_orbits();
	void count_number_of_orbits_and_get_orbit_reps(
			int *&orbit_reps, int &nb_orbits);
	int determine_depth_recursion(
		int n, int *pts, int *prev,
		int *depth, int *ancestor, int pos);
	void relabel_points(
		action_on_factor_space *AF,
		int verbose_level);
	void orbit_of_point(
			int pt, int *&orbit_elts, int &orbit_len,
			int verbose_level);
	void init_from_schreier(schreier *S,
		int f_trivial_group, int verbose_level);
	void init_shallow_schreier_forest(schreier *S,
		int f_trivial_group, int verbose_level);
	void export_tree_as_layered_graph(
			int orbit_no, int orbit_rep,
			const char *fname_mask,
			int verbose_level);
	void trace_back(int pt, int &depth);
};

int schreier_vector_determine_depth_recursion(
	int n, int *pts, int *prev,
	int *depth, int *ancestor, int pos);



// #############################################################################
// set_and_stabilizer.C:
// #############################################################################


//! a set and its known set stabilizer



class set_and_stabilizer {

public:
	action *A;
	action *A2;
	int *data;
	int sz;
	longinteger_object target_go;
	strong_generators *Strong_gens;
	sims *Stab;

	set_and_stabilizer();
	~set_and_stabilizer();
	void null();
	void freeself();
	void init(action *A, action *A2, int verbose_level);
	void init_everything(action *A, action *A2, int *Set, int set_sz, 
		strong_generators *gens, int verbose_level);
	void allocate_data(int sz, int verbose_level);
	set_and_stabilizer *create_copy(int verbose_level);
	void init_data(int *data, int sz, int verbose_level);
	void init_stab_from_data(int *data_gens, 
		int data_gens_size, int nb_gens, const char *ascii_target_go, 
		int verbose_level);
	void init_stab_from_file(const char *fname_gens, 
		int verbose_level);
	void print_set_tex(ostream &ost);
	void print_set_tex_for_inline_text(ostream &ost);
	void print_generators_tex(ostream &ost);
	//set_and_stabilizer *apply(int *Elt, int verbose_level);
	void apply_to_self(int *Elt, int verbose_level);
	void apply_to_self_inverse(int *Elt, int verbose_level);
	void apply_to_self_element_raw(int *Elt_data, int verbose_level);
	void apply_to_self_inverse_element_raw(int *Elt_data, 
		int verbose_level);
	void rearrange_by_orbits(int *&orbit_first, 
		int *&orbit_length, int *&orbit, 
		int &nb_orbits, int verbose_level);
	action *create_restricted_action_on_the_set(int verbose_level);
	void print_restricted_action_on_the_set(int verbose_level);
	void test_if_group_acts(int verbose_level);
	
	void init_surface(surface *Surf, action *A, action *A2, 
		int q, int no, int verbose_level);
};

// #############################################################################
// union_find.C:
// #############################################################################


//! a union find data structure (used in the poset classification)




class union_find {

public:
	action *A;
	int *prev;


	union_find();
	~union_find();
	void freeself();
	void null();
	void init(action *A, int verbose_level);
	int ancestor(int i);
	int count_ancestors();
	int count_ancestors_above(int i0);
	void do_union(int a, int b);
	void print();
	void add_generators(vector_ge *gens, int verbose_level);
	void add_generator(int *Elt, int verbose_level);
};

// #############################################################################
// union_find_on_k_subsets.C:
// #############################################################################

//! a union find data structure (used in the poset classification)



class union_find_on_k_subsets {

public:

	int *set;
	int set_sz;
	int k;

	sims *S;

	int *interesting_k_subsets;
	int nb_interesting_k_subsets;
	
	action *A_original;
	action *Ar; // restricted action on the set
	action *Ar_perm;
	action *Ark; // Ar_perm on k_subsets
	action *Arkr; // Ark restricted to interesting_k_subsets

	vector_ge *gens_perm;

	union_find *UF;


	union_find_on_k_subsets();
	~union_find_on_k_subsets();
	void freeself();
	void null();
	void init(action *A_original, sims *S, 
		int *set, int set_sz, int k, 
		int *interesting_k_subsets, int nb_interesting_k_subsets, 
		int verbose_level);
	int is_minimal(int rk, int verbose_level);
};


