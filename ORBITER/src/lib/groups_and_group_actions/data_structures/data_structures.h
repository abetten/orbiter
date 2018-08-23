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

	INT f_has_ascii_coding;
	char *ascii_coding;

	INT f_has_strong_generators;
	vector_ge *SG;
	INT *tl;
	
	INT f_has_sims;
	sims *S;
	
	group();
	~group();
	void null();
	void freeself();
	group(action *A);
	group(action *A, const char *ascii_coding);
	group(action *A, vector_ge &SG, INT *tl);
	void init(action *A);
	void init_ascii_coding_to_sims(const char *ascii_coding);
	void init_ascii_coding(const char *ascii_coding);
	void delete_ascii_coding();
	void delete_sims();
	void init_strong_generators_empty_set();
	void init_strong_generators(vector_ge &SG, INT *tl);
	void init_strong_generators_by_hdl(INT nb_gen, INT *gen_hdl, 
		INT *tl, INT verbose_level);
	void delete_strong_generators();
	void require_ascii_coding();
	void require_strong_generators();
	void require_sims();
	void group_order(longinteger_object &go);
	void print_group_order(ostream &ost);
	void print_tl();
	void code_ascii(INT verbose_level);
	void decode_ascii(INT verbose_level);
	void schreier_sims(INT verbose_level);
	void get_strong_generators(INT verbose_level);
	void point_stabilizer(group &stab, INT pt, INT verbose_level);
	void point_stabilizer_with_action(action *A2, 
		group &stab, INT pt, INT verbose_level);
	void induced_action(action &induced_action, 
		group &H, group &K, INT verbose_level);
	void extension(group &N, group &H, INT verbose_level);
		// N needs to have strong generators, 
		// H needs to have sims
		// N and H may have different actions, 
		// the action of N is taken for the extension.
	void print_strong_generators(ostream &ost, 
		INT f_print_as_permutation);
	void print_strong_generators_with_different_action(
		ostream &ost, action *A2);
	void print_strong_generators_with_different_action_verbose(
		ostream &ost, action *A2, INT verbose_level);

};

// #############################################################################
// orbit_transversal.C:
// #############################################################################

//! a container data structure for a poset classification from Orbiter output


class orbit_transversal {

public:
	action *A;
	action *A2;
	
	INT nb_orbits;
	set_and_stabilizer *Reps;

	orbit_transversal();
	~orbit_transversal();
	void null();
	void freeself();
	void read_from_file(action *A, action *A2, 
		const BYTE *fname, INT verbose_level);
};

// #############################################################################
// page_storage.C:
// #############################################################################

//! a data structure to store group elements in compressed form



class page_storage {

public:
	INT overall_length;
	
	INT entry_size; // in BYTE
	INT page_length_log; // number of bits
	INT page_length; // entries per page
	INT page_size; // size in BYTE of one page
	INT allocation_table_length;
		// size in BYTE of one allocation table
	
	INT page_ptr_used;
	INT page_ptr_allocated;
	INT page_ptr_oversize;
	
	UBYTE **pages;
	UBYTE **allocation_tables;
	
	INT next_free_entry;
	INT nb_free_entries;
	
	void init(INT entry_size, INT page_length_log, 
		INT verbose_level);
	void add_elt_print_function(
		void (* elt_print)(void *p, void *data, ostream &ost), 
		void *elt_print_data);
	void print();
	UBYTE *s_i_and_allocate(INT i);
	UBYTE *s_i_and_deallocate(INT i);
	UBYTE *s_i(INT i);
	UBYTE *s_i_and_allocation_bit(INT i, INT &f_allocated);
	void check_allocation_table();
	INT store(UBYTE *elt);
	void dispose(INT hdl);
	void check_free_list();
	page_storage();
	~page_storage();
	void print_storage_used();
	
	INT f_elt_print_function;
	void (* elt_print)(void *p, void *data, ostream &ost);
	void *elt_print_data;
};

void test_page_storage(INT f_v);


// #############################################################################
// projective_space_with_action.C:
// #############################################################################


//! projective space PG(n,q) with automorphism group PGGL(n+1,q)



class projective_space_with_action {

public:

	INT n;
	INT d; // n + 1
	INT q;
	finite_field *F; // do not free
	INT f_semilinear;
	INT f_init_incidence_structure;

	projective_space *P;
	

	action *A; // linear group PGGL(d,q)
	action *A_on_lines; // linear group PGGL(d,q) acting on lines
	sims *S; // linear group PGGL(d,q)

	INT *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, INT n, INT f_semilinear, 
		INT f_init_incidence_structure, INT verbose_level);
	void init_group(INT f_semilinear, INT verbose_level);
	strong_generators *set_stabilizer(
		INT *set, INT set_size, INT &canonical_pt, 
		INT *canonical_set_or_NULL, 
		INT f_save_incma_in_and_out, 
		const BYTE *save_incma_in_and_out_prefix, 
		INT f_compute_canonical_form, 
		UBYTE *&canonical_form, INT &canonical_form_len, 
		INT verbose_level);
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP, 
		INT f_save_incma_in_and_out, 
		const BYTE *save_incma_in_and_out_prefix, 
		INT f_compute_canonical_form, 
		UBYTE *&canonical_form, 
		INT &canonical_form_len, 
		INT verbose_level);
	void report_fixed_objects_in_PG_3_tex(
		INT *Elt, ostream &ost, 
		INT verbose_level);
};

// #############################################################################
// schreier_vector.C:
// #############################################################################


INT schreier_vector_coset_rep_inv_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT f_allow_failure, INT verbose_level);
// determines pt0 to be the first point of the orbit containing pt.
// cosetrep will be a group element that maps pt to pt0.
INT schreier_vector_coset_rep_inv_compact_general(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT f_allow_failure, INT verbose_level);
void schreier_vector_coset_rep_inv(action *A, INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_compact, INT f_check_image, 
	INT verbose_level);
	// determines pt0 to be the first point of the 
	// orbit containing pt.
	// cosetrep will be a group element that maps pt to pt0.
void schreier_vector_coset_rep_inv_compact(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3, 
	INT f_trivial_group, INT f_check_image, 
	INT verbose_level);
void schreier_vector_coset_rep_inv1(action *A, 
	INT *sv, INT *hdl_gen, INT pt, 
	INT &pt0, INT *cosetrep, INT *Elt1, INT *Elt2, INT *Elt3);


void schreier_vector_print(INT *sv);
void schreier_vector_print_tree(INT *sv, INT verbose_level);
INT schreier_vector_compute_depth_recursively(INT n, 
	INT *Depth, INT *pts, INT *prev, INT pt);
INT sv_number_of_orbits(INT *sv);
void analyze_schreier_vector(INT *sv, INT verbose_level);
	// we assume that the group is not trivial
INT schreier_vector_determine_depth_recursion(INT n, INT *pts, INT *prev, 
	INT *depth, INT *ancestor, INT pos);

// #############################################################################
// set_and_stabilizer.C:
// #############################################################################


//! a set and its known set stabilizer



class set_and_stabilizer {

public:
	action *A;
	action *A2;
	INT *data;
	INT sz;
	longinteger_object target_go;
	strong_generators *Strong_gens;
	sims *Stab;

	set_and_stabilizer();
	~set_and_stabilizer();
	void null();
	void freeself();
	void init(action *A, action *A2, INT verbose_level);
	void init_everything(action *A, action *A2, INT *Set, INT set_sz, 
		strong_generators *gens, INT verbose_level);
	void allocate_data(INT sz, INT verbose_level);
	set_and_stabilizer *create_copy(INT verbose_level);
	void init_data(INT *data, INT sz, INT verbose_level);
	void init_stab_from_data(INT *data_gens, 
		INT data_gens_size, INT nb_gens, const BYTE *ascii_target_go, 
		INT verbose_level);
	void init_stab_from_file(const BYTE *fname_gens, 
		INT verbose_level);
	void print_set_tex(ostream &ost);
	void print_set_tex_for_inline_text(ostream &ost);
	void print_generators_tex(ostream &ost);
	//set_and_stabilizer *apply(INT *Elt, INT verbose_level);
	void apply_to_self(INT *Elt, INT verbose_level);
	void apply_to_self_inverse(INT *Elt, INT verbose_level);
	void apply_to_self_element_raw(INT *Elt_data, INT verbose_level);
	void apply_to_self_inverse_element_raw(INT *Elt_data, 
		INT verbose_level);
	void rearrange_by_orbits(INT *&orbit_first, 
		INT *&orbit_length, INT *&orbit, 
		INT &nb_orbits, INT verbose_level);
	action *create_restricted_action_on_the_set(INT verbose_level);
	void print_restricted_action_on_the_set(INT verbose_level);
	void test_if_group_acts(INT verbose_level);
	
	void init_surface(surface *Surf, action *A, action *A2, 
		INT q, INT no, INT verbose_level);
};

// #############################################################################
// union_find.C:
// #############################################################################


//! a union find data structure (used in the poset classification)




class union_find {

public:
	action *A;
	INT *prev;


	union_find();
	~union_find();
	void freeself();
	void null();
	void init(action *A, INT verbose_level);
	INT ancestor(INT i);
	INT count_ancestors();
	INT count_ancestors_above(INT i0);
	void do_union(INT a, INT b);
	void print();
	void add_generators(vector_ge *gens, INT verbose_level);
	void add_generator(INT *Elt, INT verbose_level);
};

// #############################################################################
// union_find_on_k_subsets.C:
// #############################################################################

//! a union find data structure (used in the poset classification)



class union_find_on_k_subsets {

public:

	INT *set;
	INT set_sz;
	INT k;

	sims *S;

	INT *interesting_k_subsets;
	INT nb_interesting_k_subsets;
	
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
		INT *set, INT set_sz, INT k, 
		INT *interesting_k_subsets, INT nb_interesting_k_subsets, 
		INT verbose_level);
	INT is_minimal(INT rk, INT verbose_level);
};


