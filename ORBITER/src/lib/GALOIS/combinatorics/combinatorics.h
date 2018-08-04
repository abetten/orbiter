// combinatorics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// brick_domain.C:
// #############################################################################

class brick_domain {

public:
	finite_field *F;
	INT q;
	INT nb_bricks;

	brick_domain();
	~brick_domain();
	void null();
	void freeself();
	void init(finite_field *F, INT verbose_level);
	void unrank(INT rk, INT &f_vertical, 
		INT &x0, INT &y0, INT verbose_level);
	INT rank(INT f_vertical, INT x0, INT y0, INT verbose_level);
	void unrank_coordinates(INT rk, 
		INT &x1, INT &y1, INT &x2, INT &y2, 
		INT verbose_level);
	INT rank_coordinates(INT x1, INT y1, INT x2, INT y2, 
		INT verbose_level);
};

void brick_test(INT q, INT verbose_level);

// #############################################################################
// classify_bitvectors.C:
// #############################################################################


class classify_bitvectors {
public:

	INT nb_types; 
		// the number of isomorphism types

	INT rep_len;
		// the number of UBYTE we need to store the canonical form of 
		// one object


	UBYTE **Type_data; 
		// Type_data[N][rep_len]
		// the canonical form of the i-th representative is 
		// Type_data[i][rep_len]
	INT *Type_rep;
		// Type_rep[N]
		// Type_rep[i] is the index of the canidate which 
		// has been chosen as representative 
		// for the i-th isomorphism type
	INT *Type_mult; 
		// Type_mult[N]
		// Type_mult[i] gives the number of candidates so far which 
		// are isomorphic to the i-th isomorphism class representative
	void **Type_extra_data;
		// Type_extra_data[N]
		// Type_extra_data[i] is a pointer that is stored with the 
		// i-th isomorphism class representative
	
	INT N; 
		// number of candidates (or objects) that we will test
	INT n; 
		// number of candidates that we have already tested

	INT *type_of;
		// type_of[N]
		// type_of[i] is the isomorphism type of the i-th candidate

	classify *C_type_of;
		// the classification of type_of[N]
		// this will be computed in finalize()

	INT *perm;
		// the permutation which lists the orbit 
		// representative in the order 
		// in which they appear in the list of candidates
	
	classify_bitvectors();
	~classify_bitvectors();
	void null();
	void freeself();
	void init(INT N, INT rep_len, INT verbose_level);
	INT add(UBYTE *data, void *extra_data, INT verbose_level);
	void finalize(INT verbose_level);
	void print_reps();
	void save(const BYTE *prefix, 
		void (*encode_function)(void *extra_data, 
			INT *&encoding, INT &encoding_sz, void *global_data),
		void (*get_group_order_or_NULL)(void *extra_data, 
			longinteger_object &go, void *global_data), 
		void *global_data, 
		INT verbose_level);

};

INT compare_func_for_bitvectors(void *a, void *b, void *data);

// #############################################################################
// combinatorics.C:
// #############################################################################

INT Hamming_distance_binary(INT a, INT b, INT n);
INT INT_factorial(INT a);
INT Kung_mue_i(INT *part, INT i, INT m);
void partition_dual(INT *part, INT *dual_part, INT n, INT verbose_level);
void make_all_partitions_of_n(INT n, INT *&Table, INT &nb, INT verbose_level);
INT count_all_partitions_of_n(INT n);
INT partition_first(INT *v, INT n);
INT partition_next(INT *v, INT n);
void partition_print(ostream &ost, INT *v, INT n);
INT INT_vec_is_regular_word(INT *v, INT len, INT q);
	// Returns TRUE if the word v of length len is regular, i.~e. 
	// lies in an orbit of length $len$ under the action of the cyclic group 
	// $C_{len}$ acting on the coordinates. 
	// Lueneburg~\cite{Lueneburg87a} p. 118.
	// v is a vector over $\{0, 1, \ldots , q-1\}$
INT INT_vec_first_regular_word(INT *v, INT len, INT Q, INT q);
INT INT_vec_next_regular_word(INT *v, INT len, INT Q, INT q);
INT is_subset_of(INT *A, INT sz_A, INT *B, INT sz_B);
INT set_find(INT *elts, INT size, INT a);
void set_complement(INT *subset, INT subset_size, INT *complement, 
	INT &size_complement, INT universal_set_size);
void set_complement_safe(INT *subset, INT subset_size, INT *complement, 
	INT &size_complement, INT universal_set_size);
// subset does not need to be in increasing order
void set_add_elements(INT *elts, INT &size, 
	INT *elts_to_add, INT nb_elts_to_add);
void set_add_element(INT *elts, INT &size, INT a);
void set_delete_elements(INT *elts, INT &size, 
	INT *elts_to_delete, INT nb_elts_to_delete);
void set_delete_element(INT *elts, INT &size, INT a);
INT compare_lexicographically(INT a_len, INT *a, INT b_len, INT *b);
INT INT_n_choose_k(INT n, INT k);
void make_t_k_incidence_matrix(INT v, INT t, INT k, INT &m, INT &n, INT *&M, 
	INT verbose_level);
void print_k_subsets_by_rank(ostream &ost, INT v, INT k);
INT f_is_subset_of(INT v, INT t, INT k, INT rk_t_subset, INT rk_k_subset);
INT rank_subset(INT *set, INT sz, INT n);
void rank_subset_recursion(INT *set, INT sz, INT n, INT a0, INT &r);
void unrank_subset(INT *set, INT &sz, INT n, INT r);
void unrank_subset_recursion(INT *set, INT &sz, INT n, INT a0, INT &r);
INT rank_k_subset(INT *set, INT n, INT k);
void unrank_k_subset(INT rk, INT *set, INT n, INT k);
INT first_k_subset(INT *set, INT n, INT k);
INT next_k_subset(INT *set, INT n, INT k);
INT next_k_subset_at_level(INT *set, INT n, INT k, INT backtrack_level);
void subset_permute_up_front(INT n, INT k, INT *set, INT *k_subset_idx, 
	INT *permuted_set);
INT ordered_pair_rank(INT i, INT j, INT n);
void ordered_pair_unrank(INT rk, INT &i, INT &j, INT n);
INT unordered_triple_pair_rank(INT i, INT j, INT k, INT l, INT m, INT n);
void unordered_triple_pair_unrank(INT rk, INT &i, INT &j, INT &k, 
	INT &l, INT &m, INT &n);
INT ij2k(INT i, INT j, INT n);
void k2ij(INT k, INT & i, INT & j, INT n);
INT ijk2h(INT i, INT j, INT k, INT n);
void h2ijk(INT h, INT &i, INT &j, INT &k, INT n);
void random_permutation(INT *random_permutation, INT n);
void perm_move(INT *from, INT *to, INT n);
void perm_identity(INT *a, INT n);
void perm_elementary_transposition(INT *a, INT n, INT f);
void perm_mult(INT *a, INT *b, INT *c, INT n);
void perm_conjugate(INT *a, INT *b, INT *c, INT n);
// c := a^b = b^-1 * a * b
void perm_inverse(INT *a, INT *b, INT n);
// b := a^-1
void perm_raise(INT *a, INT *b, INT e, INT n);
// b := a^e (e >= 0)
void perm_direct_product(INT n1, INT n2, INT *perm1, INT *perm2, INT *perm3);
void perm_print_list(ostream &ost, INT *a, INT n);
void perm_print_list_offset(ostream &ost, INT *a, INT n, INT offset);
void perm_print_product_action(ostream &ost, INT *a, INT m_plus_n, INT m, 
	INT offset, INT f_cycle_length);
void perm_print(ostream &ost, INT *a, INT n);
void perm_print_with_cycle_length(ostream &ost, INT *a, INT n);
void perm_print_counting_from_one(ostream &ost, INT *a, INT n);
void perm_print_offset(ostream &ost, INT *a, INT n, INT offset, 
	INT f_cycle_length, 
	INT f_max_cycle_length, INT max_cycle_length, INT f_orbit_structure);
INT perm_order(INT *a, INT n);
INT perm_signum(INT *perm, INT n);
void first_lehmercode(INT n, INT *v);
INT next_lehmercode(INT n, INT *v);
void lehmercode_to_permutation(INT n, INT *code, INT *perm);
INT disjoint_binary_representation(INT u, INT v);
INT hall_test(INT *A, INT n, INT kmax, INT *memo, INT verbose_level);
INT philip_hall_test(INT *A, INT n, INT k, INT *memo, INT verbose_level);
// memo points to free memory of n INT's
INT philip_hall_test_dual(INT *A, INT n, INT k, INT *memo, INT verbose_level);
// memo points to free memory of n INT's
void print_01_matrix_with_stars(ostream &ost, INT *A, INT m, INT n);
void print_INT_matrix(ostream &ost, INT *A, INT m, INT n);
INT create_roots_H4(finite_field *F, INT *roots);
INT generalized_binomial(INT n, INT k, INT q);
void print_tableau(INT *Tableau, INT l1, INT l2, 
	INT *row_parts, INT *col_parts);


