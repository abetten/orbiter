// combinatorics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

namespace orbiter {
namespace foundations {


// #############################################################################
// brick_domain.C:
// #############################################################################

//! a problem of Neil Sloane

class brick_domain {

public:
	finite_field *F;
	int q;
	int nb_bricks;

	brick_domain();
	~brick_domain();
	void null();
	void freeself();
	void init(finite_field *F, int verbose_level);
	void unrank(int rk, int &f_vertical, 
		int &x0, int &y0, int verbose_level);
	int rank(int f_vertical, int x0, int y0, int verbose_level);
	void unrank_coordinates(int rk, 
		int &x1, int &y1, int &x2, int &y2, 
		int verbose_level);
	int rank_coordinates(int x1, int y1, int x2, int y2, 
		int verbose_level);
};

void brick_test(int q, int verbose_level);

// #############################################################################
// combinatorics_domain.cpp
// #############################################################################

//! a collection of combinatorial functions

class combinatorics_domain {

public:
	combinatorics_domain();
	~combinatorics_domain();
	int Hamming_distance_binary(int a, int b, int n);
	int int_factorial(int a);
	int Kung_mue_i(int *part, int i, int m);
	void partition_dual(int *part, int *dual_part, int n, int verbose_level);
	void make_all_partitions_of_n(int n, int *&Table, int &nb, int verbose_level);
	int count_all_partitions_of_n(int n);
	int partition_first(int *v, int n);
	int partition_next(int *v, int n);
	void partition_print(std::ostream &ost, int *v, int n);
	int int_vec_is_regular_word(int *v, int len, int q);
		// Returns TRUE if the word v of length len is regular, i.~e.
		// lies in an orbit of length $len$ under the action of the cyclic group
		// $C_{len}$ acting on the coordinates.
		// Lueneburg~\cite{Lueneburg87a} p. 118.
		// v is a vector over $\{0, 1, \ldots , q-1\}$
	int int_vec_first_regular_word(int *v, int len, int Q, int q);
	int int_vec_next_regular_word(int *v, int len, int Q, int q);
	int is_subset_of(int *A, int sz_A, int *B, int sz_B);
	int set_find(int *elts, int size, int a);
	void set_complement(int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	void set_complement_safe(int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	// subset does not need to be in increasing order
	void set_add_elements(int *elts, int &size,
		int *elts_to_add, int nb_elts_to_add);
	void set_add_element(int *elts, int &size, int a);
	void set_delete_elements(int *elts, int &size,
		int *elts_to_delete, int nb_elts_to_delete);
	void set_delete_element(int *elts, int &size, int a);
	int compare_lexicographically(int a_len, int *a, int b_len, int *b);
	int int_n_choose_k(int n, int k);
	void make_t_k_incidence_matrix(int v, int t, int k, int &m, int &n, int *&M,
		int verbose_level);
	void print_k_subsets_by_rank(std::ostream &ost, int v, int k);
	int f_is_subset_of(int v, int t, int k, int rk_t_subset, int rk_k_subset);
	int rank_subset(int *set, int sz, int n);
	void rank_subset_recursion(int *set, int sz, int n, int a0, int &r);
	void unrank_subset(int *set, int &sz, int n, int r);
	void unrank_subset_recursion(int *set, int &sz, int n, int a0, int &r);
	int rank_k_subset(int *set, int n, int k);
	void unrank_k_subset(int rk, int *set, int n, int k);
	int first_k_subset(int *set, int n, int k);
	int next_k_subset(int *set, int n, int k);
	int next_k_subset_at_level(int *set, int n, int k, int backtrack_level);
	void subset_permute_up_front(int n, int k, int *set, int *k_subset_idx,
		int *permuted_set);
	int ordered_pair_rank(int i, int j, int n);
	void ordered_pair_unrank(int rk, int &i, int &j, int n);
	void set_partition_4_into_2_unrank(int rk, int *v);
	int set_partition_4_into_2_rank(int *v);
	int unordered_triple_pair_rank(int i, int j, int k, int l, int m, int n);
	void unordered_triple_pair_unrank(int rk, int &i, int &j, int &k,
		int &l, int &m, int &n);
	int ij2k(int i, int j, int n);
	void k2ij(int k, int & i, int & j, int n);
	int ijk2h(int i, int j, int k, int n);
	void h2ijk(int h, int &i, int &j, int &k, int n);
	void random_permutation(int *random_permutation, int n);
	void perm_move(int *from, int *to, int n);
	void perm_identity(int *a, int n);
	int perm_is_identity(int *a, int n);
	void perm_elementary_transposition(int *a, int n, int f);
	void perm_mult(int *a, int *b, int *c, int n);
	void perm_conjugate(int *a, int *b, int *c, int n);
	// c := a^b = b^-1 * a * b
	void perm_inverse(int *a, int *b, int n);
	// b := a^-1
	void perm_raise(int *a, int *b, int e, int n);
	// b := a^e (e >= 0)
	void perm_direct_product(int n1, int n2, int *perm1, int *perm2, int *perm3);
	void perm_print_list(std::ostream &ost, int *a, int n);
	void perm_print_list_offset(std::ostream &ost, int *a, int n, int offset);
	void perm_print_product_action(std::ostream &ost, int *a, int m_plus_n, int m,
		int offset, int f_cycle_length);
	void perm_print(std::ostream &ost, int *a, int n);
	void perm_print_with_cycle_length(std::ostream &ost, int *a, int n);
	void perm_print_counting_from_one(std::ostream &ost, int *a, int n);
	void perm_print_offset(std::ostream &ost, int *a, int n, int offset,
		int f_cycle_length,
		int f_max_cycle_length, int max_cycle_length, int f_orbit_structure);
	void perm_cycle_type(int *perm, int degree, int *cycles, int &nb_cycles);
	int perm_order(int *a, int n);
	int perm_signum(int *perm, int n);
	int is_permutation(int *perm, int n);
	void first_lehmercode(int n, int *v);
	int next_lehmercode(int n, int *v);
	void lehmercode_to_permutation(int n, int *code, int *perm);
	int disjoint_binary_representation(int u, int v);
	int hall_test(int *A, int n, int kmax, int *memo, int verbose_level);
	int philip_hall_test(int *A, int n, int k, int *memo, int verbose_level);
	// memo points to free memory of n int's
	int philip_hall_test_dual(int *A, int n, int k, int *memo, int verbose_level);
	// memo points to free memory of n int's
	void print_01_matrix_with_stars(std::ostream &ost, int *A, int m, int n);
	void print_int_matrix(std::ostream &ost, int *A, int m, int n);
	int create_roots_H4(finite_field *F, int *roots);
	int generalized_binomial(int n, int k, int q);
	void print_tableau(int *Tableau, int l1, int l2,
		int *row_parts, int *col_parts);
	int ijk_rank(int i, int j, int k, int n);
	void ijk_unrank(int &i, int &j, int &k, int n, int rk);
	int largest_binomial2_below(int a2);
	int largest_binomial3_below(int a3);
	int binomial2(int a);
	int binomial3(int a);
	int minus_one_if_positive(int i);
	//int int_ij2k(int i, int j, int n);
	//void int_k2ij(int k, int & i, int & j, int n);
};

// combinatorics.cpp:
int callback_ij2k(int i, int j, int n);


}}
