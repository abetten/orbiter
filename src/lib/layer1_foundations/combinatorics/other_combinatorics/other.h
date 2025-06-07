// combinatorics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



#ifndef ORBITER_SRC_LIB_FOUNDATIONS_COMBINATORICS_COMBINATORICS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_COMBINATORICS_COMBINATORICS_H_



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace other_combinatorics {









// #############################################################################
// combinatorics_domain.cpp
// #############################################################################

//! a collection of combinatorial functions

class combinatorics_domain {

public:

	special_functions::permutations *Permutations;


	combinatorics_domain();
	~combinatorics_domain();
	int int_factorial(
			int a);
	int Kung_mue_i(
			int *part, int i, int m);
	void partition_dual(
			int *part, int *dual_part, int n,
			int verbose_level);
	void make_all_partitions_of_n(
			int n, int *&Table, int &nb, int verbose_level);
	int count_all_partitions_of_n(
			int n);
	int partition_first(
			int *v, int n);
	int partition_next(
			int *v, int n);
	void partition_print(
			std::ostream &ost, int *v, int n);
	int int_vec_is_regular_word(
			int *v, int len, int q);
		// Returns true if the word v of length len is regular, i.~e.
		// lies in an orbit of length $len$ under the action of the cyclic group
		// $C_{len}$ acting on the coordinates.
		// Lueneburg~\cite{Lueneburg87a} p. 118.
		// v is a vector over $\{0, 1, \ldots , q-1\}$
	int int_vec_first_regular_word(
			int *v, int len, int q);
	int int_vec_next_regular_word(
			int *v, int len, int q);
	void int_vec_splice(
			int *v, int *w, int len, int p);
	int is_subset_of(
			int *A, int sz_A, int *B, int sz_B);
	int set_find(
			int *elts, int size, int a);
	void set_complement(
			int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	void set_complement_lint(
			long int *subset, int subset_size, long int *complement,
		int &size_complement, int universal_set_size);
	void set_complement_safe(
			int *subset, int subset_size, int *complement,
		int &size_complement, int universal_set_size);
	// subset does not need to be in increasing order
	void set_add_elements(
			int *elts, int &size,
		int *elts_to_add, int nb_elts_to_add);
	void set_add_element(
			int *elts, int &size, int a);
	void set_delete_elements(
			int *elts, int &size,
		int *elts_to_delete, int nb_elts_to_delete);
	void set_delete_element(
			int *elts, int &size, int a);
	int compare_lexicographically(
			int a_len, long int *a, int b_len, long int *b);
	long int int_n_choose_k(
			int n, int k);
	void make_t_k_incidence_matrix(
			int v, int t, int k, int &m, int &n, int *&M,
		int verbose_level);
	void print_k_subsets_by_rank(
			std::ostream &ost, int v, int k);
	int f_is_subset_of(
			int v, int t, int k,
			int rk_t_subset, int rk_k_subset);
	int rank_subset(
			int *set, int sz, int n);
	void rank_subset_recursion(
			int *set, int sz, int n, int a0, int &r);
	void unrank_subset(
			int *set, int &sz, int n, int r);
	void unrank_subset_recursion(
			int *set, int &sz, int n, int a0, int &r);
	int rank_k_subset(
			int *set, int n, int k);
	void unrank_k_subset(
			int rk, int *set, int n, int k);
	void unrank_k_subset_and_complement(
			int rk, int *set, int n, int k);
	int first_k_subset(
			int *set, int n, int k);
	int next_k_subset(
			int *set, int n, int k);
	int next_k_subset_at_level(
			int *set, int n, int k, int backtrack_level);
	void rank_k_subsets(
			int *Mtx, int nb_rows, int n, int k, int *&Ranks,
			int verbose_level);
	void rank_k_subsets_and_sort(
			int *Mtx, int nb_rows, int n, int k, int *&Ranks,
			int verbose_level);
	void subset_permute_up_front(
			int n, int k, int *set, int *k_subset_idx,
		int *permuted_set);
	int ordered_pair_rank(
			int i, int j, int n);
	void ordered_pair_unrank(
			int rk, int &i, int &j, int n);
	void set_partition_4_into_2_unrank(
			int rk, int *v);
	int set_partition_4_into_2_rank(
			int *v);
	int unordered_triple_pair_rank(
			int i, int j, int k, int l, int m, int n);
	void unordered_triple_pair_unrank(
			int rk, int &i, int &j, int &k,
		int &l, int &m, int &n);
	long int ij2k_lint(
			long int i, long int j, long int n);
	void k2ij_lint(
			long int k, long int & i, long int & j, long int n);
	int ij2k(
			int i, int j, int n);
	void k2ij(
			int k, int & i, int & j, int n);
	int ijk2h(
			int i, int j, int k, int n);
	void h2ijk(
			int h, int &i, int &j, int &k, int n);
	int disjoint_binary_representation(
			int u, int v);
	int hall_test(
			int *A, int n, int kmax, int *memo, int verbose_level);
	int philip_hall_test(
			int *A, int n, int k, int *memo, int verbose_level);
	int philip_hall_test_dual(
			int *A, int n, int k, int *memo, int verbose_level);
	void print_01_matrix_with_stars(
			std::ostream &ost, int *A, int m, int n);
	void print_int_matrix(
			std::ostream &ost, int *A, int m, int n);
	int create_roots_H4(
			algebra::field_theory::finite_field *F, int *roots);
	long int generalized_binomial(
			int n, int k, int q);
	void print_tableau(
			int *Tableau, int l1, int l2,
		int *row_parts, int *col_parts);
	int ijk_rank(
			int i, int j, int k, int n);
	void ijk_unrank(
			int &i, int &j, int &k, int n, int rk);
	long int largest_binomial2_below(
			int a2);
	long int largest_binomial3_below(
			int a3);
	long int binomial2(
			int a);
	long int binomial3(
			int a);
	int minus_one_if_positive(
			int i);
	void make_partitions(
			int n, int *Part, int cnt);
	int count_partitions(
			int n);
	int next_partition(
			int n, int *part);
	long int binomial_lint(
			int n, int k);
	void binomial(
			algebra::ring_theory::longinteger_object &a,
			int n, int k, int verbose_level);
	void binomial_with_table(
			algebra::ring_theory::longinteger_object &a,
			int n, int k);
	void size_of_conjugacy_class_in_sym_n(
			algebra::ring_theory::longinteger_object &a,
			int n, int *part);
	void q_binomial_with_table(
			algebra::ring_theory::longinteger_object &a,
		int n, int k, int q, int verbose_level);
	void q_binomial(
			algebra::ring_theory::longinteger_object &a,
		int n, int k, int q, int verbose_level);
	void q_binomial_no_table(
			algebra::ring_theory::longinteger_object &a,
		int n, int k, int q, int verbose_level);
	void krawtchouk_with_table(
			algebra::ring_theory::longinteger_object &a,
		int n, int q, int k, int x);
	void krawtchouk(
			algebra::ring_theory::longinteger_object &a,
			int n, int q, int k, int x);
	void make_elementary_symmetric_functions(
			int n, int k_max, int verbose_level);
	std::string stringify_elementary_symmetric_function(
			int nb_vars, int k, int verbose_level);
	void Dedekind_numbers(
			int n_min, int n_max, int q_min, int q_max,
			int verbose_level);
	void do_q_binomial(
			int n, int k, int q,
			int verbose_level);
	void do_read_poset_file(
			std::string &fname,
			int f_grouping, double x_stretch, int verbose_level);
	// creates a layered graph file from a text file
	// which was created by DISCRETA/sgls2.cpp
	void do_make_tree_of_all_k_subsets(
			int n, int k, int verbose_level);
	void create_random_permutation(
			int deg,
			std::string &fname_csv, int verbose_level);
	void create_random_k_subsets(
			int n, int k, int nb,
			std::string &fname_csv, int verbose_level);


	void free_global_data();
	void free_tab_q_binomials();



};










}}}}




#endif /* ORBITER_SRC_LIB_FOUNDATIONS_COMBINATORICS_COMBINATORICS_H_ */




