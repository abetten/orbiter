// orbits.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


#ifndef ORBITER_SRC_LIB_TOP_LEVEL_ORBITS_ORBITS_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_ORBITS_ORBITS_H_




namespace orbiter {
namespace layer4_classification {
namespace orbits_schreier {



// #############################################################################
// orbit_of_equations.cpp
// #############################################################################


//! orbit of homogeneous equations using a Schreier tree


class orbit_of_equations {
public:
	actions::action *A;
	induced_actions::action_on_homogeneous_polynomials *AonHPD;
	algebra::field_theory::finite_field *F;
	groups::strong_generators *SG;

	int nb_monomials;
	int sz; // = 1 + nb_monomials
	int sz_for_compare; // = 1 + nb_monomials
	int *data_tmp; // [sz]

	int position_of_original_object;
	int allocation_length;
	int used_length;

	int **Equations; // [allocation_length][sz]
	int *prev; // [allocation_length]
	int *label; // [allocation_length]

	int f_has_print_function;
	void (*print_function)(
			int *object,
			int sz, void *print_function_data);
	void *print_function_data;

	int f_has_reduction;
	void (*reduction_function)(
			int *object,
			void *reduction_function_data);
	void *reduction_function_data;


	orbit_of_equations();
	~orbit_of_equations();
	void init(
			actions::action *A,
			algebra::field_theory::finite_field *F,
			induced_actions::action_on_homogeneous_polynomials
				*AonHPD,
		groups::strong_generators *SG,
		int *coeff_in,
		int verbose_level);
	void map_an_equation(
			int *object_in, int *object_out,
		int *Elt, int verbose_level);
	void print_orbit();
	void print_orbit_as_equations_tex(
			std::ostream &ost);
	void compute_orbit(
			int *coeff, int verbose_level);
	void reallocate(
			int *&Q, int Q_len, int verbose_level);
	void get_table(
			std::string *&Table, std::string *&Headings,
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void get_transporter(
			int idx, int *transporter, int verbose_level);
		// transporter is an element which maps 
		// the orbit representative to the given subspace.
	void get_random_schreier_generator(
			int *Elt, int verbose_level);
	void get_canonical_form(
			int *canonical_equation,
			int *transporter_to_canonical_form,
			groups::strong_generators
				*&gens_stab_of_canonical_equation,
				algebra::ring_theory::longinteger_object
				&full_group_order,
			int verbose_level);
	groups::strong_generators *stabilizer_orbit_rep(
			algebra::ring_theory::longinteger_object &full_group_order,
			int verbose_level);
	void stabilizer_orbit_rep_work(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &go,
			groups::sims *&Stab, int verbose_level);
		// this function allocates a sims structure into Stab.
	groups::strong_generators *stabilizer_any_point(
			algebra::ring_theory::longinteger_object &full_group_order,
			int idx,
		int verbose_level);
	int search_equation(
			int *eqn, int &idx, int verbose_level);
	int search_data(
			int *data, int &idx, int verbose_level);
	void save_csv(
			std::string &fname, int verbose_level);
};


// #############################################################################
// orbit_of_sets.cpp
// #############################################################################






//! orbit of sets using a Schreier tree, used in packing::make_spread_table


class orbit_of_sets {
public:
	actions::action *A;
	actions::action *A2;
	data_structures_groups::vector_ge *gens;
	long int *set;
		// the set whose orbit we want to compute;
		// it is of size 'sz'
	int sz;

	int position_of_original_set;
		// = 0; never changes
	int allocation_length;
		// number of entries allocated in Sets
	int old_length;
	int used_length;
		// number of sets currently stored in Sets
	long int **Sets;
		// the sets are stored in the order in which they
		// are discovered and added to the tree
	int *Extra;
		// [allocation_length * 2]
		// Extra[i * 2 + 0] is the index of the ancestor node of node i.
		// Extra[i * 2 + 1] is the label of the generator that maps
		// the ancestor of node i to node i.
		// Here, node i means the set in Sets[i].
		// Node 0 is the root node, i.e. the set in 'set'.
	int *cosetrep; // the result of coset_rep()
	int *cosetrep_tmp; // temporary storage for coset_rep()

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// two sets may have the same hash value.
		// map cannot handle that.

	orbit_of_sets();
	~orbit_of_sets();
	void init(
			actions::action *A,
			actions::action *A2,
			long int *set, int sz,
			data_structures_groups::vector_ge *gens,
			int verbose_level);
	void compute(
			int verbose_level);
	int find_set(
			long int *new_set, int &pos, uint32_t &hash);
	void setup_root_node(
			long int *Q, int &Q_len, int verbose_level);
	void reallocate(
			long int *&Q, int Q_len, int verbose_level);
	void dump_tables_of_hash_values();
	void get_table_of_orbits(
			long int *&Table, int &orbit_length,
		int &set_size, int verbose_level);
	void get_table_of_orbits_and_hash_values(
			long int *&Table,
			int &orbit_length,
			int &set_size, int verbose_level);
	void make_table_of_coset_reps(
			data_structures_groups::vector_ge *&Coset_reps,
			int verbose_level);
	void get_path(
			std::vector<int> &path,
			int j);
	void coset_rep(
			int j);
		// result is in cosetrep
		// determines an element in the group
		// that moves the orbit representative
	// to the j-th element in the orbit.
	void get_orbit_of_points(
			std::vector<long int> &Orbit,
			int verbose_level);
	void get_prev(
			std::vector<int> &Prev,
			int verbose_level);
	void get_label(
			std::vector<int> &Label,
			int verbose_level);
	void export_tree_as_layered_graph(
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);

};


// #############################################################################
// orbit_of_subspaces.cpp
// #############################################################################



//! orbit of subspaces using a Schreier tree



class orbit_of_subspaces {
public:
	actions::action *A;
	actions::action *A2;
	algebra::field_theory::finite_field *F;
	data_structures_groups::vector_ge *gens;
	int f_lint;
	int k;
	int n;
	int kn;
	int sz; // = 1 + k + kn
	//int sz_for_compare; // = 1 + k + kn
	int f_has_desired_pivots;
	int *desired_pivots; // [k]
	int *subspace_by_rank; // [k]
	long int *subspace_by_rank_lint; // [k]
	int *data_tmp; // [sz]
	int *Mtx1;
	int *Mtx2;
	int *Mtx3;

	int f_has_rank_functions;
	void *rank_unrank_data;
	int (*rank_vector_callback)(
			int *v, int n,
		void *data, int verbose_level);
	long int (*rank_vector_lint_callback)(
			int *v, int n,
		void *data, int verbose_level);
	void (*unrank_vector_callback)(
			int rk, int *v,
		int n, void *data, int verbose_level);
	void (*unrank_vector_lint_callback)(
			long int rk, int *v,
		int n, void *data, int verbose_level);
	void (*compute_image_of_vector_callback)(
			int *v, int *w,
		int *Elt, void *data, int verbose_level);
	void *compute_image_of_vector_callback_data;

	int position_of_original_subspace;
	int allocation_length;
	int old_length;
	int used_length;
	int **Subspaces;
	long int **Subspaces_lint;
	int *prev;
	int *label;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// two sets may have the same hash value.
		// map cannot handle that.


	orbit_of_subspaces();
	~orbit_of_subspaces();
	void init(
			actions::action *A,
			actions::action *A2,
			algebra::field_theory::finite_field *F,
		int *subspace, int k, int n, 
		int f_has_desired_pivots, int *desired_pivots, 
		int f_has_rank_functions, void *rank_unrank_data, 
		int (*rank_vector_callback)(
				int *v, int n,
			void *data, int verbose_level), 
		void (*unrank_vector_callback)(
				int rk, int *v,
			int n, void *data, int verbose_level), 
		void (*compute_image_of_vector_callback)(
				int *v,
			int *w, int *Elt, void *data, int verbose_level), 
		void *compute_image_of_vector_callback_data, 
		data_structures_groups::vector_ge *gens,
		int verbose_level);
	void init_lint(
			actions::action *A,
			actions::action *A2,
			algebra::field_theory::finite_field *F,
		long int *subspace_by_rank, int k, int n,
		int f_has_desired_pivots, int *desired_pivots,
		int f_has_rank_functions, void *rank_unrank_data,
		long int (*rank_vector_lint_callback)(
				int *v, int n,
				void *data, int verbose_level),
		void (*unrank_vector_lint_callback)(
				long int rk, int *v, int n,
				void *data, int verbose_level),
		void (*compute_image_of_vector_callback)(
				int *v, int *w,
				int *Elt, void *data, int verbose_level),
		void *compute_image_of_vector_callback_data,
		data_structures_groups::vector_ge *gens,
		int verbose_level);
	int rank_vector(
			int *v, int verbose_level);
	long int rank_vector_lint(
			int *v, int verbose_level);
	void unrank_vector(
			int rk, int *v, int verbose_level);
	void unrank_vector_lint(
			long int rk, int *v, int verbose_level);
	void unrank_subspace(
			int subspace_idx,
			int *subspace_basis, int verbose_level);
	void rank_subspace(
			int *subspace_basis, int verbose_level);
	uint32_t hash_subspace();
	void unrank(
			int *rk, int *subspace_basis, int verbose_level);
	void unrank_lint(
			long int *rk, int *subspace_basis, int verbose_level);
	void rank(
			int *rk, int *subspace_basis, int verbose_level);
	void rank_lint(
			long int *rk, int *subspace_basis, int verbose_level);
	void rref(
			int *subspace, int verbose_level);
	void rref_and_rank(
			int *subspace, int *rk, int verbose_level);
	void rref_and_rank_lint(
			int *subspace, long int *rk, int verbose_level);
	void map_a_subspace(
			int *basis, int *image_basis, int *Elt,
		int verbose_level);
	void print_orbit();
	int rank_hash_and_find(
			int *subspace,
			int &idx, uint32_t &h, int verbose_level);
	void compute(
			int verbose_level);
	void get_transporter(
			int idx, int *transporter, int verbose_level);
		// transporter is an element which maps the orbit 
		// representative to the given subspace.
	int find_subspace(
			int *subspace_ranks,
			int &idx, int verbose_level);
	int find_subspace_lint(
			long int *subspace_ranks,
			int &idx, int verbose_level);
	void get_random_schreier_generator(
			int *Elt, int verbose_level);
	groups::strong_generators *stabilizer_orbit_rep(
			algebra::ring_theory::longinteger_object &full_group_order,
			int verbose_level);
	void compute_stabilizer(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &go,
			groups::sims *&Stab, int verbose_level);
		// this function allocates a sims structure into Stab.
};


}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_ORBITS_ORBITS_H_ */




