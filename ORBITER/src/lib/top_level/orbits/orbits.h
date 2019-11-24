// orbits.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09



namespace orbiter {
namespace top_level {



// #############################################################################
// kramer_mesner.cpp
// #############################################################################


//! poset classification and orbital matrices

class kramer_mesner {

public:

	int n;
	int q, p, h;



	char *override_poly;

	finite_field *F;


	int f_linear;
	linear_group_description *Descr;
	linear_group *LG;

	action *A;
	action *A2;
	int f_A_is_allocated;

	action *final_A;



	int f_list;
	int f_arc;
	int f_surface;
	surface_domain *Surf;
	int f_KM;

	int f_orbits_t;
	int orbits_t;
	int f_orbits_k;
	int orbits_k;


	int f_draw_poset;
	int f_embedded;
	int f_sideways;


	int nb_identify;
	char **Identify_label;
	long int **Identify_data;
	int *Identify_length;




	poset *Poset;
	poset_classification *gen;




	kramer_mesner();
	~kramer_mesner();
	void read_arguments(int argc, const char **argv, int &verbose_level);
	void init_group(sims *&S, int verbose_level);
	void orbits(int argc, const char **argv, sims *S, int verbose_level);
};


int kramer_mesner_test_arc(int len, long int *S, void *data, int verbose_level);
int kramer_mesner_test_surface(int len, long int *S, void *data, int verbose_level);


// #############################################################################
// orbit_of_equations.cpp
// #############################################################################


//! orbit of homogeneous equations using a Schreier tree


class orbit_of_equations {
public:
	action *A;
	action_on_homogeneous_polynomials *AonHPD;
	finite_field *F;
	strong_generators *SG;
	int nb_monomials;
	int sz; // = 1 + nb_monomials
	int sz_for_compare; // = 1 + nb_monomials
	int *data_tmp; // [sz]

	int position_of_original_object;
	int allocation_length;
	int used_length;
	int **Equations;
	int *prev;
	int *label;

	int f_has_print_function;
	void (*print_function)(int *object, int sz, void *print_function_data);
	void *print_function_data;

	int f_has_reduction;
	void (*reduction_function)(int *object, void *reduction_function_data);
	void *reduction_function_data;


	orbit_of_equations();
	~orbit_of_equations();
	void null();
	void freeself();
	void init(action *A, finite_field *F, 
		action_on_homogeneous_polynomials *AonHPD, 
		strong_generators *SG, int *coeff_in, 
		int verbose_level);
	void map_an_equation(int *object_in, int *object_out, 
		int *Elt, int verbose_level);
	void print_orbit();
	void compute_orbit(int *coeff, int verbose_level);
	void get_transporter(int idx, int *transporter, int verbose_level);
		// transporter is an element which maps 
		// the orbit representative to the given subspace.
	void get_random_schreier_generator(int *Elt, int verbose_level);
	void compute_stabilizer(action *default_action, 
		longinteger_object &go, 
		sims *&Stab, int verbose_level);
		// this function allocates a sims structure into Stab.
	strong_generators *stabilizer_orbit_rep(
		longinteger_object &full_group_order, int verbose_level);
	strong_generators *stabilizer_any_point(
		longinteger_object &full_group_order, int idx,
		int verbose_level);
	int search_data(int *data, int &idx);
	void save_csv(const char *fname, int verbose_level);
};

int orbit_of_equations_compare_func(void *a, void *b, void *data);

// #############################################################################
// orbit_of_sets.cpp
// #############################################################################




//! orbit of sets using a Schreier tree

// used in packing::make_spread_table

class orbit_of_sets {
public:
	action *A;
	action *A2;
	vector_ge *gens;
	long int *set; // the set whose orbit we want to compute; it has size 'sz'
	int sz;

	int position_of_original_set; // = 0; never changes
	int allocation_length; // number of entries allocated in Sets
	int old_length;
	int used_length; // number of sets currently stored in Sets
	long int **Sets;
		// the sets are stored in the order in which they
		// are discovered and added to the tree
	int *Extra; // [allocation_length * 2]
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
		// it happens that two sets have the same hash value.
		// map cannot handle that.

	orbit_of_sets();
	~orbit_of_sets();
	void null();
	void freeself();
	void init(action *A, action *A2,
			long int *set, int sz,
			vector_ge *gens, int verbose_level);
	void compute(int verbose_level);
	void dump_tables_of_hash_values();
	void get_table_of_orbits(long int *&Table, int &orbit_length,
		int &set_size, int verbose_level);
	void get_table_of_orbits_and_hash_values(long int *&Table,
			int &orbit_length, int &set_size, int verbose_level);
	void coset_rep(int j);
		// result is in cosetrep
		// determines an element in the group
		// that moves the orbit representative
	// to the j-th element in the orbit.
};

int orbit_of_sets_compare_func(void *a, void *b, void *data);

// #############################################################################
// orbit_of_subspaces.cpp
// #############################################################################



//! orbit of subspaces using a Schreier tree



class orbit_of_subspaces {
public:
	action *A;
	action *A2;
	finite_field *F;
	vector_ge *gens;
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
	int (*rank_vector_callback)(int *v, int n, 
		void *data, int verbose_level);
	long int (*rank_vector_lint_callback)(int *v, int n,
		void *data, int verbose_level);
	void (*unrank_vector_callback)(int rk, int *v, 
		int n, void *data, int verbose_level);
	void (*unrank_vector_lint_callback)(long int rk, int *v,
		int n, void *data, int verbose_level);
	void (*compute_image_of_vector_callback)(int *v, int *w, 
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
		// it happens that two sets have the same hash value.
		// map cannot handle that.


	orbit_of_subspaces();
	~orbit_of_subspaces();
	void null();
	void freeself();
	void init(action *A, action *A2, finite_field *F, 
		int *subspace, int k, int n, 
		int f_has_desired_pivots, int *desired_pivots, 
		int f_has_rank_functions, void *rank_unrank_data, 
		int (*rank_vector_callback)(int *v, int n, 
			void *data, int verbose_level), 
		void (*unrank_vector_callback)(int rk, int *v, 
			int n, void *data, int verbose_level), 
		void (*compute_image_of_vector_callback)(int *v, 
			int *w, int *Elt, void *data, int verbose_level), 
		void *compute_image_of_vector_callback_data, 
		vector_ge *gens, int verbose_level);
	void init_lint(
		action *A, action *A2, finite_field *F,
		long int *subspace_by_rank, int k, int n,
		int f_has_desired_pivots, int *desired_pivots,
		int f_has_rank_functions, void *rank_unrank_data,
		long int (*rank_vector_lint_callback)(int *v, int n,
				void *data, int verbose_level),
		void (*unrank_vector_lint_callback)(long int rk, int *v, int n,
				void *data, int verbose_level),
		void (*compute_image_of_vector_callback)(int *v, int *w,
				int *Elt, void *data, int verbose_level),
		void *compute_image_of_vector_callback_data,
		vector_ge *gens, int verbose_level);
	int rank_vector(int *v, int verbose_level);
	long int rank_vector_lint(int *v, int verbose_level);
	void unrank_vector(int rk, int *v, int verbose_level);
	void unrank_vector_lint(long int rk, int *v, int verbose_level);
	void unrank_subspace(
			int subspace_idx, int *subspace_basis, int verbose_level);
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
	void rref(int *subspace, int verbose_level);
	void rref_and_rank(
			int *subspace, int *rk, int verbose_level);
	void rref_and_rank_lint(
			int *subspace, long int *rk, int verbose_level);
	void map_a_subspace(int *basis, int *image_basis, int *Elt,
		int verbose_level);
	void print_orbit();
	int rank_hash_and_find(int *subspace, int &idx, uint32_t &h, int verbose_level);
	void compute(int verbose_level);
	void get_transporter(int idx, int *transporter, int verbose_level);
		// transporter is an element which maps the orbit 
		// representative to the given subspace.
	int find_subspace(
			int *subspace_ranks, int &idx, int verbose_level);
	int find_subspace_lint(
			long int *subspace_ranks, int &idx, int verbose_level);
	void get_random_schreier_generator(int *Elt, int verbose_level);
	strong_generators *stabilizer_orbit_rep(
		longinteger_object &full_group_order, int verbose_level);
	void compute_stabilizer(action *default_action, longinteger_object &go, 
		sims *&Stab, int verbose_level);
		// this function allocates a sims structure into Stab.
};


// #############################################################################
// subspace_orbits.cpp
// #############################################################################

//! poset classification for the orbits of a group acting on the subspace lattice


class subspace_orbits {
public:

	linear_group *LG;
	int n;
	finite_field *F;
	int q;
	int depth;

	int f_print_generators;
	

	int *tmp_M; // [n * n]
	int *tmp_M2; // [n * n]
	int *tmp_M3; // [n * n]
	int *base_cols; // [n]
	int *v; // [n]
	int *w; // [n]
	int *weights; // [n + 1]

	vector_space *VS;
	poset *Poset;
	poset_classification *Gen;

	int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_implicit_fusion;
	int f_debug;

	int f_has_strong_generators;
	strong_generators *Strong_gens;

	int f_has_extra_test_func;
	int (*extra_test_func)(subspace_orbits *SubOrb, 
		int len, long int *S, void *data, int verbose_level);
	void *extra_test_func_data;

	int test_dim;


	subspace_orbits();
	~subspace_orbits();
	void init(int argc, const char **argv, 
		linear_group *LG, int depth, 
		int verbose_level);
	void init_group(int verbose_level);
	void compute_orbits(int verbose_level);
	void unrank_set_to_M(int len, long int *S, int verbose_level);
	void unrank_set_to_matrix(int len, long int *S, int *M, int verbose_level);
	void rank_set_from_matrix(int len, long int *S, int *M, int verbose_level);
	void Kramer_Mesner_matrix(int t, int k, int f_print_matrix, 
		int f_read_solutions, const char *solution_fname, 
		int verbose_level);
	void print_all_solutions(diophant *D, int k, long int *Sol, int nb_sol,
		int **Subspace_ranks, int &nb_subspaces, int verbose_level);
	void print_one_solution(diophant *D, int k, long int *sol,
		int *&subspace_ranks, int &nb_subspaces, int verbose_level);
	int test_dim_C_cap_Cperp_property(int len, long int *S, int d);
	int compute_minimum_distance(int len, long int *S);
	void print_set(std::ostream &ost, int len, long int *S);
	int test_set(int len, long int *S, int verbose_level);
	int test_minimum_distance(int len, long int *S,
		int mindist, int verbose_level);
	int test_if_self_orthogonal(int len, long int *S,
		int f_doubly_even, int verbose_level);
};


long int subspace_orbits_rank_point_func(int *v, void *data);
void subspace_orbits_unrank_point_func(int *v, long int rk, void *data);
void subspace_orbits_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

}}


