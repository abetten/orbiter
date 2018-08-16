// orbits.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09

// #############################################################################
// orbit_of_equations.C
// #############################################################################

class orbit_of_equations {
public:
	action *A;
	action_on_homogeneous_polynomials *AonHPD;
	finite_field *F;
	strong_generators *SG;
	INT nb_monomials;
	INT sz; // = 1 + nb_monomials
	INT sz_for_compare; // = 1 + nb_monomials
	INT *data_tmp; // [sz]

	INT position_of_original_object;
	INT allocation_length;
	INT used_length;
	INT **Equations;
	INT *prev;
	INT *label;


	orbit_of_equations();
	~orbit_of_equations();
	void null();
	void freeself();
	void init(action *A, finite_field *F, 
		action_on_homogeneous_polynomials *AonHPD, 
		strong_generators *SG, INT *coeff_in, 
		INT verbose_level);
	void map_an_equation(INT *object_in, INT *object_out, 
		INT *Elt, INT verbose_level);
	void print_orbit();
	void compute_orbit(INT *coeff, INT verbose_level);
	void get_transporter(INT idx, INT *transporter, INT verbose_level);
		// transporter is an element which maps 
		// the orbit representative to the given subspace.
	void get_random_schreier_generator(INT *Elt, INT verbose_level);
	void compute_stabilizer(action *default_action, 
		longinteger_object &go, 
		sims *&Stab, INT verbose_level);
		// this function allocates a sims structure into Stab.
	strong_generators *generators_for_stabilizer_of_orbit_rep(
		longinteger_object &full_group_order, INT verbose_level);
	INT search_data(INT *data, INT &idx);
	void save_csv(const BYTE *fname, INT verbose_level);
};

INT orbit_of_equations_compare_func(void *a, void *b, void *data);

// #############################################################################
// orbit_of_sets.C
// #############################################################################

class orbit_of_sets {
public:
	action *A;
	action *A2;
	vector_ge *gens;
	INT *set;
	INT sz;

	INT position_of_original_set;
	INT allocation_length;
	INT used_length;
	INT **Sets;

	orbit_of_sets();
	~orbit_of_sets();
	void null();
	void freeself();
	void init(action *A, action *A2, INT *set, INT sz, 
		vector_ge *gens, INT verbose_level);
	void compute(INT verbose_level);
	void get_table_of_orbits(INT *&Table, INT &orbit_length, 
		INT &set_size, INT verbose_level);
};

INT orbit_of_sets_compare_func(void *a, void *b, void *data);

// #############################################################################
// orbit_of_subspaces.C
// #############################################################################

class orbit_of_subspaces {
public:
	action *A;
	action *A2;
	finite_field *F;
	vector_ge *gens;
	INT k;
	INT n;
	INT kn;
	INT sz; // = 1 + k + kn
	INT sz_for_compare; // = 1 + k + kn
	INT f_has_desired_pivots;
	INT *desired_pivots; // [k]
	INT *subspace_by_rank; // [k]
	INT *data_tmp; // [sz]

	INT f_has_rank_functions;
	void *rank_unrank_data;
	INT (*rank_vector_callback)(INT *v, INT n, 
		void *data, INT verbose_level);
	void (*unrank_vector_callback)(INT rk, INT *v, 
		INT n, void *data, INT verbose_level);
	void (*compute_image_of_vector_callback)(INT *v, INT *w, 
		INT *Elt, void *data, INT verbose_level);
	void *compute_image_of_vector_callback_data;

	INT position_of_original_subspace;
	INT allocation_length;
	INT used_length;
	INT **Subspaces;
	INT *prev;
	INT *label;


	orbit_of_subspaces();
	~orbit_of_subspaces();
	void null();
	void freeself();
	void init(action *A, action *A2, finite_field *F, 
		INT *subspace, INT k, INT n, 
		INT f_has_desired_pivots, INT *desired_pivots, 
		INT f_has_rank_functions, void *rank_unrank_data, 
		INT (*rank_vector_callback)(INT *v, INT n, 
			void *data, INT verbose_level), 
		void (*unrank_vector_callback)(INT rk, INT *v, 
			INT n, void *data, INT verbose_level), 
		void (*compute_image_of_vector_callback)(INT *v, 
			INT *w, INT *Elt, void *data, INT verbose_level), 
		void *compute_image_of_vector_callback_data, 
		vector_ge *gens, INT verbose_level);
	INT rank_vector(INT *v, INT verbose_level);
	void unrank_vector(INT rk, INT *v, INT verbose_level);
	void rref(INT *subspace, INT verbose_level);
	void rref_and_rank_and_hash(INT *subspace, INT verbose_level);
	void map_a_subspace(INT *subspace, INT *image_subspace, 
		INT *Elt, INT verbose_level);
	void map_a_basis(INT *basis, INT *image_basis, INT *Elt, 
		INT verbose_level);
	void print_orbit();
	void compute(INT verbose_level);
	void get_transporter(INT idx, INT *transporter, INT verbose_level);
		// transporter is an element which maps the orbit 
		// representative to the given subspace.
	void get_random_schreier_generator(INT *Elt, INT verbose_level);
	strong_generators *generators_for_stabilizer_of_orbit_rep(
		longinteger_object &full_group_order, INT verbose_level);
	void compute_stabilizer(action *default_action, longinteger_object &go, 
		sims *&Stab, INT verbose_level);
		// this function allocates a sims structure into Stab.
	INT search_data(INT *data, INT &idx);
	INT search_data_raw(INT *data_raw, INT &idx, INT verbose_level);
};

INT orbit_of_subspaces_compare_func(void *a, void *b, void *data);

// #############################################################################
// orbit_rep.C
// #############################################################################

class orbit_rep {
public:
	BYTE prefix[1000];
	action *A;
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level);
	void *early_test_func_callback_data;
	
	INT level;
	INT orbit_at_level;
	INT nb_cases;

	INT *rep;

	sims *Stab;
	strong_generators *Strong_gens;

	longinteger_object *stab_go;
	INT *candidates;
	INT nb_candidates;


	orbit_rep();
	~orbit_rep();
	void null();
	void freeself();
	void init_from_file(action *A, BYTE *prefix, 
		INT level, INT orbit_at_level, INT level_of_candidates_file, 
		void (*early_test_func_callback)(INT *S, INT len, 
			INT *candidates, INT nb_candidates, 
			INT *good_candidates, INT &nb_good_candidates, 
			void *data, INT verbose_level), 
		void *early_test_func_callback_data, 
		INT verbose_level);
	
};

// #############################################################################
// representatives.C
// #############################################################################

class representatives {
public:
	action *A;

	BYTE prefix[1000];
	BYTE fname_rep[1000];
	BYTE fname_stabgens[1000];
	BYTE fname_fusion[1000];
	BYTE fname_fusion_ge[1000];



	// flag orbits:
	INT nb_objects;
	INT *fusion; // [nb_objects]
		// fusion[i] == -2 means that the flag orbit i 
		// has not yet been processed by the
		// isomorphism testing procedure.
		// fusion[i] = i means that flag orbit [i] 
		// in an orbit representative
		// Otherwise, fusion[i] is an earlier flag_orbit, 
		// and handle[i] is a group element that maps 
		// to it
	INT *handle; // [nb_objects]
		// handle[i] is only relevant if fusion[i] != i,
		// i.e., if flag orbit i is not a representative
		// of an isomorphism type.
		// In this case, handle[i] is the (handle of a) 
		// group element moving flag orbit i to flag orbit fusion[i]. 


	// classified objects:
	INT count;
	INT *rep; // [count] 
	sims **stab; // [count] 



	//BYTE *elt;
	INT *Elt1;
	INT *tl; // [A->base_len]

	INT nb_open;
	INT nb_reps;
	INT nb_fused;


	representatives();
	void null();
	~representatives();
	void free();
	void init(action *A, INT nb_objects, BYTE *prefix, INT verbose_level);
	void write_fusion(INT verbose_level);
	void read_fusion(INT verbose_level);
	void write_representatives_and_stabilizers(INT verbose_level);
	void read_representatives_and_stabilizers(INT verbose_level);
	void save(INT verbose_level);
	void load(INT verbose_level);
	void calc_fusion_statistics();
	void print_fusion_statistics();
};

// #############################################################################
// subspace_orbits.C
// #############################################################################


class subspace_orbits {
public:

	linear_group *LG;
	INT n;
	finite_field *F;
	INT q;
	INT depth;

	INT f_print_generators;
	

	INT *tmp_M; // [n * n]
	INT *tmp_M2; // [n * n]
	INT *tmp_M3; // [n * n]
	INT *base_cols; // [n]
	INT *v; // [n]
	INT *w; // [n]
	INT *weights; // [n + 1]

	generator *Gen;

	INT schreier_depth;
	INT f_use_invariant_subset_if_available;
	INT f_implicit_fusion;
	INT f_debug;

	INT f_has_strong_generators;
	strong_generators *Strong_gens;

	INT f_has_extra_test_func;
	INT (*extra_test_func)(subspace_orbits *SubOrb, 
		INT len, INT *S, void *data, INT verbose_level);
	void *extra_test_func_data;

	INT test_dim;


	subspace_orbits();
	~subspace_orbits();
	void init(int argc, const char **argv, 
		linear_group *LG, INT depth, 
		INT verbose_level);
	void init_group(INT verbose_level);
	void compute_orbits(INT verbose_level);
	void unrank_set_to_M(INT len, INT *S);
	void unrank_set_to_matrix(INT len, INT *S, INT *M);
	void rank_set_from_matrix(INT len, INT *S, INT *M);
	void Kramer_Mesner_matrix(INT t, INT k, INT f_print_matrix, 
		INT f_read_solutions, const BYTE *solution_fname, 
		INT verbose_level);
	void print_all_solutions(diophant *D, INT k, INT *Sol, INT nb_sol, 
		INT **Subspace_ranks, INT &nb_subspaces, INT verbose_level);
	void print_one_solution(diophant *D, INT k, INT *sol, 
		INT *&subspace_ranks, INT &nb_subspaces, INT verbose_level);
	INT test_dim_C_cap_Cperp_property(INT len, INT *S, INT d);
	INT compute_minimum_distance(INT len, INT *S);
	void print_set(INT len, INT *S);
	INT test_set(INT len, INT *S, INT verbose_level);
	INT test_minimum_distance(INT len, INT *S, 
		INT mindist, INT verbose_level);
	INT test_if_self_orthogonal(INT len, INT *S, 
		INT f_doubly_even, INT verbose_level);
};


INT subspace_orbits_rank_point_func(INT *v, void *data);
void subspace_orbits_unrank_point_func(INT *v, INT rk, void *data);
void subspace_orbits_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);



