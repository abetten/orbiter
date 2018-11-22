// linear_set.h
// 
// Anton Betten
// July 8, 2014
//
// 
//
//

#include <orbiter.h>


typedef class linear_set linear_set;


//! classification of linear sets




class linear_set {
public:
	int s;
	int n;
	int m; // n = s * m
	int q;
	int Q; // Q = q^s
	int depth;
	int f_semilinear;
	int schreier_depth;
	int f_use_invariant_subset_if_available;
	//int f_lex;
	int f_debug;
	int f_has_extra_test_func;
	int (*extra_test_func)(void *, int len, int *S, 
		void *extra_test_func_data, int verbose_level);
	void *extra_test_func_data;
	int *Basis; // [depth * vector_space_dimension]
	int *base_cols;
	
	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	projective_space *P;
	action *Aq;
	action *AQ;
	action *A_PGLQ;
	poset *Poset1;
	poset_classification *Gen;
	int vector_space_dimension; // = n
	strong_generators *Strong_gens;
	desarguesian_spread *D;
	int n1;
	int m1;
	desarguesian_spread *D1;
	int *spread_embedding; // [D->N]

	int f_identify;
	int k;
	int order;
	spread *T;
	


	int secondary_level;
	int secondary_orbit_at_level;
	int secondary_depth;
	int *secondary_candidates;
	int secondary_nb_candidates;
	int secondary_schreier_depth;
	poset *Poset_stab;
	poset_classification *Gen_stab;
	poset *Poset2;
	poset_classification *Gen2;
	int *is_allowed;

	linear_set();
	~linear_set();
	void null();
	void freeself();
	void init(int argc, const char **argv, 
		int s, int n, int q, 
		const char *poly_q, const char *poly_Q, 
		int depth, int f_identify, int verbose_level);
	void do_classify(int verbose_level);
	int test_set(int len, int *S, int verbose_level);
	void compute_intersection_types_at_level(int level, 
		int &nb_nodes, int *&Intersection_dimensions, 
		int verbose_level);
	void calculate_intersections(int depth, int verbose_level);
	void read_data_file(int depth, int verbose_level);
	void print_orbits_at_level(int level);
	void classify_secondary(int argc, const char **argv, 
		int level, int orbit_at_level, 
		strong_generators *strong_gens, 
		int verbose_level);
	void init_secondary(int argc, const char **argv, 
		int *candidates, int nb_candidates, 
		strong_generators *Strong_gens_previous, 
		int verbose_level);
	void do_classify_secondary(int verbose_level);
	int test_set_secondary(int len, int *S, int verbose_level);
	void compute_stabilizer_of_linear_set(int argc, const char **argv, 
		int level, int orbit_at_level, 
		strong_generators *&strong_gens, 
		int verbose_level);
	void init_compute_stabilizer(int argc, const char **argv, 
		int level, int orbit_at_level,  
		int *candidates, int nb_candidates, 
		strong_generators *Strong_gens_previous, 
		strong_generators *&strong_gens, 
		int verbose_level);
	void do_compute_stabilizer(int level, int orbit_at_level, 
		int *candidates, int nb_candidates, 
		strong_generators *&strong_gens, 
		int verbose_level);

	//linear_set2.C:
	void construct_semifield(int orbit_for_W, int verbose_level);

};


int linear_set_rank_point_func(int *v, void *data);
void linear_set_unrank_point_func(int *v, int rk, void *data);
void linear_set_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void linear_set_secondary_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);


