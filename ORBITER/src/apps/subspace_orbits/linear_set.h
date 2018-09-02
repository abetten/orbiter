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
	INT s;
	INT n;
	INT m; // n = s * m
	INT q;
	INT Q; // Q = q^s
	INT depth;
	INT f_semilinear;
	INT schreier_depth;
	INT f_use_invariant_subset_if_available;
	//INT f_lex;
	INT f_debug;
	INT f_has_extra_test_func;
	INT (*extra_test_func)(void *, INT len, INT *S, 
		void *extra_test_func_data, INT verbose_level);
	void *extra_test_func_data;
	INT *Basis; // [depth * vector_space_dimension]
	INT *base_cols;
	
	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	projective_space *P;
	action *Aq;
	action *AQ;
	action *A_PGLQ;
	poset_classification *Gen;
	INT vector_space_dimension; // = n
	strong_generators *Strong_gens;
	desarguesian_spread *D;
	INT n1;
	INT m1;
	desarguesian_spread *D1;
	INT *spread_embedding; // [D->N]

	INT f_identify;
	INT k;
	INT order;
	spread *T;
	


	INT secondary_level;
	INT secondary_orbit_at_level;
	INT secondary_depth;
	INT *secondary_candidates;
	INT secondary_nb_candidates;
	INT secondary_schreier_depth;
	poset_classification *Gen_stab;
	poset_classification *Gen2;
	INT *is_allowed;

	linear_set();
	~linear_set();
	void null();
	void freeself();
	void init(int argc, const char **argv, 
		INT s, INT n, INT q, 
		const char *poly_q, const char *poly_Q, 
		INT depth, INT f_identify, INT verbose_level);
	void do_classify(INT verbose_level);
	INT test_set(INT len, INT *S, INT verbose_level);
	void compute_intersection_types_at_level(INT level, 
		INT &nb_nodes, INT *&Intersection_dimensions, 
		INT verbose_level);
	void calculate_intersections(INT depth, INT verbose_level);
	void read_data_file(INT depth, INT verbose_level);
	void print_orbits_at_level(INT level);
	void classify_secondary(int argc, const char **argv, 
		INT level, INT orbit_at_level, 
		strong_generators *strong_gens, 
		INT verbose_level);
	void init_secondary(int argc, const char **argv, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens_previous, 
		INT verbose_level);
	void do_classify_secondary(INT verbose_level);
	INT test_set_secondary(INT len, INT *S, INT verbose_level);
	void compute_stabilizer_of_linear_set(int argc, const char **argv, 
		INT level, INT orbit_at_level, 
		strong_generators *&strong_gens, 
		INT verbose_level);
	void init_compute_stabilizer(int argc, const char **argv, 
		INT level, INT orbit_at_level,  
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens_previous, 
		strong_generators *&strong_gens, 
		INT verbose_level);
	void do_compute_stabilizer(INT level, INT orbit_at_level, 
		INT *candidates, INT nb_candidates, 
		strong_generators *&strong_gens, 
		INT verbose_level);

	//linear_set2.C:
	void construct_semifield(INT orbit_for_W, INT verbose_level);

};


INT linear_set_rank_point_func(INT *v, void *data);
void linear_set_unrank_point_func(INT *v, INT rk, void *data);
void linear_set_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void linear_set_secondary_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


