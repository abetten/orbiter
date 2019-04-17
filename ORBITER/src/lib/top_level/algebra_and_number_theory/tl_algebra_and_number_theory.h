// tl_algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


namespace orbiter {
namespace top_level {


// #############################################################################
// analyze_group.C:
// #############################################################################

void analyze_group(action *A, sims *S, vector_ge *SG, 
	vector_ge *gens2, int verbose_level);
void compute_regular_representation(action *A, sims *S, 
	vector_ge *SG, int *&perm, int verbose_level);
void presentation(action *A, sims *S, int goi, vector_ge *gens, 
	int *primes, int verbose_level);


// #############################################################################
// factor_group.C:
// #############################################################################


//! auxiliary class for create_factor_group, which is used in analyze_group.cpp

struct factor_group {
	int goi;
	action *A;
	sims *S;
	int size_subgroup;
	int *subgroup;
	int *all_cosets;
	int nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	int goi_factor_group;
};

void create_factor_group(action *A, sims *S, int goi, 
	int size_subgroup, int *subgroup, factor_group *F, int verbose_level);



// #############################################################################
// semifield_classify.cpp
// #############################################################################



class semifield_classify {
public:

	int k;
	int k2; // = k * k
	finite_field *F;
	int q;
	int order; // q^k

	int f_level_two_prefix;
	const char *level_two_prefix;

	int f_level_three_prefix;
	const char *level_three_prefix;


	spread *T;

	action *A; // = T->A = PGL_n_q
	int *Elt1;
	sims *G; // = T->R->A0_linear->Sims

	action *A0;
	action *A0_linear;


	action_on_spread_set *A_on_S;
	action *AS;

	strong_generators *Strong_gens;
		// the stabilizer of two components in a spread:
		// infinity and zero


	poset *Poset;
	poset_classification *Gen;
	sims *Symmetry_group;

	//semifield_starter *SFS;


	int vector_space_dimension; // = k * k
	int schreier_depth;

	semifield_classify();
	~semifield_classify();
	void null();
	void freeself();
	void init(int argc, const char **argv,
		int order, int n, int k,
		finite_field *F,
		const char *prefix,
		int verbose_level);
	void compute_orbits(int depth, int verbose_level);
	void list_points();
	int rank_point(int *v, int verbose_level);
	void unrank_point(int *v, int rk, int verbose_level);
	void matrix_unrank(int rk, int *Mtx);
	int matrix_rank(int *Mtx);
	void early_test_func(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};

void semifield_classify_early_test_func(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int semifield_classify_rank_point_func(int *v, void *data);
void semifield_classify_unrank_point_func(int *v, int rk, void *data);


// #############################################################################
// young.C
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	int n;
	action *A;
	sims *S;
	longinteger_object go;
	int goi;
	int *Elt;
	int *v;

	action *Aconj;
	action_by_conjugation *ABC;
	schreier *Sch;
	strong_generators *SG;
	int nb_classes;
	int *class_size;
	int *class_rep;
	a_domain *D;

	int l1, l2;
	int *row_parts;
	int *col_parts;
	int *Tableau;

	set_of_sets *Row_partition;
	set_of_sets *Col_partition;

	vector_ge *gens1, *gens2;
	sims *S1, *S2;


	young();
	~young();
	void null();
	void freeself();
	void init(int n, int verbose_level);
	void create_module(int *h_alpha, 
		int *&Base, int *&base_cols, int &rk, 
		int verbose_level);
	void create_representations(int *Base, int *Base_inv, int rk, 
		int verbose_level);
	void create_representation(int *Base, int *base_cols, int rk, 
		int group_elt, int *Mtx, int verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_int]
	void young_symmetrizer(int *row_parts, int nb_row_parts, 
		int *tableau, 
		int *elt1, int *elt2, int *elt3, 
		int verbose_level);
	void compute_generators(int &go1, int &go2, int verbose_level);
	void Maschke(int *Rep, 
		int dim_of_module, int dim_of_submodule, 
		int *&Mu, 
		int verbose_level);
};


}}

