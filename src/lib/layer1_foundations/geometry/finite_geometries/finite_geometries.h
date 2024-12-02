/*
 * finite_geometries.h
 *
 *  Created on: Nov 30, 2024
 *      Author: betten
 */

// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



#ifndef SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_FINITE_GEOMETRIES_FINITE_GEOMETRIES_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_FINITE_GEOMETRIES_FINITE_GEOMETRIES_H_

namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace finite_geometries {



// #############################################################################
// andre_construction.cpp
// #############################################################################

//! Andre / Bruck / Bose construction of a translation plane from a spread


class andre_construction {
public:
	int order; // = q^k
	int spread_size; // order + 1
	int n; // = 2 * k
	int k;
	int q;
	int N; // order^2 + order + 1


	projective_geometry::grassmann *Grass;
	algebra::field_theory::finite_field *F;

	long int *spread_elements_numeric; // [spread_size]
	long int *spread_elements_numeric_sorted; // [spread_size]

	long int *spread_elements_perm;
	long int *spread_elements_perm_inv;

	int *spread_elements_genma; // [spread_size * k * n]
	int *pivot; //[spread_size * k]
	int *non_pivot; //[spread_size * (n - k)]


	andre_construction();
	~andre_construction();
	void init(
			algebra::field_theory::finite_field *F,
			int k, long int *spread_elements_numeric,
		int verbose_level);
	void points_on_line(
			andre_construction_line_element *Line,
		int *pts_on_line, int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);

};




// #############################################################################
// andre_construction_point_element.cpp
// #############################################################################


//! a point in the projective plane created using the Andre construction


class andre_construction_point_element {
public:
	andre_construction *Andre;
	int k, n, q, spread_size;
	algebra::field_theory::finite_field *F;
	int point_rank;
	int f_is_at_infinity;
	int at_infinity_idx;
	int affine_numeric;
	int *coordinates; // [n]

	andre_construction_point_element();
	~andre_construction_point_element();
	void init(
			andre_construction *Andre, int verbose_level);
	void unrank(
			int point_rank, int verbose_level);
	int rank(
			int verbose_level);
};


// #############################################################################
// andre_construction_line_element.cpp
// #############################################################################


//! a line in the projective plane created using the Andre construction


class andre_construction_line_element {
public:
	andre_construction *Andre;
	int k, n, q, spread_size;
	algebra::field_theory::finite_field *F;
	int line_rank;
	int f_is_at_infinity;
	int affine_numeric;
	int parallel_class_idx;
	int coset_idx;
	int *pivots; // [k]
	int *non_pivots; // [n - k]
	int *coset; // [n - k]
	int *coordinates; // [(k + 1) * n], last row is special vector

	andre_construction_line_element();
	~andre_construction_line_element();
	void init(
			andre_construction *Andre, int verbose_level);
	void unrank(
			int line_rank, int verbose_level);
	int rank(
			int verbose_level);
	int make_affine_point(
			int idx, int verbose_level);
		// 0 \le idx \le order
};


// #############################################################################
// buekenhout_metz.cpp
// #############################################################################

//! Buekenhout-Metz unitals


class buekenhout_metz {
public:
	algebra::field_theory::finite_field *FQ, *Fq;
	int q;
	int Q;

	algebra::field_theory::subfield_structure *SubS;

	int f_classical;
	int f_Uab;
	int parameter_a;
	int parameter_b;

	projective_geometry::projective_space *P2; // PG(2,q^2), where the unital lives
	projective_geometry::projective_space *P3; // PG(3,q), where the ovoid lives

	int *v; // [3]
	int *w1; // [6]
	int *w2; // [6]
	int *w3; // [6]
	int *w4; // [6]
	int *w5; // [6]


	long int *ovoid;
	long int *U;
	int sz;
	int alpha, t0, t1, T0, T1;
	long int theta_3;
	int minus_t0, sz_ovoid;
	int e1, one_1, one_2;


	// compute_the_design:
	long int *secant_lines;
	int nb_secant_lines;
	long int *tangent_lines;
	int nb_tangent_lines;
	long int *Intersection_sets;
	int *Design_blocks;
	long int *block;
	int block_size;
	int *idx_in_unital;
	int *idx_in_secants;
	int *tangent_line_at_point;
	int *point_of_tangency;
	int *f_is_tangent_line;
	int *f_is_Baer;


	// the block that we choose:
	int nb_good_points;
	int *good_points; // = q + 1


	buekenhout_metz();
	~buekenhout_metz();
	void buekenhout_metz_init(
			algebra::field_theory::finite_field *Fq,
			algebra::field_theory::finite_field *FQ,
		int f_Uab, int a, int b,
		int f_classical, int verbose_level);
	void init_ovoid(
			int verbose_level);
	void init_ovoid_Uab_even(
			int a, int b, int verbose_level);
	void create_unital(
			int verbose_level);
	void create_unital_tex(
			int verbose_level);
	void create_unital_Uab_tex(
			int verbose_level);
	void compute_the_design(
			int verbose_level);
	void write_unital_to_file();
	void get_name(
			std::string &name);

};


// #############################################################################
// desarguesian_spread.cpp
// #############################################################################


//! desarguesian spread



class desarguesian_spread {
public:
	int n;
	int m;
	int s;
	int q;
	int Q;
	algebra::field_theory::finite_field *Fq;
	algebra::field_theory::finite_field *FQ;
	algebra::field_theory::subfield_structure *SubS;
	projective_geometry::grassmann *Gr;

	int N;
		// = number of points in PG(m - 1, Q)

	int nb_points;
		// = number of points in PG(n - 1, q)

	int nb_points_per_spread_element;
		// = number of points in PG(s - 1, q)

	int spread_element_size;
		// = s * n

	int *Spread_elements;
		// [N * spread_element_size]

	long int *Rk;
		// [N]

	int *List_of_points;
		// [N * nb_points_per_spread_element]

	desarguesian_spread();
	~desarguesian_spread();
	void init(
			int n, int m, int s,
			algebra::field_theory::subfield_structure *SubS,
		int verbose_level);
	void calculate_spread_elements(
			int verbose_level);
	void compute_intersection_type(
			int k, int *subspace,
		int *intersection_dimensions, int verbose_level);
	// intersection_dimensions[h]
	void compute_shadow(
			int *Basis, int basis_sz,
		int *is_in_shadow, int verbose_level);
	void compute_linear_set(
			int *Basis, int basis_sz,
		long int *&the_linear_set, int &the_linear_set_sz,
		int verbose_level);
	void print_spread_element_table_tex(
			std::ostream &ost);
	void print_spread_elements_tex(
			std::ostream &ost);
	void print_linear_set_tex(
			long int *set, int sz);
	void print_linear_set_element_tex(
			long int a, int sz);
	void create_latex_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);

};



// #############################################################################
// knarr.cpp
// #############################################################################

//! the Knarr construction of a GQ from a BLT-set
/*! The Knarr construction of a GQ(q^2,q) from a BLT set of lines in W(3,q):
 *
 * let P = (1,0,0,0,0,0) in W(5,q)
 *
 * Let B be a BLT-set of lines in W(3,q),
 * lifted into P^\perp in W(5,q)
 *
 * type i) points:
 * the q^5 points in W(5,q) \setminus P^\perp
 * type ii) points:
 * lines in the BLT-planes, not containing the point P
 * there are (q+1)*q^2 of them (q^2 for each BLT-plane)
 * type iii) points:
 * The unique point P=(1,0,0,0,0,0)
 *
 * For a total of q^5 + q^3 + q^2 + 1 = (q^2 + 1)(q^3 + 1) points
 *
 * type a) lines:
 * t.i. planes \pi, not containing P,
 * with \pi \cap P^\perp a line of a BLT-plane (such that the line avoids P),
 * i.e. a point of type ii).
 * There are (q+1)*q^3 such planes
 *
 * type b) lines:
 * the q+1 elements of the BLT set,
 * lifted to become t.i. planes containing P in W(5,q)
 *
 * For a total of
 * q^4 + q^3 + q + 1 = (q + 1)*(q^3 + 1) lines
 *
 * This is the required number for a GQ(q^2,q).
 * Recall that a GQ(s,t) has
 * (s+1)(st+1) points and
 * (t+1)(st+1) lines.
 */



class knarr {
public:
	int q;
	int BLT_no;

	W3q *W;
	projective_geometry::projective_space *P5;
	projective_geometry::grassmann *G63;
	algebra::field_theory::finite_field *F;
	long int *BLT;
	int *BLT_line_idx;
	int *Basis;
	int *Basis2;
	int *subspace_basis;
	int *Basis_Pperp;
	algebra::ring_theory::longinteger_object *six_choose_three_q;
	int six_choose_three_q_int;
	int f_show;
	int dim_intersection;
	int *Basis_intersection;
	other::data_structures::fancy_set *type_i_points, *type_ii_points, *type_iii_points;
	other::data_structures::fancy_set *type_a_lines, *type_b_lines;
	int *type_a_line_BLT_idx;
	int q2;
	int q5;
	int v5[5];
	int v6[6];

	knarr();
	~knarr();
	void init(
			algebra::field_theory::finite_field *F,
			int BLT_no, int verbose_level);
	void points_and_lines(
			int verbose_level);
	void incidence_matrix(
			int *&Inc, int &nb_points,
		int &nb_lines,
		int verbose_level);

};



// #############################################################################
// spread_domain.cpp
// #############################################################################

#define SPREAD_OF_TYPE_FTWKB 1
#define SPREAD_OF_TYPE_KANTOR 2
#define SPREAD_OF_TYPE_KANTOR2 3
#define SPREAD_OF_TYPE_GANLEY 4
#define SPREAD_OF_TYPE_LAW_PENTTILA 5
#define SPREAD_OF_TYPE_DICKSON_KANTOR 6
#define SPREAD_OF_TYPE_HUDSON 7

//! spreads of PG(k-1,q) in PG(n-1,q) where k divides n


class spread_domain {

public:

	algebra::field_theory::finite_field *F;

	int n; // = a multiple of k
	int k;
	int kn; // = k * n
	int q;

	long int nCkq; // = {n choose k}_q
		// used in print_elements, print_elements_and_points
	long int nC1q; // = {n choose 1}_q
	long int kC1q; // = {k choose 1}_q

	long int qn; // q^n
	long int qk; // q^k

	int order; // q^k
	int spread_size; // = order + 1

	long int r;
	long int nb_pts;
	long int nb_points_total; // = nb_pts = {n choose 1}_q
	//long int block_size;
	// = r = {k choose 1}_q, used in spread_lifting.spp

	projective_geometry::grassmann *Grass;
		// {n choose k}_q

	// for check_function and check_function_incremental:
	int *tmp_M1;
	int *tmp_M2;
	int *tmp_M3;
	int *tmp_M4;

	// only if n = 2 * k:
	projective_geometry::klein_correspondence *Klein;
	layer1_foundations::geometry::orthogonal_geometry::orthogonal *O;


	int *Data1;
		// for early_test_func
		// [max_depth * kn],
		// previously [Nb * n], which was too much
	int *Data2;
		// for early_test_func
		// [n * n]

	spread_domain();
	~spread_domain();
	void init_spread_domain(
			algebra::field_theory::finite_field *F,
			int n, int k,
			int verbose_level);
	void unrank_point(
			int *v, long int a);
	long int rank_point(
			int *v);
	void unrank_subspace(
			int *M, long int a);
	long int rank_subspace(
			int *M);
	void print_points();
	void print_points(
			long int *pts, int len);
	void print_elements();
	void print_elements_and_points();
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_function(
			int len, long int *S, int verbose_level);
	int incremental_check_function(
			int len, long int *S, int verbose_level);
	void compute_dual_spread(
			int *spread, int *dual_spread,
		int verbose_level);
	void print(
			std::ostream &ost, int len, long int *S);
	void czerwinski_oakden(
			int level, int verbose_level);
	void write_spread_to_file(
			int type_of_spread, int verbose_level);
	void make_spread(
			long int *data, int type_of_spread,
			int verbose_level);
	void make_spread_from_q_clan(
			long int *data, int type_of_spread,
		int verbose_level);
	void read_and_print_spread(
			std::string &fname, int verbose_level);
	void HMO(
			std::string &fname, int verbose_level);
	void print_spread(
			std::ostream &ost, long int *data, int sz);

};


// #############################################################################
// spread_tables.cpp
// #############################################################################

//! tables with line-spreads in PG(3,q)


class spread_tables {

public:
	int q;
	int d; // = 4
	algebra::field_theory::finite_field *F;
	projective_geometry::projective_space *P; // PG(3,q)
	projective_geometry::grassmann *Gr; // Gr_{4,2}
	long int nb_lines;
	int spread_size;
	int nb_iso_types_of_spreads;

	std::string prefix;

	std::string fname_dual_line_idx;
	std::string fname_self_dual_lines;
	std::string fname_spreads;
	std::string fname_isomorphism_type_of_spreads;
	std::string fname_dual_spread;
	std::string fname_self_dual_spreads;
	std::string fname_schreier_table;

	int *dual_line_idx; // [nb_lines]
	int *self_dual_lines; // [nb_self_dual_lines]
	int nb_self_dual_lines;

	int nb_spreads;
	long int *spread_table; // [nb_spreads * spread_size]
	int *spread_iso_type; // [nb_spreads]
	long int *dual_spread_idx; // [nb_spreads]
	long int *self_dual_spreads; // [nb_self_dual_spreads]
	int nb_self_dual_spreads;

	int *schreier_table; // [nb_spreads * 4]

	spread_tables();
	~spread_tables();
	void init(
			projective_geometry::projective_space *P,
			int f_load,
			int nb_iso_types_of_spreads,
			std::string &path_to_spread_tables,
			int verbose_level);
	void create_file_names(
			int verbose_level);
	void init_spread_table(
			int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			int verbose_level);
	void init_tables(
			int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			long int *dual_spread_idx,
			long int *self_dual_spreads, int nb_self_dual_spreads,
			int verbose_level);
	void init_schreier_table(
			int *schreier_table,
			int verbose_level);
	void init_reduced(
			int nb_select, int *select,
			spread_tables *old_spread_table,
			std::string &path_to_spread_tables,
			int verbose_level);
	long int *get_spread(
			int spread_idx);
	void find_spreads_containing_two_lines(
			std::vector<int> &v,
			int line1, int line2, int verbose_level);

	void classify_self_dual_spreads(
			int *&type,
			other::data_structures::set_of_sets *&SoS,
			int verbose_level);
	int files_exist(
			int verbose_level);
	void save(
			int verbose_level);
	void load(
			int verbose_level);
	void compute_adjacency_matrix(
			other::data_structures::bitvector *&Bitvec,
			int verbose_level);
	int test_if_spreads_are_disjoint(
			int a, int b);
	void compute_dual_spreads(
			long int **Sets,
			long int *&Dual_spread_idx,
			long int *&self_dual_spread_idx,
			int &nb_self_dual_spreads,
			int verbose_level);
	int test_if_pair_of_sets_are_adjacent(
			long int *set1, int sz1,
			long int *set2, int sz2,
			int verbose_level);
	int test_if_set_of_spreads_is_line_disjoint(
			long int *set, int len);
	int test_if_set_of_spreads_is_line_disjoint_and_complain_if_not(
			long int *set, int len);
	void make_exact_cover_problem(
			combinatorics::solvers::diophant *&Dio,
			long int *live_point_index, int nb_live_points,
			long int *live_blocks, int nb_live_blocks,
			int nb_needed,
			int verbose_level);
	void compute_list_of_lines_from_packing(
			long int *list_of_lines,
			long int *packing, int sz_of_packing,
			int verbose_level);
	// list_of_lines[sz_of_packing * spread_size]
	void compute_iso_type_invariant(
			int *Partial_packings, int nb_pp, int sz,
			int *&Iso_type_invariant,
			int verbose_level);
	void report_one_spread(
			std::ostream &ost, int a);
	void make_graph_of_disjoint_spreads(
			combinatorics::graph_theory::colored_graph *&CG,
			int verbose_level);

};





// #############################################################################
// W3q.cpp
// #############################################################################

//! isomorphism between the W(3,q) and the Q(4,q) generalized quadrangles


class W3q {
public:
	int q;

	projective_geometry::projective_space *P3;
	orthogonal_geometry::orthogonal *Q4;
	algebra::field_theory::finite_field *F;

	int nb_lines;
		// number of absolute lines of W(3,q)
		// = number of points on Q(4,q)

	int *Lines; // [nb_lines]
		// Lines[] is a list of all absolute lines of PG(3,q)
		// under the chosen symplectic form.
		// The symplectic form is defined
		// in the function evaluate_symplectic_form(),
		// which relies on
		// F->Linear_algebra->evaluate_symplectic_form.
		// The form consists of 2x2 blocks
		// of the form (0,1,-1,0)
		// along the diagonal

	int *Q4_rk; // [nb_lines]
	int *Line_idx; // [nb_lines]
		// Q4_rk[] and Line_idx[] are inverse permutations
		// for a line a, Q4_rk[a] is the point b
		// on the quadric corresponding to it.
		// For a point b on the quadric,
		// Line_idx[b] is the index b of the corresponding line


	W3q();
	~W3q();
	void init(
			algebra::field_theory::finite_field *F, int verbose_level);
	void find_lines(
			int verbose_level);
	void print_lines();
	int evaluate_symplectic_form(
			int *x4, int *y4);
	void isomorphism_Q4q(
			int *x4, int *y4, int *v);
	void print_by_lines();
	void print_by_points();
	int find_line(
			int line);
};



}}}}



#endif /* SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_FINITE_GEOMETRIES_FINITE_GEOMETRIES_H_ */
