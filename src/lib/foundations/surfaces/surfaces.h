/*
 * surfaces.h
 *
 *  Created on: Jul 29, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_SURFACES_SURFACES_H_
#define SRC_LIB_FOUNDATIONS_SURFACES_SURFACES_H_

namespace orbiter {
namespace foundations {



// #############################################################################
// arc_lifting_with_two_lines.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting_with_two_lines {

public:

	int q;
	finite_field *F; // do not free

	surface_domain *Surf; // do not free

	//surface_with_action *Surf_A;

	long int *Arc6;
	int arc_size; // = 6

	long int line1, line2;

	long int plane_rk;

	int *Arc_coords; // [6 * 4]

	long int P[6];

	long int transversal_01;
	long int transversal_23;
	long int transversal_45;

	long int transversal[4];

	long int input_Lines[9];

	int coeff[20];
	long int lines27[27];

	arc_lifting_with_two_lines();
	~arc_lifting_with_two_lines();
	void null();
	void freeself();
	void create_surface(
		surface_domain *Surf,
		long int *Arc6, long int line1, long int line2,
		int verbose_level);
	// The arc must be given as points in PG(3,q), not in PG(2,q).
};



// #############################################################################
// clebsch_map.cpp
// #############################################################################

//! records the images of a specific Clebsch map


class clebsch_map {

public:
	surface_domain *Surf;
	surface_object *SO;
	finite_field *F;

	int hds, ds, ds_row;

	int line1, line2;
	int transversal;
	int tritangent_plane_idx;

	int line_idx[2];
	int plane_rk_global;

	int intersection_points[6];
	int intersection_points_local[6];
	int Plane[16];
	int base_cols[4];


	long int *Clebsch_map; // [SO->nb_pts]
	int *Clebsch_coeff; // [SO->nb_pts * 4]


	clebsch_map();
	~clebsch_map();
	void freeself();
	void init_half_double_six(surface_object *SO,
			int hds, int verbose_level);
	void init(surface_object *SO, int *line_idx, long int plane_rk_global, int verbose_level);

};


// #############################################################################
// eckardt_point_info.cpp
// #############################################################################

//! information about the Eckardt points of a surface derived from a six-arc


class eckardt_point_info {

public:

	surface_domain *Surf;
	projective_space *P;
	long int arc6[6];

	int *bisecants; // [15]
	int *Intersections; // [15 * 15]
	int *B_pts; // [nb_B_pts]
	int *B_pts_label; // [nb_B_pts * 3]
	int nb_B_pts; // at most 15
	int *E2; // [6 * 5 * 2] Eckardt points of the second type
	int nb_E2; // at most 30
	int *conic_coefficients; // [6 * 6]
	eckardt_point *E;
	int nb_E;

	eckardt_point_info();
	~eckardt_point_info();
	void null();
	void freeself();
	void init(surface_domain *Surf, projective_space *P,
			long int *arc6, int verbose_level);
	void print_bisecants(std::ostream &ost, int verbose_level);
	void print_intersections(std::ostream &ost, int verbose_level);
	void print_conics(std::ostream &ost, int verbose_level);
	void print_Eckardt_points(std::ostream &ost, int verbose_level);

};


// #############################################################################
// eckardt_point.cpp
// #############################################################################

//! Eckardt point on a cubic surface using the Schlaefli labeling


class eckardt_point {

public:

	int len;
	int pt;
	int index[3];


	eckardt_point();
	~eckardt_point();
	void null();
	void freeself();
	void print();
	void latex(std::ostream &ost);
	void latex_index_only(std::ostream &ost);
	void latex_to_str(char *str);
	void latex_to_str_without_E(char *str);
	void init2(int i, int j);
	void init3(int ij, int kl, int mn);
	void init6(int i, int j, int k, int l, int m, int n);
	void init_by_rank(int rk);
	void three_lines(surface_domain *S, int *three_lines);
	int rank();
	void unrank(int rk, int &i, int &j, int &k, int &l, int &m, int &n);

};


// #############################################################################
// seventytwo_cases.cpp
// #############################################################################

//! description of a Clebsch map with a fixed tritangent plane

class seventytwo_cases {

public:

	surface_domain *Surf;

	int f;

	int tritangent_plane_idx;  // = t
		// the tritangent plane picked for the Clebsch map,
		// using the Schlaefli labeling, in [0,44].


	int three_lines_idx[3];
		// the index into Lines[] of the
		// three lines in the chosen tritangent plane
		// This is computed from the Schlaefli labeling
		// using the eckardt point class.

	long int three_lines[3];
		// the three lines in the chosen tritangent plane


	long int tritangent_plane_rk;

	int Basis_pi[16];
	int Basis_pi_inv[17]; // in case it is semilinear

	int line_idx;  // = i
		// the index of the line chosen to be P1,P2 in three_lines[3]
		// three_lines refers to class surfaces_arc_lifting_upstep

	int m1, m2, m3;
		// rearrangement of three_lines_idx[3]
		// m1 = line_idx is the line through P1 and P2.
		// m2 and m3 are the two other lines.

	int l1, l2;
		// the indices of the two lines defining the Clebsch map.
		// They pass through m1.

	int line_l1_l2_idx; // = j

	int transversals[5];
		// the 5 transversals of l1 and l2 in Schlaefli labeling

	long int transversals4[4];
		// the 4 transversals different from m1 in Schlaefli labeling

	long int half_double_six[6];
		// long int because surf->find_half_double_six() requires it that way

	int half_double_six_index;

	long int P6[6];
		// the points of intersection of l1, l2, and of the 4 transversals
		// with the tritangent plane

	long int P6a[6];
		// the arc after the plane has been moved

	long int L1, L2; // images of l1 and l2 under Alpha1 * Alpha2 * Beta1 * Beta2

	long int P6_local[6];
		// the moved arc in local coordinates

	long int P6_local_canonical[6];
		// the canonical form of P6_local[]

	long int P6_perm[6];
	long int P6_perm_mapped[6];
	long int pair[2];
	int the_rest[4];

	int orbit_not_on_conic_idx;
	int pair_orbit_idx;

	int partition_orbit_idx;
	int the_partition4[4];

	int f2;

	seventytwo_cases();
	~seventytwo_cases();
	void init(surface_domain *Surf, int f, int tritangent_plane_idx,
			int *three_lines_idx, long int *three_lines,
			int line_idx, int m1, int m2, int m3, int line_l1_l2_idx, int l1, int l2);
	void compute_arc(surface_object *SO, int verbose_level);
	// We have chosen a tritangent planes and we know the three lines m1, m2, m3 in it.
	// The lines l1 and l2 intersect m1 in the first two points.
	// Computes the 5 transversals to the two lines l1 and l2.
	// One of these lines must be m1, so we remove that to have 4 lines.
	// These 4 lines intersect the two other lines m2 and m3 in the other 4 points.
	// This makes up the arc of 6 points.
	// They will be stored in P6[6].
	void compute_partition(int verbose_level);
	void compute_half_double_six(surface_object *SO, int verbose_level);
	void print();
	void report_seventytwo_maps_line(std::ostream &ost);
	void report_seventytwo_maps_top(std::ostream &ost);
	void report_seventytwo_maps_bottom(std::ostream &ost);
	void report_single_Clebsch_map(std::ostream &ost, int verbose_level);
	void report_Clebsch_map_details(std::ostream &ost, surface_object *SO, int verbose_level);
	void report_Clebsch_map_aut_coset(std::ostream &ost, int coset, int relative_order, int verbose_level);
};




// #############################################################################
// surface_domain.cpp
// #############################################################################

//! cubic surfaces in PG(3,q) with 27 lines


class surface_domain {

public:
	int q;
	int n; // = 4
	int n2; // = 2 * n
	finite_field *F;
	projective_space *P; // PG(3,q)
	projective_space *P2; // PG(2,q)
	grassmann *Gr; // Gr_{4,2}
	grassmann *Gr3; // Gr_{4,3}
	long int nb_lines_PG_3;
	int nb_pts_on_surface; // q^2 + 7q + 1

	orthogonal *O;
	klein_correspondence *Klein;


	// allocated in init_line_data:
	long int *Sets; // [30 * 2]
	int *M; // [6 * 6]
	long int *Sets2; // [15 * 2]


	int Basis0[16];
	int Basis1[16];
	int Basis2[16];
	int o_rank[27];

	int *v; // [n]
	int *v2; // [(n * (n-1)) / 2]
	int *w2; // [(n * (n-1)) / 2]

	int nb_monomials;

	int max_pts; // 27 * (q + 1)
	int *Pts; // [max_pts * n] point coordinates
	long int *pt_list;
		// [max_pts] list of points,
		// used only in compute_system_in_RREF
	int *System; // [max_pts * nb_monomials]
	int *base_cols; // [nb_monomials]

	std::string *Line_label; // [27]
	std::string *Line_label_tex; // [27]

	int *Trihedral_pairs; // [nb_trihedral_pairs * 9]
	std::string *Trihedral_pair_labels; // [nb_trihedral_pairs]
	int *Trihedral_pairs_row_sets; // [nb_trihedral_pairs * 3]
	int *Trihedral_pairs_col_sets; // [nb_trihedral_pairs * 3]
	int nb_trihedral_pairs; // = 120

	tally *Classify_trihedral_pairs_row_values;
	tally *Classify_trihedral_pairs_col_values;

	int nb_Eckardt_points; // = 45
	eckardt_point *Eckardt_points;

	std::string *Eckard_point_label; // [nb_Eckardt_points]
	std::string *Eckard_point_label_tex; // [nb_Eckardt_points]


	int nb_trihedral_to_Eckardt; // nb_trihedral_pairs * 6
	long int *Trihedral_to_Eckardt;
		// [nb_trihedral_pairs * 6]
		// first the three rows, then the three columns
		// long int so that we can induce the action on it

	int nb_collinear_Eckardt_triples;
		// nb_trihedral_pairs * 2
	int *collinear_Eckardt_triples_rank;
		// as three subsets of 45 = nb_Eckardt_points

	tally *Classify_collinear_Eckardt_triples;

	homogeneous_polynomial_domain *Poly1;
		// linear polynomials in three variables
	homogeneous_polynomial_domain *Poly2;
		// quadratic polynomials in three variables
	homogeneous_polynomial_domain *Poly3;
		// cubic polynomials in three variables

	homogeneous_polynomial_domain *Poly1_x123;
		// linear polynomials in three variables
	homogeneous_polynomial_domain *Poly2_x123;
		// quadratic polynomials in three variables
	homogeneous_polynomial_domain *Poly3_x123;
		// cubic polynomials in three variables
	homogeneous_polynomial_domain *Poly4_x123;
		// quartic polynomials in three variables

	homogeneous_polynomial_domain *Poly1_4;
		// linear polynomials in four variables
	homogeneous_polynomial_domain *Poly2_4;
		// quadratic polynomials in four variables
	homogeneous_polynomial_domain *Poly3_4;
		// cubic polynomials in four variables

	long int *Double_six; // [36 * 12]
	std::string *Double_six_label_tex; // [36]


	long int *Half_double_sixes; // [72 * 6]
		// warning: the half double sixes are sorted individually,
		// so the pairing between the lines
		// in the associated double six is gone.
	std::string *Half_double_six_label_tex; // [72]

	int *Half_double_six_to_double_six; // [72]
	int *Half_double_six_to_double_six_row; // [72]

	int f_has_large_polynomial_domains;
	homogeneous_polynomial_domain *Poly2_27;
	homogeneous_polynomial_domain *Poly4_27;
	homogeneous_polynomial_domain *Poly6_27;
	homogeneous_polynomial_domain *Poly3_24;

	int nb_monomials2, nb_monomials4, nb_monomials6;
	int nb_monomials3;

	int *Clebsch_Pij;
	int **Clebsch_P;
	int **Clebsch_P3;

	int *Clebsch_coeffs; // [4 * Poly3->nb_monomials * nb_monomials3]
	int **CC; // [4 * Poly3->nb_monomials]

	int *adjacency_matrix_of_lines;
		// [27 * 27]
		// indexed by the lines in Schlaefli labeling

	int *incidence_lines_vs_tritangent_planes;
		// [27 * 45]
		// indexed by the lines and tritangent planes in Schlaefli labeling

	long int *Lines_in_tritangent_planes;
		// [45 * 3]
		// long int so that we can induce the action on it


	surface_domain();
	~surface_domain();
	void freeself();
	void null();
	void init(finite_field *F, int verbose_level);
	void init_polynomial_domains(int verbose_level);
	void init_large_polynomial_domains(int verbose_level);
	void label_variables_3(homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_x123(homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_4(homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_27(homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_24(homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void init_system(int verbose_level);
	int index_of_monomial(int *v);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void unrank_plane(int *v, long int rk);
	long int rank_plane(int *v);
	int test(int len, long int *S, int verbose_level);
	void enumerate_points(int *coeff, long int *Pts, int &nb_pts,
		int verbose_level);
	void substitute_semilinear(int *coeff_in, int *coeff_out,
		int f_semilinear, int frob, int *Mtx_inv, int verbose_level);
	void list_starter_configurations(long int *Lines, int nb_lines,
		set_of_sets *line_intersections, int *&Table, int &N,
		int verbose_level);
	void create_starter_configuration(int line_idx, int subset_idx,
		set_of_sets *line_neighbors, long int *Lines, long int *S,
		int verbose_level);
	void wedge_to_klein(int *W, int *K);
	void klein_to_wedge(int *K, int *W);
	long int line_to_wedge(long int line_rk);
	void line_to_wedge_vec(long int *Line_rk, long int *Wedge_rk, int len);
	void line_to_klein_vec(long int *Line_rk, long int *Klein_rk, int len);
	long int klein_to_wedge(long int klein_rk);
	void klein_to_wedge_vec(long int *Klein_rk, long int *Wedge_rk, int len);
	void save_lines_in_three_kinds(const char *fname_csv,
		long int *Lines_wedge, long int *Lines, long int *Lines_klein, int nb_lines);
	void find_tritangent_planes_intersecting_in_a_line(int line_idx,
		int &plane1, int &plane2, int verbose_level);
	void make_trihedral_pairs(int verbose_level);
	void process_trihedral_pairs(int verbose_level);
	void make_Tijk(int *T, int i, int j, int k);
	void make_Tlmnp(int *T, int l, int m, int n, int p);
	void make_Tdefght(int *T, int d, int e, int f, int g, int h, int t);
	void make_Eckardt_points(int verbose_level);
	void init_Trihedral_to_Eckardt(int verbose_level);
	int Eckardt_point_from_tritangent_plane(int *tritangent_plane);
	void init_collinear_Eckardt_triples(int verbose_level);
	void find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
		int *E_idx, int nb_E,
		int *&T_idx, int &nb_T, int verbose_level);


	// surface_domain2.cpp:
	void multiply_conic_times_linear(int *six_coeff, int *three_coeff,
		int *ten_coeff, int verbose_level);
	void multiply_linear_times_linear_times_linear(int *three_coeff1,
		int *three_coeff2, int *three_coeff3, int *ten_coeff,
		int verbose_level);
	void multiply_linear_times_linear_times_linear_in_space(
		int *four_coeff1, int *four_coeff2, int *four_coeff3,
		int *twenty_coeff, int verbose_level);
	void multiply_Poly2_3_times_Poly2_3(int *input1, int *input2,
		int *result, int verbose_level);
	void multiply_Poly1_3_times_Poly3_3(int *input1, int *input2,
		int *result, int verbose_level);
	void create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		int *The_six_plane_equations, int *The_surface_equations,
		int verbose_level);
		// The_surface_equations[(q + 1) * 20]
	int plane_from_three_lines(long int *three_lines, int verbose_level);
	void Trihedral_pairs_to_planes(long int *Lines, long int *Planes_by_rank,
		int verbose_level);
		// Planes_by_rank[nb_trihedral_pairs * 6]
	void compute_tritangent_planes_slow(long int *Lines,
		long int *&Tritangent_planes, int &nb_tritangent_planes,
		long int *&Unitangent_planes, int &nb_unitangent_planes,
		long int *&Lines_in_tritangent_plane,
		long int *&Line_in_unitangent_plane,
		int verbose_level);
	void init_double_sixes(int verbose_level);
	void create_half_double_sixes(int verbose_level);
	int find_half_double_six(long int *half_double_six);
	void ijklm2n(int i, int j, int k, int l, int m, int &n);
	void ijkl2mn(int i, int j, int k, int l, int &m, int &n);
	void ijk2lmn(int i, int j, int k, int &l, int &m, int &n);
	void ij2klmn(int i, int j, int &k, int &l, int &m, int &n);
	void get_half_double_six_associated_with_Clebsch_map(
		int line1, int line2, int transversal,
		int hds[6],
		int verbose_level);
	void prepare_clebsch_map(int ds, int ds_row, int &line1,
		int &line2, int &transversal, int verbose_level);
	int clebsch_map(long int *Lines, long int *Pts, int nb_pts,
		int line_idx[2], long int plane_rk,
		long int *Image_rk, int *Image_coeff,
		int verbose_level);
	void clebsch_cubics(int verbose_level);
	void multiply_222_27_and_add(int *M1, int *M2, int *M3,
		int scalar, int *MM, int verbose_level);
	void minor22(int **P3, int i1, int i2, int j1, int j2,
		int scalar, int *Ad, int verbose_level);
	void multiply42_and_add(int *M1, int *M2, int *MM,
		int verbose_level);
	void prepare_system_from_FG(int *F_planes, int *G_planes,
		int lambda, int *&system, int verbose_level);
	void compute_nine_lines(int *F_planes, int *G_planes,
		long int *nine_lines, int verbose_level);
	void compute_nine_lines_by_dual_point_ranks(long int *F_planes_rank,
		long int *G_planes_rank, long int *nine_lines, int verbose_level);
	void split_nice_equation(int *nice_equation, int *&f1,
		int *&f2, int *&f3, int verbose_level);
	void assemble_tangent_quadric(int *f1, int *f2, int *f3,
		int *&tangent_quadric, int verbose_level);
	void tritangent_plane_to_trihedral_pair_and_position(
		int tritangent_plane_idx,
		int &trihedral_pair_idx, int &position, int verbose_level);
	void do_arc_lifting_with_two_lines(
		long int *Arc6, int p1_idx, int p2_idx, int partition_rk,
		long int line1, long int line2,
		int *coeff20, long int *lines27,
		int verbose_level);
	void compute_local_coordinates_of_arc(
			long int *P6, long int *P6_local, int verbose_level);
	int choose_tritangent_plane_for_Clebsch_map(int line_a, int line_b,
				int transversal_line, int verbose_level);



	// surface_domain_lines.cpp:
	void init_line_data(int verbose_level);
	void init_Schlaefli_labels(int verbose_level);
	void unrank_line(int *v, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	int line_ai(int i);
	int line_bi(int i);
	int line_cij(int i, int j);
	int type_of_line(int line);
		// 0 = a_i, 1 = b_i, 2 = c_ij
	void index_of_line(int line, int &i, int &j);
		// returns i for a_i, i for b_i and (i,j) for c_ij
	int third_line_in_tritangent_plane(int l1, int l2, int verbose_level);
	long int rank_line(int *v);
	void build_cubic_surface_from_lines(int len, long int *S, int *coeff,
		int verbose_level);
	int compute_system_in_RREF(int len, long int *S, int verbose_level);
	void compute_intersection_points(int *Adj,
		long int *Lines, int nb_lines,
		long int *&Intersection_pt,
		int verbose_level);
	void compute_intersection_points_and_indices(int *Adj,
		long int *Points, int nb_points,
		long int *Lines, int nb_lines,
		int *&Intersection_pt, int *&Intersection_pt_idx,
		int verbose_level);
	void lines_meet3_and_skew3(long int *lines_meet3, long int *lines_skew3,
		long int *&lines, int &nb_lines, int verbose_level);
	void perp_of_three_lines(long int *three_lines, long int *&perp, int &perp_sz,
		int verbose_level);
	int perp_of_four_lines(long int *four_lines, long int *trans12, int &perp_sz,
		int verbose_level);
	int rank_of_four_lines_on_Klein_quadric(long int *four_lines,
		int verbose_level);
	int create_double_six_from_five_lines_with_a_common_transversal(
		long int *five_pts, long int *double_six,
		int verbose_level);
	int create_double_six_from_six_disjoint_lines(long int *single_six,
			long int *double_six, int verbose_level);
	void create_the_fifteen_other_lines(long int *double_six,
		long int *fifteen_other_lines, int verbose_level);
	void init_adjacency_matrix_of_lines(int verbose_level);
	void init_incidence_matrix_of_lines_vs_tritangent_planes(int verbose_level);
	void set_adjacency_matrix_of_lines(int i, int j);
	int get_adjacency_matrix_of_lines(int i, int j);
	void compute_adjacency_matrix_of_line_intersection_graph(
		int *&Adj,
		long int *S, int n, int verbose_level);
	void compute_adjacency_matrix_of_line_disjointness_graph(
		int *&Adj,
		long int *S, int n, int verbose_level);
	void compute_points_on_lines(
			long int *Pts_on_surface,
			int nb_points_on_surface,
			long int *Lines, int nb_lines,
			set_of_sets *&pts_on_lines,
			int verbose_level);
	int compute_rank_of_any_four(
			long int *&Rk, int &nb_subsets, long int *lines,
		int sz, int verbose_level);
	void rearrange_lines_according_to_a_given_double_six(long int *Lines,
			int *given_double_six,
			long int *New_lines,
			int verbose_level);
	void rearrange_lines_according_to_double_six(long int *Lines,
		int verbose_level);
	void rearrange_lines_according_to_starter_configuration(
		long int *Lines, long int *New_lines,
		int line_idx, int subset_idx, int *Adj,
		set_of_sets *line_intersections, int verbose_level);
	int intersection_of_four_lines_but_not_b6(int *Adj,
		int *four_lines_idx, int b6, int verbose_level);
	int intersection_of_five_lines(int *Adj, int *five_lines_idx,
		int verbose_level);
	void rearrange_lines_according_to_a_given_double_six(long int *Lines,
		long int *New_lines, long int *double_six, int verbose_level);
	void create_lines_from_plane_equations(int *The_plane_equations,
		long int *Lines, int verbose_level);
	int identify_two_lines(long int *lines, int verbose_level);
	int identify_three_lines(long int *lines, int verbose_level);
	void create_remaining_fifteen_lines(
		long int *double_six, long int *fifteen_lines,
		int verbose_level);
	long int compute_cij(long int *double_six,
		int i, int j, int verbose_level);
	int compute_transversals_of_any_four(
			long int *&Trans, int &nb_subsets,
			long int *lines, int sz, int verbose_level);

	// surface_domain_io.cpp:
	void print_equation(std::ostream &ost, int *coeffs);
	void print_equation_tex(std::ostream &ost, int *coeffs);
	void print_equation_tex_lint(std::ostream &ost, long int *coeffs);
	void latex_double_six(std::ostream &ost, long int *double_six);
	void make_spreadsheet_of_lines_in_three_kinds(spreadsheet *&Sp,
		long int *Wedge_rk, long int *Line_rk, long int *Klein_rk, int nb_lines,
		int verbose_level);
	void print_line(std::ostream &ost, int rk);
	void latex_table_of_double_sixes(std::ostream &ost);
	void latex_table_of_half_double_sixes(std::ostream &ost);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_abstract_trihedral_pair(std::ostream &ost, int t_idx);
	void latex_trihedral_pair(std::ostream &ost, int *T, long int *TE);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost);
	void latex_half_double_six(std::ostream &ost, int idx);
	void latex_table_of_Eckardt_points(std::ostream &ost);
	void latex_table_of_tritangent_planes(std::ostream &ost);
	void print_equation_in_trihedral_form(std::ostream &ost,
		int *the_six_plane_equations, int lambda, int *the_equation);
	void print_equation_wrapped(std::ostream &ost, int *the_equation);
	void print_lines_tex(std::ostream &ost, long int *Lines);
	void print_clebsch_P(std::ostream &ost);
	void print_clebsch_P_matrix_only(std::ostream &ost);
	void print_clebsch_cubics(std::ostream &ost);
	void print_system(std::ostream &ost, int *system);
	void print_trihedral_pair_in_dual_coordinates_in_GAP(
		long int *F_planes_rank, long int *G_planes_rank);
	void print_basics(std::ostream &ost);
	void print_polynomial_domains(std::ostream &ost);
	void print_Schlaefli_labelling(std::ostream &ost);
	void print_set_of_lines_tex(std::ostream &ost, long int *v, int len);
	void latex_table_of_clebsch_maps(std::ostream &ost);
	void print_half_double_sixes_in_GAP();
	void sstr_line_label(std::stringstream &sstr, long int pt);


	// surface_domain_families.cpp:
	void create_equation_F13(int a, int *coeff, int verbose_level);
	void create_equation_G13(int a, int *coeff, int verbose_level);
	int create_surface_F13(int a,
		int *coeff20,
		long int *Lines27,
		int &nb_E,
		int verbose_level);
	int create_surface_G13(int a,
		int *coeff20,
		long int *Lines27,
		int &nb_E,
		int verbose_level);
	void create_equation_HCV(int a, int b, int *coeff, int verbose_level);
	int test_HCV_form_alpha_beta(int *coeff, int &alpha, int &beta,
		int verbose_level);
	void create_HCV_double_six(long int *double_six, int a, int b,
		int verbose_level);
	void create_HCV_fifteen_lines(long int *fifteen_lines, int a, int b,
		int verbose_level);
	void create_surface_family_HCV(int a,
		long int *Lines27,
		int *equation20, int verbose_level);
	int create_surface_HCV(int a, int b,
		int *coeff20,
		long int *Lines27,
		int &alpha, int &beta, int &nb_E,
		int verbose_level);


};

void callback_surface_domain_sstr_line_label(std::stringstream &sstr, long int pt, void *data);

// #############################################################################
// surface_object.cpp
// #############################################################################

//! a particular cubic surface in PG(3,q), given by its equation


class surface_object {

public:
	int q;
	finite_field *F;
	surface_domain *Surf;

	long int Lines[27];
	int eqn[20];

	long int *Pts;
	int nb_pts;

	int nb_planes;

	set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	set_of_sets *lines_on_point;

	long int *Eckardt_points;
	int *Eckardt_points_index;
	int nb_Eckardt_points;

	long int *Double_points;
	int *Double_points_index;
	int nb_Double_points;

	long int *Pts_not_on_lines;
	int nb_pts_not_on_lines;

	int *plane_type_by_points;
	int *plane_type_by_lines;

	tally *C_plane_type_by_points;
	tally *Type_pts_on_lines;
	tally *Type_lines_on_point;

	long int *Tritangent_plane_rk; // [45]
		// list of tritangent planes in Schlaefli labeling
	int nb_tritangent_planes;

#if 0
	long int *Tritangent_planes; // [nb_tritangent_planes]
	int nb_tritangent_planes;
	long int *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]
	int *Tritangent_plane_dual; // [nb_tritangent_planes]

	int *iso_type_of_tritangent_plane; // [nb_tritangent_planes]
	tally *Type_iso_tritangent_planes;


	long int *Unitangent_planes; // [nb_unitangent_planes]
	int nb_unitangent_planes;
	long int *Line_in_unitangent_plane; // [nb_unitangent_planes]

	int *Tritangent_planes_on_lines; // [27 * 5]
	int *Tritangent_plane_to_Eckardt; // [nb_tritangent_planes]
	int *Eckardt_to_Tritangent_plane; // [nb_tritangent_planes]
	long int *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]
	int *Unitangent_planes_on_lines; // [27 * (q + 1 - 5)]
#endif

	long int *Lines_in_tritangent_planes; // [nb_tritangent_planes * 3]

	long int *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]


	long int *All_Planes; // [nb_trihedral_pairs * 6]
	int *Dual_point_ranks; // [nb_trihedral_pairs * 6]

	int *Adj_line_intersection_graph; // [27 * 27]
	set_of_sets *Line_neighbors;
	int *Line_intersection_pt; // [27 * 27]
	int *Line_intersection_pt_idx; // [27 * 27]

	surface_object();
	~surface_object();
	void freeself();
	void null();
	int init_equation(surface_domain *Surf, int *eqn, int verbose_level);
		// returns FALSE if the surface does not have 27 lines
	void init(surface_domain *Surf, long int *Lines, int *eqn,
		int f_find_double_six_and_rearrange_lines, int verbose_level);
	void compute_properties(int verbose_level);
	void find_double_six_and_rearrange_lines(long int *Lines, int verbose_level);
	void enumerate_points(int verbose_level);
	void compute_adjacency_matrix_of_line_intersection_graph(
		int verbose_level);
	void print_neighbor_sets(std::ostream &ost);
	int Adj_ij(int i, int j);
	void compute_plane_type_by_points(int verbose_level);
	void compute_tritangent_planes_by_rank(int verbose_level);
	void compute_Lines_in_tritangent_planes(int verbose_level);
	void compute_Trihedral_pairs_as_tritangent_planes(int verbose_level);
	//void compute_tritangent_planes(int verbose_level);
	void compute_planes_and_dual_point_ranks(int verbose_level);
	void print_everything(std::ostream &ost, int verbose_level);
	void report_properties(std::ostream &ost, int verbose_level);
	void print_line_intersection_graph(std::ostream &ost);
	void print_adjacency_list(std::ostream &ost);
	void print_adjacency_matrix(std::ostream &ost);
	void print_adjacency_matrix_with_intersection_points(std::ostream &ost);
	void print_planes_in_trihedral_pairs(std::ostream &ost);
	void print_tritangent_planes(std::ostream &ost);
	void print_single_tritangent_planes(std::ostream &ost, int plane_idx);
	//void print_generalized_quadrangle(std::ostream &ost);
	void print_plane_type_by_points(std::ostream &ost);
	void print_lines(std::ostream &ost);
	void print_lines_with_points_on_them(std::ostream &ost);
	void print_equation(std::ostream &ost);
	void print_general(std::ostream &ost);
	void print_affine_points_in_source_code(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_Eckardt_points(std::ostream &ost);
	void print_double_points(std::ostream &ost);
	void print_points_on_surface(std::ostream &ost);
	void print_points_on_lines(std::ostream &ost);
	void print_points_on_surface_but_not_on_a_line(std::ostream &ost);
	void print_double_sixes(std::ostream &ost);
	void print_half_double_sixes(std::ostream &ost);
	void print_half_double_sixes_numerically(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost);
	void print_trihedral_pairs_numerically(std::ostream &ost);
	void latex_table_of_trihedral_pairs_and_clebsch_system(std::ostream &ost,
		int *T, int nb_T);
	void latex_table_of_trihedral_pairs(std::ostream &ost, int *T, int nb_T);
	void latex_trihedral_pair(std::ostream &ost, int t_idx);
	void make_equation_in_trihedral_form(int t_idx,
		int *F_planes, int *G_planes, int &lambda, int *equation,
		int verbose_level);
	void print_equation_in_trihedral_form(std::ostream &ost,
		int *F_planes, int *G_planes, int lambda);
	void print_equation_in_trihedral_form_equation_only(std::ostream &ost,
		int *F_planes, int *G_planes, int lambda);
	void make_and_print_equation_in_trihedral_form(std::ostream &ost, int t_idx);
#if 0
	void identify_double_six_from_trihedral_pair(int *Lines,
		int t_idx, int *nine_lines, int *double_sixes,
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_one(int *Lines,
		int t_idx, int *nine_line_idx, int *double_sixes,
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_two(int *Lines,
		int t_idx, int *nine_line_idx, int *double_sixes,
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_three(int *Lines,
		int t_idx, int *nine_line_idx, int *double_sixes,
		int verbose_level);
	void find_common_transversals_to_two_disjoint_lines(int a, int b,
		int *transversals5);
	void find_common_transversals_to_three_disjoint_lines(int a1, int a2,
		int a3, int *transversals3);
	void find_common_transversals_to_four_disjoint_lines(int a1, int a2,
		int a3, int a4, int *transversals2);
	int find_tritangent_plane_through_two_lines(int line_a, int line_b);
	void get_planes_through_line(int *new_lines,
		int line_idx, int *planes5);
	void find_two_lines_in_plane(int plane_idx, int forbidden_line,
		int &line1, int &line2);
	int find_unique_line_in_plane(int plane_idx, int forbidden_line1,
		int forbidden_line2);
	int choose_tritangent_plane(int line_a, int line_b,
		int transversal_line, int verbose_level);
	void find_all_tritangent_planes(
		int line_a, int line_b, int transversal_line,
		int *tritangent_planes3,
		int verbose_level);
#endif
	void identify_lines(long int *lines, int nb_lines, int *line_idx,
		int verbose_level);
	void print_nine_lines_latex(std::ostream &ost, long int *nine_lines,
		int *nine_lines_idx);
	int compute_transversal_line(int line_a, int line_b,
		int verbose_level);
	void compute_transversal_lines(
		int line_a, int line_b, int *transversals5,
		int verbose_level);
	void clebsch_map_find_arc_and_lines(long int *Clebsch_map,
		long int *Arc, long int *Blown_up_lines, int verbose_level);
	void clebsch_map_print_fibers(long int *Clebsch_map);
	//void compute_clebsch_maps(int verbose_level);
	void compute_clebsch_map(int line_a, int line_b,
		int transversal_line,
		long int &tritangent_plane_rk,
		long int *Clebsch_map, int *Clebsch_coeff,
		int verbose_level);
	// Clebsch_map[nb_pts]
	// Clebsch_coeff[nb_pts * 4]
	void clebsch_map_latex(std::ostream &ost,
			long int *Clebsch_map, int *Clebsch_coeff);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void latex_trihedral_pair(std::ostream &ost, int *T, long int *TE);


};



// #############################################################################
// web_of_cubic_curves.cpp
// #############################################################################

//! a web of cubic curves which is used to create an algebraic variety


class web_of_cubic_curves {

public:
	surface_domain *Surf;


	long int arc6[6];

	eckardt_point_info *E;

	int *E_idx;

	int *T_idx;
	int nb_T;


	int base_curves4[4];
	int t_idx0;
	long int row_col_Eckardt_points[6];
	int six_curves[6 * 10];
	int *Web_of_cubic_curves; // [45 * 10]
	int *Tritangent_plane_equations; // [45 * 4]
	int *base_curves; // [4 * 10]
	long int *The_plane_rank; // [45]
	long int *The_plane_duals; // [45]
	long int *Dual_point_ranks; // [nb_T * 6]
	long int Lines27[27];

	web_of_cubic_curves();
	~web_of_cubic_curves();
	void init(surface_domain *Surf, long int *arc6, int verbose_level);
	void compute_web_of_cubic_curves(long int *arc6, int verbose_level);
	void rank_of_foursubsets(int *&rk, int &N, int verbose_level);
	void create_web_and_equations_based_on_four_tritangent_planes(
			long int *arc6, int *base_curves4,
			int verbose_level);
	void find_Eckardt_points(int verbose_level);
	void find_trihedral_pairs(int verbose_level);
	void extract_six_curves_from_web(
		int verbose_level);
	void create_surface_equation_from_trihedral_pair(long int *arc6,
		int t_idx, int *surface_equation,
		int &lambda,
		int verbose_level);
	void create_lambda_from_trihedral_pair_and_arc(
		long int *arc6, int t_idx,
		int &lambda, int &lambda_rk,
		int verbose_level);
	void find_point_not_on_six_curves(int &pt, int &f_point_was_found,
		int verbose_level);
	void print_lines(std::ostream &ost);
	void print_trihedral_plane_equations(std::ostream &ost);
	void print_the_six_plane_equations(
		int *The_six_plane_equations,
		long int *plane6, std::ostream &ost);
	void print_surface_equations_on_line(
		int *The_surface_equations,
		int lambda, int lambda_rk, std::ostream &ost);
	void print_dual_point_ranks(std::ostream &ost);
	void print_Eckardt_point_data(std::ostream &ost, int verbose_level);
	void report_basics(std::ostream &ost, int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void print_web_of_cubic_curves(long int *arc6, std::ostream &ost);

};




}}





#endif /* SRC_LIB_FOUNDATIONS_SURFACES_SURFACES_H_ */
