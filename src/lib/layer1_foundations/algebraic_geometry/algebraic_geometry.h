/*
 * surfaces.h
 *
 *  Created on: Jul 29, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_ALGEBRAIC_GEOMETRY_ALGEBRAIC_GEOMETRY_H_
#define SRC_LIB_FOUNDATIONS_ALGEBRAIC_GEOMETRY_ALGEBRAIC_GEOMETRY_H_

namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {




// #############################################################################
// algebraic_geometry_global.cpp
// #############################################################################

//! catch all class for everything related to algebraic geometry


class algebraic_geometry_global {

public:

	algebraic_geometry_global();
	~algebraic_geometry_global();
	void analyze_del_Pezzo_surface(
			geometry::projective_space *P,
			expression_parser::formula *Formula,
			std::string &evaluate_text,
			int verbose_level);
	void report_grassmannian(
			geometry::projective_space *P,
			int k,
			int verbose_level);
	void map(
			geometry::projective_space *P,
			std::string &ring_label,
			std::string &formula_label,
			std::string &evaluate_text,
			long int *&Image_pts,
			int &N_points,
			int verbose_level);


};



// #############################################################################
// arc_lifting_with_two_lines.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting_with_two_lines {

public:

	int q;
	field_theory::finite_field *F; // do not free

	surface_domain *Surf; // do not free

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
	void create_surface(
		surface_domain *Surf,
		long int *Arc6, long int line1, long int line2,
		int verbose_level);
	// The arc must be given as points in PG(3,q), not in PG(2,q).
};



// #############################################################################
// clebsch_map.cpp
// #############################################################################

//! records the images of a specific Clebsch map, used by class layer5_applications::applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_clebsch_map


class clebsch_map {

public:
	surface_domain *Surf;
	surface_object *SO;
	field_theory::finite_field *F;

	int hds, ds, ds_row;

	int line1, line2;
	int transversal;
	int tritangent_plane_idx;

	int line_idx[2];
	long int plane_rk_global;

	int intersection_points[6];
	int intersection_points_local[6];
	int Plane[16];
	int base_cols[4];


	long int *Clebsch_map; // [SO->nb_pts], = Image_rk
	int *Clebsch_coeff; // [SO->nb_pts * 4], = Image_coeff

	long int Arc[6];
	long int Blown_up_lines[6];

	clebsch_map();
	~clebsch_map();
	void init_half_double_six(surface_object *SO,
			int hds, int verbose_level);
	void compute_Clebsch_map_down(int verbose_level);
	int compute_Clebsch_map_down_worker(
		long int *Image_rk, int *Image_coeff,
		int verbose_level);
	// assuming:
	// In:
	// SO->Lines[27]
	// SO->Pts[SO->nb_pts]
	// Out:
	// Image_rk[nb_pts]  (image point in the plane in local coordinates)
	//   Note Image_rk[i] is -1 if Pts[i] does not have an image.
	// Image_coeff[nb_pts * 4] (image point in the plane in PG(3,q) coordinates)
	void clebsch_map_print_fibers();
	void clebsch_map_find_arc_and_lines(
		//long int *Clebsch_map,
		//long int *Arc, long int *Blown_up_lines,
		int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};


// #############################################################################
// cubic_curve.cpp
// #############################################################################

//! cubic curves in PG(2,q)


class cubic_curve {

public:
	int q;
	field_theory::finite_field *F;
	geometry::projective_space *P; // PG(2,q)


	int nb_monomials;


	ring_theory::homogeneous_polynomial_domain *Poly;
		// cubic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly2;
		// quadratic polynomials in three variables

	ring_theory::partial_derivative *Partials;

	int *gradient; // 3 * Poly2->nb_monomials


	cubic_curve();
	~cubic_curve();
	void init(field_theory::finite_field *F, int verbose_level);
	int compute_system_in_RREF(
			int nb_pts, long int *pt_list, int verbose_level);
	void compute_gradient(
			int *eqn_in, int verbose_level);
	void compute_singular_points(
			int *eqn_in,
			long int *Pts_on_curve, int nb_pts_on_curve,
			long int *Pts, int &nb_pts,
			int verbose_level);
	void compute_inflexion_points(
			int *eqn_in,
			long int *Pts_on_curve, int nb_pts_on_curve,
			long int *Pts, int &nb_pts,
			int verbose_level);

};




// #############################################################################
// del_pezzo_surface_of_degree_two_domain.cpp
// #############################################################################

//! domain for del Pezzo surfaces of degree two


class del_pezzo_surface_of_degree_two_domain {

public:
	field_theory::finite_field *F;
	geometry::projective_space *P3;
	geometry::projective_space *P2;
	geometry::grassmann *Gr; // Gr_{4,2}
	geometry::grassmann *Gr3; // Gr_{4,3}
	long int nb_lines_PG_3;
	ring_theory::homogeneous_polynomial_domain *Poly4_3;
		// quartic polynomials in three variables

	del_pezzo_surface_of_degree_two_domain();
	~del_pezzo_surface_of_degree_two_domain();
	void init(
			geometry::projective_space *P3,
			ring_theory::homogeneous_polynomial_domain *Poly4_3,
			int verbose_level);
	void enumerate_points(int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	void print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);

};


// #############################################################################
// del_pezzo_surface_of_degree_two_object.cpp
// #############################################################################

//! a del Pezzo surface of degree two


class del_pezzo_surface_of_degree_two_object {

public:
	del_pezzo_surface_of_degree_two_domain *Dom;

	expression_parser::formula *RHS;
	expression_parser::syntax_tree_node **Subtrees;
	int *Coefficient_vector;

	geometry::points_and_lines *pal;


	del_pezzo_surface_of_degree_two_object();
	~del_pezzo_surface_of_degree_two_object();
	void init(
			del_pezzo_surface_of_degree_two_domain *Dom,
			expression_parser::formula *RHS,
			expression_parser::syntax_tree_node **Subtrees,
			int *Coefficient_vector,
			int verbose_level);
	void enumerate_points_and_lines(int verbose_level);
	void create_latex_report(std::string &label, std::string &label_tex, int verbose_level);
	void report_properties(std::ostream &ost, int verbose_level);
	void print_equation(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_all_points_on_surface(std::ostream &ost);
	void print_lines(std::ostream &ost);

};



// #############################################################################
// eckardt_point_info.cpp
// #############################################################################

//! information about the Eckardt points of a surface derived from a six-arc


class eckardt_point_info {

public:

	//surface_domain *Surf;
	geometry::projective_space *P2;
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
	void init(geometry::projective_space *P2,
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
	void print();
	void latex(std::ostream &ost);
	void latex_index_only(std::ostream &ost);
	void latex_to_string(std::string &s);
	void latex_to_str_without_E(std::string &s);
	void init2(int i, int j);
	void init3(int ij, int kl, int mn);
	void init6(int i, int j, int k, int l, int m, int n);
	void init_by_rank(int rk);
	void three_lines(surface_domain *S, int *three_lines);
	int rank();
	void unrank(int rk, int &i, int &j, int &k, int &l, int &m, int &n);

};

// #############################################################################
// quartic_curve_domain.cpp
// #############################################################################

//! domain for quartic curves in PG(2,q) with 28 bitangents


class quartic_curve_domain {

public:
	field_theory::finite_field *F;
	geometry::projective_space *P;

	// we use the monomial ordering t_PART in all polynomial rings:

	ring_theory::homogeneous_polynomial_domain *Poly1_3;
		// linear polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly2_3;
		// quadratic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly3_3;
		// cubic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly4_3;
		// quartic polynomials in three variables

	ring_theory::homogeneous_polynomial_domain *Poly3_4;
		// cubic polynomials in four variables

	ring_theory::partial_derivative *Partials; // [3]

	algebraic_geometry::schlaefli_labels *Schlaefli;

	quartic_curve_domain();
	~quartic_curve_domain();
	void init(field_theory::finite_field *F, int verbose_level);
	void init_polynomial_domains(int verbose_level);
	void print_equation_maple(std::stringstream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs);
	void print_gradient_with_line_breaks_tex(std::ostream &ost, int *coeffs);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	void unrank_line_in_dual_coordinates(int *v, long int rk);
	void print_lines_tex(std::ostream &ost, long int *Lines, int nb_lines);
	void compute_points_on_lines(
			long int *Pts, int nb_points,
			long int *Lines, int nb_lines,
			data_structures::set_of_sets *&pts_on_lines,
			int *&f_is_on_line,
			int verbose_level);
	void multiply_conic_times_conic(int *six_coeff_a,
		int *six_coeff_b, int *fifteen_coeff,
		int verbose_level);
	void multiply_conic_times_line(int *six_coeff,
		int *three_coeff, int *ten_coeff,
		int verbose_level);
	void multiply_line_times_line(int *line1,
		int *line2, int *six_coeff,
		int verbose_level);
	void multiply_three_lines(int *line1, int *line2, int *line3,
		int *ten_coeff,
		int verbose_level);
	void multiply_four_lines(int *line1, int *line2, int *line3, int *line4,
		int *fifteen_coeff,
		int verbose_level);
	void assemble_cubic_surface(int *f1, int *f2, int *f3, int *eqn20,
		int verbose_level);
	void create_surface(quartic_curve_object *Q, int *eqn20, int verbose_level);
	// Given a quartic Q in X1,X2,X3, compute an associated cubic surface
	// whose projection from (1,0,0,0) gives back the quartic Q.
	// Pick 4 bitangents L0,L1,L2,L3 so that the 8 points of tangency lie on a conic C.
	// Then, create the cubic surface with equation
	// (- lambda * mu) / 4 * X0^2 * L0 (the equation of the first of the four bitangents)
	// + X0 * lambda * C (the conic equation)
	// + L1 * L2 * L3 (the product of the equations of the last three bitangents)
	// Here 1, lambda, mu are the coefficients of a linear dependency between
	// Q (the quartic), C^2, L0*L1*L2*L3, so
	// Q + lambda * C^2 + mu * L0*L1*L2*L3 = 0.
	void compute_gradient(int *equation15, int *&gradient, int verbose_level);

};


// #############################################################################
// quartic_curve_object_properties.cpp
// #############################################################################

//! properties of a particular quartic curve surface in PG(2,q), as defined by an object of class quartic_curve_object


class quartic_curve_object_properties {

public:

	quartic_curve_object *QO;


	data_structures::set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	int *f_is_on_line; // [QO->nb_pts]

	data_structures::tally *Bitangent_line_type;
	int line_type_distribution[3];

	data_structures::set_of_sets *lines_on_point;
	data_structures::tally *Point_type;

	int f_fullness_has_been_established;
	int f_is_full;
	int nb_Kovalevski;
	int nb_Kovalevski_on;
	int nb_Kovalevski_off;
	int *Kovalevski_point_idx;
	long int *Kovalevski_points;

	long int *Pts_off;
	int nb_pts_off;

	data_structures::set_of_sets *pts_off_on_lines;
	int *f_is_on_line2; // [QO->nb_pts]

	data_structures::set_of_sets *lines_on_points_off;
	data_structures::tally *Point_off_type;


	int *gradient;

	long int *singular_pts;
	int nb_singular_pts;
	int nb_non_singular_pts;

	long int *tangent_line_rank_global; // [QO->nb_pts]
	long int *tangent_line_rank_dual; // [nb_non_singular_pts]



	quartic_curve_object_properties();
	~quartic_curve_object_properties();
	void init(quartic_curve_object *QO, int verbose_level);
	void create_summary_file(std::string &fname,
			std::string &surface_label, std::string &col_postfix, int verbose_level);
	void report_properties_simple(std::ostream &ost, int verbose_level);
	void print_equation(std::ostream &ost);
	void print_gradient(std::ostream &ost);
	void print_general(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_all_points(std::ostream &ost);
	void print_bitangents(std::ostream &ost);
	void print_lines_with_points_on_them(std::ostream &ost,
			long int *Lines, int nb_lines, data_structures::set_of_sets *SoS);
	//void print_bitangents_with_points_on_them(std::ostream &ost);
	void points_on_curve_on_lines(int verbose_level);
	void report_bitangent_line_type(std::ostream &ost);
	void compute_gradient(int verbose_level);
	void compute_singular_points_and_tangent_lines(int verbose_level);
	// a singular point is a point where all partials vanish
	// We compute the set of singular points into Pts[nb_pts]

};



// #############################################################################
// quartic_curve_object.cpp
// #############################################################################

//! a particular quartic curve in PG(2,q), given by its equation


class quartic_curve_object {

public:
	int q;
	field_theory::finite_field *F;
	quartic_curve_domain *Dom;

	long int *Pts; // in increasing order
	int nb_pts;


	//long int *Lines;
	//int nb_lines;

	int eqn15[15];

	int f_has_bitangents;
	long int bitangents28[28];

	quartic_curve_object_properties *QP;



	quartic_curve_object();
	~quartic_curve_object();
	void init_equation_but_no_bitangents(quartic_curve_domain *Dom,
			int *eqn15,
			int verbose_level);
	void init_equation_and_bitangents(quartic_curve_domain *Dom,
			int *eqn15, long int *bitangents28,
			int verbose_level);
	void init_equation_and_bitangents_and_compute_properties(quartic_curve_domain *Dom,
			int *eqn15, long int *bitangents28,
			int verbose_level);
	void enumerate_points(int verbose_level);
	void compute_properties(int verbose_level);
	void recompute_properties(int verbose_level);
	void identify_lines(long int *lines, int nb_lines, int *line_idx,
		int verbose_level);
	int find_point(long int P, int &idx);

};




// #############################################################################
// schlaefli_labels.cpp
// #############################################################################

//! schlaefli labeling of objects in cubic surfaces with 27 lines


class schlaefli_labels {

public:

	long int *Sets; // [30 * 2]
	int *M; // [6 * 6]
	long int *Sets2; // [15 * 2]


	std::string *Line_label; // [27]
	std::string *Line_label_tex; // [27]

	schlaefli_labels();
	~schlaefli_labels();
	void init(int verbose_level);
};


// #############################################################################
// schlaefli.cpp
// #############################################################################

//! schlaefli labeling of objects in cubic surfaces with 27 lines


class schlaefli {

public:

	surface_domain *Surf;


	schlaefli_labels *Labels;

#if 0
	long int *Sets; // [30 * 2]
	int *M; // [6 * 6]
	long int *Sets2; // [15 * 2]

	// Schlaefli stuff:

	std::string *Line_label; // [27]
	std::string *Line_label_tex; // [27]
#endif

	int *Trihedral_pairs; // [nb_trihedral_pairs * 9]
	std::string *Trihedral_pair_labels; // [nb_trihedral_pairs]
	int *Trihedral_pairs_row_sets; // [nb_trihedral_pairs * 3]
	int *Trihedral_pairs_col_sets; // [nb_trihedral_pairs * 3]
	int nb_trihedral_pairs; // = 120

	int *Triads;
	int nb_triads; // = 40

	data_structures::tally *Classify_trihedral_pairs_row_values;
	data_structures::tally *Classify_trihedral_pairs_col_values;

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

	data_structures::tally *Classify_collinear_Eckardt_triples;


	// Schlaefli stuff, part2:


	long int *Double_six; // [36 * 12]
	std::string *Double_six_label_tex; // [36]


	long int *Half_double_sixes; // [72 * 6]
		// warning: the half double sixes are sorted individually,
		// so the pairing between the lines
		// in the associated double six is gone.
	std::string *Half_double_six_label_tex; // [72]

	int *Half_double_six_to_double_six; // [72]
	int *Half_double_six_to_double_six_row; // [72]


	// Schlaefli stuff, part3:

	int *adjacency_matrix_of_lines;
		// [27 * 27]
		// indexed by the lines in Schlaefli labeling

	int *incidence_lines_vs_tritangent_planes;
		// [27 * 45]
		// indexed by the lines and tritangent planes in Schlaefli labeling

	long int *Lines_in_tritangent_planes;
		// [45 * 3]
		// long int so that we can induce the action on it



	schlaefli();
	~schlaefli();
	void init(surface_domain *Surf, int verbose_level);
	void init_line_data(int verbose_level);
	void init_Schlaefli_labels(int verbose_level);
	void find_tritangent_planes_intersecting_in_a_line(
		int line_idx,
		int &plane1, int &plane2, int verbose_level);
	void make_triads(int verbose_level);
	void make_trihedral_pair_disjointness_graph(int *&Adj, int verbose_level);
	void make_trihedral_pairs(int verbose_level);
	void process_trihedral_pairs(int verbose_level);
	int line_ai(int i);
	int line_bi(int i);
	int line_cij(int i, int j);
	int type_of_line(int line);
		// 0 = a_i, 1 = b_i, 2 = c_ij
	void index_of_line(int line, int &i, int &j);
		// returns i for a_i, i for b_i and (i,j) for c_ij
	int third_line_in_tritangent_plane(int l1, int l2, int verbose_level);
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
	void init_adjacency_matrix_of_lines(int verbose_level);
	void init_incidence_matrix_of_lines_vs_tritangent_planes(int verbose_level);
	void set_adjacency_matrix_of_lines(int i, int j);
	int get_adjacency_matrix_of_lines(int i, int j);
	int choose_tritangent_plane_for_Clebsch_map(int line_a, int line_b,
				int transversal_line, int verbose_level);

	void latex_table_of_double_sixes(std::ostream &ost);
	void latex_table_of_half_double_sixes(std::ostream &ost);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_abstract_trihedral_pair(std::ostream &ost, int t_idx);
	void latex_table_of_Schlaefli_labeling_of_lines(std::ostream &ost);
	void latex_trihedral_pair(std::ostream &ost, int *T, long int *TE);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void latex_triads(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost);
	void latex_half_double_six(std::ostream &ost, int idx);
	void latex_table_of_Eckardt_points(std::ostream &ost);
	void latex_table_of_tritangent_planes(std::ostream &ost);
	void print_line(std::ostream &ost, int rk);
	void print_Schlaefli_labelling(std::ostream &ost);
	void print_set_of_lines_tex(std::ostream &ost, long int *v, int len);
	void latex_table_of_clebsch_maps(std::ostream &ost);
	void print_half_double_sixes_in_GAP();
	int identify_Eckardt_point(int line1, int line2, int line3, int verbose_level);

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
	field_theory::finite_field *F;
	geometry::projective_space *P; // PG(3,q)
	geometry::projective_space *P2; // PG(2,q)
	geometry::grassmann *Gr; // Gr_{4,2}
	geometry::grassmann *Gr3; // Gr_{4,3}
	long int nb_lines_PG_3;
	int nb_pts_on_surface_with_27_lines; // q^2 + 7q + 1

	orthogonal_geometry::orthogonal *O;
	geometry::klein_correspondence *Klein;


	int Basis0[16];
	int Basis1[16];
	int Basis2[16];

	// used in line_to_wedge and klein_to_wedge:
	int *v; // [n]
	int *v2; // [(n * (n-1)) / 2]
	int *w2; // [(n * (n-1)) / 2]

	int nb_monomials; // = 20

	schlaefli *Schlaefli;


	ring_theory::homogeneous_polynomial_domain *Poly1;
		// linear polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly2;
		// quadratic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly3;
		// cubic polynomials in three variables

	ring_theory::homogeneous_polynomial_domain *Poly1_x123;
		// linear polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly2_x123;
		// quadratic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly3_x123;
		// cubic polynomials in three variables
	ring_theory::homogeneous_polynomial_domain *Poly4_x123;
		// quartic polynomials in three variables

	ring_theory::homogeneous_polynomial_domain *Poly1_4;
		// linear polynomials in four variables
	ring_theory::homogeneous_polynomial_domain *Poly2_4;
		// quadratic polynomials in four variables
	ring_theory::homogeneous_polynomial_domain *Poly3_4;
		// cubic polynomials in four variables

	ring_theory::partial_derivative *Partials; // [4]


	int f_has_large_polynomial_domains;
	ring_theory::homogeneous_polynomial_domain *Poly2_27;
	ring_theory::homogeneous_polynomial_domain *Poly4_27;
	ring_theory::homogeneous_polynomial_domain *Poly6_27;
	ring_theory::homogeneous_polynomial_domain *Poly3_24;

	int nb_monomials2, nb_monomials4, nb_monomials6;
	int nb_monomials3;

	int *Clebsch_Pij;
	int **Clebsch_P;
	int **Clebsch_P3;

	int *Clebsch_coeffs; // [4 * Poly3->nb_monomials * nb_monomials3]
	int **CC; // [4 * Poly3->nb_monomials]

	surface_domain();
	~surface_domain();
	void init(field_theory::finite_field *F, int verbose_level);
	void init_polynomial_domains(int verbose_level);
	void init_large_polynomial_domains(int verbose_level);
	void label_variables_3(ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_x123(ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_4(ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_27(ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	void label_variables_24(ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level);
	int index_of_monomial(int *v);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	void unrank_plane(int *v, long int rk);
	long int rank_plane(int *v);
	void enumerate_points(int *coeff,
			std::vector<long int> &Pts,
			int verbose_level);
	void substitute_semilinear(int *coeff_in, int *coeff_out,
		int f_semilinear, int frob, int *Mtx_inv, int verbose_level);
	void list_starter_configurations(long int *Lines, int nb_lines,
			data_structures::set_of_sets *line_intersections, int *&Table, int &N,
		int verbose_level);
	void create_starter_configuration(int line_idx, int subset_idx,
			data_structures::set_of_sets *line_neighbors, long int *Lines, long int *S,
		int verbose_level);
	void wedge_to_klein(int *W, int *K);
	void klein_to_wedge(int *K, int *W);
	long int line_to_wedge(long int line_rk);
	void line_to_wedge_vec(long int *Line_rk, long int *Wedge_rk, int len);
	void line_to_klein_vec(long int *Line_rk, long int *Klein_rk, int len);
	long int klein_to_wedge(long int klein_rk);
	void klein_to_wedge_vec(long int *Klein_rk, long int *Wedge_rk, int len);
	void save_lines_in_three_kinds(std::string &fname_csv,
		long int *Lines_wedge, long int *Lines, long int *Lines_klein, int nb_lines);
	int build_surface_from_double_six_and_count_Eckardt_points(long int *double_six, int verbose_level);


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
	long int plane_from_three_lines(long int *three_lines, int verbose_level);
	void Trihedral_pairs_to_planes(long int *Lines, long int *Planes_by_rank,
		int verbose_level);
		// Planes_by_rank[nb_trihedral_pairs * 6]
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
	void compute_gradient(int *equation20, int *&gradient, int verbose_level);
	long int compute_tangent_plane(int *pt_coords, int *equation20, int verbose_level);




	// surface_domain_lines.cpp:
	void init_Schlaefli(int verbose_level);
	void unrank_line(int *v, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	long int rank_line(int *v);
	void build_cubic_surface_from_lines(int len, long int *S, int *coeff,
		int verbose_level);
	int rank_of_system(int len, long int *S,
			int verbose_level);
	void create_system(int len, long int *S,
			int *&System, int &nb_rows, int verbose_level);
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
	int test_double_six_property(long int *S12, int verbose_level);
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
			data_structures::set_of_sets *&pts_on_lines,
			int *&f_is_on_line,
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
		data_structures::set_of_sets *line_intersections, int verbose_level);
	int intersection_of_four_lines_but_not_b6(int *Adj,
		int *four_lines_idx, int b6, int verbose_level);
	int intersection_of_five_lines(int *Adj, int *five_lines_idx,
		int verbose_level);
	void rearrange_lines_according_to_a_given_double_six(long int *Lines,
		long int *New_lines, long int *double_six, int verbose_level);
	void create_lines_from_plane_equations(int *The_plane_equations,
		long int *Lines, int verbose_level);
	void create_remaining_fifteen_lines(
		long int *double_six, long int *fifteen_lines,
		int verbose_level);
	long int compute_cij(long int *double_six,
		int i, int j, int verbose_level);
	int compute_transversals_of_any_four(
			long int *&Trans, int &nb_subsets,
			long int *lines, int sz, int verbose_level);
	int read_schlaefli_label(const char *p);
	void read_string_of_schlaefli_labels(std::string &str, int *&v, int &sz, int verbose_level);

	// surface_domain_io.cpp:
	void print_equation(std::ostream &ost, int *coeffs);
	void print_equation_maple(std::stringstream &ost, int *coeffs);
	void print_equation_tex(std::ostream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs);
	void print_equation_tex_lint(std::ostream &ost, long int *coeffs);
	void latex_double_six(std::ostream &ost, long int *double_six);
	void make_spreadsheet_of_lines_in_three_kinds(data_structures::spreadsheet *&Sp,
		long int *Wedge_rk, long int *Line_rk, long int *Klein_rk, int nb_lines,
		int verbose_level);
	void print_equation_in_trihedral_form(std::ostream &ost,
		int *the_six_plane_equations, int lambda, int *the_equation);
	void print_equation_wrapped(std::ostream &ost, int *the_equation);
	void print_lines_tex(std::ostream &ost, long int *Lines, int nb_lines);
	//void alice(std::ostream &ost, long int *Lines, int nb_lines);
	void print_clebsch_P(std::ostream &ost);
	void print_clebsch_P_matrix_only(std::ostream &ost);
	void print_clebsch_cubics(std::ostream &ost);
	void print_system(std::ostream &ost, int *system);
	void print_trihedral_pair_in_dual_coordinates_in_GAP(
		long int *F_planes_rank, long int *G_planes_rank);
	void print_basics(std::ostream &ost);
	void print_polynomial_domains(std::ostream &ost);
	void sstr_line_label(std::stringstream &sstr, long int pt);
	void make_table_of_surfaces(int verbose_level);
	void make_table_of_surfaces_detailed(
			int *Q_table, int Q_table_len, int verbose_level);
	void make_table_of_surfaces2(std::ostream &ost,
			int *Q_table, int Q_table_len, int verbose_level);
	void table_top(std::ostream &ost);
	void table_bottom(std::ostream &ost);
	void compute_table_E(
			int *field_orders, int nb_fields,
			long int *&Table,
			int *&Q, int &nb_Q,
			int *&E, int &nb_E_types, int verbose_level);


	// surface_domain_families.cpp:
	void create_equation_general_abcd(int a, int b, int c, int d, int *coeff, int verbose_level);
	void create_equation_Cayley_klmn(int k, int l, int m, int n, int *coeff, int verbose_level);
	void create_equation_bes(int a, int c, int *coeff, int verbose_level);
	void create_equation_F13(int a, int *coeff, int verbose_level);
	void create_equation_G13(int a, int *coeff, int verbose_level);
	surface_object *create_surface_general_abcd(int a, int b, int c, int d,
		int verbose_level);
	surface_object *create_surface_bes(int a, int c,
		int verbose_level);
	surface_object *create_surface_F13(int a, int verbose_level);
	surface_object *create_surface_G13(int a, int verbose_level);
	surface_object *create_Eckardt_surface(int a, int b,
		int &alpha, int &beta,
		int verbose_level);
	void create_equation_Eckardt_surface(int a, int b, int *coeff, int verbose_level);
	int test_Eckardt_form_alpha_beta(int *coeff, int &alpha, int &beta,
		int verbose_level);
	void create_Eckardt_double_six(long int *double_six, int a, int b,
		int verbose_level);
	void create_Eckardt_fifteen_lines(long int *fifteen_lines, int a, int b,
		int verbose_level);

};

void callback_surface_domain_sstr_line_label(std::stringstream &sstr, long int pt, void *data);

// #############################################################################
// surface_object_properties.cpp
// #############################################################################

//! properties of a particular cubic surface in PG(3,q), as defined by an object of class surface_object


class surface_object_properties {

public:

	surface_object *SO;


	data_structures::set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	int *f_is_on_line; // [SO->nb_pts]


	data_structures::set_of_sets *lines_on_point;
	data_structures::tally *Type_pts_on_lines;
	data_structures::tally *Type_lines_on_point;

	long int *Eckardt_points; // the orbiter rank of the Eckardt points
	int *Eckardt_points_index; // index into SO->Pts
	int *Eckardt_points_schlaefli_labels; // Schlaefli labels
	int *Eckardt_point_bitvector_in_Schlaefli_labeling;
		// true if the i-th Eckardt point in the Schlaefli labeling is present
	int nb_Eckardt_points;

	int *Eckardt_points_line_type; // [nb_Eckardt_points + 1]
	int *Eckardt_points_plane_type; // [SO->Surf->P->Nb_subspaces[2]]

	long int *Hesse_planes;
	int nb_Hesse_planes;
	int *Eckardt_point_Hesse_plane_incidence; // [nb_Eckardt_points * nb_Hesse_planes]


	int nb_axes;
	int *Axes_index; // [nb_axes] two times the index into trihedral pairs + 0 or +1
	long int *Axes_Eckardt_points; // [nb_axes * 3] the Eckardt points in Schlaefli labels that lie on the axes
	long int *Axes_line_rank;


	long int *Double_points;
	int *Double_points_index;
	int nb_Double_points;

	long int *Single_points;
	int *Single_points_index;
	int nb_Single_points;

	long int *Pts_not_on_lines;
	int nb_pts_not_on_lines;

	int nb_planes;
	int *plane_type_by_points;
	int *plane_type_by_lines;
	data_structures::tally *C_plane_type_by_points;

	long int *Tritangent_plane_rk; // [45]
		// list of tritangent planes in Schlaefli labeling
	int nb_tritangent_planes;

	long int *Lines_in_tritangent_planes; // [nb_tritangent_planes * 3]

	long int *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]

	long int *All_Planes; // [nb_trihedral_pairs * 6]
	int *Dual_point_ranks; // [nb_trihedral_pairs * 6]

	int *Adj_line_intersection_graph; // [SO->nb_lines * SO->nb_lines]
	data_structures::set_of_sets *Line_neighbors;
	int *Line_intersection_pt; // [SO->nb_lines * SO->nb_lines]
	int *Line_intersection_pt_idx; // [SO->nb_lines * SO->nb_lines]


	int *gradient; // [4 * SO->Surf->Poly2_4->get_nb_monomials()]

	long int *singular_pts;
	int nb_singular_pts;
	int nb_non_singular_pts;

	long int *tangent_plane_rank_global; // [SO->nb_pts]
	long int *tangent_plane_rank_dual; // [nb_non_singular_pts]

	surface_object_properties();
	~surface_object_properties();
	void init(surface_object *SO, int verbose_level);
	void compute_properties(int verbose_level);
	void compute_axes(int verbose_level);
	void compute_gradient(int verbose_level);
	void compute_singular_points_and_tangent_planes(int verbose_level);
	void compute_adjacency_matrix_of_line_intersection_graph(
		int verbose_level);
	int Adj_ij(int i, int j);
	void compute_plane_type_by_points(int verbose_level);
	void compute_tritangent_planes_by_rank(int verbose_level);
	void compute_Lines_in_tritangent_planes(int verbose_level);
	void compute_Trihedral_pairs_as_tritangent_planes(int verbose_level);

	void compute_planes_and_dual_point_ranks(int verbose_level);
	//void print_everything(std::ostream &ost, int verbose_level);
	void report_properties(std::ostream &ost, int verbose_level);
	void report_properties_simple(std::ostream &ost, int verbose_level);
	void print_line_intersection_graph(std::ostream &ost);
	void print_adjacency_list(std::ostream &ost);
	void print_adjacency_matrix(std::ostream &ost);
	void print_adjacency_matrix_with_intersection_points(
			std::ostream &ost);
	void print_neighbor_sets(std::ostream &ost);
	void print_planes_in_trihedral_pairs(std::ostream &ost);
	void print_tritangent_planes(std::ostream &ost);
	void print_single_tritangent_plane(std::ostream &ost, int plane_idx);


	void print_plane_type_by_points(std::ostream &ost);
	void print_lines(std::ostream &ost);
	void print_lines_with_points_on_them(std::ostream &ost);
	void print_equation(std::ostream &ost);
	void print_summary(std::ostream &ost);
	void print_affine_points_in_source_code(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_Eckardt_points(std::ostream &ost);
	void print_Hesse_planes(std::ostream &ost);
	void print_axes(std::ostream &ost);
	void print_singular_points(std::ostream &ost);
	void print_double_points(std::ostream &ost);
	void print_single_points(std::ostream &ost);
	void print_points_on_surface(std::ostream &ost);
	void print_all_points_on_surface(std::ostream &ost);
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

	int compute_transversal_line(int line_a, int line_b,
		int verbose_level);
	void compute_transversal_lines(
		int line_a, int line_b, int *transversals5,
		int verbose_level);
	void clebsch_map_find_arc_and_lines(long int *Clebsch_map,
		long int *Arc, long int *Blown_up_lines, int verbose_level);
	void clebsch_map_latex(std::ostream &ost,
			long int *Clebsch_map, int *Clebsch_coeff);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void latex_trihedral_pair(std::ostream &ost, int *T, long int *TE);
	void compute_reduced_set_of_points_not_on_lines_wrt_P(int P_idx, int *&f_deleted, int verbose_level);
	int test_full_del_pezzo(int P_idx, int *f_deleted, int verbose_level);
	void create_summary_file(std::string &fname,
			std::string &surface_label, std::string &col_postfix, int verbose_level);

};



// #############################################################################
// surface_object.cpp
// #############################################################################

//! a particular cubic surface in PG(3,q), given by its equation


class surface_object {

public:
	int q;
	field_theory::finite_field *F;
	surface_domain *Surf;

	long int *Pts; // in increasing order
	int nb_pts;


	long int *Lines;
	int nb_lines;

	int eqn[20];

	surface_object_properties *SOP;



	surface_object();
	~surface_object();
	void init_equation_points_and_lines_only(surface_domain *Surf, int *eqn,
		int verbose_level);
	void init_equation(surface_domain *Surf, int *eqn, int verbose_level);
	void enumerate_points(int verbose_level);
	void enumerate_points_and_lines(int verbose_level);
	void find_real_lines(std::vector<long int> &The_Lines, int verbose_level);
	void init_with_27_lines(surface_domain *Surf, long int *Lines27, int *eqn,
		int f_find_double_six_and_rearrange_lines, int verbose_level);
	void compute_properties(int verbose_level);
	void recompute_properties(int verbose_level);
	void find_double_six_and_rearrange_lines(long int *Lines, int verbose_level);
	void identify_lines(long int *lines, int nb_lines, int *line_idx,
		int verbose_level);
	void print_nine_lines_latex(std::ostream &ost, long int *nine_lines,
		int *nine_lines_idx);
	int find_point(long int P, int &idx);
	void export_something(std::string &what,
			std::string &fname_base, int verbose_level);


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




}}}





#endif /* SRC_LIB_FOUNDATIONS_ALGEBRAIC_GEOMETRY_ALGEBRAIC_GEOMETRY_H_ */
