// globals.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_




namespace orbiter {
namespace layer1_foundations {


// #############################################################################
// function_command.cpp
// #############################################################################

//! an individual command which is part of a function expressed in reverse polish notation

class function_command {
public:

	int type;
	// 1 = push labeled constant
	// 2 = push immediate constant
	// 3 = push variable
	// 4 = store variable
	// 5 = mult
	// 6 = add
	// 7 = cos
	// 8 = sin
	// 9 = return
	// 10 = sqrt

	int f_has_argument;
	int arg;
	double val; // for push immediate constant

	function_command();
	~function_command();
	void init_with_argument(int type, int arg);
	void init_push_immediate_constant(double val);
	void init_simple(int type);

};

// #############################################################################
// function_polish_description.cpp
// #############################################################################

//! description of a function in reverse polish notation from the command line


class function_polish_description {
public:
	int nb_constants;
	std::vector<std::string> const_names;
	std::vector<std::string> const_values;
	int nb_variables;
	std::vector<std::string> variable_names;
	int code_sz;
	std::vector<std::string> code;

	function_polish_description();
	~function_polish_description();
	void null();
	void freeself();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
};


// #############################################################################
// function_polish.cpp
// #############################################################################

//! a set of functions in reverse polish notation

class function_polish {
public:

	function_polish_description *Descr;

	std::vector<std::string > Variables;

	std::vector<std::pair<std::string, double> > Constants;

	std::vector<int> Entry;
	std::vector<int> Len;

	std::vector<function_command> Code;


	function_polish();
	~function_polish();
	void init(
			function_polish_description *Descr,
			int verbose_level);
	void print_code_complete(
			int verbose_level);
	void print_code(
			int i0,  int len,
			int verbose_level);
	void evaluate(
			double *variable_values,
			double *output_values,
			int verbose_level);

};



// #############################################################################
// numerics.cpp
// #############################################################################



//! numerical functions, mostly concerned with double

class numerics {
public:
	numerics();
	~numerics();
	void vec_print(double *a, int len);
	void vec_linear_combination1(double c1, double *v1,
			double *w, int len);
	void vec_linear_combination(double c1, double *v1,
			double c2, double *v2, double *v3, int len);
	void vec_linear_combination3(
			double c1, double *v1,
			double c2, double *v2,
			double c3, double *v3,
			double *w, int len);
	void vec_add(double *a, double *b, double *c, int len);
	void vec_subtract(double *a, double *b, double *c, int len);
	void vec_scalar_multiple(double *a, double lambda, int len);
	int Gauss_elimination(double *A, int m, int n,
		int *base_cols, int f_complete,
		int verbose_level);
	void print_system(double *A, int m, int n);
	void get_kernel(double *M, int m, int n,
		int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n,
		double *kernel);
	// kernel must point to the appropriate amount of memory!
	// (at least n * (n - nb_base_cols) doubles)
	// m is not used!
	int Null_space(double *M, int m, int n, double *K,
		int verbose_level);
	// K will be k x n
	// where k is the return value.
	void vec_normalize_from_back(double *v, int len);
	void vec_normalize_to_minus_one_from_back(double *v, int len);
	int triangular_prism(double *P1, double *P2, double *P3,
		double *abc3, double *angles3, double *T3,
		int verbose_level);
	int general_prism(double *Pts, int nb_pts, double *Pts_xy,
		double *abc3, double *angles3, double *T3,
		int verbose_level);
	void mult_matrix(double *v, double *R, double *vR);
	void mult_matrix_matrix(
			double *A, double *B, double *C, int m, int n, int o);
	// A is m x n, B is n x o, C is m x o
	void print_matrix(double *R);
	void make_Rz(double *R, double phi);
	void make_Ry(double *R, double psi);
	void make_Rx(double *R, double chi);
	double atan_xy(double x, double y);
	double dot_product(double *u, double *v, int len);
	void cross_product(double *u, double *v, double *n);
	double distance_euclidean(double *x, double *y, int len);
	double distance_from_origin(double x1, double x2, double x3);
	double distance_from_origin(double *x, int len);
	void make_unit_vector(double *v, int len);
	void center_of_mass(double *Pts, int len,
			int *Pt_idx, int nb_pts, double *c);
	void plane_through_three_points(double *p1, double *p2, double *p3,
		double *n, double &d);
	void orthogonal_transformation_from_point_to_basis_vector(
		double *from,
		double *A, double *Av, int verbose_level);
	void output_double(double a, std::ostream &ost);
	void mult_matrix_4x4(double *v, double *R, double *vR);
	void transpose_matrix_4x4(double *A, double *At);
	void transpose_matrix_nxn(double *A, double *At, int n);
	void substitute_quadric_linear(double *coeff_in, double *coeff_out,
		double *A4_inv, int verbose_level);
	void substitute_cubic_linear_using_povray_ordering(
			double *coeff_in, double *coeff_out,
			double *A4_inv, int verbose_level);
	void substitute_quartic_linear_using_povray_ordering(
		double *coeff_in, double *coeff_out,
		double *A4_inv, int verbose_level);
	void make_transform_t_varphi_u_double(int n, double *varphi, double *u,
		double *A, double *Av, int verbose_level);
	// varphi are the dual coordinates of a plane.
	// u is a vector such that varphi(u) \neq -1.
	// A = I + varphi * u.
	void matrix_double_inverse(double *A, double *Av, int n, int verbose_level);
	int line_centered(double *pt1_in, double *pt2_in,
		double *pt1_out, double *pt2_out, double r, int verbose_level);
	int line_centered_tolerant(double *pt1_in, double *pt2_in,
		double *pt1_out, double *pt2_out, double r, int verbose_level);
	int sign_of(double a);
	void eigenvalues(double *A, int n, double *lambda, int verbose_level);
	void eigenvectors(double *A, double *Basis,
			int n, double *lambda, int verbose_level);
	double rad2deg(double phi);
	void vec_copy(double *from, double *to, int len);
	void vec_swap(double *from, double *to, int len);
	void vec_print(std::ostream &ost, double *v, int len);
	void vec_scan(const char *s, double *&v, int &len);
	void vec_scan(std::string &s, double *&v, int &len);
	void vec_scan_from_stream(std::istream & is, double *&v, int &len);


	double cos_grad(double phi);
	double sin_grad(double phi);
	double tan_grad(double phi);
	double atan_grad(double x);
	void adjust_coordinates_double(double *Px, double *Py, int *Qx, int *Qy,
		int N, double xmin, double ymin, double xmax, double ymax,
		int verbose_level);
	void Intersection_of_lines(double *X, double *Y,
		double *a, double *b, double *c, int l1, int l2, int pt);
	void intersection_of_lines(double a1, double b1, double c1,
		double a2, double b2, double c2,
		double &x, double &y);
	void Line_through_points(double *X, double *Y,
		double *a, double *b, double *c,
		int pt1, int pt2, int line_idx);
	void line_through_points(double pt1_x, double pt1_y,
		double pt2_x, double pt2_y, double &a, double &b, double &c);
	void intersect_circle_line_through(double rad, double x0, double y0,
		double pt1_x, double pt1_y,
		double pt2_x, double pt2_y,
		double &x1, double &y1, double &x2, double &y2);
	void intersect_circle_line(double rad, double x0, double y0,
		double a, double b, double c,
		double &x1, double &y1, double &x2, double &y2);
	void affine_combination(double *X, double *Y,
		int pt0, int pt1, int pt2, double alpha, int new_pt);
	void on_circle_double(double *Px, double *Py, int idx,
		double angle_in_degree, double rad);
	void affine_pt1(int *Px, int *Py, int p0, int p1, int p2,
		double f1, int p3);
	void affine_pt2(int *Px, int *Py, int p0, int p1, int p1b,
		double f1, int p2, int p2b, double f2, int p3);
	double norm_of_vector_2D(int x1, int x2, int y1, int y2);
	void transform_llur(int *in, int *out, int &x, int &y);
	void transform_dist(int *in, int *out, int &x, int &y);
	void transform_dist_x(int *in, int *out, int &x);
	void transform_dist_y(int *in, int *out, int &y);
	void transform_llur_double(double *in, double *out, double &x, double &y);
	void on_circle_int(int *Px, int *Py, int idx, int angle_in_degree, int rad);
	double power_of(double x, int n);
	double bernoulli(double p, int n, int k);
	void local_coordinates_wrt_triangle(double *pt,
			double *triangle_points, double &x, double &y,
			int verbose_level);
	int intersect_line_and_line(
			double *line1_pt1_coords, 	double *line1_pt2_coords,
			double *line2_pt1_coords, 	double *line2_pt2_coords,
			double &lambda,
			double *pt_coords,
			int verbose_level);
	void clebsch_map_up(
			double *line1_pt1_coords, 	double *line1_pt2_coords,
			double *line2_pt1_coords, 	double *line2_pt2_coords,
		double *pt_in, double *pt_out,
		double *Cubic_coords_povray_ordering,
		int line1_idx, int line2_idx,
		int verbose_level);
	void project_to_disc(int f_projection_on, int f_transition, int step, int nb_steps,
		double rad, double height, double x, double y, double &xp, double &yp);

};





}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_ */



