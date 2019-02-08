// globals.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

namespace orbiter {
namespace foundations {


// #############################################################################
// galois_global.C:
// #############################################################################

void test_unipoly();
void test_unipoly2();
char *search_for_primitive_polynomial_of_given_degree(int p, 
	int degree, int verbose_level);
void search_for_primitive_polynomials(int p_min, int p_max, 
	int n_min, int n_max, int verbose_level);
void make_linear_irreducible_polynomials(int q, int &nb, int *&table, 
	int verbose_level);
void gl_random_matrix(int k, int q, int verbose_level);
void save_as_colored_graph_easy(const char *fname_base, int n, int *Adj, 
	int verbose_level);
void save_colored_graph(const char *fname, int nb_vertices, int nb_colors, 
	int *vertex_labels, int *vertex_colors, 
	int *data, int data_sz, 
	uchar *bitvector_adjacency, int bitvector_length,
	int verbose_level);
void load_colored_graph(const char *fname, int &nb_vertices, int &nb_colors, 
	int *&vertex_labels, int *&vertex_colors, 
	int *&user_data, int &user_data_size, 
	uchar *&bitvector_adjacency, int &bitvector_length,
	int verbose_level);
int is_diagonal_matrix(int *A, int n);
int is_association_scheme(int *color_graph, int n, int *&Pijk, 
	int *&colors, int &nb_colors, int verbose_level);
void print_Pijk(int *Pijk, int nb_colors);
void write_colored_graph(ofstream &ost, char *label, 
	int point_offset, 
	int nb_points, 
	int f_has_adjacency_matrix, int *Adj, 
	int f_has_adjacency_list, int *adj_list, 
	int f_has_bitvector, uchar *bitvector_adjacency, 
	int f_has_is_adjacent_callback, 
	int (*is_adjacent_callback)(int i, int j, void *data), 
	void *is_adjacent_callback_data, 
	int f_colors, int nb_colors, int *point_color, 
	int f_point_labels, int *point_label);
int str2int(string &str);
void print_longinteger_after_multiplying(ostream &ost, int *factors, int len);
void andre_preimage(projective_space *P2, projective_space *P4, 
	int *set2, int sz2, int *set4, int &sz4, int verbose_level);
void determine_conic(int q, const char *override_poly, int *input_pts, 
	int nb_pts, int verbose_level);
void compute_decomposition_of_graph_wrt_partition(int *Adj, int N, 
	int *first, int *len, int nb_parts, int *&R, int verbose_level);
void create_Levi_graph_from_incidence_matrix(colored_graph *&CG, int *M, 
	int nb_rows, int nb_cols, 
	int f_point_labels, int *point_labels, int verbose_level);


// #############################################################################
// magma_interface.C
// #############################################################################

void magma_write_permutation_group(const char *fname_base, int group_order, 
	int *Table, int *gens, int nb_gens, int verbose_level);
void magma_normalizer_in_Sym_n(const char *fname_base, int group_order, 
	int *Table, int *gens, int nb_gens, 
	int *&N_gens, int &N_nb_gens, int &N_go, int verbose_level);
void read_magma_permutation_group(const char *fname, int degree, 
	int *&gens, int &nb_gens, int &go, int verbose_level);

// #############################################################################
// numerics.C:
// #############################################################################


void double_vec_print(double *a, int len);
void double_vec_linear_combination1(double c1, double *v1,
		double *w, int len);
void double_vec_linear_combination(double c1, double *v1,
		double c2, double *v2, double *v3, int len);
void double_vec_linear_combination3(
		double c1, double *v1,
		double c2, double *v2,
		double c3, double *v3,
		double *w, int len);
void double_vec_add(double *a, double *b, double *c, int len);
void double_vec_subtract(double *a, double *b, double *c, int len);
void double_vec_scalar_multiple(double *a, double lambda, int len);
int Gauss_elimination(double *A, int m, int n, int *base_cols, 
	int f_complete, int verbose_level);
void print_system(double *A, int m, int n);
void get_kernel(double *M, int m, int n, int *base_cols, int nb_base_cols, 
	int &kernel_m, int &kernel_n, double *kernel);
int Null_space(double *M, int m, int n, double *K, int verbose_level);
// K will be k x n
// where k is the return value.
void double_vec_normalize_from_back(double *v, int len);
void double_vec_normalize_to_minus_one_from_back(double *v, int len);
int triangular_prism(double *P1, double *P2, double *P3, 
	double *abc3, double *angles3, double *T3, 
	int verbose_level);
int general_prism(double *Pts, int nb_pts, double *Pts_xy, 
	double *abc3, double *angles3, double *T3, 
	int verbose_level);
double rad2deg(double phi);
void mult_matrix(double *v, double *R, double *vR);
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
void center_of_mass(double *Pts, int len, int *Pt_idx, int nb_pts, double *c);
void plane_through_three_points(double *p1, double *p2, double *p3, 
	double *n, double &d);
void orthogonal_transformation_from_point_to_basis_vector(double *from, 
	double *A, double *Av, int verbose_level);
void output_double(double a, ostream &ost);
void mult_matrix_4x4(double *v, double *R, double *vR);
void transpose_matrix_4x4(double *A, double *At);
void substitute_quadric_linear(double *coeff_in, double *coeff_out, 
	double *A4_inv, int verbose_level);
void substitute_cubic_linear(double *coeff_in, double *coeff_out, 
	double *A4_inv, int verbose_level);
void make_transform_t_varphi_u_double(int n, double *varphi, double *u, 
	double *A, double *Av, int verbose_level);
// varphi are the dual coordinates of a plane.
// u is a vector such that varphi(u) \neq -1.
// A = I + varphi * u.
void matrix_double_inverse(double *A, double *Av, int n, int verbose_level);
int line_centered(double *pt1_in, double *pt2_in, 
	double *pt1_out, double *pt2_out, double r);

}}


