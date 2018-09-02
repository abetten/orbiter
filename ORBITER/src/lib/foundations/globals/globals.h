// globals.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// galois_global.C:
// #############################################################################

void test_unipoly();
void test_unipoly2();
char *search_for_primitive_polynomial_of_given_degree(INT p, 
	INT degree, INT verbose_level);
void search_for_primitive_polynomials(INT p_min, INT p_max, 
	INT n_min, INT n_max, INT verbose_level);
void make_linear_irreducible_polynomials(INT q, INT &nb, INT *&table, 
	INT verbose_level);
void gl_random_matrix(INT k, INT q, INT verbose_level);
void save_as_colored_graph_easy(const char *fname_base, INT n, INT *Adj, 
	INT verbose_level);
void save_colored_graph(const char *fname, INT nb_vertices, INT nb_colors, 
	INT *vertex_labels, INT *vertex_colors, 
	INT *data, INT data_sz, 
	uchar *bitvector_adjacency, INT bitvector_length,
	INT verbose_level);
void load_colored_graph(const char *fname, INT &nb_vertices, INT &nb_colors, 
	INT *&vertex_labels, INT *&vertex_colors, 
	INT *&user_data, INT &user_data_size, 
	uchar *&bitvector_adjacency, INT &bitvector_length,
	INT verbose_level);
INT is_diagonal_matrix(INT *A, INT n);
INT is_association_scheme(INT *color_graph, INT n, INT *&Pijk, 
	INT *&colors, INT &nb_colors, INT verbose_level);
void print_Pijk(INT *Pijk, INT nb_colors);
void write_colored_graph(ofstream &ost, char *label, 
	INT point_offset, 
	INT nb_points, 
	INT f_has_adjacency_matrix, INT *Adj, 
	INT f_has_adjacency_list, INT *adj_list, 
	INT f_has_bitvector, uchar *bitvector_adjacency, 
	INT f_has_is_adjacent_callback, 
	INT (*is_adjacent_callback)(INT i, INT j, void *data), 
	void *is_adjacent_callback_data, 
	INT f_colors, INT nb_colors, INT *point_color, 
	INT f_point_labels, INT *point_label);
int str2int(string &str);
void print_longinteger_after_multiplying(ostream &ost, INT *factors, INT len);
void andre_preimage(projective_space *P2, projective_space *P4, 
	INT *set2, INT sz2, INT *set4, INT &sz4, INT verbose_level);
void determine_conic(INT q, const char *override_poly, INT *input_pts, 
	INT nb_pts, INT verbose_level);
void compute_decomposition_of_graph_wrt_partition(INT *Adj, INT N, 
	INT *first, INT *len, INT nb_parts, INT *&R, INT verbose_level);
void INT_vec_print_classified(INT *v, INT len);
void create_Levi_graph_from_incidence_matrix(colored_graph *&CG, INT *M, 
	INT nb_rows, INT nb_cols, 
	INT f_point_labels, INT *point_labels, INT verbose_level);


// #############################################################################
// magma_interface.C
// #############################################################################

void magma_write_permutation_group(const char *fname_base, INT group_order, 
	INT *Table, INT *gens, INT nb_gens, INT verbose_level);
void magma_normalizer_in_Sym_n(const char *fname_base, INT group_order, 
	INT *Table, INT *gens, INT nb_gens, 
	INT *&N_gens, INT &N_nb_gens, INT &N_go, INT verbose_level);
void read_magma_permutation_group(const char *fname, INT degree, 
	INT *&gens, INT &nb_gens, INT &go, INT verbose_level);

// #############################################################################
// numerics.C:
// #############################################################################


void double_vec_print(double *a, INT len);
void double_vec_add(double *a, double *b, double *c, INT len);
void double_vec_subtract(double *a, double *b, double *c, INT len);
void double_vec_scalar_multiple(double *a, double lambda, INT len);
INT Gauss_elimination(double *A, INT m, INT n, INT *base_cols, 
	INT f_complete, INT verbose_level);
void print_system(double *A, INT m, INT n);
void get_kernel(double *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, double *kernel);
INT Null_space(double *M, INT m, INT n, double *K, INT verbose_level);
// K will be k x n
// where k is the return value.
void double_vec_normalize_from_back(double *v, INT len);
void double_vec_normalize_to_minus_one_from_back(double *v, INT len);
INT triangular_prism(double *P1, double *P2, double *P3, 
	double *abc3, double *angles3, double *T3, 
	INT verbose_level);
INT general_prism(double *Pts, INT nb_pts, double *Pts_xy, 
	double *abc3, double *angles3, double *T3, 
	INT verbose_level);
double rad2deg(double phi);
void mult_matrix(double *v, double *R, double *vR);
void print_matrix(double *R);
void make_Rz(double *R, double phi);
void make_Ry(double *R, double psi);
void make_Rx(double *R, double chi);
double atan_xy(double x, double y);
double dot_product(double *u, double *v, INT len);
void cross_product(double *u, double *v, double *n);
double distance_euclidean(double *x, double *y, INT len);
double distance_from_origin(double x1, double x2, double x3);
double distance_from_origin(double *x, INT len);
void make_unit_vector(double *v, INT len);
void center_of_mass(double *Pts, INT len, INT *Pt_idx, INT nb_pts, double *c);
void plane_through_three_points(double *p1, double *p2, double *p3, 
	double *n, double &d);
void orthogonal_transformation_from_point_to_basis_vector(double *from, 
	double *A, double *Av, INT verbose_level);
void output_double(double a, ostream &ost);
void mult_matrix_4x4(double *v, double *R, double *vR);
void transpose_matrix_4x4(double *A, double *At);
void substitute_quadric_linear(double *coeff_in, double *coeff_out, 
	double *A4_inv, INT verbose_level);
void substitute_cubic_linear(double *coeff_in, double *coeff_out, 
	double *A4_inv, INT verbose_level);
void make_transform_t_varphi_u_double(INT n, double *varphi, double *u, 
	double *A, double *Av, INT verbose_level);
// varphi are the dual coordinates of a plane.
// u is a vector such that varphi(u) \neq -1.
// A = I + varphi * u.
void matrix_double_inverse(double *A, double *Av, INT n, INT verbose_level);
INT line_centered(double *pt1_in, double *pt2_in, 
	double *pt1_out, double *pt2_out, double r);


