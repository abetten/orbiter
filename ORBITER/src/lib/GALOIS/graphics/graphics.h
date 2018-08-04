// graphics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// draw.C
// #############################################################################

void transform_llur(INT *in, INT *out, INT &x, INT &y);
void transform_dist(INT *in, INT *out, INT &x, INT &y);
void transform_dist_x(INT *in, INT *out, INT &x);
void transform_dist_y(INT *in, INT *out, INT &y);
void transform_llur_double(double *in, double *out, double &x, double &y);
void draw(BYTE *fname);
void on_circle_int(INT *Px, INT *Py, INT idx, INT angle_in_degree, INT rad);
void on_circle_double(double *Px, double *Py, INT idx, 
	double angle_in_degree, double rad);
void polygon3D(mp_graphics &G, INT *Px, INT *Py, INT dim, 
	INT x0, INT y0, INT z0, INT x1, INT y1, INT z1);
void integer_4pts(mp_graphics &G, INT *Px, INT *Py, 
	INT p1, INT p2, INT p3, INT p4, 
	const BYTE *align, INT a);
void text_4pts(mp_graphics &G, INT *Px, INT *Py, 
	INT p1, INT p2, INT p3, INT p4, 
	const BYTE *align, const BYTE *str);
void affine_pt1(INT *Px, INT *Py, INT p0, INT p1, INT p2, 
	double f1, INT p3);
void affine_pt2(INT *Px, INT *Py, INT p0, INT p1, INT p1b, 
	double f1, INT p2, INT p2b, double f2, INT p3);
INT C3D(INT i, INT j, INT k);
INT C2D(INT i, INT j);
double cos_grad(double phi);
double sin_grad(double phi);
double tan_grad(double phi);
double atan_grad(double x);
void adjust_coordinates_double(double *Px, double *Py, INT *Qx, INT *Qy, 
	INT N, double xmin, double ymin, double xmax, double ymax, 
	INT verbose_level);
void Intersection_of_lines(double *X, double *Y, 
	double *a, double *b, double *c, INT l1, INT l2, INT pt);
void intersection_of_lines(double a1, double b1, double c1, 
	double a2, double b2, double c2, 
	double &x, double &y);
void Line_through_points(double *X, double *Y, 
	double *a, double *b, double *c, 
	INT pt1, INT pt2, INT line_idx);
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
	INT pt0, INT pt1, INT pt2, double alpha, INT new_pt);
void draw_graph(mp_graphics *G, INT x, INT y, INT dx, INT dy, 
	INT nb_V, INT *Edges, INT nb_E);
void draw_graph_with_distinguished_edge(mp_graphics *G, INT x, INT y, 
	INT dx, INT dy, INT nb_V, INT *Edges, INT nb_E, 
	INT distinguished_edge, INT verbose_level);
void draw_graph_on_multiple_circles(mp_graphics *G, INT x, INT y, 
	INT dx, INT dy, INT nb_V, 
	INT *Edges, INT nb_E, INT nb_circles);
void draw_graph_on_2D_grid(mp_graphics *G, 
	INT x, INT y, INT dx, INT dy, 
	INT rad, INT nb_V, 
	INT *Edges, INT nb_E, INT *coords_2D, INT *Base, 
	INT f_point_labels, INT point_label_offset, INT f_directed);
void draw_tournament(mp_graphics *G, 
	INT x, INT y, INT dx, INT dy, INT nb_V, 
	INT *Edges, INT nb_E, INT verbose_level);
void draw_bitmatrix(const BYTE *fname_base, INT f_dots, 
	INT f_partition, INT nb_row_parts, INT *row_part_first, 
	INT nb_col_parts, INT *col_part_first, 
	INT f_row_grid, INT f_col_grid, 
	INT f_bitmatrix, UBYTE *D, INT *M, 
	INT m, INT n, INT xmax_in, INT ymax_in, INT xmax, INT ymax, 
	double scale, double line_width, 
	INT f_has_labels, INT *labels);
void draw_bitmatrix2(mp_graphics &G, INT f_dots, 
	INT f_partition, INT nb_row_parts, INT *row_part_first, 
	INT nb_col_parts, INT *col_part_first, 
	INT f_row_grid, INT f_col_grid, 
	INT f_bitmatrix, UBYTE *D, INT *M, 
	INT m, INT n, INT xmax, INT ymax, 
	INT f_has_labels, INT *labels);


// #############################################################################
// mp_graphics.C:
// #############################################################################


struct grid_frame {
	INT f_matrix_notation;
	double origin_x;
	double origin_y;
	INT m; // number of rows in the grid
	INT n; // number of columns in the grid
	double dx;
	double dy;
};

class mp_graphics {

	char fname_base[1000];
	char fname_mp[1000];
	char fname_log[1000];
	char fname_tikz[1000];
	ofstream fp_mp;
	ofstream fp_log;
	ofstream fp_tikz;
	INT f_file_open;
	
	
	// coordinate systems:
	
	INT user[4]; // llx/lly/urx/ury 
	INT dev[4]; // llx/lly/urx/ury 

	INT x_min, x_max, y_min, y_max, f_min_max_set;

	INT txt_halign;
		// 0=left aligned, 
		// 1=centered, 
		// 2=right aligned; 
		// default=0
	INT txt_valign; // 0=bottom, 1=middle, 2=top; default=0
	INT txt_boxed; //  default=0
	INT txt_overwrite; // default=0
	INT txt_rotate; // default = 0 (in degree)

	INT line_beg_style; // default=0
	INT line_end_style; // 0=nothing, 1=arrow; default=0

	INT line_thickness; // 1,2,3
	INT line_color; // 0=white, 1=black, 2=red, 3=green


	INT fill_interior;
		// in 1/100th, 
		// 0= none (used for pie-drawing); 
		// default=0
	INT fill_color; // 0 = white, 1 = black; default=0
	INT fill_shape; // 0 =  .., 1 = -- ; default=1
	INT fill_outline; // default=0
	INT fill_nofill; // default=0


	INT line_dashing;
		// 0 = no dashing, 
		// otherwise scaling factor 1/100th evenly
		// default=0
	
	INT cur_path;

	INT f_embedded;
		// have a header so that the file 
		// can be compiled standalone (for tikz)
	INT f_sideways;

public:
	// for tikz:
	double tikz_global_scale; // .45 works
	double tikz_global_line_width; // 1.5 works


	mp_graphics();
	mp_graphics(const char *file_name, 
		INT xmin, INT ymin, INT xmax, INT ymax, 
		INT f_embedded, INT f_sideways);
	~mp_graphics();
	void default_values();
	void init(const char *file_name, 
		INT xmin, INT ymin, INT xmax, INT ymax, 
		INT f_embedded, INT f_sideways);
	void exit(ostream &ost, INT verbose_level);
	void setup(const char *fname_base, 
		INT in_xmin, INT in_ymin, INT in_xmax, INT in_ymax, 
		INT xmax, INT ymax, INT f_embedded, INT f_sideways, 
		double scale, double line_width);
	void set_parameters(double scale, double line_width);
	void set_scale(double scale);
	void frame(double move_out);
	void frame_constant_aspect_ratio(double move_out);
	void finish(ostream &ost, INT verbose_level);

	INT& out_xmin();
	INT& out_ymin();
	INT& out_xmax();
	INT& out_ymax();

	void user2dev(INT &x, INT &y);
	void dev2user(INT &x, INT &y);
	void user2dev_dist_x(INT &x);
	void user2dev_dist_y(INT &y);

	void draw_polar_grid(double r_max, INT nb_circles, 
		INT nb_rays, double x_stretch);
	void draw_axes_and_grid(
		double x_min, double x_max, 
		double y_min, double y_max, 
		double x_stretch, double y_stretch, 
		INT f_x_axis_at_y_min, INT f_y_axis_at_x_min, 
		INT x_mod, INT y_mod, INT x_tick_mod, INT y_tick_mod, 
		double x_labels_offset, double y_labels_offset, 
		double x_tick_half_width, double y_tick_half_width, 
		INT f_v_lines, INT subdivide_v, 
		INT f_h_lines, INT subdivide_h);
	void plot_curve(INT N, INT *f_DNE, 
		double *Dx, double *Dy, double dx, double dy);
	void nice_circle(INT x, INT y, INT rad);
	void grid_polygon2(grid_frame *F, INT x0, INT y0, 
		INT x1, INT y1);
	void grid_polygon4(grid_frame *F, INT x0, INT y0, 
		INT x1, INT y1, INT x2, INT y2, INT x3, INT y3);
	void grid_polygon5(grid_frame *F, INT x0, INT y0, 
		INT x1, INT y1, INT x2, INT y2, 
		INT x3, INT y3, INT x4, INT y4);
	void polygon(INT *Px, INT *Py, INT n);
	void polygon2(INT *Px, INT *Py, INT i1, INT i2);
	void polygon3(INT *Px, INT *Py, INT i1, INT i2, INT i3);
	void polygon4(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4);
	void polygon5(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5);
	void polygon6(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6);
	void polygon7(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7);
	void polygon8(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8);
	void polygon9(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, INT i9);
	void polygon10(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, INT i9, 
		INT i10);
	void polygon11(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, INT i9, 
		INT i10, INT i11);
	void polygon_idx(INT *Px, INT *Py, INT *Idx, INT n);
	void bezier(INT *Px, INT *Py, INT n);
	void bezier2(INT *Px, INT *Py, INT i1, INT i2);
	void bezier3(INT *Px, INT *Py, INT i1, INT i2, INT i3);
	void bezier4(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4);
	void bezier5(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5);
	void bezier6(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6);
	void bezier7(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7);
	void bezier_idx(INT *Px, INT *Py, INT *Idx, INT n);
	void grid_fill_polygon4(grid_frame *F, 
		INT x0, INT y0, INT x1, INT y1, INT x2, 
		INT y2, INT x3, INT y3);
	void grid_fill_polygon5(grid_frame *F, 
		INT x0, INT y0, INT x1, INT y1, 
		INT x2, INT y2, INT x3, INT y3, 
		INT x4, INT y4);
	void fill_polygon3(INT *Px, INT *Py, INT i1, INT i2, INT i3);
	void fill_polygon4(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4);
	void fill_polygon5(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5);
	void fill_polygon6(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6);
	void fill_polygon7(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7);
	void fill_polygon8(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8);
	void fill_polygon9(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, INT i9);
	void fill_polygon10(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, INT i9, INT i10);
	void fill_polygon11(INT *Px, INT *Py, INT i1, INT i2, INT i3, 
		INT i4, INT i5, INT i6, INT i7, INT i8, 
		INT i9, INT i10, INT i11);
	void polygon2_arrow_halfway(INT *Px, INT *Py, INT i1, INT i2);
	void polygon2_arrow_halfway_and_label(INT *Px, INT *Py, INT i1, INT i2, 
		const BYTE *alignment, const BYTE *txt);
	void grid_aligned_text(grid_frame *F, INT x, INT y, 
		const char *alignment, const char *p);
	void aligned_text(INT x, INT y, const char *alignment, const char *p);
	void aligned_text_array(INT *Px, INT *Py, INT idx, 
		const char *alignment, const char *p);
	void aligned_text_with_offset(INT x, INT y, INT xoffset, INT yoffset, 
		const char *alignment, const char *p);

	void st_alignment(INT txt_halign, INT txt_valign);
	void sl_udsty(INT line_dashing);
	void sl_ends(INT line_beg_style, INT line_end_style);
	void sl_thickness(INT line_thickness);
	void sl_color(INT line_color);
	void sf_interior(INT fill_interior);
	void sf_color(INT fill_color);
	void sf_shape(INT fill_shape);
	void sf_outline(INT fill_outline);
	void sf_nofill(INT fill_nofill);
	void st_boxed(INT txt_boxed);
	void st_overwrite(INT txt_overwrite);
	void st_rotate(INT txt_rotate);
	void coords_min_max(INT x, INT y);

	// output commands:
	void header();
	void footer();
	void begin_figure(INT factor_1000);
	void end_figure();

	void comment(const BYTE *p);
	void text(INT x, INT y, const char *p);
	void circle(INT x, INT y, INT rad);
	void circle_text(INT x, INT y, INT rad, const char *text);
	void polygon_or_bezier_idx(INT *Px, INT *Py, INT *Idx, INT n, 
		const char *symbol, INT f_cycle);
	void fill_idx(INT *Px, INT *Py, INT *Idx, INT n, 
		const char *symbol, INT f_cycle);


	// output commands log file:
	void header_log(BYTE *str_date);
	void footer_log();
	void comment_log(const BYTE *p);
	void st_alignment_log();
	void sl_udsty_log();
	void sl_ends_log();
	void sl_thickness_log();
	void sl_color_log();
	void sf_interior_log();
	void sf_color_log();
	void sf_shape_log();
	void sf_outline_log();
	void sf_nofill_log();
	void st_boxed_log();
	void st_overwrite_log();
	void st_rotate_log();
	void bezier_idx_log(INT *Px, INT *Py, INT *Idx, INT n);
	void polygon_log(INT *Px, INT *Py, INT n);
	void polygon_idx_log(INT *Px, INT *Py, INT *Idx, INT n);
	void text_log(INT x1, INT y1, const char *p);
	void circle_log(INT x1, INT y1, INT rad);


	// output commands metapost:
	void header_mp(BYTE *str_date);
	void footer_mp();
	void comment_mp(const BYTE *p);
	void text_mp(INT x1, INT y1, const char *p);
	void begin_figure_mp(INT factor_1000);
	void end_figure_mp();
	void circle_mp(INT x, INT y, INT rad);
	void output_circle_text_mp(INT x, INT y, INT idx, const char *text);
	void polygon_or_bezier_idx_mp(INT *Px, INT *Py, INT *Idx, 
		INT n, const char *symbol, INT f_cycle);
	void color_tikz(ofstream &fp, INT color);
	void fill_idx_mp(INT *Px, INT *Py, INT *Idx, INT n, 
		const char *symbol, INT f_cycle);
	void output_xy_metapost(INT x, INT y);
	void output_x_metapost(INT x);
	void output_y_metapost(INT y);
	INT get_label(INT x, INT y);
	void get_alignment_mp(BYTE *align);
	void line_thickness_mp();

	// output commands tikz:
	void header_tikz(BYTE *str_date);
	void footer_tikz();
	void comment_tikz(const BYTE *p);
	void text_tikz(INT x1, INT y1, const char *p);
	void circle_tikz(INT x, INT y, INT rad);
	void output_circle_text_tikz(INT x, INT y, INT idx, INT rad, 
		const char *text);
	void polygon_or_bezier_idx_tikz(INT *Px, INT *Py, INT *Idx, INT n, 
		const char *symbol, INT f_cycle);
	void fill_idx_tikz(ofstream &fp, INT *Px, INT *Py, INT *Idx, INT n, 
		const char *symbol, INT f_cycle);
	void output_xy_tikz(INT x, INT y);
	void output_x_tikz(INT x);
	void output_y_tikz(INT y);




};


// #############################################################################
// plot.C:
// #############################################################################

void draw_density(BYTE *prefix, INT *the_set, INT set_size,
	INT f_title, const BYTE *title, INT out_of, 
	const BYTE *label_x, 
	INT f_circle, INT circle_at, INT circle_rad, 
	INT f_mu, INT f_sigma, INT nb_standard_deviations, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid, 
	INT xmax, INT ymax, INT offset_x, 
	INT f_switch_x, INT no, INT f_embedded, 
	INT verbose_level);
void draw_density_multiple_curves(BYTE *prefix,
	INT **Data, INT *Data_size, INT nb_data_sets, 
	INT f_title, const BYTE *title, INT out_of, 
	const BYTE *label_x, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid, 
	INT xmax, INT ymax, INT offset_x, INT f_switch_x, 
	INT f_v_logarithmic, double log_base, INT no, INT f_embedded, 
	INT verbose_level);
void draw_density2(mp_graphics &G, INT no, 
	INT *outline_value, INT *outline_number, INT outline_sz, 
	INT min_value, INT max_value, INT offset_x, INT f_switch_x, 
	INT f_title, const BYTE *title, 
	const BYTE *label_x, 
	INT f_circle, INT circle_at, INT circle_rad, 
	INT f_mu, INT f_sigma, INT nb_standard_deviations, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid);
void draw_density2_multiple_curves(mp_graphics &G, INT no, 
	INT **outline_value, INT **outline_number, 
	INT *outline_sz, INT nb_curves, 
	INT min_x, INT max_x, INT min_y, INT max_y, 
	INT offset_x, INT f_switch_x, 
	INT f_title, const BYTE *title, 
	const BYTE *label_x, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid, 
	INT f_v_logarithmic, double log_base);
void read_numbers_from_file(const BYTE *fname, 
	INT *&the_set, INT &set_size, INT verbose_level);
void get_coord(INT *Px, INT *Py, INT idx, INT x, INT y, 
	INT min_x, INT min_y, INT max_x, INT max_y, INT f_switch_x);
void get_coord_log(INT *Px, INT *Py, INT idx, INT x, INT y, 
	INT min_x, INT min_y, INT max_x, INT max_y, 
	double log_base, INT f_switch_x);
void y_to_pt_on_curve(INT y_in, INT &x, INT &y,  
	INT *outline_value, INT *outline_number, INT outline_sz);
void projective_plane_draw_grid(const char *fname, INT xmax, INT ymax, 
	INT f_with_points, INT rad, 
	INT q, INT *Table, INT nb, 
	INT f_point_labels, BYTE **Point_labels, 
	INT f_embedded, INT f_sideways, 
	INT verbose_level);
void projective_plane_draw_grid2(mp_graphics &G, INT q, INT *Table, 
	INT nb, INT f_with_points, INT rad, 
	INT f_point_labels, BYTE **Point_labels, INT verbose_level);
void projective_plane_make_affine_point(INT q, INT x1, INT x2, INT x3, 
	double &a, double &b);

// #############################################################################
// scene.C
// #############################################################################

#define SCENE_MAX_LINES 100000
#define SCENE_MAX_EDGES 100000
#define SCENE_MAX_POINTS 100000
#define SCENE_MAX_PLANES 10000
#define SCENE_MAX_QUADRICS 10000
#define SCENE_MAX_CUBICS 10000
#define SCENE_MAX_FACES 10000

// #############################################################################
// scene.C:
// #############################################################################


class scene {
public:
	
	INT nb_lines;
	double *Line_coords;
		// [nb_lines * 6] a line is given by two points
	
	INT nb_edges;
	INT *Edge_points;
		// [nb_edges * 2]

	INT nb_points;
	double *Point_coords;
		// [nb_points * 3]

	INT nb_planes;
	double *Plane_coords;
		// [nb_planes * 4]

	INT nb_quadrics;
	double *Quadric_coords;
		// [nb_quadrics * 10]

	INT nb_cubics;
	double *Cubic_coords;
		// [nb_cubics * 20]

	INT nb_faces;
	INT *Nb_face_points; // [nb_faces]
	INT **Face_points; // [nb_faces]


	
	void *extra_data;


	scene();
	~scene();
	void null();
	void freeself();
	void init(INT verbose_level);
	scene *transformed_copy(double *A4, double *A4_inv, 
		double rad, INT verbose_level);
	void print();
	void transform_lines(scene *S, double *A4, double *A4_inv, 
		double rad, INT verbose_level);
	void copy_edges(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	void transform_points(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	void transform_planes(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	void transform_quadrics(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	void transform_cubics(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	void copy_faces(scene *S, double *A4, double *A4_inv, 
		INT verbose_level);
	INT line_pt_and_dir(double *x6, double rad);
	INT line6(double *x6);
	INT line(double x1, double x2, double x3, 
		double y1, double y2, double y3);
	INT line_through_two_points(INT pt1, INT pt2, 
		double rad);
	INT edge(INT pt1, INT pt2);
	void points(double *Coords, INT nb_points);
	INT point(double x1, double x2, double x3);
	INT point_center_of_mass_of_face(INT face_idx);
	INT point_center_of_mass_of_edge(INT edge_idx);
	INT point_center_of_mass(INT *Pt_idx, INT nb_pts);
	INT triangle(INT line1, INT line2, INT line3, INT verbose_level);
	INT point_as_intersection_of_two_lines(INT line1, INT line2);
	INT plane_from_dual_coordinates(double *x4);
	INT plane(double x1, double x2, double x3, double a);
		// A plane is called a polynomial shape because 
		// it is defined by a first order polynomial equation. 
		// Given a plane: plane { <A, B, C>, D }
		// it can be represented by the equation 
		// A*x + B*y + C*z - D*sqrt(A^2 + B^2 + C^2) = 0.
		// see http://www.povray.org/documentation/view/3.6.1/297/
	INT plane_through_three_points(INT pt1, INT pt2, INT pt3);
	INT quadric_through_three_lines(INT line_idx1, 
		INT line_idx2, INT line_idx3, INT verbose_level);
	INT quadric(double *coeff);
	// povray ordering of monomials:
	// http://www.povray.org/documentation/view/3.6.1/298/
	// 1: x^2
	// 2: xy
	// 3: xz
	// 4: x
	// 5: y^2
	// 6: yz
	// 7: y
	// 8: z^2
	// 9: z
	// 10: 1
	INT cubic(double *coeff);
	// povray ordering of monomials:
	// http://www.povray.org/documentation/view/3.6.1/298/
	// 1: x^3
	// 2: x^2y
	// 3: x^2z
	// 4: x^2
	// 5: xy^2
	// 6: xyz
	// 7: xy
	// 8: xz^2
	// 9: xz
	// 10: x
	// 11: y^3
	// 12: y^2z
	// 13: y^2
	// 14: yz^2
	// 15: yz
	// 16: y
	// 17: z^3
	// 18: z^2
	// 19: z
	// 20: 1
	INT face(INT *pts, INT nb_pts);
	INT face3(INT pt1, INT pt2, INT pt3);
	INT face4(INT pt1, INT pt2, INT pt3, INT pt4);
	INT face5(INT pt1, INT pt2, INT pt3, INT pt4, INT pt5);
	void draw_lines_with_selection(INT *selection, INT nb_select, 
		double r, const BYTE *options, ostream &ost);
	void draw_line_with_selection(INT line_idx, 
		double r, const BYTE *options, ostream &ost);
	void draw_lines_cij_with_selection(INT *selection, INT nb_select, 
		ostream &ost);
	void draw_lines_cij(ostream &ost);
	void draw_lines_ai_with_selection(INT *selection, INT nb_select, 
		ostream &ost);
	void draw_lines_ai(ostream &ost);
	void draw_lines_bj_with_selection(INT *selection, INT nb_select, 
		ostream &ost);
	void draw_lines_bj(ostream &ost);
	void draw_edges_with_selection(INT *selection, INT nb_select, 
		double rad, const BYTE *options, ostream &ost);
	void draw_faces_with_selection(INT *selection, INT nb_select, 
		double thickness_half, const BYTE *options, ostream &ost);
	void draw_face(INT idx, double thickness_half, const BYTE *options, 
		ostream &ost);
	void draw_text(const BYTE *text, double thickness_half, double extra_spacing, 
			double scale, 
			double off_x, double off_y, double off_z, 
			const BYTE *color_options, 
			double x, double y, double z, 
			double up_x, double up_y, double up_z, 
			double view_x, double view_y, double view_z, 
			ostream &ost, INT verbose_level);
	void draw_planes_with_selection(INT *selection, INT nb_select, 
		const BYTE *options, ostream &ost);
	void draw_points_with_selection(INT *selection, INT nb_select, 
		double rad, const BYTE *options, ostream &ost);
	void draw_cubic_with_selection(INT *selection, INT nb_select, 
		const BYTE *options, ostream &ost);
	void draw_quadric_with_selection(INT *selection, INT nb_select, 
		const BYTE *options, ostream &ost);
	INT intersect_line_and_plane(INT line_idx, INT plane_idx, 
		INT &intersection_point_idx, 
		INT verbose_level);
	INT intersect_line_and_line(INT line1_idx, INT line2_idx, 
		double &lambda, 
		INT verbose_level);
#if 0
	INT line_centered(double *pt1_in, double *pt2_in, 
		double *pt1_out, double *pt2_out, 
		double r);
#endif
	INT line_extended(double x1, double x2, double x3, 
		double y1, double y2, double y3, 
		double r);
	void map_a_line(INT line1, INT line2, 
		INT plane_idx, INT line_idx, double spread, 
		INT nb_pts, 
		INT *New_line_idx, INT &nb_new_lines, 
		INT *New_pt_idx, INT &nb_new_points, INT verbose_level);
	INT map_a_point(INT line1, INT line2, 
		INT plane_idx, double pt_in[3], 
		INT &new_line_idx, INT &new_pt_idx, 
		INT verbose_level);
	void lines_a();
	void lines_b();
	void lines_cij();
	void Eckardt_points();
	void fourD_cube(double rad_desired);
	void rescale(INT first_pt_idx, double rad_desired);
	double euclidean_distance(INT pt1, INT pt2);
	double distance_from_origin(INT pt);
	void fourD_cube_edges(INT first_pt_idx);
	void hypercube(INT n, double rad_desired);
	void Dodecahedron_points();
	void Dodecahedron_edges(INT first_pt_idx);
	void Dodecahedron_planes(INT first_pt_idx);
	void tritangent_planes();
	void clebsch_cubic();
	double distance_between_two_points(INT pt1, INT pt2);
	void create_five_plus_one();
	void create_Hilbert_model();

};


// #############################################################################
// tree.C:
// #############################################################################

class tree {

public:

	tree_node *root;
	
	INT nb_nodes;
	INT max_depth;
	
	INT *path;

	INT f_count_leaves;
	INT leaf_count;

	tree();
	~tree();
	void init(const BYTE *fname, INT xmax, INT ymax, INT verbose_level);
	void draw(char *fname, INT xmax_in, INT ymax_in, INT xmax, INT ymax, 
		INT rad, 
		INT f_circle, INT f_circletext, INT f_i, INT f_edge_labels, 
		INT f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, 
			INT *v, INT layer, tree_node *N, 
			INT x, INT y, INT dx, INT dy), 
		INT f_embedded, INT f_sideways, INT f_on_circle, 
		double tikz_global_scale, double tikz_global_line_width
		);
	void circle_center_and_radii(INT xmax, INT ymax, INT max_depth, 
		INT &x0, INT &y0, INT *&rad);
	void compute_DFS_ranks(INT &nb_nodes, INT verbose_level);
};

// #############################################################################
// tree_node.C:
// #############################################################################

class tree_node {

public:
	tree_node *parent;
	INT depth;
	INT f_value;
	INT value;
	
	INT f_int_data;
	INT int_data;
	BYTE *char_data;
	INT nb_children;
	tree_node **children;

	INT weight;
	INT placement_x;
	INT placement_y;
	INT width;

	INT DFS_rank;

	tree_node();
	~tree_node();
	void init(INT depth, tree_node *parent, INT f_value, INT value, 
		INT f_i_data, INT i_data, BYTE *c_data, INT verbose_level);
	void print_path();
	void print_depth_first();
	void compute_DFS_rank(INT &rk);
	void get_coordinates(INT &idx, INT *coord_xy);
	void get_coordinates_and_width(INT &idx, INT *coord_xyw);
	void calc_weight();
	void place_xy(INT left, INT right, INT ymax, INT max_depth);
	void place_on_circle(INT xmax, INT ymax, INT max_depth);
	void add_node(INT l, INT depth, INT *path, INT i_data, BYTE *c_data, 
		INT verbose_level);
	INT find_child(INT val);
	void get_values(INT *v);
	void draw_edges(mp_graphics &G, INT rad, INT f_circle, 
		INT f_circletext, INT f_i, 
		INT f_has_parent, INT parent_x, INT parent_y, 
		INT max_depth, INT f_edge_labels, 
		INT f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, INT *v, 
			INT layer, tree_node *N, INT x, INT y, INT dx, INT dy),
		tree *T
		);
	void draw_vertices(mp_graphics &G, INT rad, INT f_circle, 
		INT f_circletext, INT f_i, 
		INT f_has_parent, INT parent_x, INT parent_y, INT max_depth, 
		INT f_edge_labels, 
		INT f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, INT *v, 
			INT layer, tree_node *N, INT x, INT y, INT dx, INT dy),
		tree *T
		);
	void draw_sideways(mp_graphics &G, INT f_circletext, INT f_i, 
		INT f_has_parent, INT parent_x, INT parent_y, 
		INT max_depth, INT f_edge_labels);
};

INT tree_node_calc_y_coordinate(INT ymax, INT l, INT max_depth);





