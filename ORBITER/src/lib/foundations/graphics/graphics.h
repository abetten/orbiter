// graphics.h
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
// animate.cpp
// #############################################################################


class animate {
public:
	scene *S;
	const char *output_mask;
	char fname_makefile[1000];
	int nb_frames;
	video_draw_options *Opt;
	std::ofstream *fpm;
	void (*draw_frame_callback)(animate *A, int frame,
					int nb_frames_this_round, int round,
					std::ostream &fp,
					int verbose_level);

	animate();
	~animate();
	void init(scene *S,
			const char *output_mask,
			int nb_frames,
			video_draw_options *Opt,
			int verbose_level);
	void animate_one_round(
		int round,
		int verbose_level);
};

// #############################################################################
// draw.C
// #############################################################################

void transform_llur(int *in, int *out, int &x, int &y);
void transform_dist(int *in, int *out, int &x, int &y);
void transform_dist_x(int *in, int *out, int &x);
void transform_dist_y(int *in, int *out, int &y);
void transform_llur_double(double *in, double *out, double &x, double &y);
void draw(char *fname);
void on_circle_int(int *Px, int *Py, int idx, int angle_in_degree, int rad);
void on_circle_double(double *Px, double *Py, int idx, 
	double angle_in_degree, double rad);
void polygon3D(mp_graphics &G, int *Px, int *Py, int dim, 
	int x0, int y0, int z0, int x1, int y1, int z1);
void integer_4pts(mp_graphics &G, int *Px, int *Py, 
	int p1, int p2, int p3, int p4, 
	const char *align, int a);
void text_4pts(mp_graphics &G, int *Px, int *Py, 
	int p1, int p2, int p3, int p4, 
	const char *align, const char *str);
void affine_pt1(int *Px, int *Py, int p0, int p1, int p2, 
	double f1, int p3);
void affine_pt2(int *Px, int *Py, int p0, int p1, int p1b, 
	double f1, int p2, int p2b, double f2, int p3);
int C3D(int i, int j, int k);
int C2D(int i, int j);
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
void draw_graph(mp_graphics *G, int x, int y, int dx, int dy, 
	int nb_V, int *Edges, int nb_E);
void draw_graph_with_distinguished_edge(mp_graphics *G, int x, int y, 
	int dx, int dy, int nb_V, int *Edges, int nb_E, 
	int distinguished_edge, int verbose_level);
void draw_graph_on_multiple_circles(mp_graphics *G, int x, int y, 
	int dx, int dy, int nb_V, 
	int *Edges, int nb_E, int nb_circles);
void draw_graph_on_2D_grid(mp_graphics *G, 
	int x, int y, int dx, int dy, 
	int rad, int nb_V, 
	int *Edges, int nb_E, int *coords_2D, int *Base, 
	int f_point_labels, int point_label_offset, int f_directed);
void draw_tournament(mp_graphics *G, 
	int x, int y, int dx, int dy, int nb_V, 
	int *Edges, int nb_E, int verbose_level);
void draw_bitmatrix(const char *fname_base, int f_dots, 
	int f_partition, int nb_row_parts, int *row_part_first, 
	int nb_col_parts, int *col_part_first, 
	int f_row_grid, int f_col_grid, 
	int f_bitmatrix, uchar *D, int *M, 
	int m, int n, int xmax_in, int ymax_in, int xmax, int ymax, 
	double scale, double line_width, 
	int f_has_labels, int *labels);
void draw_bitmatrix2(mp_graphics &G, int f_dots, 
	int f_partition, int nb_row_parts, int *row_part_first, 
	int nb_col_parts, int *col_part_first, 
	int f_row_grid, int f_col_grid, 
	int f_bitmatrix, uchar *D, int *M, 
	int m, int n, int xmax, int ymax, 
	int f_has_labels, int *labels);


// #############################################################################
// mp_graphics.C:
// #############################################################################

//! a class to help with drawing elements in a 2D grid fashion

struct grid_frame {
	int f_matrix_notation;
	double origin_x;
	double origin_y;
	int m; // number of rows in the grid
	int n; // number of columns in the grid
	double dx;
	double dy;
};


//! a general 2D graphical output interface (metapost, tikz, postscript)


class mp_graphics {

	char fname_base[1000];
	char fname_mp[1000];
	char fname_log[1000];
	char fname_tikz[1000];
	std::ofstream fp_mp;
	std::ofstream fp_log;
	std::ofstream fp_tikz;
	int f_file_open;
	
	
	// coordinate systems:
	
	int user[4]; // llx/lly/urx/ury 
	int dev[4]; // llx/lly/urx/ury 

	int x_min, x_max, y_min, y_max, f_min_max_set;

	int txt_halign;
		// 0=left aligned, 
		// 1=centered, 
		// 2=right aligned; 
		// default=0
	int txt_valign; // 0=bottom, 1=middle, 2=top; default=0
	int txt_boxed; //  default=0
	int txt_overwrite; // default=0
	int txt_rotate; // default = 0 (in degree)

	int line_beg_style; // default=0
	int line_end_style; // 0=nothing, 1=arrow; default=0

	int line_thickness; // 1,2,3
	int line_color; // 0=white, 1=black, 2=red, 3=green


	int fill_interior;
		// in 1/100th, 
		// 0= none (used for pie-drawing); 
		// default=0
	int fill_color; // 0 = white, 1 = black; default=0
	int fill_shape; // 0 =  .., 1 = -- ; default=1
	int fill_outline; // default=0
	int fill_nofill; // default=0


	int line_dashing;
		// 0 = no dashing, 
		// otherwise scaling factor 1/100th evenly
		// default=0
	
	int cur_path;

	int f_embedded;
		// have a header so that the file 
		// can be compiled standalone (for tikz)
	int f_sideways;

public:
	// for tikz:
	double tikz_global_scale; // .45 works
	double tikz_global_line_width; // 1.5 works


	mp_graphics();
	mp_graphics(const char *file_name, 
		int xmin, int ymin, int xmax, int ymax, 
		int f_embedded, int f_sideways);
	~mp_graphics();
	void default_values();
	void init(const char *file_name, 
		int xmin, int ymin, int xmax, int ymax, 
		int f_embedded, int f_sideways);
	void exit(std::ostream &ost, int verbose_level);
	void setup(const char *fname_base, 
		int in_xmin, int in_ymin, int in_xmax, int in_ymax, 
		int xmax, int ymax, int f_embedded, int f_sideways, 
		double scale, double line_width);
	void set_parameters(double scale, double line_width);
	void set_scale(double scale);
	void frame(double move_out);
	void frame_constant_aspect_ratio(double move_out);
	void finish(std::ostream &ost, int verbose_level);

	int& out_xmin();
	int& out_ymin();
	int& out_xmax();
	int& out_ymax();

	void user2dev(int &x, int &y);
	void dev2user(int &x, int &y);
	void user2dev_dist_x(int &x);
	void user2dev_dist_y(int &y);

	void draw_polar_grid(double r_max, int nb_circles, 
		int nb_rays, double x_stretch);
	void draw_axes_and_grid(
		double x_min, double x_max, 
		double y_min, double y_max, 
		double x_stretch, double y_stretch, 
		int f_x_axis_at_y_min, int f_y_axis_at_x_min, 
		int x_mod, int y_mod, int x_tick_mod, int y_tick_mod, 
		double x_labels_offset, double y_labels_offset, 
		double x_tick_half_width, double y_tick_half_width, 
		int f_v_lines, int subdivide_v, 
		int f_h_lines, int subdivide_h);
	void plot_curve(int N, int *f_DNE, 
		double *Dx, double *Dy, double dx, double dy);
	void nice_circle(int x, int y, int rad);
	void grid_polygon2(grid_frame *F, int x0, int y0, 
		int x1, int y1);
	void grid_polygon4(grid_frame *F, int x0, int y0, 
		int x1, int y1, int x2, int y2, int x3, int y3);
	void grid_polygon5(grid_frame *F, int x0, int y0, 
		int x1, int y1, int x2, int y2, 
		int x3, int y3, int x4, int y4);
	void polygon(int *Px, int *Py, int n);
	void polygon2(int *Px, int *Py, int i1, int i2);
	void polygon3(int *Px, int *Py, int i1, int i2, int i3);
	void polygon4(int *Px, int *Py, int i1, int i2, int i3, 
		int i4);
	void polygon5(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5);
	void polygon6(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6);
	void polygon7(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7);
	void polygon8(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8);
	void polygon9(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, int i9);
	void polygon10(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, int i9, 
		int i10);
	void polygon11(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, int i9, 
		int i10, int i11);
	void polygon_idx(int *Px, int *Py, int *Idx, int n);
	void bezier(int *Px, int *Py, int n);
	void bezier2(int *Px, int *Py, int i1, int i2);
	void bezier3(int *Px, int *Py, int i1, int i2, int i3);
	void bezier4(int *Px, int *Py, int i1, int i2, int i3, 
		int i4);
	void bezier5(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5);
	void bezier6(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6);
	void bezier7(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7);
	void bezier_idx(int *Px, int *Py, int *Idx, int n);
	void grid_fill_polygon4(grid_frame *F, 
		int x0, int y0, int x1, int y1, int x2, 
		int y2, int x3, int y3);
	void grid_fill_polygon5(grid_frame *F, 
		int x0, int y0, int x1, int y1, 
		int x2, int y2, int x3, int y3, 
		int x4, int y4);
	void fill_polygon3(int *Px, int *Py, int i1, int i2, int i3);
	void fill_polygon4(int *Px, int *Py, int i1, int i2, int i3, 
		int i4);
	void fill_polygon5(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5);
	void fill_polygon6(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6);
	void fill_polygon7(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7);
	void fill_polygon8(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8);
	void fill_polygon9(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, int i9);
	void fill_polygon10(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, int i9, int i10);
	void fill_polygon11(int *Px, int *Py, int i1, int i2, int i3, 
		int i4, int i5, int i6, int i7, int i8, 
		int i9, int i10, int i11);
	void polygon2_arrow_halfway(int *Px, int *Py, int i1, int i2);
	void polygon2_arrow_halfway_and_label(int *Px, int *Py, int i1, int i2, 
		const char *alignment, const char *txt);
	void grid_aligned_text(grid_frame *F, int x, int y, 
		const char *alignment, const char *p);
	void aligned_text(int x, int y, const char *alignment, const char *p);
	void aligned_text_array(int *Px, int *Py, int idx, 
		const char *alignment, const char *p);
	void aligned_text_with_offset(int x, int y, int xoffset, int yoffset, 
		const char *alignment, const char *p);

	void st_alignment(int txt_halign, int txt_valign);
	void sl_udsty(int line_dashing);
	void sl_ends(int line_beg_style, int line_end_style);
	void sl_thickness(int line_thickness);
	void sl_color(int line_color);
	void sf_interior(int fill_interior);
	void sf_color(int fill_color);
	void sf_shape(int fill_shape);
	void sf_outline(int fill_outline);
	void sf_nofill(int fill_nofill);
	void st_boxed(int txt_boxed);
	void st_overwrite(int txt_overwrite);
	void st_rotate(int txt_rotate);
	void coords_min_max(int x, int y);

	// output commands:
	void header();
	void footer();
	void begin_figure(int factor_1000);
	void end_figure();

	void comment(const char *p);
	void text(int x, int y, const char *p);
	void circle(int x, int y, int rad);
	void circle_text(int x, int y, int rad, const char *text);
#if 0
	void polygon_or_bezier_idx(int *Px, int *Py, int *Idx, int n, 
		const char *symbol, int f_cycle);
#endif
	void polygon_idx2(int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void bezier_idx2(int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void fill_idx(int *Px, int *Py, int *Idx, int n, 
		const char *symbol, int f_cycle);


	// output commands log file:
	void header_log(char *str_date);
	void footer_log();
	void comment_log(const char *p);
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
	void bezier_idx_log(int *Px, int *Py, int *Idx, int n);
	void polygon_log(int *Px, int *Py, int n);
	void polygon_idx_log(int *Px, int *Py, int *Idx, int n);
	void text_log(int x1, int y1, const char *p);
	void circle_log(int x1, int y1, int rad);


	// output commands metapost:
	void header_mp(char *str_date);
	void footer_mp();
	void comment_mp(const char *p);
	void text_mp(int x1, int y1, const char *p);
	void begin_figure_mp(int factor_1000);
	void end_figure_mp();
	void circle_mp(int x, int y, int rad);
	void output_circle_text_mp(int x, int y, int idx, const char *text);
#if 0
	void polygon_or_bezier_idx_mp(int *Px, int *Py, int *Idx, 
		int n, const char *symbol, int f_cycle);
#endif
	void polygon_idx_mp(int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void bezier_idx_mp(int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void color_tikz(std::ofstream &fp, int color);
	void fill_idx_mp(int *Px, int *Py, int *Idx, int n, 
		const char *symbol, int f_cycle);
	void output_xy_metapost(int x, int y);
	void output_x_metapost(int x);
	void output_y_metapost(int y);
	int get_label(int x, int y);
	void get_alignment_mp(char *align);
	void line_thickness_mp();

	// output commands tikz:
	void header_tikz(char *str_date);
	void footer_tikz();
	void comment_tikz(const char *p);
	void text_tikz(int x1, int y1, const char *p);
	void circle_tikz(int x, int y, int rad);
	void output_circle_text_tikz(int x, int y, int idx, int rad, 
		const char *text);
#if 0
	void polygon_or_bezier_idx_tikz(int *Px, int *Py, int *Idx, int n, 
		const char *symbol, int f_cycle);
#endif
	void polygon_idx_tikz(int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void bezier_idx_tikz(int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void fill_idx_tikz(std::ofstream &fp,
		int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle);
	void output_xy_tikz(int x, int y);
	void output_x_tikz(int x);
	void output_y_tikz(int y);




};


// #############################################################################
// plot.C:
// #############################################################################

void draw_density(char *prefix, int *the_set, int set_size,
	int f_title, const char *title, int out_of, 
	const char *label_x, 
	int f_circle, int circle_at, int circle_rad, 
	int f_mu, int f_sigma, int nb_standard_deviations, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int xmax, int ymax, int offset_x, 
	int f_switch_x, int no, int f_embedded, 
	int verbose_level);
void draw_density_multiple_curves(char *prefix,
	int **Data, int *Data_size, int nb_data_sets, 
	int f_title, const char *title, int out_of, 
	const char *label_x, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int xmax, int ymax, int offset_x, int f_switch_x, 
	int f_v_logarithmic, double log_base, int no, int f_embedded, 
	int verbose_level);
void draw_density2(mp_graphics &G, int no, 
	int *outline_value, int *outline_number, int outline_sz, 
	int min_value, int max_value, int offset_x, int f_switch_x, 
	int f_title, const char *title, 
	const char *label_x, 
	int f_circle, int circle_at, int circle_rad, 
	int f_mu, int f_sigma, int nb_standard_deviations, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid);
void draw_density2_multiple_curves(mp_graphics &G, int no, 
	int **outline_value, int **outline_number, 
	int *outline_sz, int nb_curves, 
	int min_x, int max_x, int min_y, int max_y, 
	int offset_x, int f_switch_x, 
	int f_title, const char *title, 
	const char *label_x, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int f_v_logarithmic, double log_base);
void read_numbers_from_file(const char *fname, 
	int *&the_set, int &set_size, int verbose_level);
void get_coord(int *Px, int *Py, int idx, int x, int y, 
	int min_x, int min_y, int max_x, int max_y, int f_switch_x);
void get_coord_log(int *Px, int *Py, int idx, int x, int y, 
	int min_x, int min_y, int max_x, int max_y, 
	double log_base, int f_switch_x);
void y_to_pt_on_curve(int y_in, int &x, int &y,  
	int *outline_value, int *outline_number, int outline_sz);
void projective_plane_draw_grid(const char *fname, int xmax, int ymax, 
	int f_with_points, int rad, 
	int q, int *Table, int nb, 
	int f_point_labels, char **Point_labels, 
	int f_embedded, int f_sideways, 
	int verbose_level);
void projective_plane_draw_grid2(mp_graphics &G, int q, int *Table, 
	int nb, int f_with_points, int rad, 
	int f_point_labels, char **Point_labels, int verbose_level);
void projective_plane_make_affine_point(int q, int x1, int x2, int x3, 
	double &a, double &b);

// #############################################################################
// povray_interface.cpp
// #############################################################################

void povray_beginning(std::ostream &ost,
		double angle,
		const char *sky,
		const char *location,
		const char *look_at,
		int f_with_background);
void povray_animation_rotate_around_origin_and_1_1_1(std::ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector(double *v,
		std::ostream &ost);
void povray_animation_rotate_around_origin_and_given_vector_by_a_given_angle(
	double *v, double angle_zero_one, std::ostream &ost);
void povray_union_start(std::ostream &ost);
void povray_union_end(std::ostream &ost, double clipping_radius);
void povray_bottom_plane(std::ostream &ost);
void povray_rotate_111(int h, int nb_frames, std::ostream &fp);
void povray_ini(std::ostream &ost, const char *fname_pov, int first_frame,
	int last_frame);



// #############################################################################
// scene.C:
// #############################################################################

#define SCENE_MAX_LINES 100000
#define SCENE_MAX_EDGES 100000
#define SCENE_MAX_POINTS 100000
#define SCENE_MAX_PLANES 10000
#define SCENE_MAX_QUADRICS 10000
#define SCENE_MAX_CUBICS 10000
#define SCENE_MAX_FACES 10000


//! a collection of 3D geometry objects



class scene {
public:
	
	double line_radius;

	int nb_lines;
	double *Line_coords;
		// [nb_lines * 6] a line is given by two points
	
	int nb_edges;
	int *Edge_points;
		// [nb_edges * 2]

	int nb_points;
	double *Point_coords;
		// [nb_points * 3]

	int nb_planes;
	double *Plane_coords;
		// [nb_planes * 4]

	int nb_quadrics;
	double *Quadric_coords;
		// [nb_quadrics * 10]

	int nb_cubics;
	double *Cubic_coords;
		// [nb_cubics * 20]

	int nb_faces;
	int *Nb_face_points; // [nb_faces]
	int **Face_points; // [nb_faces]


	
	void *extra_data;

	int f_has_affine_space;
	int affine_space_q;
	int affine_space_starting_point;


	scene();
	~scene();
	void null();
	void freeself();
	void init(int verbose_level);
	scene *transformed_copy(double *A4, double *A4_inv, 
		double rad, int verbose_level);
	void print();
	void transform_lines(scene *S, double *A4, double *A4_inv, 
		double rad, int verbose_level);
	void copy_edges(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	void transform_points(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	void transform_planes(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	void transform_quadrics(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	void transform_cubics(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	void copy_faces(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	int line_pt_and_dir(double *x6, double rad);
	int line_through_two_pts(double *x6, double rad);
	int line6(double *x6);
	int line(double x1, double x2, double x3, 
		double y1, double y2, double y3);
	int line_through_two_points(int pt1, int pt2, 
		double rad);
	int edge(int pt1, int pt2);
	void points(double *Coords, int nb_points);
	int point(double x1, double x2, double x3);
	int point_center_of_mass_of_face(int face_idx);
	int point_center_of_mass_of_edge(int edge_idx);
	int point_center_of_mass(int *Pt_idx, int nb_pts);
	int triangle(int line1, int line2, int line3, int verbose_level);
	int point_as_intersection_of_two_lines(int line1, int line2);
	int plane_from_dual_coordinates(double *x4);
	int plane(double x1, double x2, double x3, double a);
		// A plane is called a polynomial shape because 
		// it is defined by a first order polynomial equation. 
		// Given a plane: plane { <A, B, C>, D }
		// it can be represented by the equation 
		// A*x + B*y + C*z - D*sqrt(A^2 + B^2 + C^2) = 0.
		// see http://www.povray.org/documentation/view/3.6.1/297/
	int plane_through_three_points(int pt1, int pt2, int pt3);
	int quadric_through_three_lines(int line_idx1, 
		int line_idx2, int line_idx3, int verbose_level);
	int quadric(double *coeff);
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
	int cubic(double *coeff);
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
	int face(int *pts, int nb_pts);
	int face3(int pt1, int pt2, int pt3);
	int face4(int pt1, int pt2, int pt3, int pt4);
	int face5(int pt1, int pt2, int pt3, int pt4, int pt5);
	void draw_lines_with_selection(int *selection, int nb_select, 
		const char *options, std::ostream &ost);
	void draw_line_with_selection(int line_idx, 
		const char *options, std::ostream &ost);
	void draw_lines_cij_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_cij(std::ostream &ost);
	void draw_lines_ai_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_ai(std::ostream &ost);
	void draw_lines_bj_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_bj(std::ostream &ost);
	void draw_edges_with_selection(int *selection, int nb_select, 
		const char *options, std::ostream &ost);
	void draw_faces_with_selection(int *selection, int nb_select, 
		double thickness_half, const char *options, std::ostream &ost);
	void draw_face(int idx, double thickness_half, const char *options, 
			std::ostream &ost);
	void draw_text(const char *text, double thickness_half, double extra_spacing, 
			double scale, 
			double off_x, double off_y, double off_z, 
			const char *color_options, 
			double x, double y, double z, 
			double up_x, double up_y, double up_z, 
			double view_x, double view_y, double view_z, 
			std::ostream &ost, int verbose_level);
	void draw_planes_with_selection(int *selection, int nb_select, 
		const char *options, std::ostream &ost);
	void draw_points_with_selection(int *selection, int nb_select, 
		double rad, const char *options, std::ostream &ost);
	void draw_cubic_with_selection(int *selection, int nb_select, 
		const char *options, std::ostream &ost);
	void draw_quadric_with_selection(int *selection, int nb_select, 
		const char *options, std::ostream &ost);
	void draw_quadric_clipped_by_plane(int quadric_idx, int plane_idx,
		const char *options, std::ostream &ost);
	void draw_line_clipped_by_plane(int line_idx, int plane_idx,
			const char *options, std::ostream &ost);
	int intersect_line_and_plane(int line_idx, int plane_idx, 
		int &intersection_point_idx, 
		int verbose_level);
	int intersect_line_and_line(int line1_idx, int line2_idx, 
		double &lambda, 
		int verbose_level);
#if 0
	int line_centered(double *pt1_in, double *pt2_in, 
		double *pt1_out, double *pt2_out, 
		double r);
#endif
	int line_extended(double x1, double x2, double x3, 
		double y1, double y2, double y3, 
		double r);
	void map_a_line(int line1, int line2, 
		int plane_idx, int line_idx, double spread, 
		int nb_pts, 
		int *New_line_idx, int &nb_new_lines, 
		int *New_pt_idx, int &nb_new_points, int verbose_level);
	int map_a_point(int line1, int line2, 
		int plane_idx, double pt_in[3], 
		int &new_line_idx, int &new_pt_idx, 
		int verbose_level);
	void lines_a();
	void lines_b();
	void lines_cij();
	void Eckardt_points();
	void fourD_cube(double rad_desired);
	void rescale(int first_pt_idx, double rad_desired);
	double euclidean_distance(int pt1, int pt2);
	double distance_from_origin(int pt);
	void fourD_cube_edges(int first_pt_idx);
	void hypercube(int n, double rad_desired);
	void Dodecahedron_points();
	void Dodecahedron_edges(int first_pt_idx);
	void Dodecahedron_planes(int first_pt_idx);
	void tritangent_planes();
	void clebsch_cubic();
	double distance_between_two_points(int pt1, int pt2);
	void create_five_plus_one();
	void create_Hilbert_model(int verbose_level);
	void create_affine_space(int q, int verbose_level);
	//void create_surface_13_1(int verbose_level);

};


// #############################################################################
// tree.C:
// #############################################################################


//! a data structure for trees


class tree {

public:

	tree_node *root;
	
	int nb_nodes;
	int max_depth;
	
	int *path;

	int f_count_leaves;
	int leaf_count;

	tree();
	~tree();
	void init(const char *fname, int xmax, int ymax, int verbose_level);
	void draw(char *fname, int xmax_in, int ymax_in, int xmax, int ymax, 
		int rad, 
		int f_circle, int f_circletext, int f_i, int f_edge_labels, 
		int f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, 
			int *v, int layer, tree_node *N, 
			int x, int y, int dx, int dy), 
		int f_embedded, int f_sideways, int f_on_circle, 
		double tikz_global_scale, double tikz_global_line_width
		);
	void circle_center_and_radii(int xmax, int ymax, int max_depth, 
		int &x0, int &y0, int *&rad);
	void compute_DFS_ranks(int &nb_nodes, int verbose_level);
};

// #############################################################################
// tree_node.C:
// #############################################################################


//! part of the data structure tree


class tree_node {

public:
	tree_node *parent;
	int depth;
	int f_value;
	int value;
	
	int f_int_data;
	int int_data;
	char *char_data;
	int nb_children;
	tree_node **children;

	int weight;
	int placement_x;
	int placement_y;
	int width;

	int DFS_rank;

	tree_node();
	~tree_node();
	void init(int depth, tree_node *parent, int f_value, int value, 
		int f_i_data, int i_data, char *c_data, int verbose_level);
	void print_path();
	void print_depth_first();
	void compute_DFS_rank(int &rk);
	void get_coordinates(int &idx, int *coord_xy);
	void get_coordinates_and_width(int &idx, int *coord_xyw);
	void calc_weight();
	void place_xy(int left, int right, int ymax, int max_depth);
	void place_on_circle(int xmax, int ymax, int max_depth);
	void add_node(int l, int depth, int *path, int i_data, char *c_data, 
		int verbose_level);
	int find_child(int val);
	void get_values(int *v);
	void draw_edges(mp_graphics &G, int rad, int f_circle, 
		int f_circletext, int f_i, 
		int f_has_parent, int parent_x, int parent_y, 
		int max_depth, int f_edge_labels, 
		int f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, int *v, 
			int layer, tree_node *N, int x, int y, int dx, int dy),
		tree *T
		);
	void draw_vertices(mp_graphics &G, int rad, int f_circle, 
		int f_circletext, int f_i, 
		int f_has_parent, int parent_x, int parent_y, int max_depth, 
		int f_edge_labels, 
		int f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, int *v, 
			int layer, tree_node *N, int x, int y, int dx, int dy),
		tree *T
		);
	void draw_sideways(mp_graphics &G, int f_circletext, int f_i, 
		int f_has_parent, int parent_x, int parent_y, 
		int max_depth, int f_edge_labels);
};

int tree_node_calc_y_coordinate(int ymax, int l, int max_depth);

// #############################################################################
// video_draw_options.cpp:
// #############################################################################


//! options for povray videos


class video_draw_options {
public:


	int f_has_global_picture_scale;
	double global_picture_scale;

	int f_has_font_size;
	int font_size;

	int f_has_stroke_width;
	int stroke_width;



	int f_W;
	int W;
	int f_H;
	int H;

	int f_default_angle; // = FALSE;
	int default_angle; // = 22;

	int f_clipping_radius; // = TRUE;
	double clipping_radius; // = 0.9;


	int nb_clipping;
	int clipping_round[1000];
	double clipping_value[1000];

	int nb_camera;
	int camera_round[1000];
	const char *camera_sky[1000];
	const char *camera_location[1000];
	const char *camera_look_at[1000];

	int nb_zoom;
	int zoom_round[1000];
	int zoom_start[1000];
	int zoom_end[1000];

	int nb_zoom_sequence;
	int zoom_sequence_round[1000];
	const char *zoom_sequence_text[1000];

	int nb_pan;
	int pan_round[1000];
	int pan_f_reverse[1000];
	double pan_from_x[1000];
	double pan_from_y[1000];
	double pan_from_z[1000];
	double pan_to_x[1000];
	double pan_to_y[1000];
	double pan_to_z[1000];
	double pan_center_x[1000];
	double pan_center_y[1000];
	double pan_center_z[1000];

	int nb_no_background;
	int no_background_round[1000];

	int nb_no_bottom_plane;
	int no_bottom_plane_round[1000];

	int cnt_nb_frames;
	int nb_frames_round[1000];
	int nb_frames_value[1000];

	int nb_round_text;
	int round_text_round[1000];
	int round_text_sustain[1000];
	const char *round_text_text[1000];

	int nb_label;
	int label_round[1000];
	int label_start[1000];
	int label_sustain[1000];
	const char *label_gravity[1000];
	const char *label_text[1000];

	int nb_latex_label;
	int latex_label_round[1000];
	int latex_label_start[1000];
	int latex_label_sustain[1000];
	const char *latex_extras_for_praeamble[1000];
	const char *latex_label_gravity[1000];
	const char *latex_label_text[1000];
	int latex_f_label_has_been_prepared[1000];
	char *latex_fname_base[1000];


	int nb_picture;
	int picture_round[1000];
	double picture_scale[1000];
	const char *picture_fname[1000];
	const char *picture_options[1000];

	int latex_file_count;
	int f_omit_bottom_plane;

	const char *sky;
	const char *location;
	const char *look_at;


	video_draw_options();
	~video_draw_options();
	int read_arguments(
			int argc, const char **argv,
			int verbose_level);
};




}}



