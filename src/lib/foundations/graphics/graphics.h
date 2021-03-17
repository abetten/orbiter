// graphics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GRAPHICS_GRAPHICS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GRAPHICS_GRAPHICS_H_



namespace orbiter {
namespace foundations {


// #############################################################################
// animate.cpp
// #############################################################################


//! creates 3D animations using Povray


class animate {
public:
	scene *S;
	std::string output_mask;
	char fname_makefile[1000];
	int nb_frames;
	video_draw_options *Opt;
	std::ofstream *fpm;
	void (*draw_frame_callback)(animate *A, int frame,
					int nb_frames_this_round, int round,
					double clipping,
					std::ostream &fp,
					int verbose_level);
	void *extra_data;
	povray_interface *Pov;

	animate();
	~animate();
	void init(scene *S,
			std::string &output_mask,
			int nb_frames,
			video_draw_options *Opt,
			void *extra_data,
			int verbose_level);
	void animate_one_round(
		int round,
		int verbose_level);
	void draw_single_line(int line_idx, std::string &color, std::ostream &fp);
	void draw_single_quadric(int idx, std::string &color, std::ostream &fp);
	void draw_single_surface(int surface_idx, std::ostream &fp);
	void draw_single_surface_with_color(int surface_idx, std::string &color, std::ostream &fp);
	void draw_Hilbert_point(int point_idx, double rad,
			std::string &options, std::ostream &fp);
	void draw_Hilbert_line(int line_idx, std::string &color, std::ostream &fp);
	void draw_Hilbert_plane(int plane_idx, std::string &color, std::ostream &fp);
	void draw_Hilbert_red_line(int idx_one_based, std::ostream &fp);
	void draw_Hilbert_blue_line(int idx_one_based, std::ostream &fp);
	void draw_Hilbert_red_lines(std::ostream &fp);
	void draw_Hilbert_blue_lines(std::ostream &fp);
	void draw_Hilbert_cube_extended_edges(std::ostream &fp);
	void draw_Hilbert_cube_faces(std::ostream &fp);
	void draw_Hilbert_cube_boxed(std::ostream &fp);
	void draw_Hilbert_tetrahedron_boxed(std::ostream &fp);
	void draw_Hilbert_tetrahedron_faces(std::ostream &fp);
	void draw_frame_Hilbert(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_surface_13_1(std::ostream &fp);
	void draw_frame_Hilbert_round_76(video_draw_options *Opt,
			int h, int nb_frames, int round,
			std::ostream &fp,
			int verbose_level);
		// tritangent plane, 6 arc points, 2 blue lines, 6 red lines, text
	void draw_frame_HCV_surface(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_frame_E4_surface(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_frame_triangulation_of_cube(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_frame_twisted_cubic(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_frame_five_plus_one(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_frame_windy(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void rotation(
			int h, int nb_frames, int round,
			std::ostream &fp);
	void union_end(
			int h, int nb_frames, int round,
			double clipping_radius,
			std::ostream &fp);
	void draw_text(std::string &text,
			double thickness_half, double extra_spacing,
			double scale,
			double off_x, double off_y, double off_z,
			std::string &color_options,
			int idx_point,
			//double x, double y, double z,
			//double up_x, double up_y, double up_z,
			//double view_x, double view_y, double view_z,
			std::ostream &ost, int verbose_level);
	void draw_text_with_selection(int *selection, int nb_select,
		double thickness_half, double extra_spacing,
		double scale,
		double off_x, double off_y, double off_z,
		std::string &options, std::string &group_options,
		std::ostream &ost, int verbose_level);
};


// #############################################################################
// drawable_set_of_objects.cpp
// #############################################################################



//! a specific description of a set of objects that should be drawn



class drawable_set_of_objects {

public:

	int group_idx;

	int type;
	// 1 = sphere
	// 2 = cylinder
	// 3 = prisms (faces)
	// 4 = planes
	// 5 = lines
	// 6 = cubics
	// 7 = quadrics
	// 8 = quartics
	// 9 = quintics
	// 10 = octics
	// 11 = label


	double d;
	double d2; // for text: scale

	std::string properties;

	drawable_set_of_objects();
	~drawable_set_of_objects();
	void init_spheres(int group_idx, double rad,
			std::string &properties, int verbose_level);
	void init_cylinders(int group_idx,
			double rad, std::string &properties, int verbose_level);
	void init_prisms(int group_idx,
			double thickness, std::string &properties, int verbose_level);
	void init_planes(int group_idx,
			std::string &properties, int verbose_level);
	void init_lines(int group_idx,
			double rad, std::string &properties, int verbose_level);
	void init_cubics(int group_idx,
			std::string &properties, int verbose_level);
	void init_quadrics(int group_idx,
			std::string &properties, int verbose_level);
	void init_quartics(int group_idx,
			std::string &properties, int verbose_level);
	void init_quintics(int group_idx,
			std::string &properties, int verbose_level);
	void init_octics(int group_idx,
			std::string &properties, int verbose_level);
	void init_labels(int group_idx,
			double thickness_half, double scale, std::string &properties, int verbose_level);
	void draw(animate *Anim, std::ostream &ost,
			int f_group_is_animated, int frame, int verbose_level);

};


// #############################################################################
// graphical_output.cpp
// #############################################################################

//! a catch-all class for things related to 2D graphics


class graphical_output {

private:

public:

	function_polish *smooth_curve_Polish;
	double parabola_a;
	double parabola_b;
	double parabola_c;


	graphical_output();
	~graphical_output();
	void draw_layered_graph_from_file(std::string &fname,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void do_create_points_on_quartic(double desired_distance, int verbose_level);
	void do_create_points_on_parabola(double desired_distance, int N,
			double a, double b, double c, int verbose_level);
	void do_smooth_curve(std::string &curve_label,
			double desired_distance, int N,
			double t_min, double t_max, double boundary,
			function_polish_description *FP_descr, int verbose_level);
};


// #############################################################################
// mp_graphics.cpp
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

	//std::string fname_base;
	std::string fname_mp;
	std::string fname_log;
	std::string fname_tikz;
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
	mp_graphics(std::string &file_name,
		int xmin, int ymin, int xmax, int ymax, 
		int f_embedded, int f_sideways, int verbose_level);
	~mp_graphics();
	void default_values();
	void init(std::string &file_name,
		int xmin, int ymin, int xmax, int ymax, 
		int f_embedded, int f_sideways, int verbose_level);
	void exit(std::ostream &ost, int verbose_level);
	void setup(std::string &fname_base,
		int in_xmin, int in_ymin, int in_xmax, int in_ymax, 
		int xmax, int ymax, int f_embedded, int f_sideways, 
		double scale, double line_width, int verbose_level);
	void set_parameters(double scale, double line_width);
	void set_scale(double scale);
	void frame(double move_out);
	void frame_constant_aspect_ratio(double move_out);
	void finish(std::ostream &ost, int verbose_level);

	int& in_xmin();
	int& in_ymin();
	int& in_xmax();
	int& in_ymax();
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
			layered_graph_draw_options *O,
		double x_min, double x_max, 
		double y_min, double y_max, 
		double dx, double dy,
		int f_x_axis_at_y_min, int f_y_axis_at_x_min, 
		int x_mod, int y_mod, int x_tick_mod, int y_tick_mod, 
		double x_labels_offset, double y_labels_offset, 
		double x_tick_half_width, double y_tick_half_width, 
		int f_v_lines, int subdivide_v, 
		int f_h_lines, int subdivide_h,
		int verbose_level);
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
	void polygon_idx2(int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void bezier_idx2(int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void fill_idx(int *Px, int *Py, int *Idx, int n, 
		const char *symbol, int f_cycle);


	// output commands log file:
	void header_log(std::string &str_date);
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
	void header_mp(std::string &str_date);
	void footer_mp();
	void comment_mp(const char *p);
	void text_mp(int x1, int y1, const char *p);
	void begin_figure_mp(int factor_1000);
	void end_figure_mp();
	void circle_mp(int x, int y, int rad);
	void output_circle_text_mp(int x, int y, int idx, const char *text);
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
	void header_tikz(std::string &str_date);
	void footer_tikz();
	void comment_tikz(const char *p);
	void text_tikz(int x1, int y1, const char *p);
	void circle_tikz(int x, int y, int rad);
	void output_circle_text_tikz(int x, int y, int idx, int rad, 
		const char *text);
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

	void polygon3D(int *Px, int *Py,
			int dim, int x0, int y0, int z0, int x1, int y1, int z1);
	void integer_4pts(int *Px, int *Py,
			int p1, int p2, int p3, int p4,
			const char *align, int a);
	void text_4pts(int *Px, int *Py, int p1, int p2, int p3, int p4,
		const char *align, const char *str);


	void draw_graph(int x, int y,
			int dx, int dy, int nb_V, long int *Edges, int nb_E);
	void draw_graph_with_distinguished_edge(
		int x, int y,
		int dx, int dy, int nb_V, long int *Edges, int nb_E,
		int distinguished_edge, int verbose_level);
	void draw_graph_on_multiple_circles(int x, int y,
			int dx, int dy, int nb_V, int *Edges, int nb_E, int nb_circles);
	void draw_graph_on_2D_grid(
			int x, int y, int dx, int dy, int rad, int nb_V,
			int *Edges, int nb_E, int *coords_2D, int *Base,
			int f_point_labels, int point_label_offset, int f_directed);
	void draw_tournament(int x, int y,
			int dx, int dy, int nb_V, long int *Edges, int nb_E,
			int verbose_level);
	void draw_bitmatrix2(int f_dots,
		int f_partition, int nb_row_parts, int *row_part_first,
		int nb_col_parts, int *col_part_first,
		int f_row_grid, int f_col_grid,
		int f_bitmatrix, bitmatrix *Bitmatrix, int *M,
		int m, int n, int xmax, int ymax,
		int f_has_labels, int *labels);

	void draw_density2(int no,
		int *outline_value, int *outline_number, int outline_sz,
		int min_value, int max_value, int offset_x, int f_switch_x,
		int f_title, const char *title,
		const char *label_x,
		int f_circle, int circle_at, int circle_rad,
		int f_mu, int f_sigma, int nb_standard_deviations,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid);
	void draw_density2_multiple_curves(int no,
		int **outline_value, int **outline_number,
		int *outline_sz, int nb_curves,
		int min_x, int max_x, int min_y, int max_y,
		int offset_x, int f_switch_x,
		int f_title, const char *title,
		const char *label_x,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int f_v_logarithmic, double log_base);
	void projective_plane_draw_grid2(
			layered_graph_draw_options *O,
			int q,
			int *Table, int nb,
			int f_point_labels, char **Point_labels, int verbose_level);
	void draw_matrix_in_color(
		int f_row_grid, int f_col_grid,
		int *Table, int nb_colors,
		int m, int n, int xmax, int ymax,
		int *color_scale, int nb_colors_in_scale,
		int f_has_labels, int *labels);
	void domino_draw1(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw2(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw3(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw4(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw5(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw6(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw7(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw8(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw9(int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw_assignment_East(int Ap, int Aq, int M,
			int i, int j, int dx, int dy, int rad);
	void domino_draw_assignment_South(int Ap, int Aq, int M,
			int i, int j, int dx, int dy, int rad);
	void domino_draw_assignment(int *A, int *matching, int *B,
			int M, int N,
			int dx, int dy,
			int rad, int edge,
			int f_grid, int f_gray, int f_numbers, int f_frame,
			int f_cost, int cost);
};


// #############################################################################
// parametric_curve_point.cpp
// #############################################################################

//! an individual point on a continuous curve, sampled through parametric_curve

class parametric_curve_point {
public:

	double t;
	int f_is_valid;
	std::vector<double> coords;

	parametric_curve_point();
	~parametric_curve_point();
	void init(double t, int f_is_valid, double *x,
			int nb_dimensions, int verbose_level);
};

// #############################################################################
// parametric_curve.cpp
// #############################################################################

//! a continuous curve sampled by individual points

class parametric_curve {
public:

	int nb_dimensions;
	double desired_distance;
	double t0, t1; // parameter interval
	int (*compute_point_function)(double t, double *pt, void *extra_data, int verbose_level);
	void *extra_data;
	double boundary;

	int nb_pts;
	std::vector<parametric_curve_point> Pts;

	parametric_curve();
	~parametric_curve();
	void init(int nb_dimensions,
			double desired_distance,
			double t0, double t1,
			int (*compute_point_function)(double t, double *pt, void *extra_data, int verbose_level),
			void *extra_data,
			double boundary,
			int N,
			int verbose_level);

};

// #############################################################################
// plot_tools.cpp
// #############################################################################

//! utility functions for plotting (graphing)


class plot_tools {

public:
	plot_tools();
	~plot_tools();

	void draw_density(char *prefix, int *the_set, int set_size,
		int f_title, const char *title, int out_of,
		const char *label_x,
		int f_circle, int circle_at, int circle_rad,
		int f_mu, int f_sigma, int nb_standard_deviations,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int xmax, int ymax, int offset_x,
		int f_switch_x, int no, int f_embedded,
		int verbose_level);
	void draw_density_multiple_curves(std::string &prefix,
		int **Data, int *Data_size, int nb_data_sets,
		int f_title, const char *title, int out_of,
		const char *label_x,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int xmax, int ymax, int offset_x, int f_switch_x,
		int f_v_logarithmic, double log_base, int no, int f_embedded,
		int verbose_level);
	void get_coord(int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y, int f_switch_x);
	void get_coord_log(int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y,
		double log_base, int f_switch_x);
	void y_to_pt_on_curve(int y_in, int &x, int &y,
		int *outline_value, int *outline_number, int outline_sz);
	void projective_plane_draw_grid(std::string &fname,
			layered_graph_draw_options *O,
			int q, int *Table, int nb,
			int f_point_labels, char **Point_labels,
			int verbose_level);
	void draw_mod_n(std::string &fname,
			layered_graph_draw_options *O,
			int number_n,
			int f_inverse,
			int f_additive_inverse,
			int f_power_cycle, int power_cycle_base,
			int f_cyclotomic_sets, int cyclotomic_sets_q, std::string &cyclotomic_sets_reps,
			int verbose_level);
	void draw_mod_n_work(mp_graphics &G,
			layered_graph_draw_options *O,
			int number,
			int f_inverse,
			int f_additive_inverse,
			int f_power_cycle, int power_cycle_base,
			int f_cyclotomic_sets, int cyclotomic_sets_q, std::string &cyclotomic_sets_reps,
			int verbose_level);

};

// #############################################################################
// povray_interface.cpp
// #############################################################################

//! povray interface for 3D graphics



class povray_interface {
public:


	std::string color_white_simple;
	std::string color_white;
	std::string color_white_very_transparent;
	std::string color_black;
	std::string color_pink;
	std::string color_pink_transparent;
	std::string color_green;
	std::string color_gold;
	std::string color_red;
	std::string color_blue;
	std::string color_yellow;
	std::string color_yellow_transparent;
	std::string color_scarlet;
	std::string color_brown;
	std::string color_orange;
	std::string color_orange_transparent;
	std::string color_orange_no_phong;
	std::string color_chrome;
	std::string color_gold_dode;
	std::string color_gold_transparent;
	std::string color_red_wine_transparent;
	std::string color_yellow_lemon_transparent;

	double sky[3];
	double location[3];
	double look_at[3];


	povray_interface();
	~povray_interface();
	void beginning(std::ostream &ost,
			double angle,
			double *sky,
			double *location,
			double *look_at,
			int f_with_background);
	void animation_rotate_around_origin_and_1_1_1(std::ostream &ost);
	void animation_rotate_around_origin_and_given_vector(double *v,
			std::ostream &ost);
	void animation_rotate_xyz(
		double angle_x_deg, double angle_y_deg, double angle_z_deg, std::ostream &ost);
	void animation_rotate_around_origin_and_given_vector_by_a_given_angle(
		double *v, double angle_zero_one, std::ostream &ost);
	void union_start(std::ostream &ost);
	void union_end(std::ostream &ost, double scale_factor, double clipping_radius);
	void union_end_box_clipping(std::ostream &ost, double scale_factor,
			double box_x, double box_y, double box_z);
	void union_end_no_clipping(std::ostream &ost, double scale_factor);
	void bottom_plane(std::ostream &ost);
	void rotate_111(int h, int nb_frames, std::ostream &fp);
	void rotate_around_z_axis(int h, int nb_frames, std::ostream &fp);
	void ini(std::ostream &ost, const char *fname_pov, int first_frame,
		int last_frame);
};


// #############################################################################
// scene.cpp
// #############################################################################

#define SCENE_MAX_LINES    100000
#define SCENE_MAX_EDGES    100000
#define SCENE_MAX_POINTS   200000
#define SCENE_MAX_PLANES    10000
#define SCENE_MAX_QUADRICS  10000
#define SCENE_MAX_OCTICS      100
#define SCENE_MAX_QUARTICS   1000
#define SCENE_MAX_QUINTICS    500
#define SCENE_MAX_CUBICS    10000
#define SCENE_MAX_FACES    200000


//! a collection of 3D geometry objects



class scene {
	
private:

	double *Line_coords;
		// [nb_lines * 6] a line is given by two points

	int *Edge_points;
		// [nb_edges * 2]

	double *Point_coords;
		// [nb_points * 3]

	double *Plane_coords;
		// [nb_planes * 4]
		// the four parameters A,B,C,D as needed for the povray command
		// plane{<A,B,C>, D}

	double *Quadric_coords;
		// [nb_quadrics * 10]

	double *Cubic_coords;
		// [nb_cubics * 20]

	double *Quartic_coords;
		// [nb_quartics * 35]

	double *Quintic_coords;
		// [nb_quintics * 56]

	double *Octic_coords;
		// [nb_quartics * 165]

	int *Nb_face_points; // [nb_faces]
	int **Face_points; // [nb_faces]

public:


	std::vector<std::pair<int, std::string> > Labels;


	double line_radius;

	int nb_lines;

	int nb_edges;

	int nb_points;

	int nb_planes;

	int nb_quadrics;

	int nb_cubics;

	int nb_quartics;

	int nb_quintics;

	int nb_octics;

	int nb_faces;

	int nb_groups;
	std::vector<std::vector<int> > group_of_things;

	std::vector<int> animated_groups;

	std::vector<drawable_set_of_objects> Drawables;


	
	void *extra_data;

	int f_has_affine_space;
	int affine_space_q;
	int affine_space_starting_point;


	scene();
	~scene();
	void null();
	void freeself();
	double label(int idx, std::string &txt);
	double point_coords(int idx, int j);
	double line_coords(int idx, int j);
	double plane_coords(int idx, int j);
	double cubic_coords(int idx, int j);
	double quadric_coords(int idx, int j);
	int edge_points(int idx, int j);
	void print_point_coords(int idx);
	double point_distance_euclidean(int pt_idx, double *y);
	double point_distance_from_origin(int pt_idx);
	double distance_euclidean_point_to_point(int pt1_idx, int pt2_idx);
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
	void transform_quartics(scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_quintics(scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void copy_faces(scene *S, double *A4, double *A4_inv, 
		int verbose_level);
	int line_pt_and_dir(double *x6, double rad, int verbose_level);
	int line_pt_and_dir_and_copy_points(double *x6, double rad, int verbose_level);
	int line_through_two_pts(double *x6, double rad);
	int line6(double *x6);
	int line(double x1, double x2, double x3, 
		double y1, double y2, double y3);
	int line_after_recentering(double x1, double x2, double x3,
		double y1, double y2, double y3, double rad);
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
	int quintic(double *coeff_56);
	int octic(double *coeff_165);
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
	int cubic_in_orbiter_ordering(double *coeff);
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
	void deformation_of_cubic_lex(int nb_frames,
			double angle_start, double angle_max, double angle_min,
			double *coeff1, double *coeff2,
			int verbose_level);
	int cubic_Goursat_ABC(double A, double B, double C);
	int quartic(double *coeff);
	int face(int *pts, int nb_pts);
	int face3(int pt1, int pt2, int pt3);
	int face4(int pt1, int pt2, int pt3, int pt4);
	int face5(int pt1, int pt2, int pt3, int pt4, int pt5);
	void draw_lines_with_selection(int *selection, int nb_select, 
			std::string &options, std::ostream &ost);
	void draw_line_with_selection(int line_idx, 
			std::string &options, std::ostream &ost);
	void draw_lines_cij_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_cij(std::ostream &ost);
	void draw_lines_cij_with_offset(int offset, int number_of_lines, std::ostream &ost);
	void draw_lines_ai_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_ai(std::ostream &ost);
	void draw_lines_ai_with_offset(int offset, std::ostream &ost);
	void draw_lines_bj_with_selection(int *selection, int nb_select, 
			std::ostream &ost);
	void draw_lines_bj(std::ostream &ost);
	void draw_lines_bj_with_offset(int offset, std::ostream &ost);
	void draw_edges_with_selection(int *selection, int nb_select, 
			std::string &options, std::ostream &ost);
	void draw_faces_with_selection(int *selection, int nb_select, 
		double thickness_half, std::string &options, std::ostream &ost);
	void draw_face(int idx, double thickness_half, std::string &options,
			std::ostream &ost);
	void draw_planes_with_selection(int *selection, int nb_select, 
			std::string &options, std::ostream &ost);
	void draw_plane(int idx, std::string &options, std::ostream &ost);
	void draw_points_with_selection(int *selection, int nb_select, 
		double rad, std::string &options, std::ostream &ost);
	void draw_cubic_with_selection(int *selection, int nb_select, 
			std::string &options, std::ostream &ost);
	void draw_quartic_with_selection(int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quintic_with_selection(int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_octic_with_selection(int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quadric_with_selection(int *selection, int nb_select, 
			std::string &options, std::ostream &ost);
	void draw_quadric_clipped_by_plane(int quadric_idx, int plane_idx,
			std::string &options, std::ostream &ost);
	void draw_line_clipped_by_plane(int line_idx, int plane_idx,
			std::string &options, std::ostream &ost);
	int intersect_line_and_plane(int line_idx, int plane_idx, 
		int &intersection_point_idx, 
		int verbose_level);
	int intersect_line_and_line(int line1_idx, int line2_idx, 
		double &lambda, 
		int verbose_level);
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

	// Clebsch version 1:
	void clebsch_cubic();
	void clebsch_cubic_lines_a();
	void clebsch_cubic_lines_b();
	void clebsch_cubic_lines_cij();
	void Clebsch_Eckardt_points();

	// Clebsch version 2:
	void clebsch_cubic_version2();
	void clebsch_cubic_version2_Hessian();
	void clebsch_cubic_version2_lines_a();
	void clebsch_cubic_version2_lines_b();
	void clebsch_cubic_version2_lines_c();

	double distance_between_two_points(int pt1, int pt2);
	void create_five_plus_one();
	void create_Clebsch_surface(int verbose_level);
	// 1 cubic, 27 lines, 7 Eckardt points
	void create_Hilbert_Cohn_Vossen_surface(int verbose_level);
		// 1 cubic, 27 lines, 54 points, 45 planes
	void create_Hilbert_model(int verbose_level);
	void create_Cayleys_nodal_cubic(int verbose_level);
	void create_Hilbert_cube(int verbose_level);
	void create_cube(int verbose_level);
	void create_cube_and_tetrahedra(int verbose_level);
	void create_affine_space(int q, int verbose_level);
	//void create_surface_13_1(int verbose_level);
	void create_HCV_surface(int N, int verbose_level);
	void create_E4_surface(int N, int verbose_level);
	void create_twisted_cubic(int N, int verbose_level);
	void create_triangulation_of_cube(int N, int verbose_level);
	void print_a_line(int line_idx);
	void print_a_plane(int plane_idx);
	void print_a_face(int face_idx);
	void read_obj_file(std::string &fname, int verbose_level);
	void add_a_group_of_things(int *Idx, int sz, int verbose_level);
	void create_regulus(int idx, int nb_lines, int verbose_level);
	void clipping_by_cylinder(int line_idx, double r, std::ostream &ost);
	int scan1(int argc, std::string *argv, int &i, int verbose_level);
	int scan2(int argc, std::string *argv, int &i, int verbose_level);
	int read_scene_objects(int argc, std::string *argv,
			int i0, int verbose_level);
};


// #############################################################################
// tree.cpp
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
	void init(std::string &fname,
			int xmax, int ymax, int verbose_level);
	void draw(std::string &fname,
		int xmax_in, int ymax_in, int xmax, int ymax,
		int rad, 
		int f_circle, int f_circletext, int f_i, int f_edge_labels, 
		int f_has_draw_vertex_callback, 
		void (*draw_vertex_callback)(tree *T, mp_graphics *G, 
			int *v, int layer, tree_node *N, 
			int x, int y, int dx, int dy), 
		int f_embedded, int f_sideways, int f_on_circle, 
		double tikz_global_scale, double tikz_global_line_width, int verbose_level);
	void circle_center_and_radii(int xmax, int ymax, int max_depth, 
		int &x0, int &y0, int *&rad);
	void compute_DFS_ranks(int &nb_nodes, int verbose_level);
};

// #############################################################################
// tree_node.cpp
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

	int f_rotate;
	int rotation_axis_type;
		// 1 = 1,1,1
		// 2 = 0,0,1
		// 3 = custom
	double rotation_axis_custom[3];
	int boundary_type;
		// 1 = sphere
		// 2 = box

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
	double camera_sky[1000 * 3];
	double camera_location[1000 * 3];
	double camera_look_at[1000 * 3];
	//const char *camera_sky[1000];
	//const char *camera_location[1000];
	//const char *camera_look_at[1000];

	int nb_zoom;
	int zoom_round[1000];
	int zoom_start[1000];
	int zoom_end[1000];
	double zoom_clipping_start[1000];
	double zoom_clipping_end[1000];

	int nb_zoom_sequence;
	int zoom_sequence_round[1000];
	std::string zoom_sequence_text[1000];

	int nb_pan;
	int pan_round[1000];
	int pan_f_reverse[1000];
	double pan_from[1000 * 3];
	double pan_to[1000 * 3];
	double pan_center[1000 * 3];

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
	std::string round_text_text[1000];

	int nb_label;
	int label_round[1000];
	int label_start[1000];
	int label_sustain[1000];
	std::string label_gravity[1000];
	std::string label_text[1000];

	int nb_latex_label;
	int latex_label_round[1000];
	int latex_label_start[1000];
	int latex_label_sustain[1000];
	std::string latex_extras_for_praeamble[1000];
	std::string latex_label_gravity[1000];
	std::string latex_label_text[1000];
	int latex_f_label_has_been_prepared[1000];
	std::string latex_fname_base[1000];


	int nb_picture;
	int picture_round[1000];
	double picture_scale[1000];
	std::string picture_fname[1000];
	std::string picture_options[1000];

	int latex_file_count;
	int f_omit_bottom_plane;

	//const char *sky;
	//const char *location;
	//const char *look_at;
	double sky[3];
	double location[3];
	double look_at[3];

	double scale_factor;

	int f_line_radius;
	double line_radius;

	video_draw_options();
	~video_draw_options();
	int read_arguments(
			int argc, std::string *argv,
			int verbose_level);
};


// draw_bitmap.cpp:

void draw_bitmap(std::string &fname, int *M, int m, int n,
		int f_partition, int part_width,
		int nb_row_parts, int *Row_part, int nb_col_parts, int *Col_part,
		int f_box_width, int box_width,
		int f_invert_colors, int bit_depth,
		int verbose_level);
// bit_depth should be either 8 or 24.



}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPHICS_GRAPHICS_H_ */


