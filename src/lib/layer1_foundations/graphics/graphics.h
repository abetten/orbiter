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
namespace layer1_foundations {
namespace graphics {


// #############################################################################
// animate.cpp
// #############################################################################


//! creates 3D animations using Povray


class animate {
public:
	povray_job_description *Povray_job_description;
	scene *S;
	std::string output_mask;
	std::string fname_makefile;
	int nb_frames;
	video_draw_options *Opt;
	std::ofstream *fpm;
	void (*draw_frame_callback)(animate *A, int frame,
					int nb_frames_this_round, int round,
					double clipping,
					std::ostream &fp,
					int verbose_level);
	void *extra_data;
	l1_interfaces::povray_interface *Pov;

	animate();
	~animate();
	void init(
			povray_job_description *Povray_job_description,
			void *extra_data,
			int verbose_level);
	void animate_one_round(
		int round,
		int verbose_level);
	void draw_single_line(int line_idx, std::string &color,
			std::ostream &fp);
	void draw_single_quadric(int idx, std::string &color,
			std::ostream &fp);
	void draw_single_surface(int surface_idx, std::ostream &fp);
	void draw_single_surface_with_color(
			int surface_idx, std::string &color, std::ostream &fp);
	void draw_Hilbert_point(
			int point_idx, double rad,
			std::string &options, std::ostream &fp);
	void draw_Hilbert_line(int line_idx, std::string &color,
			std::ostream &fp);
	void draw_Hilbert_plane(int plane_idx, std::string &color,
			std::ostream &fp);
	void draw_Hilbert_red_line(int idx_one_based,
			std::ostream &fp);
	void draw_Hilbert_blue_line(int idx_one_based,
			std::ostream &fp);
	void draw_Hilbert_red_lines(
			std::ostream &fp);
	void draw_Hilbert_blue_lines(
			std::ostream &fp);
	void draw_Hilbert_cube_extended_edges(
			std::ostream &fp);
	void draw_Hilbert_cube_faces(
			std::ostream &fp);
	void draw_Hilbert_cube_boxed(
			std::ostream &fp);
	void draw_Hilbert_tetrahedron_boxed(
			std::ostream &fp);
	void draw_Hilbert_tetrahedron_faces(
			std::ostream &fp);
	void draw_frame_Hilbert(
		int h, int nb_frames, int round,
		double clipping_radius,
		std::ostream &fp,
		int verbose_level);
	void draw_surface_13_1(
			std::ostream &fp);
	void draw_frame_Hilbert_round_76(
			video_draw_options *Opt,
			int h, int nb_frames, int round,
			std::ostream &fp,
			int verbose_level);
		// tritangent plane, 6 arc points, 2 blue lines, 6 red lines, text
	void draw_frame_Eckardt_surface(
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
	void draw_text(
			std::string &text,
			double thickness_half, double extra_spacing,
			double scale,
			double off_x, double off_y, double off_z,
			std::string &color_options,
			int idx_point,
			std::ostream &ost, int verbose_level);
	void draw_text_with_selection(
			int *selection, int nb_select,
		double thickness_half, double extra_spacing,
		double scale,
		double off_x, double off_y, double off_z,
		std::string &options, std::string &group_options,
		std::ostream &ost, int verbose_level);
};


// #############################################################################
// draw_bitmap_control.cpp
// #############################################################################


//! options for drawing bitmap files


class draw_bitmap_control {

public:


	// TABLES/draw_bitmap_control.tex

	int f_input_csv_file;
	std::string input_csv_file_name;

	int f_secondary_input_csv_file;
	std::string secondary_input_csv_file_name;

	int f_input_object;
	std::string input_object_label;

	int f_partition;
	int part_width;
	std::string part_row;
	std::string part_col;

	int f_box_width;
	int box_width;

	int f_invert_colors;
	int bit_depth;

	int f_grayscale;


	// not a command line argument:
	int f_input_matrix;
	int *M;
	int *M2;
	int m;
	int n;


	draw_bitmap_control();
	~draw_bitmap_control();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
};


// #############################################################################
// draw_incidence_structure_description.cpp
// #############################################################################


//! options for drawing an incidence structure


class draw_incidence_structure_description {
public:

	// TABLES/draw_incidence_structure_description.tex

	int f_width;
	int width;

	int f_width_10;
	int width_10;

	// width for one box in 0.1mm
	// width_10 is 1 10th of width
	// example: width = 40, width_10 = 4 */

	int f_outline_thin;

	int f_unit_length;
	std::string unit_length;

	int f_thick_lines;
	std::string thick_lines;

	int f_thin_lines;
	std::string thin_lines;

	int f_geo_line_width;
	std::string geo_line_width;

	int v;
	int b;
	int V;
	int B;
	int *Vi;
	int *Bj;

	int f_labelling_points;
	std::string *point_labels;

	int f_labelling_blocks;
	std::string *block_labels;

	draw_incidence_structure_description();
	~draw_incidence_structure_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// draw_mod_n_description.cpp
// #############################################################################


//! options for drawing modulo n


class draw_mod_n_description {
public:

	// TABLES/draw_mod_n.tex

	int f_n;
	int n;

	int f_mod_s;
	int mod_s;

	int f_divide_out_by;
	int divide_out_by;

	int f_file;
	std::string fname;
	int f_label_nodes;
	int f_inverse;
	int f_additive_inverse;
	int f_power_cycle;
	int power_cycle_base;

	int f_cyclotomic_sets;
	int cyclotomic_sets_q;
	std::string cyclotomic_sets_reps;

	int f_cyclotomic_sets_thickness;
	int cyclotomic_sets_thickness;

	int f_eigenvalues;
	double eigenvalues_A[4];

	int f_draw_options;
	std::string draw_options_label;

	draw_mod_n_description();
	~draw_mod_n_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// draw_projective_curve_description.cpp
// #############################################################################


//! options for drawing a projective curve


class draw_projective_curve_description {
public:

	int f_number;
	int number;

	int f_file;
	std::string fname;

	int f_animate;
	int animate_nb_of_steps;

	int f_animate_with_transition;
	int animate_transition_nb_of_steps;

	int f_title_page;
	int f_trailer_page;

	int f_draw_options;
	std::string draw_options_label;



	draw_projective_curve_description();
	~draw_projective_curve_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// drawable_set_of_objects.cpp
// #############################################################################



//! a set of objects that should be drawn with certain povray properties



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
	void init_spheres(
			int group_idx, double rad,
			std::string &properties, int verbose_level);
	void init_cylinders(
			int group_idx,
			double rad, std::string &properties, int verbose_level);
	void init_prisms(
			int group_idx,
			double thickness,
			std::string &properties, int verbose_level);
	void init_planes(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_lines(
			int group_idx,
			double rad, std::string &properties, int verbose_level);
	void init_cubics(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_quadrics(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_quartics(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_quintics(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_octics(
			int group_idx,
			std::string &properties, int verbose_level);
	void init_labels(
			int group_idx,
			double thickness_half, double scale,
			std::string &properties, int verbose_level);
	void draw(
			animate *Anim, std::ostream &ost,
			int f_group_is_animated, int frame, int verbose_level);

};


// #############################################################################
// graphical_output.cpp
// #############################################################################

//! a catch-all class for things related to 2D graphics


class graphical_output {

private:

public:

	polish::function_polish *smooth_curve_Polish;
	double parabola_a;
	double parabola_b;
	double parabola_c;


	graphical_output();
	~graphical_output();
	void draw_layered_graph_from_file(
			std::string &fname,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void do_domino_portrait(
			int D, int s,
			std::string &photo_label,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void do_create_points_on_quartic(
			double desired_distance, int verbose_level);
	void do_create_points_on_parabola(
			double desired_distance, int N,
			double a, double b, double c, int verbose_level);
	void do_smooth_curve(
			std::string &curve_label,
			double desired_distance, int N,
			double t_min, double t_max, double boundary,
			polish::function_polish_description *FP_descr,
			int verbose_level);
	void draw_projective_curve(
			draw_projective_curve_description *Descr,
			int verbose_level);
	void draw_projective(
			mp_graphics &G,
			int number, int animate_step, int animate_nb_of_steps,
		int f_transition, int transition_step, int transition_nb_steps,
		int f_title_page, int title_page_step,
		int f_trailer_page, int trailer_page_step);
	void tree_draw(
			tree_draw_options *Tree_draw_options,
			int verbose_level);
	void animate_povray(
			povray_job_description *Povray_job_description,
			int verbose_level);

};


// #############################################################################
// layered_graph_draw_options.cpp
// #############################################################################

//! options for drawing an object of type layered_graph

class layered_graph_draw_options {
public:

	// TABLES/layered_graph_draw_options_1.tex
	// Section 17.1

	int f_paperheight;
	int paperheight;

	int f_paperwidth;
	int paperwidth;

	int xin; // Assume input $x$-coordinates are in the interval $[0,xin]$. Default value: 10000.
	int yin; // Assume input $y$-coordinates are in the interval $[0,yin]$. Default value: 10000.
	int xout; // Assume output $x$-coordinates are in the interval $[0,xout]$. Default value: 1000000.
	int yout; // Assume output $y$-coordinates are in the interval $[0,yout]$. Default value: 1000000.


	int f_spanning_tree;


	int f_circle;
	int f_corners;
	int rad; // Default value: 200.
	int f_embedded;
	int f_sideways;

	int f_show_level_info; // undocumented

	int f_label_edges;

	int f_x_stretch;
	double x_stretch; // Apply $x$-axis scaling by a factor of $s$. Default value: $s=1.0$.
	int f_y_stretch;
	double y_stretch; // Apply $y$-axis scaling by a factor of $s$. Default value: $s=1.0$.


	// TABLES/layered_graph_draw_options_2.tex


	int f_scale;
	double scale; // Use Tikz global scale-factor of $s$. Default value: $s=0.45$.

	int f_line_width;
	double line_width; // Set Tikz line width to $s$. Default value: $s=1.5$.

	int f_rotated; // Rotate the output.


	int f_nodes; // Turn on node drawing.
	int f_nodes_empty; // Do not label the nodes. Default value: off.

	int f_show_colors; // indicate the color in the subscript of the vertex label

	int f_select_layers;
	std::string select_layers; // Draw layers whose index is given in the list $S$ only.
	int nb_layer_select;
	int *layer_select;


	// undocumented

	int f_has_draw_begining_callback;
	void (*draw_begining_callback)(
			graph_theory::layered_graph *LG, mp_graphics *G,
		int x_max, int y_max, int f_rotated, int dx, int dy);
	int f_has_draw_ending_callback;
	void (*draw_ending_callback)(
			graph_theory::layered_graph *LG, mp_graphics *G,
		int x_max, int y_max, int f_rotated, int dx, int dy);
	int f_has_draw_vertex_callback;
	void (*draw_vertex_callback)(
			graph_theory::layered_graph *LG, mp_graphics *G,
		int layer, int node, int x, int y, int dx, int dy);


	int f_paths_in_between;
	int layer1, node1;
	int layer2, node2;

	//Draw all paths from node $(l_1,i_1)$ to node $(l_2,i_2)$.
	//Here, $(l,i)$ is the $i$-th node at layer $l$ (counting from zero).
	//Delete all other edges between layers $l_1$ and $l_2.$


	layered_graph_draw_options();
	~layered_graph_draw_options();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();
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

	layered_graph_draw_options *Draw_options;

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
		// see mp_graphics::color_tikz for tikz colors

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


public:

	mp_graphics();
	~mp_graphics();
	void init(
			std::string &file_name,
			layered_graph_draw_options *Draw_options,
		int verbose_level);
	void exit(
			std::ostream &ost, int verbose_level);
	void frame(
			double move_out);
	void frame_constant_aspect_ratio(
			double move_out);
	void finish(
			std::ostream &ost, int verbose_level);

	int& in_xmin();
	int& in_ymin();
	int& in_xmax();
	int& in_ymax();
	int& out_xmin();
	int& out_ymin();
	int& out_xmax();
	int& out_ymax();

	void user2dev(
			int &x, int &y);
	void dev2user(
			int &x, int &y);
	void user2dev_dist_x(
			int &x);
	void user2dev_dist_y(
			int &y);

	void draw_polar_grid(
			double r_max, int nb_circles,
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
	void plot_curve(
			int N, int *f_DNE,
		double *Dx, double *Dy, double dx, double dy);
	void nice_circle(
			int x, int y, int rad);
	void grid_polygon2(
			grid_frame *F, int x0, int y0,
		int x1, int y1);
	void grid_polygon4(
			grid_frame *F, int x0, int y0,
		int x1, int y1, int x2, int y2, int x3, int y3);
	void grid_polygon5(
			grid_frame *F, int x0, int y0,
		int x1, int y1, int x2, int y2, 
		int x3, int y3, int x4, int y4);
	void polygon(
			int *Px, int *Py, int n);
	void polygon2(
			int *Px, int *Py, int i1, int i2);
	void polygon3(
			int *Px, int *Py, int i1, int i2, int i3);
	void polygon4(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4);
	void polygon5(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5);
	void polygon6(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6);
	void polygon7(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6, int i7);
	void polygon8(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8);
	void polygon9(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, int i9);
	void polygon10(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, int i9, 
		int i10);
	void polygon11(
			int *Px, int *Py, int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, int i9, 
		int i10, int i11);
	void polygon_idx(
			int *Px, int *Py, int *Idx, int n);
	void bezier(
			int *Px, int *Py, int n);
	void bezier2(
			int *Px, int *Py,
			int i1, int i2);
	void bezier3(
			int *Px, int *Py,
			int i1, int i2, int i3);
	void bezier4(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4);
	void bezier5(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5);
	void bezier6(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6);
	void bezier7(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7);
	void bezier_idx(
			int *Px, int *Py, int *Idx, int n);
	void grid_fill_polygon4(
			grid_frame *F,
		int x0, int y0, int x1, int y1, int x2, 
		int y2, int x3, int y3);
	void grid_fill_polygon5(
			grid_frame *F,
		int x0, int y0, int x1, int y1, 
		int x2, int y2, int x3, int y3, 
		int x4, int y4);
	void fill_polygon3(
			int *Px, int *Py,
			int i1, int i2, int i3);
	void fill_polygon4(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4);
	void fill_polygon5(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5);
	void fill_polygon6(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6);
	void fill_polygon7(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7);
	void fill_polygon8(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8);
	void fill_polygon9(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, int i9);
	void fill_polygon10(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, int i9, int i10);
	void fill_polygon11(
			int *Px, int *Py,
			int i1, int i2, int i3,
		int i4, int i5, int i6, int i7, int i8, 
		int i9, int i10, int i11);
	void polygon2_arrow_halfway(
			int *Px, int *Py, int i1, int i2);
	void polygon2_arrow_halfway_and_label(
			int *Px, int *Py, int i1, int i2,
		const char *alignment, std::string &s);
	void grid_aligned_text(
			grid_frame *F, int x, int y,
		const char *alignment, std::string &s);
	void aligned_text(
			int x, int y, const char *alignment, std::string &s);
	void aligned_text_array(
			int *Px, int *Py, int idx,
		const char *alignment, std::string &s);
	void aligned_text_with_offset(
			int x, int y, int xoffset, int yoffset,
		const char *alignment, std::string &s);

	void st_alignment(
			int txt_halign, int txt_valign);
	void sl_udsty(
			int line_dashing);
	void sl_ends(
			int line_beg_style, int line_end_style);
	void sl_thickness(
			int line_thickness);
	void sl_color(
			int line_color);
	void sf_interior(
			int fill_interior);
	void sf_color(
			int fill_color);
	void sf_shape(
			int fill_shape);
	void sf_outline(
			int fill_outline);
	void sf_nofill(
			int fill_nofill);
	void st_boxed(
			int txt_boxed);
	void st_overwrite(
			int txt_overwrite);
	void st_rotate(
			int txt_rotate);
	void coords_min_max(
			int x, int y);

	// output commands:
	void header();
	void footer();
	void begin_figure(
			int factor_1000);
	void end_figure();

	void comment(
			std::string &s);
	void text(
			int x, int y, std::string &s);
	void circle(
			int x, int y, int rad);
	void circle_text(
			int x, int y, int rad, std::string &s);
	void polygon_idx2(
			int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void bezier_idx2(
			int *Px, int *Py, int *Idx, int n,
			int f_cycle);
	void fill_idx(
			int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle);


	// output commands log file:
	void header_log(
			std::string &str_date);
	void footer_log();
	void comment_log(
			std::string &s);
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
	void bezier_idx_log(
			int *Px, int *Py, int *Idx, int n);
	void polygon_log(
			int *Px, int *Py, int n);
	void polygon_idx_log(
			int *Px, int *Py, int *Idx, int n);
	void text_log(
			int x1, int y1, std::string &s);
	void circle_log(
			int x1, int y1, int rad);


	// output commands metapost:
	void header_mp(
			std::string &str_date);
	void footer_mp();
	void comment_mp(
			std::string &s);
	void text_mp(
			int x1, int y1, std::string &s);
	void begin_figure_mp(
			int factor_1000);
	void end_figure_mp();
	void circle_mp(
			int x, int y, int rad);
	void output_circle_text_mp(
			int x, int y, int idx, std::string &s);
	void polygon_idx_mp(
			int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void bezier_idx_mp(
			int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void color_tikz(
			std::ofstream &fp, int color);
	void fill_idx_mp(
			int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle);
	void output_xy_metapost(
			int x, int y);
	void output_x_metapost(
			int x);
	void output_y_metapost(
			int y);
	int get_label(
			int x, int y);
	void get_alignment_mp(
			char *align);
	void line_thickness_mp();

	// output commands tikz:
	void header_tikz(
			std::string &str_date);
	void footer_tikz();
	void comment_tikz(
			std::string &s);
	void text_tikz(
			int x1, int y1, std::string &s);
	void circle_tikz(
			int x, int y, int rad);
	void output_circle_text_tikz(
			int x, int y, int idx, int rad,
		const char *text);
	void polygon_idx_tikz(
			int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void bezier_idx_tikz(
			int *Px, int *Py,
			int *Idx, int n, int f_cycle);
	void fill_idx_tikz(
			std::ofstream &fp,
		int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle);
	void output_xy_tikz(
			int x, int y);
	void output_x_tikz(
			int x);
	void output_y_tikz(
			int y);

	void polygon3D(
			int *Px, int *Py,
			int dim, int x0, int y0, int z0, int x1, int y1, int z1);
	void integer_4pts(
			int *Px, int *Py,
			int p1, int p2, int p3, int p4,
			const char *align, int a);
	void text_4pts(
			int *Px, int *Py, int p1, int p2, int p3, int p4,
		const char *align, std::string &s);


	void draw_graph(
			int x, int y,
			int dx, int dy, int nb_V,
			long int *Edges, int nb_E, int radius,
			int verbose_level);
	void draw_graph_with_distinguished_edge(
		int x, int y,
		int dx, int dy, int nb_V, long int *Edges, int nb_E,
		int distinguished_edge, int verbose_level);
	void draw_graph_on_multiple_circles(
			int x, int y,
			int dx, int dy, int nb_V,
			int *Edges, int nb_E, int nb_circles);
	void draw_graph_on_2D_grid(
			int x, int y, int dx, int dy, int rad, int nb_V,
			int *Edges, int nb_E, int *coords_2D, int *Base,
			int f_point_labels,
			int point_label_offset, int f_directed);
	void draw_tournament(
			int x, int y,
			int dx, int dy, int nb_V, long int *Edges, int nb_E,
			int radius,
			int verbose_level);
	void draw_bitmatrix2(
			int f_dots,
		int f_partition,
		int nb_row_parts, int *row_part_first,
		int nb_col_parts, int *col_part_first,
		int f_row_grid, int f_col_grid,
		int f_bitmatrix,
		data_structures::bitmatrix *Bitmatrix, int *M,
		int m, int n,
		int f_has_labels, int *labels);

	void draw_density2(
			int no,
		int *outline_value, int *outline_number, int outline_sz,
		int min_value, int max_value,
		int offset_x, int f_switch_x,
		int f_title, std::string &title,
		std::string &label_x,
		int f_circle, int circle_at, int circle_rad,
		int f_mu, int f_sigma, int nb_standard_deviations,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid);
	void draw_density2_multiple_curves(
			int no,
		int **outline_value, int **outline_number,
		int *outline_sz, int nb_curves,
		int min_x, int max_x, int min_y, int max_y,
		int offset_x, int f_switch_x,
		int f_title, std::string &title,
		std::string &label_x,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int f_v_logarithmic, double log_base);
	void projective_plane_draw_grid2(
			layered_graph_draw_options *O,
			int q,
			int *Table, int nb,
			int f_point_labels,
			std::string *Point_labels, int verbose_level);
	void draw_matrix_in_color(
		int f_row_grid, int f_col_grid,
		int *Table, int nb_colors,
		int m, int n,
		int *color_scale, int nb_colors_in_scale,
		int f_has_labels, int *labels);
	void domino_draw1(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw2(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw3(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw4(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw5(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw6(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw7(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw8(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw9(
			int M,
			int i, int j, int dx, int dy, int rad, int f_horizontal);
	void domino_draw_assignment_East(
			int Ap, int Aq, int M,
			int i, int j, int dx, int dy, int rad);
	void domino_draw_assignment_South(
			int Ap, int Aq, int M,
			int i, int j, int dx, int dy, int rad);
	void domino_draw_assignment(
			int *A, int *matching, int *B,
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
	void init(
			double t, int f_is_valid, double *x,
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
	int (*compute_point_function)(
			double t,
			double *pt, void *extra_data, int verbose_level);
	void *extra_data;
	double boundary;

	int nb_pts;
	std::vector<parametric_curve_point> Pts;

	parametric_curve();
	~parametric_curve();
	void init(
			int nb_dimensions,
			double desired_distance,
			double t0, double t1,
			int (*compute_point_function)(
					double t,
					double *pt, void *extra_data, int verbose_level),
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

	void draw_density(
			layered_graph_draw_options *Draw_options,
			std::string &prefix, int *the_set, int set_size,
		int f_title, std::string &title, int out_of,
		std::string &label_x,
		int f_circle, int circle_at, int circle_rad,
		int f_mu, int f_sigma, int nb_standard_deviations,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int offset_x,
		int f_switch_x, int no, int f_embedded,
		int verbose_level);
	void draw_density_multiple_curves(
			layered_graph_draw_options *Draw_options,
			std::string &prefix,
		int **Data, int *Data_size, int nb_data_sets,
		int f_title, std::string &title, int out_of,
		std::string &label_x,
		int f_v_grid, int v_grid, int f_h_grid, int h_grid,
		int offset_x, int f_switch_x,
		int f_v_logarithmic, double log_base, int no, int f_embedded,
		int verbose_level);
	void get_coord(
			int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y, int f_switch_x);
	void get_coord_log(
			int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y,
		double log_base, int f_switch_x);
	void y_to_pt_on_curve(
			int y_in, int &x, int &y,
		int *outline_value, int *outline_number, int outline_sz);
	void projective_plane_draw_grid(
			std::string &fname,
			layered_graph_draw_options *O,
			int q, int *Table, int nb,
			int f_point_labels, std::string *Point_labels,
			int verbose_level);
	void draw_mod_n(
			draw_mod_n_description *Descr,
			int verbose_level);
	void draw_mod_n_work(
			mp_graphics &G,
			layered_graph_draw_options *O,
			draw_mod_n_description *Descr,
			int verbose_level);
	void draw_point_set_in_plane(
		std::string &fname,
		layered_graph_draw_options *O,
		geometry::projective_geometry::projective_space *P,
		long int *Pts, int nb_pts,
		int f_point_labels,
		int verbose_level);

};




// #############################################################################
// povray_job_description.cpp
// #############################################################################

//! description of a povray job



class povray_job_description {
public:

	int f_output_mask;
	std::string output_mask;
	int f_nb_frames_default;
	int nb_frames_default;
	int f_round;
	int round;
	int f_rounds;
	std::string rounds_as_string;
	video_draw_options *Video_draw_options;

	// for povray_worker:
	scene *S;

	povray_job_description();
	~povray_job_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// scene_element_of_type_edge.cpp
// #############################################################################


//! a scene element of type edge



class scene_element_of_type_edge {

private:

	std::vector<std::string> Idx;
		// labels of two points

public:
	scene_element_of_type_edge();
	~scene_element_of_type_edge();
	void init(
			std::string &pt1, std::string &pt2);
	void print();

};





// #############################################################################
// scene_element_of_type_face.cpp
// #############################################################################


//! a scene element of type face



class scene_element_of_type_face {

private:

	std::vector<std::string> Pts;
		// labels of the points

public:
	scene_element_of_type_face();
	~scene_element_of_type_face();
	void init(
			std::vector<std::string> &pts);
	void print();

};






// #############################################################################
// scene_element_of_type_line.cpp
// #############################################################################


//! a scene element of type line



class scene_element_of_type_line {

private:

	double Line_coords[6];
	// a line is given by two points

public:
	scene_element_of_type_line();
	~scene_element_of_type_line();
	void init(
			double *coord6);
	void print();

};




// #############################################################################
// scene_element_of_type_plane.cpp
// #############################################################################


//! a scene element of type plane



class scene_element_of_type_plane {

private:

	double Plane_coords[4];

public:
	scene_element_of_type_plane();
	~scene_element_of_type_plane();
	void init(
			double *coord4);
	void print();

};







// #############################################################################
// scene_element_of_type_point.cpp
// #############################################################################


//! a scene element of type point



class scene_element_of_type_point {

private:

	double Point_coords[3];

public:
	scene_element_of_type_point();
	~scene_element_of_type_point();
	void init(
			double *coord3);
	void print();

};



// #############################################################################
// scene_element_of_type_surface.cpp
// #############################################################################


//! a scene element of type surface



class scene_element_of_type_surface {

private:

	int d;
	int nb_coeffs;
	double *Eqn;

public:
	scene_element_of_type_surface();
	~scene_element_of_type_surface();
	void init(
			int d, int nb_coeffs, double *coords);
	void print();

};





// #############################################################################
// scene.cpp
// #############################################################################

#define SCENE_MAX_LINES    100000
#define SCENE_MAX_EDGES    100000
#define SCENE_MAX_POINTS   200000
#define SCENE_MAX_PLANES    10000
#define SCENE_MAX_FACES    200000


#define SCENE_MAX_QUADRICS  10000
#define SCENE_MAX_OCTICS      100
#define SCENE_MAX_QUARTICS   1000
#define SCENE_MAX_QUINTICS    500
#define SCENE_MAX_CUBICS    10000


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

	// scene_init.cpp:
	int line6(
			double *x6);
	int line(
			double x1, double x2, double x3,
		double y1, double y2, double y3);
	int point(
			double x1, double x2, double x3);
	int edge(
			int pt1, int pt2);
	int plane(
			double x1, double x2, double x3, double a);
		// A plane is called a polynomial shape because
		// it is defined by a first order polynomial equation.
		// Given a plane: plane { <A, B, C>, D }
		// it can be represented by the equation
		// A*x + B*y + C*z - D*sqrt(A^2 + B^2 + C^2) = 0.
		// see http://www.povray.org/documentation/view/3.6.1/297/
	int quadric(
			double *coeff10);
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
	int cubic(
			double *coeff20);
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
	int quartic(
			double *coeff35);
	int quintic(
			double *coeff_56);
	int octic(
			double *coeff_165);
	int face(
			int *pts, int nb_pts);
	int face3(
			int pt1, int pt2, int pt3);
	int face4(
			int pt1, int pt2, int pt3, int pt4);
	int face5(
			int pt1, int pt2, int pt3, int pt4, int pt5);



	int line_pt_and_dir(
			double *x6, double rad, int verbose_level);
	int line_pt_and_dir_and_copy_points(
			double *x6, double rad, int verbose_level);
	int line_through_two_pts(
			double *x6, double rad);
	int line_after_recentering(
			double x1, double x2, double x3,
		double y1, double y2, double y3, double rad);
	int line_through_two_points(
			int pt1, int pt2,
		double rad);
	int plane_through_three_points(
			int pt1, int pt2, int pt3);
	int quadric_through_three_lines(
			int line_idx1,
		int line_idx2, int line_idx3, int verbose_level);
	int cubic_in_orbiter_ordering(
			double *coeff);
	void deformation_of_cubic_lex(
			int nb_frames,
			double angle_start, double angle_max, double angle_min,
			double *coeff1, double *coeff2,
			int verbose_level);
	int cubic_Goursat_ABC(
			double A, double B, double C);
	int line_extended(
			double x1, double x2, double x3,
		double y1, double y2, double y3,
		double r);


	scene();
	~scene();
	double label(
			int idx, std::string &txt);
	double point_coords(
			int idx, int j);
	double line_coords(
			int idx, int j);
	double plane_coords(
			int idx, int j);
	double cubic_coords(
			int idx, int j);
	double quadric_coords(
			int idx, int j);
	int edge_points(
			int idx, int j);
	void print_point_coords(
			int idx);
	double point_distance_euclidean(
			int pt_idx, double *y);
	double point_distance_from_origin(
			int pt_idx);
	double distance_euclidean_point_to_point(
			int pt1_idx, int pt2_idx);
	void init(
			int verbose_level);
	scene *transformed_copy(
			double *A4, double *A4_inv,
		double rad, int verbose_level);
	void print();
	void transform_lines(
			scene *S, double *A4, double *A4_inv,
		double rad, int verbose_level);
	void copy_edges(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_points(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_planes(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_quadrics(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_cubics(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_quartics(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void transform_quintics(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void copy_faces(
			scene *S, double *A4, double *A4_inv,
		int verbose_level);
	void points(
			double *Coords, int nb_points);
	int point_center_of_mass_of_face(
			int face_idx);
	int point_center_of_mass_of_edge(
			int edge_idx);
	int point_center_of_mass(
			int *Pt_idx, int nb_pts);
	int triangle(
			int line1, int line2, int line3,
			int verbose_level);
	int point_as_intersection_of_two_lines(
			int line1, int line2);
	int plane_from_dual_coordinates(
			double *x4);
	void draw_lines_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_line_with_selection(
			int line_idx,
			std::string &options, std::ostream &ost);
	void draw_lines_cij_with_selection(
			int *selection, int nb_select,
			std::ostream &ost);
	void draw_lines_cij(
			std::ostream &ost);
	void draw_lines_cij_with_offset(
			int offset,
			int number_of_lines, std::ostream &ost);
	void draw_lines_ai_with_selection(
			int *selection, int nb_select,
			std::ostream &ost);
	void draw_lines_ai(
			std::ostream &ost);
	void draw_lines_ai_with_offset(
			int offset, std::ostream &ost);
	void draw_lines_bj_with_selection(
			int *selection, int nb_select,
			std::ostream &ost);
	void draw_lines_bj(
			std::ostream &ost);
	void draw_lines_bj_with_offset(
			int offset, std::ostream &ost);
	void draw_edges_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_faces_with_selection(
			int *selection, int nb_select,
		double thickness_half, std::string &options, std::ostream &ost);
	void draw_face(
			int idx, double thickness_half, std::string &options,
			std::ostream &ost);
	void draw_planes_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_plane(
			int idx, std::string &options, std::ostream &ost);
	void draw_points_with_selection(
			int *selection, int nb_select,
		double rad, std::string &options, std::ostream &ost);
	void draw_cubic_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quartic_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quintic_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_octic_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quadric_with_selection(
			int *selection, int nb_select,
			std::string &options, std::ostream &ost);
	void draw_quadric_clipped_by_plane(
			int quadric_idx, int plane_idx,
			std::string &options, std::ostream &ost);
	void draw_line_clipped_by_plane(
			int line_idx, int plane_idx,
			std::string &options, std::ostream &ost);
	int intersect_line_and_plane(
			int line_idx, int plane_idx,
		int &intersection_point_idx, 
		int verbose_level);
	int intersect_line_and_line(
			int line1_idx, int line2_idx,
		double &lambda, 
		int verbose_level);
	void map_a_line(
			int line1, int line2,
		int plane_idx, int line_idx, double spread, 
		int nb_pts, 
		int *New_line_idx, int &nb_new_lines, 
		int *New_pt_idx, int &nb_new_points, int verbose_level);
	int map_a_point(
			int line1, int line2,
		int plane_idx, double pt_in[3], 
		int &new_line_idx, int &new_pt_idx, 
		int verbose_level);
	void fourD_cube(
			double rad_desired);
	void rescale(
			int first_pt_idx, double rad_desired);
	double euclidean_distance(
			int pt1, int pt2);
	double distance_from_origin(
			int pt);
	void fourD_cube_edges(
			int first_pt_idx);
	void hypercube(
			int n, double rad_desired);
	void Dodecahedron_points();
	void Dodecahedron_edges(
			int first_pt_idx);
	void Dodecahedron_planes(
			int first_pt_idx);
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

	double distance_between_two_points(
			int pt1, int pt2);
	void create_five_plus_one();
	void create_Clebsch_surface(
			int verbose_level);
	// 1 cubic, 27 lines, 7 Eckardt points
	void create_Hilbert_Cohn_Vossen_surface(
			int verbose_level);
		// 1 cubic, 27 lines, 54 points, 45 planes
	void create_Hilbert_model(
			int verbose_level);
	void create_Cayleys_nodal_cubic(
			int verbose_level);
	void create_Hilbert_cube(
			int verbose_level);
	void create_cube(
			int verbose_level);
	void create_cube_and_tetrahedra(
			int verbose_level);
	void create_affine_space(
			int q, int verbose_level);
	//void create_surface_13_1(int verbose_level);
	void create_Eckardt_surface(
			int N, int verbose_level);
	void create_E4_surface(
			int N, int verbose_level);
	void create_twisted_cubic(
			int N, int verbose_level);
	void create_triangulation_of_cube(
			int N, int verbose_level);
	void print_a_line(
			int line_idx);
	void print_a_plane(
			int plane_idx);
	void print_a_face(
			int face_idx);
	void read_obj_file(
			std::string &fname, int verbose_level);
	void add_a_group_of_things(
			int *Idx, int sz, int verbose_level);
	void create_regulus(
			int idx, int nb_lines, int verbose_level);
	void clipping_by_cylinder(
			int line_idx, double r, std::ostream &ost);
	int scan1(
			int argc, std::string *argv, int &i, int verbose_level);
	int scan2(
			int argc, std::string *argv, int &i, int verbose_level);
	int read_scene_objects(
			int argc, std::string *argv,
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
	int *f_node_select;
	
	int *path;

	int f_count_leaves;
	int leaf_count;

	tree();
	~tree();
	void init(
			graphics::tree_draw_options *Tree_draw_options,
			int xmax, int ymax,
			int verbose_level);
	void draw(
			std::string &fname,
			graphics::tree_draw_options *Tree_draw_options,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void draw_preprocess(
			std::string &fname,
			graphics::tree_draw_options *Tree_draw_options,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void circle_center_and_radii(
			int xmax, int ymax, int max_depth,
		int &x0, int &y0, int *&rad);
	void compute_DFS_ranks(
			int &nb_nodes, int verbose_level);
};

// #############################################################################
// tree_draw_options.cpp
// #############################################################################


//! options for drawing a tree


class tree_draw_options {

public:

	// TABLES/tree_draw_options.tex

	int f_file;
	std::string file_name;

	int f_restrict;
	int restrict_excluded_color;

	int f_select_path;
	std::string select_path_text;

	int f_has_draw_vertex_callback;
	void (*draw_vertex_callback)(tree *T,
		mp_graphics *G, int *v, int layer, tree_node *N,
		int x, int y, int dx, int dy);

	int f_draw_options;
	std::string draw_options_label;

	tree_draw_options();
	~tree_draw_options();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

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
	
	int f_has_color;
	int color;

	std::string label;

	int nb_children;
	tree_node **children;

	int weight;
	int placement_x;
	int placement_y;
	int width;

	int DFS_rank;

	tree_node();
	~tree_node();
	void init(
			int depth, tree_node *parent, int f_value, int value,
		int f_has_color, int color, std::string &label,
		int verbose_level);
	void print_path();
	void print_depth_first();
	void compute_DFS_rank(
			int &rk);
	int find_node(
			int &DFS_rk, int *path, int sz, int verbose_level);
	int find_node_and_path(
			std::vector<int> &Rk, int *path,
			int sz, int verbose_level);
	void get_coordinates(
			int &idx, int *coord_xy);
	void get_coordinates_and_width(
			int &idx, int *coord_xyw);
	void calc_weight();
	void place_xy(
			int left, int right, int ymax, int max_depth);
	void place_on_circle(
			int xmax, int ymax, int max_depth);
	void add_node(
			int l, int depth, int *path,
			int color, std::string &label,
		int verbose_level);
	int find_child(
			int val);
	void get_values(
			int *v, int verbose_level);
	void draw_edges(
			mp_graphics &G,
			tree_draw_options *Tree_draw_options,
			layered_graph_draw_options *Opt,
		int f_has_parent, int parent_x, int parent_y, int max_depth,
		tree *T, int verbose_level);
	void draw_vertices(
			mp_graphics &G,
			tree_draw_options *Tree_draw_options,
			layered_graph_draw_options *Opt,
		int f_has_parent, int parent_x, int parent_y, int max_depth,
		tree *T, int verbose_level);
	void draw_sideways(
			mp_graphics &G, int f_circletext, int f_i,
		int f_has_parent, int parent_x, int parent_y, 
		int max_depth, int f_edge_labels);
	int calc_y_coordinate(
			int ymax, int l, int max_depth);

};


// #############################################################################
// video_draw_options.cpp:
// #############################################################################


//! options for povray videos


class video_draw_options {
public:


	// TABLES/video_draw_options_1.tex

	int f_rotate;
	int rotation_axis_type;
		// 1 = 1,1,1
		// 2 = 0,0,1
		// 3 = custom

	double rotation_axis_custom[3];

	int boundary_type;
		// 1 = sphere
		// 2 = box

	int f_has_font_size;
	int font_size;

	int f_has_stroke_width;
	int stroke_width;


	int f_omit_bottom_plane;

	int f_W;
	int W;
	int f_H;
	int H;



	double sky[3];
	double location[3];
	int f_look_at;
	double look_at[3];


	int f_has_global_picture_scale;
	double global_picture_scale;



	int f_default_angle; // = false;
	int default_angle; // = 22;

	int f_clipping_radius; // = true;
	double clipping_radius; // = 0.9;

	int nb_zoom;
	int zoom_round[1000];
	int zoom_start[1000];
	int zoom_end[1000];
	double zoom_clipping_start[1000];
	double zoom_clipping_end[1000];


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


	int nb_camera;
	int camera_round[1000];
	double camera_sky[1000 * 3];
	double camera_location[1000 * 3];
	double camera_look_at[1000 * 3];




	// TABLES/video_draw_options_2.tex

	int nb_clipping;
	int clipping_round[1000];
	double clipping_value[1000];

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



	int nb_zoom_sequence;
	int zoom_sequence_round[1000];
	std::string zoom_sequence_text[1000];

	int cnt_nb_frames;
	int nb_frames_round[1000];
	int nb_frames_value[1000];



	int nb_picture;
	int picture_round[1000];
	double picture_scale[1000];
	std::string picture_fname[1000];
	std::string picture_options[1000];

	int latex_file_count;

	int f_scale_factor;
	double scale_factor;

	int f_line_radius;
	double line_radius;

	video_draw_options();
	~video_draw_options();
	int read_arguments(
			int argc, std::string *argv,
			int verbose_level);
	void print();
};





}}}



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPHICS_GRAPHICS_H_ */


