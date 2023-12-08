// mp_graphics.cpp
//
// Anton Betten
// March 6, 2003

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {


static void projective_plane_make_affine_point(
		int q, int x1, int x2, int x3,
	double &a, double &b);
static int compute_dd(int dx);


mp_graphics::mp_graphics()
{
	Draw_options = NULL;

	//std::string fname_mp;
	//std::string fname_log;
	//std::string fname_tikz;
	//std::ofstream fp_mp;
	//std::ofstream fp_log;
	//std::ofstream fp_tikz;


	f_file_open = false;
	//int user[4]; // llx/lly/urx/ury
	//int dev[4]; // llx/lly/urx/ury

	x_min = INT_MAX;
	x_max = INT_MIN;
	y_min = INT_MAX;
	y_max = INT_MIN;
	f_min_max_set = false;

	txt_halign = 0;
	txt_valign = 0;
	txt_boxed = 0;
	txt_overwrite = 0;
	txt_rotate = 0;
	line_beg_style = 0;
	line_end_style = 0; // if 1, draw an arrow
	line_dashing = 0; // between 0 and 100  sl_udsty
	line_thickness = 100; // 100 is the old 1
	line_color = 1;
	fill_interior = 0;
	fill_color = 0;
	fill_shape = 1;
	fill_outline = 0;
	fill_nofill = 0;

	cur_path = 1;

}


mp_graphics::~mp_graphics()
{
	// cout << "mp_graphics::~mp_graphics()" < endl;
	exit(cout, 1);
}

void mp_graphics::init(
		std::string &file_name,
		layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "mp_graphics::init" << endl;
		cout << "mp_graphics::init file_name=" << file_name << endl;
	}
	mp_graphics::Draw_options = Draw_options;

	fname_mp.assign(file_name);
	fname_log.assign(file_name);
	fname_tikz.assign(file_name);

	ST.replace_extension_with(fname_mp, ".mp");
	ST.replace_extension_with(fname_log, ".commands");
	ST.replace_extension_with(fname_tikz, ".tex");
	
	fp_mp.open(fname_mp);
	fp_log.open(fname_log);
	fp_tikz.open(fname_tikz);
	f_file_open = true;
	
	user[0] = 0; //xmin;
	user[1] = 0; //ymin;
	user[2] = Draw_options->xin; // xmax;
	user[3] = Draw_options->yin; // ymax;
	
	// the identity transformation:
	
	dev[0] = 0; // xmin;
	dev[1] = 0; // ymin;
	dev[2] = Draw_options->xout; // xmax;
	dev[3] = Draw_options->yout; // ymax;
	if (f_v) {
		cout << "mp_graphics::init done" << endl;
	}
}

void mp_graphics::exit(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "mp_graphics::exit" << endl;
	}
	if (f_file_open) {
		fp_mp.close();
		fp_log.close();
		fp_tikz.close();
		if (f_v) {
			ost << "mp_graphics::exit "
					"written file " << fname_mp
					<< " of size " << Fio.file_size(fname_mp) << endl;
			ost << "written file " << fname_log
					<< " of size " << Fio.file_size(fname_log) << endl;
			ost << "written file " << fname_tikz
					<< " of size " << Fio.file_size(fname_tikz) << endl;
		}
		f_file_open = false;
	}
	else {
		if (f_v) {
			cout << "mp_graphics::exit f_file_open is false" << endl;
		}
	}
	if (f_v) {
		cout << "mp_graphics::exit done" << endl;
	}
}

void mp_graphics::frame(double move_out)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int Px[30];
	int Py[30];
	int x, y, dx, dy, ox, oy, Ox, Oy;
	int xmin, ymin, xmax, ymax;
	

	if (f_v) {
		cout << "mp_graphics::frame" << endl;
	}


	if (f_min_max_set) {

		if (f_v) {
			cout << "mp_graphics::frame f_min_max_set" << endl;
			cout << "out_xmin()=" << out_xmin() << endl;
			cout << "out_xmax()=" << out_xmax() << endl;
			cout << "out_ymin()=" << out_ymin() << endl;
			cout << "out_ymax()=" << out_ymax() << endl;
			cout << "x_min=" << x_min << endl;
			cout << "x_max=" << x_max << endl;
			cout << "y_min=" << y_min << endl;
			cout << "y_max=" << y_max << endl;
		}
		int xmin_d, xmax_d;
		int ymin_d, ymax_d;

		xmin_d = x_min;
		xmax_d = x_max;
		ymin_d = y_min;
		ymax_d = y_max;
		user2dev(xmin_d, ymin_d);
		user2dev(xmax_d, ymax_d);
		if (f_v) {
			cout << "xmin_d=" << xmin_d << endl;
			cout << "xmax_d=" << xmax_d << endl;
			cout << "ymin_d=" << ymin_d << endl;
			cout << "ymax_d=" << ymax_d << endl;
		}

		xmin = out_xmin();
		xmax = out_xmax();
		ymin = ymin_d;
		ymax = ymax_d;

	}
	else {
		xmin = out_xmin();
		xmax = out_xmax();
		ymin = out_ymin();
		ymax = out_ymax();


	}

	dx = out_xmax() - out_xmin();
	dy = out_ymax() - out_ymin();
	
	ox = (int)(dx * .05);
	oy = (int)(dy * .05);
	Ox = (int)(dx * move_out);
	Oy = (int)(dy * move_out);
	xmin -= Ox;
	xmax += Ox;
	ymin -= Oy;
	ymax += Oy;


	x = xmin;
	y = ymin;
	dev2user(x, y);
	Px[0] = x;
	Py[0] = y;
	//cout << "Px[0]=" << Px[0] << endl;
	//cout << "Py[0]=" << Py[0] << endl;
	x = xmin + ox;
	y = ymin;
	dev2user(x, y);
	Px[1] = x;
	Py[1] = y;
	x = xmin;
	y = ymin + oy;
	dev2user(x, y);
	Px[2] = x;
	Py[2] = y;
	
	x = xmax;
	y = ymin;
	dev2user(x, y);
	Px[3] = x;
	Py[3] = y;
	x = xmax - ox;
	y = ymin;
	dev2user(x, y);
	Px[4] = x;
	Py[4] = y;
	x = xmax;
	y = ymin + oy;
	dev2user(x, y);
	Px[5] = x;
	Py[5] = y;
	
	x = xmax;
	y = ymax;
	dev2user(x, y);
	Px[6] = x;
	Py[6] = y;
	//cout << "Px[6]=" << Px[6] << endl;
	//cout << "Py[6]=" << Py[6] << endl;
	x = xmax - ox;
	y = ymax;
	dev2user(x, y);
	Px[7] = x;
	Py[7] = y;
	x = xmax;
	y = ymax - oy;
	dev2user(x, y);
	Px[8] = x;
	Py[8] = y;
	
	x = xmin;
	y = ymax;
	dev2user(x, y);
	Px[9] = x;
	Py[9] = y;
	x = xmin + ox;
	y = ymax;
	dev2user(x, y);
	Px[10] = x;
	Py[10] = y;
	x = xmin;
	y = ymax - oy;
	dev2user(x, y);
	Px[11] = x;
	Py[11] = y;

	if (f_v) {
		for (int i = 0; i < 12; i++) {
			cout << "Px[" << i << "]=" << Px[i] << " ";
			cout << "Py[" << i << "]=" << Py[i] << " ";
			cout << endl;
		}
	}

	polygon3(Px, Py, 2, 0, 1);
	polygon3(Px, Py, 4, 3, 5);
	polygon3(Px, Py, 7, 6, 8);
	polygon3(Px, Py, 11, 9, 10);
}

void mp_graphics::frame_constant_aspect_ratio(double move_out)
{
	int Px[30];
	int Py[30];
	int x, y, dx, dy, ox, oy, Ox, Oy;
	int xmin, ymin, xmax, ymax;
	
	xmin = out_xmin();
	xmax = out_xmax();
	ymin = out_ymin();
	ymax = out_ymax();
	
	cout << "mp_graphics::frame_constant_aspect_ratio:" << endl;
	cout << xmin << "," << xmax << "," << ymin << "," << ymax << endl;
	dx = xmax - xmin;
	dy = ymax - ymin;
	
	int adjust_x = 0;
	int adjust_y = 0;
	
	if (dx > dy) {
		adjust_y = (dx - dy) >> 1;
		cout << "mp_graphics::frame_constant_aspect_ratio "
				"adjust_y=" << adjust_y << endl;
	}
	else {
		adjust_x = (dy - dx) >> 1;
		cout << "mp_graphics::frame_constant_aspect_ratio "
				"adjust_x=" << adjust_x << endl;
	}
	xmin -= adjust_x;
	xmax += adjust_x;
	ymin -= adjust_y;
	ymax += adjust_y;
	cout << "mp_graphics::frame_constant_aspect_ratio after "
			"adjustment:" << endl;
	cout << xmin << "," << xmax << "," << ymin << "," << ymax << endl;
	dx = xmax - xmin;
	dy = ymax - ymin;
	
	ox = (int)(dx * .05);
	oy = (int)(dy * .05);
	Ox = (int)(dx * move_out);
	Oy = (int)(dy * move_out);
	xmin -= Ox;
	xmax += Ox;
	ymin -= Oy;
	ymax += Oy;

	x = xmin;
	y = ymin;
	dev2user(x, y);
	Px[0] = x;
	Py[0] = y;
	//cout << "Px[0]=" << Px[0] << endl;
	//cout << "Py[0]=" << Py[0] << endl;
	x = xmin + ox;
	y = ymin;
	dev2user(x, y);
	Px[1] = x;
	Py[1] = y;
	x = xmin;
	y = ymin + oy;
	dev2user(x, y);
	Px[2] = x;
	Py[2] = y;
	
	x = xmax;
	y = ymin;
	dev2user(x, y);
	Px[3] = x;
	Py[3] = y;
	x = xmax - ox;
	y = ymin;
	dev2user(x, y);
	Px[4] = x;
	Py[4] = y;
	x = xmax;
	y = ymin + oy;
	dev2user(x, y);
	Px[5] = x;
	Py[5] = y;
	
	x = xmax;
	y = ymax;
	dev2user(x, y);
	Px[6] = x;
	Py[6] = y;
	//cout << "Px[6]=" << Px[6] << endl;
	//cout << "Py[6]=" << Py[6] << endl;
	x = xmax - ox;
	y = ymax;
	dev2user(x, y);
	Px[7] = x;
	Py[7] = y;
	x = xmax;
	y = ymax - oy;
	dev2user(x, y);
	Px[8] = x;
	Py[8] = y;
	
	x = xmin;
	y = ymax;
	dev2user(x, y);
	Px[9] = x;
	Py[9] = y;
	x = xmin + ox;
	y = ymax;
	dev2user(x, y);
	Px[10] = x;
	Py[10] = y;
	x = xmin;
	y = ymax - oy;
	dev2user(x, y);
	Px[11] = x;
	Py[11] = y;

	polygon3(Px, Py, 2, 0, 1);
	polygon3(Px, Py, 4, 3, 5);
	polygon3(Px, Py, 7, 6, 8);
	polygon3(Px, Py, 11, 9, 10);
}

void mp_graphics::finish(std::ostream &ost, int verbose_level)
{
	end_figure();
	footer();
	exit(cout, verbose_level - 1);
}

int& mp_graphics::in_xmin()
{
	return user[0];
}

int& mp_graphics::in_ymin()
{
	return user[1];
}

int& mp_graphics::in_xmax()
{
	return user[2];
}

int& mp_graphics::in_ymax()
{
	return user[3];
}

int& mp_graphics::out_xmin()
{
	return dev[0];
}

int& mp_graphics::out_ymin()
{
	return dev[1];
}

int& mp_graphics::out_xmax()
{
	return dev[2];
}

int& mp_graphics::out_ymax()
{
	return dev[3];
}

void mp_graphics::user2dev(int &x, int &y)
{
	orbiter_kernel_system::numerics Num;

	Num.transform_llur(user, dev, x, y);
}

void mp_graphics::dev2user(int &x, int &y)
{
	orbiter_kernel_system::numerics Num;

	Num.transform_llur(dev, user, x, y);
}

void mp_graphics::user2dev_dist_x(int &x)
{
	orbiter_kernel_system::numerics Num;

	Num.transform_dist_x(user, dev, x);
}

void mp_graphics::user2dev_dist_y(int &y)
{
	orbiter_kernel_system::numerics Num;

	Num.transform_dist_y(user, dev, y);
}

void mp_graphics::draw_polar_grid(double r_max,
		int nb_circles, int nb_rays, double x_stretch)
{
	int N = 1000;
	int number;
	int i;
	double dr;
	int *Px;
	int *Py;
	double *Dx;
	double *Dy;
	double dx = 1.;
	double dy = 1.;
	orbiter_kernel_system::numerics Num;

	dr = r_max / nb_circles;

	sl_thickness(100);	
	//G.sf_color(1);
	//G.sf_interior(10);
	Px = NEW_int(N);
	Py = NEW_int(N);
	Dx = new double[N];
	Dy = new double[N];

	sf_interior(0);
	sf_color(0);
	Px[0] = 0;
	Py[0] = 0;
	for (i = 1; i <= nb_circles; i++) {
		circle(Px[0], Py[0], (int)dx * dr * i * x_stretch);
	}

	number = nb_rays;
	for (i = 0; i < number; i++) {
		Num.on_circle_double(Dx, Dy, i, i * 360 / number, r_max);
	}
	for (i = 0; i < number; i++) {
		Px[i] = Dx[i] * dx * x_stretch;
		Py[i] = Dy[i] * dy * x_stretch;
	}
	Px[number] = 0;
	Py[number] = 0;
	for (i = 0; i < number; i++) {
		polygon2(Px, Py, number, i);
	}
		
}

void mp_graphics::draw_axes_and_grid(
		layered_graph_draw_options *O,
	double x_min, double x_max,
	double y_min, double y_max,
	double dx, double dy,
	int f_x_axis_at_y_min, int f_y_axis_at_x_min, 
	int x_mod, int y_mod, int x_tick_mod, int y_tick_mod, 
	double x_labels_offset, double y_labels_offset, 
	double x_tick_half_width, double y_tick_half_width, 
	int f_v_lines, int subdivide_v, int f_h_lines, int subdivide_h,
	int verbose_level)
{
	double *Dx, *Dy;
	int *Px, *Py;
	int N = 1000;
	int n;
	int i, j, h;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mp_graphics::draw_axes_and_grid" << endl;
		cout << "mp_graphics::draw_axes_and_grid "
				"dx=" << dx << " dy=" << dy << endl;
		cout << "mp_graphics::draw_axes_and_grid "
				"x_min=" << x_min << " x_max=" << x_max << endl;
		cout << "mp_graphics::draw_axes_and_grid "
				"y_min=" << y_min << " y_max=" << y_max << endl;
	}

	

	sl_thickness(100);	
	//sf_color(1);
	//sf_interior(10);
	Px = NEW_int(N);
	Py = NEW_int(N);
	Dx = new double[N];
	Dy = new double[N];


	double y0;
	double x0;

	if (f_x_axis_at_y_min) {
		y0 = y_min;
	}
	else {
		y0 = 0.;
	}
	if (f_y_axis_at_x_min) {
		x0 = x_min;
	}
	else {
		x0 = 0.;
	}
	// draw axes (with arrows):
	Dx[0] = x_min;
	Dy[0] = y0;
	Dx[1] = x_max - y_labels_offset;
	Dy[1] = y0;

	Dx[2] = x0;
	Dy[2] = y_min;
	Dx[3] = x0;
	Dy[3] = y_max - x_labels_offset;


	for (i = 0; i < 4; i++) {
		Px[i] = Dx[i] * dx;
		if (f_v) {
			cout << "mp_graphics::draw_axes_and_grid "
					"Dx[i]=" << Dx[i] << " dx=" << dx << " Px[i]=" << Px[i] << endl;
		}
		Py[i] = Dy[i] * dy;
	}
	
	sl_ends(0 /* line_beg_style */, 1 /* line_end_style */);
	polygon2(Px, Py, 0, 1);
	polygon2(Px, Py, 2, 3);

	sl_ends(0 /* line_beg_style */, 0 /* line_end_style */);

	// draw x_ticks:
	j = 0;
	for (i = (int)x_min; i <= (int)x_max; i++) {
		if ((i % x_tick_mod) && (i != (int)x_min) && (i != (int)x_max)) {
			continue;
			}
		Dx[j] = i;
		Dy[j] = y0 + (-1) * x_tick_half_width;
		j++;
		Dx[j] = i;
		Dy[j] = y0 + 1 * x_tick_half_width;
		j++;
	}
	n = j;
	for (i = 0; i < n; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
	}
	j = 0;
	for (i = (int)x_min; i <= (int)x_max; i++) {
		if ((i % x_tick_mod) && (i != (int)x_min) && (i != (int)x_max)) {
			continue;
		}
		polygon2(Px, Py, j, j + 1);
		j += 2;
	}


	// draw y_ticks:
	j = 0;
	for (i = (int)y_min; i <= (int)y_max; i++) {
		if ((i % y_tick_mod) && (i != (int)y_min) && (i != (int)y_max)) {
			continue;
		}
		Dx[j] = x0 + (-1) * y_tick_half_width;
		Dy[j] = i;
		j++;
		Dx[j] = x0 + 1 * y_tick_half_width;
		Dy[j] = i;
		j++;
	}
	n = j;
	for (i = 0; i < n; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
		}
	j = 0;
	for (i = (int)y_min; i <= (int)y_max; i++) {
		if ((i % y_tick_mod) && (i != (int)y_min) && (i != (int)y_max)) {
			continue;
		}
		polygon2(Px, Py, j, j + 1);
		j += 2;
	}

	// draw x_labels:
	j = 0;
	for (i = (int)x_min; i <= (int)x_max; i++) {
		if ((i % x_tick_mod) && (i != (int)x_min) && (i != (int)x_max)) {
			continue;
		}
		Dx[j] = i;
		Dy[j] = y_min + x_labels_offset;
		j++;
	}
	n = j;
	for (i = 0; i < n; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
	}
	j = 0;
	for (i = (int)x_min; i <= (int)x_max; i++) {
		if ((i % x_tick_mod) && (i != (int)x_min) && (i != (int)x_max)) {
			continue;
		}
		string s;
		s = std::to_string(i);
		//cout << "str='" << str << "'" << endl;
		aligned_text_array(Px, Py, j, "", s);
		j += 1;
	}

	// draw y_labels:
	j = 0;
	for (i = (int)y_min; i <= (int)y_max; i++) {
		if ((i % y_tick_mod) && (i != (int)y_min) && (i != (int)y_max)) {
			continue;
		}
		Dy[j] = i;
		Dx[j] = x_min + y_labels_offset;
		j++;
	}
	n = j;
	for (i = 0; i < n; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
	}
	j = 0;
	for (i = (int)y_min; i <= (int)y_max; i++) {
		if ((i % y_tick_mod) && (i != (int)y_min) && (i != (int)y_max)) {
			continue;
		}
		string s;
		s = std::to_string(i);
		//cout << "str='" << str << "'" << endl;
		aligned_text_array(Px, Py, j, "", s);
		j += 1;
	}


	if (f_v_lines) {
		// draw vertical lines:

		double ddx = (double) 1 / subdivide_v;

		sl_thickness(35);	
		h = 0;
		for (i = (int)x_min; i <= (int)x_max; i++) {
			if ((i % x_mod) && (i != (int)x_min) && (i != (int)x_max)) {
				continue;
			}
			for (j = 0; j <= subdivide_v; j++) {
				if (j && i == (int)x_max) {
					continue;
				}
				Dx[h] = i + (double) j * ddx;
				Dy[h] = y_min;
				h++;
				Dx[h] = i + (double) j * ddx;
				Dy[h] = y_max;
				h++;
			}
		}
		n = h;
		for (i = 0; i < n; i++) {
			Px[i] = Dx[i] * dx;
			Py[i] = Dy[i] * dy;
		}
		h = 0;
		for (i = (int)x_min; i <= (int)x_max; i++) {
			if ((i % x_mod) && (i != (int)x_min) && (i != (int)x_max)) {
				continue;
			}
			for (j = 0; j <= subdivide_v; j++) {
				if (j && i == (int)x_max) {
					continue;
				}
				polygon2(Px, Py, h, h + 1);
				h += 2;
			}
		}
	}

	if (f_h_lines) {
		// draw horizontal lines:


		double ddy = (double) 1 / subdivide_h;

		sl_thickness(35);	
		h = 0;
		for (i = (int)y_min; i <= (int)y_max; i++) {
			if ((i % y_mod) && (i != (int)y_min) && (i != (int)y_max)) {
				continue;
			}
			for (j = 0; j <= subdivide_h; j++) {
				if (j && i == (int)y_max) {
					continue;
				}
				Dy[h] = i + (double) j * ddy;
				Dx[h] = x_min;
				h++;
				Dy[h] = i + (double) j * ddy;
				Dx[h] = x_max;
				h++;
			}
		}
		n = h;
		for (i = 0; i < n; i++) {
			Px[i] = Dx[i] * dx;
			Py[i] = Dy[i] * dy;
		}
		h = 0;
		for (i = (int)y_min; i <= (int)y_max; i++) {
			if ((i % y_mod) && (i != (int)y_min) && (i != (int)y_max)) {
				continue;
			}
			for (j = 0; j <= subdivide_h; j++) {
				if (j && i == (int)y_max) {
					continue;
				}
				polygon2(Px, Py, h, h + 1);
				h += 2;
			}
		}
	}

	sl_thickness(100);	
	FREE_int(Px);
	FREE_int(Py);
	delete [] Dx;
	delete [] Dy;



}

void mp_graphics::plot_curve(
		int N, int *f_DNE,
		double *Dx, double *Dy, double dx, double dy)
{
	int *Px;
	int *Py;
	int i, j;
	//int Dx1, Dy1, Dx2, Dy2, L1, L2;

	Px = NEW_int(N);
	Py = NEW_int(N);
	j = 0;
	for (i = 0; i < N; i++) {
		if (f_DNE[i] == false) {
			Px[j] = Dx[i] * dx;
			Py[j] = Dy[i] * dy;
			j++;
			//cout << "i=" << i << " Px[i]=" << Px[i]
			// << " Py[i]=" << Py[i] << endl;

#if 0
			if (i > 2 && f_DNE[i - 1] == false && f_DNE[i - 2] == false) {
				Dx1 = Px[i - 1] - Px[i - 2];
				Dy1 = Py[i - 1] - Py[i - 2];
				L1 = Dx1 * Dx1 + Dy1 * Dy1;
				Dx2 = Px[i] - Px[i - 1];
				Dy2 = Py[i] - Py[i - 1];
				L2 = Dx2 * Dx2 + Dy2 * Dy2;
				if (L2 > 10 * L1) {
					f_DNE[i] = true;
				}
			}
#endif
		}
	}
	N = j;

#if 0
	for (i = 0; i < N; i += 5) {
		for (j = 0; j < 6; j++) {
			if (i + j == N || f_DNE[i + j]) {
				break;
			}
		}
		
		if (j == 6) {
			polygon6(Px, Py, i + 0, i + 1, i + 2, i + 3, i + 4, i + 5);
		}
		else if (j == 5) {
			polygon5(Px, Py, i + 0, i + 1, i + 2, i + 3, i + 4);
		}
		else if (j == 4) {
			polygon4(Px, Py, i + 0, i + 1, i + 2, i + 3);
		}
		else if (j == 3) {
			polygon3(Px, Py, i + 0, i + 1, i + 2);
		}
		else if (j == 2) {
			polygon2(Px, Py, i + 0, i + 1);
		}
	}
#else
	for (i = 0; i < N - 1; i++) {
#if 0
		if (f_DNE[i] || f_DNE[i + 1]) {
		}
		else {
			polygon2(Px, Py, i + 0, i + 1);
		}
#endif
		polygon2(Px, Py, i + 0, i + 1);
	}
#endif
	FREE_int(Px);
	FREE_int(Py);
}


void mp_graphics::nice_circle(
		int x, int y, int rad)
{
	//fp_log << "NiceCircle " << x << " " << y << " " << rad << endl;

	sf_interior(100);
	//sf_color(0); // 1 = black, 0 = white
	circle(x, y, rad);
	sf_interior(0);
	//sf_color(1); // 1 = black, 0 = white
	circle(x, y, rad);
}

void mp_graphics::grid_polygon2(
		grid_frame *F,
	int x0, int y0, int x1, int y1)
{
	int *Px, *Py, *Idx;
	int i;

	Px = NEW_int(2);
	Py = NEW_int(2);
	Idx = NEW_int(2);

	for (i = 0; i < 2; i++) {
		Idx[i] = i;
	}
	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y0 * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x0) * F->dy);
		Px[1] = (int)(F->origin_x + y1 * F->dx);
		Py[1] = (int)(F->origin_y + (F->m - x1) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x0 * F->dx);
		Py[0] = (int)(F->origin_y + y0 * F->dy);
		Px[1] = (int)(F->origin_x + x1 * F->dx);
		Py[1] = (int)(F->origin_y + y1 * F->dy);
	}
	polygon_idx(Px, Py, Idx, 2);
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Idx);
}

void mp_graphics::grid_polygon4(
		grid_frame *F,
	int x0, int y0, int x1, int y1,
	int x2, int y2, int x3, int y3)
{
	int *Px, *Py, *Idx;
	int i;

	Px = NEW_int(4);
	Py = NEW_int(4);
	Idx = NEW_int(4);

	for (i = 0; i < 4; i++) {
		Idx[i] = i;
	}
	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y0 * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x0) * F->dy);
		Px[1] = (int)(F->origin_x + y1 * F->dx);
		Py[1] = (int)(F->origin_y + (F->m - x1) * F->dy);
		Px[2] = (int)(F->origin_x + y2 * F->dx);
		Py[2] = (int)(F->origin_y + (F->m - x2) * F->dy);
		Px[3] = (int)(F->origin_x + y3 * F->dx);
		Py[3] = (int)(F->origin_y + (F->m - x3) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x0 * F->dx);
		Py[0] = (int)(F->origin_y + y0 * F->dy);
		Px[1] = (int)(F->origin_x + x1 * F->dx);
		Py[1] = (int)(F->origin_y + y1 * F->dy);
		Px[2] = (int)(F->origin_x + x2 * F->dx);
		Py[2] = (int)(F->origin_y + y2 * F->dy);
		Px[3] = (int)(F->origin_x + x3 * F->dx);
		Py[3] = (int)(F->origin_y + y3 * F->dy);
	}
	polygon_idx(Px, Py, Idx, 4);
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Idx);
}

void mp_graphics::grid_polygon5(
		grid_frame *F,
	int x0, int y0, int x1, int y1,
	int x2, int y2, int x3, int y3,
	int x4, int y4)
{
	int *Px, *Py, *Idx;
	int i;

	Px = NEW_int(5);
	Py = NEW_int(5);
	Idx = NEW_int(5);

	for (i = 0; i < 5; i++) {
		Idx[i] = i;
	}
	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y0 * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x0) * F->dy);
		Px[1] = (int)(F->origin_x + y1 * F->dx);
		Py[1] = (int)(F->origin_y + (F->m - x1) * F->dy);
		Px[2] = (int)(F->origin_x + y2 * F->dx);
		Py[2] = (int)(F->origin_y + (F->m - x2) * F->dy);
		Px[3] = (int)(F->origin_x + y3 * F->dx);
		Py[3] = (int)(F->origin_y + (F->m - x3) * F->dy);
		Px[4] = (int)(F->origin_x + y4 * F->dx);
		Py[4] = (int)(F->origin_y + (F->m - x4) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x0 * F->dx);
		Py[0] = (int)(F->origin_y + y0 * F->dy);
		Px[1] = (int)(F->origin_x + x1 * F->dx);
		Py[1] = (int)(F->origin_y + y1 * F->dy);
		Px[2] = (int)(F->origin_x + x2 * F->dx);
		Py[2] = (int)(F->origin_y + y2 * F->dy);
		Px[3] = (int)(F->origin_x + x3 * F->dx);
		Py[3] = (int)(F->origin_y + y3 * F->dy);
		Px[4] = (int)(F->origin_x + x4 * F->dx);
		Py[4] = (int)(F->origin_y + y4 * F->dy);
	}
	polygon_idx(Px, Py, Idx, 5);
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Idx);
}

void mp_graphics::polygon(
		int *Px, int *Py, int n)
{
	int *Idx = NEW_int(n);
	int i;
	
	polygon_log(Px, Py, n);
	for (i = 0; i < n; i++) {
		Idx[i] = i;
		}
	polygon_idx(Px, Py, Idx, n);
	FREE_int(Idx);
}

void mp_graphics::polygon2(
		int *Px, int *Py, int i1, int i2)
{
	int Idx[2];
	Idx[0] = i1;
	Idx[1] = i2;
	polygon_idx(Px, Py, Idx, 2);
}

void mp_graphics::polygon3(
		int *Px, int *Py,
		int i1, int i2, int i3)
{
	int Idx[3];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	polygon_idx(Px, Py, Idx, 3);
}

void mp_graphics::polygon4(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4)
{
	int Idx[4];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	polygon_idx(Px, Py, Idx, 4);
}

void mp_graphics::polygon5(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5)
{
	int Idx[5];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;

	polygon_idx(Px, Py, Idx, 5);
}

void mp_graphics::polygon6(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	polygon_idx(Px, Py, Idx, 6);
}

void mp_graphics::polygon7(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	polygon_idx(Px, Py, Idx, 7);
}

void mp_graphics::polygon8(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	polygon_idx(Px, Py, Idx, 8);
}

void mp_graphics::polygon9(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	polygon_idx(Px, Py, Idx, 9);
}

void mp_graphics::polygon10(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9, int i10)
{
	int Idx[20];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	Idx[9] = i10;
	polygon_idx(Px, Py, Idx, 10);
}

void mp_graphics::polygon11(
		int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9, int i10, int i11)
{
	int Idx[20];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	Idx[9] = i10;
	Idx[10] = i11;
	polygon_idx(Px, Py, Idx, 11);
}

void mp_graphics::polygon_idx(
		int *Px, int *Py, int *Idx, int n)
{
	polygon_idx_log(Px, Py, Idx, n);
	polygon_idx2(Px, Py, Idx, n, false);
}

void mp_graphics::bezier(int *Px, int *Py, int n)
{
	int *Idx = NEW_int(n);
	int i;
	
	for (i = 0; i < n; i++) {
		Idx[i] = i;
		}
	bezier_idx(Px, Py, Idx, n);
	FREE_int(Idx);
}

void mp_graphics::bezier2(int *Px, int *Py,
		int i1, int i2)
{
	int Idx[2];
	Idx[0] = i1;
	Idx[1] = i2;
	bezier_idx(Px, Py, Idx, 2);
}

void mp_graphics::bezier3(int *Px, int *Py,
		int i1, int i2, int i3)
{
	int Idx[3];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	bezier_idx(Px, Py, Idx, 3);
}

void mp_graphics::bezier4(int *Px, int *Py,
		int i1, int i2, int i3, int i4)
{
	int Idx[4];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	bezier_idx(Px, Py, Idx, 4);
}

void mp_graphics::bezier5(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5)
{
	int Idx[5];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	bezier_idx(Px, Py, Idx, 5);
}

void mp_graphics::bezier6(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6)
{
	int Idx[6];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	bezier_idx(Px, Py, Idx, 6);
}

void mp_graphics::bezier7(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	int Idx[7];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	bezier_idx(Px, Py, Idx, 7);
}

void mp_graphics::bezier_idx(int *Px, int *Py, int *Idx, int n)
{
	bezier_idx_log(Px, Py, Idx, n);
	bezier_idx2(Px, Py, Idx, n, false);
}

void mp_graphics::grid_fill_polygon4(grid_frame *F, 
	int x0, int y0, int x1, int y1,
	int x2, int y2, int x3, int y3)
{
	int *Px, *Py, *Idx;
	int i;

	Px = NEW_int(4);
	Py = NEW_int(4);
	Idx = NEW_int(4);

	for (i = 0; i < 4; i++) {
		Idx[i] = i;
	}
	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y0 * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x0) * F->dy);
		Px[1] = (int)(F->origin_x + y1 * F->dx);
		Py[1] = (int)(F->origin_y + (F->m - x1) * F->dy);
		Px[2] = (int)(F->origin_x + y2 * F->dx);
		Py[2] = (int)(F->origin_y + (F->m - x2) * F->dy);
		Px[3] = (int)(F->origin_x + y3 * F->dx);
		Py[3] = (int)(F->origin_y + (F->m - x3) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x0 * F->dx);
		Py[0] = (int)(F->origin_y + y0 * F->dy);
		Px[1] = (int)(F->origin_x + x1 * F->dx);
		Py[1] = (int)(F->origin_y + y1 * F->dy);
		Px[2] = (int)(F->origin_x + x2 * F->dx);
		Py[2] = (int)(F->origin_y + y2 * F->dy);
		Px[3] = (int)(F->origin_x + x3 * F->dx);
		Py[3] = (int)(F->origin_y + y3 * F->dy);
	}
	fill_idx(Px, Py, Idx, 4, "--", true);
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Idx);
}

void mp_graphics::grid_fill_polygon5(grid_frame *F, 
	int x0, int y0, int x1, int y1, int x2, int y2,
	int x3, int y3, int x4, int y4)
{
	int *Px, *Py, *Idx;
	int i;

	Px = NEW_int(5);
	Py = NEW_int(5);
	Idx = NEW_int(5);

	for (i = 0; i < 5; i++) {
		Idx[i] = i;
	}
	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y0 * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x0) * F->dy);
		Px[1] = (int)(F->origin_x + y1 * F->dx);
		Py[1] = (int)(F->origin_y + (F->m - x1) * F->dy);
		Px[2] = (int)(F->origin_x + y2 * F->dx);
		Py[2] = (int)(F->origin_y + (F->m - x2) * F->dy);
		Px[3] = (int)(F->origin_x + y3 * F->dx);
		Py[3] = (int)(F->origin_y + (F->m - x3) * F->dy);
		Px[4] = (int)(F->origin_x + y4 * F->dx);
		Py[4] = (int)(F->origin_y + (F->m - x4) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x0 * F->dx);
		Py[0] = (int)(F->origin_y + y0 * F->dy);
		Px[1] = (int)(F->origin_x + x1 * F->dx);
		Py[1] = (int)(F->origin_y + y1 * F->dy);
		Px[2] = (int)(F->origin_x + x2 * F->dx);
		Py[2] = (int)(F->origin_y + y2 * F->dy);
		Px[3] = (int)(F->origin_x + x3 * F->dx);
		Py[3] = (int)(F->origin_y + y3 * F->dy);
		Px[4] = (int)(F->origin_x + x4 * F->dx);
		Py[4] = (int)(F->origin_y + y4 * F->dy);
	}
	fill_idx(Px, Py, Idx, 5, "--", true);
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Idx);
}

void mp_graphics::fill_polygon3(int *Px, int *Py,
		int i1, int i2, int i3)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	fill_idx(Px, Py, Idx, 3, "--", false);
}

void mp_graphics::fill_polygon4(int *Px, int *Py,
		int i1, int i2, int i3, int i4)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	fill_idx(Px, Py, Idx, 4, "--", false);
}

void mp_graphics::fill_polygon5(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	fill_idx(Px, Py, Idx, 5, "--", false);
}

void mp_graphics::fill_polygon6(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	fill_idx(Px, Py, Idx, 6, "--", false);
}

void mp_graphics::fill_polygon7(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	fill_idx(Px, Py, Idx, 7, "--", false);
}

void mp_graphics::fill_polygon8(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	fill_idx(Px, Py, Idx, 8, "--", false);
}

void mp_graphics::fill_polygon9(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	fill_idx(Px, Py, Idx, 9, "--", false);
}

void mp_graphics::fill_polygon10(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9, int i10)
{
	int Idx[20];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	Idx[9] = i10;
	fill_idx(Px, Py, Idx, 10, "--", false);
}

void mp_graphics::fill_polygon11(int *Px, int *Py,
		int i1, int i2, int i3, int i4, int i5, int i6,
		int i7, int i8, int i9, int i10, int i11)
{
	int Idx[20];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	Idx[4] = i5;
	Idx[5] = i6;
	Idx[6] = i7;
	Idx[7] = i8;
	Idx[8] = i9;
	Idx[9] = i10;
	Idx[10] = i11;
	fill_idx(Px, Py, Idx, 11, "--", false);
}

void mp_graphics::polygon2_arrow_halfway(
		int *Px, int *Py,
		int i1, int i2)
{
	int Idx[3];
	int X[3], Y[3];
	
	X[0] = Px[i1];
	X[1] = Px[i2];
	X[2] = (Px[i1] + Px[i2]) >> 1;
	Y[0] = Py[i1];
	Y[1] = Py[i2];
	Y[2] = (Py[i1] + Py[i2]) >> 1;

	sl_ends(0, 0);
	Idx[0] = 0;
	Idx[1] = 1;
	
	polygon_idx(X, Y, Idx, 2);

	sl_ends(0, 1);

	Idx[0] = 0;
	Idx[1] = 2;
	polygon_idx(X, Y, Idx, 2);

	sl_ends(0, 0);
}

void mp_graphics::polygon2_arrow_halfway_and_label(
		int *Px, int *Py,
	int i1, int i2,
	const char *alignment, std::string &s)
{
	int Idx[3];
	int X[3], Y[3];
	
	X[0] = Px[i1];
	X[1] = Px[i2];
	X[2] = (Px[i1] + Px[i2]) >> 1;
	Y[0] = Py[i1];
	Y[1] = Py[i2];
	Y[2] = (Py[i1] + Py[i2]) >> 1;

	sl_ends(0, 0);
	Idx[0] = 0;
	Idx[1] = 1;
	
	polygon_idx(X, Y, Idx, 2);

	sl_ends(0, 1);

	Idx[0] = 0;
	Idx[1] = 2;
	polygon_idx(X, Y, Idx, 2);

	sl_ends(0, 0);
	
	aligned_text(X[2], Y[2], alignment, s);
}

void mp_graphics::grid_aligned_text(grid_frame *F,
		int x, int y, const char *alignment, std::string &s)
{
	int *Px, *Py;

	Px = NEW_int(1);
	Py = NEW_int(1);

	if (F->f_matrix_notation) {
		Px[0] = (int)(F->origin_x + y * F->dx);
		Py[0] = (int)(F->origin_y + (F->m - x) * F->dy);
	}
	else {
		Px[0] = (int)(F->origin_x + x * F->dx);
		Py[0] = (int)(F->origin_y + y * F->dy);
	}
	aligned_text(Px[0], Py[0], alignment, s);
	FREE_int(Px);
	FREE_int(Py);
}

void mp_graphics::aligned_text(int x, int y,
		const char *alignment, std::string &s)
{
	//fp_log << "AlignedText " << x << " " << y << " "
	// << alignment << " \"" << p << "\"" << endl;
	aligned_text_with_offset(x, y, 0, 0, alignment, s);
}

void mp_graphics::aligned_text_array(int *Px, int *Py, int idx,
		const char *alignment, std::string &s)
{
	aligned_text(Px[idx], Py[idx], alignment, s);
}

void mp_graphics::aligned_text_with_offset(int x, int y,
		int xoffset, int yoffset,
	const char *alignment, std::string &s)
{
	int h_align = 1, v_align = 1;
	int l, i;
	char c;
	
	//fp_log << "AlignedText " << x << " " << y << " " << xoffset
	// << " " << yoffset << " " << alignment << " \"" << s << "\"" << endl;
	l = strlen(alignment);
	for (i = 0; i < l; i++) {
		c = alignment[i];
		if (c == 'r')
			h_align = 2;
		else if (c == 'l')
			h_align = 0;
		else if (c == 'b')
			v_align = 0;
		else if (c == 't')
			v_align = 2;
		else {
			cout << "mp_graphics::aligned_text: "
					"unknown alignment character " << c << endl;
		}
	}
	//cout << "xoffset=" << xoffset << endl;
	//cout << "yoffset=" << yoffset << endl;
	//cout << "text=" << s << endl;
	st_alignment(h_align, v_align);
	text(x + xoffset, y + yoffset, s);
}







void mp_graphics::st_alignment(int txt_halign, int txt_valign)
{
	mp_graphics::txt_halign = txt_halign;
	mp_graphics::txt_valign = txt_valign;
	st_alignment_log();
}

void mp_graphics::sl_udsty(int line_dashing)
{
	mp_graphics::line_dashing = line_dashing;
	sl_udsty_log();
}

void mp_graphics::sl_ends(int line_beg_style, int line_end_style)
{
	mp_graphics::line_beg_style = line_beg_style;
	mp_graphics::line_end_style = line_end_style;
	sl_ends_log();
}

void mp_graphics::sl_thickness(int line_thickness)
{
	mp_graphics::line_thickness = line_thickness;
	line_thickness_mp();
	sl_thickness_log();
}

void mp_graphics::sl_color(int line_color)
{
	mp_graphics::line_color = line_color;
	sl_color_log();
}

void mp_graphics::sf_interior(int fill_interior)
{
	mp_graphics::fill_interior = fill_interior;
	sf_interior_log();
}

void mp_graphics::sf_color(int fill_color)
{
	mp_graphics::fill_color = fill_color;
	sf_color_log();
}

void mp_graphics::sf_outline(int fill_outline)
{
	mp_graphics::fill_outline = fill_outline;
	sf_outline_log();
}

void mp_graphics::sf_shape(int fill_shape)
{
	mp_graphics::fill_shape = fill_shape;
	sf_shape_log();
}

void mp_graphics::sf_nofill(int fill_nofill)
{
	mp_graphics::fill_nofill = fill_nofill;
	sf_nofill_log();
}

void mp_graphics::st_boxed(int txt_boxed)
{
	mp_graphics::txt_boxed = txt_boxed;
	st_boxed_log();
}

void mp_graphics::st_overwrite(int txt_overwrite)
{
	mp_graphics::txt_overwrite = txt_overwrite;
	st_overwrite_log();
}

void mp_graphics::st_rotate(int txt_rotate)
{
	mp_graphics::txt_rotate = txt_rotate;
	st_rotate_log();
}




void mp_graphics::coords_min_max(int x, int y)
{
	if (!f_min_max_set) {
		x_min = x_max = x;
		y_min = y_max = y;
	}
	else {
		x_min = MINIMUM(x_min, x);
		y_min = MINIMUM(y_min, y);
		x_max = MAXIMUM(x_max, x);
		y_max = MAXIMUM(y_max, y);
	}
	f_min_max_set = true;
}


// #############################################################################
// output commands:
// #############################################################################

void mp_graphics::header()
{
	
	f_min_max_set = false;
	//system("rm a");

	orbiter_kernel_system::os_interface Os;
	string str;

	Os.get_date(str);

	header_log(str);
	header_mp(str);
	header_tikz(str);
}

void mp_graphics::footer()
{
	footer_log();
	footer_mp();
	footer_tikz();
}

void mp_graphics::begin_figure(int factor_1000)
{
	begin_figure_mp(factor_1000);
}

void mp_graphics::end_figure()
{
	end_figure_mp();
}

void mp_graphics::comment(std::string &s)
{
	comment_log(s);
	comment_mp(s);
	comment_tikz(s);
}




void mp_graphics::text(int x, int y, std::string &s)
{
	int x1, y1;
	
	//fp_log << "Text " << x << " " << y << " \"" << p << "\"" << endl;

	x1 = x;
	y1 = y;
	coords_min_max(x1, y1);
	user2dev(x1, y1);
	
	text_log(x1, y1, s);
	text_mp(x1, y1, s);
	text_tikz(x1, y1, s);
}

void mp_graphics::circle(int x, int y, int rad)
{
	//fp_log << "Circle " << x << " " << y << " " << rad << endl;

	//cout << "mp_graphics::circle x=" << x << " y=" << y << " rad=" << rad << endl;
	coords_min_max(x, y);
	user2dev(x, y);
	user2dev_dist_x(rad);
	
	if (rad <= 0) rad = 1;
	
	circle_log(x, y, rad);
	circle_mp(x, y, rad);
	circle_tikz(x, y, rad);
}

void mp_graphics::circle_text(int x, int y, int rad, std::string &s)
{
	//fp_log << "CircleText " << x << " " << y << " \"" << s << "\"" << endl;

#if 0
	coords_min_max(x, y);
	user2dev(x, y);
	user2dev_dist_x(rad);
#endif

	nice_circle(x, y, rad);
	text(x, y, s);
	//output_circle_text_mp(x, y, idx, p);
	//output_circle_text_tikz(x, y, idx, rad, p);
}

#if 0
void mp_graphics::polygon_or_bezier_idx(int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle)
{
	polygon_or_bezier_idx_mp(Px, Py, Idx, n, symbol, f_cycle);
	polygon_or_bezier_idx_tikz(Px, Py, Idx, n, symbol, f_cycle);
}
#endif

void mp_graphics::polygon_idx2(int *Px, int *Py, int *Idx, int n,
		int f_cycle)
{
	polygon_idx_mp(Px, Py, Idx, n, f_cycle);
	polygon_idx_tikz(Px, Py, Idx, n, f_cycle);
}

void mp_graphics::bezier_idx2(int *Px, int *Py, int *Idx, int n,
		int f_cycle)
{
	bezier_idx_mp(Px, Py, Idx, n, f_cycle);
	bezier_idx_tikz(Px, Py, Idx, n, f_cycle);
}

void mp_graphics::fill_idx(int *Px, int *Py, int *Idx, int n,
		const char *symbol, int f_cycle)
{
	fill_idx_mp(Px, Py, Idx, n, symbol, f_cycle);
	fill_idx_tikz(fp_tikz, Px, Py, Idx, n, symbol, f_cycle);
}



// #############################################################################
// device specific output commands: log file
// #############################################################################

void mp_graphics::header_log(std::string &str_date)
{
	fp_log << "% file: " << fname_log << endl;
	fp_log << "% created by Orbiter graphics interface" << endl;
	fp_log << "% creation date: " << str_date << endl << endl;
	fp_log << "DeviceCoordinates " << dev[0] << " " << dev[1] << " " << dev[2] << " " << dev[3] << endl;
	fp_log << "UserCoordinates " << user[0] << " " << user[1] << " " << user[2] << " " << user[3] << endl;
	
}

void mp_graphics::footer_log()
{
	fp_log << "END_OF_FILE" << endl << endl;
}

void mp_graphics::comment_log(std::string &s)
{
	fp_log << "Comment " << s << endl;
}

void mp_graphics::st_alignment_log()
{
	fp_log << "st_alignment " << txt_halign << ", " << txt_valign << endl;
}

void mp_graphics::sl_udsty_log()
{
	fp_log << "sl_udsty " << line_dashing << endl;
}

void mp_graphics::sl_ends_log()
{
	fp_log << "sl_ends " << line_beg_style << ", " << line_end_style << endl;
}

void mp_graphics::sl_thickness_log()
{
	fp_log << "sl_thickness " << line_thickness << endl;
}

void mp_graphics::sl_color_log()
{
	fp_log << "sl_color " << line_color << endl;
}

void mp_graphics::sf_interior_log()
{
	fp_log << "sf_interior " << fill_interior << endl;
}

void mp_graphics::sf_color_log()
{
	fp_log << "sf_color " << fill_color << endl;
}

void mp_graphics::sf_shape_log()
{
	fp_log << "sf_shape " << fill_shape << endl;
}

void mp_graphics::sf_outline_log()
{
	fp_log << "sf_outline " << fill_outline << endl;
}

void mp_graphics::sf_nofill_log()
{
	fp_log << "sf_nofill " << fill_nofill << endl;
}

void mp_graphics::st_boxed_log()
{
	fp_log << "st_boxed " << txt_boxed << endl;
}

void mp_graphics::st_overwrite_log()
{
	fp_log << "st_overwrite " << txt_overwrite << endl;
}

void mp_graphics::st_rotate_log()
{
	fp_log << "st_rotate " << txt_rotate << endl;
}

void mp_graphics::bezier_idx_log(int *Px, int *Py, int *Idx, int n)
{
	int i;
	
	fp_log << "Bezier " << n;
	for (i = 0; i < n; i++) {
		fp_log << " " << Px[Idx[i]] << " " << Py[Idx[i]];
	}
	fp_log << endl;
}

void mp_graphics::polygon_log(int *Px, int *Py, int n)
{
	int i;
	
	fp_log << "Polygon " << n;
	for (i = 0; i < n; i++) {
		fp_log << " " << Px[i] << " " << Py[i];
	}
	fp_log << endl;
}

void mp_graphics::polygon_idx_log(int *Px, int *Py, int *Idx, int n)
{
	int i;
	
	fp_log << "Polygon " << n;
	for (i = 0; i < n; i++) {
		fp_log << " " << Px[Idx[i]] << " " << Py[Idx[i]];
	}
	fp_log << endl;
}

void mp_graphics::text_log(int x1, int y1, std::string &s)
{
	fp_log << "Text " << x1 << ", " << y1 << ", \"" << s << "\"" << endl;
}

void mp_graphics::circle_log(int x1, int y1, int rad)
{
	fp_log << "Circle " << x1 << ", " << y1 << ", " << rad << endl;
}


// #############################################################################
// device specific output commands: metapost
// #############################################################################


void mp_graphics::header_mp(std::string &str_date)
{
	fp_mp << "% file: " << fname_mp << endl;
	fp_mp << "% created by Orbiter metapost interface" << endl;
	fp_mp << "% creation date: " << str_date << endl << endl;
	fp_mp << "input boxes" << endl << endl;
	
	fp_mp << "verbatimtex" << endl;
	fp_mp << "\\documentclass[10pt]{article}" << endl;
	fp_mp << "\\usepackage[noBBpl]{mathpazo}" << endl;
	fp_mp << "\\usepackage{amsmath}" << endl;
	fp_mp << "\\usepackage{amssymb}" << endl;
	fp_mp << "\\begin{document}" << endl;
	fp_mp << "\\scriptsize" << endl;
	fp_mp << "etex;" << endl;
}

void mp_graphics::footer_mp()
{
	fp_mp << "end" << endl << endl;
}

void mp_graphics::begin_figure_mp(int factor_1000)
{
	double d;
	char str[1000];
	int i, l;
	
	d = (double) factor_1000 * 0.001  * 0.1;
	
	//fp_mp << "defaultfont:=\"cmr7\";\n";

	snprintf(str, sizeof(str), "%lf", d);
	l = strlen(str);
	for (i = 0; i < l; i++) {
		if (str[i] == ',')
			str[i] = '.';
	}
	fp_mp << "u=" << str << "mm;\n";
	fp_mp << "beginfig(1);" << endl;
	fp_mp << "path p[];" << endl;
}

void mp_graphics::end_figure_mp()
{
	fp_mp << "endfig;" << endl << endl;
}

void mp_graphics::comment_mp(std::string &s)
{
	fp_mp << "% " << s << endl;
}

void mp_graphics::text_mp(int x1, int y1, std::string &s)
{
	char align[64];
	int lab;

	get_alignment_mp(align);
	if (txt_boxed) {
		lab = get_label(x1, y1);
		fp_mp << "boxit.l" << lab << "(btex " << s << " etex);" << endl;
		fp_mp << lab << ".c=";
		output_xy_metapost(x1, y1);
		fp_mp << endl;
		if (txt_overwrite) {
			fp_mp << "unfill bpath " << lab << ";" << endl;
		}
		fp_mp << "drawboxed(" << lab << ");" << endl;
		
	}
	else {
		fp_mp << "label" << align << "(btex " << s << " etex";
		if (txt_rotate) {
			fp_mp << " rotated " << txt_rotate;
		}
		fp_mp << ", ";
		output_xy_metapost(x1, y1);
		fp_mp << ");" << endl;
	}
}

void mp_graphics::circle_mp(int x, int y, int rad)
{
	int X[10], Y[10], i;

	X[0] = x +  rad;
	Y[0] = y;
	X[1] = x;
	Y[1] = y +  rad;
	X[2] = x -  rad;
	Y[2] = y;
	X[3] = x;
	Y[3] = y -  rad;
	X[4] = x +  rad;
	Y[4] = y;
	fp_mp << "path pp;" << endl;
	//fp_mp << "pickup pencircle scaled " << line_thickness << "pt;" << endl;
	fp_mp << "pp = ";
	for (i = 0; i < 5; i++) {
		if (i) {
			fp_mp << " .. ";
		}
		output_xy_metapost(X[i], Y[i]);
	}
	fp_mp << " .. cycle;" << endl;
	if (fill_interior > 0) {
		fp_mp << "fill pp withcolor ";
		if (fill_interior > 99) {
			fp_mp << "1 ";
		}
		else {
			fp_mp << "." << fill_interior << " ";
			// fprintf(mp_draw_fp, ".%02ld ", fill_interior);
		}
		if (fill_color == 1)
			fp_mp << "black";
		else
			fp_mp << "white";
		fp_mp << ";" << endl;
	}
	else {
		fp_mp << "draw pp";
		if (line_dashing) {
			fp_mp << " dashed evenly";
			if (line_dashing != 100) {
				fp_mp << " scaled " << (double) line_dashing / 100.;
			}
		}
		fp_mp << ";" << endl;
	}
}

void mp_graphics::output_circle_text_mp(
		int x, int y,
		int idx, std::string &s)
{
	fp_mp << "circleit.l" << idx << "(btex " << s << " etex);" << endl;
	fp_mp << "l" << idx << ".c = ";
	output_xy_metapost(x, y);
	fp_mp << endl;
	fp_mp << "unfill bpath l" << idx << ";" << endl;
	fp_mp << "drawboxed(l" << idx << ");" << endl;
}

void mp_graphics::polygon_idx_mp(
		int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int x, y, i;

	//f << "pickup pencircle scaled " << line_thickness << "pt;" << endl;
	if (line_end_style == 1)
		fp_mp << "drawarrow ";
	else
		fp_mp << "draw ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);

		if (i) {
			fp_mp << "--";
		}
		//fp_mp << "(" << x << "u," << y << "u)";
		output_xy_metapost(x, y);
		if (((i + 1) % 30) == 0)
			fp_mp << endl;
	}
	if (f_cycle) {
		fp_mp << " " << "--" << " cycle ";
	}

	if (line_dashing) {
		fp_mp << " dashed evenly";
		if (line_dashing != 100) {
			fp_mp << " scaled " << (double) line_dashing / 100.;
		}
	}
	fp_mp << ";" << endl;
}

void mp_graphics::bezier_idx_mp(
		int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int x, y, i;

	//f << "pickup pencircle scaled " << line_thickness << "pt;" << endl;
	if (line_end_style == 1)
		fp_mp << "drawarrow ";
	else
		fp_mp << "draw ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);
		
		if (i) {
			fp_mp << "..";
		}
		//fp_mp << "(" << x << "u," << y << "u)";
		output_xy_metapost(x, y);
		if (((i + 1) % 30) == 0)
			fp_mp << endl;
	}
	if (f_cycle) {
		fp_mp << " " << ".." << " cycle ";
	}

	if (line_dashing) {
		fp_mp << " dashed evenly";
		if (line_dashing != 100) {
			fp_mp << " scaled " << (double) line_dashing / 100.;
		}
	}
	fp_mp << ";" << endl;
}

void mp_graphics::fill_idx_mp(
		int *Px, int *Py,
		int *Idx, int n, const char *symbol, int f_cycle)
{
	int x, y, i;

	fp_mp << "fill buildcycle(";
	//fp_mp << "p[" << cur_path << "] = buildcycle(";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);
		
		if (i) {
			fp_mp << symbol;
		}
		output_xy_metapost(x, y);
		if (((i + 1) % 30) == 0)
			fp_mp << endl;
	}
	if (f_cycle) {
		fp_mp << " " << symbol << " cycle ";
	}
	fp_mp << ")";

	//fp_mp << "fill p" << cur_path << " withcolor ";
	fp_mp << " withcolor ";
	fp_mp << ((double)fill_interior / (double) 100);
	if (fill_color == 0) {
		fp_mp << " white;" << endl;
	}
	else if (fill_color == 1) {
		fp_mp << " black;" << endl;
	}
	cur_path++;
	
}

void mp_graphics::output_xy_metapost(int x, int y)
{
	fp_mp << "(";
	output_x_metapost(x);
	fp_mp << ",";
	output_y_metapost(y);
	fp_mp << ")";
}

void mp_graphics::output_x_metapost(int x)
{
	double d;

	d = (double) x;
	d /= 1000.;
	fp_mp << d << "u ";
}

void mp_graphics::output_y_metapost(int y)
{
	double d;

	d = (double) y;
	d /= 1000.;
	fp_mp << d << "u ";
}

int mp_graphics::get_label(int x, int y)
{
	static int i = 0;
	
	return i++;
}

void mp_graphics::get_alignment_mp(char *align)
{
	if (txt_halign == 2) {
			// right aligned, text to the
			// left of the current position
		if (txt_valign == 2) 
			strcpy(align, ".llft");
		else if (txt_valign == 1) 
			strcpy(align, ".lft");
		else if (txt_valign == 0) 
			strcpy(align, ".ulft");
	}
	else if (txt_halign == 1) { // horizontally centered
		if (txt_valign == 2) 
			strcpy(align, ".bot");
		else if (txt_valign == 1) 
			strcpy(align, "");
		else if (txt_valign == 0) 
			strcpy(align, ".top");
	}
	else if (txt_halign == 0) {
		if (txt_valign == 2) 
			strcpy(align, ".lrt");
		else if (txt_valign == 1) 
			strcpy(align, ".rt");
		else if (txt_valign == 0) 
			strcpy(align, ".urt");
	}
}

void mp_graphics::line_thickness_mp()
{
	double d;

	d = line_thickness * 0.01;
	fp_mp << "pickup pencircle scaled " << d << "pt;" << endl;

	//cout << "mp_graphics::line_thickness = "
	// << mp_graphics::line_thickness << endl;
}




// #############################################################################
// device specific output commands: tikz
// #############################################################################

void mp_graphics::header_tikz(
		std::string &str_date)
{
	fp_tikz << "% file: " << fname_tikz << endl;
	fp_tikz << "% created by Orbiter tikz interface" << endl;

	fp_tikz << "% creation date: " << str_date << endl;

	// no extra spaces in tikz mode so that we do not create a line feed.
	// this allows for multiple tikz pictures on the same line
	fp_tikz << "% DeviceCoordinates " << dev[0] << " " << dev[1]
			<< " " << dev[2] << " " << dev[3] << endl;
	fp_tikz << "% UserCoordinates " << user[0] << " " << user[1]
			<< " " << user[2] << " " << user[3] << endl;
	
	if (Draw_options->f_embedded) {
		fp_tikz << "\\documentclass{standalone}" << endl;
		//fp_tikz << "\\documentclass[12pt]{article}" << endl;
		fp_tikz << "\\usepackage{amsmath, amssymb, amsthm}" << endl;
		fp_tikz << "\\usepackage{tikz} " << endl;
		if (Draw_options->f_sideways) {
			fp_tikz << "\\usepackage{rotating} " << endl;
		}
		fp_tikz << "%\\usepackage{anysize}" << endl;


		if (Draw_options->f_paperheight) {
			fp_tikz << "\\paperheight=" << (double) Draw_options->paperheight * 0.1 << "in" << endl;
		}
		if (Draw_options->f_paperwidth) {
			fp_tikz << "\\paperwidth=" << (double) Draw_options->paperwidth * 0.1 << "in" << endl;
		}

		fp_tikz << "\\begin{document}" << endl;
		fp_tikz << "%\\bibliographystyle{plain}" << endl;
		fp_tikz << "\\pagestyle{empty}" << endl;
	}

	if (Draw_options->f_sideways) {
		fp_tikz << "\\begin{sideways}" << endl;
	}
	fp_tikz << "\\begin{tikzpicture}[scale=" << Draw_options->scale
		<< ",line width = " << Draw_options->line_width << "pt]" << endl;
	//fp_tikz << "\\begin{tikzpicture}[scale=.05,line width = 0.5pt]" << endl;
}

void mp_graphics::footer_tikz()
{
	fp_tikz << "\\end{tikzpicture}" << endl;
	if (Draw_options->f_sideways) {
		fp_tikz << "\\end{sideways}" << endl;
	}
	if (Draw_options->f_embedded) {
		fp_tikz << "\\end{document}" << endl;
	}
}

void mp_graphics::comment_tikz(
		std::string &s)
{
	fp_tikz << "% " << s << endl;
}


void mp_graphics::text_tikz(
		int x1, int y1, std::string &s)
{
	if (txt_overwrite) {
		fp_tikz << "\\draw ";
		output_xy_tikz(x1, y1);
		fp_tikz << " node[fill=white] {";
		fp_tikz << s;
		fp_tikz << "};" << endl;
	}
	else {
		fp_tikz << "\\draw ";
		output_xy_tikz(x1, y1);
		fp_tikz << " node{";
		fp_tikz << s;
		fp_tikz << "};" << endl;
	}
}


void mp_graphics::circle_tikz(
		int x, int y, int rad)
{
	//cout << "mp_graphics::circle_tikz x=" << x << " y=" << y << " rad=" << rad << endl;
	if (fill_interior > 0) {
		fp_tikz << "\\filldraw[color=";
		color_tikz(fp_tikz, fill_color);
		fp_tikz << "] "; 
		output_xy_tikz(x, y);
		fp_tikz << " circle [radius = ";
		output_x_tikz(rad * 1);
		fp_tikz << "cm];" << endl;
	}
	else {
		fp_tikz << "\\draw "; 
		output_xy_tikz(x, y);
		fp_tikz << " circle [radius = ";
		output_x_tikz(rad * 1);
		fp_tikz << "cm];" << endl;
	}
}


void mp_graphics::output_circle_text_tikz(
		int x, int y,
		int idx, int rad, const char *text)
{

	fp_tikz << "\\draw "; 
	output_xy_tikz(x, y);
	fp_tikz << " circle [fill=white,radius = ";
	output_x_tikz(rad * 1);
	fp_tikz << "cm];" << endl;

	fp_tikz << "\\draw ";
	output_xy_tikz(x, y);
	fp_tikz << " node{" << text;
	//fp_tikz << " node[fill=white]{" << text;
	fp_tikz << "};" << endl;

}


void mp_graphics::polygon_idx_tikz(
		int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int f_need_comma = false;
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int x, y, i;

	fp_tikz << "\\draw [";
	if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "->";
		f_need_comma = true;
	}
	else if (line_end_style == 0 && line_beg_style == 1) {
		fp_tikz << "<-";
		f_need_comma = true;
	}
	else if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "<->";
		f_need_comma = true;
	}
	if (line_thickness != 100) {
		if (f_need_comma) {
			fp_tikz << ",";
		}
		fp_tikz << "line width=" << ((double)line_thickness * 0.01) << "mm";
		f_need_comma = true;
	}
	if (line_color != 1) {
		if (f_need_comma) {
			fp_tikz << ",";
		}
		fp_tikz << "color=";
		color_tikz(fp_tikz, line_color);
		f_need_comma = true;
	}

	fp_tikz << "] ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		if (f_v) {
			cout << "mp_graphics::polygon_idx_tikz "
					"x=" << x << " y=" << y << endl;
		}
		coords_min_max(x, y);
		user2dev(x, y);
		if (f_v) {
			cout << "mp_graphics::polygon_idx_tikz "
					"x'=" << x << " y'=" << y << endl;
		}

		if (i) {
			fp_tikz << " -- ";
		}
		output_xy_tikz(x, y);
	}
	fp_tikz << ";" << endl;
}

void mp_graphics::bezier_idx_tikz(
		int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int f_need_comma = false;
	int x, y, i;
	
	fp_tikz << "\\draw [";
	if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "->";
		f_need_comma = true;
	}
	else if (line_end_style == 0 && line_beg_style == 1) {
		fp_tikz << "<-";
		f_need_comma = true;
	}
	else if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "<->";
		f_need_comma = true;
	}
	if (line_thickness != 100) {
		if (f_need_comma) {
			fp_tikz << ",";
		}
		fp_tikz << "line width=" << ((double)line_thickness * 0.01) << "mm";
		f_need_comma = true;
	}
	if (line_color != 1) {
		if (f_need_comma) {
			fp_tikz << ",";
		}
		fp_tikz << "color=";
		color_tikz(fp_tikz, line_color);
		f_need_comma = true;
	}
	
	fp_tikz << "] ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);

		if (i) {
			fp_tikz << " .. ";
		}
#if 0
		if (i == 1) {
			fp_tikz << " .. controls ";
		}
		else if (i == n - 1) {
			fp_tikz << " .. ";
		}
		else if (i) {
			fp_tikz << " and ";
		}
#endif
		output_xy_tikz(x, y);
	}
	fp_tikz << ";" << endl;
}

void mp_graphics::color_tikz(
		std::ofstream &fp, int color)
{
	if (color == 0)
		fp << "white";
	else if (color == 1)
		fp << "black";
	else if (color == 2)
		fp << "red";
	else if (color == 3)
		fp << "green";
	else if (color == 4)
		fp << "blue";
	else if (color == 5)
		fp << "cyan";
	else if (color == 6)
		fp << "magenta";
	else if (color == 7)
		fp << "pink";
	else if (color == 8)
		fp << "orange";
	else if (color == 9)
		fp << "lightgray";
	else if (color == 10)
		fp << "brown";
	else if (color == 11)
		fp << "lime";
	else if (color == 12)
		fp << "olive";
	else if (color == 13)
		fp << "gray";
	else if (color == 14)
		fp << "purple";
	else if (color == 15)
		fp << "teal";
	else if (color == 16)
		fp << "violet";
	else if (color == 17)
		fp << "darkgray";
	else if (color == 18)
		fp << "lightgray";
	else if (color == 19)
		fp << "yellow";
	else if (color == 20)
		fp << "green!50!red";
	else if (color == 21)
		fp << "violet!50!red";
	else if (color == 22)
		fp << "cyan!50!red";
	else if (color == 23)
		fp << "green!50!blue";
	else if (color == 24)
		fp << "brown!50!red";
	else if (color == 25)
		fp << "purple!50!red";
	else {
		cout << "mp_graphics::color_tikz color = " << color 
			<< ", we don't have a color for this" << endl;
	}
}


// red, green, blue, cyan, magenta, yellow, black, gray,
// darkgray, lightgray, brown, lime, olive, orange, pink,
// purple, teal, violet and white.

void mp_graphics::fill_idx_tikz(
		std::ofstream &fp, int *Px, int *Py,
		int *Idx, int n, const char *symbol, int f_cycle)
{
	int f_need_comma;
	int i, x, y;
	
	fp << "\\fill [color=";

	color_tikz(fp, fill_color);
#if 0
	if (fill_color == 0)
		fp << "white";
	else if (fill_color == 1)
		fp << "black";
	else if (fill_color == 2)
		fp << "red";
	else if (fill_color == 3)
		fp << "green";
#endif

	if (fill_interior < 100) {
		fp << "!" << fill_interior;
	}
	f_need_comma = true;
	if (line_end_style == 1 && line_beg_style == 0) {
		if (f_need_comma) {
			fp << ", ";
		}
		fp << "->";
	}
	else if (line_end_style == 0 && line_beg_style == 1) {
		if (f_need_comma) {
			fp << ", ";
		}
		fp << "<-";
	}
	else if (line_end_style == 1 && line_beg_style == 0) {
		if (f_need_comma) {
			fp << ", ";
		}
		fp << "<->";
	}
	if (line_thickness != 100) {
		if (f_need_comma) {
			fp << ", ";
		}
		fp << "line width=" << ((double)line_thickness * 0.01) << "mm";
	}
	fp << "] ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);

		if (i) {
			fp << " -- ";
		}
		output_xy_tikz(x, y);
	}
	fp << ";" << endl;

}

#if 1
void mp_graphics::output_xy_tikz(
		int x, int y)
{
	fp_tikz << "(";
	output_x_tikz(x);
	fp_tikz << ",";
	output_y_tikz(y);
	fp_tikz << ")";
}
#else
void mp_graphics::output_xy_tikz(int x, int y)
{
	double dx, dy, x2, y2, f;
	
	dx = (double) x / 30000;
	dy = (double) y / 30000;
	f = 1. / sqrt(3. * 3. + dx * dx + dy * dy);
	x2 = dx * f;
	y2 = dy * f;
	fp_tikz << "(";
	if (ABS(x2) < 0.0001) {
		fp_tikz << 0;
		}
	else {
		fp_tikz << x2 * 10;
		}
	//output_x_tikz(x);
	fp_tikz << ",";
	if (ABS(y2) < 0.0001) {
		fp_tikz << 0;
		}
	else {
		fp_tikz << y2 * 10;
		}
	//output_y_tikz(y);
	fp_tikz << ")";
}
#endif

void mp_graphics::output_x_tikz(int x)
{
	double d;

	d = (double) x;
	d /= 30000.;
	if (ABS(d) < 0.0001) {
		fp_tikz << 0;
	}
	else {
		fp_tikz << d;
	}
}

void mp_graphics::output_y_tikz(int y)
{
	double d;

	d = (double) y;
	d /= 30000.;
	if (ABS(d) < 0.0001) {
		fp_tikz << 0;
	}
	else {
		fp_tikz << d;
	}
}



void mp_graphics::polygon3D(
		int *Px, int *Py,
		int dim, int x0, int y0, int z0, int x1, int y1, int z1)
{
	int idx0, idx1;
	idx0 = x0 * 9 + y0 * 3 + z0;
	idx1 = x1 * 9 + y1 * 3 + z1;
	polygon2(Px, Py, idx0, idx1);
}

void mp_graphics::integer_4pts(
		int *Px, int *Py,
		int p1, int p2, int p3, int p4,
		const char *align, int a)
{
	char str[100];
	string s;

	snprintf(str, sizeof(str), "%d", a);
	s.assign(str);
	text_4pts(Px, Py, p1, p2, p3, p4, align, s);
}

void mp_graphics::text_4pts(
		int *Px, int *Py,
		int p1, int p2, int p3, int p4,
		const char *align, std::string &s)
{
	int x = Px[p1] + Px[p2] + Px[p3] + Px[p4];
	int y = Py[p1] + Py[p2] + Py[p3] + Py[p4];
	x >>= 2;
	y >>= 2;
	aligned_text(x, y, align, s);
}


void mp_graphics::draw_graph(
		int x, int y,
		int dx, int dy, int nb_V,
		long int *Edges, int nb_E, int radius,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mp_graphics::draw_graph "
				"nb_V=" << nb_V << " nb_E=" << nb_E << endl;
	}
	double *X, *Y;
	double h = dy * .7;
	double w = dx * .7;
	int i, e, u, v;
	double phi = M_PI * 2. / nb_V;
	int Px[2];
	int Py[2];
	int rad = radius;
	//int rad = (int)(dx * .05);
	combinatorics::combinatorics_domain Combi;

	//cout << "draw_graph nb_V=" << nb_V << endl;

	if (f_v) {
		cout << "mp_graphics::draw_graph edges=";
		//Orbiter->Int_vec.print(cout, Edges, nb_E);
		Lint_vec_print(cout, Edges, nb_E);
		cout << endl;
	}

	sl_thickness(100); // was 30

	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		X[i] = cos(i * phi) * w;
		Y[i] = sin(i * phi) * h;
	}
	for (i = 0; i < nb_E; i++) {
		e = Edges[i];
		Combi.k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		polygon2(Px, Py, 0, 1);
	}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		nice_circle(Px[0], Py[0], rad);
	}
	delete [] X;
	delete [] Y;
	if (f_v) {
		cout << "mp_graphics::draw_graph done" << endl;
	}
}

void mp_graphics::draw_graph_with_distinguished_edge(
	int x, int y,
	int dx, int dy, int nb_V, long int *Edges, int nb_E,
	int distinguished_edge, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double *X, *Y;
	double h = dy * .7;
	double w = dx * .7;
	int i, e, u, v;
	double phi = M_PI * 2. / nb_V;
	int Px[2];
	int Py[2];
	int rad = (int)(dx * .05);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "mp_graphics::draw_graph_with_distinguished_edge "
				"nb_V=" << nb_V << endl;
	}
	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		X[i] = cos(i * phi) * w;
		Y[i] = sin(i * phi) * h;
	}

	sl_thickness(30);

	for (i = 0; i < nb_E; i++) {
		e = Edges[i];
		Combi.k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		if (distinguished_edge == i) {
			sl_thickness(80);
		}
		polygon2(Px, Py, 0, 1);
		if (distinguished_edge == i) {
			sl_thickness(30);
		}
	}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		nice_circle(Px[0], Py[0], rad);
	}
	delete [] X;
	delete [] Y;
}

void mp_graphics::draw_graph_on_multiple_circles(
		int x, int y,
		int dx, int dy, int nb_V,
		int *Edges, int nb_E, int nb_circles)
{
	double *X, *Y;
	double h = dy * .7;
	double w = dx * .7;
	int i, j, e, u, v;
	double phi;
	int Px[2];
	int Py[2];
	int rad = (int)(dx * .05);
	int nb_per_circle;
	combinatorics::combinatorics_domain Combi;

	cout << "mp_graphics::draw_graph_on_multiple_circles "
			"nb_V=" << nb_V << endl;
	nb_per_circle = nb_V / nb_circles;
	cout << "nb_per_circle = " << nb_per_circle << endl;
	phi = M_PI * 2. / nb_per_circle;
	X = new double [nb_V];
	Y = new double [nb_V];
	for (j = 0; j < nb_circles; j++) {
		for (i = 0; i < nb_per_circle; i++) {
			X[j * nb_per_circle + i] = cos(i * phi) * w;
			Y[j * nb_per_circle + i] = sin(i * phi) * h;
		}
		w = w * .5;
		h = h * .5;
	}
	for (i = 0; i < nb_E; i++) {
		e = Edges[i];
		Combi.k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		polygon2(Px, Py, 0, 1);
	}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		nice_circle(Px[0], Py[0], rad);
	}
	delete [] X;
	delete [] Y;
}

void mp_graphics::draw_graph_on_2D_grid(
		int x, int y, int dx, int dy, int rad, int nb_V,
		int *Edges, int nb_E, int *coords_2D, int *Base,
		int f_point_labels,
		int point_label_offset, int f_directed)
{
	double *X, *Y;
	//double h = dy * .7;
	//double w = dx * .7;
	int i, u, v;
	int Px[2];
	int Py[2];
	//int rad = (int)(dx * .1);

	cout << "mp_graphics::draw_graph_on_2D_grid "
			"nb_V=" << nb_V << endl;
	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		u = coords_2D[2 * i + 0];
		v = coords_2D[2 * i + 1];
		X[i] = u * Base[0] + v * Base[2];
		Y[i] = u * Base[1] + v * Base[3];
	}

	if (f_directed) {
		sl_ends(0, 1);
	}
	for (i = 0; i < nb_E; i++) {
		u = Edges[2 * i + 0];
		v = Edges[2 * i + 1];
		//k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		polygon2(Px, Py, 0, 1);
	}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		nice_circle(Px[0], Py[0], rad);
	}

	if (f_point_labels) {

		for (i = 0; i < nb_V; i++) {
			string s;

			s = std::to_string(point_label_offset);

			Px[0] = x + (int) X[i];
			Py[0] = y + (int) Y[i];
			aligned_text(Px[0], Py[0], "", s);
		}
	}
	if (f_directed) {
		sl_ends(0, 0);
	}
	delete [] X;
	delete [] Y;
}

void mp_graphics::draw_tournament(
		int x, int y,
		int dx, int dy, int nb_V, long int *Edges, int nb_E,
		int radius,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double *X, *Y;
	double h = dy * .7;
	double w = dx * .7;
	int i, a, a2, swap, u, v;
	double phi = M_PI * 2. / nb_V;
	int Px[3];
	int Py[3];
	int rad = radius;
	//int rad = (int)(dx * .05);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "mp_graphics::draw_tournament nb_V=" << nb_V << endl;
	}
	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		X[i] = cos(i * phi) * w;
		Y[i] = sin(i * phi) * h;
	}

	// draw the edges:
	for (i = 0; i < nb_E; i++) {
		a = Edges[i];



		swap = a % 2;
		a2 = a / 2;
		Combi.k2ij(a2, u, v, nb_V);



		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		if (swap) {
			Px[2] = (3 * Px[0] + Px[1]) >> 2;
			Py[2] = (3 * Py[0] + Py[1]) >> 2;
			sl_ends(0, 1);
			polygon2(Px, Py, 1, 2);
		}
		else {
			Px[2] = (Px[0] + 3 * Px[1]) >> 2;
			Py[2] = (Py[0] + 3 * Py[1]) >> 2;
			sl_ends(0, 1);
			polygon2(Px, Py, 0, 2);
		}
	}

	// now draw vertices:

	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		nice_circle(Px[0], Py[0], rad);
	}
	delete [] X;
	delete [] Y;
}

void mp_graphics::draw_bitmatrix2(
		int f_dots,
	int f_partition,
	int nb_row_parts, int *row_part_first,
	int nb_col_parts, int *col_part_first,
	int f_row_grid, int f_col_grid,
	int f_bitmatrix,
	data_structures::bitmatrix *Bitmatrix,
	int *M, int m, int n,
	int f_has_labels, int *labels)
{
	string s;
	grid_frame F;
	int i, j, ii, jj, ij, a, cnt, mn, mtn, mtn1;
	int indent = 0;

	mn = MAXIMUM(m, n);
	F.f_matrix_notation = true;
	F.m = m;
	F.n = n;
	F.origin_x = 0.;
	F.origin_y = 0.;
	F.dx = ONE_MILLION / (10 * mn);
	F.dy = ONE_MILLION / (10 * mn);

	cout << "draw_it2" << endl;
	cout << "dx=" << F.dx << endl;
	cout << "dy=" << F.dy << endl;

	// draw a box around it:

	s.assign("box outline");
	comment(s);

	grid_polygon2(&F, 0, 0, 10 * m, 0);
	grid_polygon2(&F, 10 * m, 0, 10 * m, 10 * n);
	grid_polygon2(&F, 10 * m, 10 * n, 0, 10 * n);
	grid_polygon2(&F, 0, 10 * n, 0, 0);

	sf_interior(100);
	sf_color(1); // black


	sl_thickness(20); // 100 is standard

	if (f_partition) {
		s.assign("row partition");
		comment(s);
		for (i = 0; i < nb_row_parts + 1; i++) {
			s = "row_part_" + std::to_string(i);
			comment(s);
			ii = row_part_first[i];
			grid_polygon2(&F, ii * 10, 0 * 10, ii * 10, (n + 0) * 10);
			//G.grid_polygon2(&F, ii * 10, -1 * 10, ii * 10, (n + 1) * 10);
		}
		s.assign("column partition");
		comment(s);
		for (j = 0; j < nb_col_parts + 1; j++) {
			s = "col_part_" + std::to_string(j);
			comment(s);
			jj = col_part_first[j];
			grid_polygon2(&F, 0 * 10, jj * 10, (m + 0) * 10, jj * 10);
			//G.grid_polygon2(&F, -1 * 10, jj * 10, (m + 1) * 10, jj * 10);
		}
	}




	sl_thickness(10); // 100 is standard

	sf_interior(0);
	sf_color(1);

	if (f_row_grid) {
		for (i = 0; i < m; i++) {

			string s;

			s = "row_" + std::to_string(i);
			if (f_has_labels) {
				s += " label " + std::to_string(labels[i]);
			}
			comment(s);
			j = 0;
			grid_fill_polygon5(&F,
				10 * i + indent, 10 * j + indent,
				10 * (i + 1) - indent, 10 * j + indent,
				10 * (i + 1) - indent, 10 * n - indent,
				10 * i + indent, 10 * n - indent,
				10 * i + indent, 10 * j + indent);
		}
	}

	if (f_col_grid) {
		for (j = 0; j < n; j++) {
			string s;

			s = "col_" + std::to_string(j);

			if (f_has_labels) {
				s += " label " + std::to_string(labels[j]);
			}
			comment(s);
			i = 0;
			grid_fill_polygon5(&F,
				10 * i + indent, 10 * j + indent,
				10 * m - indent, 10 * j + indent,
				10 * m - indent, 10 * (j + 1) - indent,
				10 * i + indent, 10 * (j + 1) - indent,
				10 * i + indent, 10 * j + indent);
		}
	}

	if (f_has_labels) {
		for (i = 0; i < m; i++) {
			string s;
			s += std::to_string(labels[i]);
			grid_aligned_text(&F, i * 10 + 5, -1 * 10, "", s);
		}
		for (j = 0; j < n; j++) {
			string s;
			s += std::to_string(labels[m + j] - m);
			grid_aligned_text(&F, -1 * 10, j * 10 + 5, "", s);
		}
	}


	sl_thickness(10); // 100 is standard

	sf_interior(100);

	cnt = 0;
	mtn = m * n;
	mtn1 = mtn / 20;
	if (mtn1 == 0) {
		mtn1 = 1;
	}
	for (i = 0; i < m; i++) {
#if 0
		if (i && (i % 1000) == 0) {
			cout << "draw_it2 " << i << " / " << m << endl;
			}
#endif
		for (j = 0; j < n; j++) {

			ij = i * n + j;
			if ((ij % mtn1) == 0) {
				cout << "draw_bitmatrix2 " << ij << " / " << mtn << endl;
			}

			//a = Aij(i, j);

			if (f_bitmatrix) {
				a = Bitmatrix->s_ij(i, j);
				//a = bitvector_s_i(D, i * n + j);
			}
			else {
				a = M[i * n + j];
			}
			if (a == 0) {
				continue;
			}
			cnt++;

			//if (cnt > 5000)  continue;
			//grid_fill_polygon4(&F, i, j, i + 1, j, i + 1, j + 1, i, j + 1);



			string s;
			s = std::to_string(i) + "_" + std::to_string(j);
			comment(s);

			if (f_dots) {
				grid_polygon2(&F, 10 * i, 10 * j, 10 * i, 10 * j);
			}
			else {
				sf_interior(100);
				sf_color(1);

#if 0
				grid_fill_polygon4(&F,
					10 * i + 1, 10 * j + 1,
					10 * (i + 1) - 1, 10 * j + 1,
					10 * (i + 1) - 1, 10 * (j + 1) - 1,
					10 * i + 1, 10 * (j + 1) - 1);
#else
				grid_fill_polygon5(&F,
					10 * i + indent, 10 * j + indent,
					10 * (i + 1) - indent, 10 * j + indent,
					10 * (i + 1) - indent, 10 * (j + 1) - indent,
					10 * i + indent, 10 * (j + 1) - indent,
					10 * i + indent, 10 * j + indent);
#endif
				//grid_polygon2(&F, i, j, i + 1, j);
				//grid_polygon2(&F, i + 1, j, i + 1, j + 1);
				//grid_polygon2(&F, i + 1, j + 1, i, j + 1);
				//grid_polygon2(&F, i, j + 1, i, j);
			}
		}
	}
	cout << "mp_graphics::draw_bitmatrix2 "
			"# of non-zero coefficients = " << cnt << endl;
}



void mp_graphics::draw_density2(
		int no,
	int *outline_value, int *outline_number, int outline_sz,
	int min_value, int max_value,
	int offset_x, int f_switch_x,
	int f_title, std::string &title,
	std::string &label_x,
	int f_circle, int circle_at, int circle_rad,
	int f_mu, int f_sigma, int nb_standard_deviations,
	int f_v_grid, int v_grid, int f_h_grid, int h_grid)
{
	int i;
	int Px[1000], Py[1000];
	//int phi = 360 / 12;
	//int rad1 = 400;
	int y_in, x, y, k;

	int min_x, max_x, min_y, max_y;
	int sum, a;
	int mini_x, i0;
	double average;
	double sigma; // standard deviation
	double sum1, f;
	plot_tools Pt;

	if (outline_value[0] == 0) {
		i0 = 1;
		mini_x = outline_number[0];
	}
	else {
		i0 = 0;
		mini_x = 0; // outline_number[0];
	}
	min_x = 0;
	max_x = outline_number[outline_sz - 1]; // number of students

	min_y = min_value;
	max_y = max_value;

	sum = 0;
	for (i = outline_sz - 1; i >= i0; i--) {
		if (i) {
			a = outline_number[i] - outline_number[i - 1];
		}
		else {
			a = outline_number[i];
		}
		sum += outline_value[i] * a;
	}

	//cout << "sum=" << sum << endl;
	average = sum / MAXIMUM((max_x - mini_x), 1);

	// now for the standard deviation:
	sum1 = 0;
	for (i = outline_sz - 1; i >= 0; i--) {
		if (i) {
			a = outline_number[i] - outline_number[i - 1];
		}
		else {
			a = outline_number[i];
		}
		f = ((double) outline_value[i]) - average;
		f = f * f;
		sum1 += f;
	}
	sigma = sqrt(sum1 / max_x);


	Pt.get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, false);
	for (i = 0; i < outline_sz; i++) {
		Pt.get_coord(Px, Py, 2, outline_number[i], outline_value[i],
			min_x, min_y, max_x, max_y, false);
		Px[1] = Px[0];
		Py[1] = Py[2];
		polygon3(Px, Py, 0, 1, 2);
		Px[0] = Px[2];
		Py[0] = Py[2];
	}
	Pt.get_coord(Px, Py, 2, max_x, max_value,
		min_x, min_y, max_x, max_y, false);
	polygon2(Px, Py, 0, 2);
	Pt.get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, false);
	Pt.get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, false);
	Pt.get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, false);
	Pt.get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, false);
	polygon5(Px, Py, 0, 1, 2, 3, 0);


	if (f_switch_x) {
		string s;
		s = "{\\bf {\\large " + std::to_string(max_x + offset_x) + "}}";
		aligned_text(Px[0], Py[0], "t", s);
		s = "{\\bf {\\large " + std::to_string(min_x + offset_x) + "}}";
		aligned_text(Px[1], Py[1], "t", s);
	}
	else {
		string s;
		s = "{\\bf {\\large " + std::to_string(min_x + offset_x) + "}}";
		aligned_text(Px[0], Py[0], "t", s);
		s = "{\\bf {\\large " + std::to_string(max_x + offset_x) + "}}";
		aligned_text(Px[1], Py[1], "t", s);
	}


	string s;


	s = "{\\bf {\\large " + std::to_string(min_y) + "}}";
	aligned_text(Px[0], Py[0], "r", s);
	s = "{\\bf {\\large " + std::to_string(max_y) + "}}";
	aligned_text(Px[3], Py[3], "r", s);



	Px[0] = 5 * 100;
	Py[0] = 0;
	s = "{\\bf {\\large " + label_x + "}}";
	aligned_text(Px[0], Py[0], "t", s);

	Px[0] = 5 * 100;
	Py[0] = -50;

	string s_mu;
	string s_sigma;
	string s_inside;

	if (f_mu) {
		s_mu = "\\overline{x}=" + std::to_string(average);
	}
	if (f_sigma) {
		s_sigma = "\\sigma=" + std::to_string(sigma);
	}

	if (f_mu && f_sigma) {
		s_inside = s_mu + "\\, " + s_sigma;
	}
	else if (f_mu) {
		s_inside = s_mu;
	}
	else {
		s_inside = s_sigma;
	}

	if (f_mu || f_sigma) {
		s = "{\\bf {\\large $" + s_inside + "$}}";
		aligned_text(Px[0], Py[0], "t", s);
	}


	if (f_mu) {
		y_in = (int) average;
		Pt.y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		Pt.get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, false);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		s.assign("$\\overline{x}$");
		aligned_text(Px[2], Py[2], "r", s);
	}


	if (f_circle) {
		Pt.y_to_pt_on_curve(circle_at, x, y,
			outline_value, outline_number, outline_sz);
		Pt.get_coord(Px, Py, 0, x, y, min_x, min_y, max_x, max_y, false);
		circle(Px[0], Py[0], circle_rad);
	}


	for (k = 1; k < nb_standard_deviations; k++) {
		y_in = (int) (average + k * sigma);
		Pt.y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		Pt.get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, false);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			s = "$\\overline{x}+" + std::to_string(k) + " \\sigma$";
		}
		else {
			s = "$\\overline{x}+\\sigma$";
		}
		aligned_text(Px[2], Py[2], "r", s);

		y_in = (int) (average - k * sigma);
		Pt.y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		Pt.get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, false);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			s = "{\\bf {\\large $\\overline{x}-" + std::to_string(k) + " \\sigma$}}";
		}
		else {
			s = "{\\bf {\\large $\\overline{x}-\\sigma$}}";
		}
		aligned_text(Px[2], Py[2], "r", s);
	}

#if 0
	y_in = (int) (average + 2 * sigma);
	y_to_pt_on_curve(y_in, x, y,
		outline_value, outline_number, outline_sz);
	get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y);
	Py[0] -= 10;
	G.polygon3(Px, Py, 0, 1, 2);
	G.aligned_text(Px[2], Py[2], "r", "$\\overline{x}+2\\sigma$");

	y_in = (int) (average - 2 * sigma);
	y_to_pt_on_curve(y_in, x, y,
		outline_value, outline_number, outline_sz);
	get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y);
	Py[0] -= 10;
	G.polygon3(Px, Py, 0, 1, 2);
	G.aligned_text(Px[2], Py[2], "r", "$\\overline{x}-2\\sigma$");
#endif




	int line_dashing = 50;
	sl_udsty(line_dashing);

	if (f_v_grid) {
		int delta;

		delta = 1000 / v_grid;
		for (i = 1; i <= v_grid - 1; i++) {
			Px[0] = i * delta;
			Py[0] = 0;
			Px[1] = i * delta;
			Py[1] = 1000;
			polygon2(Px, Py, 0, 1);
		}
	}
	if (f_h_grid) {
		int delta;

		delta = 1000 / h_grid;
		for (i = 1; i <= h_grid - 1; i++) {
			Px[0] = 0;
			Py[0] = i * delta;
			Px[1] = 1000;
			Py[1] = i * delta;
			polygon2(Px, Py, 0, 1);
		}
	}


	if (f_title) {
		Px[0] = 5 * 100;
		Py[0] = 1050;
		s = "{\\bf {\\large " + title + "}}";
		aligned_text(Px[0], Py[0], "b", s);
	}

}

void mp_graphics::draw_density2_multiple_curves(
		int no,
	int **outline_value, int **outline_number, int *outline_sz, int nb_curves,
	int min_x, int max_x, int min_y, int max_y,
	int offset_x, int f_switch_x,
	int f_title, std::string &title,
	std::string &label_x,
	int f_v_grid, int v_grid, int f_h_grid, int h_grid,
	int f_v_logarithmic, double log_base)
{
	int i;
	int Px[1000], Py[1000];
	//int phi = 360 / 12;
	//int rad1 = 400;
	string s;
	int curve;
	plot_tools Pt;

#if 0
	int min_x, max_x, min_y, max_y;

	min_x = INT_MAX;
	max_x = int_MIN;
	for (curve = 0; curve < nb_curves; curve++) {
		min_x = MINIMUM(min_x, outline_number[curve][0]);
		min_x = MINIMUM(min_x, outline_number[curve][outline_sz[curve] - 1]);
		max_x = MAXIMUM(max_x, outline_number[curve][0]);
		max_x = MAXIMUM(max_x, outline_number[curve][outline_sz[curve] - 1]);
		}

	cout << "min_x=" << min_x << endl;
	cout << "max_x=" << max_x << endl;

	min_y = min_value;
	max_y = max_value;
#endif


	for (curve = 0; curve < nb_curves; curve++) {
		if (f_v_logarithmic) {
			Pt.get_coord_log(Px, Py, 0,
					min_x, min_y, min_x, min_y, max_x, max_y,
					log_base, f_switch_x);
		}
		else {
			Pt.get_coord(Px, Py, 0,
					min_x, min_y, min_x, min_y, max_x, max_y, f_switch_x);
		}
		for (i = 0; i < outline_sz[curve]; i++) {
			if (f_v_logarithmic) {
				Pt.get_coord_log(Px, Py, 2,
					outline_number[curve][i], outline_value[curve][i],
					min_x, min_y, max_x, max_y, log_base, f_switch_x);
			}
			else {
				Pt.get_coord(Px, Py, 2,
					outline_number[curve][i], outline_value[curve][i],
					min_x, min_y, max_x, max_y, f_switch_x);
			}
			Px[1] = Px[0];
			Py[1] = Py[2];
			polygon3(Px, Py, 0, 1, 2);
			Px[0] = Px[2];
			Py[0] = Py[2];
		}
		if (f_v_logarithmic) {
			Pt.get_coord_log(Px, Py, 2, max_x, max_y,
				min_x, min_y, max_x, max_y, log_base, f_switch_x);
		}
		else {
			Pt.get_coord(Px, Py, 2, max_x, max_y,
				min_x, min_y, max_x, max_y, f_switch_x);
		}
		polygon2(Px, Py, 0, 2);
	}


	if (f_v_logarithmic) {
		Pt.get_coord_log(Px, Py, 0,
				min_x, min_y, min_x, min_y, max_x, max_y, log_base, false);
		Pt.get_coord_log(Px, Py, 1,
				max_x, min_y, min_x, min_y, max_x, max_y, log_base, false);
		Pt.get_coord_log(Px, Py, 2,
				max_x, max_y, min_x, min_y, max_x, max_y, log_base, false);
		Pt.get_coord_log(Px, Py, 3,
				min_x, max_y, min_x, min_y, max_x, max_y, log_base, false);
	}
	else {
		Pt.get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, false);
		Pt.get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, false);
	}
	polygon5(Px, Py, 0, 1, 2, 3, 0);



	if (f_switch_x) {
		s = "{\\bf {\\large " + std::to_string(max_x + offset_x) + "}}";
		aligned_text(Px[1], Py[1], "t", s);
		s = "{\\bf {\\large " + std::to_string(min_x + offset_x) + "}}";
		aligned_text(Px[0], Py[0], "t", s);
	}
	else {
		s = "{\\bf {\\large " + std::to_string(min_x + offset_x) + "}}";
		aligned_text(Px[0], Py[0], "t", s);
		s = "{\\bf {\\large " + std::to_string(max_x + offset_x) + "}}";
		aligned_text(Px[1], Py[1], "t", s);
	}

	s = "{\\bf {\\large " + std::to_string(min_y) + "}}";
	aligned_text(Px[0], Py[0], "r", s);
	s = "{\\bf {\\large " + std::to_string(max_y) + "}}";
	aligned_text(Px[3], Py[3], "r", s);



	Px[0] = 5 * 100;
	Py[0] = 0;
	s = "{\\bf {\\large " + label_x + "}}";
	aligned_text(Px[0], Py[0], "t", s);





	int line_dashing = 50;
	int line_thickness = 15;
	sl_udsty(line_dashing);
	sl_thickness(line_thickness);

	if (f_v_grid) {
		if (false) {
			double delta, a;

			delta = log10(max_x - min_x) / v_grid;
			for (i = 1; i <= v_grid - 1; i++) {
				a = min_x + pow(10, i * delta);
				Px[0] = (int)a;
				Py[0] = 0;
				Px[1] = (int)a;
				Py[1] = 1000;
				polygon2(Px, Py, 0, 1);
			}
		}
		else {
			int delta;
			delta = 1000 / v_grid;
			for (i = 1; i <= v_grid - 1; i++) {
				Px[0] = i * delta;
				Py[0] = 0;
				Px[1] = i * delta;
				Py[1] = 1000;
				polygon2(Px, Py, 0, 1);
			}
		}
	}
	if (f_h_grid) {
		if (f_v_logarithmic) {
			double delta, a;

			delta = (log(max_y - min_y + 1) / log(log_base))/ h_grid;
			for (i = 1; i <= h_grid - 1; i++) {
				a = min_y + pow(log_base, i * delta);
				Pt.get_coord_log(Px, Py, 2, min_x, (int)a,
						min_x, min_y, max_x, max_y, log_base,
						false /* f_switch_x */);
				Px[0] = Px[2];
				Py[0] = Py[2];
				Px[1] = 1000;
				Py[1] = Py[2];
				polygon2(Px, Py, 0, 1);

				s = "{" + std::to_string(a) + "}";
				aligned_text(Px[0], Py[0], "r", s);
			}
		}
		else {
			int delta;

			delta = 1000 / h_grid;
			for (i = 1; i <= h_grid - 1; i++) {
				Px[0] = 0;
				Py[0] = i * delta;
				Px[1] = 1000;
				Py[1] = i * delta;
				polygon2(Px, Py, 0, 1);
			}
		}
	}


	if (f_title) {
		Px[0] = 5 * 100;
		Py[0] = 1050;
		s = "{\\bf {\\large " + title + "}}";
		aligned_text(Px[0], Py[0], "b", s);
	}

}

void mp_graphics::projective_plane_draw_grid2(
		layered_graph_draw_options *O,
		int q,
		int *Table, int nb,
		int f_point_labels,
		std::string *Point_labels, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double a, b;
	int x1, x2, x3;
	string s;

	//int rad = 17000;
	int i, h;
	//double x_stretch = 1.0 / (double) (q + 1);
	//double y_stretch = 1.0 / (double) (q + 1);
	//double x_stretch = 0.01;
	//double y_stretch = 0.01;

	double *Dx, *Dy;
	int *Px, *Py;
	double dx = O->xin * 0.5 / (double) (q + 1);
	double dy = O->yin * 0.5 / (double) (q + 1); // stretch factor
	int N = 1000;


	if (f_v) {
		cout << "projective_plane_draw_grid2" << endl;
		cout << "projective_plane_draw_grid2 q=" << q << endl;
		//cout << "projective_plane_draw_grid2 x_stretch=" << x_stretch << " y_stretch" << y_stretch << endl;
		cout << "projective_plane_draw_grid2 dx=" << dx << " dy=" << dy << endl;
	}


	Px = NEW_int(N);
	Py = NEW_int(N);
	Dx = new double[N];
	Dy = new double[N];






	if (f_v) {
		cout << "projective_plane_draw_grid2 "
				"before G.draw_axes_and_grid" << endl;
		}


	draw_axes_and_grid(O,
		0., (double)(q - 1), 0., (double)(q - 1), dx, dy,
		true /* f_x_axis_at_y_min */, true /* f_y_axis_at_x_min */,
		1 /* x_mod */, 1 /* y_mod */, 1, 1,
		-1. /* x_labels_offset */, -1. /* y_labels_offset */,
		0.5 /* x_tick_half_width */, 0.5 /* y_tick_half_width */,
		true /* f_v_lines */, 1 /* subdivide_v */,
		true /* f_h_lines */, 1 /* subdivide_h */,
		verbose_level - 1);



	if (f_v) {
		cout << "projective_plane_draw_grid2 "
				"after G.draw_axes_and_grid" << endl;
		}

	Dx[0] = q;
	Dy[0] = -1;
	Dx[1] = -1;
	Dy[1] = q;
	for (i = 0; i < 2; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
		}
	s.assign("$x$");
	text(Px[0], Py[0], s);
	s.assign("$y$");
	text(Px[1], Py[1], s);

	sl_thickness(100);

	projective_plane_make_affine_point(q, 0, 1, 0, a, b);
	Dx[0] = a;
	Dy[0] = b;
	projective_plane_make_affine_point(q, 1, 0, 0, a, b);
	Dx[1] = a;
	Dy[1] = b;
	if (EVEN(q)) {
		projective_plane_make_affine_point(q, q / 2, 1, 0, a, b);
		Dx[2] = a;
		Dy[2] = b;
		Dx[3] = a;
		Dy[3] = b;
		}
	else {
		projective_plane_make_affine_point(q, q / 2, 1, 0, a, b);
		Dx[2] = a;
		Dy[2] = b;
		projective_plane_make_affine_point(q, (q + 1) / 2, 1, 0, a, b);
		Dx[3] = a;
		Dy[3] = b;
		}

	for (i = 0; i < 4; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
		}
	polygon4(Px, Py, 0, 2, 3, 1);

	if (!O->f_nodes_empty) {

		if (f_v) {
			cout << "projective_plane_draw_grid2 "
					"drawing points, nb=" << nb << endl;
			}

		sl_thickness(50);

#if 0
		if (nb >= 40) {
			rad = 2000;
			}
#endif
		for (h = 0; h < nb; h++) {
			x1 = Table[3 * h + 0];
			x2 = Table[3 * h + 1];
			x3 = Table[3 * h + 2];
			//get_ab(q, x1, x2, x3, a, b);
			projective_plane_make_affine_point(q, x1, x2, x3, a, b);

			if (f_v) {
				cout << "projective_plane_draw_grid2 "
						"point " << h << " : " << x1 << ", " << x2
						<< ", " << x3 << " : " << a << ", " << b << endl;
			}

			Dx[0] = a;
			Dy[0] = b;

			for (i = 0; i < 1; i++) {
				Px[i] = Dx[i] * dx;
				Py[i] = Dy[i] * dy;
				}

			//G.nice_circle(Px[a * Q + b], Py[a * Q + b], rad);
			nice_circle(Px[0], Py[0], O->rad);
			if (f_point_labels) {
				text(Px[0], Py[0], Point_labels[h]);
			}
		}


	}
	else {
		cout << "projective_plane_draw_grid2 not drawing any points" << endl;
	}





	FREE_int(Px);
	FREE_int(Py);
	delete [] Dx;
	delete [] Dy;



	if (f_v) {
		cout << "projective_plane_draw_grid2 done" << endl;
		}
}


void mp_graphics::draw_matrix_in_color(
	int f_row_grid, int f_col_grid,
	int *Table, int nb_colors,
	int m, int n,
	int *color_scale, int nb_colors_in_scale,
	int f_has_labels, int *labels)
{
	grid_frame F;
	int i, j, a, mn;
	int indent = 0;
	string s;

	mn = MAXIMUM(m, n);
	F.f_matrix_notation = true;
	F.m = m;
	F.n = n;
	F.origin_x = 0.;
	F.origin_y = 0.;
	F.dx = ONE_MILLION / (10 * mn);
	F.dy = ONE_MILLION / (10 * mn);

	cout << "mp_graphics::draw_matrix_in_color" << endl;
	cout << "dx=" << F.dx << endl;
	cout << "dy=" << F.dy << endl;

	// draw a box around it:

	s.assign("box outline");
	comment(s);

	grid_polygon2(&F, 0, 0, 10 * m, 0);
	grid_polygon2(&F, 10 * m, 0, 10 * m, 10 * n);
	grid_polygon2(&F, 10 * m, 10 * n, 0, 10 * n);
	grid_polygon2(&F, 0, 10 * n, 0, 0);

	sf_interior(100);
	sf_color(1); // black


	//sl_thickness(20); // 100 is standard



	sl_thickness(10); // 100 is standard

	sf_interior(0);
	sf_color(1);

	if (f_row_grid) {
		for (i = 0; i < m; i++) {
			s = "row_" + std::to_string(i);
			if (f_has_labels) {
				s += " label " + std::to_string(labels[i]);
				}
			comment(s);
			j = 0;
			grid_fill_polygon5(&F,
				10 * i + indent, 10 * j + indent,
				10 * (i + 1) - indent, 10 * j + indent,
				10 * (i + 1) - indent, 10 * n - indent,
				10 * i + indent, 10 * n - indent,
				10 * i + indent, 10 * j + indent);
			}
		}

	if (f_col_grid) {
		for (j = 0; j < n; j++) {
			s = "col_" + std::to_string(j);
			if (f_has_labels) {
				s += " label " + std::to_string(labels[j]);
				}
			comment(s);
			i = 0;
			grid_fill_polygon5(&F,
				10 * i + indent, 10 * j + indent,
				10 * m - indent, 10 * j + indent,
				10 * m - indent, 10 * (j + 1) - indent,
				10 * i + indent, 10 * (j + 1) - indent,
				10 * i + indent, 10 * j + indent);
			}
		}

	if (f_has_labels) {
		for (i = 0; i < m; i++) {
			s = std::to_string(labels[i]);
			grid_aligned_text(&F, i * 10 + 5, -1 * 10, "", s);
			}
		for (j = 0; j < n; j++) {
			s = std::to_string(labels[m + j] - m);
			grid_aligned_text(&F, -1 * 10, j * 10 + 5, "", s);
			}
		}


	sl_thickness(10); // 100 is standard

	sf_interior(100);

	//cnt = 0;


	double color_step = (double) nb_colors / (double) (nb_colors_in_scale);
	//double shade_step = (double) color_step * 100 / ((double) nb_colors / (double)nb_colors_in_scale);
	double f_sufficiently_many_colors;
	int c, a1, c1;

	cout << "color_step=" << color_step << endl;
	//cout << "shade_step=" << shade_step << endl;

	if (nb_colors_in_scale > nb_colors) {
		f_sufficiently_many_colors = true;
	}
	else {
		f_sufficiently_many_colors = false;
	}

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {


			a = Table[i * n + j];

			//cnt++;

			//if (cnt > 5000)  continue;
			//grid_fill_polygon4(&F, i, j, i + 1, j, i + 1, j + 1, i, j + 1);




			if (a) {

				s = std::to_string(i) + "_" + std::to_string(j);
				comment(s);

				c = (int)((double) a / (double) color_step) + 1;
				if (f_sufficiently_many_colors) {
					sf_interior(100);
					sf_color(a);
				}
				else {
					a1 = a - c * color_step;
					c1 = a1 / color_step * 40 + 60;
					sf_color(c);
					sf_interior(c1);
				}

				grid_fill_polygon5(&F,
					10 * i + indent, 10 * j + indent,
					10 * (i + 1) - indent, 10 * j + indent,
					10 * (i + 1) - indent, 10 * (j + 1) - indent,
					10 * i + indent, 10 * (j + 1) - indent,
					10 * i + indent, 10 * j + indent);
			}

			//grid_polygon2(&F, i, j, i + 1, j);
			//grid_polygon2(&F, i + 1, j, i + 1, j + 1);
			//grid_polygon2(&F, i + 1, j + 1, i, j + 1);
			//grid_polygon2(&F, i, j + 1, i, j);
		} // next j
	} // next i
	cout << "mp_graphics::draw_matrix_in_color done" << endl;
}




static void projective_plane_make_affine_point(
		int q, int x1, int x2, int x3, double &a, double &b)
{
	if (x3 == 0) {
		if (x2 == 0) {
			a = 3 * q / 2;
			b = q / 2;
			}
		else {
			a = q / 2 + x1;
			b = 3 * q / 2 - x1;
			}
		// make it lie on the rim:
		// if x1 < q/2, we decrease the y coordinate.
		// if x1 > q/2, we decrease the x coordinate.
		if (x2 == 0) {
			a = q;
			}
		else {
			if (x1 < (q / 2)) {
				b = q;
				}
			if (x1 > q / 2) {
				a = q;
				}
			}
		}
	else {
		a = x1;
		b = x2;
		}
}

void mp_graphics::domino_draw1(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	circle(Px[0], Py[0], rad);
}


void mp_graphics::domino_draw2(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);
	if (f_horizontal) {
		Px[1] = Px[0] - ddx;
		Py[1] = Py[0] + ddy;
		Px[2] = Px[0] + ddx;
		Py[2] = Py[0] - ddy;
		}
	else {
		Px[1] = Px[0] + ddx;
		Py[1] = Py[0] + ddy;
		Px[2] = Px[0] - ddx;
		Py[2] = Py[0] - ddy;
		}

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
}

void mp_graphics::domino_draw3(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);
	if (f_horizontal) {
		Px[1] = Px[0] - ddx;
		Py[1] = Py[0] + ddy;
		Px[2] = Px[0] + ddx;
		Py[2] = Py[0] - ddy;
		}
	else {
		Px[1] = Px[0] + ddx;
		Py[1] = Py[0] + ddy;
		Px[2] = Px[0] - ddx;
		Py[2] = Py[0] - ddy;
		}

	circle(Px[0], Py[0], rad);
	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
}

void mp_graphics::domino_draw4(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
}

void mp_graphics::domino_draw5(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	circle(Px[0], Py[0], rad);
	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
}

void mp_graphics::domino_draw6(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	if (f_horizontal) {
		Px[5] = Px[0];
		Py[5] = Py[0] + ddy;
		Px[6] = Px[0];
		Py[6] = Py[0] - ddy;
		}
	else {
		Px[5] = Px[0] - ddx;
		Py[5] = Py[0];
		Px[6] = Px[0] + ddx;
		Py[6] = Py[0];
		}

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
	circle(Px[5], Py[5], rad);
	circle(Px[6], Py[6], rad);
}

void mp_graphics::domino_draw7(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	if (f_horizontal) {
		Px[5] = Px[0];
		Py[5] = Py[0] + ddy;
		Px[6] = Px[0];
		Py[6] = Py[0] - ddy;
		Px[7] = Px[0];
		Py[7] = Py[0];
		}
	else {
		Px[5] = Px[0] - ddx;
		Py[5] = Py[0];
		Px[6] = Px[0] + ddx;
		Py[6] = Py[0];
		Px[7] = Px[0];
		Py[7] = Py[0];
		}

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
	circle(Px[5], Py[5], rad);
	circle(Px[6], Py[6], rad);
	circle(Px[7], Py[7], rad);
}

void mp_graphics::domino_draw8(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	if (f_horizontal) {
		Px[5] = Px[0];
		Py[5] = Py[0] + ddy;
		Px[6] = Px[0];
		Py[6] = Py[0] - ddy;
		Px[7] = Px[0];
		Py[7] = Py[0];
		Px[8] = Px[0] - ddx;
		Py[8] = Py[0];
		Px[9] = Px[0] + ddx;
		Py[9] = Py[0];
		}
	else {
		Px[5] = Px[0] - ddx;
		Py[5] = Py[0];
		Px[6] = Px[0] + ddx;
		Py[6] = Py[0];
		Px[7] = Px[0];
		Py[7] = Py[0];
		Px[8] = Px[0];
		Py[8] = Py[0] + ddy;
		Px[9] = Px[0];
		Py[9] = Py[0] - ddy;
		}

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
	circle(Px[5], Py[5], rad);
	circle(Px[6], Py[6], rad);
	//circle(Px[7], Py[7], rad);
	circle(Px[8], Py[8], rad);
	circle(Px[9], Py[9], rad);
}

void mp_graphics::domino_draw9(
		int M,
		int i, int j, int dx, int dy, int rad, int f_horizontal)
{
	int Px[100], Py[100];
	int ddx = compute_dd(dx);
	int ddy = compute_dd(dy);

	Px[0] = j * dx + (dx >> 1);
	Py[0] = (M - i) * dy - (dy >> 1);

	Px[1] = Px[0] - ddx;
	Py[1] = Py[0] + ddy;
	Px[2] = Px[0] + ddx;
	Py[2] = Py[0] + ddy;
	Px[3] = Px[0] + ddx;
	Py[3] = Py[0] - ddy;
	Px[4] = Px[0] - ddx;
	Py[4] = Py[0] - ddy;

	if (f_horizontal) {
		Px[5] = Px[0];
		Py[5] = Py[0] + ddy;
		Px[6] = Px[0];
		Py[6] = Py[0] - ddy;
		Px[7] = Px[0];
		Py[7] = Py[0];
		Px[8] = Px[0] - ddx;
		Py[8] = Py[0];
		Px[9] = Px[0] + ddx;
		Py[9] = Py[0];
		}
	else {
		Px[5] = Px[0] - ddx;
		Py[5] = Py[0];
		Px[6] = Px[0] + ddx;
		Py[6] = Py[0];
		Px[7] = Px[0];
		Py[7] = Py[0];
		Px[8] = Px[0];
		Py[8] = Py[0] + ddy;
		Px[9] = Px[0];
		Py[9] = Py[0] - ddy;
		}

	circle(Px[1], Py[1], rad);
	circle(Px[2], Py[2], rad);
	circle(Px[3], Py[3], rad);
	circle(Px[4], Py[4], rad);
	circle(Px[5], Py[5], rad);
	circle(Px[6], Py[6], rad);
	circle(Px[7], Py[7], rad);
	circle(Px[8], Py[8], rad);
	circle(Px[9], Py[9], rad);
}


#define DD_MULTIPLIER 8.5


static int compute_dd(int dx)
{
	return (int)(((double) dx) / 32 * DD_MULTIPLIER);
}

void mp_graphics::domino_draw_assignment_East(
		int Ap, int Aq, int M,
		int i, int j, int dx, int dy, int rad)
{
	if (Ap == 1)
		domino_draw1(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 1)
		domino_draw1(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 2)
		domino_draw2(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 2)
		domino_draw2(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 3)
		domino_draw3(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 3)
		domino_draw3(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 4)
		domino_draw4(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 4)
		domino_draw4(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 5)
		domino_draw5(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 5)
		domino_draw5(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 6)
		domino_draw6(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 6)
		domino_draw6(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 7)
		domino_draw7(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 7)
		domino_draw7(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 8)
		domino_draw8(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 8)
		domino_draw8(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
	if (Ap == 9)
		domino_draw9(M, i, j, dx, dy, rad, true /* f_horizontal */);
	if (Aq == 9)
		domino_draw9(M, i, j + 1, dx, dy, rad, true /* f_horizontal */);
}

void mp_graphics::domino_draw_assignment_South(
		int Ap, int Aq, int M,
		int i, int j, int dx, int dy, int rad)
{
	if (Ap == 1)
		domino_draw1(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 1)
		domino_draw1(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 2)
		domino_draw2(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 2)
		domino_draw2(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 3)
		domino_draw3(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 3)
		domino_draw3(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 4)
		domino_draw4(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 4)
		domino_draw4(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 5)
		domino_draw5(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 5)
		domino_draw5(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 6)
		domino_draw6(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 6)
		domino_draw6(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 7)
		domino_draw7(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 7)
		domino_draw7(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 8)
		domino_draw8(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 8)
		domino_draw8(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
	if (Ap == 9)
		domino_draw9(M, i, j, dx, dy, rad, false /* f_horizontal */);
	if (Aq == 9)
		domino_draw9(M, i + 1, j, dx, dy, rad, false /* f_horizontal */);
}


void mp_graphics::domino_draw_assignment(
		int *A, int *matching, int *B,
		int M, int N,
		int dx, int dy,
		int rad, int edge,
		int f_grid, int f_gray, int f_numbers, int f_frame,
		int f_cost, int cost)
{
	int Px[100], Py[100];
	string s;
	int i, j, a, p, q;

	if (f_cost) {
		Px[0] = (N * dx) / 2;
		Py[0] = (M + 1) * dy;
		s = "${" + std::to_string(cost) + "}$";
		aligned_text(Px[0], Py[0], "", s);
	}


	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			Px[0] = j * dx;
			Py[0] = (M - i) * dy;
			Px[1] = (j + 1) * dx;
			Py[1] = (M - i) * dy;
			Px[2] = (j + 1) * dx;
			Py[2] = (M - i - 1) * dy;
			Px[3] = j * dx;
			Py[3] = (M - i - 1) * dy;

			if (f_grid) {
				polygon5(Px, Py, 0, 1, 2, 3, 0);
			}
			if (f_gray) {
				a = B[i * N + j];
				if (a < 0) {
					a = -a;
				}
				sf_interior(100 - 10 * a);
				sf_color(0);
				fill_polygon5(Px, Py, 0, 1, 2, 3, 0);
			}
			if (f_numbers) {
				//Px[4] = j * dx + (dx >> 1);
				//Py[4] = (M - i) * dy - (dy >> 1);
				s = "$" + std::to_string(B[i * N + j]) + "$";
				aligned_text(Px[2], Py[2], "br", s);
			}
			//cout << "i=" << i << "j=" << j << "p=" << p << endl;
		}
	}

	sf_interior(100 /* fill_interior */);
	sf_color(1 /* fill_color */);

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			p = i * N + j;
			//cout << "i=" << i << "j=" << j << "p=" << p << endl;
			if (matching[p] == 3) {
				//cout << "East" << endl;
				Px[0] = j * dx;
				Py[0] = (M - i) * dy;
				Px[1] = (j + 2) * dx;
				Py[1] = (M - i) * dy;
				Px[2] = (j + 2) * dx;
				Py[2] = (M - i - 1) * dy;
				Px[3] = j * dx;
				Py[3] = (M - i - 1) * dy;

				Px[4] = Px[0] + edge;
				Py[4] = Py[0] - edge;
				Px[5] = Px[1] - edge;
				Py[5] = Py[1] - edge;
				Px[6] = Px[2] - edge;
				Py[6] = Py[2] + edge;
				Px[7] = Px[3] + edge;
				Py[7] = Py[3] + edge;


				polygon5(Px, Py, 4, 5, 6, 7, 4);
				q = i * N + j + 1;

				domino_draw_assignment_East(A[p], A[q], M,
							i, j, dx, dy, rad);
			}
			else if (matching[p] == 6) {
				//cout << "South" << endl;
				Px[0] = j * dx;
				Py[0] = (M - i) * dy;
				Px[1] = (j + 1) * dx;
				Py[1] = (M - i) * dy;
				Px[2] = (j + 1) * dx;
				Py[2] = (M - i - 2) * dy;
				Px[3] = j * dx;
				Py[3] = (M - i - 2) * dy;


				Px[4] = Px[0] + edge;
				Py[4] = Py[0] - edge;
				Px[5] = Px[1] - edge;
				Py[5] = Py[1] - edge;
				Px[6] = Px[2] - edge;
				Py[6] = Py[2] + edge;
				Px[7] = Px[3] + edge;
				Py[7] = Py[3] + edge;


				polygon5(Px, Py, 4, 5, 6, 7, 4);
				q = (i + 1) * N + j;

				domino_draw_assignment_South(A[p], A[q], M,
						i, j, dx, dy, rad);
			}
		}
	}

	if (f_frame) {
		sl_udsty(50);
		Px[0] = 0 * dx;
		Py[0] = M * dy;
		Px[1] = 0 * dx;
		Py[1] = 0 * dy;
		Px[2] = N * dx;
		Py[2] = 0 * dy;
		Px[3] = N * dx;
		Py[3] = M * dy;
		polygon5(Px, Py, 0, 1, 2, 3, 0);
	}
}

}}}


