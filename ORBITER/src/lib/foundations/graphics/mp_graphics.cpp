// mp_graphics.C
//
// Anton Betten
// March 6, 2003

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


mp_graphics::mp_graphics()
{
	default_values();
}

mp_graphics::mp_graphics(const char *file_name,
		int xmin, int ymin, int xmax, int ymax,
		int f_embedded, int f_sideways)
{
	
	default_values();
	init(file_name, xmin, ymin, xmax, ymax, f_embedded, f_sideways);
}

mp_graphics::~mp_graphics()
{
	// cout << "mp_graphics::~mp_graphics()" < endl;
	exit(cout, FALSE);
}

void mp_graphics::default_values()
{
	f_file_open = FALSE;
	f_min_max_set = FALSE;
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
	
	f_embedded = FALSE;
	f_sideways = FALSE;

	tikz_global_scale = .45;
	tikz_global_line_width = 1.5;
}

void mp_graphics::init(const char *file_name,
	int xmin, int ymin, int xmax, int ymax,
	int f_embedded, int f_sideways)
{
	mp_graphics::f_embedded = f_embedded;
	mp_graphics::f_sideways = f_sideways;
	
	strcpy(fname_mp, file_name);
	
	get_fname_base(file_name, fname_base);
	sprintf(fname_log, "%s.commands", fname_base);
	sprintf(fname_tikz, "%s.tex", fname_base);

	fp_mp.open(fname_mp);
	fp_log.open(fname_log);
	fp_tikz.open(fname_tikz);
	f_file_open = TRUE;
	
	user[0] = xmin;
	user[1] = ymin;
	user[2] = xmax;
	user[3] = ymax;
	
	// the identity transformation:
	
	dev[0] = xmin;
	dev[1] = ymin;
	dev[2] = xmax;
	dev[3] = ymax;
}

void mp_graphics::exit(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_file_open) {
		fp_mp.close();
		fp_log.close();
		fp_tikz.close();
		if (f_v) {
			ost << "mp_graphics::exit "
					"written file " << fname_mp
					<< " of size " << file_size(fname_mp) << endl;
			ost << "written file " << fname_log
					<< " of size " << file_size(fname_log) << endl;
			ost << "written file " << fname_tikz
					<< " of size " << file_size(fname_tikz) << endl;
			}
		f_file_open = FALSE;
		}
}

void mp_graphics::setup(const char *fname_base, 
	int in_xmin, int in_ymin, int in_xmax, int in_ymax, 
	int xmax, int ymax, int f_embedded, int f_sideways, 
	double scale, double line_width)
{
	//int x_min = 0, x_max = 1000;
	//int y_min = 0, y_max = 1000;
	int factor_1000 = 1000;
	char fname_full[1000];
	
	sprintf(fname_full, "%s.mp", fname_base);
	init(fname_full, in_xmin, in_ymin,
			in_xmax, in_ymax, f_embedded, f_sideways);
#if 0
	out_xmin() = -(xmax >> 1);
	out_ymin() = -(ymax >> 1);
	out_xmax() = xmax >> 1;
	out_ymax() = ymax >> 1;
#else
	out_xmin() = 0;
	out_ymin() = 0;
	out_xmax() = xmax;
	out_ymax() = ymax;
#endif
	//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	tikz_global_scale = scale;
	tikz_global_line_width = line_width;
	header();
	begin_figure(factor_1000);
}

void mp_graphics::set_parameters(double scale, double line_width)
{
	tikz_global_scale = scale;
	tikz_global_line_width = line_width;
}

void mp_graphics::set_scale(double scale)
{
	tikz_global_scale = scale;
}


void mp_graphics::frame(double move_out)
{
	int Px[30];
	int Py[30];
	int x, y, dx, dy, ox, oy, Ox, Oy;
	int xmin, ymin, xmax, ymax;
	
	xmin = out_xmin();
	xmax = out_xmax();
	ymin = out_ymin();
	ymax = out_ymax();
	
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

void mp_graphics::finish(ostream &ost, int verbose_level)
{
	end_figure();
	footer();
	exit(cout, verbose_level - 1);
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
	transform_llur(user, dev, x, y);
}

void mp_graphics::dev2user(int &x, int &y)
{
	transform_llur(dev, user, x, y);
}

void mp_graphics::user2dev_dist_x(int &x)
{
	transform_dist_x(user, dev, x);
}

void mp_graphics::user2dev_dist_y(int &y)
{
	transform_dist_y(user, dev, y);
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
	numerics Num;

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
	double x_min, double x_max,
	double y_min, double y_max,
	double x_stretch, double y_stretch,
	int f_x_axis_at_y_min, int f_y_axis_at_x_min, 
	int x_mod, int y_mod, int x_tick_mod, int y_tick_mod, 
	double x_labels_offset, double y_labels_offset, 
	double x_tick_half_width, double y_tick_half_width, 
	int f_v_lines, int subdivide_v, int f_h_lines, int subdivide_h)
{
	double *Dx, *Dy;
	int *Px, *Py;
	double dx = x_stretch;//ONE_MILLION * 50 * x_stretch;
	double dy = y_stretch;//ONE_MILLION * 50 * y_stretch;
	int N = 1000;
	int n;
	int i, j, h;

	

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
		char str[1000];
		sprintf(str, "%d", i);
		//cout << "str='" << str << "'" << endl;
		aligned_text_array(Px, Py, j, "", str);
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
		char str[1000];
		sprintf(str, "%d", i);
		//cout << "str='" << str << "'" << endl;
		aligned_text_array(Px, Py, j, "", str);
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

void mp_graphics::plot_curve(int N, int *f_DNE,
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
		if (f_DNE[i] == FALSE) {
			Px[j] = Dx[i] * dx;
			Py[j] = Dy[i] * dy;
			j++;
			//cout << "i=" << i << " Px[i]=" << Px[i]
			// << " Py[i]=" << Py[i] << endl;

#if 0
			if (i > 2 && f_DNE[i - 1] == FALSE && f_DNE[i - 2] == FALSE) {
				Dx1 = Px[i - 1] - Px[i - 2];
				Dy1 = Py[i - 1] - Py[i - 2];
				L1 = Dx1 * Dx1 + Dy1 * Dy1;
				Dx2 = Px[i] - Px[i - 1];
				Dy2 = Py[i] - Py[i - 1];
				L2 = Dx2 * Dx2 + Dy2 * Dy2;
				if (L2 > 10 * L1) {
					f_DNE[i] = TRUE;
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


void mp_graphics::nice_circle(int x, int y, int rad)
{
	//fp_log << "NiceCircle " << x << " " << y << " " << rad << endl;

	sf_interior(100);
	//sf_color(0); // 1 = black, 0 = white
	circle(x, y, rad);
	sf_interior(0);
	//sf_color(1); // 1 = black, 0 = white
	circle(x, y, rad);
}

void mp_graphics::grid_polygon2(grid_frame *F, 
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

void mp_graphics::grid_polygon4(grid_frame *F, 
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

void mp_graphics::grid_polygon5(grid_frame *F, 
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

void mp_graphics::polygon(int *Px, int *Py, int n)
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

void mp_graphics::polygon2(int *Px, int *Py, int i1, int i2)
{
	int Idx[2];
	Idx[0] = i1;
	Idx[1] = i2;
	polygon_idx(Px, Py, Idx, 2);
}

void mp_graphics::polygon3(int *Px, int *Py,
		int i1, int i2, int i3)
{
	int Idx[3];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	polygon_idx(Px, Py, Idx, 3);
}

void mp_graphics::polygon4(int *Px, int *Py,
		int i1, int i2, int i3, int i4)
{
	int Idx[4];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	polygon_idx(Px, Py, Idx, 4);
}

void mp_graphics::polygon5(int *Px, int *Py,
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

void mp_graphics::polygon6(int *Px, int *Py,
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

void mp_graphics::polygon7(int *Px, int *Py,
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

void mp_graphics::polygon8(int *Px, int *Py,
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

void mp_graphics::polygon9(int *Px, int *Py,
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

void mp_graphics::polygon10(int *Px, int *Py,
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

void mp_graphics::polygon11(int *Px, int *Py,
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

void mp_graphics::polygon_idx(int *Px, int *Py, int *Idx, int n)
{
	polygon_idx_log(Px, Py, Idx, n);
	polygon_idx2(Px, Py, Idx, n, FALSE);
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
	bezier_idx2(Px, Py, Idx, n, FALSE);
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
	fill_idx(Px, Py, Idx, 4, "--", TRUE);
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
	fill_idx(Px, Py, Idx, 5, "--", TRUE);
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
	fill_idx(Px, Py, Idx, 3, "--", FALSE);
}

void mp_graphics::fill_polygon4(int *Px, int *Py,
		int i1, int i2, int i3, int i4)
{
	int Idx[10];
	Idx[0] = i1;
	Idx[1] = i2;
	Idx[2] = i3;
	Idx[3] = i4;
	fill_idx(Px, Py, Idx, 4, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 5, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 6, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 7, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 8, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 9, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 10, "--", FALSE);
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
	fill_idx(Px, Py, Idx, 11, "--", FALSE);
}

void mp_graphics::polygon2_arrow_halfway(int *Px, int *Py,
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

void mp_graphics::polygon2_arrow_halfway_and_label(int *Px, int *Py,
	int i1, int i2,
	const char *alignment, const char *txt)
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
	
	aligned_text(X[2], Y[2], alignment, txt);
}

void mp_graphics::grid_aligned_text(grid_frame *F,
		int x, int y, const char *alignment, const char *p)
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
	aligned_text(Px[0], Py[0], alignment, p);
	FREE_int(Px);
	FREE_int(Py);
}

void mp_graphics::aligned_text(int x, int y,
		const char *alignment, const char *p)
{
	//fp_log << "AlignedText " << x << " " << y << " "
	// << alignment << " \"" << p << "\"" << endl;
	aligned_text_with_offset(x, y, 0, 0, alignment, p);
}

void mp_graphics::aligned_text_array(int *Px, int *Py, int idx,
		const char *alignment, const char *p)
{
	aligned_text(Px[idx], Py[idx], alignment, p);
}

void mp_graphics::aligned_text_with_offset(int x, int y,
		int xoffset, int yoffset,
	const char *alignment, const char *p)
{
	int h_align = 1, v_align = 1;
	int l, i;
	char c;
	
	//fp_log << "AlignedText " << x << " " << y << " " << xoffset
	// << " " << yoffset << " " << alignment << " \"" << p << "\"" << endl;
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
	//cout << "text=" << p << endl;
	st_alignment(h_align, v_align);
	text(x + xoffset, y + yoffset, p);
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
	f_min_max_set = TRUE;
}


// #############################################################################
// output commands:
// #############################################################################

void mp_graphics::header()
{
	char str[1024];
	
	f_min_max_set = FALSE;
	//system("rm a");

	system("date >a");
	{
	ifstream f1("a");
	f1.getline(str, sizeof(str));
	}

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

void mp_graphics::comment(const char *p)
{
	comment_log(p);
	comment_mp(p);
	comment_tikz(p);
}




void mp_graphics::text(int x, int y, const char *p)
{
	int x1, y1;
	
	//fp_log << "Text " << x << " " << y << " \"" << p << "\"" << endl;

	x1 = x;
	y1 = y;
	coords_min_max(x1, y1);
	user2dev(x1, y1);
	
	text_log(x1, y1, p);
	text_mp(x1, y1, p);
	text_tikz(x1, y1, p);
}

void mp_graphics::circle(int x, int y, int rad)
{
	//fp_log << "Circle " << x << " " << y << " " << rad << endl;

	coords_min_max(x, y);
	user2dev(x, y);
	user2dev_dist_x(rad);
	
	if (rad <= 0) rad = 1;
	
	circle_log(x, y, rad);
	circle_mp(x, y, rad);
	circle_tikz(x, y, rad);
}

void mp_graphics::circle_text(int x, int y, int rad, const char *p)
{
	//fp_log << "CircleText " << x << " " << y << " \"" << p << "\"" << endl;

#if 0
	coords_min_max(x, y);
	user2dev(x, y);
	user2dev_dist_x(rad);
#endif

	nice_circle(x, y, rad);
	text(x, y, p);
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

void mp_graphics::header_log(char *str_date)
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

void mp_graphics::comment_log(const char *p)
{
	fp_log << "Comment " << p << endl;
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

void mp_graphics::text_log(int x1, int y1, const char *p)
{
	fp_log << "Text " << x1 << ", " << y1 << ", \"" << p << "\"" << endl;
}

void mp_graphics::circle_log(int x1, int y1, int rad)
{
	fp_log << "Circle " << x1 << ", " << y1 << ", " << rad << endl;
}


// #############################################################################
// device specific output commands: metapost
// #############################################################################


void mp_graphics::header_mp(char *str_date)
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

	sprintf(str, "%lf", d);
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

void mp_graphics::comment_mp(const char *p)
{
	fp_mp << "% " << p << endl;
}

void mp_graphics::text_mp(int x1, int y1, const char *p)
{
	char align[64];
	int lab;

	get_alignment_mp(align);
	if (txt_boxed) {
		lab = get_label(x1, y1);
		fp_mp << "boxit.l" << lab << "(btex " << p << " etex);" << endl;
		fp_mp << lab << ".c=";
		output_xy_metapost(x1, y1);
		fp_mp << endl;
		if (txt_overwrite) {
			fp_mp << "unfill bpath " << lab << ";" << endl;
			}
		fp_mp << "drawboxed(" << lab << ");" << endl;
		
		}
	else {
		fp_mp << "label" << align << "(btex " << p << " etex";
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

void mp_graphics::output_circle_text_mp(int x, int y,
		int idx, const char *text)
{
	fp_mp << "circleit.l" << idx << "(btex " << text << " etex);" << endl;
	fp_mp << "l" << idx << ".c = ";
	output_xy_metapost(x, y);
	fp_mp << endl;
	fp_mp << "unfill bpath l" << idx << ";" << endl;
	fp_mp << "drawboxed(l" << idx << ");" << endl;
}

void mp_graphics::polygon_idx_mp(int *Px, int *Py,
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

void mp_graphics::bezier_idx_mp(int *Px, int *Py,
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

void mp_graphics::fill_idx_mp(int *Px, int *Py,
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

void mp_graphics::header_tikz(char *str_date)
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
	
	if (f_embedded) {
		fp_tikz << "\\documentclass[12pt]{article}" << endl;
		fp_tikz << "\\usepackage{amsmath, amssymb, amsthm}" << endl;
		fp_tikz << "\\usepackage{tikz} " << endl;
		if (f_sideways) {
			fp_tikz << "\\usepackage{rotating} " << endl;
			}
		fp_tikz << "%\\usepackage{anysize}" << endl;
		fp_tikz << "\\begin{document}" << endl;
		fp_tikz << "%\\bibliographystyle{plain}" << endl;
		fp_tikz << "\\pagestyle{empty}" << endl;
		}

	if (f_sideways) {
		fp_tikz << "\\begin{sideways}" << endl;
		}
	fp_tikz << "\\begin{tikzpicture}[scale=" << tikz_global_scale
		<< ",line width = " << tikz_global_line_width << "pt]" << endl;
	//fp_tikz << "\\begin{tikzpicture}[scale=.05,line width = 0.5pt]" << endl;
}

void mp_graphics::footer_tikz()
{
	fp_tikz << "\\end{tikzpicture}" << endl;
	if (f_sideways) {
		fp_tikz << "\\end{sideways}" << endl;
		}
	if (f_embedded) {
		fp_tikz << "\\end{document}" << endl;
		}
}

void mp_graphics::comment_tikz(const char *p)
{
	fp_tikz << "% " << p << endl;
}


void mp_graphics::text_tikz(int x1, int y1, const char *p)
{
	if (txt_overwrite) {
		fp_tikz << "\\draw ";
		output_xy_tikz(x1, y1);
		fp_tikz << " node[fill=white] {";
		fp_tikz << p;
		fp_tikz << "};" << endl;
		}
	else {
		fp_tikz << "\\draw ";
		output_xy_tikz(x1, y1);
		fp_tikz << " node{";
		fp_tikz << p;
		fp_tikz << "};" << endl;
		}
}


void mp_graphics::circle_tikz(int x, int y, int rad)
{
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


void mp_graphics::output_circle_text_tikz(int x, int y,
		int idx, int rad, const char *text)
{
	//char str[1000];

	//sprintf(str, "%d", idx);

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


void mp_graphics::polygon_idx_tikz(int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int f_need_comma = FALSE;
	int x, y, i;

	fp_tikz << "\\draw [";
	if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "->";
		f_need_comma = TRUE;
		}
	else if (line_end_style == 0 && line_beg_style == 1) {
		fp_tikz << "<-";
		f_need_comma = TRUE;
		}
	else if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "<->";
		f_need_comma = TRUE;
		}
	if (line_thickness != 100) {
		if (f_need_comma) {
			fp_tikz << ",";
			}
		fp_tikz << "line width=" << ((double)line_thickness * 0.01) << "mm";
		f_need_comma = TRUE;
		}
	if (line_color != 1) {
		if (f_need_comma) {
			fp_tikz << ",";
			}
		fp_tikz << "color=";
		color_tikz(fp_tikz, line_color);
		f_need_comma = TRUE;
		}

	fp_tikz << "] ";
	for (i = 0; i < n; i++) {
		x = Px[Idx[i]];
		y = Py[Idx[i]];
		coords_min_max(x, y);
		user2dev(x, y);

		if (i) {
			fp_tikz << " -- ";
			}
		output_xy_tikz(x, y);
		}
	fp_tikz << ";" << endl;
}

void mp_graphics::bezier_idx_tikz(int *Px, int *Py,
		int *Idx, int n, int f_cycle)
{
	int f_need_comma = FALSE;
	int x, y, i;
	
	fp_tikz << "\\draw [";
	if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "->";
		f_need_comma = TRUE;
		}
	else if (line_end_style == 0 && line_beg_style == 1) {
		fp_tikz << "<-";
		f_need_comma = TRUE;
		}
	else if (line_end_style == 1 && line_beg_style == 0) {
		fp_tikz << "<->";
		f_need_comma = TRUE;
		}
	if (line_thickness != 100) {
		if (f_need_comma) {
			fp_tikz << ",";
			}
		fp_tikz << "line width=" << ((double)line_thickness * 0.01) << "mm";
		f_need_comma = TRUE;
		}
	if (line_color != 1) {
		if (f_need_comma) {
			fp_tikz << ",";
			}
		fp_tikz << "color=";
		color_tikz(fp_tikz, line_color);
		f_need_comma = TRUE;
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
		} else if (i == n - 1) {
			fp_tikz << " .. ";
		} else if (i) {
			fp_tikz << " and ";
		}
#endif
		output_xy_tikz(x, y);
		}
	fp_tikz << ";" << endl;
}

void mp_graphics::color_tikz(ofstream &fp, int color)
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
	else if (color == 16)
		fp << "darkgray";
	else {
		cout << "mp_graphics::color_tikz color = " << color 
			<< ", we don't have a color for this" << endl;
		}
}
// red, green, blue, cyan, magenta, yellow, black, gray,
// darkgray, lightgray, brown, lime, olive, orange, pink,
// purple, teal, violet and white.

void mp_graphics::fill_idx_tikz(ofstream &fp, int *Px, int *Py,
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
	f_need_comma = TRUE;
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
void mp_graphics::output_xy_tikz(int x, int y)
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



void mp_graphics::polygon3D(int *Px, int *Py,
		int dim, int x0, int y0, int z0, int x1, int y1, int z1)
{
	int idx0, idx1;
	idx0 = x0 * 9 + y0 * 3 + z0;
	idx1 = x1 * 9 + y1 * 3 + z1;
	polygon2(Px, Py, idx0, idx1);
}

void mp_graphics::integer_4pts(int *Px, int *Py,
		int p1, int p2, int p3, int p4,
		const char *align, int a)
{
	char str[100];

	sprintf(str, "%d", a);
	text_4pts(Px, Py, p1, p2, p3, p4, align, str);
}

void mp_graphics::text_4pts(int *Px, int *Py,
		int p1, int p2, int p3, int p4,
		const char *align, const char *str)
{
	int x = Px[p1] + Px[p2] + Px[p3] + Px[p4];
	int y = Py[p1] + Py[p2] + Py[p3] + Py[p4];
	x >>= 2;
	y >>= 2;
	aligned_text(x, y, align, str);
}


void mp_graphics::draw_graph(int x, int y,
		int dx, int dy, int nb_V, int *Edges, int nb_E)
{
	double *X, *Y;
	double h = dy * .7;
	double w = dx * .7;
	int i, e, u, v;
	double phi = M_PI * 2. / nb_V;
	int Px[2];
	int Py[2];
	int rad = (int)(dx * .05);

	//cout << "draw_graph nb_V=" << nb_V << endl;

	sl_thickness(30);

	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		X[i] = cos(i * phi) * w;
		Y[i] = sin(i * phi) * h;
		}
	for (i = 0; i < nb_E; i++) {
		e = Edges[i];
		k2ij(e, u, v, nb_V);
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
	delete X;
	delete Y;
}

void mp_graphics::draw_graph_with_distinguished_edge(
	int x, int y,
	int dx, int dy, int nb_V, int *Edges, int nb_E,
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
		k2ij(e, u, v, nb_V);
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
	delete X;
	delete Y;
}

void mp_graphics::draw_graph_on_multiple_circles(int x, int y,
		int dx, int dy, int nb_V, int *Edges, int nb_E, int nb_circles)
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
		k2ij(e, u, v, nb_V);
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
	delete X;
	delete Y;
}

void mp_graphics::draw_graph_on_2D_grid(
		int x, int y, int dx, int dy, int rad, int nb_V,
		int *Edges, int nb_E, int *coords_2D, int *Base,
		int f_point_labels, int point_label_offset, int f_directed)
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
		char str[1000];

		for (i = 0; i < nb_V; i++) {
			sprintf(str, "%d", i+ point_label_offset);
			Px[0] = x + (int) X[i];
			Py[0] = y + (int) Y[i];
			aligned_text(Px[0], Py[0], "", str);
			}
		}
	if (f_directed) {
		sl_ends(0, 0);
		}
	delete X;
	delete Y;
}

void mp_graphics::draw_tournament(int x, int y,
		int dx, int dy, int nb_V, int *Edges, int nb_E,
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
	int rad = (int)(dx * .05);

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
		k2ij(a2, u, v, nb_V);



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
	delete X;
	delete Y;
}

void mp_graphics::draw_bitmatrix2(int f_dots,
	int f_partition, int nb_row_parts, int *row_part_first,
	int nb_col_parts, int *col_part_first,
	int f_row_grid, int f_col_grid,
	int f_bitmatrix, uchar *D, int *M,
	int m, int n, int xmax, int ymax,
	int f_has_labels, int *labels)
{
	char str[1000];
	grid_frame F;
	int i, j, ii, jj, ij, a, cnt, mn, mtn, mtn1;
	int indent = 0;

	mn = MAXIMUM(m, n);
	F.f_matrix_notation = TRUE;
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

	sprintf(str, "box outline");
	comment(str);

	grid_polygon2(&F, 0, 0, 10 * m, 0);
	grid_polygon2(&F, 10 * m, 0, 10 * m, 10 * n);
	grid_polygon2(&F, 10 * m, 10 * n, 0, 10 * n);
	grid_polygon2(&F, 0, 10 * n, 0, 0);

	sf_interior(100);
	sf_color(1); // black


	sl_thickness(20); // 100 is standard

	if (f_partition) {
		sprintf(str, "row partition");
		comment(str);
		for (i = 0; i < nb_row_parts + 1; i++) {
			sprintf(str, "part_%d", i);
			comment(str);
			ii = row_part_first[i];
			grid_polygon2(&F, ii * 10, 0 * 10, ii * 10, (n + 0) * 10);
			//G.grid_polygon2(&F, ii * 10, -1 * 10, ii * 10, (n + 1) * 10);
			}
		sprintf(str, "column partition");
		comment(str);
		for (j = 0; j < nb_col_parts + 1; j++) {
			sprintf(str, "part_%d", j);
			comment(str);
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
			sprintf(str, "row_%d", i);
			if (f_has_labels) {
				sprintf(str + strlen(str), " label %d", labels[i]);
				}
			comment(str);
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
			sprintf(str, "col_%d", j);
			if (f_has_labels) {
				sprintf(str + strlen(str), " label %d", labels[j]);
				}
			comment(str);
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
			sprintf(str, "%d", labels[i]);
			grid_aligned_text(&F, i * 10 + 5, -1 * 10, "", str);
			}
		for (j = 0; j < n; j++) {
			sprintf(str, "%d", labels[m + j] - m);
			grid_aligned_text(&F, -1 * 10, j * 10 + 5, "", str);
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
				a = bitvector_s_i(D, i * n + j);
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



			sprintf(str, "%d_%d", i, j);
			comment(str);

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



void mp_graphics::draw_density2(int no,
	int *outline_value, int *outline_number, int outline_sz,
	int min_value, int max_value, int offset_x, int f_switch_x,
	int f_title, const char *title,
	const char *label_x,
	int f_circle, int circle_at, int circle_rad,
	int f_mu, int f_sigma, int nb_standard_deviations,
	int f_v_grid, int v_grid, int f_h_grid, int h_grid)
{
	int i;
	int Px[1000], Py[1000];
	//int phi = 360 / 12;
	//int rad1 = 400;
	char str[1000];
	int y_in, x, y, k;

	int min_x, max_x, min_y, max_y;
	int sum, a;
	int mini_x, i0;
	double average;
	double sigma; // standard deviation
	double sum1, f;

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


	get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, FALSE);
	for (i = 0; i < outline_sz; i++) {
		get_coord(Px, Py, 2, outline_number[i], outline_value[i],
			min_x, min_y, max_x, max_y, FALSE);
		Px[1] = Px[0];
		Py[1] = Py[2];
		polygon3(Px, Py, 0, 1, 2);
		Px[0] = Px[2];
		Py[0] = Py[2];
		}
	get_coord(Px, Py, 2, max_x, max_value,
		min_x, min_y, max_x, max_y, FALSE);
	polygon2(Px, Py, 0, 2);
	get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, FALSE);
	polygon5(Px, Py, 0, 1, 2, 3, 0);


	if (f_switch_x) {
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		aligned_text(Px[1], Py[1], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		aligned_text(Px[1], Py[1], "t", str);
		}
	sprintf(str, "{\\bf {\\large %d}}", min_y);
	aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %d}}", max_y);
	aligned_text(Px[3], Py[3], "r", str);



	Px[0] = 5 * 100;
	Py[0] = 0;
	sprintf(str, "{\\bf {\\large %s}}", label_x);
	aligned_text(Px[0], Py[0], "t", str);

	Px[0] = 5 * 100;
	Py[0] = -50;
	sprintf(str, "{\\bf {\\large $");
	if (f_mu) {
		sprintf(str + strlen(str), "\\overline{x}=%.1lf", average);
		}
	if (f_sigma) {
		if (f_mu) {
			sprintf(str + strlen(str), "\\, ");
			}
		sprintf(str + strlen(str), "\\sigma=%.1lf", sigma);
		}
	if (f_mu || f_sigma) {
		sprintf(str + strlen(str), "$}}");
		aligned_text(Px[0], Py[0], "t", str);
		}


	if (f_mu) {
		y_in = (int) average;
		y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		aligned_text(Px[2], Py[2], "r", "$\\overline{x}$");
		}


	if (f_circle) {
		y_to_pt_on_curve(circle_at, x, y,
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, y, min_x, min_y, max_x, max_y, FALSE);
		circle(Px[0], Py[0], circle_rad);
		}


	for (k = 1; k < nb_standard_deviations; k++) {
		y_in = (int) (average + k * sigma);
		y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "$\\overline{x}+%d \\sigma$", k);
			}
		else {
			sprintf(str, "$\\overline{x}+\\sigma$");
			}
		aligned_text(Px[2], Py[2], "r", str);

		y_in = (int) (average - k * sigma);
		y_to_pt_on_curve(y_in, x, y,
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "{\\bf {\\large $\\overline{x}-%d \\sigma$}}", k);
			}
		else {
			sprintf(str, "{\\bf {\\large $\\overline{x}-\\sigma$}}");
			}
		aligned_text(Px[2], Py[2], "r", str);
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
		sprintf(str, "{\\bf {\\large %s}}", title);
		aligned_text(Px[0], Py[0], "b", str);
		}

}

void mp_graphics::draw_density2_multiple_curves(int no,
	int **outline_value, int **outline_number, int *outline_sz, int nb_curves,
	int min_x, int max_x, int min_y, int max_y,
	int offset_x, int f_switch_x,
	int f_title, const char *title,
	const char *label_x,
	int f_v_grid, int v_grid, int f_h_grid, int h_grid,
	int f_v_logarithmic, double log_base)
{
	int i;
	int Px[1000], Py[1000];
	//int phi = 360 / 12;
	//int rad1 = 400;
	char str[1000];
	int curve;

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
			get_coord_log(Px, Py, 0,
					min_x, min_y, min_x, min_y, max_x, max_y,
					log_base, f_switch_x);
			}
		else {
			get_coord(Px, Py, 0,
					min_x, min_y, min_x, min_y, max_x, max_y, f_switch_x);
			}
		for (i = 0; i < outline_sz[curve]; i++) {
			if (f_v_logarithmic) {
				get_coord_log(Px, Py, 2,
					outline_number[curve][i], outline_value[curve][i],
					min_x, min_y, max_x, max_y, log_base, f_switch_x);
				}
			else {
				get_coord(Px, Py, 2,
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
			get_coord_log(Px, Py, 2, max_x, max_y,
				min_x, min_y, max_x, max_y, log_base, f_switch_x);
			}
		else {
			get_coord(Px, Py, 2, max_x, max_y,
				min_x, min_y, max_x, max_y, f_switch_x);
			}
		polygon2(Px, Py, 0, 2);
		}


	if (f_v_logarithmic) {
		get_coord_log(Px, Py, 0,
				min_x, min_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 1,
				max_x, min_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 2,
				max_x, max_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 3,
				min_x, max_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		}
	else {
		get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, FALSE);
		}
	polygon5(Px, Py, 0, 1, 2, 3, 0);



	if (f_switch_x) {
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		aligned_text(Px[1], Py[1], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		aligned_text(Px[0], Py[0], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		aligned_text(Px[1], Py[1], "t", str);
		}

	sprintf(str, "{\\bf {\\large %d}}", min_y);
	aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %d}}", max_y);
	aligned_text(Px[3], Py[3], "r", str);



	Px[0] = 5 * 100;
	Py[0] = 0;
	sprintf(str, "{\\bf {\\large %s}}", label_x);
	aligned_text(Px[0], Py[0], "t", str);





	int line_dashing = 50;
	int line_thickness = 15;
	sl_udsty(line_dashing);
	sl_thickness(line_thickness);

	if (f_v_grid) {
		if (FALSE) {
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
				get_coord_log(Px, Py, 2, min_x, (int)a,
						min_x, min_y, max_x, max_y, log_base,
						FALSE /* f_switch_x */);
				Px[0] = Px[2];
				Py[0] = Py[2];
				Px[1] = 1000;
				Py[1] = Py[2];
				polygon2(Px, Py, 0, 1);
				sprintf(str, "{%d}", (int)a);
				aligned_text(Px[0], Py[0], "r", str);
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
		sprintf(str, "{\\bf {\\large %s}}", title);
		aligned_text(Px[0], Py[0], "b", str);
		}

}

void mp_graphics::projective_plane_draw_grid2(int q,
	int *Table, int nb, int f_with_points, int rad,
	int f_point_labels, char **Point_labels, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double a, b;
	int x1, x2, x3;

	//int rad = 17000;
	int i, h;
	double x_stretch = 0.0010;
	double y_stretch = 0.0010;
	//double x_stretch = 0.01;
	//double y_stretch = 0.01;

	double *Dx, *Dy;
	int *Px, *Py;
	int dx = ONE_MILLION * 50 * x_stretch;
	int dy = ONE_MILLION * 50 * y_stretch;
	int N = 1000;


	if (f_v) {
		cout << "projective_plane_draw_grid2" << endl;
		}


	Px = NEW_int(N);
	Py = NEW_int(N);
	Dx = new double[N];
	Dy = new double[N];






	if (f_v) {
		cout << "projective_plane_draw_grid2 "
				"before G.draw_axes_and_grid" << endl;
		}


	draw_axes_and_grid(
		0., (double)(q - 1), 0., (double)(q - 1), x_stretch, y_stretch,
		TRUE /* f_x_axis_at_y_min */, TRUE /* f_y_axis_at_x_min */,
		1 /* x_mod */, 1 /* y_mod */, 1, 1,
		-1. /* x_labels_offset */, -1. /* y_labels_offset */,
		0.5 /* x_tick_half_width */, 0.5 /* y_tick_half_width */,
		TRUE /* f_v_lines */, 1 /* subdivide_v */,
		TRUE /* f_h_lines */, 1 /* subdivide_h */);


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
	text(Px[0], Py[0], "$x$");
	text(Px[1], Py[1], "$y$");

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

	if (f_with_points) {

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
			nice_circle(Px[0], Py[0], rad);
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




}
}


