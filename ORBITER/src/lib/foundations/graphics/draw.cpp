// draw.C
//
// Anton Betten
//
// moved here from mp.C:  February 18, 2014
//



#include "foundations.h"
#include <math.h>

namespace orbiter {
namespace foundations {



void orbits_point(int *Px, int *Py, int idx, int sx, int sy, int i, int j)
{
	Px[idx] = 0 + sx * j;
	Py[idx] = 0 - sy * i;
}

void orbits_draw_dot(mp_graphics &MP, int sx, int sy, int i, int j)
{
	int Px[1000], Py[1000];
	
	orbits_point(Px, Py, 0, sx, sy, i, j);
	MP.sf_interior(100);
	MP.sf_color(1);
	MP.circle(Px[0], Py[0], 4);
}

void orbits_draw_circle(mp_graphics &MP, int sx, int sy, int i0, int j0, int i1, int j1)
{
	int Px[1000], Py[1000];
	int dx, dy, r;
	
	orbits_point(Px, Py, 0, sx, sy, i0, j0);
	orbits_point(Px, Py, 1, sx, sy, i1, j1);
	dx = Px[1] - Px[0];
	dy = Py[1] - Py[0];
	r = (int)sqrt((double)(dx * dx + dy * dy));
	// MP.sf_interior(100);
	// MP.sf_color(1);
	MP.circle(Px[0], Py[0], r);
}

void orbits_draw_line2(mp_graphics &MP, int sx, int sy, int i0, int j0, int i1, int j1)
{
	int Px[1000], Py[1000];
	
	orbits_point(Px, Py, 0, sx, sy, i0, j0);
	orbits_point(Px, Py, 1, sx, sy, i1, j1);
	MP.polygon2(Px, Py, 0, 1);
}

void orbits_draw_line3(mp_graphics &MP, int sx, int sy, int i0, int j0, int i1, int j1, int i2, int j2)
{
	int Px[1000], Py[1000];
	
	orbits_point(Px, Py, 0, sx, sy, i0, j0);
	orbits_point(Px, Py, 1, sx, sy, i1, j1);
	orbits_point(Px, Py, 2, sx, sy, i2, j2);
	MP.polygon3(Px, Py, 0, 1, 2);
}

void orbits_draw_line4(mp_graphics &MP, int sx, int sy, int i0, int j0, int i1, int j1, int i2, int j2, int i3, int j3)
{
	int Px[1000], Py[1000];
	
	orbits_point(Px, Py, 0, sx, sy, i0, j0);
	orbits_point(Px, Py, 1, sx, sy, i1, j1);
	orbits_point(Px, Py, 2, sx, sy, i2, j2);
	orbits_point(Px, Py, 3, sx, sy, i3, j3);
	MP.polygon4(Px, Py, 0, 1, 2, 3);
}

void orbits_draw_box(mp_graphics &MP, int sx, int sy, int i0, int j0, int i1, int j1)
{
	orbits_draw_line2(MP, sx, sy, i0, j0, i1, j0);
	orbits_draw_line2(MP, sx, sy, i1, j0, i1, j1);
	orbits_draw_line2(MP, sx, sy, i1, j1, i0, j1);
	orbits_draw_line2(MP, sx, sy, i0, j1, i0, j0);
}

#undef DEBUG_TRANSFORM_LLUR

void transform_llur(int *in, int *out, int &x, int &y)
{
	int dx, dy; //, rad;
	double a, b; //, f;

#ifdef DEBUG_TRANSFORM_LLUR
	cout << "transform_llur: ";
	cout << "In=" << in[0] << "," << in[1] << "," << in[2] << "," << in[3] << endl;
	cout << "Out=" << out[0] << "," << out[1] << "," << out[2] << "," << out[3] << endl;
#endif
	dx = x - in[0];
	dy = y - in[1];
	//rad = MIN(out[2] - out[0], out[3] - out[1]);
	a = (double) dx / (double)(in[2] - in[0]);
	b = (double) dy / (double)(in[3] - in[1]);
	//a = a / 300000.;
	//b = b / 300000.;
#ifdef DEBUG_TRANSFORM_LLUR
	cout << "transform_llur: (x,y)=(" << x << "," << y << ") in[2] - in[0]=" << in[2] - in[0] << " in[3] - in[1]=" << in[3] - in[1] << " (a,b)=(" << a << "," << b << ") -> ";
#endif
	
	// projection on a disc of radius 1:
	//f = 300000 / sqrt(1. + a * a + b * b);
#ifdef DEBUG_TRANSFORM_LLUR
	cout << "f=" << f << endl;
#endif
	//a = f * a;
	//b = f * b;

	//dx = (int)(a * f);
	//dy = (int)(b * f);
	dx = (int)(a * (double)(out[2] - out[0]));
	dy = (int)(b * (double)(out[3] - out[1]));
	x = dx + out[0];
	y = dy + out[1];
#ifdef DEBUG_TRANSFORM_LLUR
	cout << x << "," << y << " a=" << a << " b=" << b << endl;
#endif
}

void transform_dist(int *in, int *out, int &x, int &y)
{
	int dx, dy;
	double a, b;

	a = (double) x / (double)(in[2] - in[0]);
	b = (double) y / (double)(in[3] - in[1]);
	dx = (int)(a * (double) (out[2] - out[0]));
	dy = (int)(b * (double) (out[3] - out[1]));
	x = dx;
	y = dy;
}

void transform_dist_x(int *in, int *out, int &x)
{
	int dx;
	double a;

	a = (double) x / (double)(in[2] - in[0]);
	dx = (int)(a * (double) (out[2] - out[0]));
	x = dx;
}

void transform_dist_y(int *in, int *out, int &y)
{
	int dy;
	double b;

	b = (double) y / (double)(in[3] - in[1]);
	dy = (int)(b * (double) (out[3] - out[1]));
	y = dy;
}

void transform_llur_double(double *in, double *out, double &x, double &y)
{
	double dx, dy;
	double a, b;

#ifdef DEBUG_TRANSFORM_LLUR
	cout << "transform_llur_double: " << x << "," << y << " -> ";
#endif
	dx = x - in[0];
	dy = y - in[1];
	a = dx / (in[2] - in[0]);
	b =  dy / (in[3] - in[1]);
	dx = a * (out[2] - out[0]);
	dy = b * (out[3] - out[1]);
	x = dx + out[0];
	y = dy + out[1];
#ifdef DEBUG_TRANSFORM_LLUR
	cout << x << "," << y << endl;
#endif
}

void draw(char *fname)
{
	int n = 100, dx = 10, dy = 10;
	int xmin = -n * dx, xmax = n * dx;
	int ymin = -n * dy, ymax = 0;
	int factor_1000 = 1000;
	//int Px[1000], Py[1000];
	char fname_full[1000];
	int f_embedded = TRUE;
	int f_sideways = FALSE;
	
	sprintf(fname_full, "%s.mp", fname);
	mp_graphics G(fname_full, xmin, ymin, xmax, ymax, f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = 1000;
	G.out_ymax() = 1000;
	
	G.header();
	G.begin_figure(factor_1000);

	//G.circle(Px[1], Py[1], 500);

#if 0
	for (k = 0; k < short_base_len; k++) {
		i = short_base[k];
		l = orbit_len[i];
		if (l == 1) {
			continue;
			}
		bi = the_base[i];
		cout << "orbit of " << bi << " : " << endl;
		Px[0] = i * dy;
		Py[0] = -(i * dx);
		l = 0;
		for (j = i; j < n; j++) {
			if (prev[j] == i /* path_exists(i, j)*/) {
				Px[1] = j * dy;
				Py[1] = -((i + 1) * dx);
				G.polygon2(Px, Py, 0, 1);
				
				G.sf_interior(100);
				G.sf_color(1);
				G.circle(Px[1], Py[1], 5);

				sprintf(str, "$%d$", j);
				G.aligned_text(Px, Py, 1, "t", str);
				}
			}
		}
#endif

	G.end_figure();
	G.footer();
}

void on_circle_int(int *Px, int *Py, int idx, int angle_in_degree, int rad)
{
	
	Px[idx] = (int)(cos_grad(angle_in_degree) * (double) rad);
	Py[idx] = (int)(sin_grad(angle_in_degree) * (double) rad);
}

void on_circle_double(double *Px, double *Py, int idx, double angle_in_degree, double rad)
{
	
	Px[idx] = cos_grad(angle_in_degree) * rad;
	Py[idx] = sin_grad(angle_in_degree) * rad;
}

void polygon3D(mp_graphics &G, int *Px, int *Py, int dim, int x0, int y0, int z0, int x1, int y1, int z1)
{
	int idx0, idx1;
	idx0 = x0 * 9 + y0 * 3 + z0;
	idx1 = x1 * 9 + y1 * 3 + z1;
	G.polygon2(Px, Py, idx0, idx1);
}

void integer_4pts(mp_graphics &G, int *Px, int *Py, int p1, int p2, int p3, int p4, 
	const char *align, int a)
{
	char str[100];
	
	sprintf(str, "%d", a);
	text_4pts(G, Px, Py, p1, p2, p3, p4, align, str);
}

void text_4pts(mp_graphics &G, int *Px, int *Py, int p1, int p2, int p3, int p4, 
	const char *align, const char *str)
{
	int x = Px[p1] + Px[p2] + Px[p3] + Px[p4];
	int y = Py[p1] + Py[p2] + Py[p3] + Py[p4];
	x >>= 2;
	y >>= 2;
	G.aligned_text(x, y, align, str);
}

void affine_pt1(int *Px, int *Py, int p0, int p1, int p2, double f1, int p3)
{
	int x = Px[p0] + (int)(f1 * (double)(Px[p2] - Px[p1]));
	int y = Py[p0] + (int)(f1 * (double)(Py[p2] - Py[p1]));
	Px[p3] = x;
	Py[p3] = y;
}

void affine_pt2(int *Px, int *Py, int p0, int p1, int p1b, 
	double f1, int p2, int p2b, double f2, int p3)
{
	int x = Px[p0] + (int)(f1 * (double)(Px[p1b] - Px[p1])) + (int)(f2 * (double)(Px[p2b] - Px[p2]));
	int y = Py[p0] + (int)(f1 * (double)(Py[p1b] - Py[p1])) + (int)(f2 * (double)(Py[p2b] - Py[p2]));
	Px[p3] = x;
	Py[p3] = y;
}

int C3D(int i, int j, int k)
{
	return i * 9 + j * 3 + k;
}

int C2D(int i, int j)
{
	return i * 5 + j;
}

#include <math.h>

double cos_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return cos(x);
}

double sin_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return sin(x);
}

double tan_grad(double phi)
{
	double x;

	x = (phi * M_PI) / 180.;
	return tan(x);
}

double atan_grad(double x)
{
	double y, phi;

	y = atan(x);
	phi = (y * 180.) / M_PI;
	return phi;
}

void adjust_coordinates_double(double *Px, double *Py, int *Qx, int *Qy, 
	int N, double xmin, double ymin, double xmax, double ymax, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	double in[4], out[4];
	double x_min, x_max;
	double y_min, y_max;
	int i;
	double x, y;
	
	x_min = x_max = Px[0];
	y_min = y_max = Py[0];
	
	for (i = 1; i < N; i++) {
		x_min = MINIMUM(x_min, Px[i]);
		x_max = MAXIMUM(x_max, Px[i]);
		y_min = MINIMUM(y_min, Py[i]);
		y_max = MAXIMUM(y_max, Py[i]);
		}
	if (f_v) {
		cout << "adjust_coordinates_double: x_min=" << x_min 
		<< "x_max=" << x_max
		<< "y_min=" << y_min 
		<< "y_max=" << y_max << endl;
		cout << "adjust_coordinates_double: ";
		cout 
			<< "xmin=" << xmin
			<< " xmax=" << xmax 
			<< " ymin=" << ymin 
			<< " ymax=" << ymax 
			<< endl; 
		}

	in[0] = x_min;
	in[1] = y_min;
	in[2] = x_max;
	in[3] = y_max;
	
	out[0] = xmin;
	out[1] = ymin;
	out[2] = xmax;
	out[3] = ymax;

	for (i = 0; i < N; i++) {
		x = Px[i];
		y = Py[i];
		if (f_vv) {
			cout << "In:" << x << "," << y << " : ";
			}
		transform_llur_double(in, out, x, y);
		Qx[i] = (int)x;
		Qy[i] = (int)y;
		if (f_vv) {
			cout << " Out: " << Qx[i] << "," << Qy[i] << endl;
			}
		}
}

void Intersection_of_lines(double *X, double *Y, double *a, double *b, double *c, int l1, int l2, int pt)
{
	intersection_of_lines(a[l1], b[l1], c[l1], a[l2], b[l2], c[l2], X[pt], Y[pt]);
}

void intersection_of_lines(double a1, double b1, double c1, double a2, double b2, double c2, 
	double &x, double &y)
{
	double d;

	d = a1 * b2 - a2 * b1;
	d = 1. / d;
	x = d * (b2 * -c1 + -b1 * -c2);
	y = d * (-a2 * -c1 + a1 * -c2);
}

void Line_through_points(double *X, double *Y, double *a, double *b, double *c, 
	int pt1, int pt2, int line_idx)
{
	line_through_points(X[pt1], Y[pt1], X[pt2], Y[pt2], a[line_idx], b[line_idx], c[line_idx]);
}

void line_through_points(double pt1_x, double pt1_y, 
	double pt2_x, double pt2_y, double &a, double &b, double &c)
{
	double s, off;
	const double EPSILON = 0.01;

	if (ABS(pt2_x - pt1_x) > EPSILON) {
		s = (pt2_y - pt1_y) / (pt2_x - pt1_x);
		off = pt1_y - s * pt1_x;
		a = s;
		b = -1;
		c = off;
		}
	else {
		s = (pt2_x - pt1_x) / (pt2_y - pt1_y);
		off = pt1_x - s * pt1_y;
		a = 1;
		b = -s;
		c = -off;
		}
}

void intersect_circle_line_through(double rad, double x0, double y0, 
	double pt1_x, double pt1_y, 
	double pt2_x, double pt2_y, 
	double &x1, double &y1, double &x2, double &y2)
{
	double a, b, c;

	line_through_points(pt1_x, pt1_y, pt2_x, pt2_y, a, b, c);
	//cout << "intersect_circle_line_through pt1_x=" << pt1_x << " pt1_y=" << pt1_y << " pt2_x=" << pt2_x << " pt2_y=" << pt2_y << endl;
	//cout << "intersect_circle_line_through a=" << a << " b=" << b << " c=" << c << endl;
	intersect_circle_line(rad, x0, y0, a, b, c, x1, y1, x2, y2);
	//cout << "intersect_circle_line_through x1=" << x1 << " y1=" << y1 
	//	<< " x2=" << x2 << " y2=" << y2 << endl << endl;
}


void intersect_circle_line(double rad, double x0, double y0, double a, double b, double c, 
	double &x1, double &y1, double &x2, double &y2)
{
	double A, B, C;
	double a2 = a * a;
	double b2 = b * b;
	double c2 = c * c;
	double x02 = x0 * x0;
	double y02 = y0 * y0;
	double r2 = rad * rad;
	double p, q, u, disc, d;
	
	cout << "a=" << a << " b=" << b << " c=" << c << endl;
	A = 1 + a2 / b2;
	B = 2 * a * c / b2 - 2 * x0 + 2 * a * y0 / b;
	C = c2 / b2 + 2 * c * y0 / b + x02 + y02 - r2;
	cout << "A=" << A << " B=" << B << " C=" << C << endl;
	p = B / A;
	q = C / A;
	u = -p / 2;
	disc =  u * u - q;
	d = sqrt(disc);
	x1 = u + d;
	x2 = u - d;
	y1 = (-a * x1 - c) / b;
	y2 = (-a * x2 - c) / b;
}

void affine_combination(double *X, double *Y, int pt0, int pt1, int pt2, double alpha, int new_pt)
//X[new_pt] = X[pt0] + alpha * (X[pt2] - X[pt1]);
//Y[new_pt] = Y[pt0] + alpha * (Y[pt2] - Y[pt1]);
{
	X[new_pt] = X[pt0] + alpha * (X[pt2] - X[pt1]);
	Y[new_pt] = Y[pt0] + alpha * (Y[pt2] - Y[pt1]);
}

void draw_graph(mp_graphics *G, int x, int y, int dx, int dy, int nb_V, int *Edges, int nb_E)
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

	G->sl_thickness(30);

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
		G->polygon2(Px, Py, 0, 1);
		}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		G->nice_circle(Px[0], Py[0], rad);
		}
	delete X;
	delete Y;
}

void draw_graph_with_distinguished_edge(mp_graphics *G, int x, int y, 
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
		cout << "draw_graph_with_distinguished_edge nb_V=" << nb_V << endl;
		}
	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		X[i] = cos(i * phi) * w;
		Y[i] = sin(i * phi) * h;
		}

	G->sl_thickness(30);

	for (i = 0; i < nb_E; i++) {
		e = Edges[i];
		k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		if (distinguished_edge == i) {
			G->sl_thickness(80);
			}
		G->polygon2(Px, Py, 0, 1);
		if (distinguished_edge == i) {
			G->sl_thickness(30);
			}
		}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		G->nice_circle(Px[0], Py[0], rad);
		}
	delete X;
	delete Y;
}

void draw_graph_on_multiple_circles(mp_graphics *G, int x, int y, int dx, int dy, int nb_V, int *Edges, int nb_E, int nb_circles)
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
	
	cout << "draw_graph_on_multiple_circles nb_V=" << nb_V << endl;
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
		G->polygon2(Px, Py, 0, 1);
		}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		G->nice_circle(Px[0], Py[0], rad);
		}
	delete X;
	delete Y;
}

void draw_graph_on_2D_grid(mp_graphics *G, int x, int y, int dx, int dy, int rad, int nb_V, 
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
	
	cout << "draw_graph_on_2D_grid nb_V=" << nb_V << endl;
	X = new double [nb_V];
	Y = new double [nb_V];
	for (i = 0; i < nb_V; i++) {
		u = coords_2D[2 * i + 0];
		v = coords_2D[2 * i + 1];
		X[i] = u * Base[0] + v * Base[2];
		Y[i] = u * Base[1] + v * Base[3];
		}

	if (f_directed) {
		G->sl_ends(0, 1);
		}
	for (i = 0; i < nb_E; i++) {
		u = Edges[2 * i + 0];
		v = Edges[2 * i + 1];
		//k2ij(e, u, v, nb_V);
		Px[0] = x + (int) X[u];
		Py[0] = y + (int) Y[u];
		Px[1] = x + (int) X[v];
		Py[1] = y + (int) Y[v];
		G->polygon2(Px, Py, 0, 1);
		}
	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		G->nice_circle(Px[0], Py[0], rad);
		}

	if (f_point_labels) {
		char str[1000];
		
		for (i = 0; i < nb_V; i++) {
			sprintf(str, "%d", i+ point_label_offset);
			Px[0] = x + (int) X[i];
			Py[0] = y + (int) Y[i];
			G->aligned_text(Px[0], Py[0], "", str);
			}
		}
	if (f_directed) {
		G->sl_ends(0, 0);
		}
	delete X;
	delete Y;
}

void draw_tournament(mp_graphics *G, int x, int y, int dx, int dy, int nb_V, int *Edges, int nb_E, int verbose_level)
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
		cout << "draw_tournament nb_V=" << nb_V << endl;
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
			G->sl_ends(0, 1);
			G->polygon2(Px, Py, 1, 2);
			}
		else {
			Px[2] = (Px[0] + 3 * Px[1]) >> 2;
			Py[2] = (Py[0] + 3 * Py[1]) >> 2;
			G->sl_ends(0, 1);
			G->polygon2(Px, Py, 0, 2);
			}
		}

	// now draw vertices:

	for (i = 0; i < nb_V; i++) {
		Px[0] = x + (int) X[i];
		Py[0] = y + (int) Y[i];
		G->nice_circle(Px[0], Py[0], rad);
		}
	delete X;
	delete Y;
}

void draw_bitmatrix(const char *fname_base, int f_dots, 
	int f_partition, int nb_row_parts, int *row_part_first, int nb_col_parts, int *col_part_first, 
	int f_row_grid, int f_col_grid, 
	int f_bitmatrix, uchar *D, int *M, 
	int m, int n, int xmax_in, int ymax_in, int xmax_out, int ymax_out, 
	double scale, double line_width, 
	int f_has_labels, int *labels)
{
	mp_graphics G;
	char fname_base2[1000];
	char fname[1000];
	int f_embedded = TRUE;
	int f_sideways = FALSE;
	//double scale = .3;
	//double line_width = 1.0;
	
	sprintf(fname_base2, "%s", fname_base);
	sprintf(fname, "%s.mp", fname_base2);
	{
	G.setup(fname_base2, 0, 0, 
		xmax_in /* ONE_MILLION */, ymax_in /* ONE_MILLION */, 
		xmax_out, ymax_out, 
		f_embedded, f_sideways, 
		scale, line_width);

	//G.frame(0.05);
	
	draw_bitmatrix2(G, f_dots, 
		f_partition, nb_row_parts, row_part_first, nb_col_parts, col_part_first, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, D, M, 
		m, n, 
		xmax_in, ymax_in, 
		f_has_labels, labels);

	G.finish(cout, TRUE);
	}
	cout << "draw_it written file " << fname << " of size " << file_size(fname) << endl;
}

void draw_bitmatrix2(mp_graphics &G, int f_dots, 
	int f_partition, int nb_row_parts, int *row_part_first, int nb_col_parts, int *col_part_first, 
	int f_row_grid, int f_col_grid, 
	int f_bitmatrix, uchar *D, int *M, 
	int m, int n, int xmax, int ymax, 
	int f_has_labels, int *labels)
{
	char comment[1000];
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

	sprintf(comment, "box outline");
	G.comment(comment);

	G.grid_polygon2(&F, 0, 0, 10 * m, 0);
	G.grid_polygon2(&F, 10 * m, 0, 10 * m, 10 * n);
	G.grid_polygon2(&F, 10 * m, 10 * n, 0, 10 * n);
	G.grid_polygon2(&F, 0, 10 * n, 0, 0);

	G.sf_interior(100);
	G.sf_color(1); // black
	

	G.sl_thickness(20); // 100 is standard

	if (f_partition) {
		sprintf(comment, "row partition");
		G.comment(comment);
		for (i = 0; i < nb_row_parts + 1; i++) {
			sprintf(comment, "part_%d", i);
			G.comment(comment);
			ii = row_part_first[i];
			G.grid_polygon2(&F, ii * 10, 0 * 10, ii * 10, (n + 0) * 10);
			//G.grid_polygon2(&F, ii * 10, -1 * 10, ii * 10, (n + 1) * 10);
			}
		sprintf(comment, "column partition");
		G.comment(comment);
		for (j = 0; j < nb_col_parts + 1; j++) {
			sprintf(comment, "part_%d", j);
			G.comment(comment);
			jj = col_part_first[j];
			G.grid_polygon2(&F, 0 * 10, jj * 10, (m + 0) * 10, jj * 10);
			//G.grid_polygon2(&F, -1 * 10, jj * 10, (m + 1) * 10, jj * 10);
			}
		}



	
	G.sl_thickness(10); // 100 is standard

	G.sf_interior(0);
	G.sf_color(1);

	if (f_row_grid) {
		for (i = 0; i < m; i++) {
			sprintf(comment, "row_%d", i);
			if (f_has_labels) {
				sprintf(comment + strlen(comment), " label %d", labels[i]);
				}
			G.comment(comment);
			j = 0;
			G.grid_fill_polygon5(&F, 
				10 * i + indent, 10 * j + indent, 
				10 * (i + 1) - indent, 10 * j + indent, 
				10 * (i + 1) - indent, 10 * n - indent, 
				10 * i + indent, 10 * n - indent, 
				10 * i + indent, 10 * j + indent);
			}
		}

	if (f_col_grid) {
		for (j = 0; j < n; j++) {
			sprintf(comment, "col_%d", j);
			if (f_has_labels) {
				sprintf(comment + strlen(comment), " label %d", labels[j]);
				}
			G.comment(comment);
			i = 0;
			G.grid_fill_polygon5(&F, 
				10 * i + indent, 10 * j + indent, 
				10 * m - indent, 10 * j + indent, 
				10 * m - indent, 10 * (j + 1) - indent, 
				10 * i + indent, 10 * (j + 1) - indent, 
				10 * i + indent, 10 * j + indent);
			}
		}

	if (f_has_labels) {
		for (i = 0; i < m; i++) {
			sprintf(comment, "%d", labels[i]);
			G.grid_aligned_text(&F, i * 10 + 5, -1 * 10, "", comment);
			}
		for (j = 0; j < n; j++) {
			sprintf(comment, "%d", labels[m + j] - m);
			G.grid_aligned_text(&F, -1 * 10, j * 10 + 5, "", comment);
			}
		}


	G.sl_thickness(10); // 100 is standard

	G.sf_interior(100);

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
			//G.grid_fill_polygon4(&F, i, j, i + 1, j, i + 1, j + 1, i, j + 1);



			sprintf(comment, "%d_%d", i, j);
			G.comment(comment);

			if (f_dots) {
				G.grid_polygon2(&F, 10 * i, 10 * j, 10 * i, 10 * j);
				}
			else {
				G.sf_interior(100);
				G.sf_color(1);

#if 0
				G.grid_fill_polygon4(&F, 
					10 * i + 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * (j + 1) - 1, 
					10 * i + 1, 10 * (j + 1) - 1);
#else
				G.grid_fill_polygon5(&F, 
					10 * i + indent, 10 * j + indent, 
					10 * (i + 1) - indent, 10 * j + indent, 
					10 * (i + 1) - indent, 10 * (j + 1) - indent, 
					10 * i + indent, 10 * (j + 1) - indent, 
					10 * i + indent, 10 * j + indent);
#endif
				//G.grid_polygon2(&F, i, j, i + 1, j);
				//G.grid_polygon2(&F, i + 1, j, i + 1, j + 1);
				//G.grid_polygon2(&F, i + 1, j + 1, i, j + 1);
				//G.grid_polygon2(&F, i, j + 1, i, j);
				}
			}
		}
	cout << "draw_it2 # of non-zero coefficients = " << cnt << endl;
}

}
}



