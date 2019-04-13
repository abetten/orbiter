// draw.C
//
// Anton Betten
//
// moved here from mp.C:  February 18, 2014
//



#include "foundations.h"
#include <math.h>


using namespace std;


namespace orbiter {
namespace foundations {



#if 0
void orbits_point(int *Px, int *Py, int idx, int sx, int sy, int i, int j)
{
	Px[idx] = 0 + sx * j;
	Py[idx] = 0 - sy * i;
}
#endif


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



void on_circle_int(int *Px, int *Py,
		int idx, int angle_in_degree, int rad)
{
	numerics Num;
	
	Px[idx] = (int)(Num.cos_grad(angle_in_degree) * (double) rad);
	Py[idx] = (int)(Num.sin_grad(angle_in_degree) * (double) rad);
}

int C3D(int i, int j, int k)
{
	return i * 9 + j * 3 + k;
}

int C2D(int i, int j)
{
	return i * 5 + j;
}

void draw_bitmatrix(const char *fname_base, int f_dots, 
	int f_partition, int nb_row_parts, int *row_part_first,
	int nb_col_parts, int *col_part_first,
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
	
	G.draw_bitmatrix2(f_dots,
		f_partition, nb_row_parts, row_part_first,
		nb_col_parts, col_part_first,
		f_row_grid, f_col_grid, 
		f_bitmatrix, D, M, 
		m, n, 
		xmax_in, ymax_in, 
		f_has_labels, labels);

	G.finish(cout, TRUE);
	}
	cout << "draw_it written file " << fname
			<< " of size " << file_size(fname) << endl;
}


}
}



