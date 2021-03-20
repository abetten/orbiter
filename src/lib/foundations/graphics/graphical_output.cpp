/*
 * graphical_output.cpp
 *
 *  Created on: Oct 11, 2020
 *      Author: betten
 */

#include "foundations.h"
#include "EasyBMP.h"


using namespace std;


namespace orbiter {
namespace foundations {



static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_on_parabola_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level);

std::vector<int> get_color(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level);
void fillBitmap(BMP &image, int i, int j, std::vector<int> color);

graphical_output::graphical_output()
{

	smooth_curve_Polish = NULL;
	parabola_a = 0.;
	parabola_b = 0.;
	parabola_c = 0.;

}

graphical_output::~graphical_output()
{

}

void graphical_output::draw_layered_graph_from_file(std::string &fname,
		layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file fname=" << fname << endl;
	}
	layered_graph *LG;
	file_io Fio;

	LG = NEW_OBJECT(layered_graph);
	if (Fio.file_size(fname) <= 0) {
		cout << "graphical_output::draw_layered_graph_from_file file " << fname << " does not exist" << endl;
		exit(1);
		}
	LG->read_file(fname, verbose_level - 1);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file Layered graph read from file" << endl;
	}

	LG->print_nb_nodes_per_level();

	int data1;


	data1 = LG->data1;

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file data1=" << data1 << endl;
	}

	if (Opt->f_y_stretch) {
		LG->place_with_y_stretch(Opt->y_stretch, verbose_level - 1);
		}
	if (Opt->f_spanning_tree) {
		// create n e w x coordinates
		LG->create_spanning_tree(TRUE /* f_place_x */, verbose_level);
		}
#if 0
	if (Opt->f_numbering_on) {
		// create depth first ranks at each node:
		LG->create_spanning_tree(FALSE /* f_place_x */, verbose_level);
		}
#endif

	if (Opt->f_x_stretch) {
		LG->scale_x_coordinates(Opt->x_stretch, verbose_level);
	}


	string fname_out;

	fname_out.assign(fname);
	chop_off_extension(fname_out);
	fname_out.append("_draw");

	//fname_out.append(".mp");

	if (Opt->f_paths_in_between) {

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file f_paths_in_between" << endl;
		}
		std::vector<std::vector<int> > All_Paths;

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file before LG->find_all_paths_between" << endl;
		}
		LG->find_all_paths_between(Opt->layer1, Opt->node1, Opt->layer2, Opt->node2,
				All_Paths,
				verbose_level - 2);
		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file after LG->find_all_paths_between" << endl;
		}

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file before LG->remove_edges" << endl;
		}
		LG->remove_edges(Opt->layer1, Opt->node1, Opt->layer2, Opt->node2,
				All_Paths,
				verbose_level - 2);
		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file after LG->remove_edges" << endl;
		}


	}

	string fname_full;

	fname_full.assign(fname_out);
	fname_full.append(".mp");

	LG->draw_with_options(fname_out, Opt, verbose_level - 10);

	int n;
	double avg;
	n = LG->nb_nodes();
	avg = LG->average_word_length();
	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file "
				"number of nodes = " << n << endl;
		cout << "graphical_output::draw_layered_graph_from_file "
				"average word length = " << avg << endl;
	}


	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file "
				"Written file " << fname_full << " of size " << Fio.file_size(fname_full) << endl;
	}

	FREE_OBJECT(LG);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file done" << endl;
	}
}

void graphical_output::do_create_points_on_quartic(double desired_distance, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::do_create_points_on_quartic" << endl;
	}

	double amin, amid, amax;
	//double epsilon = 0.001;
	int N = 200;
	int i;

	//a0 = 16. / 25.;
	//b0 = 16. / 25.;

	amin = 0;
	amid = 16. / 25.;
	amax = 100;

	int nb;

	{
		parametric_curve C1;
		parametric_curve C2;

		C1.init(2 /* nb_dimensions */,
				desired_distance,
				amin, amid,
				do_create_points_on_quartic_compute_point_function,
				this /* extra_data */,
				100. /* boundary */,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C1.Pts.size()=" << C1.Pts.size() << endl;


		C2.init(2 /* nb_dimensions */,
				desired_distance,
				amid, amax,
				do_create_points_on_quartic_compute_point_function,
				this /* extra_data */,
				100. /* boundary */,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C2.Pts.size()=" << C2.Pts.size() << endl;


		for (i = 0; i < (int) C1.Pts.size(); i++) {
			cout << C1.Pts[i].t << " : " << C1.Pts[i].coords[0] << ", " << C1.Pts[i].coords[1] << endl;
		}

		double *Pts;
		int nb_pts;

		nb_pts = 4 * (C1.Pts.size() + C2.Pts.size());
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C2.Pts[i].coords[1];
			nb++;
		}
		file_io Fio;

		string fname;

		fname.assign("points.csv");

		Fio.double_matrix_write_csv(fname, Pts, nb, 2);

		cout << "created curve 1 with " << C1.Pts.size() << " many points" << endl;
		cout << "created curve 2 with " << C2.Pts.size() << " many points" << endl;
	}
	cout << "created 4  curves with " << nb << " many points" << endl;



	if (f_v) {
		cout << "graphical_output::do_create_points_on_quartic done" << endl;
	}
}

void graphical_output::do_create_points_on_parabola(
		double desired_distance, int N,
		double a, double b, double c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::do_create_points_on_parabola" << endl;
	}

	double amin, amax;
	double boundary;
	int i;

	amin = -10;
	amax = 3.08;
	boundary = 10;

	int nb;

	graphical_output::parabola_a = a;
	graphical_output::parabola_b = b;
	graphical_output::parabola_c = c;

	{
		parametric_curve C;

		C.init(2 /* nb_dimensions */,
				desired_distance,
				amin, amax,
				do_create_points_on_parabola_compute_point_function,
				this /* extra_data */,
				boundary,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C.Pts.size()=" << C.Pts.size() << endl;




		for (i = 0; i < (int) C.Pts.size(); i++) {
			cout << C.Pts[i].t << " : " << C.Pts[i].coords[0] << ", " << C.Pts[i].coords[1] << endl;
		}

		{
		double *Pts;
		int nb_pts;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C.Pts[i].coords[1];
			nb++;
		}
		file_io Fio;
		char str[1000];
		string fname;

		snprintf(str, 1000, "parabola_N%d_%lf_%lf_%lf_points.csv", N, a, b, c);
		fname.assign(str);

		Fio.double_matrix_write_csv(fname, Pts, nb, 2);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 6];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			Pts[nb * 6 + 0] = C.Pts[i].coords[0];
			Pts[nb * 6 + 1] = C.Pts[i].coords[1];
			Pts[nb * 6 + 2] = 0.;
			Pts[nb * 6 + 3] = 0.;
			Pts[nb * 6 + 4] = 0.;
			Pts[nb * 6 + 5] = 1.;
			nb++;
		}
		file_io Fio;
		char str[1000];
		string fname;
		snprintf(str, 1000, "parabola_N%d_%lf_%lf_%lf_projection_from_center.csv", N, a, b, c);
		fname.assign(str);

		Fio.double_matrix_write_csv(fname, Pts, nb, 6);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		double x, y, H, f;
		double h = 1.;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 6];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			x = C.Pts[i].coords[0];
			y = C.Pts[i].coords[1];
			Pts[nb * 6 + 0] = x;
			Pts[nb * 6 + 1] = y;
			Pts[nb * 6 + 2] = 0.;

			H = sqrt(h * h + x * x + y * y);
			f = h / H;

			Pts[nb * 6 + 3] = x * f;
			Pts[nb * 6 + 4] = y * f;
			Pts[nb * 6 + 5] = 1. - f;
			nb++;
		}
		file_io Fio;

		char str[1000];
		string fname;
		snprintf(str, 1000, "parabola_N%d_%lf_%lf_%lf_projection_from_sphere.csv", N, a, b, c);
		fname.assign(str);


		Fio.double_matrix_write_csv(fname, Pts, nb, 6);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		double x, y, H, f;
		double h = 1.;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 3];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			x = C.Pts[i].coords[0];
			y = C.Pts[i].coords[1];

			H = sqrt(h * h + x * x + y * y);
			f = h / H;

			Pts[nb * 3 + 0] = x * f;
			Pts[nb * 3 + 1] = y * f;
			Pts[nb * 3 + 2] = 1. - f;
			nb++;
		}
		file_io Fio;

		char str[1000];
		string fname;
		snprintf(str, 1000, "parabola_N%d_%lf_%lf_%lf_points_projected.csv", N, a, b, c);
		fname.assign(str);


		Fio.double_matrix_write_csv(fname, Pts, nb, 3);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}


	}
	cout << "created curve with " << nb << " many points" << endl;



	if (f_v) {
		cout << "graphical_output::do_create_points_on_parabola done" << endl;
	}
}

void graphical_output::do_smooth_curve(std::string &curve_label,
		double desired_distance, int N,
		double t_min, double t_max, double boundary,
		function_polish_description *FP_descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_dimensions;

	if (f_v) {
		cout << "graphical_output::do_smooth_curve" << endl;
	}

	smooth_curve_Polish = NEW_OBJECT(function_polish);

	if (f_v) {
		cout << "graphical_output::do_smooth_curve before smooth_curve_Polish->init" << endl;
	}
	smooth_curve_Polish->init(FP_descr, verbose_level);
	if (f_v) {
		cout << "graphical_output::do_smooth_curve after smooth_curve_Polish->init" << endl;
	}
#if 0
	if (smooth_curve_Polish->Variables.size() != 1) {
		cout << "interface_projective::do_smooth_curve number of variables should be 1, is "
				<< smooth_curve_Polish->Variables.size() << endl;
		exit(1);
	}
#endif
	nb_dimensions = smooth_curve_Polish->Entry.size();
	if (f_v) {
		cout << "graphical_output::do_smooth_curve nb_dimensions = " << nb_dimensions << endl;
	}


	{
		parametric_curve C;

		C.init(nb_dimensions,
				desired_distance,
				t_min, t_max,
				do_create_points_smooth_curve_compute_point_function,
				this /* extra_data */,
				boundary,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C.Pts.size()=" << C.Pts.size() << endl;

		{
		double *Pts;
		int nb_pts;
		int i, j, nb;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * nb_dimensions];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			if (C.Pts[i].f_is_valid) {
				for (j = 0; j < nb_dimensions; j++) {
					Pts[nb * nb_dimensions + j] = C.Pts[i].coords[j];
				}
				nb++;
			}
		}
		file_io Fio;

		char str[1000];
		string fname;
		snprintf(str, 1000, "function_%s_N%d_points.csv", curve_label.c_str(), N);
		fname.assign(str);


		Fio.double_matrix_write_csv(fname, Pts, nb, nb_dimensions);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		int i, j, nb, n;
		double d; // euclidean distance to the previous point
		numerics Num;

		nb_pts = C.Pts.size();
		n = 1 + nb_dimensions + 1;
		Pts = new double[nb_pts * n];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			if (C.Pts[i].f_is_valid) {
				Pts[nb * n + 0] = C.Pts[i].t;
				for (j = 0; j < nb_dimensions; j++) {
					Pts[nb * n + 1 + j] = C.Pts[i].coords[j];
				}
				if (nb) {
					d = Num.distance_euclidean(Pts + (nb - 1) * n + 1, Pts + nb * n + 1, 3);
				}
				else {
					d = 0;
				}
				Pts[nb * n + 1 + 4 + 0] = d;
				nb++;
			}
		}
		file_io Fio;

		char str[1000];
		string fname;
		snprintf(str, 1000, "function_%s_N%d_points_plus.csv", curve_label.c_str(), N);
		fname.assign(str);

		Fio.double_matrix_write_csv(fname, Pts, nb, n);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

	}

	if (f_v) {
		cout << "graphical_output::do_smooth_curve done" << endl;
	}
}


static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level)
{
	double num, denom, b;
	double epsilon = 0.00001;

	num = 4. - 4. * t;
	denom = 4. - 25. * t * 0.25;
	if (ABS(denom) < epsilon) {
		return FALSE;
	}
	else {
		b = num / denom;
		if (b < 0) {
			return FALSE;
		}
		else {
			pt[0] = sqrt(t);
			pt[1] = sqrt(b);
		}
	}
	cout << "created point " << pt[0] << ", " << pt[1] << endl;
	return TRUE;
}

static int do_create_points_on_parabola_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level)
{
	graphical_output *I = (graphical_output *) extra_data;
	double a = I->parabola_a;
	double b = I->parabola_b;
	double c = I->parabola_c;

	pt[0] = t;
	pt[1] = a * t * t + b * t + c;
	//cout << "created point " << pt[0] << ", " << pt[1] << endl;
	return TRUE;
}


static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graphical_output *I = (graphical_output *) extra_data;
	int ret = FALSE;
	double epsilon = 0.0001;
	double *input; // to store the input variable and all local variables during evaluate


	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function t = " << t << endl;
	}
	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function before evaluate" << endl;
	}
	input = new double[I->smooth_curve_Polish->Variables.size()];
	input[0] = t;
	I->smooth_curve_Polish->evaluate(
			input /* variable_values */,
			output,
			verbose_level);
	delete [] input;

	if (I->smooth_curve_Polish->Entry.size() == 4) {
		if (ABS(output[3]) < epsilon) {
			ret = FALSE;
		}
		else {
			double av = 1. / output[3];
			output[0] *= av;
			output[1] *= av;
			output[2] *= av;
			output[3] *= av;
			ret = TRUE;
		}
	}
	else {
		ret = TRUE;
	}
	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function after evaluate t = " << t << endl;
	}
	return ret;
}

void graphical_output::draw_bitmap(draw_bitmap_control *C, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::draw_bitmap" << endl;
	}

	if (C->f_input_csv_file) {

		file_io Fio;

		Fio.int_matrix_read_csv(C->input_csv_file_name,
				C->M, C->m, C->n,
				verbose_level);

	}
	else {

	}

	if (f_v) {
		cout << "graphical_output::draw_bitmap drawing matrix of size " << C->m << " x " << C->n << endl;
	}

	if (C->f_partition) {
		cout << "row_part: ";
		Orbiter->Int_vec.print(cout, C->Row_parts, C->nb_row_parts);
		cout << endl;
		cout << "col_part: ";
		Orbiter->Int_vec.print(cout, C->Col_parts, C->nb_col_parts);
		cout << endl;
	}
	int i;
	int max_value;

	max_value = Orbiter->Int_vec.maximum(C->M, C->m * C->n);
	cout << "max_value=" << max_value << endl;

	//max_value += 5;
	//cout << "max_value after adjustment=" << max_value << endl;


	string fname_out;

	if (C->f_input_csv_file) {
		fname_out.assign(C->input_csv_file_name);
	}
	else {
		fname_out.assign("bitmatrix.csv");

	}
	replace_extension_with(fname_out, "_draw.bmp");

	//int bit_depth = 8;

	BMP image;

	//int bit_depth = 24;


	if (max_value > 10000) {
		cout << "draw_bitmap max_value > 10000" << endl;
		exit(1);
	}
	if (max_value == 0) {
		max_value = 1;
	}
	for (i = max_value; i >= 0; i--) {
		std::vector<int> color = get_color(C->bit_depth, max_value, i, C->f_invert_colors, 1);

		cout << i << " : " << color[0] << "," << color[1] << "," << color[2] << endl;
		}


	int width, height;
	//int *Table;
	geometry_global Gg;

	width = C->n;
	height = C->m;

	cout << "width=" << width << endl;

	if (C->f_box_width) {
		image.SetSize(width * C->box_width, height * C->box_width);
	}
	else {
		image.SetSize(width, height);
	}

	image.SetBitDepth(C->bit_depth);

	int j, d;
	int N, N100, cnt;

	N = height * width;
	N100 = N / 100 + 1;

	cout << "N100=" << N100 << endl;

	cnt = 0;
	for (i = 0; i < height; i++) {



		for (j = 0; j < width; j++, cnt++) {


			if ((cnt % N100) == 0) {
				cout << "we are at " << ((double) cnt / (double) N) * 100. << " %" << endl;
			}
			d = C->M[i * width + j];
			//std::vector<int> color = getColor(M[idx_x * width + idx_z]);
			std::vector<int> color = get_color(C->bit_depth, max_value, d, C->f_invert_colors, 0);

			// Here the pixel is set on the image.
			if (C->f_box_width) {
				int I, J, u, v;

				I = i * C->box_width;
				J = j * C->box_width;
				for (u = 0; u < C->box_width; u++) {
					for (v = 0; v < C->box_width; v++) {
						fillBitmap(image, J + v, I + u, color);
					}
				}

			}
			else {
				fillBitmap(image, j, i, color);
			}
		}
	}
	if (C->f_partition) {

		cout << "drawing the partition" << endl;
		int i0, j0;
		int h, t, I, J;
		std::vector<int> color = get_color(C->bit_depth, max_value, 1, C->f_invert_colors, 0);

		// row partition:
		i0 = 0;
		for (h = 0; h <= C->nb_row_parts; h++) {
			for (t = 0; t < C->part_width; t++) {
				if (C->f_box_width) {
					for (j = 0; j < width * C->box_width; j++) {
						I = i0 * C->box_width;
						if (h == C->nb_row_parts) {
							fillBitmap(image, j, I - 1 - t, color);
						}
						else {
							fillBitmap(image, j, I + t, color);
						}
					}
				}
			}
			if (h < C->nb_row_parts) {
				i0 += C->Row_parts[h];
			}
		}

		// col partition:
		j0 = 0;
		for (h = 0; h <= C->nb_col_parts; h++) {
			for (t = 0; t < C->part_width; t++) {
				if (C->f_box_width) {
					for (i = 0; i < height * C->box_width; i++) {
						J = j0 * C->box_width;
						if (h == C->nb_col_parts) {
							fillBitmap(image, J - 1 - t, i, color);
						}
						else {
							fillBitmap(image, J + t, i, color);
						}
					}
				}
			}
			if (h < C->nb_col_parts) {
				j0 += C->Col_parts[h];
			}
		}
	}

	cout << "before writing the image to file as " << fname_out << endl;

	  image.WriteToFile(fname_out.c_str());

	  std::cout << "Written file " << fname_out << std::endl;
	  {
		  file_io Fio;
		  cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
	  }




	if (f_v) {
		cout << "graphical_output::draw_bitmap done" << endl;
	}

}

std::vector<int> get_color(int bit_depth, int max_value, int loopCount, int f_invert_colors, int verbose_level)
{
	int f_v = (verbose_level>= 1);
	int r, g, b;
#if 0
		Black	#000000	(0,0,0)
		 	White	#FFFFFF	(255,255,255)
		 	Red	#FF0000	(255,0,0)
		 	Lime	#00FF00	(0,255,0)
		 	Blue	#0000FF	(0,0,255)
		 	Yellow	#FFFF00	(255,255,0)
		 	Cyan / Aqua	#00FFFF	(0,255,255)
		 	Magenta / Fuchsia	#FF00FF	(255,0,255)
		 	Silver	#C0C0C0	(192,192,192)
		 	Gray	#808080	(128,128,128)
		 	Maroon	#800000	(128,0,0)
		 	Olive	#808000	(128,128,0)
		 	Green	#008000	(0,128,0)
		 	Purple	#800080	(128,0,128)
		 	Teal	#008080	(0,128,128)
		 	Navy	#000080	(0,0,128)
#endif
	int table[] = {
			255,255,255, // white
			0,0,0, // black
			255,0,0,
			0,255,0,
			0,0,255,
			255,255,0,
			0,255,255,
			255,0,255,
			192,192,192,
			128,128,128,
			128,0,0,
			128,128,0,
			0,128,0,
			128,0,128,
			0,128,128,
			0,0,128
	};


	if (loopCount < 16 && bit_depth == 8) {
		r = table[loopCount * 3 + 0];
		g = table[loopCount * 3 + 1];
		b = table[loopCount * 3 + 2];
	}
	else {
		double a1, a2, x, y, z;
		int max_color;

		max_color = (1 << bit_depth) - 1;

		if (loopCount > max_value) {
			cout << "loopCount > max_value" << endl;
			cout << "loopCount=" << loopCount << endl;
			cout << "max_value=" << max_value << endl;
			exit(1);
		}

		if (loopCount < 16) {
			r = table[loopCount * 3 + 0];
			g = table[loopCount * 3 + 1];
			b = table[loopCount * 3 + 2];
			return { r, g, b};
		}
		loopCount -= 16;

		a1 = (double) loopCount / (double) max_value;


		if (f_invert_colors) {
			a2 = 1. - a1;
		}
		else {
			a2 = a1;
		}
		x = a2;
		y = a2 * a2;
		z = y * a2;
		r = x * max_color;
		g = y * max_color;
		b = z * max_color;
		if (f_v) {
			cout << loopCount << " : " << max_value << " : "
					<< a1 << " : " << a2 << " : " << x << "," << y << "," << z << " : " << r << "," << g << "," << b << endl;
		}
	}
	return { r, g, b};
}

void fillBitmap(BMP &image, int i, int j, std::vector<int> color)
{
	// The pixel is set using its image
	// location and stacks 3 variables (RGB) into the vector word.
	image(i, j)->Red = color[0];
	image(i, j)->Green = color[1];
	image(i, j)->Blue = color[2];
};


}}

