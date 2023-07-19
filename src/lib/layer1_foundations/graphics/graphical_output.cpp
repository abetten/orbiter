/*
 * graphical_output.cpp
 *
 *  Created on: Oct 11, 2020
 *      Author: betten
 */

#include <l1_interfaces/EasyBMP.h>
#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {



static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_on_parabola_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level);

static void interface_povray_draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	std::ostream &fp,
	int verbose_level);


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
	graph_theory::layered_graph *LG;
	orbiter_kernel_system::file_io Fio;

	LG = NEW_OBJECT(graph_theory::layered_graph);
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
		// create updated x coordinates
		LG->create_spanning_tree(true /* f_place_x */, verbose_level);
		}
#if 0
	if (Opt->f_numbering_on) {
		// create depth first ranks at each node:
		LG->create_spanning_tree(false /* f_place_x */, verbose_level);
		}
#endif

	if (Opt->f_x_stretch) {
		LG->scale_x_coordinates(Opt->x_stretch, verbose_level);
	}


	string fname_out;
	data_structures::string_tools ST;

	fname_out = fname;
	ST.chop_off_extension(fname_out);
	fname_out += "_draw";


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

	fname_full = fname_out + ".mp";

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

void graphical_output::do_domino_portrait(int D, int s,
		std::string &photo_label,
		layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"D=" << D << " s=" << s
				<< " photo_label=" << photo_label << endl;
	}


	string fname;
	//seed_random_generator_with_system_time();

	srand(777);

	if (f_v) {
		cout << "graphical_output::do_domino_portrait" << endl;
	}


	combinatorics::domino_assignment *Domino_Assignment;

	Domino_Assignment = NEW_OBJECT(combinatorics::domino_assignment);

	Domino_Assignment->initialize_assignment(D, s, verbose_level - 1);

	Domino_Assignment->read_photo(photo_label, verbose_level - 1);


	if (f_v) {
		cout << "graphical_output::do_domino_portrait before stage 0" << endl;
	}
	Domino_Assignment->stage0(verbose_level - 1);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait after stage 0" << endl;
	}


	if (f_v) {
		cout << "graphical_output::do_domino_portrait before stage 1" << endl;
	}
	Domino_Assignment->stage1(verbose_level - 1);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait after stage 1" << endl;
	}


	if (f_v) {
		cout << "graphical_output::do_domino_portrait before stage 2" << endl;
	}
	Domino_Assignment->stage2(verbose_level - 1);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait after stage 2" << endl;
	}



#if 0
	sorting Sorting;

	Sorting.Heapsort_general((void *)Assi_table_sort[hd], tot_dom,
			compare_assignment,
			swap_assignment,
			Assi_table[hd]);
	//Sorting.quicksort_array(tot_dom,
	//		(void **) Assi_table_sort[hd],
	//		compare_assignment,
	//		Assi_table[hd] /* void *data */);

	//quicksort_array(tot_dom,
	//		Assi_table_sort[hd],
	//		compare_assignment,
	//		Assi_table[hd] /* void *data */);
#endif

	//cout << "solution " << hd << " after sort" << endl;
	//print_assignment(hd);


	fname = photo_label + "_solution_" + std::to_string(0);



	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"calling draw_domino_matrix" << endl;
	}

	int cost;

	cost = Domino_Assignment->cost_function();
	Domino_Assignment->draw_domino_matrix(fname,
			Domino_Assignment->tot_dom,
			true /* f_has_cost */, cost,
			Opt,
			verbose_level - 1);

	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"after draw_domino_matrix" << endl;
	}


	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"calling prepare_latex" << endl;
	}
	Domino_Assignment->prepare_latex(photo_label, verbose_level);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"after prepare_latex" << endl;
	}

	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"nb_changes=" << Domino_Assignment->nb_changes << endl;
	}
	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"before classify_changes_by_type" << endl;
	}

	Domino_Assignment->classify_changes_by_type(verbose_level);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"after classify_changes_by_type" << endl;
	}

	int *Cost;
	int len;
	string fname_cost;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools String;

	fname_cost = photo_label;
	String.chop_off_extension(fname_cost);
	fname_cost += "_cost.csv";

	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"before get_cost_function" << endl;
	}
	Domino_Assignment->get_cost_function(Cost, len, verbose_level);
	if (f_v) {
		cout << "graphical_output::do_domino_portrait "
				"after get_cost_function" << endl;
		cout << "graphical_output::do_domino_portrait "
				"after get_cost_function len=" << len << endl;
	}

	string label;

	label = "Cost";
	Fio.Csv_file_support->int_vec_write_csv(
			Cost, len, fname_cost, label);

	if (f_v) {
		cout << "Written file " << fname_cost
				<< " of size " << Fio.file_size(fname_cost) << endl;
	}


	if (f_v) {
		cout << "graphical_output::do_domino_portrait done" << endl;
	}
}

void graphical_output::do_create_points_on_quartic(
		double desired_distance, int verbose_level)
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
		orbiter_kernel_system::file_io Fio;

		string fname;

		fname.assign("points.csv");

		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, 2);

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
		orbiter_kernel_system::file_io Fio;
		string fname;

		fname = "parabola_N" + std::to_string(N) + "_"
				+ std::to_string(a) + "_"
				+ std::to_string(b) + "_"
				+ std::to_string(c) + "_points.csv";

		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, 2);

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
		orbiter_kernel_system::file_io Fio;

		string fname;
		fname = "parabola_N" + std::to_string(N) + "_"
				+ std::to_string(a) + "_"
				+ std::to_string(b) + "_"
				+ std::to_string(c) + "_projection_from_center.csv";

		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, 6);

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
		orbiter_kernel_system::file_io Fio;

		string fname;
		fname = "parabola_N" + std::to_string(N) + "_"
				+ std::to_string(a) + "_"
				+ std::to_string(b) + "_"
				+ std::to_string(c) + "_projection_from_sphere.csv";


		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, 6);

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
		orbiter_kernel_system::file_io Fio;

		string fname;
		fname = "parabola_N" + std::to_string(N) + "_"
				+ std::to_string(a) + "_"
				+ std::to_string(b) + "_"
				+ std::to_string(c) + "_points_projected.csv";


		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, 3);

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
		polish::function_polish_description *FP_descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_dimensions;

	if (f_v) {
		cout << "graphical_output::do_smooth_curve" << endl;
	}

	smooth_curve_Polish = NEW_OBJECT(polish::function_polish);

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
		orbiter_kernel_system::file_io Fio;

		string fname;
		fname = "function_" + curve_label + "_N" + std::to_string(N) + "_points.csv";


		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, nb_dimensions);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		int i, j, nb, n;
		double d; // euclidean distance to the previous point
		orbiter_kernel_system::numerics Num;

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
		orbiter_kernel_system::file_io Fio;

		string fname;
		fname = "function_" + curve_label + "_N" + std::to_string(N) + "_points_plus.csv";

		Fio.Csv_file_support->double_matrix_write_csv(
				fname, Pts, nb, n);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

	}

	if (f_v) {
		cout << "graphical_output::do_smooth_curve done" << endl;
	}
}


//

static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level)
{
	double num, denom, b;
	double epsilon = 0.00001;

	num = 4. - 4. * t;
	denom = 4. - 25. * t * 0.25;
	if (ABS(denom) < epsilon) {
		return false;
	}
	else {
		b = num / denom;
		if (b < 0) {
			return false;
		}
		else {
			pt[0] = sqrt(t);
			pt[1] = sqrt(b);
		}
	}
	cout << "created point " << pt[0] << ", " << pt[1] << endl;
	return true;
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
	return true;
}


static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graphical_output *I = (graphical_output *) extra_data;
	int ret = false;
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
			ret = false;
		}
		else {
			double av = 1. / output[3];
			output[0] *= av;
			output[1] *= av;
			output[2] *= av;
			output[3] *= av;
			ret = true;
		}
	}
	else {
		ret = true;
	}
	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function after evaluate t = " << t << endl;
	}
	return ret;
}


void graphical_output::draw_projective_curve(draw_projective_curve_description *Descr,
		layered_graph_draw_options *Opt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::draw_projective_curve" << endl;
	}

	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();
	//int xmax = Opt->xin; //1500;
	//int ymax = Opt->yin; //1500;
	int i;


	if (Descr->f_animate) {

		for (i = 0; i <= Descr->animate_nb_of_steps; i++) {


			if (f_v) {
				cout << "animate step " << i << " / " << Descr->animate_nb_of_steps << ":" << endl;
				}
			mp_graphics G;


			string fname;
			fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);

			G.init(fname, Opt, verbose_level);
			//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
			//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width, verbose_level - 1);
			//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax);


			//G.frame(0.05);


			draw_projective(G, Descr->number, i, Descr->animate_nb_of_steps, false, 0, 0, false, 0, false, 0);
			G.finish(cout, true);

		}
	}
	else if (Descr->f_animate_with_transition) {
		int frame;

		frame = 0;

		if (Descr->f_title_page) {

			for (i = 0; i < 4; i++, frame++) {
				mp_graphics G;

				string fname;
				fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);


				G.init(fname, Opt, verbose_level);
				//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
				//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width,
				//		verbose_level - 1);

				draw_projective(G, Descr->number, 0, Descr->animate_nb_of_steps, true, i, Descr->animate_transition_nb_of_steps, true, i, false, 0);


				G.finish(cout, true);


			}
		}

		for (i = 0; i <= Descr->animate_transition_nb_of_steps; i++, frame++) {

			if (f_v) {
				cout << "frame " << frame << " transition in step " << i << " / " << Descr->animate_transition_nb_of_steps << ":" << endl;
			}
			mp_graphics G;

			string fname;
			fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);


			G.init(fname, Opt, verbose_level);
			//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
			//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width,
			//		verbose_level - 1);
			//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax);

			//G.frame(0.05);


			draw_projective(G, Descr->number, 0, Descr->animate_nb_of_steps, true, i, Descr->animate_transition_nb_of_steps, false, 0, false, 0);
			G.finish(cout, true);

		}

		for (i = 0; i <= Descr->animate_nb_of_steps; i++, frame++) {


			if (f_v) {
				cout << "frame " << frame << " animate step " << i << " / " << Descr->animate_nb_of_steps << ":" << endl;
			}
			mp_graphics G;

			string fname;
			fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);

			G.init(fname, Opt, verbose_level);
			//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
			//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width,
			//		verbose_level - 1);
			//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax);

			//G.frame(0.05);


			draw_projective(G, Descr->number, i, Descr->animate_nb_of_steps, false, 0, 0, false, 0, false, 0);
			G.finish(cout, true);

		}

		for (i = 0; i <= Descr->animate_transition_nb_of_steps; i++, frame++) {

			if (f_v) {
				cout << "frame " << frame << " transition in step " << i << " / " << Descr->animate_transition_nb_of_steps << ":" << endl;
			}
			mp_graphics G;

			string fname;
			fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);

			G.init(fname, Opt, verbose_level);
			//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
			//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width,
			//		verbose_level - 1);
			//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax);

			//G.frame(0.05);


			draw_projective(G, Descr->number,
					Descr->animate_nb_of_steps, Descr->animate_nb_of_steps, true,
					Descr->animate_transition_nb_of_steps - i, Descr->animate_transition_nb_of_steps,
					false, 0, false, 0);
			G.finish(cout, true);

		}
		if (Descr->f_trailer_page) {

			for (i = 0; i <= 7; i++, frame++) {
				mp_graphics G;

				string fname;
				fname = Descr->fname + "_" + std::to_string(Descr->number) + "_" + std::to_string(i);

				G.init(fname, Opt, verbose_level);
				//G.setup(fname, 0, 0, ONE_MILLION, ONE_MILLION,
				//		xmax, ymax, Opt->f_embedded, Opt->f_sideways, Opt->scale, Opt->line_width,
				//		verbose_level - 1);

				draw_projective(G, Descr->number, 0,
						Descr->animate_nb_of_steps, true, i, Descr->animate_transition_nb_of_steps,
						false, 0, true, i);


				G.finish(cout, true);


			}
		}


		cout << "frame=" << frame << endl;
	}



	Os.time_check(cout, t0);
	cout << endl;
}



void graphical_output::draw_projective(mp_graphics &G,
		int number, int animate_step, int animate_nb_of_steps,
	int f_transition, int transition_step, int transition_nb_steps,
	int f_title_page, int title_page_step,
	int f_trailer_page, int trailer_page_step)
{
	double *Dx, *Dy;
	int *Px, *Py;
	double x_stretch = 1.;
	double y_stretch = 1.;
	double dx = ONE_MILLION * 50 * x_stretch;
	double dy = ONE_MILLION * 50 * y_stretch; // stretch factor
	//double x_labels_offset = -.5;
	//double y_labels_offset = -.5;
	//double x_tick_half_width = 0.1;
	//double y_tick_half_width = 0.1;
	int N = 30;
	int i;
	double x_min = -10;
	double x_max = 10;
	double t_min = -1.13;
	double t_max = 5;
	double Delta_t;
	double step;
	double y_min = 0;
	double y_max = 2;
	double x, y, t;
	//int subdivide_v = 4;
	//int subdivide_h = 4;
	int f_plot_grid = true;
	int f_plot_curve = true;
	//int x_mod = 1;
	//int y_mod = 1;
	//int x_tick_mod = 1;
	//int y_tick_mod = 1;
	double height = 3.;
	double R, R2, X, Y;
	int mirror;
	int f_projection_on = true;
	double radius = 10.;
	int N_curve = 500;
	orbiter_kernel_system::numerics Num;


	cout << "draw_projective number=" << number
			<< " animate_step=" << animate_step << " animate_nb_of_steps=" << animate_nb_of_steps << endl;

	if (number == 1 || number == 3) {
		x_min = -10;
		x_max = 10;
		y_min = -10;
		y_max = 10;
		x_stretch = .7;
		y_stretch = .7;
		dx = ONE_MILLION * 50 * x_stretch;
		dy = ONE_MILLION * 50 * y_stretch; // stretch factor
		t_min = -1.119437527;
		t_max = 4;
#if 0
		x_mod = 100;
		y_mod = 100;
		x_tick_mod = 1;
		y_tick_mod = 2;
		subdivide_v = 1;
		subdivide_h = 1;
#endif
		f_plot_curve = true;
#if 0
		x_labels_offset = -.5;
		y_labels_offset = -.5;
		x_tick_half_width = 0.2;
		y_tick_half_width = 0.1;
#endif
		f_plot_grid = true;
		f_plot_curve = true;
		height = 6;
		R = 20;
		f_projection_on = true;
		radius = 10.;
		}
	else if (number == 2 || number == 4) {
		x_min = -10;
		x_max = 10;
		y_min = -10;
		y_max = 10;
		x_stretch = 0.25;
		y_stretch = 0.25;
		dx = ONE_MILLION * 50 * x_stretch;
		dy = ONE_MILLION * 50 * y_stretch; // stretch factor
		t_min = -1.119437527;
		t_max = 4;
#if 0
		x_mod = 100;
		y_mod = 100;
		x_tick_mod = 1;
		y_tick_mod = 2;
		subdivide_v = 1;
		subdivide_h = 1;
#endif
		f_plot_curve = true;
#if 0
		x_labels_offset = -.5;
		y_labels_offset = -.5;
		x_tick_half_width = 0.2;
		y_tick_half_width = 0.1;
#endif
		f_plot_grid = true;
		f_plot_curve = true;
		height = 6;
		R = 20;
		f_projection_on = false;
		radius = 10.;
		}
	else if (number == 5) {
		x_min = -10;
		x_max = 10;
		y_min = -10;
		y_max = 10;
		x_stretch = 0.25;
		y_stretch = 0.25;
		dx = ONE_MILLION * 50 * x_stretch;
		dy = ONE_MILLION * 50 * y_stretch; // stretch factor
		t_min = 0;
		t_max = 4;
#if 0
		x_mod = 100;
		y_mod = 100;
		x_tick_mod = 1;
		y_tick_mod = 2;
		subdivide_v = 1;
		subdivide_h = 1;
#endif
		f_plot_curve = true;
#if 0
		x_labels_offset = -.5;
		y_labels_offset = -.5;
		x_tick_half_width = 0.2;
		y_tick_half_width = 0.1;
#endif
		f_plot_grid = true;
		f_plot_curve = true;
		height = 6;
		R = 20;
		f_projection_on = true;
		radius = 10.;
		}
	else if (number == 7 || number == 8) {
		x_min = -10;
		x_max = 10;
		y_min = -10;
		y_max = 10;
		x_stretch = 0.25;
		y_stretch = 0.25;
		dx = ONE_MILLION * 50 * x_stretch;
		dy = ONE_MILLION * 50 * y_stretch; // stretch factor
		t_min = 0;
		t_max = 10;
#if 0
		x_mod = 100;
		y_mod = 100;
		x_tick_mod = 1;
		y_tick_mod = 2;
		subdivide_v = 1;
		subdivide_h = 1;
#endif
		f_plot_curve = true;
#if 0
		x_labels_offset = -.5;
		y_labels_offset = -.5;
		x_tick_half_width = 0.2;
		y_tick_half_width = 0.1;
#endif
		f_plot_grid = true;
		f_plot_curve = true;
		height = 6;
		R = 20;
		if (number == 8) {
			f_projection_on = false;
			}
		else {
			f_projection_on = true;
			}
		radius = 10.;
		N_curve = 2000;
		}

	Delta_t = t_max - t_min;

	G.sl_thickness(100);
	//G.sf_color(1);
	//G.sf_interior(10);
	Px = new int[N];
	Py = new int[N];
	Dx = new double[N];
	Dy = new double[N];


	cout << "draw_projective dx=" << dx << " dy=" << dy << endl;

	double box_x_min = x_min * 1.2;
	double box_x_max = x_max * 1.2;
	double box_y_min = y_min * 1.2;
	double box_y_max = y_max * 1.2;

	// draw a black frame:
	Dx[0] = box_x_min;
	Dy[0] = box_y_min;
	Dx[1] = box_x_max;
	Dy[1] = box_y_min;
	Dx[2] = box_x_max;
	Dy[2] = box_y_max;
	Dx[3] = box_x_min;
	Dy[3] = box_y_max;
	for (i = 0; i < 4; i++) {
		//project_to_disc(f_projection_on, radius, height, Dx[i], Dy[i], Dx[i], Dy[i]);
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
		}
	G.polygon5(Px, Py, 0, 1, 2, 3, 0);


	if (f_title_page) {

		X = 0;
		Y = 9;
		for (i = 0; i < 11; i++) {
			Dx[i] = X;
			Dy[i] = Y;
			Y = Y - 1.8;
			}

		for (i = 0; i < 11; i++) {
			Px[i] = Dx[i] * dx;
			Py[i] = Dy[i] * dy;
			}

		string s;


		s.assign("Transforming a Parabola");
		G.aligned_text_array(Px, Py, 0, "", s);

		s.assign("into a Hyperbola");
		G.aligned_text_array(Px, Py, 1, "", s);
		if (title_page_step == 0) {
			return;
			}
		s.assign("Step 1: Move into");
		G.aligned_text_array(Px, Py, 4, "", s);

		s.assign("the projective plane");
		G.aligned_text_array(Px, Py, 5, "", s);
		if (title_page_step == 1) {
			return;
			}

		s.assign("Step 2: Transform the equation");
		G.aligned_text_array(Px, Py, 6, "", s);
		if (title_page_step == 2) {
			return;
			}
		s.assign("Step 3: Move back");
		G.aligned_text_array(Px, Py, 7, "", s);

		s.assign("in the affine plane");
		G.aligned_text_array(Px, Py, 8, "", s);
		if (title_page_step == 3) {
			return;
			}

		s.assign("Created by Anton Betten 2017");
		G.aligned_text_array(Px, Py, 10, "", s);
		return;

		}


	if (f_trailer_page) {

		string s;

		X = 0;
		Y = 9.5;
		for (i = 0; i < 18; i++) {
			Dx[i] = X;
			Dy[i] = Y;
			Y = Y - 1.4;
			}

		for (i = 0; i < 18; i++) {
			Px[i] = Dx[i] * dx;
			Py[i] = Dy[i] * dy;
			}

		s.assign("Thanks for watching!");
		G.aligned_text_array(Px, Py, 0, "", s);
		if (trailer_page_step == 0) {
			return;
			}

		s.assign("credits:");
		G.aligned_text_array(Px, Py, 2, "", s);
		if (trailer_page_step == 1) {
			return;
			}

		s.assign("Felix Klein:");
		G.aligned_text_array(Px, Py, 4, "", s);
		if (trailer_page_step == 2) {
			return;
			}

		s.assign("Introduction to");
		G.aligned_text_array(Px, Py, 5, "", s);
		s.assign("non-euclidean geometry");
		G.aligned_text_array(Px, Py, 6, "", s);
		s.assign("(in German) 1928");
		G.aligned_text_array(Px, Py, 7, "", s);
		if (trailer_page_step == 3) {
			return;
			}
		s.assign("Latex: Donald Knuth");
		G.aligned_text_array(Px, Py, 9, "", s);
		s.assign("Leslie Lamport");
		G.aligned_text_array(Px, Py, 10, "", s);
		if (trailer_page_step == 4) {
			return;
			}
		s.assign("Tikz: Till Tantau");
		G.aligned_text_array(Px, Py, 11, "", s);
		if (trailer_page_step == 5) {
			return;
			}
		s.assign("ImageMagick Studio LLC");
		G.aligned_text_array(Px, Py, 12, "", s);
		if (trailer_page_step == 6) {
			return;
			}
		s.assign("Created by Anton Betten 2017");
		G.aligned_text_array(Px, Py, 14, "", s);
		return;

		}

#if 1
	if (f_plot_grid) {

		int *f_DNE;

		f_DNE = NEW_int(N);


		G.sl_thickness(10);

		for (x = 0; x < R; x++) {

			for (mirror = 0; mirror < 2; mirror++) {
				R2 = sqrt(R * R - x * x);

				// vertical line:
				t_min = -R2;
				t_max = R2;

				Delta_t = t_max - t_min;
				step = Delta_t / (double) N;

				for (i = 0; i < N; i++) {
					f_DNE[i] = false;
					t = t_min + i * step;


					if (mirror == 0) {
						X = x;
						Y = t;
						}
					else {
						X = -x;
						Y = t;
						}


					if (f_DNE[i] == false) {
						Dx[i] = X;
						Dy[i] = Y;
						Num.project_to_disc(f_projection_on, f_transition, transition_step, transition_nb_steps, radius, height, Dx[i], Dy[i], Dx[i], Dy[i]);
						if (Dx[i] < box_x_min || Dx[i] > box_x_max || Dy[i] < box_y_min || Dy[i] > box_y_max) {
							f_DNE[i] = true;
							}
						}
					}
				G.plot_curve(N, f_DNE, Dx, Dy, dx, dy);
				}
			}
		for (y = 0; y < R; y++) {

			for (mirror = 0; mirror < 2; mirror++) {
				R2 = sqrt(R * R - y * y);

				// horizontal line:
				t_min = -R2;
				t_max = R2;

				Delta_t = t_max - t_min;
				step = Delta_t / (double) N;

				for (i = 0; i < N; i++) {
					f_DNE[i] = false;
					t = t_min + i * step;


					if (mirror == 0) {
						X = t;
						Y = y;
						}
					else {
						X = t;
						Y = -y;
						}


					if (f_DNE[i] == false) {
						Dx[i] = X;
						Dy[i] = Y;
						Num.project_to_disc(f_projection_on, f_transition, transition_step, transition_nb_steps, radius, height, Dx[i], Dy[i], Dx[i], Dy[i]);
						if (Dx[i] < box_x_min || Dx[i] > box_x_max || Dy[i] < box_y_min || Dy[i] > box_y_max) {
							f_DNE[i] = true;
							}
						}
					}
				G.plot_curve(N, f_DNE, Dx, Dy, dx, dy);
				}
			}

		FREE_int(f_DNE);
		}
#endif

	if (f_plot_curve) {


		G.sl_color(2);

		double omega;

		omega = -1 * animate_step * M_PI / (2 * animate_nb_of_steps);
		cout << "animate_step=" << animate_step << " omega=" << omega << endl;
		double cos_omega, sin_omega;

		cos_omega = cos(omega);
		sin_omega = sin(omega);
		cout << "sin_omega=" << sin_omega << " cos_omega=" << cos_omega << endl;

		N = N_curve;

		delete [] Px;
		delete [] Py;
		delete [] Dx;
		delete [] Dy;

		int *f_DNE;

		f_DNE = NEW_int(N);
		Px = new int[N];
		Py = new int[N];
		Dx = new double[N];
		Dy = new double[N];


		G.sl_thickness(100);

		// draw the function as a parametric curve:

		double s_min, s_max, s, Delta_s;
		int h;

		for (h = 0; h < 2; h++) {
			if (number == 1 || number == 2) {
				s_min = 0;
				}
			else if (number == 5) {
				s_min = -30;
				}
			else if (number == 7 || number == 8) {
				s_min = -30;
				}
			else {
				s_min = -1.119437527;
				}
			s_max = 30;

			//t_min = -R;
			//t_max = R;

			//Delta_t = t_max - t_min;
			Delta_s = s_max - s_min;
			step = Delta_s / (double) N;

			cout << "Delta_s=" << Delta_s << " step=" << step << endl;
			cout << "draw_projective dx=" << dx << " dy=" << dy << endl;

			for (i = 0; i < N; i++) {


				f_DNE[i] = false;

				s = exp(s_min + i * step);
					// this allows us to get many very small values and many very big values as well.

#if 0
				if (f_projection_on) {
					if (s > 10) {
						s = 10 + exp(s - 10);
						}
					}
#endif
				t = s;
				//t = exp(s);
				//t = t_min + i * step;

				//cout << "i=" << i << " s=" << s << " t=" << t << endl;


				if (number == 1 || number == 2) {
					X = t;
					Y = t * t;
					}
				else if (number == 3 || number == 4) {
					X = t;
					Y = t * t * t + 5 * t + 7;
					if (Y < 0) {
						f_DNE[i] = true;
						}
					else {
						Y = sqrt(Y);
						}
					}
				else if (number == 5) {
					double denom, x, y;
					x = t;
					y = t * t;
					denom = x * sin_omega + cos_omega;

					if (ABS(denom) < 0.0000000001) {
						f_DNE[i] = true;
						}
					else {
						if (h == 0) {
							X = (x * cos_omega - sin_omega) / denom;
							}
						else {
							X = (-x * cos_omega - sin_omega) / denom;
							}
						Y = y / denom;
						}
					}
				else if (number == 7 || number == 8) {
					X = t;
					if (t < 0) {
						f_DNE[i] = true;
						}
					else {
						Y = log(t);
						if (ABS(Y) < 0.0001) {
							f_DNE[i] = true;
							}
						}
					}

#if 0
				if (!f_DNE[i]) {
					double z;

					z = sqrt(X * X + Y * Y);
					if (z > 2 * R) {
						f_DNE[i] = true;
						//cout << endl;
						//cout << "x=" << x << " y=" << y << " is out of bounds" << endl;
						}
					}
#endif

#if 0
				cout << "i=" << i << " s=" << s << " t=" << t << " f_DNE[i]=" << f_DNE[i];
				if (f_DNE[i] == false) {
					cout << " X=" << X << " Y=" << Y << endl;
					}
				else {
					cout << endl;
					}
#endif

				if (f_DNE[i] == false) {
					//double z;

					Dx[i] = X;
					Dy[i] = Y;
#if 0
					if (animate_step == 8) {
						cout << "X=" << X << " Y=" << Y << endl;
						}
#endif
					Num.project_to_disc(f_projection_on, f_transition, transition_step, transition_nb_steps, radius, height, Dx[i], Dy[i], Dx[i], Dy[i]);

					//z = sqrt(Dx[i] * Dx[i] + Dy[i] * Dy[i]);
					if (Dx[i] < box_x_min || Dx[i] > box_x_max || Dy[i] < box_y_min || Dy[i] > box_y_max) {
						f_DNE[i] = true;
						//cout << endl;
						//cout << "x=" << x << " y=" << y << " is out of bounds" << endl;
						}
					if (!f_DNE[i] && isnan(Dx[i])) {
						f_DNE[i] = true;
						}
					if (!f_DNE[i] && isnan(Dy[i])) {
						f_DNE[i] = true;
						}
					if (!f_DNE[i] && ABS(Dx[i]) < 0.0001) {
						f_DNE[i] = true;
						}
					if (!f_DNE[i] && ABS(Dy[i]) < 0.0001) {
						f_DNE[i] = true;
						}
					}
				cout << i << " : s=" << s << " : " << " : t=" << t << " : ";
				if (f_DNE[i]) {
					cout << "-";
					}
				else {
					cout << Dx[i] << ", " << Dy[i];
					}
				cout << endl;
				}

			if (false) {
				cout << "before plot_curve:" << endl;
				for (i = 0; i < N; i++) {
					cout << i << " : ";
					if (f_DNE[i]) {
						cout << "-";
						}
					else {
						cout << Dx[i] << ", " << Dy[i];
						}
					cout << endl;
					}
				}

			G.plot_curve(N, f_DNE, Dx, Dy, dx, dy);
			} // next h


		FREE_int(f_DNE);
		}


}



void graphical_output::tree_draw(tree_draw_options *Tree_draw_options, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::tree_draw" << endl;
	}

	if (!Tree_draw_options->f_file) {
		cout << "graphical_output::tree_draw please use -file <fname>" << endl;
		exit(1);
	}
	tree T;
	orbiter_kernel_system::file_io Fio;
	std::string fname2;

	cout << "Trying to read file " << Tree_draw_options->file_name << " of size "
			<< Fio.file_size(Tree_draw_options->file_name) << endl;

	if (Fio.file_size(Tree_draw_options->file_name) <= 0) {
		cout << "treedraw.out the input file " << Tree_draw_options->file_name
				<< " does not exist" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "graphical_output::tree_draw reading input file " << Tree_draw_options->file_name << endl;
	}
	T.init(Tree_draw_options,
			orbiter_kernel_system::Orbiter->draw_options->xin,
			orbiter_kernel_system::Orbiter->draw_options->yin,
			verbose_level);
	if (f_v) {
		cout << "graphical_output::tree_draw reading input file " << Tree_draw_options->file_name << " finished" << endl;
	}

#if 0
	if (/* T.nb_nodes > 200 ||*/ f_no_circletext) {
		f_circletext = false;
		}
	if (f_on_circle) {
		T.root->place_on_circle(xmax, ymax, T.max_depth);
		}

	if (f_count_leaves) {
		T.f_count_leaves = true;
		}
#endif

	data_structures::string_tools ST;

	fname2 = Tree_draw_options->file_name;
	ST.chop_off_extension(fname2);
	fname2 += "_draw";

	if (f_v) {
		cout << "graphical_output::tree_draw before T.draw" << endl;
	}
	T.draw(fname2,
			Tree_draw_options,
			orbiter_kernel_system::Orbiter->draw_options,
			verbose_level);
	if (f_v) {
		cout << "graphical_output::tree_draw after T.draw" << endl;
	}

#if 0
	if (f_graph) {
		cout << "treedraw.out drawing as graph" << endl;
		T.draw(fname_out,
			xmax, ymax, xmax_out, ymax_out, rad, f_circle, f_circletext,
			f_i, f_e, true, draw_vertex_callback,
			f_embedded, f_sideways, f_on_circle,
			scale, line_width, verbose_level - 1);
		}
	else if (f_placeholder_labels) {
		T.draw(fname_out,
			xmax, ymax, xmax_out, ymax_out, rad, f_circle, f_circletext,
			f_i, f_e, true, draw_vertex_callback_placeholders,
			f_embedded, f_sideways, f_on_circle,
			scale, line_width, verbose_level - 1);
		}
	else {
		T.draw(fname_out,
			xmax, ymax, xmax_out, ymax_out, rad, f_circle, f_circletext,
			f_i, f_e, true, draw_vertex_callback_standard,
			f_embedded, f_sideways, f_on_circle,
			scale, line_width, verbose_level - 1);
		}
#endif
	if (f_v) {
		cout << "graphical_output::tree_draw done" << endl;
	}

}





void graphical_output::animate_povray(
		povray_job_description *Povray_job_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::animate_povray" << endl;
	}
	animate *A;

	A = NEW_OBJECT(animate);

	A->init(Povray_job_description,
			NULL /* extra_data */,
			verbose_level);


	A->draw_frame_callback = interface_povray_draw_frame;



	{
		ofstream fpm(A->fname_makefile);

		A->fpm = &fpm;

		fpm << "all:" << endl;

		if (Povray_job_description->f_rounds) {

			int *rounds;
			int nb_rounds;

			Int_vec_scan(Povray_job_description->rounds_as_string, rounds, nb_rounds);

			cout << "Doing the following " << nb_rounds << " rounds: ";
			Int_vec_print(cout, rounds, nb_rounds);
			cout << endl;

			int r, this_round;

			for (r = 0; r < nb_rounds; r++) {


				this_round = rounds[r];

				cout << "round " << r << " / " << nb_rounds
						<< " is " << this_round << endl;

				//round = first_round + r;

				A->animate_one_round(
						this_round,
						verbose_level);

			}
		}
		else {
			cout << "round " << Povray_job_description->round << endl;


			A->animate_one_round(
					Povray_job_description->round,
					verbose_level);

		}

		fpm << endl;
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << A->fname_makefile << " of size "
			<< Fio.file_size(A->fname_makefile) << endl;



	FREE_OBJECT(A);
	A = NULL;

}


static void interface_povray_draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i, j;



	Anim->Pov->union_start(fp);


	if (round == 0) {


		for (i = 0; i < (int) Anim->S->Drawables.size(); i++) {
			drawable_set_of_objects D;
			int f_group_is_animated = false;

			if (false) {
				cout << "drawable " << i << ":" << endl;
			}
			D = Anim->S->Drawables[i];

			for (j = 0; j < Anim->S->animated_groups.size(); j++) {
				if (Anim->S->animated_groups[j] == i) {
					break;
				}
			}
			if (j < Anim->S->animated_groups.size()) {
				f_group_is_animated = true;
			}
			if (false) {
				if (f_group_is_animated) {
					cout << "is animated" << endl;
				}
				else {
					cout << "is not animated" << endl;
				}
			}
			D.draw(Anim, fp, f_group_is_animated, h, verbose_level);
		}


	}

	//Anim->S->clipping_by_cylinder(0, 1.7 /* r */, fp);

	Anim->rotation(h, nb_frames, round, fp);
	Anim->union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

}




}}}


