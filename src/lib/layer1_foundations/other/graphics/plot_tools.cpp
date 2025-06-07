// plot_tools.cpp
//
// Anton Betten
// July 18, 2012

#include "foundations.h"
#include <math.h>


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace graphics {

plot_tools::plot_tools()
{
	Record_birth();
}

plot_tools::~plot_tools()
{
	Record_death();


}



void plot_tools::draw_density(
		layered_graph_draw_options *Draw_options,
		std::string &prefix, int *the_set, int set_size,
	int f_title, std::string &title, int out_of,
	std::string &label_x,
	int f_circle, int circle_at, int circle_rad, 
	int f_mu, int f_sigma, int nb_standard_deviations, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int offset_x,
	int f_switch_x, int no, int f_embedded,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *set;
	int factor_1000 = 1000;
	string fname_full;
	int i, prev;
	int *outline_value;
	int *outline_number;
	int outline_sz = 0;
	data_structures::sorting Sorting;
	

	if (f_v) {
		cout << "plot_tools::draw_density" << endl;
	}

	set = NEW_int(set_size);
	for (i = 0; i < set_size; i++) {
		set[i] = the_set[i];
	}
	
	if (f_vv) {
		cout << "plot_tools::draw_density read the "
				"following " << set_size << " numbers:" << endl;
		for (i = 0; i < set_size; i++) {
			cout << the_set[i] << endl;
		}
	}

	Sorting.int_vec_heapsort(set, set_size);
	if (f_vv) {
		cout << "plot_tools::draw_density after sorting:" << endl;
		for (i = 0; i < set_size; i++) {
			cout << set[i] << endl;
		}
	}
	
	outline_value = NEW_int(set_size);
	outline_number = NEW_int(set_size);
	outline_sz = 0;
	prev = set[0];
	//prev_start = 0;
	for (i = 1; i <= set_size; i++) {
		if (i == set_size || set[i] != prev) {
			outline_value[outline_sz] = prev;
			outline_number[outline_sz] = i - 1;
			outline_sz++;
			prev = set[i];
			//prev_start = i;
		}
	}
	if (f_vv) {
		cout << "plot_tools::draw_density "
				"outline of size " << outline_sz << ":" << endl;
		for (i = 0; i < outline_sz; i++) {
			cout << outline_value[i] << " " << outline_number[i] << endl;
		}
	}


	fname_full = prefix  + "_" + std::to_string(no) + ".mp";

	
	{

		mp_graphics G;

		G.init(fname_full, Draw_options, verbose_level - 1);
#if 0
		mp_graphics G(fname_full,
				x_min, y_min, x_max, y_max, f_embedded, f_sideways, verbose_level - 1);
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = xmax;
		G.out_ymax() = ymax;
		if (f_vv) {
			cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
#endif

		G.header();
		G.begin_figure(factor_1000);

		G.draw_density2(no,
			outline_value, outline_number, outline_sz,
			0, out_of, offset_x, f_switch_x,
			f_title, title,
			label_x,
			f_circle, circle_at, circle_rad,
			f_mu, f_sigma, nb_standard_deviations,
			f_v_grid, v_grid, f_h_grid, h_grid);
	
	
		G.end_figure();
		G.footer();
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "plot_tools::draw_density written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}
	FREE_int(set);
	
}

void plot_tools::draw_density_multiple_curves(
		layered_graph_draw_options *Draw_options,
		std::string &prefix,
	int **Data, int *Data_size, int nb_data_sets, 
	int f_title, std::string &title, int out_of,
	std::string &label_x,
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int offset_x, int f_switch_x,
	int f_v_logarithmic, double log_base, int no, int f_embedded, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	int **Data2;
	int factor_1000 = 1000;
	string fname_full;
	int i, prev;
	int **outline_value;
	int **outline_number;
	int *outline_sz;
	int curve;
	data_structures::sorting Sorting;
	

	if (f_v) {
		cout << "plot_tools::draw_density_multiple_curves" << endl;
		}

	Data2 = NEW_pint(nb_data_sets);
	for (curve = 0; curve < nb_data_sets; curve++) {
		Data2[curve] = NEW_int(Data_size[curve]);
		for (i = 0; i < Data_size[curve]; i++) {
			Data2[curve][i] = Data[curve][i];
			}
		Sorting.int_vec_heapsort(Data2[curve], Data_size[curve]);
		if (f_v5) {
			cout << "after sorting:" << endl;
			for (i = 0; i < Data_size[curve]; i++) {
				cout << Data2[curve][i] << endl;
				}
			}
		}
	int max_x; 
	
	max_x = 0;
	outline_value = NEW_pint(nb_data_sets);
	outline_number = NEW_pint(nb_data_sets);
	outline_sz = NEW_int(nb_data_sets);
	for (curve = 0; curve < nb_data_sets; curve++) {
		outline_value[curve] = NEW_int(Data_size[curve]);
		outline_number[curve] = NEW_int(Data_size[curve]);
		outline_sz[curve] = 0;
		max_x = MAXIMUM(max_x, Data_size[curve]);
		prev = Data2[curve][0];
		//prev_start = 0;
		for (i = 1; i <= Data_size[curve]; i++) {
			if (i == Data_size[curve] || Data2[curve][i] != prev) {
				outline_value[curve][outline_sz[curve]] = prev;
				outline_number[curve][outline_sz[curve]] = i - 1;
				outline_sz[curve]++;
				prev = Data2[curve][i];
				//prev_start = i;
			}
		}
		if (f_v5) {
			cout << "plot_tools::draw_density_multiple_curves outline "
					"of size " << outline_sz[curve] << ":" << endl;
			for (i = 0; i < outline_sz[curve]; i++) {
				cout << outline_value[curve][i] << " "
						<< outline_number[curve][i] << endl;
			}



		
		}
	}


	fname_full = prefix + "_" + std::to_string(no) + ".mp";

	

	{
		mp_graphics G;

		G.init(fname_full, Draw_options, verbose_level - 1);

#if 0
		mp_graphics G(fname_full,
				x_min, y_min, x_max, y_max, f_embedded, f_sideways, verbose_level - 1);
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = xmax;
		G.out_ymax() = ymax;
		if (f_v5) {
			cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
#endif

		G.header();
		G.begin_figure(factor_1000);

		G.draw_density2_multiple_curves(no,
			outline_value, outline_number, outline_sz, nb_data_sets,
			0, max_x - 1, 0, out_of,
			offset_x, f_switch_x,
			f_title, title,
			label_x,
			f_v_grid, v_grid, f_h_grid, h_grid,
			f_v_logarithmic, log_base);
	
	
		G.end_figure();
		G.footer();
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "plot_tools::draw_density written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}
	for (curve = 0; curve < nb_data_sets; curve++) {
		FREE_int(Data2[curve]);
		FREE_int(outline_value[curve]);
		FREE_int(outline_number[curve]);
	}
	FREE_pint(Data2);
	FREE_pint(outline_value);
	FREE_pint(outline_number);
	FREE_int(outline_sz);
	
}


void plot_tools::get_coord(
		int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y, int f_switch_x)
{
	Px[idx] = (int)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
	}
	Py[idx] = (int)(1000 * (double)(y - min_y) / (double)(max_y - min_y));
}

void plot_tools::get_coord_log(
		int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y,
		double log_base, int f_switch_x)
{
	Px[idx] = (int)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
	}
	Py[idx] = (int)(1000 * log((double)(y - min_y + 1)) /  
		((double)log(max_y - min_y + 1)));
}



void plot_tools::y_to_pt_on_curve(
		int y_in, int &x, int &y,
	int *outline_value, int *outline_number, int outline_sz)
{
	int f_v = false;
	int idx, f_found;
	data_structures::sorting Sorting;

	f_found = Sorting.int_vec_search(outline_value, outline_sz, y_in, idx);
	if (f_found) {
		x = outline_number[idx];
		y = outline_value[idx];
		if (f_v) {
			cout << "y-value " << y_in << " found at " << x << endl;
		}
	}
	else {
		if (f_v) {
			cout << "y-value " << y_in << " not found" << endl;
		}
		x = outline_number[idx];
		y = outline_value[idx];
		if (f_v) {
			cout << "x1=" << x << " y1=" << outline_value[idx] << endl;
		}
		if (idx - 1 >= 0) {
			x = outline_number[idx - 1];
			if (f_v) {
				cout << "x2=" << x << " y2=" << outline_value[idx - 1] << endl;
			}
			x = outline_number[idx - 1];
			y = y_in;
		}
		else {
			if (f_v) {
				cout << "at the bottom" << endl;
			}
		}
	}

}

void plot_tools::projective_plane_draw_grid(
		std::string &fname,
		layered_graph_draw_options *O,
	int q, int *Table, int nb, 
	int f_point_labels, std::string *Point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int factor_1000 = 1000;
	string fname_full;
	
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid" << endl;
	}


	fname_full = fname + "_draw.mp";


	{
		mp_graphics G;

		G.init(fname_full, O, verbose_level - 1);

#if 0
		mp_graphics G(fname_full,
				0, 0,
				O->xin, O->yin,
				O->f_embedded, O->f_sideways, verbose_level - 1);
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = O->xout;
		G.out_ymax() = O->yout;
		if (f_v) {
			cout << "plot_tools::projective_plane_draw_grid" << endl;
		}
#endif

		G.header();
		G.begin_figure(factor_1000);

		if (f_v) {
			cout << "plot_tools::projective_plane_draw_grid "
					"before projective_plane_draw_grid2" << endl;
		}

		G.projective_plane_draw_grid2(
				O,
				q, Table, nb,
				f_point_labels, Point_labels,
				verbose_level - 1);

		if (f_v) {
			cout << "plot_tools::projective_plane_draw_grid "
					"after projective_plane_draw_grid2" << endl;
		}


		G.end_figure();
		G.footer();
	}
	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname_full << " of size "
			<< Fio.file_size(fname_full) << endl;
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid done" << endl;
	}
	
}



void plot_tools::draw_something(
		draw_mod_n_description *Descr,
		std::string &options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int factor_1000 = 1000;
	string fname_full;

	if (f_v) {
		cout << "plot_tools::draw_something" << endl;
	}

	graphics::layered_graph_draw_options *O;


	if (!Descr->f_file) {
		cout << "please use -file <fname>" << endl;
		exit(1);
	}

	if (!Descr->f_draw_options) {
		cout << "please use -draw_options" << endl;
		exit(1);
	}

	O = Get_draw_options(Descr->draw_options_label);

	fname_full = Descr->fname + "_draw.mp";



	std::map<std::string, std::string> symbol_table;

	other::data_structures::string_tools ST;

	int f_mod_n = false;
	int f_desargues = false;

	if (options.length()) {

		ST.parse_value_pairs(symbol_table,
				options, verbose_level - 1);

		if (f_v) {
			cout << "plot_tools::draw_something parsing option pairs" << endl;
		}



		{
			std::map<std::string, std::string>::iterator it = symbol_table.begin();


			// Iterate through the map and print the elements
			while (it != symbol_table.end()) {
				string label;
				string val;

				label = it->first;
				val = it->second;
				if (f_v) {
					cout << "plot_tools::draw_something key = " << label << " value = " << val << endl;
				}
				//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
				//assignment.insert(std::make_pair(label, a));
				if (label == "mod_n" /*ST.stringcmp(label, "dense") == 0*/) {
					if (val == "on" /*ST.stringcmp(val, "on") == 0*/) {
						f_mod_n = true;
						if (f_v) {
							cout << "plot_tools::draw_something f_mod_n = true" << endl;
						}
					}
					else {
						cout << "plot_tools::draw_something unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
					if (f_v) {
						cout << "plot_tools::draw_something f_mod_n = true" << endl;
					}
				}
				else if (label == "desargues" /*ST.stringcmp(label, "dense") == 0*/) {
					if (val == "on" /*ST.stringcmp(val, "on") == 0*/) {
						f_desargues = true;
						if (f_v) {
							cout << "plot_tools::draw_something f_desargues = true" << endl;
						}
					}
					else {
						cout << "plot_tools::draw_something unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
					if (f_v) {
						cout << "plot_tools::draw_something f_desargues = true" << endl;
					}
				}

				else {
					cout << "plot_tools::draw_something unknown option "
							<< label << " with value " << val << endl;
					exit(1);
				}
				++it;
			}
		}
	}



	{
		mp_graphics G;

		G.init(fname_full, O, verbose_level);

		G.header();
		G.begin_figure(factor_1000);



		if (f_mod_n) {
			if (f_v) {
				cout << "plot_tools::draw_something "
						"before draw_mod_n_work" << endl;
			}
			draw_mod_n_work(
					G,
					O,
					Descr,
					verbose_level);
			if (f_v) {
				cout << "plot_tools::draw_something "
						"after draw_mod_n_work" << endl;
			}
		}
		else if (f_desargues) {
			if (f_v) {
				cout << "plot_tools::draw_something "
						"before draw_desargues" << endl;
			}
			draw_desargues(
					G,
					O,
					Descr,
					verbose_level);
			if (f_v) {
				cout << "plot_tools::draw_something "
						"after draw_desargues" << endl;
			}

		}
		else {
			cout << "plot_tools::draw_something please specify options" << endl;
			exit(1);
		}

		double move_out = 0.01;

		if (O->f_corners) {
			G.frame(move_out);
		}



		G.end_figure();
		G.footer();
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname_full << " of size "
				<< Fio.file_size(fname_full) << endl;
	}

	if (f_v) {
		cout << "plot_tools::draw_something done" << endl;
	}

}



#ifndef M_PI
#define M_PI 3.141516
#endif


void plot_tools::draw_mod_n_work(
		mp_graphics &G,
		layered_graph_draw_options *O,
		draw_mod_n_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double *Dx, *Dy;
	double *D2x, *D2y;
	int *Px, *Py;
	double x_stretch = 1.;
	double y_stretch = 1.;
	int dx = O->xin * 0.25;
	int dy = O->yin * 0.25; // stretch factor
	int N = 1000;
	int i, j;
	double start_angle = 0;
	orbiter_kernel_system::numerics Num;
	int f_do_it = false;


	int n = Descr->n;

	if (f_v) {
		cout << "plot_tools::draw_mod_n_work n=" << n << endl;
	}


	G.sl_thickness(100);
	//G.sf_color(1);
	//G.sf_interior(10);
	Px = new int[N];
	Py = new int[N];
	Dx = new double[N];
	Dy = new double[N];

	int M, N0;

	N0 = 4 * n + 1 + 4;
	M = N0 + n;

	for (i = 0; i < n; i++) {
		Num.on_circle_double(Dx, Dy, i, start_angle + i * 360. / (double) n, 1.0);
	}
	for (i = 0; i < n; i++) {
		Num.on_circle_double(Dx, Dy, n + 1 + i, start_angle + i * 360. / (double) n, 1.2);
	}
	for (i = 0; i < n; i++) {
		Num.on_circle_double(Dx, Dy, 2 * n + 1 + i, start_angle + i * 360. / (double) n, .9);
	}
	for (i = 0; i < n; i++) {
		Num.on_circle_double(Dx, Dy, 3 * n + 1 + i, start_angle + i * 360. / (double) n, 1.1);
	}
	Num.on_circle_double(Dx, Dy, n, 0, 0.0);
	Num.on_circle_double(Dx, Dy, 4 * n + 1 + 0, 0., 1.3);
	Num.on_circle_double(Dx, Dy, 4 * n + 1 + 1, 90, 1.3);
	Num.on_circle_double(Dx, Dy, 4 * n + 1 + 2, 180, 1.3);
	Num.on_circle_double(Dx, Dy, 4 * n + 1 + 3, 270, 1.3);

	for (i = 0; i < n; i++) {
		cout << "i=" << i << " Dx=" << Dx[i] << " Dy=" << Dy[i] << endl;
	}

	for (i = 0; i < M; i++) {
		Px[i] = Dx[i] * dx * x_stretch;
		Py[i] = Dy[i] * dy * y_stretch;
	}

	// big circle:
	G.circle(Px[n], Py[n], (int) dx * x_stretch);


	if (Descr->f_eigenvalues) {
		D2x = new double[n];
		D2y = new double[n];

		double v[2];
		double w[2];
		for (i = 0; i < n; i++) {
			v[0] = cos((start_angle + i * 360. / (double) n) / 360. * 2 * M_PI);
			v[1] = sin((start_angle + i * 360. / (double) n) / 360. * 2 * M_PI);
			w[0] = Descr->eigenvalues_A[0] * v[0] + Descr->eigenvalues_A[1] * v[1];
			w[1] = Descr->eigenvalues_A[2] * v[0] + Descr->eigenvalues_A[3] * v[1];
			D2x[i] = w[0];
			D2y[i] = w[1];
		}
		for (i = 0; i < n; i++) {
			Px[N0 + i] = D2x[i] * dx * x_stretch;
			Py[N0 + i] = D2y[i] * dy * y_stretch;
		}

	}

	G.sf_interior(100);
	G.sf_color(1);
	for (i = 0; i < n; i++) {
		cout << "drawing circle " << i << " at " << Px[i] << ", " << Py[i]
			<< " with rad=" << O->rad << endl;

		if (Descr->f_mod_s) {
			if ((i % Descr->mod_s) == 0) {
				f_do_it = true;
			}
			else {
				f_do_it = false;
			}
		}
		else {
			f_do_it = true;
		}

		if (f_do_it) {
			G.circle(Px[i], Py[i], O->rad);
		}
		if (Descr->f_eigenvalues) {
			G.sl_ends(0, 1);
			G.sl_color(2); // red
			G.polygon2(Px, Py, n, i);
			G.sl_color(4); // blue
			G.polygon2(Px, Py, i, N0 + i);
			G.sl_ends(0, 0);
			G.sl_color(1); // black
		}
	}

	if (Descr->f_label_nodes) {

		for (i = 0; i < n; i++) {

			string str;

			if (O->f_nodes_empty) {
				str = "";
			}
			else {

				if (Descr->f_divide_out_by) {
					j = i / Descr->divide_out_by;
				}
				else {
					j = i;
				}
				str = std::to_string(j);
			}

			if (Descr->f_mod_s) {
				if ((i % Descr->mod_s) == 0) {
					f_do_it = true;
				}
				else {
					f_do_it = false;
				}
			}
			else {
				f_do_it = true;
			}
			if (f_do_it) {
				string s;
				s.assign(str);
				G.text(Px[n + 1 + i], Py[n + 1 + i], s);
			}
		}
	}


	if (Descr->f_inverse) {
		//finite_field *F;
		long int a, b, g, u, v;
		algebra::number_theory::number_theory_domain NT;

		//F = NEW_OBJECT(finite_field);
		//F->init(q);


		for (a = 1; a < n; a++) {

			g = NT.gcd_lint(a, n);

			if (g == 1) {

				NT.extended_gcd_lint(a, n, g, u, v);
				b = u;
				while (b < 0) {
					b += n;
				}
				b = b % n;

				cout << "inverse of " << a << " mod " << n << " is " << b << endl;
				//b = F->inverse(a);
				if (a == b) {
					G.polygon2(Px, Py, 2 * n + 1 + a, 3 * n + 1 + a);
				}
				else {
					G.polygon2(Px, Py, a, b);
				}
			}
			else {
				cout << "the element " << a << " does not have an inverse mod " << n << endl;
			}
		}
		//FREE_OBJECT(F);
	}

	if (Descr->f_additive_inverse) {
		//finite_field *F;
		long int a, b; //, g, u, v;
		algebra::number_theory::number_theory_domain NT;

		//F = NEW_OBJECT(finite_field);
		//F->init(q);


		for (a = 0; a < n; a++) {

			b = (n - a) % n;

			//b = F->inverse(a);
			cout << "negative of " << a << " mod " << n << " is " << b << endl;

			if (a == b) {
				G.polygon2(Px, Py, 2 * n + 1 + a, 3 * n + 1 + a);
			}
			else {
				G.polygon2(Px, Py, a, b);
			}
		}
		//FREE_OBJECT(F);
	}


	G.polygon2(Px, Py, 4 * n + 1 + 0, 4 * n + 1 + 2);
	G.polygon2(Px, Py, 4 * n + 1 + 1, 4 * n + 1 + 3);

	if (Descr->f_power_cycle) {
		//finite_field *F;
		long int a1, a, b; //, g, u, v;
		algebra::number_theory::number_theory_domain NT;

		cout << "f_power_cycle base = " << Descr->power_cycle_base << endl;

		if (Descr->power_cycle_base == -1) {
			a1 = NT.primitive_root(n, verbose_level);
		}
		else {
			a1 = Descr->power_cycle_base;
		}

		cout << "a1= " << a1 << endl;
		cout << "n= " << n << endl;

		//F = NEW_OBJECT(finite_field);
		//F->init(q);


		a = a1;
		for (i = 0; ; i++) {

			b = (a * a1) % n;

			//b = F->inverse(a);
			cout << "a= " << a << " b= " << b << endl;

			G.polygon2(Px, Py, a, b);

			if (b == a1) {
				break;
			}
			a = b;
		}
		//FREE_OBJECT(F);
	}

	if (Descr->f_cyclotomic_sets) {


		algebra::number_theory::number_theory_domain NT;
		int *reps;
		int nb_reps;

		Int_vec_scan(Descr->cyclotomic_sets_reps, reps, nb_reps);

		cout << "cyclotomic sets of ";
		Int_vec_print(cout, reps, nb_reps);
		cout << " modulo " << Descr->cyclotomic_sets_q << endl;

		if (Descr->f_cyclotomic_sets_thickness) {
			G.sl_thickness(Descr->cyclotomic_sets_thickness);
		}
		else {
			G.sl_thickness(110);
		}

		for (i = 0; i < nb_reps; i++) {
			std::vector<int> cyclotomic_set;
			int a, b, h;

			NT.cyclotomic_set(cyclotomic_set,
					reps[i], Descr->cyclotomic_sets_q, n,
					verbose_level);

			G.sl_color(3 + i);
			for (h = 0; h < cyclotomic_set.size(); h++) {
				a = cyclotomic_set[h];
				if (h < cyclotomic_set.size() - 1) {
					b = cyclotomic_set[h + 1];
				}
				else {
					b = cyclotomic_set[0];
				}
				G.polygon2(Px, Py, a, b);

			}
			G.sl_color(0);
		}
		G.sl_thickness(100);
		//cyclotomic_sets_q, std::string &cyclotomic_sets_reps

	}

	#if 0
	G.sf_interior(0);
	G.sf_color(0);
	i = number;

	G.sf_interior(100);
	G.sf_color(1);
	for (i = 0; i < number; i++) {
		G.circle(Px[i], Py[i], rad3);
	}

#endif

#if 0
	if (f_plot_curve) {
		cout << "log(10) = " << log(10) << endl;
		cout << "log(2.782) = " << log(2.782) << endl;

		G.sl_thickness(100);
		// draw function
		step = Delta_x / (double) N;
		lo = log(0.75); // logarithm base e
		cout << "lo=" << lo << endl;
		for (i = 0; i < N; i++) {
			x = x_min + i * step;
			y = exp(lo * x);
			//cout << "i=" << i << " x=" << x << " y=" << y << endl;
			Px[i] = x * dx;
			Py[i] = y * dy;
			}
		for (i = 0; i < N; i += 5) {
			if (i < N - 5) {
				G.polygon6(Px, Py, i + 0, i + 1, i + 2, i + 3, i + 4, i + 5);
				}
			else {
				G.polygon5(Px, Py, i + 0, i + 1, i + 2, i + 3, i + 4);
				}
			}
		}

	if (f_draw_triangles) {
		// draw triangles
		for (i = 0; i < (int)x_max; i++) {
			x1 =  (double) i;
			y1 = exp(lo * x1);
			x2 =  (double) i + 1;
			y2 = y1;
			x3 =  (double) i + 1;
			y3 = exp(lo * x3);
			//cout << "i=" << i << " x=" << x << " y=" << y << endl;
			Px[0] = x1 * dx;
			Py[0] = y1 * dy;
			Px[1] = x2 * dx;
			Py[1] = y2 * dy;
			Px[2] = x3 * dx;
			Py[2] = y3 * dy;
			G.polygon3(Px, Py, 0, 1, 2);
			}
		}
#endif

}


void plot_tools::draw_desargues(
		mp_graphics &G,
		layered_graph_draw_options *O,
		draw_mod_n_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double *Dx, *Dy;
	int *Px, *Py;
	double x_stretch = 1.;
	double y_stretch = 1.;
	int dx = O->xin * 0.25;
	int dy = O->yin * 0.25; // stretch factor
	int N = 1000;
	int i;
	double start_angle = 0;
	orbiter_kernel_system::numerics Num;


	//int n = Descr->n;
	int n;

	n = 40;

	if (f_v) {
		cout << "plot_tools::draw_desargues n=" << n << endl;
	}


	G.sl_thickness(100);
	//G.sf_color(1);
	//G.sf_interior(10);
	Px = new int[N];
	Py = new int[N];
	Dx = new double[N];
	Dy = new double[N];

	int M, N0;

	N0 = 4 * n;
	M = N0 + n;

	for (i = 0; i < n; i++) {
		Num.on_circle_double(Dx, Dy, i, start_angle + i * 360. / (double) n, 1.0);
	}



	for (i = 0; i < n; i++) {
		cout << "i=" << i << " Dx=" << Dx[i] << " Dy=" << Dy[i] << endl;
	}

	for (i = 0; i < M; i++) {
		Px[i] = Dx[i] * dx * x_stretch;
		Py[i] = Dy[i] * dy * y_stretch;
	}

	// big circle:
	//G.circle(Px[n], Py[n], (int) dx * x_stretch);


	int n0, n1, n2, n3, n4, n5, n6, n7, n8, n9;

	n0 = 10;
	n1 = n0 + n / 2 - 5;
	n2 = n0 + n / 2 + 1;
	n3 = n0 + n / 2 + 5;

	n4 = n;
	n5 = n + 1;
	n6 = n + 2;
	n7 = n + 4;
	n8 = n + 5;
	n9 = n + 6;


	orbiter_kernel_system::numerics Numerics;

	double f1 = 0.35;
	double f2 = 0.7;
	double f3 = 0.6;

	if (f_v) {
		cout << "plot_tools::draw_desargues before Numerics.affine_pt1" << endl;
	}
	Numerics.affine_pt1(
			Px, Py,
			n0, n0, n1, f1, n4);
	Numerics.affine_pt1(
			Px, Py,
			n0, n0, n2, f2, n5);
	Numerics.affine_pt1(
			Px, Py,
			n0, n0, n3, f3, n6);

	if (f_v) {
		cout << "plot_tools::draw_desargues before G.circle_with_array" << endl;
	}


	double a1, b1, c1;
	double a2, b2, c2;

	double x, y;


	// compute point 7:

	Numerics.line_through_points(
			Px[n1], Py[n1],
			Px[n2], Py[n2],
			a1, b1, c1);

	Numerics.line_through_points(
			Px[n4], Py[n4],
			Px[n5], Py[n5],
			a2, b2, c2);

	Numerics.intersection_of_lines(
			a1, b1, c1,
			a2, b2, c2,
			x, y);

	Px[n7] = x;
	Py[n7] = y;


	// compute point 8:

	Numerics.line_through_points(
			Px[n1], Py[n1],
			Px[n3], Py[n3],
			a1, b1, c1);

	Numerics.line_through_points(
			Px[n4], Py[n4],
			Px[n6], Py[n6],
			a2, b2, c2);

	Numerics.intersection_of_lines(
			a1, b1, c1,
			a2, b2, c2,
			x, y);

	Px[n8] = x;
	Py[n8] = y;

	// compute point 9:

	Numerics.line_through_points(
			Px[n2], Py[n2],
			Px[n3], Py[n3],
			a1, b1, c1);

	Numerics.line_through_points(
			Px[n5], Py[n5],
			Px[n6], Py[n6],
			a2, b2, c2);

	Numerics.intersection_of_lines(
			a1, b1, c1,
			a2, b2, c2,
			x, y);

	Px[n9] = x;
	Py[n9] = y;





	int Idx[3];

	Idx[0] = n1;
	Idx[1] = n2;
	Idx[2] = n3;

	//int fill_color = 1; // black
	int fill_color = 2; // red
	//int fill_color = 3; //

	G.sf_color(fill_color);
	G.sf_interior(100);

	G.fill_idx(
			Px, Py, Idx, 3,
			"", true /* f_cycle*/);


	Idx[0] = n4;
	Idx[1] = n5;
	Idx[2] = n6;

	fill_color = 4; // 4 = blue

	G.sf_color(fill_color);
	G.sf_interior(100);

	G.fill_idx(
			Px, Py, Idx, 3,
			"", true /* f_cycle*/);



	G.polygon2(Px, Py, n0, n1);
	G.polygon2(Px, Py, n0, n2);
	G.polygon2(Px, Py, n0, n3);


	G.polygon2(Px, Py, n8, n9);

	G.polygon2(Px, Py, n4, n8);
	G.polygon2(Px, Py, n1, n8);

	G.polygon2(Px, Py, n1, n7);
	G.polygon2(Px, Py, n4, n7);

	G.polygon2(Px, Py, n3, n9);
	G.polygon2(Px, Py, n6, n9);

	string s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;

	s0 = "0";
	s1 = "1";
	s2 = "2";
	s3 = "3";
	s4 = "4";
	s5 = "5";
	s6 = "6";
	s7 = "7";
	s8 = "8";
	s9 = "9";


	G.sf_color(0 /*fill_color*/);

	G.circle_text_array(Px, Py, n0, O->rad, s0);
	G.circle_text_array(Px, Py, n1, O->rad, s1);
	G.circle_text_array(Px, Py, n2, O->rad, s2);
	G.circle_text_array(Px, Py, n3, O->rad, s3);
	G.circle_text_array(Px, Py, n4, O->rad, s4);
	G.circle_text_array(Px, Py, n5, O->rad, s5);
	G.circle_text_array(Px, Py, n6, O->rad, s6);

	G.circle_text_array(Px, Py, n7, O->rad, s7);
	G.circle_text_array(Px, Py, n8, O->rad, s8);
	G.circle_text_array(Px, Py, n9, O->rad, s9);


	if (f_v) {
		cout << "plot_tools::draw_desargues after G.circle_with_array" << endl;
	}

}



void plot_tools::draw_point_set_in_plane(
	std::string &fname,
	layered_graph_draw_options *O,
	geometry::projective_geometry::projective_space *P,
	long int *Pts, int nb_pts,
	int f_point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q, i;
	int *Table;

	if (f_v) {
		cout << "plot_tools::draw_point_set_in_plane" << endl;
	}
	if (P->Subspaces->n != 2) {
		cout << "plot_tools::draw_point_set_in_plane n != 2" << endl;
		exit(1);
	}
	q = P->Subspaces->F->q;
	Table = NEW_int(nb_pts * 3);
	for (i = 0; i < nb_pts; i++) {
		P->unrank_point(Table + i * 3, Pts[i]);
	}
	if (f_point_labels) {
		std::string *Labels;

		Labels = new std::string[nb_pts];
		for (i = 0; i < nb_pts; i++) {
			Labels[i] = std::to_string(Pts[i]);
		}
		if (f_v) {
			cout << "plot_tools::draw_point_set_in_plane "
					"before projective_plane_draw_grid" << endl;
		}
		projective_plane_draw_grid(
				fname, O,
			q, Table, nb_pts, true, Labels,
			verbose_level - 1);
		if (f_v) {
			cout << "plot_tools::draw_point_set_in_plane "
					"after projective_plane_draw_grid" << endl;
		}
		delete [] Labels;
	}
	else {
		if (f_v) {
			cout << "plot_tools::draw_point_set_in_plane "
					"before projective_plane_draw_grid" << endl;
		}
		projective_plane_draw_grid(
				fname, O,
			q, Table, nb_pts, false, NULL,
			verbose_level - 1);
		if (f_v) {
			cout << "plot_tools::draw_point_set_in_plane "
					"after projective_plane_draw_grid" << endl;
		}
	}
	FREE_int(Table);
	if (f_v) {
		cout << "plot_tools::draw_point_set_in_plane done" << endl;
	}
}


}}}}







