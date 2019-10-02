// plot_tools.cpp
//
// Anton Betten
// July 18, 2012

#include "foundations.h"
#include <math.h>


using namespace std;


namespace orbiter {
namespace foundations {

plot_tools::plot_tools()
{
}

plot_tools::~plot_tools()
{


}



void plot_tools::draw_density(char *prefix, int *the_set, int set_size,
	int f_title, const char *title, int out_of, 
	const char *label_x, 
	int f_circle, int circle_at, int circle_rad, 
	int f_mu, int f_sigma, int nb_standard_deviations, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int xmax, int ymax, int offset_x,
	int f_switch_x, int no, int f_embedded,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v5 = (verbose_level >= 5);
	int *set;
	int x_min = 0, x_max = 1000;
	int y_min = 0, y_max = 1000;
	int factor_1000 = 1000;
	//char ext[1000];
	char fname_full[1000];
	int i, prev; //, prev_start;
	int *outline_value;
	int *outline_number;
	int outline_sz = 0;
	int f_sideways = FALSE;
	sorting Sorting;
	

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
		cout << "plot_tools::draw_density outline of size " << outline_sz << ":" << endl;
		for (i = 0; i < outline_sz; i++) {
			cout << outline_value[i] << " " << outline_number[i] << endl;
			}
		}

	
	sprintf(fname_full, "%s_%d.mp", prefix, no);
	{
	mp_graphics G(fname_full,
			x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	if (f_vv) {
		cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
	
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
	file_io Fio;

	if (f_v) {
		cout << "plot_tools::draw_density written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
		}
	FREE_int(set);
	
}

void plot_tools::draw_density_multiple_curves(char *prefix,
	int **Data, int *Data_size, int nb_data_sets, 
	int f_title, const char *title, int out_of, 
	const char *label_x, 
	int f_v_grid, int v_grid, int f_h_grid, int h_grid, 
	int xmax, int ymax, int offset_x, int f_switch_x, 
	int f_v_logarithmic, double log_base, int no, int f_embedded, 
	int verbose_level)
{
	verbose_level += 6;
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	int **Data2;
	//int *set;
	int x_min = 0, x_max = 1000;
	int y_min = 0, y_max = 1000;
	int factor_1000 = 1000;
	//char ext[1000];
	char fname_full[1000];
	int i, prev; //, prev_start;
	int **outline_value;
	int **outline_number;
	int *outline_sz;
	int curve;
	int f_sideways = FALSE;
	sorting Sorting;
	

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

	
	sprintf(fname_full, "%s_%d.mp", prefix, no);
	{
	mp_graphics G(fname_full,
			x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	if (f_v5) {
		cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
	
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
	file_io Fio;

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


void plot_tools::get_coord(int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y, int f_switch_x)
{
	Px[idx] = (int)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
		}
	Py[idx] = (int)(1000 * (double)(y - min_y) / (double)(max_y - min_y));
}

void plot_tools::get_coord_log(int *Px, int *Py, int idx, int x, int y,
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



void plot_tools::y_to_pt_on_curve(int y_in, int &x, int &y,
	int *outline_value, int *outline_number, int outline_sz)
{
	int f_v = FALSE;
	int idx, f_found;
	sorting Sorting;

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

void plot_tools::projective_plane_draw_grid(const char *fname,
	int xmax, int ymax, int f_with_points, int rad,
	int q, int *Table, int nb, 
	int f_point_labels, char **Point_labels, 
	int f_embedded, int f_sideways, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x_min = 0, x_max = 1000000;
	int y_min = 0, y_max = 1000000;
	int factor_1000 = 1000;
	char fname_full[1000];
	//int f_embedded = TRUE;
	//int f_sideways = FALSE;
	
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid" << endl;
		}
	sprintf(fname_full, "%s.mp", fname);
	{
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max,
			f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid" << endl;
		cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
	
	G.header();
	G.begin_figure(factor_1000);
	
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid "
				"before projective_plane_draw_grid2" << endl;
		}
	G.projective_plane_draw_grid2(q, Table, nb,
			f_with_points, rad, f_point_labels, Point_labels,
			verbose_level);
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid "
				"after projective_plane_draw_grid2" << endl;
		}


	G.end_figure();
	G.footer();
	}
	file_io Fio;

	cout << "written file " << fname_full << " of size "
			<< Fio.file_size(fname_full) << endl;
	if (f_v) {
		cout << "plot_tools::projective_plane_draw_grid done" << endl;
		}
	
}



}}





