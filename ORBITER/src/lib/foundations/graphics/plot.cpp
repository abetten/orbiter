// plot.C
//
// Anton Betten
// July 18, 2012

#include "foundations.h"
#include <math.h>

namespace orbiter {
namespace foundations {


void draw_density(char *prefix, int *the_set, int set_size,
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
	

	if (f_v) {
		cout << "draw_density" << endl;
		}

	set = NEW_int(set_size);
	for (i = 0; i < set_size; i++) {
		set[i] = the_set[i];
		}
	
	if (f_vv) {
		cout << "draw_density read the "
				"following " << set_size << " numbers:" << endl;
		for (i = 0; i < set_size; i++) {
			cout << the_set[i] << endl;
			}
		}

	int_vec_heapsort(set, set_size);
	if (f_vv) {
		cout << "draw_density after sorting:" << endl;
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
		cout << "draw_density outline of size " << outline_sz << ":" << endl;
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
	
	draw_density2(G, no, 
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
	if (f_v) {
		cout << "draw_density written file " << fname_full
				<< " of size " << file_size(fname_full) << endl;
		}
	FREE_int(set);
	
}

void draw_density_multiple_curves(char *prefix,
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
	

	if (f_v) {
		cout << "draw_density_multiple_curves" << endl;
		}

	Data2 = NEW_pint(nb_data_sets);
	for (curve = 0; curve < nb_data_sets; curve++) {
		Data2[curve] = NEW_int(Data_size[curve]);
		for (i = 0; i < Data_size[curve]; i++) {
			Data2[curve][i] = Data[curve][i];
			}
		int_vec_heapsort(Data2[curve], Data_size[curve]);
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
			cout << "draw_density_multiple_curves outline "
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
	
	draw_density2_multiple_curves(G, no, 
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
	if (f_v) {
		cout << "draw_density written file " << fname_full
				<< " of size " << file_size(fname_full) << endl;
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


void draw_density2(mp_graphics &G, int no, 
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
		G.polygon3(Px, Py, 0, 1, 2);
		Px[0] = Px[2];
		Py[0] = Py[2];
		}
	get_coord(Px, Py, 2, max_x, max_value, 
		min_x, min_y, max_x, max_y, FALSE);
	G.polygon2(Px, Py, 0, 2);
	get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, FALSE);
	get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, FALSE);
	G.polygon5(Px, Py, 0, 1, 2, 3, 0);


	if (f_switch_x) {
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}
	sprintf(str, "{\\bf {\\large %d}}", min_y);
	G.aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %d}}", max_y);
	G.aligned_text(Px[3], Py[3], "r", str);



	Px[0] = 5 * 100;
	Py[0] = 0;
	sprintf(str, "{\\bf {\\large %s}}", label_x);
	G.aligned_text(Px[0], Py[0], "t", str);

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
		G.aligned_text(Px[0], Py[0], "t", str);
		}


	if (f_mu) {
		y_in = (int) average;
		y_to_pt_on_curve(y_in, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		G.polygon3(Px, Py, 0, 1, 2);
		G.aligned_text(Px[2], Py[2], "r", "$\\overline{x}$");
		}


	if (f_circle) {
		y_to_pt_on_curve(circle_at, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, y, min_x, min_y, max_x, max_y, FALSE);
		G.circle(Px[0], Py[0], circle_rad);
		}


	for (k = 1; k < nb_standard_deviations; k++) {
		y_in = (int) (average + k * sigma);
		y_to_pt_on_curve(y_in, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		G.polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "$\\overline{x}+%d \\sigma$", k);
			}
		else {
			sprintf(str, "$\\overline{x}+\\sigma$");
			}
		G.aligned_text(Px[2], Py[2], "r", str);

		y_in = (int) (average - k * sigma);
		y_to_pt_on_curve(y_in, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		G.polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "{\\bf {\\large $\\overline{x}-%d \\sigma$}}", k);
			}
		else {
			sprintf(str, "{\\bf {\\large $\\overline{x}-\\sigma$}}");
			}
		G.aligned_text(Px[2], Py[2], "r", str);
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
	G.sl_udsty(line_dashing);

	if (f_v_grid) {
		int delta;

		delta = 1000 / v_grid;
		for (i = 1; i <= v_grid - 1; i++) {
			Px[0] = i * delta;
			Py[0] = 0;
			Px[1] = i * delta;
			Py[1] = 1000;
			G.polygon2(Px, Py, 0, 1);
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
			G.polygon2(Px, Py, 0, 1);
			}
		}


	if (f_title) {
		Px[0] = 5 * 100;
		Py[0] = 1050;
		sprintf(str, "{\\bf {\\large %s}}", title);
		G.aligned_text(Px[0], Py[0], "b", str);
		}
	
}

void draw_density2_multiple_curves(mp_graphics &G, int no, 
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
			G.polygon3(Px, Py, 0, 1, 2);
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
		G.polygon2(Px, Py, 0, 2);
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
	G.polygon5(Px, Py, 0, 1, 2, 3, 0);



	if (f_switch_x) {
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %d}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %d}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}

	sprintf(str, "{\\bf {\\large %d}}", min_y);
	G.aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %d}}", max_y);
	G.aligned_text(Px[3], Py[3], "r", str);



	Px[0] = 5 * 100;
	Py[0] = 0;
	sprintf(str, "{\\bf {\\large %s}}", label_x);
	G.aligned_text(Px[0], Py[0], "t", str);





	int line_dashing = 50;
	int line_thickness = 15;
	G.sl_udsty(line_dashing);
	G.sl_thickness(line_thickness);

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
				G.polygon2(Px, Py, 0, 1);
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
				G.polygon2(Px, Py, 0, 1);
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
				G.polygon2(Px, Py, 0, 1);
				sprintf(str, "{%d}", (int)a);
				G.aligned_text(Px[0], Py[0], "r", str);
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
				G.polygon2(Px, Py, 0, 1);
				}
			}
		}


	if (f_title) {
		Px[0] = 5 * 100;
		Py[0] = 1050;
		sprintf(str, "{\\bf {\\large %s}}", title);
		G.aligned_text(Px[0], Py[0], "b", str);
		}
	
}

void get_coord(int *Px, int *Py, int idx, int x, int y,
		int min_x, int min_y, int max_x, int max_y, int f_switch_x)
{
	Px[idx] = (int)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
		}
	Py[idx] = (int)(1000 * (double)(y - min_y) / (double)(max_y - min_y));
}

void get_coord_log(int *Px, int *Py, int idx, int x, int y,
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



void read_numbers_from_file(const char *fname, 
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	double d;
	
	if (f_v) {
		cout << "read_numbers_from_file opening file " << fname
				<< " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);
	
	set_size = 1000;
	the_set = NEW_int(set_size);
	
	for (i = 0; TRUE; i++) {
		if (f.eof()) {
			break;
			}
		f >> d;
		a = (int) d;
		if (f_vv) {
			cout << "read_set_from_file: the " << i
					<< "-th number is " << d << " which becomes "
					<< a << endl;
			}
		if (a == -1)
			break;
		the_set[i] = a;
		if (i >= set_size) {
			cout << "i >= set_size" << endl;
			exit(1);
			}
		}
	set_size = i;
	if (f_v) {
		cout << "read a set of size " << set_size
				<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void y_to_pt_on_curve(int y_in, int &x, int &y,  
	int *outline_value, int *outline_number, int outline_sz)
{
	int f_v = FALSE;
	int idx, f_found;

	f_found = int_vec_search(outline_value, outline_sz, y_in, idx);
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

void projective_plane_draw_grid(const char *fname,
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
		cout << "projective_plane_draw_grid" << endl;
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
		cout << "projective_plane_draw_grid" << endl;
		cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
		}
	
	G.header();
	G.begin_figure(factor_1000);
	
	if (f_v) {
		cout << "projective_plane_draw_grid "
				"before projective_plane_draw_grid2" << endl;
		}
	projective_plane_draw_grid2(G, q, Table, nb,
			f_with_points, rad, f_point_labels, Point_labels, verbose_level);
	if (f_v) {
		cout << "projective_plane_draw_grid "
				"after projective_plane_draw_grid2" << endl;
		}


	G.end_figure();
	G.footer();
	}
	cout << "written file " << fname_full << " of size "
			<< file_size(fname_full) << endl;
	if (f_v) {
		cout << "projective_plane_draw_grid done" << endl;
		}
	
}


void projective_plane_draw_grid2(mp_graphics &G, int q, 
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


	G.draw_axes_and_grid(
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
	G.text(Px[0], Py[0], "$x$");
	G.text(Px[1], Py[1], "$y$");

	G.sl_thickness(100);	

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
	G.polygon4(Px, Py, 0, 2, 3, 1);

	if (f_with_points) {

		if (f_v) {
			cout << "projective_plane_draw_grid2 "
					"drawing points, nb=" << nb << endl;
			}

		G.sl_thickness(50);	

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
			G.nice_circle(Px[0], Py[0], rad);
			if (f_point_labels) {
				G.text(Px[0], Py[0], Point_labels[h]);
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

void projective_plane_make_affine_point(
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

}
}





