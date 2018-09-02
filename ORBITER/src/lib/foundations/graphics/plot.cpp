// plot.C
//
// Anton Betten
// July 18, 2012

#include "foundations.h"
#include <math.h>


void draw_density(char *prefix, INT *the_set, INT set_size,
	INT f_title, const char *title, INT out_of, 
	const char *label_x, 
	INT f_circle, INT circle_at, INT circle_rad, 
	INT f_mu, INT f_sigma, INT nb_standard_deviations, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid, 
	INT xmax, INT ymax, INT offset_x, INT f_switch_x, INT no, INT f_embedded, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_v5 = (verbose_level >= 5);
	INT *set;
	INT x_min = 0, x_max = 1000;
	INT y_min = 0, y_max = 1000;
	INT factor_1000 = 1000;
	//char ext[1000];
	char fname_full[1000];
	INT i, prev; //, prev_start;
	INT *outline_value;
	INT *outline_number;
	INT outline_sz = 0;
	INT f_sideways = FALSE;
	

	if (f_v) {
		cout << "draw_density" << endl;
		}

	set = NEW_INT(set_size);
	for (i = 0; i < set_size; i++) {
		set[i] = the_set[i];
		}
	
	if (f_vv) {
		cout << "draw_density read the following " << set_size << " numbers:" << endl;
		for (i = 0; i < set_size; i++) {
			cout << the_set[i] << endl;
			}
		}

	INT_vec_heapsort(set, set_size);
	if (f_vv) {
		cout << "draw_density after sorting:" << endl;
		for (i = 0; i < set_size; i++) {
			cout << set[i] << endl;
			}
		}
	
	outline_value = NEW_INT(set_size);
	outline_number = NEW_INT(set_size);
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

	
	sprintf(fname_full, "%s_%ld.mp", prefix, no);
	{
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
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
		cout << "draw_density written file " << fname_full << " of size " << file_size(fname_full) << endl;
		}
	FREE_INT(set);
	
}

void draw_density_multiple_curves(char *prefix,
	INT **Data, INT *Data_size, INT nb_data_sets, 
	INT f_title, const char *title, INT out_of, 
	const char *label_x, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid, 
	INT xmax, INT ymax, INT offset_x, INT f_switch_x, 
	INT f_v_logarithmic, double log_base, INT no, INT f_embedded, 
	INT verbose_level)
{
	verbose_level += 6;
	INT f_v = (verbose_level >= 1);
	INT f_v5 = (verbose_level >= 5);
	INT **Data2;
	//INT *set;
	INT x_min = 0, x_max = 1000;
	INT y_min = 0, y_max = 1000;
	INT factor_1000 = 1000;
	//char ext[1000];
	char fname_full[1000];
	INT i, prev; //, prev_start;
	INT **outline_value;
	INT **outline_number;
	INT *outline_sz;
	INT curve;
	INT f_sideways = FALSE;
	

	if (f_v) {
		cout << "draw_density_multiple_curves" << endl;
		}

	Data2 = NEW_PINT(nb_data_sets);
	for (curve = 0; curve < nb_data_sets; curve++) {
		Data2[curve] = NEW_INT(Data_size[curve]);
		for (i = 0; i < Data_size[curve]; i++) {
			Data2[curve][i] = Data[curve][i];
			}
		INT_vec_heapsort(Data2[curve], Data_size[curve]);
		if (f_v5) {
			cout << "after sorting:" << endl;
			for (i = 0; i < Data_size[curve]; i++) {
				cout << Data2[curve][i] << endl;
				}
			}
		}
	INT max_x; 
	
	max_x = 0;
	outline_value = NEW_PINT(nb_data_sets);
	outline_number = NEW_PINT(nb_data_sets);
	outline_sz = NEW_INT(nb_data_sets);
	for (curve = 0; curve < nb_data_sets; curve++) {
		outline_value[curve] = NEW_INT(Data_size[curve]);
		outline_number[curve] = NEW_INT(Data_size[curve]);
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
			cout << "draw_density_multiple_curves outline of size " << outline_sz[curve] << ":" << endl;
			for (i = 0; i < outline_sz[curve]; i++) {
				cout << outline_value[curve][i] << " " << outline_number[curve][i] << endl;
				}



		
			}
		}

	
	sprintf(fname_full, "%s_%ld.mp", prefix, no);
	{
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
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
		cout << "draw_density written file " << fname_full << " of size " << file_size(fname_full) << endl;
		}
	for (curve = 0; curve < nb_data_sets; curve++) {
		FREE_INT(Data2[curve]);
		FREE_INT(outline_value[curve]);
		FREE_INT(outline_number[curve]);
		}
	FREE_PINT(Data2);
	FREE_PINT(outline_value);
	FREE_PINT(outline_number);
	FREE_INT(outline_sz);
	
}


void draw_density2(mp_graphics &G, INT no, 
	INT *outline_value, INT *outline_number, INT outline_sz, 
	INT min_value, INT max_value, INT offset_x, INT f_switch_x, 
	INT f_title, const char *title, 
	const char *label_x, 
	INT f_circle, INT circle_at, INT circle_rad, 
	INT f_mu, INT f_sigma, INT nb_standard_deviations, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid)
{
	INT i;
	INT Px[1000], Py[1000];
	//INT phi = 360 / 12;
	//INT rad1 = 400;
	char str[1000];
	INT y_in, x, y, k;
	
	INT min_x, max_x, min_y, max_y;
	INT sum, a;
	INT mini_x, i0;
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
		sprintf(str, "{\\bf {\\large %ld}}", max_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %ld}}", min_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %ld}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %ld}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}
	sprintf(str, "{\\bf {\\large %ld}}", min_y);
	G.aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %ld}}", max_y);
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
		y_in = (INT) average;
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
		y_in = (INT) (average + k * sigma);
		y_to_pt_on_curve(y_in, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		G.polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "$\\overline{x}+%ld \\sigma$", k);
			}
		else {
			sprintf(str, "$\\overline{x}+\\sigma$");
			}
		G.aligned_text(Px[2], Py[2], "r", str);

		y_in = (INT) (average - k * sigma);
		y_to_pt_on_curve(y_in, x, y, 
			outline_value, outline_number, outline_sz);
		get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y, FALSE);
		Py[0] -= 10;
		G.polygon3(Px, Py, 0, 1, 2);
		if (k > 1) {
			sprintf(str, "{\\bf {\\large $\\overline{x}-%ld \\sigma$}}", k);
			}
		else {
			sprintf(str, "{\\bf {\\large $\\overline{x}-\\sigma$}}");
			}
		G.aligned_text(Px[2], Py[2], "r", str);
		}

#if 0
	y_in = (INT) (average + 2 * sigma);
	y_to_pt_on_curve(y_in, x, y, 
		outline_value, outline_number, outline_sz);
	get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y);
	Py[0] -= 10;
	G.polygon3(Px, Py, 0, 1, 2);
	G.aligned_text(Px[2], Py[2], "r", "$\\overline{x}+2\\sigma$");

	y_in = (INT) (average - 2 * sigma);
	y_to_pt_on_curve(y_in, x, y, 
		outline_value, outline_number, outline_sz);
	get_coord(Px, Py, 0, x, min_y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 1, x, y, min_x, min_y, max_x, max_y);
	get_coord(Px, Py, 2, min_x, y, min_x, min_y, max_x, max_y);
	Py[0] -= 10;
	G.polygon3(Px, Py, 0, 1, 2);
	G.aligned_text(Px[2], Py[2], "r", "$\\overline{x}-2\\sigma$");
#endif




	INT line_dashing = 50;
	G.sl_udsty(line_dashing);

	if (f_v_grid) {
		INT delta;

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
		INT delta;

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

void draw_density2_multiple_curves(mp_graphics &G, INT no, 
	INT **outline_value, INT **outline_number, INT *outline_sz, INT nb_curves, 
	INT min_x, INT max_x, INT min_y, INT max_y, 
	INT offset_x, INT f_switch_x, 
	INT f_title, const char *title, 
	const char *label_x, 
	INT f_v_grid, INT v_grid, INT f_h_grid, INT h_grid,
	INT f_v_logarithmic, double log_base)
{
	INT i;
	INT Px[1000], Py[1000];
	//INT phi = 360 / 12;
	//INT rad1 = 400;
	char str[1000];
	INT curve;
	
#if 0
	INT min_x, max_x, min_y, max_y;
	
	min_x = INT_MAX;
	max_x = INT_MIN;
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
			get_coord_log(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, log_base, f_switch_x);
			}
		else {
			get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, f_switch_x);
			}
		for (i = 0; i < outline_sz[curve]; i++) {
			if (f_v_logarithmic) {
				get_coord_log(Px, Py, 2, outline_number[curve][i], outline_value[curve][i], 
					min_x, min_y, max_x, max_y, log_base, f_switch_x);
				}
			else {
				get_coord(Px, Py, 2, outline_number[curve][i], outline_value[curve][i], 
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
		get_coord_log(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		get_coord_log(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, log_base, FALSE);
		}
	else {
		get_coord(Px, Py, 0, min_x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 1, max_x, min_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 2, max_x, max_y, min_x, min_y, max_x, max_y, FALSE);
		get_coord(Px, Py, 3, min_x, max_y, min_x, min_y, max_x, max_y, FALSE);
		}
	G.polygon5(Px, Py, 0, 1, 2, 3, 0);



	if (f_switch_x) {
		sprintf(str, "{\\bf {\\large %ld}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		sprintf(str, "{\\bf {\\large %ld}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		}
	else {
		sprintf(str, "{\\bf {\\large %ld}}", min_x + offset_x);
		G.aligned_text(Px[0], Py[0], "t", str);
		sprintf(str, "{\\bf {\\large %ld}}", max_x + offset_x);
		G.aligned_text(Px[1], Py[1], "t", str);
		}

	sprintf(str, "{\\bf {\\large %ld}}", min_y);
	G.aligned_text(Px[0], Py[0], "r", str);
	sprintf(str, "{\\bf {\\large %ld}}", max_y);
	G.aligned_text(Px[3], Py[3], "r", str);



	Px[0] = 5 * 100;
	Py[0] = 0;
	sprintf(str, "{\\bf {\\large %s}}", label_x);
	G.aligned_text(Px[0], Py[0], "t", str);





	INT line_dashing = 50;
	INT line_thickness = 15;
	G.sl_udsty(line_dashing);
	G.sl_thickness(line_thickness);

	if (f_v_grid) {
		if (FALSE) {
			double delta, a;
			
			delta = log10(max_x - min_x) / v_grid;
			for (i = 1; i <= v_grid - 1; i++) {
				a = min_x + pow(10, i * delta);
				Px[0] = (INT)a;
				Py[0] = 0;
				Px[1] = (INT)a;
				Py[1] = 1000;
				G.polygon2(Px, Py, 0, 1);
				}
			}
		else {
			INT delta;
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
				get_coord_log(Px, Py, 2, min_x, (INT)a, min_x, min_y, max_x, max_y, log_base, FALSE /* f_switch_x */);
				Px[0] = Px[2];
				Py[0] = Py[2];
				Px[1] = 1000;
				Py[1] = Py[2];
				G.polygon2(Px, Py, 0, 1);
				sprintf(str, "{%ld}", (INT)a);
				G.aligned_text(Px[0], Py[0], "r", str);
				}
			}
		else {
			INT delta;

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

void get_coord(INT *Px, INT *Py, INT idx, INT x, INT y, INT min_x, INT min_y, INT max_x, INT max_y, INT f_switch_x)
{
	Px[idx] = (INT)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
		}
	Py[idx] = (INT)(1000 * (double)(y - min_y) / (double)(max_y - min_y));
}

void get_coord_log(INT *Px, INT *Py, INT idx, INT x, INT y, INT min_x, INT min_y, INT max_x, INT max_y, double log_base, INT f_switch_x)
{
	Px[idx] = (INT)(1000 * (double)(x - min_x) / (double)(max_x - min_x));
	if (f_switch_x) {
		Px[idx] = 1000 - Px[idx];
		}
	Py[idx] = (INT)(1000 * log((double)(y - min_y + 1)) /  
		((double)log(max_y - min_y + 1)));
}



void read_numbers_from_file(const char *fname, 
	INT *&the_set, INT &set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a;
	double d;
	
	if (f_v) {
		cout << "read_numbers_from_file opening file " << fname << " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);
	
	set_size = 1000;
	the_set = NEW_INT(set_size);
	
	for (i = 0; TRUE; i++) {
		if (f.eof()) {
			break;
			}
		f >> d;
		a = (INT) d;
		if (f_vv) {
			cout << "read_set_from_file: the " << i << "-th number is " << d << " which becomes " << a << endl;
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
		cout << "read a set of size " << set_size << " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		INT_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void y_to_pt_on_curve(INT y_in, INT &x, INT &y,  
	INT *outline_value, INT *outline_number, INT outline_sz)
{
	INT f_v = FALSE;
	INT idx, f_found;

	f_found = INT_vec_search(outline_value, outline_sz, y_in, idx);
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

void projective_plane_draw_grid(const char *fname, INT xmax, INT ymax, INT f_with_points, INT rad, 
	INT q, INT *Table, INT nb, 
	INT f_point_labels, char **Point_labels, 
	INT f_embedded, INT f_sideways, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT x_min = 0, x_max = 1000000;
	INT y_min = 0, y_max = 1000000;
	INT factor_1000 = 1000;
	char fname_full[1000];
	//INT f_embedded = TRUE;
	//INT f_sideways = FALSE;
	
	if (f_v) {
		cout << "projective_plane_draw_grid" << endl;
		}
	sprintf(fname_full, "%s.mp", fname);
	{
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G.header();
	G.begin_figure(factor_1000);
	
	projective_plane_draw_grid2(G, q, Table, nb, f_with_points, rad, f_point_labels, Point_labels, verbose_level);


	G.end_figure();
	G.footer();
	}
	cout << "written file " << fname_full << " of size " << file_size(fname_full) << endl;
	if (f_v) {
		cout << "projective_plane_draw_grid done" << endl;
		}
	
}


void projective_plane_draw_grid2(mp_graphics &G, INT q, 
	INT *Table, INT nb, INT f_with_points, INT rad, 
	INT f_point_labels, char **Point_labels, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	double a, b;
	INT x1, x2, x3;

	//INT rad = 17000;
	INT i, h;
	double x_stretch = 0.0010;
	double y_stretch = 0.0010;
	//double x_stretch = 0.01;
	//double y_stretch = 0.01;

	double *Dx, *Dy;
	INT *Px, *Py;
	INT dx = ONE_MILLION * 50 * x_stretch;
	INT dy = ONE_MILLION * 50 * y_stretch;
	INT N = 1000;


	Px = NEW_INT(N);
	Py = NEW_INT(N);
	Dx = new double[N];
	Dy = new double[N];

	
	if (f_v) {
		cout << "projective_plane_draw_grid2" << endl;
		}




	if (f_v) {
		cout << "drawing grid" << endl;
		}


	G.draw_axes_and_grid(
		0., (double)(q - 1), 0., (double)(q - 1), x_stretch, y_stretch, 
		TRUE /* f_x_axis_at_y_min */, TRUE /* f_y_axis_at_x_min */, 
		1 /* x_mod */, 1 /* y_mod */, 1, 1, 
		-1. /* x_labels_offset */, -1. /* y_labels_offset */, 
		0.5 /* x_tick_half_width */, 0.5 /* y_tick_half_width */, 
		TRUE /* f_v_lines */, 1 /* subdivide_v */, TRUE /* f_h_lines */, 1 /* subdivide_h */);

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
			cout << "drawing points, nb=" << nb << endl;
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

			cout << "point " << h << " : " << x1 << ", " << x2 << ", " << x3 << " : " << a << ", " << b << endl;
			
			Dx[0] = a;
			Dy[0] = b;
			
			for (i = 0; i < 1; i++) {
				Px[i] = Dx[i] * dx;
				Py[i] = Dy[i] * dy;
				}

			//G.nice_circle(Px[a * Q + b], Py[a * Q + b], rad);
			G.nice_circle(Px[0], Py[0], rad);
			G.text(Px[0], Py[0], Point_labels[h]);
			}


		}
	else {
		cout << "projective_plane_draw_grid2 not drawing any points" << endl;
		}





	delete [] Px;
	delete [] Py;
	delete [] Dx;
	delete [] Dy;



	if (f_v) {
		cout << "projective_plane_draw_grid2 done" << endl;
		}
}

void projective_plane_make_affine_point(INT q, INT x1, INT x2, INT x3, double &a, double &b)
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





