/*
 * tdo_refinement.cpp
 *
 *  Created on: Oct 28, 2019
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {
namespace combinatorics {


static void print_distribution(std::ostream &ost,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions);
static int compare_func_int_vec(void *a, void *b, void *data);
static int compare_func_int_vec_inverse(void *a, void *b, void *data);
static void distribution_reverse_sorting(int f_increasing,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions);


tdo_refinement::tdo_refinement()
{
	Descr = NULL;

	t0 = 0;
	cnt = 0;
	//p_buf = NULL;


	//geo_parameter GP;

	//geo_parameter GP2;


	f_doit = FALSE;
	nb_written = 0;
	nb_written_tactical = 0;
	nb_tactical = 0;
	cnt_second_system = 0;

}

tdo_refinement::~tdo_refinement()
{
}

void tdo_refinement::init(tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;

	if (f_v) {
		cout << "tdo_refinement::init" << endl;
	}

	t0 = Os.os_ticks();

	tdo_refinement::Descr = Descr;

	GP2.part_nb_alloc = 10000;
	GP2.entries_nb_alloc = 1000000;
	GP2.part = NEW_int(GP2.part_nb_alloc);
	GP2.entries = NEW_int(GP2.entries_nb_alloc);

	if (f_v) {
		cout << "tdo_refinement::init done" << endl;
	}
}


void tdo_refinement::main_loop(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refinement::main_loop" << endl;
	}

	if (!Descr->f_input_file) {
		cout << "please use option -input_file <fanme>" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "tdo_refinement::main_loop "
			"opening file " << Descr->fname_in << " for reading" << endl;
	}
	ifstream f(Descr->fname_in);
	char str[1000];
	data_structures::string_tools ST;

	fname.assign(Descr->fname_in);
	//strcpy(str, Descr->fname_in);
	//get_extension_if_present(str, ext);
	//chop_off_extension_if_present(str, ext);
	ST.chop_off_extension(fname);


	fname_out.assign(fname);
	//sprintf(fname_out, "%s", str);
	if (Descr->f_range) {
		sprintf(str, "_r%d_%d", Descr->range_first, Descr->range_len);
		fname_out.append(str);
	}
	if (Descr->f_select) {
		fname_out.append("_S");
		fname_out.append(Descr->select_label);
	}
	fname_out.append("r.tdo");
	//sprintf(fname_out + strlen(fname_out), "r.tdo");
	{

		if (f_v) {
			cout << "tdo_parameter_calculation::main_loop "
					"opening file " << fname_out << " for writing" << endl;
		}
		ofstream g(fname_out);

		for (cnt = 0; ; cnt++) {

			if (f_v) {
				cout << "tdo_parameter_calculation::main_loop "
						"cnt=" << cnt << endl;
			}

			if (f.eof()) {
				cout << "eof reached" << endl;
				break;
			}

	#if 0
			if (cnt && (cnt % 1000) == 0) {
				cout << cnt << endl;
				registry_dump();
				}
	#endif

			if (!GP.input_mode_stack(f, 0 /*verbose_level - 1*/)) {
				//cout << "GP.input_mode_stack returns FALSE" << endl;
				break;
			}

			if (f) {
				cout << "tdo_refinement::main_loop "
						"cnt=" << cnt << " read input TDO" << endl;
			}

			f_doit = TRUE;
			if (Descr->f_range) {
				if (cnt + 1 < Descr->range_first || cnt + 1 >= Descr->range_first + Descr->range_len) {
					f_doit = FALSE;
				}
			}
			if (Descr->f_select) {
				if (strcmp(GP.label.c_str(), Descr->select_label.c_str())) {
					continue;
				}
			}
			if (f_doit) {
				if (f_v) {
					cout << "tdo_refinement::main_loop "
							"read decomposition " << cnt << endl;
				}
				if (f_vv) {
					GP.print_schemes();
				}
				if (FALSE) {
					cout << "after print_schemes" << endl;
				}
				do_it(g, verbose_level - 1);
			}


		} // next cnt



		g << -1 << " " << nb_written << " TDOs, with " << nb_written_tactical << " being tactical" << endl;
		cout << "tdo_refinement::main_loop " << nb_written
				<< " TDOs, with " << nb_written_tactical << " being tactical" << endl;
	}
	if (f_v) {
		cout << "tdo_refinement::main_loop done" << endl;
	}
}

void tdo_refinement::do_it(ofstream &g, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	tdo_scheme_synthetic G;
	data_structures::partitionstack P;

	if (f_v) {
		cout << "tdo_refinement::do_it "
				"read TDO " << cnt << " " << GP.label << endl;
	}

	GP.init_tdo_scheme(G, verbose_level - 1);
	if (f_vv) {
		cout << "tdo_refinement::do_it "
				"after init_tdo_scheme" << endl;
		GP.print_schemes(G);
	}


	if (f_vvv) {
		cout << "tdo_refinement::do_it "
				"calling init_partition_stack" << endl;
	}
	G.init_partition_stack(verbose_level - 4);
	if (f_vvv) {
		cout << "tdo_refinement::do_it "
				"row_level=" << GP.row_level << endl;
		cout << "tdo_parameter_calculation::do_it "
				"col_level=" << GP.col_level << endl;
	}

	if (GP.col_level > GP.row_level) {
		if (f_vvv) {
			cout << "tdo_refinement::do_it "
					"calling do_row_refinement" << endl;
		}
		do_row_refinement(g, G, P, verbose_level);
		if (f_vvv) {
			cout << "tdo_refinement::do_it "
					"after do_row_refinement" << endl;
		}
	}
	else if (GP.col_level < GP.row_level) {
		if (f_vvv) {
			cout << "tdo_refinement::do_it "
					"calling do_col_refinement" << endl;
		}
		do_col_refinement(g, G, P, verbose_level);
		if (f_vvv) {
			cout << "tdo_refinement::do_it "
					"after do_col_refinement" << endl;
		}
	}
	else {
		GP.write_mode_stack(g, GP.label);
#if 0
		tdo_write(g, GP.label, GP.part, GP.nb_parts, GP.entries, GP.nb_entries,
			GP.row_level, GP.col_level, GP.lambda_level,
			GP.extra_row_level, GP.extra_col_level);
#endif
		if (f_vv) {
			cout << "tdo_refinement::do_it "
					<< GP.label << " written" << endl;
		}
		nb_written++;
		nb_written_tactical++;
	}

	if (f_v) {
		cout << "tdo_refinement::do_it done" << endl;
	}

}

void tdo_refinement::do_row_refinement(
	ofstream &g, tdo_scheme_synthetic &G,
	data_structures::partitionstack &P,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refinement::do_row_refinement "
				"col_level > row_level" << endl;
	}
	int *point_types, nb_point_types, point_type_len;
	int *distributions, nb_distributions;
	int f_success;

	if (Descr->f_lambda3) {
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before G.td3_refine_rows" << endl;
		}
		f_success = G.td3_refine_rows(verbose_level - 1, Descr->f_once,
				Descr->lambda3, Descr->block_size,
			point_types, nb_point_types, point_type_len,
			distributions, nb_distributions);
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after G.td3_refine_rows" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before G.refine_rows" << endl;
		}
		f_success = G.refine_rows(verbose_level - 1,
				Descr->f_use_mckay_solver, Descr->f_once, P,
			point_types, nb_point_types, point_type_len,
			distributions, nb_distributions,
			cnt_second_system, Descr->Sol,
			Descr->f_omit1, Descr->omit1, Descr->f_omit2, Descr->omit2,
			Descr->f_use_packing_numbers,
			Descr->f_dual_is_linear_space,
			Descr->f_do_the_geometric_test);
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after G.refine_rows" << endl;
		}
	}

	if (f_success) {
		if (Descr->f_reverse || Descr->f_reverse_inverse) {
			distribution_reverse_sorting(Descr->f_reverse_inverse,
				point_types, nb_point_types, point_type_len,
				distributions, nb_distributions);
		}
		if (verbose_level >= 5) {
			print_distribution(cout,
				point_types, nb_point_types, point_type_len,
				distributions, nb_distributions);
		}

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before do_all_row_refinements" << endl;
			}
		do_all_row_refinements(GP.label, g, G,
			point_types, nb_point_types, point_type_len,
			distributions, nb_distributions, nb_tactical,
			verbose_level - 2);
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after do_all_row_refinements, found "
					<< nb_distributions << " refinements" << endl;
		}

		nb_written += nb_distributions;
		nb_written_tactical += nb_tactical;
		FREE_int(point_types);
		FREE_int(distributions);
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"Case " << GP.label << ", found " << 0
				<< " row refinements, out of which "
				<< 0 << " are tactical" << endl;
		}
	}
	if (f_v) {
		cout << "tdo_refinement::do_row_refinement done" << endl;
	}
}

void tdo_refinement::do_col_refinement(
		ofstream &g, tdo_scheme_synthetic &G,
		data_structures::partitionstack &P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int *line_types, nb_line_types, line_type_len;
	int *distributions, nb_distributions;
	int f_success;

	if (f_v) {
		cout << "tdo_refinement::do_col_refinement "
				"col_level < row_level" << endl;
	}
	if (Descr->f_lambda3) {
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before G.td3_refine_columns" << endl;
		}

		f_success = G.td3_refine_columns(verbose_level - 1,
				Descr->f_once,
				Descr->lambda3, Descr->block_size,
				Descr->f_scale, Descr->scaling,
			line_types, nb_line_types, line_type_len,
			distributions, nb_distributions);

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"after G.td3_refine_columns" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before G.refine_columns" << endl;
			}
		f_success = G.refine_columns(verbose_level - 1,
				Descr->f_once, P,
			line_types, nb_line_types, line_type_len,
			distributions, nb_distributions,
			cnt_second_system, Descr->Sol,
			Descr->f_omit1, Descr->omit1, Descr->f_omit2, Descr->omit2,
			Descr->f_D1_upper_bound_x0, Descr->D1_upper_bound_x0,
			Descr->f_use_mckay_solver,
			Descr->f_use_packing_numbers);
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"after G.refine_columns" << endl;
			}
		}
	if (f_success) {
		if (Descr->f_reverse || Descr->f_reverse_inverse) {
			if (f_v) {
				cout << "tdo_refinement::do_col_refinement "
						"before G.distribution_reverse_sorting" << endl;
				}
			distribution_reverse_sorting(Descr->f_reverse_inverse,
				line_types, nb_line_types, line_type_len,
				distributions, nb_distributions);
			if (f_v) {
				cout << "tdo_refinement::do_col_refinement "
						"after G.distribution_reverse_sorting" << endl;
				}
			}
		if (verbose_level >= 5) {
			print_distribution(cout,
				line_types, nb_line_types, line_type_len,
				distributions, nb_distributions);
			}

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before do_all_column_refinements" << endl;
			}
		do_all_column_refinements(GP.label, g, G,
			line_types, nb_line_types, line_type_len,
			distributions, nb_distributions, nb_tactical,
			verbose_level - 1);
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
				"after do_all_column_refinements" << endl;
		}
		nb_written += nb_distributions;
		nb_written_tactical += nb_tactical;
		FREE_int(line_types);
		FREE_int(distributions);
		}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
				"Case " << GP.label << ", found " << 0
				<< " col refinements, out of which "
				<< 0 << " are tactical" << endl;
			}
		}
	if (f_v) {
		cout << "tdo_refinement::do_col_refinement done" << endl;
		}
}

void tdo_refinement::do_all_row_refinements(
	std::string &label_in, ofstream &g, tdo_scheme_synthetic &G,
	int *point_types, int nb_point_types, int point_type_len,
	int *distributions, int nb_distributions, int &nb_tactical,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, t;

	if (f_v) {
		cout << "tdo_refinement::do_all_row_refinements" << endl;
	}
	nb_tactical = 0;
	for (i = 0; i < GP.nb_parts; i++) {
		GP2.part[i] = GP.part[i];
	}
	for (i = 0; i < 4 * GP.nb_entries; i++) {
		GP2.entries[i] = GP.entries[i];
	}

	for (t = 0; t < nb_distributions; t++) {

		if (do_row_refinement(t, label_in, g, G, point_types, nb_point_types,
			point_type_len, distributions, nb_distributions,
			verbose_level - 5)) {
			nb_tactical++;
		}

	}
	if (f_v) {
		cout << "Case " << label_in << ", found " << nb_distributions
			<< " row refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
	}
	if (f_v) {
		cout << "tdo_refinement::do_all_row_refinements done" << endl;
	}

}

void tdo_refinement::do_all_column_refinements(
		std::string &label_in, ofstream &g, tdo_scheme_synthetic &G,
	int *line_types, int nb_line_types, int line_type_len,
	int *distributions, int nb_distributions, int &nb_tactical,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, t;


	if (f_v) {
		cout << "tdo_refinement::do_all_column_refinements" << endl;
	}
	nb_tactical = 0;
	for (i = 0; i < GP.nb_parts; i++) {
		GP2.part[i] = GP.part[i];
	}
	for (i = 0; i < 4 * GP.nb_entries; i++) {
		GP2.entries[i] = GP.entries[i];
	}

	for (t = 0; t < nb_distributions; t++) {

		//cout << "tdo_refinement::do_all_column_refinements t=" << t << endl;
		if (do_column_refinement(t, label_in, g, G, line_types, nb_line_types,
			line_type_len, distributions, nb_distributions,
			verbose_level - 5)) {
			nb_tactical++;
		}

	}
	if (f_v) {
		cout << "Case " << label_in << ", found " << nb_distributions
			<< " column refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
		}
	if (f_v) {
		cout << "tdo_refinement::do_all_column_refinements done" << endl;
	}
}


int tdo_refinement::do_row_refinement(
	int t, std::string &label_in, ofstream &g, tdo_scheme_synthetic &G,
	int *point_types, int nb_point_types, int point_type_len,
	int *distributions, int nb_distributions,
	int verbose_level)
// returns TRUE or FALSE depending on whether the
// refinement gave a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	//char label_out[1000];
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "tdo_refinement::do_row_refinement t=" << t << endl;
	}

	type_index = NEW_int(nb_point_types);
	for (i = 0; i < nb_point_types; i++) {
		type_index[i] = -1;
	}

	new_nb_parts = GP.nb_parts;
	if (G.row_level >= 2) {
		R = G.nb_row_classes[ROW_SCHEME];
	}
	else {
		R = 1;
	}
	i = 0;
	h = 0;
	S = 0;
	for (r = 0; r < R; r++) {
		if (G.row_level >= 2) {
			l = G.row_classes_len[ROW_SCHEME][r];
		}
		else {
			//partitionstack &P = G.PB.P;
			l = G.P->startCell[1];
		}
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
		}
		while (i < nb_point_types) {
			a = distributions[t * nb_point_types + i];
			if (a == 0) {
				i++;
				continue;
			}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
			}
			type_index[h++] = i;
			if (s == 0) {
			}
			else {
				GP2.part[new_nb_parts++] = S + s;
			}
			s += a;
			i++;
			if (s == l) {
				break;
			}
			if (s > l) {
				cout << "tdo_refinement::do_row_refinement: s > l" << endl;
				exit(1);
			}
		}
		S += l;
	}
	if (S != G.m) {
		cout << "tdo_refinement::do_row_refinement: S != G.m" << endl;
		exit(1);
	}

	new_nb_entries = GP.nb_entries;
	GP2.part[new_nb_parts] = -1;
	GP2.entries[new_nb_entries * 4 + 0] = -1;
	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++)
			cout << GP2.part[i] << " ";
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++)
			cout << type_index[i] << " ";
		cout << endl;
	}



	{
		tdo_scheme_synthetic G2;

		G2.init_part_and_entries_int(GP2.part, GP2.entries, verbose_level - 2);

		G2.row_level = new_nb_parts;
		G2.col_level = G.col_level;
		G2.extra_row_level = G.row_level; // GP.extra_row_level;
		G2.extra_col_level = GP.extra_col_level;
		G2.lambda_level = G.lambda_level;
		G2.level[ROW_SCHEME] = new_nb_parts;
		G2.level[COL_SCHEME] = G.col_level;
		G2.level[EXTRA_ROW_SCHEME] = G.row_level; // G.extra_row_level;
		G2.level[EXTRA_COL_SCHEME] = G.extra_col_level;
		G2.level[LAMBDA_SCHEME] = G.lambda_level;

		G2.init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a " << G2.nb_row_classes[ROW_SCHEME] << " x " << G2.nb_col_classes[ROW_SCHEME] << " scheme" << endl;
		}
		for (i = 0; i < G2.nb_row_classes[ROW_SCHEME]; i++) {
			c1 = G2.row_classes[ROW_SCHEME][i];
			for (j = 0; j < point_type_len /*G2.nb_col_classes[ROW]*/; j++) {
				c2 = G2.col_classes[ROW_SCHEME][j];
				idx = type_index[i];
				if (idx == -1) {
					continue;
				}
				a = point_types[idx * point_type_len + j];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
				}
				GP2.entries[new_nb_entries * 4 + 0] = new_nb_parts;
				GP2.entries[new_nb_entries * 4 + 1] = c1;
				GP2.entries[new_nb_entries * 4 + 2] = c2;
				GP2.entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
			}
		}

		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << GP2.entries[i * 4 + j] << " ";
				}
				cout << endl;
			}
		}

		char str[1000];

		//sprintf(GP2.label, "%s.%d", label_in, t + 1);
		sprintf(str, ".%d", t + 1);
		GP2.label.assign(label_in);
		GP2.label.append(str);

		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = new_nb_parts;
		GP2.col_level = G.col_level;
		GP2.lambda_level = G.lambda_level;
		GP2.extra_row_level = G.row_level;
		GP2.extra_col_level = G.extra_col_level;

		GP2.write_mode_stack(g, GP2.label);

#if 0
		tdo_write(g, GP2.label,
			GP2.part, new_nb_parts, GP2.entries, new_nb_entries,
			new_nb_parts, G.col_level, G.lambda_level,
			G.row_level/*G.extra_row_level*/, G.extra_col_level);
#endif

		if (f_vv) {
			cout << GP2.label << " written" << endl;
		}
		if (new_nb_parts == G.col_level) {
			f_tactical = TRUE;
		}
		else {
			f_tactical = FALSE;
		}
	}

	FREE_int(type_index);
	if (f_v) {
		cout << "tdo_refinement::do_row_refinement t=" << t << " done" << endl;
	}
	return f_tactical;
}

int tdo_refinement::do_column_refinement(
	int t, std::string &label_in,
	ofstream &g, tdo_scheme_synthetic &G,
	int *line_types, int nb_line_types, int line_type_len,
	int *distributions, int nb_distributions,
	int verbose_level)
// returns TRUE or FALSE depending on whether the
// refinement gave a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	//char label_out[1000];
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "tdo_refinement::do_column_refinement t=" << t << endl;
	}

	type_index = NEW_int(nb_line_types);

	for (i = 0; i < nb_line_types; i++) {
		type_index[i] = -1;
	}
	new_nb_parts = GP.nb_parts;
	R = G.nb_col_classes[COL_SCHEME];
	i = 0;
	h = 0;
	S = G.m;
	for (r = 0; r < R; r++) {
		l = G.col_classes_len[COL_SCHEME][r];
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
		}
		while (i < nb_line_types) {
			a = distributions[t * nb_line_types + i];
			if (a == 0) {
				i++;
				continue;
			}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
			}
			type_index[h++] = i;
			if (s == 0) {
			}
			else {
				GP2.part[new_nb_parts++] = S + s;
			}
			s += a;
			i++;
			if (s == l) {
				break;
			}
			if (s > l) {
				cout << "tdo_refinement::do_column_refinement: s > l" << endl;
				cout << "r=" << r << endl;
				cout << "s=" << s << endl;
				cout << "l=" << l << endl;
				cout << "a=" << a << endl;
				Orbiter->Int_vec->print(cout, distributions + t * nb_line_types, nb_line_types);
				cout << endl;
				exit(1);
			}
		}
		S += l;
	}
	if (S != G.m + G.n) {
		cout << "tdo_refinement::do_column_refinement: S != G.m + G.n" << endl;
		exit(1);
	}

	new_nb_entries = G.nb_entries;
	GP2.part[new_nb_parts] = -1;
	GP2.entries[new_nb_entries * 4 + 0] = -1;
	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++) {
			cout << GP2.part[i] << " ";
		}
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++) {
			cout << type_index[i] << " ";
		}
		cout << endl;
	}

	{
		tdo_scheme_synthetic *G2;

		G2 = NEW_OBJECT(tdo_scheme_synthetic);

		G2->init_part_and_entries_int(GP2.part, GP2.entries, verbose_level - 2);

		G2->row_level = GP.row_level;
		G2->col_level = new_nb_parts;
		G2->extra_row_level = GP.extra_row_level;
		G2->extra_col_level = GP.col_level; // GP.extra_col_level;
		G2->lambda_level = G.lambda_level;
		G2->level[ROW_SCHEME] = G.row_level;
		G2->level[COL_SCHEME] = new_nb_parts;
		G2->level[EXTRA_ROW_SCHEME] = G.extra_row_level;
		G2->level[EXTRA_COL_SCHEME] = GP.col_level; // G.extra_col_level;
		G2->level[LAMBDA_SCHEME] = G.lambda_level;

		G2->init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a " << G2->nb_row_classes[COL_SCHEME] << " x " << G2->nb_col_classes[COL_SCHEME] << " scheme" << endl;
		}
		for (i = 0; i < G2->nb_row_classes[COL_SCHEME]; i++) {
			c1 = G2->row_classes[COL_SCHEME][i];
			for (j = 0; j < G2->nb_col_classes[COL_SCHEME]; j++) {
				c2 = G2->col_classes[COL_SCHEME][j];
				idx = type_index[j];
				if (idx == -1) {
					continue;
				}
				a = line_types[idx * line_type_len + i];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
				}
				GP2.entries[new_nb_entries * 4 + 0] = new_nb_parts;
				GP2.entries[new_nb_entries * 4 + 1] = c2;
				GP2.entries[new_nb_entries * 4 + 2] = c1;
				GP2.entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
			}
		}

		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << GP2.entries[i * 4 + j] << " ";
				}
				cout << endl;
			}
		}

		char str[1000];

		//sprintf(GP2.label, "%s.%d", label_in, t + 1);
		sprintf(str, ".%d", t + 1);
		GP2.label.assign(label_in);
		GP2.label.append(str);

		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = G.row_level;
		GP2.col_level = new_nb_parts;
		GP2.lambda_level = G.lambda_level;
		GP2.extra_row_level = G.extra_row_level;
		GP2.extra_col_level = G.col_level;

		GP2.write_mode_stack(g, GP2.label);

#if 0

		tdo_write(g, GP2.label, GP2.part, new_nb_parts, GP2.entries, new_nb_entries,
			G.row_level, new_nb_parts, G.lambda_level,
			G.extra_row_level, GP.col_level /*G.extra_col_level*/);
#endif

		if (f_vv) {
			cout << GP2.label << " written" << endl;
		}
		if (new_nb_parts == G.row_level) {
			f_tactical = TRUE;
		}
		else {
			f_tactical = FALSE;
		}
		FREE_OBJECT(G2);
	}

	FREE_int(type_index);
	if (f_v) {
		cout << "tdo_refinement::do_column_refinement t=" << t << " done" << endl;
	}
	return f_tactical;
}

// global stuff:


static void print_distribution(ostream &ost,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions)
{
	int i, j;


	ost << "types:" << endl;
	for (i = 0; i < nb_types; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < type_len; j++) {
			ost << setw(3) << types[i * type_len + j];
		}
		ost << endl;
	}
	ost << endl;


	for (j = 0; j < type_len; j++) {
		ost << setw(3) << j + 1 << " & ";
		for (i = 0; i < nb_types; i++) {
			ost << setw(2) << types[i * type_len + j];
			if (i < nb_types - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << endl;

	ost << "distributions:" << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(3) << distributions[i * nb_types + j];
		}
		ost << endl;
	}
	ost << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(2) << distributions[i * nb_types + j];
			if (j < nb_types - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << endl;

	ost << "distributions (in compact format):" << endl;
	int f_first, a;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		f_first = TRUE;
		for (j = 0; j < nb_types; j++) {
			a = distributions[i * nb_types + j];
			if (a == 0) {
				continue;
			}
			if (!f_first) {
				ost << ",";
			}
			ost << nb_types - 1 - j << "^{" << a << "}";
			f_first = FALSE;
		}
		ost << "\\\\" << endl;
	}
	ost << endl;
}


int static compare_func_int_vec(void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;

	for (i = 0; i < size; i++) {
		if (p[i] > q[i]) {
			return -1;
		}
		if (p[i] < q[i]) {
			return 1;
		}
	}
	return 0;
}

int static compare_func_int_vec_inverse(void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;

	for (i = 0; i < size; i++) {
		if (p[i] > q[i]) {
			return 1;
		}
		if (p[i] < q[i]) {
			return -1;
		}
	}
	return 0;
}

static void distribution_reverse_sorting(int f_increasing,
	int *types, int nb_types, int type_len,
	int *distributions, int nb_distributions)
{
	int i, j;
	int *D;
	int **P;
	data_structures::sorting Sorting;

	D = NEW_int(nb_distributions * nb_types);
	P = NEW_pint(nb_distributions);

	for (i = 0; i < nb_distributions; i++) {
		P[i] = D + i * nb_types;
		for (j = 0; j < nb_types; j++) {
			D[i * nb_types + nb_types - 1 - j] = distributions[i * nb_types + j];
		}
	}

	int p[1];

	p[0] = nb_types;

	if (f_increasing) {
		Sorting.quicksort_array(nb_distributions, (void **) P,
				compare_func_int_vec_inverse, (void *)p);
	}
	else {
		Sorting.quicksort_array(nb_distributions, (void **) P,
				compare_func_int_vec, (void *)p);
	}

	for (i = 0; i < nb_distributions; i++) {
		for (j = 0; j < nb_types; j++) {
			distributions[i * nb_types + j] = P[i][nb_types - 1 - j];
		}
	}
	FREE_int(D);
	FREE_pint(P);

}


}}}



