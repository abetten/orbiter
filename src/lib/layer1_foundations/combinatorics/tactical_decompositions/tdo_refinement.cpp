/*
 * tdo_refinement.cpp
 *
 *  Created on: Oct 28, 2019
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {



tdo_refinement::tdo_refinement()
{
	Record_birth();
	Descr = NULL;

	t0 = 0;
	cnt = 0;


	//geo_parameter GP;

	//geo_parameter GP2;




	f_doit = false;
	nb_written = 0;
	nb_written_tactical = 0;
	nb_tactical = 0;
	cnt_second_system = 0;

	Tdo_scheme_synthetic = NULL;

	//P = NULL;

}

tdo_refinement::~tdo_refinement()
{
	Record_death();

	if (Tdo_scheme_synthetic) {
		FREE_OBJECT(Tdo_scheme_synthetic);
	}
}

void tdo_refinement::init(
		tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::os_interface Os;

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


void tdo_refinement::main_loop(
		int verbose_level)
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
	other::data_structures::string_tools ST;

	fname.assign(Descr->fname_in);
	ST.chop_off_extension(fname);


	fname_out.assign(fname);
	if (Descr->f_range) {
		fname_out += "_r" + std::to_string(Descr->range_first) + "_" + std::to_string(Descr->range_len);
	}
	if (Descr->f_select) {
		fname_out += "_S" + Descr->select_label;
	}
	fname_out += "r.tdo";
	{

		if (f_v) {
			cout << "tdo_parameter_calculation::main_loop "
					"opening file " << fname_out << " for writing" << endl;
		}
		ofstream ost(fname_out);

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
				//cout << "GP.input_mode_stack returns false" << endl;
				break;
			}

			if (f) {
				cout << "tdo_refinement::main_loop "
						"cnt=" << cnt << " read input TDO" << endl;
			}

			f_doit = true;
			if (Descr->f_range) {
				if (cnt + 1 < Descr->range_first || cnt + 1 >= Descr->range_first + Descr->range_len) {
					f_doit = false;
				}
			}
			if (Descr->f_select) {
				if (GP.label != Descr->select_label) {
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
				if (false) {
					cout << "after print_schemes" << endl;
				}
				if (f_v) {
					cout << "tdo_refinement::main_loop "
							"before create_all_refinements" << endl;
				}
				create_all_refinements(ost, verbose_level - 1);
				if (f_v) {
					cout << "tdo_refinement::main_loop "
							"after create_all_refinements" << endl;
				}
			}


		} // next cnt



		ost << -1 << " " << nb_written << " TDOs, with " << nb_written_tactical << " being tactical" << endl;
		cout << "tdo_refinement::main_loop " << nb_written
				<< " TDOs, with " << nb_written_tactical << " being tactical" << endl;
	}
	if (f_v) {
		cout << "tdo_refinement::main_loop done" << endl;
	}
}

void tdo_refinement::create_all_refinements(
		std::ofstream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	//tdo_scheme_synthetic G;
	//other::data_structures::partitionstack P;

	Tdo_scheme_synthetic = NEW_OBJECT(tdo_scheme_synthetic);
	//P = NEW_OBJECT(other::data_structures::partitionstack);



	if (f_v) {
		cout << "tdo_refinement::create_all_refinements "
				"read TDO " << cnt << " " << GP.label << endl;
	}

	Tdo_scheme_synthetic->init(Descr, verbose_level - 1);


	GP.init_tdo_scheme(*Tdo_scheme_synthetic, verbose_level - 1);

	if (f_vv) {
		cout << "tdo_refinement::create_all_refinements "
				"after init_tdo_scheme" << endl;
		GP.print_schemes(*Tdo_scheme_synthetic);
	}


	if (f_vvv) {
		cout << "tdo_refinement::create_all_refinements "
				"calling init_partition_stack" << endl;
	}

	Tdo_scheme_synthetic->init_partition_stack(verbose_level - 4);
		// tdo_scheme_synthetic has a different partition,
		// namely the refinement partition of the classes of
		// the current partition w.r.t. the previous (coarser) partition

	if (f_vvv) {
		cout << "tdo_refinement::create_all_refinements "
				"row_level=" << GP.row_level << endl;
		cout << "tdo_parameter_calculation::do_it "
				"col_level=" << GP.col_level << endl;
	}

	if (GP.col_level > GP.row_level) {
		if (f_vvv) {
			cout << "tdo_refinement::create_all_refinements "
					"calling do_row_refinement" << endl;
		}
		do_row_refinement(ost, verbose_level);
		if (f_vvv) {
			cout << "tdo_refinement::create_all_refinements "
					"after do_row_refinement" << endl;
		}
	}
	else if (GP.col_level < GP.row_level) {
		if (f_vvv) {
			cout << "tdo_refinement::do_it "
					"calling do_col_refinement" << endl;
		}
		do_col_refinement(ost, verbose_level);
		if (f_vvv) {
			cout << "tdo_refinement::create_all_refinements "
					"after do_col_refinement" << endl;
		}
	}
	else {
		GP.write_mode_stack(ost, GP.label);
		if (f_vv) {
			cout << "tdo_refinement::create_all_refinements "
					<< GP.label << " written" << endl;
		}
		nb_written++;
		nb_written_tactical++;
	}

	//FREE_OBJECT(G);
	//FREE_OBJECT(P);

	//G = NULL;
	//P = NULL;

	if (f_v) {
		cout << "tdo_refinement::create_all_refinements done" << endl;
	}

}

void tdo_refinement::do_row_refinement(
	std::ofstream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refinement::do_row_refinement "
				"col_level > row_level" << endl;
	}

	int f_success;
	tdo_refinement_output *Output = NULL;


	if (Descr->f_lambda3) {

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"refining 3-design" << endl;
		}

		tdo_refine_3designs *Tdo_refine_3designs;


		Tdo_refine_3designs = NEW_OBJECT(tdo_refine_3designs);

		Tdo_refine_3designs->init(
				Tdo_scheme_synthetic,
				verbose_level);



		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before Tdo_refine_3designs->td3_refine_rows" << endl;
		}

		f_success = Tdo_refine_3designs->td3_refine_rows(
				Output,
				verbose_level - 1);

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after Tdo_refine_3designs->td3_refine_rows" << endl;
		}

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"nb_distributions = " << Output->nb_distributions << endl;
		}
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"refining rows" << endl;
		}

		tdo_refine_rows *Tdo_refine_rows;


		Tdo_refine_rows = NEW_OBJECT(tdo_refine_rows);

		Tdo_refine_rows->init(
				Tdo_scheme_synthetic,
				verbose_level);


		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before Tdo_refine_rows->refine_rows" << endl;
		}
		f_success = Tdo_refine_rows->refine_rows(
				Output,
				cnt_second_system,
				verbose_level - 1);
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after Tdo_refine_rows->refine_rows" << endl;
		}

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"nb_distributions = " << Output->nb_distributions << endl;
		}

		FREE_OBJECT(Tdo_refine_rows);

	}

	if (f_success) {
		if (Descr->f_reverse || Descr->f_reverse_inverse) {

			Output->distribution_reverse_sorting(
					Descr->f_reverse_inverse, verbose_level - 1);

		}
		if (verbose_level >= 5) {
			Output->print_distribution(cout);
		}

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"before do_all_row_refinements" << endl;
		}

		do_all_row_refinements(
				GP.label, ost,
				Output,
			nb_tactical,
			verbose_level - 2);

		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"after do_all_row_refinements, nb_distributions = "
					<< Output->nb_distributions << endl;
		}

		nb_written += Output->nb_distributions;
		nb_written_tactical += nb_tactical;
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_row_refinement "
					"Case " << GP.label << ", found " << 0
				<< " row refinements, out of which "
				<< 0 << " are tactical" << endl;
		}
	}

	if (Output) {
		FREE_OBJECT(Output);
	}

	if (f_v) {
		cout << "tdo_refinement::do_row_refinement done" << endl;
	}
}

void tdo_refinement::do_col_refinement(
		std::ofstream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	tdo_refinement_output *Output = NULL;

	int f_success;

	if (f_v) {
		cout << "tdo_refinement::do_col_refinement "
				"col_level < row_level" << endl;
	}
	if (Descr->f_lambda3) {

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"refining 3-design" << endl;
		}

		tdo_refine_3designs *Tdo_refine_3designs;


		Tdo_refine_3designs = NEW_OBJECT(tdo_refine_3designs);

		Tdo_refine_3designs->init(
				Tdo_scheme_synthetic,
				verbose_level);




		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before Tdo_refine_3designs->td3_refine_columns" << endl;
		}

		f_success = Tdo_refine_3designs->td3_refine_columns(
				Output,
				verbose_level - 1);

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"after Tdo_refine_3designs->td3_refine_columns" << endl;
		}


		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"nb_distributions = " << Output->nb_distributions << endl;
		}
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"refining columns" << endl;
		}

		tdo_refine_cols *Tdo_refine_cols;


		Tdo_refine_cols = NEW_OBJECT(tdo_refine_cols);

		Tdo_refine_cols->init(
				Tdo_scheme_synthetic,
				verbose_level);



		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before Tdo_refine_cols->refine_columns" << endl;
		}
		f_success = Tdo_refine_cols->refine_columns(
				Output,
				cnt_second_system,
				verbose_level - 1);
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"after Tdo_refine_cols->refine_columns" << endl;
		}


		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"nb_distributions = " << Output->nb_distributions << endl;
		}

		FREE_OBJECT(Tdo_refine_cols);
	}
	if (f_success) {
		if (Descr->f_reverse || Descr->f_reverse_inverse) {
			if (f_v) {
				cout << "tdo_refinement::do_col_refinement "
						"before G.distribution_reverse_sorting" << endl;
			}

			Output->distribution_reverse_sorting(Descr->f_reverse_inverse, verbose_level - 1);

			if (f_v) {
				cout << "tdo_refinement::do_col_refinement "
						"after G.distribution_reverse_sorting" << endl;
			}
		}
		if (verbose_level >= 5) {

			Output->print_distribution(cout);
		}

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
					"before do_all_column_refinements" << endl;
		}

		do_all_column_refinements(
				GP.label, ost,
				Output,
				nb_tactical,
			verbose_level - 1);

		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
				"after do_all_column_refinements" << endl;
		}

		nb_written += Output->nb_distributions;
		nb_written_tactical += nb_tactical;
	}
	else {
		if (f_v) {
			cout << "tdo_refinement::do_col_refinement "
				"Case " << GP.label << ", found " << 0
				<< " col refinements, out of which "
				<< 0 << " are tactical" << endl;
		}
	}

	if (Output) {
		FREE_OBJECT(Output);
	}

	if (f_v) {
		cout << "tdo_refinement::do_col_refinement done" << endl;
	}
}

void tdo_refinement::do_all_row_refinements(
	std::string &label_in,
	std::ofstream &ost,
	tdo_refinement_output *Output,
	int &nb_tactical,
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

	for (t = 0; t < Output->nb_distributions; t++) {

		if (f_v) {
			cout << "tdo_refinement::do_all_row_refinements "
					"case " << t << " / " << Output->nb_distributions
					<< " before do_row_refinement" << endl;
		}

		if (do_row_refinement(
				t, label_in, ost,
				Output,
				verbose_level - 5)) {
				nb_tactical++;
		}

		if (f_v) {
			cout << "tdo_refinement::do_all_row_refinements "
					"case " << t << " / " << Output->nb_distributions
					<< " after do_row_refinement, nb_tactical = " << nb_tactical << endl;
		}


	}
	if (f_v) {
		cout << "Case " << label_in << ", found " << Output->nb_distributions
			<< " row refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
	}
	if (f_v) {
		cout << "tdo_refinement::do_all_row_refinements done" << endl;
	}

}

void tdo_refinement::do_all_column_refinements(
		std::string &label_in,
		std::ofstream &ost,
		tdo_refinement_output *Output,
		int &nb_tactical,
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

	for (t = 0; t < Output->nb_distributions; t++) {

		//cout << "tdo_refinement::do_all_column_refinements t=" << t << endl;
		if (f_v) {
			cout << "tdo_refinement::do_all_column_refinements "
					"case " << t << " / " << Output->nb_distributions
					<< " before do_column_refinement" << endl;
		}

		if (do_column_refinement(
				t, label_in, ost,
				Output,
				verbose_level - 5)) {
			nb_tactical++;
		}

		if (f_v) {
			cout << "tdo_refinement::do_all_column_refinements "
					"case " << t << " / " << Output->nb_distributions
					<< " after do_column_refinement, nb_tactical = " << nb_tactical << endl;
		}

	}
	if (f_v) {
		cout << "Case " << label_in << ", found " << Output->nb_distributions
			<< " column refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
		}
	if (f_v) {
		cout << "tdo_refinement::do_all_column_refinements done" << endl;
	}
}


int tdo_refinement::do_row_refinement(
	int t,
	std::string &label_in,
	std::ofstream &ost,
	tdo_refinement_output *Output,
	int verbose_level)
// returns true or false depending on whether the
// refinement has produced a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "tdo_refinement::do_row_refinement t=" << t << endl;
	}

	type_index = NEW_int(Output->nb_types);
	for (i = 0; i < Output->nb_types; i++) {
		type_index[i] = -1;
	}

	new_nb_parts = GP.nb_parts;
	if (Tdo_scheme_synthetic->row_level >= 2) {
		R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	}
	else {
		R = 1;
	}
	i = 0;
	h = 0;
	S = 0;
	for (r = 0; r < R; r++) {
		if (Tdo_scheme_synthetic->row_level >= 2) {
			l = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r];
		}
		else {
			//partitionstack &P = G.PB.P;
			l = Tdo_scheme_synthetic->Partition_refinement->startCell[1];
		}
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
		}
		while (i < Output->nb_types) {
			a = Output->distributions[t * Output->nb_types + i];
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
	if (S != Tdo_scheme_synthetic->m) {
		cout << "tdo_refinement::do_row_refinement: S != Tdo_scheme_synthetic->m" << endl;
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

		G2.init_part_and_entries(GP2.part, GP2.entries, verbose_level - 2);

		G2.row_level = new_nb_parts;
		G2.col_level = Tdo_scheme_synthetic->col_level;
		G2.extra_row_level = Tdo_scheme_synthetic->row_level; // GP.extra_row_level;
		G2.extra_col_level = GP.extra_col_level;
		G2.lambda_level = Tdo_scheme_synthetic->lambda_level;
		G2.level[ROW_SCHEME] = new_nb_parts;
		G2.level[COL_SCHEME] = Tdo_scheme_synthetic->col_level;
		G2.level[EXTRA_ROW_SCHEME] = Tdo_scheme_synthetic->row_level; // G.extra_row_level;
		G2.level[EXTRA_COL_SCHEME] = Tdo_scheme_synthetic->extra_col_level;
		G2.level[LAMBDA_SCHEME] = Tdo_scheme_synthetic->lambda_level;

		G2.init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a scheme of size "
					<< G2.nb_row_classes[ROW_SCHEME] << " x " << G2.nb_col_classes[ROW_SCHEME] << endl;
		}
		for (i = 0; i < G2.nb_row_classes[ROW_SCHEME]; i++) {
			c1 = G2.row_classes[ROW_SCHEME][i];
			for (j = 0; j < Output->type_len; j++) {
				c2 = G2.col_classes[ROW_SCHEME][j];
				idx = type_index[i];
				if (idx == -1) {
					continue;
				}
				a = Output->types[idx * Output->type_len + j];
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

		GP2.label = label_in + "." + std::to_string(t + 1);

		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = new_nb_parts;
		GP2.col_level = Tdo_scheme_synthetic->col_level;
		GP2.lambda_level = Tdo_scheme_synthetic->lambda_level;
		GP2.extra_row_level = Tdo_scheme_synthetic->row_level;
		GP2.extra_col_level = Tdo_scheme_synthetic->extra_col_level;

		GP2.write_mode_stack(ost, GP2.label);


		if (f_vv) {
			cout << GP2.label << " written" << endl;
		}
		if (new_nb_parts == Tdo_scheme_synthetic->col_level) {
			f_tactical = true;
		}
		else {
			f_tactical = false;
		}
	}

	FREE_int(type_index);
	if (f_v) {
		cout << "tdo_refinement::do_row_refinement t=" << t << " done" << endl;
	}
	return f_tactical;
}

int tdo_refinement::do_column_refinement(
	int t,
	std::string &label_in,
	std::ofstream &ost,
	tdo_refinement_output *Output,
	int verbose_level)
// returns true or false depending on whether the
// refinement has produced a tactical decomposition
{
	int r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	int *type_index;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 6);
	int f_vvv = (verbose_level >= 7);
	int f_tactical;

	if (f_v) {
		cout << "tdo_refinement::do_column_refinement t=" << t << endl;
	}

	type_index = NEW_int(Output->nb_types);

	for (i = 0; i < Output->nb_types; i++) {
		type_index[i] = -1;
	}

	new_nb_parts = GP.nb_parts;
	R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	i = 0;
	h = 0;
	S = Tdo_scheme_synthetic->m;

	for (r = 0; r < R; r++) {
		l = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][r];
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
		}
		while (i < Output->nb_types) {
			a = Output->distributions[t * Output->nb_types + i];
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
				Int_vec_print(cout, Output->distributions + t * Output->nb_types, Output->nb_types);
				cout << endl;
				exit(1);
			}
		}
		S += l;
	}
	if (S != Tdo_scheme_synthetic->m + Tdo_scheme_synthetic->n) {
		cout << "tdo_refinement::do_column_refinement: S != Tdo_scheme_synthetic->m + Tdo_scheme_synthetic->n" << endl;
		exit(1);
	}

	new_nb_entries = Tdo_scheme_synthetic->nb_entries;
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

		G2->init_part_and_entries(
				GP2.part, GP2.entries, verbose_level - 2);

		G2->row_level = GP.row_level;
		G2->col_level = new_nb_parts;
		G2->extra_row_level = GP.extra_row_level;
		G2->extra_col_level = GP.col_level; // GP.extra_col_level;
		G2->lambda_level = Tdo_scheme_synthetic->lambda_level;
		G2->level[ROW_SCHEME] = Tdo_scheme_synthetic->row_level;
		G2->level[COL_SCHEME] = new_nb_parts;
		G2->level[EXTRA_ROW_SCHEME] = Tdo_scheme_synthetic->extra_row_level;
		G2->level[EXTRA_COL_SCHEME] = GP.col_level; // G.extra_col_level;
		G2->level[LAMBDA_SCHEME] = Tdo_scheme_synthetic->lambda_level;

		G2->init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a scheme of size "
					<< G2->nb_row_classes[COL_SCHEME] << " x " << G2->nb_col_classes[COL_SCHEME] << endl;
		}
		for (i = 0; i < G2->nb_row_classes[COL_SCHEME]; i++) {
			c1 = G2->row_classes[COL_SCHEME][i];
			for (j = 0; j < G2->nb_col_classes[COL_SCHEME]; j++) {
				c2 = G2->col_classes[COL_SCHEME][j];
				idx = type_index[j];
				if (idx == -1) {
					continue;
				}
				a = Output->types[idx * Output->type_len + i];
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

		GP2.label = label_in + "." + std::to_string(t + 1);

		GP2.nb_parts = new_nb_parts;
		GP2.nb_entries = new_nb_entries;
		GP2.row_level = Tdo_scheme_synthetic->row_level;
		GP2.col_level = new_nb_parts;
		GP2.lambda_level = Tdo_scheme_synthetic->lambda_level;
		GP2.extra_row_level = Tdo_scheme_synthetic->extra_row_level;
		GP2.extra_col_level = Tdo_scheme_synthetic->col_level;

		GP2.write_mode_stack(ost, GP2.label);


		if (f_vv) {
			cout << GP2.label << " written" << endl;
		}
		if (new_nb_parts == Tdo_scheme_synthetic->row_level) {
			f_tactical = true;
		}
		else {
			f_tactical = false;
		}
		FREE_OBJECT(G2);
	}

	FREE_int(type_index);
	if (f_v) {
		cout << "tdo_refinement::do_column_refinement t=" << t << " done" << endl;
	}
	return f_tactical;
}



}}}}




