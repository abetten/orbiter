// tdo_scheme_synthetic.cpp
//
// Anton Betten 8/27/07
//
//
// refine_rows and refine_columns started: December 2006
// moved away from inc_gen: 8/27/07
// separated out from refine: 1/24/08
// corrected a memory problem in refine_rows_hard: Dec 6 2010


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {


tdo_scheme_synthetic::tdo_scheme_synthetic()
{
	Record_birth();
	int i;
	
	Descr = NULL;

	Partition_refinement = NULL;

	part_length = 0;
	part = NULL;
	nb_entries = 0;
	entries = NULL;

	row_level = 0;
	col_level = 0;
	lambda_level = 0;
	extra_row_level = 0;
	extra_col_level = 0;
	mn = 0;
	m = 0;
	n = 0;
	for (i = 0; i < NUMBER_OF_SCHEMES; i++) {
		row_classes[i] = NULL;
		col_classes[i] = NULL;
		row_class_index[i] = NULL;
		col_class_index[i] = NULL;
		row_classes_first[i] = NULL;
		row_classes_len[i] = NULL;
		row_class_no[i] = NULL;
		col_classes_first[i] = NULL;
		col_classes_len[i] = NULL;
		col_class_no[i] = NULL;
	}
	the_row_scheme = NULL;
	the_col_scheme = NULL;
	the_extra_row_scheme = NULL;
	the_extra_col_scheme = NULL;
	the_row_scheme_cur = NULL;
	the_col_scheme_cur = NULL;
	the_extra_row_scheme_cur = NULL;
	the_extra_col_scheme_cur = NULL;

}

tdo_scheme_synthetic::~tdo_scheme_synthetic()
{
	Record_death();
	int i;
	
	if (part) {
		FREE_int(part);
		part = NULL;
	}
	if (entries) {
		FREE_int(entries);
		entries = NULL;
	}
	for (i = 0; i < NUMBER_OF_SCHEMES; i++) {
		free_partition(i);
	}
	if (the_row_scheme) {
		FREE_int(the_row_scheme);
		the_row_scheme = NULL;
	}
	if (the_col_scheme) {
		FREE_int(the_col_scheme);
		the_col_scheme = NULL;
	}
	if (the_extra_row_scheme) {
		FREE_int(the_extra_row_scheme);
		the_extra_row_scheme = NULL;
	}
	if (the_extra_col_scheme) {
		FREE_int(the_extra_col_scheme);
		the_extra_col_scheme = NULL;
	}
	if (the_row_scheme_cur) {
		FREE_int(the_row_scheme_cur);
		the_row_scheme_cur = NULL;
	}
	if (the_col_scheme_cur) {
		FREE_int(the_col_scheme_cur);
		the_col_scheme_cur = NULL;
	}
	if (the_extra_row_scheme_cur) {
		FREE_int(the_extra_row_scheme_cur);
		the_extra_row_scheme_cur = NULL;
	}
	if (the_extra_col_scheme_cur) {
		FREE_int(the_extra_col_scheme_cur);
		the_extra_col_scheme_cur = NULL;
	}
	if (Partition_refinement) {
		FREE_OBJECT(Partition_refinement);
		Partition_refinement = NULL;
	}
}

void tdo_scheme_synthetic::init(
		tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "tdo_scheme_synthetic::init" << endl;
	}

	tdo_scheme_synthetic::Descr = Descr;

	if (f_v) {
		cout << "tdo_scheme_synthetic::init done" << endl;
	}
}

void tdo_scheme_synthetic::check_init()
{
	if (Descr == NULL) {
		cout << "tdo_scheme_synthetic::check_init "
				"init has not yet been called" << endl;
		exit(1);
	}
}


void tdo_scheme_synthetic::init_part_and_entries(
	int *Part, int *Entries, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "tdo_scheme_synthetic::init_part_and_entries" << endl;
	}
	int i;

	for (part_length = 0; ; part_length++) {
		if (Part[part_length] == -1) {
			break;
		}
	}
	if (f_v) {
		cout << "partition of length " << part_length << endl;
	}
	
	for (nb_entries = 0; ; nb_entries++) {
		if (Entries[4 * nb_entries + 0] == -1) {
			break;
		}
	}
	if (f_v) {
		cout << "nb_entries = " << nb_entries << endl;
	}

	if (part) {
		FREE_int(part);
	}
	if (entries) {
		FREE_int(entries);
	}
	part = NEW_int(part_length + 1);
	Int_vec_copy(Part, part, part_length + 1);

	entries = NEW_int(4 * nb_entries + 1);
	Int_vec_copy(Entries, entries, 4 * nb_entries + 1);

	if (f_v) {
		cout << "tdo_scheme_synthetic::init_part_and_entries done" << endl;
	}
}


void tdo_scheme_synthetic::init_TDO(
		int *Part, int *Entries,
	int Row_level, int Col_level,
	int Extra_row_level, int Extra_col_level,
	int Lambda_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme::init_TDO" << endl;
	}

	init_part_and_entries(Part, Entries, verbose_level);
	if (f_vv) {
		cout << "partition of length " << part_length << endl;
	}
	if (f_vv) {
		cout << "nb_entries = " << nb_entries << endl;
	}

	row_level = Row_level;
	col_level = Col_level;
	extra_row_level = Extra_row_level;
	extra_col_level = Extra_col_level;
	lambda_level = Lambda_level;
	if (f_vvv) {
		cout << "row_level = " << row_level << endl;
		cout << "col_level = " << col_level << endl;
		cout << "extra_row_level = " << extra_row_level << endl;
		cout << "extra_col_level = " << extra_col_level << endl;
		cout << "lambda_level = " << lambda_level << endl;
	}
	level[ROW_SCHEME] = row_level;
	level[COL_SCHEME] = col_level;
	level[EXTRA_ROW_SCHEME] = extra_row_level;
	level[EXTRA_COL_SCHEME] = extra_col_level;
	level[LAMBDA_SCHEME] = lambda_level;

	init_partition_stack(verbose_level - 2);
	
	//cout << "after init_partition_stack" << endl;
	
	//print_row_test_data();
	
}

void tdo_scheme_synthetic::exit_TDO()
{
	exit_partition_stack();
	
	if (the_row_scheme_cur) {
		FREE_int(the_row_scheme_cur);
		the_row_scheme_cur = NULL;
	}
	if (the_col_scheme_cur) {
		FREE_int(the_col_scheme_cur);
		the_col_scheme_cur = NULL;
	}
	if (the_extra_row_scheme_cur) {
		FREE_int(the_extra_row_scheme_cur);
		the_extra_row_scheme_cur = NULL;
	}
	if (the_extra_col_scheme_cur) {
		FREE_int(the_extra_col_scheme_cur);
		the_extra_col_scheme_cur = NULL;
	}
}

void tdo_scheme_synthetic::init_partition_stack(
		int verbose_level)
{
	int k, at, f, c, l, i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::init_partition_stack" << endl;
	}
	if (f_vv) {
		cout << "tdo_scheme_synthetic::init_partition_stack part_length=" << part_length << endl;
		cout << "tdo_scheme_synthetic::init_partition_stack row_level=" << row_level << endl;
		cout << "tdo_scheme_synthetic::init_partition_stack col_level=" << col_level << endl;
		cout << "tdo_scheme_synthetic::init_partition_stack verbose_level=" << verbose_level << endl;
	}
	mn = part[0];
	m = part[1];
	n = mn - m;
	if (part_length < 2) {
		cout << "tdo_scheme_synthetic::init_partition_stack part_length < 2" << endl;
		exit(1);
	}
	if (f_vvv) {
		cout << "tdo_scheme_synthetic::init_partition_stack: m=" << m << " n=" << n << endl;
		Int_vec_print(cout, part, part_length + 1);
		cout << endl;
	}
	
	Partition_refinement = NEW_OBJECT(other::data_structures::partitionstack);
	Partition_refinement->allocate(m + n, 0 /* verbose_level */);
	//PB.init_partition_backtrack_basic(m, n, verbose_level - 10);
	if (f_vvv) {
		cout << "tdo_scheme_synthetic::init_partition_stack after P->allocate" << endl;
	}

	//partitionstack &P = PB.P;

	if (f_vvv) {
		cout << "tdo_scheme_synthetic::init_partition_stack initial Partition_refinement: " << endl;
		Partition_refinement->print(cout);
	}
	for (k = 1; k < part_length; k++) {
		at = part[k];
		c = Partition_refinement->cellNumber[at];
		f = Partition_refinement->startCell[c];
		l = Partition_refinement->cellSize[c];
		if (f_vvv) {
			cout << "tdo_scheme_synthetic::init_partition_stack part[" << k << "]=" << at << endl;
			cout << "P->cellNumber[at]=" << c << endl;
			cout << "P->startCell[c]=" << f << endl;
			cout << "P->cellSize[c]=" << l << endl;
			cout << "f + l - at=" << f + l - at << endl;
		}
		Partition_refinement->subset_contiguous(at, f + l - at);
		Partition_refinement->split_cell(false);
		if (f_vvv) {
			cout << "tdo_scheme_synthetic::init_partition_stack after splitting at " << at << endl;
			Partition_refinement->print(cout);
		}
		if (Partition_refinement->ht == row_level) {
			l = Partition_refinement->ht;
			if (the_row_scheme) {
				FREE_int(the_row_scheme);
				the_row_scheme = NULL;
			}
			the_row_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_row_scheme[i] = -1;
			}
			get_partition(ROW_SCHEME, l, verbose_level - 3);
			get_row_or_col_scheme(ROW_SCHEME, l, verbose_level - 3);
		}
			
		if (Partition_refinement->ht == col_level) {
			l = Partition_refinement->ht;
			if (the_col_scheme) {
				FREE_int(the_col_scheme);
				the_col_scheme = NULL;
			}
			the_col_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_col_scheme[i] = -1;
			}
			get_partition(COL_SCHEME, l, verbose_level - 3);
			get_row_or_col_scheme(COL_SCHEME, l, verbose_level - 3);
		}
			
		if (Partition_refinement->ht == extra_row_level) {
			l = Partition_refinement->ht;
			if (the_extra_row_scheme) {
				FREE_int(the_extra_row_scheme);
				the_extra_row_scheme = NULL;
			}
			the_extra_row_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_extra_row_scheme[i] = -1;
			}
			get_partition(EXTRA_ROW_SCHEME, l, verbose_level - 3);
			get_row_or_col_scheme(EXTRA_ROW_SCHEME, l, verbose_level - 3);
		}
			
		if (Partition_refinement->ht == extra_col_level) {
			l = Partition_refinement->ht;
			if (the_extra_col_scheme) {
				FREE_int(the_extra_col_scheme);
				the_extra_col_scheme = NULL;
			}
			the_extra_col_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_extra_col_scheme[i] = -1;
			}
			get_partition(EXTRA_COL_SCHEME, l, verbose_level - 3);
			get_row_or_col_scheme(EXTRA_COL_SCHEME, l, verbose_level - 3);
		}
			
		if (Partition_refinement->ht == lambda_level) {
			l = Partition_refinement->ht;
			get_partition(LAMBDA_SCHEME, l, verbose_level - 3);
		}
			
	} // next k
	
	if (f_vvv) {
		cout << "tdo_scheme_synthetic::init_partition_stack before complete_partition_info" << endl;
	}
	if (row_level >= 2) {
		complete_partition_info(ROW_SCHEME, 0/*verbose_level*/);
	}
	if (col_level >= 2) {
		complete_partition_info(COL_SCHEME, 0/*verbose_level*/);
	}
	if (extra_row_level >= 2) {
		complete_partition_info(EXTRA_ROW_SCHEME, 0/*verbose_level*/);
	}
	if (extra_col_level >= 2 && extra_col_level < part_length) {
		complete_partition_info(EXTRA_COL_SCHEME, 0/*verbose_level*/);
	}
	complete_partition_info(LAMBDA_SCHEME, 0/*verbose_level*/);
	
	if (f_vv) {
		if (row_level >= 2) {
			print_scheme(ROW_SCHEME, false);
		}
		if (col_level >= 2) {
			print_scheme(COL_SCHEME, false);
		}
		if (extra_row_level >= 2) {
			print_scheme(EXTRA_ROW_SCHEME, false);
		}
		if (extra_col_level >= 2) {
			print_scheme(EXTRA_COL_SCHEME, false);
		}
		print_scheme(LAMBDA_SCHEME, false);
	}
}

void tdo_scheme_synthetic::exit_partition_stack()
{
	if (the_row_scheme) {
		FREE_int(the_row_scheme);
		the_row_scheme = NULL;
	}
	if (the_col_scheme) {
		FREE_int(the_col_scheme);
		the_col_scheme = NULL;
	}
	if (the_extra_row_scheme) {
		FREE_int(the_extra_row_scheme);
		the_extra_row_scheme = NULL;
	}
	if (the_extra_col_scheme) {
		FREE_int(the_extra_col_scheme);
		the_extra_col_scheme = NULL;
	}
	free_partition(ROW_SCHEME);
	free_partition(COL_SCHEME);
	//if (extra_row_level >= 0)
		free_partition(EXTRA_ROW_SCHEME);
	//if (extra_col_level >= 0)
		free_partition(EXTRA_COL_SCHEME);
	free_partition(LAMBDA_SCHEME);

}

void tdo_scheme_synthetic::get_partition(
		int h, int l, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 10);
	int i;
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_partition h=" << h << " l=" << l
			<< " m=" << m << " n=" << n << endl;
	}
	if (l < 0) {
		cout << "tdo_scheme_synthetic::get_partition l is negative" << endl;
		exit(1);
	}
	free_partition(h);
	row_classes[h] = NEW_int(l);
	col_classes[h] = NEW_int(l);
	row_class_index[h] = NEW_int(l);
	col_class_index[h] = NEW_int(l);
	row_classes_first[h] = NEW_int(l);
	row_classes_len[h] = NEW_int(l);
	col_classes_first[h] = NEW_int(l);
	col_classes_len[h] = NEW_int(l);
	row_class_no[h] = NEW_int(m);
	col_class_no[h] = NEW_int(n);

	for (i = 0; i < l; i++) {
		row_class_index[h][i] = -1;
		col_class_index[h][i] = -1;
	}
			
	Partition_refinement->get_row_and_col_classes_old_fashioned(
			row_classes[h], nb_row_classes[h],
		col_classes[h], nb_col_classes[h],
		verbose_level - 1);
				
	for (i = 0; i < nb_row_classes[h]; i++) {
		row_class_index[h][row_classes[h][i]] = i;
	}
	for (i = 0; i < nb_col_classes[h]; i++) {
		col_class_index[h][col_classes[h][i]] = i;
	}


	if (f_vv) {
		for (i = 0; i < nb_row_classes[h]; i++) {
			cout << "row_class_index[h][" << row_classes[h][i] << "] = "
			<< row_class_index[h][row_classes[h][i]] << endl;
		}
		for (i = 0; i < nb_col_classes[h]; i++) {
			cout << "col_class_index[h][" << col_classes[h][i] << "] = "
			<< col_class_index[h][col_classes[h][i]] << endl;
		}
	}
	if (f_vv) {
		cout << "nb_row_classes[h]=" << nb_row_classes[h] << endl;
		cout << "nb_col_classes[h]=" << nb_col_classes[h] << endl;
	}
}

void tdo_scheme_synthetic::free_partition(
		int i)
{
	if (row_classes[i]) {
		FREE_int(row_classes[i]);
		row_classes[i] = NULL;
	}
	if (col_classes[i]) {
		FREE_int(col_classes[i]);
		col_classes[i] = NULL;
	}
	if (row_class_index[i]) {
		FREE_int(row_class_index[i]);
		row_class_index[i] = NULL;
	}
	if (col_class_index[i]) {
		FREE_int(col_class_index[i]);
		col_class_index[i] = NULL;
	}
	if (row_classes_first[i]) {
		FREE_int(row_classes_first[i]);
		row_classes_first[i] = NULL;
	}
	if (row_classes_len[i]) {
		FREE_int(row_classes_len[i]);
		row_classes_len[i] = NULL;
	}
	if (row_class_no[i]) {
		FREE_int(row_class_no[i]);
		row_class_no[i] = NULL;
	}
	if (col_classes_first[i]) {
		FREE_int(col_classes_first[i]);
		col_classes_first[i] = NULL;
	}
	if (col_classes_len[i]) {
		FREE_int(col_classes_len[i]);
		col_classes_len[i] = NULL;
	}
	if (col_class_no[i]) {
		FREE_int(col_class_no[i]);
		col_class_no[i] = NULL;
	}
}

void tdo_scheme_synthetic::complete_partition_info(
		int h, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::complete_partition_info h=" << h << endl;
		cout << "# of row classes = " << nb_row_classes[h] << endl;
		cout << "# of col classes = " << nb_col_classes[h] << endl;
	}

	int f, i, j, c1, S, k;

	f = 0;
	for (i = 0; i < nb_row_classes[h]; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
		}
		c1 = row_classes[h][i];
		if (f_vv) {
			cout << "c1=" << c1 << endl;
		}
		S = Partition_refinement->cellSizeAtLevel(c1, level[h]);
		if (f_vv) {
			cout << "S=" << S << endl;
		}
		row_classes_first[h][i] = f;
		row_classes_len[h][i] = S;
		for (k = 0; k < S; k++) {
			row_class_no[h][f + k] = i;
			if (f_vv) {
				cout << "row_class_no[h][" << f + k << "]="
					<< row_class_no[h][f + k] << endl;
			}
		}
		f += S;
	}
	f = 0;
	for (j = 0; j < nb_col_classes[h]; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
		}
		c1 = col_classes[h][j];
		if (f_vv) {
			cout << "c1=" << c1 << endl;
		}
		S = Partition_refinement->cellSizeAtLevel(c1, level[h]);
		if (f_vv) {
			cout << "S=" << S << endl;
		}
		col_classes_first[h][j] = f;
		col_classes_len[h][j] = S;
		for (k = 0; k < S; k++) {
			col_class_no[h][f + k] = j;
			if (f_vv) {
				cout << "col_class_no[h][" << f + k << "]="
					<< col_class_no[h][f + k] << endl;
			}
		}
		f += S;
	}
}

void tdo_scheme_synthetic::get_row_or_col_scheme(
		int h, int l, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_row_or_col_scheme" << endl;
	}

	int i, d, c1, c2, s1, s2, v;

	for (i = 0; i < nb_entries; i++) {
		d = entries[i * 4 + 0];
		c1 = entries[i * 4 + 1];
		c2 = entries[i * 4 + 2];
		v = entries[i * 4 + 3];
		if (d == l) {
			//cout << "entry " << i << " : " << d << " "
			//<< c1 << " " << c2 << " " << v << endl;
			if (h == ROW_SCHEME && row_class_index[h][c1] >= 0) {
				// row scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_row_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_row_scheme[s1 * nb_col_classes[h] + s2] = v;
			}
			else if (h == COL_SCHEME && col_class_index[h][c1] >= 0) {
				// col scheme
				s1 = row_class_index[h][c2];
				s2 = col_class_index[h][c1];
				//cout << "the_col_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_col_scheme[s1 * nb_col_classes[h] + s2] = v;
			}
			else if (h == EXTRA_ROW_SCHEME && row_class_index[h][c1] >= 0) {
				// col scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_extra_row_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_extra_row_scheme[s1 * nb_col_classes[h] + s2] = v;
			}
			else if (h == EXTRA_COL_SCHEME && col_class_index[h][c1] >= 0) {
				// col scheme
				s1 = row_class_index[h][c2];
				s2 = col_class_index[h][c1];
				//cout << "EXTRA_COL:" << endl;
				//cout << "c1=" << c1 << endl;
				//cout << "c2=" << c2 << endl;
				//cout << "s1=" << s1 << endl;
				//cout << "s2=" << s2 << endl;
				//cout << "the_extra_col_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + "
				//<< s2 << "] = " << v << endl;
				the_extra_col_scheme[s1 * nb_col_classes[h] + s2] = v;
			}
			//print_row_test_data();
		} // if
	} // next i
	if (h == ROW_SCHEME) {
		if (the_row_scheme_cur) {
			FREE_int(the_row_scheme_cur);
			the_row_scheme_cur = NULL;
		}
		the_row_scheme_cur = NEW_int(m * nb_col_classes[h]);
		Int_vec_zero(the_row_scheme_cur, m * nb_col_classes[h]);
		//print_row_test_data();
	}
	if (h == COL_SCHEME) {
		if (the_col_scheme_cur) {
			FREE_int(the_col_scheme_cur);
			the_col_scheme_cur = NULL;
		}
		the_col_scheme_cur = NEW_int(n * nb_row_classes[h]);
		Int_vec_zero(the_col_scheme_cur, n * nb_row_classes[h]);
	}
	if (h == EXTRA_ROW_SCHEME) {
		if (the_extra_row_scheme_cur) {
			FREE_int(the_extra_row_scheme_cur);
			the_extra_row_scheme_cur = NULL;
		}
		the_extra_row_scheme_cur = NEW_int(m * nb_col_classes[h]);
		Int_vec_zero(the_extra_row_scheme_cur, m * nb_col_classes[h]);
	}
	if (h == EXTRA_COL_SCHEME) {
		if (the_extra_col_scheme_cur) {
			FREE_int(the_extra_col_scheme_cur);
			the_extra_col_scheme_cur = NULL;
		}
		the_extra_col_scheme_cur = NEW_int(n * nb_row_classes[h]);
		Int_vec_zero(the_extra_col_scheme_cur, n * nb_row_classes[h]);
	}
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_row_or_col_scheme finished" << endl;
	}
}

other::data_structures::partitionstack *tdo_scheme_synthetic::get_column_split_partition(
		int verbose_level)
// this function computes the column split partition based on Partition_refinement
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_column_split_partition" << endl;
	}

	int i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;

	other::data_structures::partitionstack *Col_split;

	Col_split = NEW_OBJECT(other::data_structures::partitionstack);


	R = nb_row_classes[ROW_SCHEME];
	l1 = nb_col_classes[ROW_SCHEME];
	l2 = nb_col_classes[COL_SCHEME];
	if (false) {
		cout << "l1=" << l1 << " at level " << level[ROW_SCHEME] << endl;
		cout << "l2=" << l2 << " at level " << level[COL_SCHEME] << endl;
		cout << "R=" << R << endl;
	}
	Col_split->allocate(l2, false);
	for (i = 0; i < l1; i++) {
		ci = col_classes[ROW_SCHEME][i];
		j1 = col_class_index[COL_SCHEME][ci];
		cc = Col_split->cellNumber[j1];
		f = Col_split->startCell[cc];
		l = Col_split->cellSize[cc];
		if (false) {
			cout << "i=" << i << " ci=" << ci << " j1=" << j1
					<< " cc=" << cc << endl;
		}
		Col_split->subset_size = 0;
		for (h = 0; h < l; h++) {
			j = Col_split->pointList[f + h];
			cj = col_classes[COL_SCHEME][j];
			if (false) {
				cout << "j=" << j << " cj=" << cj << endl;
			}
			if (!Partition_refinement->is_descendant_of_at_level(
					cj, ci,
					level[ROW_SCHEME], false)) {
				if (false) {
					cout << j << "/" << cj << " is not a "
							"descendant of " << i << "/" << ci << endl;
				}
				Col_split->subset[Col_split->subset_size++] = j;
			}
		}
		if (false) {
			cout << "non descendants of " << i << "/" << ci << " : ";
			other::orbiter_kernel_system::Orbiter->Int_vec->set_print(cout, Col_split->subset, Col_split->subset_size);
			cout << endl;
		}
		if (Col_split->subset_size > 0) {
			Col_split->split_cell(false);
			if (false) {
				Col_split->print(cout);
			}
		}
	}
	if (f_vv) {
		cout << "tdo_scheme_synthetic::get_column_split_partition column-split partition:" << endl;
		Col_split->print(cout);
	}
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_column_split_partition done" << endl;
	}
	return Col_split;
}

other::data_structures::partitionstack *tdo_scheme_synthetic::get_row_split_partition(
		int verbose_level)
// this function computes the row split partition based on Partition_refinement
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_row_split_partition" << endl;
	}

	int i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;

	other::data_structures::partitionstack *Row_split;

	Row_split = NEW_OBJECT(other::data_structures::partitionstack);


	R = nb_col_classes[COL_SCHEME];
	l1 = nb_row_classes[COL_SCHEME];
	l2 = nb_row_classes[ROW_SCHEME];
	if (false) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "R=" << R << endl;
	}
	Row_split->allocate(l2, false);
	for (i = 0; i < l1; i++) {
		ci = row_classes[COL_SCHEME][i];
		j1 = row_class_index[ROW_SCHEME][ci];
		cc = Row_split->cellNumber[j1];
		f = Row_split->startCell[cc];
		l = Row_split->cellSize[cc];
		if (false) {
			cout << "i=" << i << " ci=" << ci << " j1=" << j1
				<< " cc=" << cc << endl;
		}
		Row_split->subset_size = 0;
		for (h = 0; h < l; h++) {
			j = Row_split->pointList[f + h];
			cj = row_classes[ROW_SCHEME][j];
			if (false) {
				cout << "j=" << j << " cj=" << cj << endl;
			}
			if (!Partition_refinement->is_descendant_of_at_level(
					cj, ci,
					level[COL_SCHEME], false)) {
				if (false) {
					cout << j << "/" << cj << " is not a descendant "
						"of " << i << "/" << ci << endl;
				}
				Row_split->subset[Row_split->subset_size++] = j;
			}
			else {
				if (false) {
					cout << cj << " is a descendant of " << ci << endl;
				}
			}
		}
		if (false) {
			cout << "non descendants of " << i << "/" << ci << " : ";
			other::orbiter_kernel_system::Orbiter->Int_vec->set_print(
					cout, Row_split->subset, Row_split->subset_size);
			cout << endl;
		}
		if (Row_split->subset_size > 0) {
			Row_split->split_cell(false);
			if (false) {
				Row_split->print(cout);
			}
		}
	}
	if (f_vv) {
		cout << "tdo_scheme_synthetic::get_row_split_partition row-split partition:" << endl;
		Row_split->print(cout);
	}
	if (f_v) {
		cout << "tdo_scheme_synthetic::get_row_split_partition done" << endl;
	}
	return Row_split;
}

void tdo_scheme_synthetic::print_all_schemes()
{
	if (lambda_level >= 2) {
		print_scheme(LAMBDA_SCHEME, false);
	}
	if (extra_row_level >= 2) {
		print_scheme(EXTRA_ROW_SCHEME, false);
	}
	if (extra_col_level >= 2) {
		print_scheme(EXTRA_COL_SCHEME, false);
	}
	if (row_level >= 2) {
		print_scheme(ROW_SCHEME, false);
	}
	if (col_level >= 2) {
		print_scheme(COL_SCHEME, false);
	}
}

void tdo_scheme_synthetic::print_scheme(
		int h, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, c1, c2, a = 0;
	
	if (h == ROW_SCHEME) {
		cout << "row_scheme at level " << level[h] << " : " << endl;
	}
	else if (h == COL_SCHEME) {
		cout << "col_scheme at level " << level[h] << " : " << endl;
	}
	else if (h == EXTRA_ROW_SCHEME) {
		cout << "extra_row_scheme at level " << level[h] << " : " << endl;
	}
	else if (h == EXTRA_COL_SCHEME) {
		cout << "extra_col_scheme at level " << level[h] << " : " << endl;
	}
	else if (h == LAMBDA_SCHEME) {
		cout << "lambda_scheme at level " << level[h] << " : " << endl;
	}
	cout << "is " << nb_row_classes[h] << " x "
		<< nb_col_classes[h] << endl;
	cout << "          | ";
	for (j = 0; j < nb_col_classes[h]; j++) {
		c2 = col_classes[h][j];
		cout << setw(3) << col_classes_len[h][j]
			<< "_{" << setw(3) << c2 << "}";
	}
	cout << endl;
	cout << "============";
	for (j = 0; j < nb_col_classes[h]; j++) {
		cout << "=========";
	}
	cout << endl;
	for (i = 0; i < nb_row_classes[h]; i++) {
		c1 = row_classes[h][i];
		cout << setw(3) << row_classes_len[h][i] << "_{"
			<< setw(3) << c1 << "} | ";
		if (h != LAMBDA_SCHEME) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				if (h == ROW_SCHEME) {
					a = the_row_scheme[i * nb_col_classes[h] + j];
				}
				else if (h == COL_SCHEME) {
					a = the_col_scheme[i * nb_col_classes[h] + j];
				}
				else if (h == EXTRA_ROW_SCHEME) {
					a = the_extra_row_scheme[i * nb_col_classes[h] + j];
				}
				else if (h == EXTRA_COL_SCHEME) {
					a = the_extra_col_scheme[i * nb_col_classes[h] + j];
				}
				
				cout << setw(9) << a;
			}
		}
		cout << endl;
	}
	cout << endl;
	if (f_v) {
		cout << "row_classes_first / len:" << endl;
		for (i = 0; i < nb_row_classes[h]; i++) {
			cout << i << " : " << row_classes_first[h][i] << " : "
				<< row_classes_len[h][i] << endl;
		}
		cout << "class_no:" << endl;
		for (i = 0; i < m; i++) {
			cout << i << " : " << row_class_no[h][i] << endl;
		}
		cout << "col_classes first / len:" << endl;
		for (i = 0; i < nb_col_classes[h]; i++) {
			cout << i << " : " << col_classes_first[h][i] << " : "
				<< col_classes_len[h][i] << endl;
		}
		cout << "col_class_no:" << endl;
		for (i = 0; i < n; i++) {
			cout << i << " : " << col_class_no[h][i] << endl;
		}
	}
}

void tdo_scheme_synthetic::print_scheme_tex(
		std::ostream &ost, int h)
{
	std::string dummy;

	print_scheme_tex_fancy(ost, h, false, true, dummy);
}

void tdo_scheme_synthetic::print_scheme_tex_fancy(
		std::ostream &ost,
	int h, int f_label, int f_subscripts, std::string &label)
{
	int i, j, a = 0, n, m, c1, c2;
	
	n = nb_row_classes[h];
	m = nb_col_classes[h];
	ost << "$$" << endl;
	ost << "\\begin{array}{r|*{" << m << "}{r}}" << endl;
	if (f_label) {
		ost << "\\multicolumn{" << m + 1 << "}{c}{\\mbox{" << label << "}}\\\\" << endl;
	}
	if (h == ROW_SCHEME || h == EXTRA_ROW_SCHEME) {
		ost << "\\rightarrow";
	}
	else if (h == COL_SCHEME || h == EXTRA_COL_SCHEME) {
		ost << "\\downarrow";
	}
	else if (h == LAMBDA_SCHEME) {
		ost << "\\lambda";
	}
	for (j = 0; j < m; j++) {
		c2 = col_classes[h][j];
		ost << " & " << setw(3) << col_classes_len[h][j];
		if (f_subscripts) {
			ost << "_{" << setw(3) << c2 << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < n; i++) {
		c1 = row_classes[h][i];
		ost << row_classes_len[h][i];
		if (f_subscripts) {
			ost << "_{" << setw(3) << c1 << "}";
		}
		for (j = 0; j < m; j++) {
			if (h == ROW_SCHEME) {
				a = the_row_scheme[i * m + j];
			}
			else if (h == COL_SCHEME) {
				a = the_col_scheme[i * m + j];
			}
			else if (h == EXTRA_ROW_SCHEME) {
				a = the_extra_row_scheme[i * m + j];
			}
			else if (h == EXTRA_COL_SCHEME) {
				a = the_extra_col_scheme[i * m + j];
			}
			ost << " & " << setw(3) << a;
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << endl;
}

void tdo_scheme_synthetic::compute_whether_first_inc_must_be_moved(
	int *f_first_inc_must_be_moved, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::compute_whether_first_inc_must_be_moved" << endl;
	}

	int i, j, ii, fi, fii, fj, row_cell0, row_cell, col_cell, a, b, c;

	for (i = 0; i < nb_row_classes[ROW_SCHEME]; i++) {
		f_first_inc_must_be_moved[i] = true;
		if (col_level < 2) {
			continue;
		}
		fi = row_classes_first[ROW_SCHEME][i];
		row_cell0 = row_class_no[COL_SCHEME][fi];
		for (j = 0; j < nb_col_classes[ROW_SCHEME]; j++) {
			a = the_row_scheme[i * nb_col_classes[ROW_SCHEME] + j];
			if (a > 0) {
				break;
			}
		}
		
		if (f_vv) {
			cout << "considering whether incidence in block " << i << ","
				<< j << " must be moved" << endl;
		}
		
		fj = col_classes_first[COL_SCHEME][j];
		col_cell = col_class_no[COL_SCHEME][fj];
		c = the_col_scheme[row_cell0 * nb_col_classes[COL_SCHEME] + col_cell];
		if (f_vvv) {
			cout << "c=" << c << endl;
		}
		if (c >= 0) {
			if (f_vvv) {
				cout << "looking at COL scheme:" << endl;
			}
			f_first_inc_must_be_moved[i] = false;
			for (ii = i + 1; ii < nb_row_classes[ROW_SCHEME]; ii++) {
				b = the_row_scheme[ii * nb_col_classes[ROW_SCHEME] + j];
				fii = row_classes_first[ROW_SCHEME][ii];
				row_cell = row_class_no[COL_SCHEME][fii];
				if (row_cell != row_cell0) {
					if (f_vvv) {
						cout << "i=" << i << " ii=" << ii
							<< " different "
							"COL fuse, hence it must not "
							"be moved" << endl;
						cout << "fi=" << fi << endl;
						cout << "fii=" << fii << endl;
						cout << "row_cell0=" << row_cell0 << endl;
						cout << "row_cell=" << row_cell << endl;
					}
					f_first_inc_must_be_moved[i] = false;
					//ii = nb_row_classes[ROW];
					break;
				}
				if (b) {
					if (f_vvv) {
						cout << "ii=" << ii << " seeing non zero entry "
							<< b << ", hence it must be moved" << endl;
					}
					f_first_inc_must_be_moved[i] = true;
					break;
				}
			} // next ii
		}
		else {
			if (f_vvv) {
				cout << "looking at EXTRA_COL scheme:" << endl;
			}
			fi = row_classes_first[ROW_SCHEME][i];
			row_cell0 = row_class_no[EXTRA_COL_SCHEME][fi];
			if (f_vvv) {
				cout << "row_cell0=" << row_cell0 << endl;
			}
			for (ii = i + 1; ii < nb_row_classes[ROW_SCHEME]; ii++) {
				b = the_row_scheme[ii * nb_col_classes[ROW_SCHEME] + j];
				fii = row_classes_first[ROW_SCHEME][ii];
				row_cell = row_class_no[EXTRA_COL_SCHEME][fii];
				if (row_cell != row_cell0) {
					if (f_vvv) {
						cout << "i=" << i << " ii=" << ii
							<< " different "
							"EXTRACOL fuse, hence it must "
							"not be moved" << endl;
						cout << "fi=" << fi << endl;
						cout << "fii=" << fii << endl;
						cout << "row_cell0=" << row_cell0 << endl;
						cout << "row_cell=" << row_cell << endl;
					}
					f_first_inc_must_be_moved[i] = false;
					//ii = nb_row_classes[ROW];
					break;
				}
				if (b) {
					if (f_vvv) {
						cout << "ii=" << ii << " seeing non zero entry "
							<< b << ", hence it must be moved" << endl;
					}
					f_first_inc_must_be_moved[i] = true;
					break;
				}
			} // next ii
		}
		
	}
	if (f_v) {
		cout << "tdo_scheme_synthetic::compute_whether_first_inc_must_be_moved done" << endl;
	}
}

int tdo_scheme_synthetic::count_nb_inc_from_row_scheme(
		int verbose_level)
{
	int f_v = (verbose_level > 1);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::count_nb_inc_from_row_scheme" << endl;
	}

	int i, j, a, b = 0, nb_inc;

	nb_inc = 0;
	for (i = 0; i < nb_row_classes[ROW_SCHEME]; i++) {
		for (j = 0; j < nb_col_classes[ROW_SCHEME]; j++) {
			a = the_row_scheme[i * nb_col_classes[ROW_SCHEME] + j];
			if (a == -1) {
				cout << "incomplete row_scheme" << endl;
				cout << "i=" << i << "j=" << j << endl;
				cout << "ignoring this" << endl;
			}
			else {
				b = a * row_classes_len[ROW_SCHEME][i];
			}
			nb_inc += b;
		}
	}
	if (f_v) {
		cout << "tdo_scheme_synthetic::count_nb_inc_from_row_scheme done nb_inc = " << nb_inc << endl;
	}
	return nb_inc;
}

int tdo_scheme_synthetic::count_nb_inc_from_extra_row_scheme(
		int verbose_level)
{
	int f_v = (verbose_level > 1);
	
	if (f_v) {
		cout << "tdo_scheme_synthetic::count_nb_inc_from_extra_row_scheme" << endl;
	}

	int i, j, a, b = 0, nb_inc;

	nb_inc = 0;
	for (i = 0; i < nb_row_classes[EXTRA_ROW_SCHEME]; i++) {
		for (j = 0; j < nb_col_classes[EXTRA_ROW_SCHEME]; j++) {
			a = the_extra_row_scheme[i * nb_col_classes[EXTRA_ROW_SCHEME] + j];
			if (a == -1) {
				cout << "incomplete extra_row_scheme" << endl;
				cout << "i=" << i << "j=" << j << endl;
				cout << "ignoring this" << endl;
			}
			else {
				b = a * row_classes_len[EXTRA_ROW_SCHEME][i];
			}
			nb_inc += b;
		}
	}
	return nb_inc;
}


}}}}


