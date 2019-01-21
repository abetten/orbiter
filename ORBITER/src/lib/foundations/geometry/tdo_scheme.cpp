// tdo_scheme.C
//
// Anton Betten 8/27/07
//
//
// refine_rows and refine_columns started: December 2006
// moved away from inc_gen: 8/27/07
// separated out from refine: 1/24/08
// corrected a memory problem in refine_rows_hard: Dec 6 2010


#include "foundations.h"

namespace orbiter {


tdo_scheme::tdo_scheme()
{
	int i;
	
	P = NULL;
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

tdo_scheme::~tdo_scheme()
{
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
	if (P) {
		FREE_OBJECT(P);
		P = NULL;
		}
}

void tdo_scheme::init_part_and_entries(
	int *Part, int *Entries, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	for (part_length = 0; ; part_length++) {
		if (Part[part_length] == -1)
			break;
		}
	if (f_v) {
		cout << "partition of length " << part_length << endl;
		}
	
	for (nb_entries = 0; ; nb_entries++) {
		if (Entries[4 * nb_entries + 0] == -1)
			break;
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
	for (i = 0; i <= part_length; i++) {
		part[i] = Part[i];
		}
	entries = NEW_int(4 * nb_entries + 1);
	for (i = 0; i <= 4 * nb_entries; i++) {
		entries[i] = Entries[i];
		}
}

void tdo_scheme::init_part_and_entries_int(
	int *Part, int *Entries, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	for (part_length = 0; ; part_length++) {
		if (Part[part_length] == -1)
			break;
		}
	if (f_v) {
		cout << "partition of length " << part_length << endl;
		}
	
	for (nb_entries = 0; ; nb_entries++) {
		if (Entries[4 * nb_entries + 0] == -1)
			break;
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
	for (i = 0; i <= part_length; i++) {
		part[i] = Part[i];
		}
	entries = NEW_int(4 * nb_entries + 1);
	for (i = 0; i <= 4 * nb_entries; i++) {
		entries[i] = Entries[i];
		}
}

void tdo_scheme::init_TDO(int *Part, int *Entries,
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
	level[ROW] = row_level;
	level[COL] = col_level;
	level[EXTRA_ROW] = extra_row_level;
	level[EXTRA_COL] = extra_col_level;
	level[LAMBDA] = lambda_level;

	init_partition_stack(verbose_level - 2);
	
	//cout << "after init_partition_stack" << endl;
	
	//print_row_test_data();
	
}

void tdo_scheme::exit_TDO()
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

void tdo_scheme::init_partition_stack(int verbose_level)
{
	int k, at, f, c, l, i;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme::init_partition_stack" << endl;
		}
	if (f_vv) {
		cout << "part_length=" << part_length << endl;
		cout << "row_level=" << row_level << endl;
		cout << "col_level=" << col_level << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	mn = part[0];
	m = part[1];
	n = mn - m;
	if (part_length < 2) {
		cout << "part_length < 2" << endl;
		exit(1);
		}
	if (f_vvv) {
		cout << "init_partition_stack: m=" << m << " n=" << n << endl;
		int_vec_print(cout, part, part_length + 1);
		cout << endl;
		}
	
	P = NEW_OBJECT(partitionstack);
	P->allocate(m + n, 0 /* verbose_level */);
	//PB.init_partition_backtrack_basic(m, n, verbose_level - 10);
	if (f_vvv) {
		cout << "after PB.init_partition_backtrack_basic" << endl;
		}

	//partitionstack &P = PB.P;

	if (f_vvv) {
		cout << "initial partition stack: " << endl;
		P->print(cout);
		}
	for (k = 1; k < part_length; k++) {
		at = part[k];
		c = P->cellNumber[at];
		f = P->startCell[c];
		l = P->cellSize[c];
		if (f_vvv) {
			cout << "part[" << k << "]=" << at << endl;
			cout << "P->cellNumber[at]=" << c << endl;
			cout << "P->startCell[c]=" << f << endl;
			cout << "P->cellSize[c]=" << l << endl;
			cout << "f + l - at=" << f + l - at << endl;
			}
		P->subset_continguous(at, f + l - at);
		P->split_cell(FALSE);
		if (f_vvv) {
			cout << "after splitting at " << at << endl;
			P->print(cout);
			}
		if (P->ht == row_level) {
			l = P->ht;
			if (the_row_scheme) {
				FREE_int(the_row_scheme);
				the_row_scheme = NULL;
				}
			the_row_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_row_scheme[i] = -1;
				}
			get_partition(ROW, l, verbose_level - 3);
			get_row_or_col_scheme(ROW, l, verbose_level - 3);
			}
			
		if (P->ht == col_level) {
			l = P->ht;
			if (the_col_scheme) {
				FREE_int(the_col_scheme);
				the_col_scheme = NULL;
				}
			the_col_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_col_scheme[i] = -1;
				}
			get_partition(COL, l, verbose_level - 3);	
			get_row_or_col_scheme(COL, l, verbose_level - 3);
			}
			
		if (P->ht == extra_row_level) {
			l = P->ht;
			if (the_extra_row_scheme) {
				FREE_int(the_extra_row_scheme);
				the_extra_row_scheme = NULL;
				}
			the_extra_row_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_extra_row_scheme[i] = -1;
				}
			get_partition(EXTRA_ROW, l, verbose_level - 3);
			get_row_or_col_scheme(EXTRA_ROW, l, verbose_level - 3);	
			}
			
		if (P->ht == extra_col_level) {
			l = P->ht;
			if (the_extra_col_scheme) {
				FREE_int(the_extra_col_scheme);
				the_extra_col_scheme = NULL;
				}
			the_extra_col_scheme = NEW_int(l * l);
			for (i = 0; i < l * l; i++) {
				the_extra_col_scheme[i] = -1;
				}
			get_partition(EXTRA_COL, l, verbose_level - 3);
			get_row_or_col_scheme(EXTRA_COL, l, verbose_level - 3);	
			}
			
		if (P->ht == lambda_level) {
			l = P->ht;
			get_partition(LAMBDA, l, verbose_level - 3);
			}
			
		} // next k
	
	if (f_vvv) {
		cout << "before complete_partition_info" << endl;
		}
	if (row_level >= 2) {
		complete_partition_info(ROW, 0/*verbose_level*/);
		}
	if (col_level >= 2) {
		complete_partition_info(COL, 0/*verbose_level*/);
		}
	if (extra_row_level >= 2) {
		complete_partition_info(EXTRA_ROW, 0/*verbose_level*/);
		}
	if (extra_col_level >= 2 && extra_col_level < part_length) {
		complete_partition_info(EXTRA_COL, 0/*verbose_level*/);
		}
	complete_partition_info(LAMBDA, 0/*verbose_level*/);
	
	if (f_vv) {
		if (row_level >= 2) {
			print_scheme(ROW, FALSE);
			}
		if (col_level >= 2) {
			print_scheme(COL, FALSE);
			}
		if (extra_row_level >= 2) {
			print_scheme(EXTRA_ROW, FALSE);
			}
		if (extra_col_level >= 2) {
			print_scheme(EXTRA_COL, FALSE);
			}
		print_scheme(LAMBDA, FALSE);
		}
}

void tdo_scheme::exit_partition_stack()
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
	free_partition(ROW);
	free_partition(COL);
	//if (extra_row_level >= 0)
		free_partition(EXTRA_ROW);
	//if (extra_col_level >= 0)
		free_partition(EXTRA_COL);
	free_partition(LAMBDA);

}

void tdo_scheme::get_partition(int h, int l, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 10);
	int i;
	
	if (f_v) {
		cout << "tdo_scheme::get_partition h=" << h << " l=" << l
			<< " m=" << m << " n=" << n << endl;
		}
	if (l < 0) {
		cout << "tdo_scheme::get_partition l is negative" << endl;
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
			
	P->get_row_and_col_classes(row_classes[h], nb_row_classes[h],
		col_classes[h], nb_col_classes[h], verbose_level - 1);
				
	for (i = 0; i < nb_row_classes[h]; i++) {
		row_class_index[h][row_classes[h][i]] = i;
		if (f_vv) {
			cout << "row_class_index[h][" << row_classes[h][i] << "] = "
				<< row_class_index[h][row_classes[h][i]] << endl;
			}
		}
	for (i = 0; i < nb_col_classes[h]; i++) {
		col_class_index[h][col_classes[h][i]] = i;
		if (f_vv) {
			cout << "col_class_index[h][" << col_classes[h][i] << "] = "
				<< col_class_index[h][col_classes[h][i]] << endl;
			}
		}
	if (f_vv) {
		cout << "nb_row_classes[h]=" << nb_row_classes[h] << endl;
		cout << "nb_col_classes[h]=" << nb_col_classes[h] << endl;
		}
}

void tdo_scheme::free_partition(int i)
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

void tdo_scheme::complete_partition_info(int h, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int f, i, j, c1, S, k;
	
	if (f_v) {
		cout << "tdo_scheme::complete_partition_info h=" << h << endl;
		cout << "# of row classes = " << nb_row_classes[h] << endl;
		cout << "# of col classes = " << nb_col_classes[h] << endl;
		}
	f = 0;
	for (i = 0; i < nb_row_classes[h]; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
			}
		c1 = row_classes[h][i];
		if (f_vv) {
			cout << "c1=" << c1 << endl;
			}
		S = P->cellSizeAtLevel(c1, level[h]);
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
		S = P->cellSizeAtLevel(c1, level[h]);
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

void tdo_scheme::get_row_or_col_scheme(int h, int l, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d, c1, c2, s1, s2, v;
	
	if (f_v) {
		cout << "tdo_scheme::get_row_or_col_scheme" << endl;
		}
	for (i = 0; i < nb_entries; i++) {
		d = entries[i * 4 + 0];
		c1 = entries[i * 4 + 1];
		c2 = entries[i * 4 + 2];
		v = entries[i * 4 + 3];
		if (d == l) {
			//cout << "entry " << i << " : " << d << " "
			//<< c1 << " " << c2 << " " << v << endl;
			if (h == ROW && row_class_index[h][c1] >= 0) {
				// row scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_row_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_row_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == COL && col_class_index[h][c1] >= 0) {
				// col scheme
				s1 = row_class_index[h][c2];
				s2 = col_class_index[h][c1];
				//cout << "the_col_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_col_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == EXTRA_ROW && row_class_index[h][c1] >= 0) {
				// col scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_extra_row_scheme[" << s1 << " * "
				//<< nb_col_classes[h] << " + " << s2 << "] = "
				//<< v << endl;
				the_extra_row_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == EXTRA_COL && col_class_index[h][c1] >= 0) {
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
	if (h == ROW) {
		if (the_row_scheme_cur) {
			FREE_int(the_row_scheme_cur);
			the_row_scheme_cur = NULL;
			}
		the_row_scheme_cur = NEW_int(m * nb_col_classes[h]);
		for (i = 0; i < m; i++) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				the_row_scheme_cur[i * nb_col_classes[h] + j] = 0;
				}
			}
		//print_row_test_data();
		}
	if (h == COL) {
		if (the_col_scheme_cur) {
			FREE_int(the_col_scheme_cur);
			the_col_scheme_cur = NULL;
			}
		the_col_scheme_cur = NEW_int(n * nb_row_classes[h]);
		for (i = 0; i < n; i++) {
			for (j = 0; j < nb_row_classes[h]; j++) {
				the_col_scheme_cur[i * nb_row_classes[h] + j] = 0;
				}
			}
		}
	if (h == EXTRA_ROW) {
		if (the_extra_row_scheme_cur) {
			FREE_int(the_extra_row_scheme_cur);
			the_extra_row_scheme_cur = NULL;
			}
		the_extra_row_scheme_cur = NEW_int(m * nb_col_classes[h]);
		for (i = 0; i < m; i++) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				the_extra_row_scheme_cur[i * nb_col_classes[h] + j] = 0;
				}
			}
		}
	if (h == EXTRA_COL) {
		if (the_extra_col_scheme_cur) {
			FREE_int(the_extra_col_scheme_cur);
			the_extra_col_scheme_cur = NULL;
			}
		the_extra_col_scheme_cur = NEW_int(n * nb_row_classes[h]);
		for (i = 0; i < n; i++) {
			for (j = 0; j < nb_row_classes[h]; j++) {
				the_extra_col_scheme_cur[i * nb_row_classes[h] + j] = 0;
				}
			}
		}
	if (f_v) {
		cout << "tdo_scheme::get_row_or_col_scheme finished" << endl;
		}
}

void tdo_scheme::get_column_split_partition(int verbose_level,
		partitionstack &P)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;
	
	if (f_v) {
		cout << "get_column_split_partition" << endl;
		}
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];
	if (FALSE) {
		cout << "l1=" << l1 << " at level " << level[ROW] << endl;
		cout << "l2=" << l2 << " at level " << level[COL] << endl;
		cout << "R=" << R << endl;
		}
	P.allocate(l2, FALSE);
	for (i = 0; i < l1; i++) {
		ci = col_classes[ROW][i];
		j1 = col_class_index[COL][ci];
		cc = P.cellNumber[j1];
		f = P.startCell[cc];
		l = P.cellSize[cc];
		if (FALSE) {
			cout << "i=" << i << " ci=" << ci << " j1=" << j1
					<< " cc=" << cc << endl;
			}
		P.subset_size = 0;
		for (h = 0; h < l; h++) {
			j = P.pointList[f + h];
			cj = col_classes[COL][j];
			if (FALSE) {
				cout << "j=" << j << " cj=" << cj << endl;
				}
			if (!tdo_scheme::P->is_descendant_of_at_level(cj, ci,
					level[ROW], FALSE)) {
				if (FALSE) {
					cout << j << "/" << cj << " is not a "
							"descendant of " << i << "/" << ci << endl;
					}
				P.subset[P.subset_size++] = j;
				}
			}
		if (FALSE) {
			cout << "non descendants of " << i << "/" << ci << " : ";
			int_set_print(cout, P.subset, P.subset_size);
			cout << endl;
			}
		if (P.subset_size > 0) {
			P.split_cell(FALSE);
			if (FALSE) {
				P.print(cout);
				}
			}
		}
	if (f_vv) {
		cout << "column-split partition:" << endl;
		P.print(cout);
		}
}

void tdo_scheme::get_row_split_partition(int verbose_level,
	partitionstack &P)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;
	
	if (f_v) {
		cout << "get_row_split_partition" << endl;
		}
	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
	if (FALSE) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "R=" << R << endl;
		}
	P.allocate(l2, FALSE);
	for (i = 0; i < l1; i++) {
		ci = row_classes[COL][i];
		j1 = row_class_index[ROW][ci];
		cc = P.cellNumber[j1];
		f = P.startCell[cc];
		l = P.cellSize[cc];
		if (FALSE) {
			cout << "i=" << i << " ci=" << ci << " j1=" << j1
				<< " cc=" << cc << endl;
			}
		P.subset_size = 0;
		for (h = 0; h < l; h++) {
			j = P.pointList[f + h];
			cj = row_classes[ROW][j];
			if (FALSE) {
				cout << "j=" << j << " cj=" << cj << endl;
				}
			if (!tdo_scheme::P->is_descendant_of_at_level(cj, ci,
					level[COL], FALSE)) {
				if (FALSE) {
					cout << j << "/" << cj << " is not a descendant "
						"of " << i << "/" << ci << endl;
					}
				P.subset[P.subset_size++] = j;
				}
			else {
				if (FALSE) {
					cout << cj << " is a descendant of " << ci << endl;
					}
				}
			}
		if (FALSE) {
			cout << "non descendants of " << i << "/" << ci << " : ";
			int_set_print(cout, P.subset, P.subset_size);
			cout << endl;
			}
		if (P.subset_size > 0) {
			P.split_cell(FALSE);
			if (FALSE) {
				P.print(cout);
				}
			}
		}
	if (f_vv) {
		cout << "row-split partition:" << endl;
		P.print(cout);
		}
}

void tdo_scheme::print_all_schemes()
{
	if (lambda_level >= 2) {
		print_scheme(LAMBDA, FALSE);
		}
	if (extra_row_level >= 2) {
		print_scheme(EXTRA_ROW, FALSE);
		}
	if (extra_col_level >= 2) {
		print_scheme(EXTRA_COL, FALSE);
		}
	if (row_level >= 2) {
		print_scheme(ROW, FALSE);
		}
	if (col_level >= 2) {
		print_scheme(COL, FALSE);
		}
}

void tdo_scheme::print_scheme(int h, int f_v)
{
	int i, j, c1, c2, a = 0;
	
	if (h == ROW) {
		cout << "row_scheme at level " << level[h] << " : " << endl;
		}
	else if (h == COL) {
		cout << "col_scheme at level " << level[h] << " : " << endl;
		}
	else if (h == EXTRA_ROW) {
		cout << "extra_row_scheme at level " << level[h] << " : " << endl;
		}
	else if (h == EXTRA_COL) {
		cout << "extra_col_scheme at level " << level[h] << " : " << endl;
		}
	else if (h == LAMBDA) {
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
		if (h != LAMBDA) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				if (h == ROW) {
					a = the_row_scheme[i * nb_col_classes[h] + j];
					}
				else if (h == COL) {
					a = the_col_scheme[i * nb_col_classes[h] + j];
					}
				else if (h == EXTRA_ROW) {
					a = the_extra_row_scheme[i * nb_col_classes[h] + j];
					}
				else if (h == EXTRA_COL) {
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

void tdo_scheme::print_scheme_tex(ostream &ost, int h)
{
	print_scheme_tex_fancy(ost, h, FALSE, NULL);
}

void tdo_scheme::print_scheme_tex_fancy(ostream &ost,
	int h, int f_label, char *label)
{
	int i, j, a = 0, n, m, c1, c2;
	
	n = nb_row_classes[h];
	m = nb_col_classes[h];
	ost << "$$" << endl;
	ost << "\\begin{array}{r|*{" << m << "}{r}}" << endl;
	if (f_label) {
		ost << "\\multicolumn{" << m + 1
			<< "}{c}{\\mbox{" << label << "}}\\\\" << endl;
		}
	if (h == ROW || h == EXTRA_ROW)
		ost << "\\rightarrow";
	else if (h == COL || h == EXTRA_COL)
		ost << "\\downarrow";
	else if (h == LAMBDA)
		ost << "\\lambda";
	for (j = 0; j < m; j++) {
		c2 = col_classes[h][j];
		ost << " & " << setw(3) << col_classes_len[h][j]
			<< "_{" << setw(3) << c2 << "}";
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < n; i++) {
		c1 = row_classes[h][i];
		ost << row_classes_len[h][i] << "_{" << setw(3) << c1 << "}";
		for (j = 0; j < m; j++) {
			if (h == ROW) {
				a = the_row_scheme[i * m + j];
				}
			else if (h == COL) {
				a = the_col_scheme[i * m + j];
				}
			else if (h == EXTRA_ROW) {
				a = the_extra_row_scheme[i * m + j];
				}
			else if (h == EXTRA_COL) {
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

void tdo_scheme::compute_whether_first_inc_must_be_moved(
	int *f_first_inc_must_be_moved, int verbose_level)
{
	int i, j, ii, fi, fii, fj, row_cell0, row_cell, col_cell, a, b, c;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme::compute_whether_first_"
				"inc_must_be_moved" << endl;
		}
	for (i = 0; i < nb_row_classes[ROW]; i++) {
		f_first_inc_must_be_moved[i] = TRUE;
		if (col_level < 2)
			continue;
		fi = row_classes_first[ROW][i];
		row_cell0 = row_class_no[COL][fi];
		for (j = 0; j < nb_col_classes[ROW]; j++) {
			a = the_row_scheme[i * nb_col_classes[ROW] + j];
			if (a > 0)
				break;
			}
		
		if (f_vv) {
			cout << "considering whether incidence in block " << i << ","
				<< j << " must be moved" << endl;
			}
		
		fj = col_classes_first[COL][j];
		col_cell = col_class_no[COL][fj];
		c = the_col_scheme[row_cell0 * nb_col_classes[COL] + col_cell];
		if (f_vvv) {
			cout << "c=" << c << endl;
			}
		if (c >= 0) {
			if (f_vvv) {
				cout << "looking at COL scheme:" << endl;
				}
			f_first_inc_must_be_moved[i] = FALSE;
			for (ii = i + 1; ii < nb_row_classes[ROW]; ii++) {
				b = the_row_scheme[ii * nb_col_classes[ROW] + j];
				fii = row_classes_first[ROW][ii];
				row_cell = row_class_no[COL][fii];
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
					f_first_inc_must_be_moved[i] = FALSE;
					//ii = nb_row_classes[ROW];
					break;
					}
				if (b) {
					if (f_vvv) {
						cout << "ii=" << ii << " seeing non zero entry "
							<< b << ", hence it must be moved" << endl;
						}
					f_first_inc_must_be_moved[i] = TRUE;
					break;
					}
				} // next ii
			}
		else {
			if (f_vvv) {
				cout << "looking at EXTRA_COL scheme:" << endl;
				}
			fi = row_classes_first[ROW][i];
			row_cell0 = row_class_no[EXTRA_COL][fi];
			if (f_vvv) {
				cout << "row_cell0=" << row_cell0 << endl;
				}
			for (ii = i + 1; ii < nb_row_classes[ROW]; ii++) {
				b = the_row_scheme[ii * nb_col_classes[ROW] + j];
				fii = row_classes_first[ROW][ii];
				row_cell = row_class_no[EXTRA_COL][fii];
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
					f_first_inc_must_be_moved[i] = FALSE;
					//ii = nb_row_classes[ROW];
					break;
					}
				if (b) {
					if (f_vvv) {
						cout << "ii=" << ii << " seeing non zero entry "
							<< b << ", hence it must be moved" << endl;
						}
					f_first_inc_must_be_moved[i] = TRUE;
					break;
					}
				} // next ii
			}
		
		}
}

int tdo_scheme::count_nb_inc_from_row_scheme(int verbose_level)
{
	int i, j, a, b = 0, nb_inc;
	int f_v = (verbose_level > 1);
	
	if (f_v) {
		cout << "tdo_scheme::count_nb_inc_from_row_scheme" << endl;
		}
	nb_inc = 0;
	for (i = 0; i < nb_row_classes[ROW]; i++) {
		for (j = 0; j < nb_col_classes[ROW]; j++) {
			a = the_row_scheme[i * nb_col_classes[ROW] + j];
			if (a == -1) {
				cout << "incomplete row_scheme" << endl;
				cout << "i=" << i << "j=" << j << endl;
				cout << "ignoring this" << endl;
				}
			else {
				b = a * row_classes_len[ROW][i];
				}
			nb_inc += b;
			}
		}
	//cout << "nb_inc=" << nb_inc << endl;
	return nb_inc;
}

int tdo_scheme::count_nb_inc_from_extra_row_scheme(int verbose_level)
{
	int i, j, a, b = 0, nb_inc;
	int f_v = (verbose_level > 1);
	
	if (f_v) {
		cout << "tdo_scheme::count_nb_inc_from_extra_row_scheme" << endl;
		}
	nb_inc = 0;
	for (i = 0; i < nb_row_classes[EXTRA_ROW]; i++) {
		for (j = 0; j < nb_col_classes[EXTRA_ROW]; j++) {
			a = the_extra_row_scheme[i * nb_col_classes[EXTRA_ROW] + j];
			if (a == -1) {
				cout << "incomplete extra_row_scheme" << endl;
				cout << "i=" << i << "j=" << j << endl;
				cout << "ignoring this" << endl;
				}
			else {
				b = a * row_classes_len[EXTRA_ROW][i];
				}
			nb_inc += b;
			}
		}
	//cout << "nb_inc=" << nb_inc << endl;
	return nb_inc;
}

int tdo_scheme::geometric_test_for_row_scheme(partitionstack &P, 
	int *point_types, int nb_point_types, int point_type_len, 
	int *distributions, int nb_distributions, 
	int f_omit1, int omit1, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_vvvv = (verbose_level >= 4);
	int f_v5 = (verbose_level >= 7);
	int i, s, d, /*l2,*/ L1, L2, cnt, new_nb_distributions; 
	int f_ruled_out;
	int *ruled_out_by;
	int *non_zero_blocks, nb_non_zero_blocks;
	
	if (f_vvv) {
		cout << "tdo_refine::geometric_test_for_row_scheme "
			"nb_distributions=" << nb_distributions << endl;
		}
	//l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level - 3);
	if (L2 != point_type_len) {
		cout << "tdo_refine::geometric_test_for_row_scheme "
				"L2 != point_type_len" << endl;
		exit(1);
		}
	
	ruled_out_by = NEW_int(nb_point_types + 1);
	non_zero_blocks = NEW_int(nb_point_types);
	for (i = 0; i <= nb_point_types; i++) {
		ruled_out_by[i] = 0;
		}

	new_nb_distributions = 0;
	for (cnt = 0; cnt < nb_distributions; cnt++) {
		nb_non_zero_blocks = 0;
		for (i = 0; i < nb_point_types; i++) {
			d = distributions[cnt * nb_point_types + i];
			if (d == 0)
				continue;
			non_zero_blocks[nb_non_zero_blocks++] = i;
			}
		
		if (f_vvvv) {
			cout << "geometric_test_for_row_scheme: testing distribution " 
				<< cnt << " / " << nb_distributions << " : ";
			int_vec_print(cout,
				distributions + cnt * nb_point_types,
				nb_point_types);
			cout << endl;
			if (f_v5) {
				cout << "that is" << endl;
				for (i = 0; i < nb_non_zero_blocks; i++) {
					d = distributions[cnt *
						nb_point_types + non_zero_blocks[i]];
					cout << setw(3) << i << " : " << setw(3) << d << " x ";
					int_vec_print(cout,
						point_types + non_zero_blocks[i] * point_type_len,
						point_type_len);
					cout << endl;
					}
				}
			}
		f_ruled_out = FALSE;
		for (s = 1; s <= nb_non_zero_blocks; s++) {
			if (!geometric_test_for_row_scheme_level_s(P, s, 
				point_types, nb_point_types, point_type_len, 
				distributions + cnt * nb_point_types, 
				non_zero_blocks, nb_non_zero_blocks, 
				f_omit1, omit1, verbose_level - 4)) {
				f_ruled_out = TRUE;
				ruled_out_by[s]++;
				if (f_vv) {
					cout << "geometric_test_for_row_scheme: distribution "
						<< cnt << " / " << nb_distributions
						<< " eliminated by test of order " << s << endl;
					}
				if (f_vvv) {
					cout << "the eliminated scheme is:" << endl;
					for (i = 0; i < nb_non_zero_blocks; i++) {
						d = distributions[cnt * nb_point_types +
										  non_zero_blocks[i]];
						cout << setw(3) << i << " : "
							<< setw(3) << d << " x ";
						int_vec_print(cout,
							point_types + non_zero_blocks[i] * point_type_len,
							point_type_len);
						cout << endl;						
						}
					cout << "we repeat the test with more printout:" << endl;
					geometric_test_for_row_scheme_level_s(P, s, 
						point_types, nb_point_types, point_type_len, 
						distributions + cnt * nb_point_types, 
						non_zero_blocks, nb_non_zero_blocks, 
						f_omit1, omit1, verbose_level - 3);
					}
				break;
				}
			}
		


		if (!f_ruled_out) {
			for (i = 0; i < nb_point_types; i++) {
				distributions[new_nb_distributions * nb_point_types + i] = 
					distributions[cnt * nb_point_types + i];
				}
			new_nb_distributions++;
			}
		} // next cnt
	if (f_v) {
		cout << "geometric_test_for_row_scheme: number of distributions "
			"reduced from " << nb_distributions << " to "
			<< new_nb_distributions << ", i.e. Eliminated " 
			<< nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "# of ruled out by test of order ";
		int_vec_print(cout, ruled_out_by, nb_point_types + 1);
		cout << endl;
		//cout << "nb ruled out by first order test  = "
		//<< nb_ruled_out_by_order1 << endl;
		//cout << "nb ruled out by second order test = "
		//<< nb_ruled_out_by_order2 << endl;
		for (i = nb_point_types; i >= 1; i--) {
			if (ruled_out_by[i])
				break;
			}
		if (i) {
			cout << "highest order test that was successfully "
					"applied is order " << i << endl;
			}
		}
	FREE_int(ruled_out_by);
	FREE_int(non_zero_blocks);
	return new_nb_distributions;
}
#if 0
int tdo_scheme::test_row_distribution(
	int *point_types, int nb_point_types, int point_type_len,
	int *distributions, int nb_distributions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int l2, cnt, J, len, k, i, d, c, new_nb_distributions, bound;
	int f_ruled_out, f_ruled_out_by_braun, f_ruled_out_by_packing;
	int nb_ruled_out_by_braun = 0, nb_ruled_out_by_packing = 0;
	int nb_ruled_out_by_both = 0;
	
	if (f_v) {
		cout << "tdo_refine::test_row_distribution "
				"nb_distributions=" << nb_distributions << endl;
		}
	l2 = nb_col_classes[COL];
	if (l2 != point_type_len) {
		cout << "tdo_refine::test_row_distribution "
				"l2 != point_type_len" << endl;
		exit(1);
		}
	
	new_nb_distributions = 0;
	
	for (cnt = 0; cnt < nb_distributions; cnt++) {
		if (f_vv) {
			cout << "testing distribution " << cnt << " : ";
			int_vec_print(cout,
				distributions + cnt * nb_point_types,
				nb_point_types);
			cout << endl;
			if (f_vvv) {
				cout << "that is" << endl;
				for (i = 0; i < nb_point_types; i++) {
					d = distributions[cnt * nb_point_types + i];
					if (d == 0)
						continue;
					cout << setw(3) << d << " x ";
					int_vec_print(cout,
						point_types + i * point_type_len,
						point_type_len);
					cout << endl;
					}
				}
			}
		f_ruled_out = FALSE;
		f_ruled_out_by_braun = FALSE;
		f_ruled_out_by_packing = FALSE;
		
		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			int *type;
			
			if (f_vvv) {
				cout << "testing distribution " << cnt << " in block "
					<< J << " len=" << len << endl;
				}
			type = NEW_int(len + 1);
			for (k = 0; k <= len; k++)
				type[k] = 0;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				c = point_types[i * point_type_len + J];
				type[c] += d;
				}
			if (f_vvv) {
				cout << "line type: ";
				int_vec_print(cout, type + 1, len);
				cout << endl;
				}
			if (!braun_test_on_line_type(len, type)) {
				if (f_vv) {
					cout << "distribution " << cnt << " is eliminated "
						"in block " << J << " using Braun test" << endl;
					}
				f_ruled_out = TRUE;
				f_ruled_out_by_braun = TRUE;
				FREE_int(type);
				break;
				}
			FREE_int(type);
			} // next J
		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			if (len == 1) 
				continue;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				if (d == 0)
					continue;
				c = point_types[i * point_type_len + J];
				// now we want d lines of size c on len points
				if (c > 1) {
					if (c > len) {
						cout << "c > len" << endl;
						cout << "J=" << J << " i=" << i << " d="
								<< d << " c=" << c << endl;
						exit(1);
						}
					bound = TDO_upper_bound(len, c);
					if (d > bound) {
						if (f_vv) {
							cout << "distribution " << cnt
								<< " is eliminated in block "
								<< J << " row-block " << i
								<< " using packing numbers" << endl;
							cout << "len=" << len << endl;
							cout << "d=" << d << endl;
							cout << "c=" << c << endl;
							cout << "bound=" << bound << endl;
							}
						f_ruled_out = TRUE;
						f_ruled_out_by_packing = TRUE;
						break;
						}
					}
				}
			if (f_ruled_out)
				break;
			}
		if (f_ruled_out) {
			if (f_ruled_out_by_braun) 
				nb_ruled_out_by_braun++;
			if (f_ruled_out_by_packing)
				nb_ruled_out_by_packing++;
			if (f_ruled_out_by_braun && f_ruled_out_by_packing)
				nb_ruled_out_by_both++;
			}
		else {
			for (i = 0; i < nb_point_types; i++) {
				distributions[new_nb_distributions * nb_point_types + i] = 
					distributions[cnt * nb_point_types + i];
				}
			new_nb_distributions++;
			}
		} // next cnt
	if (f_v) {
		cout << "number of distributions reduced from "
			<< nb_distributions << " to "
			<< new_nb_distributions << ", i.e. Eliminated " 
			<< nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "nb_ruled_out_by_braun = "
			<< nb_ruled_out_by_braun << endl;
		cout << "nb_ruled_out_by_packing = "
			<< nb_ruled_out_by_packing << endl;
		cout << "nb_ruled_out_by_both = "
			<< nb_ruled_out_by_both << endl;
		}
	return new_nb_distributions;
}
#endif

int tdo_scheme::geometric_test_for_row_scheme_level_s(
	partitionstack &P, int s,
	int *point_types, int nb_point_types, int point_type_len, 
	int *distribution, 
	int *non_zero_blocks, int nb_non_zero_blocks, 
	int f_omit1, int omit1, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	int set[1000];
	int J, L1, L2, len, max, cur, u, D, d, c;
	int nb_inc, e, f, nb_ordererd_pairs;
	
	if (f_vvv) {
		cout << "geometric_test_for_row_scheme_level_s s=" << s << endl;
		}
	if (s >= 1000) {
		cout << "level too deep" << endl;
		exit(1);
		}
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level - 3);
	first_k_subset(set, nb_non_zero_blocks, s);
	while (TRUE) {
		D = 0;
		for (u = 0; u < s; u++) {
			d = distribution[non_zero_blocks[set[u]]];
			D += d;
			}
		max = D * (D - 1);
		cur = 0;
		for (J = 0; J < L2; J++) {
			len = col_classes_len[COL][J];
			nb_inc = 0;
			for (u = 0; u < s; u++) {
				c = point_types[non_zero_blocks[set[u]] * point_type_len + J];
				d = distribution[non_zero_blocks[set[u]]];
				// we have d rows with c incidences in len columns
				nb_inc += d * c;
				}
			
			e = nb_inc % len; // the number of incidences in the extra row
			f = nb_inc / len; // the number of full rows

			nb_ordererd_pairs = 0;
			if (n) {
				nb_ordererd_pairs = e * (f + 1) * f + (len - e) * f * (f - 1);
				}
			cur += nb_ordererd_pairs;
			if (cur > max) {
				if (f_v) {
					cout << "tdo_scheme::geometric_test_for_row_scheme_"
						"level_s s=" << s << " failure in point type ";
					int_vec_print(cout, set, s);
					cout << endl;
					cout << "max=" << max << endl;
					cout << "J=" << J << endl;
					cout << "nb_inc=" << nb_inc << endl;
					cout << "nb_ordererd_pairs=" << nb_ordererd_pairs << endl;
					cout << "cur=" << cur << endl;
					}
				return FALSE;
				}
			} // next J
		if (!next_k_subset(set, nb_non_zero_blocks, s))
			break;
		}
	return TRUE;
}

// #############################################################################
// parameter refinement: refine rows
// #############################################################################

int tdo_scheme::refine_rows(int verbose_level,
	int f_use_mckay, int f_once, 
	partitionstack &P, 
	int *&point_types, int &nb_point_types, int &point_type_len, 
	int *&distributions, int &nb_distributions, 
	int &cnt_second_system, solution_file_data *Sol,
	int f_omit1, int omit1,
	int f_omit2, int omit2,
	int f_use_packing_numbers,
	int f_dual_is_linear_space,
	int f_do_the_geometric_test)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_easy;
	int l1, l2, R;
	
	if (f_v) {
		cout << "refine_rows" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit2=" << f_omit2 << " omit2=" << omit2 << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		cout << "f_use_mckay=" << f_use_mckay << endl;
		}
	if (row_level >= 2) {
		R = nb_row_classes[ROW];
		l1 = nb_col_classes[ROW];
		l2 = nb_col_classes[COL];
		if (f_vv) {
			cout << "l1=" << l1 << " at level " << level[ROW] << endl;
			cout << "l2=" << l2 << " at level " << level[COL] << endl;
			cout << "R=" << R << endl;
			}
		get_column_split_partition(0 /*verbose_level*/, P);
		if (f_vv) {
			cout << "column split partition: " << P << endl;
			}
		if (P.ht != l1) {
			cout << "P.ht != l1" << endl;
			exit(1);
			}
		if ((R == 1) && (l1 == 1) && (the_row_scheme[0] == -1)) {
			if (!refine_rows_easy(verbose_level - 1, 
				point_types, nb_point_types, point_type_len, 
				distributions, nb_distributions, cnt_second_system)) {
				return FALSE;
				}
			}
		else {
			if (!refine_rows_hard(P,
				verbose_level - 1, f_use_mckay, f_once,
				point_types, nb_point_types, point_type_len, 
				distributions, nb_distributions, cnt_second_system,
				f_omit1, omit1, f_omit2, omit2, 
				f_use_packing_numbers, f_dual_is_linear_space)) {
				return FALSE;
				}
			}
		}
	else {
		if (!refine_rows_easy(verbose_level - 1, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, cnt_second_system)) {
			return FALSE;
			}
		}

	if (f_do_the_geometric_test) {
		nb_distributions = geometric_test_for_row_scheme(P, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, 
			f_omit1, omit1, 
			verbose_level);
		}
	return TRUE;
}

int tdo_scheme::refine_rows_easy(int verbose_level, 
	int *&point_types, int &nb_point_types, int &point_type_len,  
	int *&distributions, int &nb_distributions, 
	int &cnt_second_system)
{
	int nb_rows;
	int i, j, J, S, l2, nb_eqns, nb_vars;
	int nb_eqns_joining, nb_eqns_upper_bound;
	int nb_sol, len, k, a2, a, b, ab;
	int f_used, j1, j2, len1, len2, cnt;
	int Nb_eqns, Nb_vars;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char label[100];

	if (f_v) {
		cout << "refine_rows_easy" << endl;
		}
	l2 = nb_col_classes[COL];
	
	//partitionstack &P = PB.P;
	nb_rows = P->startCell[1];
	S = nb_rows - 1;
	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
		}
	
	nb_vars = l2 + 1; // 1 slack variable
	nb_eqns = 1;
	
	diophant D;
	
	D.open(nb_eqns, nb_vars);
		
	// 1st equation: connections within the same row-partition
	for (J = 0; J < nb_vars; J++) {
		D.Aij(0, J) = minus_one_if_positive(the_col_scheme[0 * l2 + J]);
		}
	D.Aij(0, nb_vars - 1) = 0;
	if (f_vv) {
		cout << "nb_rows=" << nb_rows << endl;
		}
	D.RHS[0] = nb_rows - 1;
	if (f_vv) {
		cout << "RHS[0]=" << D.RHS[0] << endl;
		}
		
	for (j = 0; j < l2; j++) {
		D.x_max[j] = col_classes_len[COL][j];
		}
	D.x_max[nb_vars - 1] = nb_rows - 1;;
	
	D.f_x_max = TRUE;
		

	D.eliminate_zero_rows_quick(verbose_level);
	D.sum = S;
	if (f_vv) {
		cout << "The first system is" << endl;
		D.print();
		}
	if (f_vv) {
		char label[1000];
			
		sprintf(label, "first");
		D.write_xml(cout, label);
		}

	nb_sol = 0;
	point_type_len = nb_vars - 1;
	
	if (D.solve_first(verbose_level - 2)) {
	
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D.solve_next())
				break;
			}
		}
	if (f_v) {
		cout << "found " << nb_sol << " point types" << endl;
		}
	if (nb_sol == 0) {
		return FALSE;
		}
	nb_point_types = nb_sol;
	
	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		if (len > 2) {
			nb_eqns_upper_bound += len - 2;
			}
		}
	nb_eqns_joining = l2 + ((l2 * (l2 - 1)) >> 1);
	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	Nb_vars = nb_sol;
	
	diophant D2;
	
	D2.open(Nb_eqns, Nb_vars);
	point_types = NEW_int(nb_point_types * point_type_len);
	if (f_v) {
		cout << "refine_rows_easy: opening second "
			<< cnt_second_system << " system with "
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
		}
	
	nb_sol = 0;
	if (D.solve_first(verbose_level - 2)) {
	
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			for (i = 0; i < point_type_len; i++) {
				D2.Aij(i, nb_sol) = D.x[i];
				point_types[nb_sol * point_type_len + i] = D.x[i];
				}
			nb_sol++;
			if (!D.solve_next())
				break;
			}
		}
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		for (i = 0; i < Nb_vars; i++) {
			a = point_types[i * point_type_len + j];
			a2 = binomial2(a);
			D2.Aij(l2 + j, i) = a2;
			}
		D2.RHS[l2 + j] = binomial2(len);
		D2.type[l2 + j] = t_LE;
		sprintf(label, "J_{%d}", j + 1);
		D2.init_eqn_label(l2 + j, label);
		}
	cnt = 0;
	for (j1 = 0; j1 < l2; j1++) {
		len1 = col_classes_len[COL][j1];
		for (j2 = j1 + 1; j2 < l2; j2++) {
			len2 = col_classes_len[COL][j2];
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j1];
				b = point_types[i * point_type_len + j2];
				ab = a * b;
				D2.Aij(l2 + l2 + cnt, i) = ab;
				}
			D2.RHS[l2 + l2 + cnt] = len1 * len2;
			D2.type[l2 + l2 + cnt] = t_LE;
			sprintf(label, "J_{%d,%d}", j1 + 1, j2 + 1);
			D2.init_eqn_label(l2 + l2 + cnt, label);
			cnt++;
			}
		}

	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		for (k = 3; k <= len; k++) {
			for (i = 0; i < Nb_vars; i++) {
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 0;
				}
			f_used = FALSE;
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j];
				if (a < k)
					continue;
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 1;
				f_used = TRUE;
				}
			if (f_used) {
				int bound = TDO_upper_bound(len, k);
				D2.RHS[l2 + nb_eqns_joining + nb_eqns_upper_bound] = bound;
				D2.type[l2 + nb_eqns_joining + nb_eqns_upper_bound] = t_LE;
				sprintf(label, "P_{%d,%d} \\,\\mbox{using}\\, "
						"P(%d,%d)=%d", j + 1, k, len, k, bound);
				D2.init_eqn_label(l2 +
						nb_eqns_joining + nb_eqns_upper_bound, label);
				nb_eqns_upper_bound++;
				}
			} // next k
		} // next j
	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	D2.m = Nb_eqns;

	if (f_v) {
		cout << "second system " << cnt_second_system << " found "
			<< nb_sol << " point types" << endl;
		}
	cnt_second_system++;
	if (nb_sol == 0) {
		FREE_int(point_types);
		return FALSE;
		}
	D2.sum = nb_rows;
	for (i = 0; i < l2; i++) {
		D2.RHS[i] = col_classes_len[COL][i] * the_col_scheme[i];
		sprintf(label, "F_{%d}", i + 1);
		D2.init_eqn_label(i, label);
		}
	D2.eliminate_zero_rows_quick(verbose_level);
	if (f_vv) {
		cout << "The second system is" << endl;
		D2.print();
		}
	if (f_vv) {
		char label[1000];
			
		sprintf(label, "second");
		D2.write_xml(cout, label);
		}
	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << D2.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D2.solve_next())
				break;
			}
		}
	nb_distributions = nb_sol;
	distributions = NEW_int(nb_distributions * nb_point_types);
	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << D2.x[i];
					}
				cout << endl;
				}
			for (i = 0; i < Nb_vars; i++) {
				distributions[nb_sol * nb_point_types + i] = D2.x[i];
				}
			nb_sol++;
			if (!D2.solve_next())
				break;
			}
		}
	if (f_v) {
		cout << "refine_rows_easy: found " << nb_distributions
			<< " point type distributions." << endl;
		}
	return TRUE;
}

int tdo_scheme::refine_rows_hard(partitionstack &P, int verbose_level, 
	int f_use_mckay, int f_once, 
	int *&point_types, int &nb_point_types, int &point_type_len,  
	int *&distributions, int &nb_distributions, 
	int &cnt_second_system, 
	int f_omit1, int omit1, int f_omit, int omit, 
	int f_use_packing_numbers, int f_dual_is_linear_space)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, r, R, l1, /*l2,*/ L1, L2;
	int nb_sol;
	int point_types_allocated;
	int h, u;
	tdo_data T;

	if (f_v) {
		cout << "refine_rows_hard" << endl;
		if (f_omit1) {
			cout << "omitting the last " << omit1
				<< " column blocks from the previous row-scheme" << endl;
			}
		if (f_omit) {
			cout << "omitting the last " << omit << " row blocks" << endl;
			}
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		cout << "f_use_mckay=" << f_use_mckay << endl;
		}
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	//l2 = nb_col_classes[COL];

	if (f_v) {
		cout << "the_row_scheme is:" << endl;
		int i, j;
		for (i = 0; i < R; i++) {
			for (j = 0; j < l1; j++) {
				cout << setw(4) << the_row_scheme[i * l1 + j];
				}
			cout << endl;
			}
		}
	
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level);

	T.allocate(R);
	
	T.types_first[0] = 0;
	
	point_types_allocated = 100;
	nb_point_types = 0;
	point_type_len = L2 + L1; // + slack variables
	point_types = NEW_int(point_types_allocated * point_type_len);
		// detected and corrected an error: Dec 6 2010
		// it was allocated to point_types_allocated * L2
		// which is not enough
		
		// when we are done, it is [point_types_allocated * L2]
	

	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_v) {
			cout << "r=" << r << endl;
			}
		
		tdo_rows_setup_first_system(verbose_level, 
			T, r, P, 
			f_omit1, omit1, 
			point_types, nb_point_types);
		
		if (f_vv) {
			char label[1000];
			
			sprintf(label, "first_%d", r);
			T.D1->write_xml(cout, label);
			}
		nb_sol = T.solve_first_system(verbose_level - 1, 
			point_types, nb_point_types, point_types_allocated);

		if (f_v) {
			cout << "r = " << r << ", found " << nb_sol
					<< " refined point types" << endl;
			}
		if (f_vv) {
			print_integer_matrix_width(cout,
				point_types + T.types_first[r] * point_type_len,
				nb_sol, point_type_len, point_type_len, 3);
			}
		
#if 0
		// MARUTA  Begin
		if (r == 1) {
			int h, a;

			for (h = nb_sol - 1; h >= 0; h--) {
				a = (point_types + (T.types_first[r] + h)
						* point_type_len)[0];
				if (a == 0) {
					cout << "removing last solution" << endl;
					nb_sol--;
					nb_point_types--;
					}
				}
			}
		// MARUTA   End
#endif


		if (f_vv) {
			print_integer_matrix_width(cout,
				point_types + T.types_first[r] * point_type_len,
				nb_sol, point_type_len, point_type_len, 3);
			}
		if (nb_sol == 0) {
			FREE_int(point_types);
			return FALSE;
			}

		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;
		
		if (nb_sol == 1) {
			if (f_v) {
				cout << "only one solution in block r=" << r << endl;
				}
			T.only_one_type[T.nb_only_one_type++] = r;
			}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
			}
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;

		} // next r
	
	// eliminate the slack variables from point_types:
	for (r = 0; r < nb_point_types; r++) {
		int f, l, a, j, J;
		
		for (i = 0; i < L1; i++) {
			f = P.startCell[i];
			l = P.cellSize[i];
			for (j = 0; j < l; j++) {
				J = f + i + j;
				a = point_types[r * point_type_len + J];
				point_types[r * L2 + f + j] = a;
				}
			}
		}
	point_type_len = L2;
	if (f_v) {
		cout << "altogether, we found " << nb_point_types
				<< " refined point types" << endl;
		}
	if (f_vv) {
		print_integer_matrix_width(cout, point_types,
			nb_point_types, point_type_len, point_type_len, 3);
		}
	

	// now we compute the distributions:

	if (!tdo_rows_setup_second_system(
		verbose_level,
		T, P, 
		f_omit1, omit1,
		f_use_packing_numbers,
		f_dual_is_linear_space,
		point_types, nb_point_types)) {
		FREE_int(point_types);
		return FALSE;
		}
	if (f_vv) {
		char label[1000];
			
		sprintf(label, "second");
		T.D2->write_xml(cout, label);
		}
	

	if (T.D2->n == 0) {
		distributions = NEW_int(1 * nb_point_types);
		nb_distributions = 0;
		for (h = 0; h < T.nb_only_one_type; h++) {
			r = T.only_one_type[h];
			u = T.types_first[r];
			//cout << "only one type, r=" << r << " u=" << u
			//<< " row_classes_len[ROW][r]="
			//<< row_classes_len[ROW][r] << endl;
			distributions[nb_distributions * nb_point_types + u] = 
				row_classes_len[ROW][r];
			}
		nb_distributions++;
		return TRUE;
		}
	

#if 0
	if (cnt_second_system == 1) {
		int j;
		int x[] = {4,1,5,0,2,0,7,2,4,0,0,0,1,0,0,4,0,0};
		cout << "testing solution:" << endl;
		int_vec_print(cout, x, 18);
		cout << endl;
		if (T.D2->n != 18) {
			cout << "T.D2->n != 18" << endl;
			}
		for (j = 0; j < 18; j++) {
			T.D2->x[j] = x[j];
			}
		T.D2->multiply_A_x_to_RHS1();
		for (i = 0; i < T.D2->m; i++) {
			cout << i << " : " << T.D2->RHS1[i] << " : "
					<< T.D2->RHS[i] - T.D2->RHS1[i] << endl;
			}
		}
#endif

	if (f_v) {
		cout << "refine_rows_hard: solving second system "
				<< cnt_second_system << " which is " << T.D2->m
				<< " x " << T.D2->n << endl;
		cout << T.nb_multiple_types << " variable blocks:" << endl;
		int f, l;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first2[i];
			l = T.types_len[r];
			cout << i << " : " << r << " : " << setw(3)
				<< row_classes_len[ROW][r] << " : " << setw(3) << f
				<< " : " << setw(3) << l << endl;
			}
		}
	if (f_omit) {
		T.solve_second_system_omit(verbose_level - 1, 
			row_classes_len[ROW], 
			point_types, nb_point_types,
			distributions, nb_distributions, omit);
		}
	else {
		int f_scale = FALSE;
		int scaling = 0;
		T.solve_second_system(verbose_level - 1,
			f_use_mckay, f_once,
			row_classes_len[ROW], f_scale, scaling, 
			point_types, nb_point_types,
			distributions, nb_distributions);
		}



	if (f_v) {
		cout << "refine_rows_hard: second system "
			<< cnt_second_system
			<< " found " << nb_distributions
			<< " distributions." << endl;
		}
	cnt_second_system++;
	return TRUE;
}

void tdo_scheme::row_refinement_L1_L2(partitionstack &P,
	int f_omit, int omit,
	int &L1, int &L2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l1, l2, omit2, i;
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += P.cellSize[i];
			}
		}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "row_refinement_L1_L2: l1 = " << l1 << " l2=" << l2
			<< " L1=" << L1 << " L2=" << L2 << endl;
		}
}

int tdo_scheme::tdo_rows_setup_first_system(int verbose_level, 
	tdo_data &T, int r, partitionstack &P, 
	int f_omit, int omit, 
	int *&point_types, int &nb_point_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int S, s_default, s_or_s_default, R, l1, l2, L1, L2;
	int J, r2, i, j, s, f, l;
	int nb_vars, nb_eqns;
	
	if (!f_omit)
		omit = 0;
	
	if (f_v) {
		cout << "tdo_rows_setup_first_system r=" << r << endl;
		if (f_omit) {
			cout << "omit=" << omit << endl;
			}
		}
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	nb_vars = L2 + L1; // possible up to L1 slack variables
	nb_eqns = R + L1;
		

	T.D1->open(nb_eqns, nb_vars);
	T.D1->fill_coefficient_matrix_with(0);
	S = 0;
		
	for (r2 = 0; r2 < R; r2++) {
		
		if (r2 == r) {
			// connections within the same row-partition
			for (i = 0; i < L1; i++) {
				f = P.startCell[i];
				l = P.cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) =
						minus_one_if_positive(
								the_col_scheme[r2 * l2 + f + j]);
					}
				T.D1->Aij(r2, f + i + l) = 0;
					// the slack variable is not needed
				}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) =
					minus_one_if_positive(the_col_scheme[r2 * l2 + J]);
				}
#endif
			T.D1->RHS[r] = row_classes_len[ROW][r] - 1;
			if (f_omit)
				T.D1->type[r] = t_LE;
			}
		else {
			// connections to the point from different row-partitions
			for (i = 0; i < L1; i++) {
				f = P.startCell[i];
				l = P.cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) = the_col_scheme[r2 * l2 + f + j];
					}
				T.D1->Aij(r2, f + i + l) = 0;
					// the slack variable is not needed
				}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) = the_col_scheme[r2 * l2 + J];
				}
#endif
			T.D1->RHS[r2] = row_classes_len[ROW][r2];
			if (f_omit)
				T.D1->type[r2] = t_LE;
			}
		}
		
	for (i = 0; i < L1; i++) {
		s = the_row_scheme[r * l1 + i];
		if (FALSE) {
			cout << "r=" << r << " i=" << i << " s=" << s << endl;
			}
		if (s == -1) {
			cout << "row scheme entry " << r << "," << i
				<< " is -1, using slack variable" << endl;
			cout << "using " << col_classes_len[ROW][i]
				<< " as upper bound" << endl;
			s_default = col_classes_len[ROW][i];
			s_or_s_default = s_default;
			}
		else {
			s_default = 0; // not needed but compiler likes it
			s_or_s_default = s;
			}
		
		T.D1->RHS[R + i] = s_or_s_default;
		S += s_or_s_default;
		
		f = P.startCell[i];
		l = P.cellSize[i];
		if (FALSE) {
			cout << "f=" << f << " l=" << l << endl;
			}
			
		for (j = 0; j < l; j++) {
			J = f + i + j; // +i for the slack variables
			T.D1->Aij(R + i, J) = 1;
			T.D1->x_max[J] = MINIMUM(col_classes_len[COL][f + j],
					s_or_s_default);
			}
		T.D1->Aij(R + i, f + i + l) = 1; // the slack variable
		if (s == -1) {
			T.D1->x_max[f + i + l] = s_default;
			}
		else {
			T.D1->x_max[f + i + l] = 0;
			}
		}
	T.D1->f_has_sum = TRUE;
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
	T.D1->eliminate_zero_rows_quick(verbose_level);
	
	if (f_v) {
		cout << "tdo_rows_setup_first_system r=" << r << " finished" << endl;
		}
	if (f_vv) {
		T.D1->print();
		}
	return TRUE;
		
}

int tdo_scheme::tdo_rows_setup_second_system(int verbose_level, 
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit,
	int f_use_packing_numbers,
	int f_dual_is_linear_space,
	int *&point_types, int &nb_point_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_eqns_joining, nb_eqns_counting, nb_eqns_packing, nb_eqns_used = 0;
	int Nb_vars, Nb_eqns;
	int l2, i, j, len, r, L1, L2;
	
	if (f_v) {
		cout << "tdo_rows_setup_second_system" << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		}

	l2 = nb_col_classes[COL];

	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_eqns_joining = L2 + binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_packing = 0;
	if (f_use_packing_numbers) {
		for (j = 0; j < L2; j++) {
			len = col_classes_len[COL][j];
			if (len > 2) {
				nb_eqns_packing += len - 2;
				}
			}
		}
	
	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_packing;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
		}
		

	T.D2->open(Nb_eqns, Nb_vars);
	T.D2->fill_coefficient_matrix_with(0);

#if 0
	if (Nb_vars == 0) {
		return TRUE;
		}
#endif

	if (f_v) {
		cout << "tdo_rows_setup_second_system: opening second system with " 
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_packing=" << nb_eqns_packing << endl;
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
		}

	if (!tdo_rows_setup_second_system_eqns_joining(verbose_level, 
		T, P, 
		f_omit, omit, f_dual_is_linear_space, 
		point_types, nb_point_types, 
		0 /*eqn_offset*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	if (!tdo_rows_setup_second_system_eqns_counting(verbose_level, 
		T, P, 
		f_omit, omit, 
		point_types, nb_point_types, 
		nb_eqns_joining /*eqn_offset*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	if (f_use_packing_numbers) {
		if (!tdo_rows_setup_second_system_eqns_packing(verbose_level, 
			T, P, 
			f_omit, omit, 
			point_types, nb_point_types,
			nb_eqns_joining + nb_eqns_counting /* eqn_start */,
			nb_eqns_used)) {
			if (f_v) {
				T.D2->print();
				}
			return FALSE;
			}
		}
	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_used;
	T.D2->m = Nb_eqns;

	T.D2->eliminate_zero_rows_quick(verbose_level);


	
	if (f_v) {
		cout << "tdo_rows_setup_second_system finished" << endl;
		}
	if (f_vv) {
		T.D2->print();
		}
	return TRUE;
}

int tdo_scheme::tdo_rows_setup_second_system_eqns_joining(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, int f_dual_is_linear_space, 
	int *point_types, int nb_point_types, 
	int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int l2, I1, I2, k, b, ab, i, j, r, I, J;
	int f, l, c, a, a2, rr, p, u, h, L1, L2;
	char label[100];
	
	if (f_v) {
		cout << "tdo_scheme::tdo_rows_setup_second_system_"
				"eqns_joining" << endl;
		}
	l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	if (f_v) {
		cout << "l2 = " << l2 << endl;
		cout << "L2 = " << L2 << endl;
		cout << "eqn_offset = " << eqn_offset << endl;
		cout << "T.nb_multiple_types = " << T.nb_multiple_types << endl;
		}

	for (I = 0; I < L2; I++) {
		sprintf(label, "J_{%d}", I + 1);
		T.D2->init_eqn_label(eqn_offset + I, label);
		}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = ij2k(I1, I2, L2);
			sprintf(label, "J_{%d,%d}", I1 + 1, I2 + 1);
			T.D2->init_eqn_label(eqn_offset + L2 + k, label);
			}
		}
	if (f_v) {
		cout << "filling coefficient matrix" << endl;
		}
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = point_types[c * L2 + I];
				a2 = binomial2(a);
				T.D2->Aij(eqn_offset + I, J) = a2;
				}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = ij2k(I1, I2, L2);
					a = point_types[c * L2 + I1];
					b = point_types[c * L2 + I2];
					ab = a * b;
					T.D2->Aij(eqn_offset + L2 + k, J) = ab;
					}
				}
			}
		}
		
	if (f_v) {
		cout << "filling RHS" << endl;
		}
	for (I = 0; I < L2; I++) {
		a = col_classes_len[COL][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2;
		if (f_dual_is_linear_space) {
			T.D2->type[eqn_offset + I] = t_EQ;
			}
		else {
			T.D2->type[eqn_offset + I] = t_LE;
			}
		}
	for (I1 = 0; I1 < L2; I1++) {
		a = col_classes_len[COL][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = col_classes_len[COL][I2];
			k = ij2k(I1, I2, L2);
			T.D2->RHS[eqn_offset + L2 + k] = a * b;
			if (f_dual_is_linear_space) {
				T.D2->type[eqn_offset + L2 + k] = t_EQ;
				}
			else {
				T.D2->type[eqn_offset + L2 + k] = t_LE;
				}
			}
		}
	if (f_v) {
		cout << "subtracting contribution from one-type blocks:" << endl;
		}
	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = row_classes_len[ROW][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = point_types[u * L2 + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_vv) {
					cout << "tdo_rows_setup_second_system_eqns_joining: "
							"RHS is negative, no solution for the "
							"distribution" << endl;
					cout << "h=" << h << endl;
					cout << "rr=T.only_one_type[h]=" << rr << endl;
					cout << "p=row_classes_len[ROW][rr]=" << p << endl;
					cout << "u=T.types_first[rr]="
							<< T.types_first[rr] << endl;
					cout << "I=" << I << endl;
					cout << "a=point_types[u * L2 + I]=" << a << endl;
					cout << "a2=binomial2(a)=" << a2 << endl;
					cout << "T.D2->RHS[eqn_offset + I]="
							<< T.D2->RHS[eqn_offset + I] << endl;
					}
				return FALSE;
				}
			}
		for (I1 = 0; I1 < L2; I1++) {
			a = point_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = point_types[u * L2 + I2];
				k = ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_offset + L2 + k] -= ab;
				if (T.D2->RHS[eqn_offset + L2 + k] < 0) {
					if (f_vv) {
						cout << "tdo_rows_setup_second_system_eqns_"
								"joining: RHS is negative, no solution "
								"for the distribution" << endl;
						cout << "h=" << h << endl;
						cout << "rr=T.only_one_type[h]=" << rr << endl;
						cout << "p=row_classes_len[ROW][rr]=" << p << endl;
						cout << "u=T.types_first[rr]="
								<< T.types_first[rr] << endl;
						cout << "I1=" << I1 << endl;
						cout << "I2=" << I2 << endl;
						cout << "k=" << k << endl;
						cout << "a=point_types[u * L2 + I1]=" << a << endl;
						cout << "b=point_types[u * L2 + I2]=" << b << endl;
						cout << "ab=" << ab << endl;
						cout << "T.D2->RHS[eqn_offset + L2 + k]="
							<< T.D2->RHS[eqn_offset + L2 + k] << endl;
						}
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

int tdo_scheme::tdo_rows_setup_second_system_eqns_counting(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, 
	int *point_types, int nb_point_types, 
	int eqn_offset)
{
	int l2, b, i, j, r, I, J, f, l, c, a, S, s, L1, L2;
	char label[100];
	//int nb_vars = T.D1->n;
	
	l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				sprintf(label, "F_{%d,%d}", I+1, r+1);
				T.D2->init_eqn_label(eqn_offset + i * (L2 + 1) + I, label);
				}
			}
		sprintf(label, "F_{%d}", r+1);
		T.D2->init_eqn_label(eqn_offset + i * (L2 + 1) + l2, label);
		}
	
	// equations counting flags
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = point_types[c * L2 + I];
				T.D2->Aij(eqn_offset + i * (L2 + 1) + I, J) = a;
				}
			T.D2->Aij(eqn_offset + i * (L2 + 1) + L2, J) = 1;
			}
		}
	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		for (I = 0; I < L2; I++) {
			a = the_col_scheme[r * nb_col_classes[COL] + I];
			b = col_classes_len[COL][I];
			T.D2->RHS[eqn_offset + i * (L2 + 1) + I] = a * b;
			}
		s = row_classes_len[ROW][r];
		T.D2->RHS[eqn_offset + i * (L2 + 1) + L2] = s;
		S += s;
		}
	
	T.D2->f_has_sum = TRUE;
	T.D2->sum = S;
	//T.D2->f_x_max = TRUE;
	return TRUE;
}

int tdo_scheme::tdo_rows_setup_second_system_eqns_packing(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, 
	int *point_types, int nb_point_types,
	int eqn_start, int &nb_eqns_used)
{
	int f_v = (verbose_level >= 1);
	int nb_eqns_packing;
	int /*l2,*/ i, r, f, l, j, c, J, JJ, k, h;
	int rr, p, u, a, len, f_used, L1, L2;
	char label[100];
	//int nb_vars = T.D1->n;
	
	//l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_eqns_packing = 0;
	for (J = 0; J < L2; J++) {
		len = col_classes_len[COL][J];
		if (len <= 2)
			continue;
		for (k = 3; k <= len; k++) {
			f_used = FALSE;
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					a = point_types[c * L2 + J];
					if (a < k)
						continue;
					JJ = T.types_first2[i] + j;
					f_used = TRUE;
					T.D2->Aij(eqn_start + nb_eqns_packing, JJ) = 1;
					}
				} // next i
			if (f_used) {
				int bound;
				bound = TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = row_classes_len[COL][rr];
					u = T.types_first[rr];
					a = point_types[u * L2 + J];
					if (a < k)
						continue;
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_scheme::tdo_rows_setup_second_"
								"system_eqns_packing RHS < 0" << endl;
							}
						return FALSE;
						}
					}
				sprintf(label, "P_{%d,%d} \\,\\mbox{using}\\, "
						"P(%d,%d)=%d", J + 1, k, len, k, bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				if (f_v) {
					cout << "packing equation " << nb_eqns_packing
							<< " J=" << J << " k=" << k
							<< " len=" << len << endl;
					}
				nb_eqns_packing++;
				}
			} // next k
		}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_rows_setup_second_system_eqns_packing "
				"nb_eqns_used = " << nb_eqns_used << endl;
		}
	return TRUE;
}

// #############################################################################
// parameter refinement: refine columns
// #############################################################################

int tdo_scheme::refine_columns(int verbose_level,
	int f_once, partitionstack &P,
	int *&line_types, int &nb_line_types, int &line_type_len, 
	int *&distributions, int &nb_distributions, 
	int &cnt_second_system, solution_file_data *Sol, 
	int f_omit1, int omit1, int f_omit, int omit, 
	int f_D1_upper_bound_x0, int D1_upper_bound_x0, 
	int f_use_mckay_solver, 
	int f_use_packing_numbers)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f_easy;
	int l1, l2, R;
	int ret = FALSE;
	
	if (f_v) {
		cout << "tdo_scheme::refine_columns" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_D1_upper_bound_x0=" << f_D1_upper_bound_x0 << endl;
		cout << "f_use_mckay_solver=" << f_use_mckay_solver << endl;
		}
	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
	if (f_vv) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "R=" << R << endl;
		}
	
	get_row_split_partition(0 /*verbose_level*/, P);
	if (f_vv) {
		cout << "tdo_scheme::refine_columns "
				"row split partition: " << P << endl;
		}
	if (P.ht != l1) {
		cout << "P.ht != l1" << endl;
		}

	if ((R == 1) && (l1 == 1) && (the_col_scheme[0] == -1)) {
		f_easy = TRUE;
		if (FALSE) {
			cout << "easy mode" << endl;
			}
		}
	else {
		f_easy = FALSE;
		if (FALSE) {
			cout << "full mode" << endl;
			}
		}


	if (f_easy) {
		cout << "tdo_scheme::refine_columns "
				"refine_cols_easy nyi" << endl;
		exit(1);
		
		}
	else {
		ret = refine_cols_hard(P, verbose_level - 1, f_once, 
			line_types, nb_line_types, line_type_len, 
			distributions, nb_distributions, cnt_second_system, Sol, 
			f_omit1, omit1, f_omit, omit, 
			f_D1_upper_bound_x0, D1_upper_bound_x0, 
			f_use_mckay_solver, 
			f_use_packing_numbers);
		}
	if (f_v) {
		cout << "tdo_scheme::refine_columns finished" << endl;
		}
	return ret;
}

int tdo_scheme::refine_cols_hard(partitionstack &P,
	int verbose_level, int f_once,
	int *&line_types, int &nb_line_types, int &line_type_len,  
	int *&distributions, int &nb_distributions, 
	int &cnt_second_system, solution_file_data *Sol, 
	int f_omit1, int omit1, int f_omit, int omit, 
	int f_D1_upper_bound_x0, int D1_upper_bound_x0, 
	int f_use_mckay_solver, 
	int f_use_packing_numbers)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int nb_eqns, nb_vars;
	int R, /*l1,*/ l2, L1, L2, r;
	int line_types_allocated;
	int nb_sol, nb_sol1, f_survive;
	{
	tdo_data T;
	int i, j, u;

	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_D1_upper_bound_x0=" << f_D1_upper_bound_x0 << endl;
		cout << "f_use_mckay_solver=" << f_use_mckay_solver << endl;
		}
	R = nb_col_classes[COL];
	//l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];

	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard "
				"the_row_scheme is:" << endl;
		for (i = 0; i < l2; i++) {
			for (j = 0; j < R; j++) {
				cout << setw(4) << the_row_scheme[i * R + j];
				}
			cout << endl;
			}
		}

	column_refinement_L1_L2(P, f_omit1, omit1,
			L1, L2, verbose_level);

	T.allocate(R);
	
	T.types_first[0] = 0;
	
	line_types_allocated = 100;
	nb_line_types = 0;
	line_types = NEW_int(line_types_allocated * l2);
	line_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	
	for (r = 0; r < R; r++) {
		
		if (f_v) {
			cout << "r=" << r << endl;
			}
		
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"before tdo_columns_setup_first_system" << endl;
		}
		if (!tdo_columns_setup_first_system(verbose_level, 
			T, r, P, 
			f_omit1, omit1, 
			line_types, nb_line_types)) {
			if (f_v) {
				cout << "tdo_scheme::refine_cols_hard "
						"tdo_columns_setup_first_system returns FALSE" << endl;
			}
			FREE_int(line_types);
			return FALSE;
			}
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"after tdo_columns_setup_first_system" << endl;
		}
		
		if (f_D1_upper_bound_x0) {
			T.D1->x_max[0] = D1_upper_bound_x0;
			cout << "setting upper bound for D1->x[0] to "
					<< T.D1->x_max[0] << endl;
			} 


#if 0
		// ATTENTION, this is from a specific problem
		// on arcs in a plane (MARUTA)
		
		// now we are interested in (42,6)_8 arcs
		// a line intersects the arc in at most 6 points:
		//T.D1->x_max[0] = 6;
		
		// now we are interested in (33,5)_8 arcs
		//T.D1->x_max[0] = 5;
		//cout << "ATTENTION: MARUTA, limiting x_max[0] to 5" << endl;
		
		// now we are interested in (49,7)_8 arcs
		//T.D1->x_max[0] = 7;
		//cout << "ATTENTION: MARUTA, limiting x_max[0] to 7" << endl;
		
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif
		
		
		if (f_vv) {
			char label[1000];
			
			sprintf(label, "first_%d", r);
			T.D1->write_xml(cout, label);
			}
				
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"before T.solve_first_system" << endl;
		}
		nb_sol = T.solve_first_system(verbose_level - 1, 
			line_types, nb_line_types, line_types_allocated);
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"after T.solve_first_system" << endl;
		}

		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"r = " << r << ", found " << nb_sol
					<< " refined line types" << endl;
			}
		if (f_vv) {
			print_integer_matrix_width(cout,
				line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}
		nb_sol1 = 0;
		for (u = 0; u < nb_sol; u++) {
			f_survive = TRUE;
			for (i = 0; i < L2; i++) {
				int len1, len2, flags;
				len1 = row_classes_len[ROW][i];
				if (len1 > 1)
					continue;
				len2 = col_classes_len[ROW][r];
				flags = the_row_scheme[i * R + r];
				if (flags == len2) {
					if (line_types[(T.types_first[r] + u) * L2 + i] == 0) {
						f_survive = FALSE;
						if (f_vv) {
							cout << "line type " << u << " eliminated, "
									"line_types[] = 0" << endl;
							cout << "row block " << i << endl;
							cout << "col block=" << r << endl;
							cout << "length of col block " << len2 << endl;
							cout << "flags " << flags << endl;
							
							}
						break;
						}
					}
				}
			if (f_survive) {
				for (i = 0; i < L2; i++) {
					line_types[(T.types_first[r] + nb_sol1) * L2 + i] = 
						line_types[(T.types_first[r] + u) * L2 + i];
					}
				nb_sol1++;
				}
			}
		if (nb_sol1 < nb_sol) {
			if (f_v) {
				cout << "tdo_scheme::refine_cols_hard "
						"eliminated " << nb_sol - nb_sol1
						<< " types" << endl;
				}
			nb_sol = nb_sol1;
			nb_line_types = T.types_first[r] + nb_sol1;
			if (f_v) {
				cout << "tdo_scheme::refine_cols_hard "
						"r = " << r << ", found " << nb_sol
						<< " refined line types" << endl;
				}
			}
		
		if (f_vv) {
			print_integer_matrix_width(cout,
				line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}
		if (nb_sol == 0) {
			FREE_int(line_types);
			return FALSE;
			}
		
		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;
		
		if (nb_sol == 1) {
			if (f_v) {
				cout << "tdo_scheme::refine_cols_hard "
						"only one solution in block "
					"r=" << r << endl;
				}
			T.only_one_type[T.nb_only_one_type++] = r;
			}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
			}
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;
		
		} // next r
	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard "
				"R=" << R << endl;
		cout << "r : T.types_first[r] : T.types_len[r]" << endl;
		for (r = 0; r < R; r++) {
			cout << r << " : " << T.types_first[r] << " : "
					<< T.types_len[r] << endl;
			}
		}
	if (f_vv) {
		print_integer_matrix_width(cout, line_types,
				nb_line_types, line_type_len, line_type_len, 3);
		}
	
	// now we compute the distributions:
	//
	int f_scale = FALSE;
	int scaling = 0;
	
	if (!tdo_columns_setup_second_system(verbose_level, 
		T, P, 
		f_omit1, omit1, 
		f_use_packing_numbers, 
		line_types, nb_line_types)) {
		FREE_int(line_types);
		return FALSE;
		}


#if 0
	// ATTENTION, this is for the classification
	// of (42,6)_8 arcs where a_1 = 0 (MARUTA)
		
	//T.D2->x_max[5] = 0; // a_1 is known to be zero
		
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif



	if (f_vv) {
		char label[1000];
			
		sprintf(label, "second");
		T.D2->write_xml(cout, label);
		}




	int idx, /*f,*/ l;
	idx = 0;
	for (r = 0; r < R; r++) {
		l = T.types_len[r];
		if (l > 1) {
			if (T.multiple_types[idx] != r) {
				cout << "T.multiple_types[idx] != r" << endl;
				exit(1);
				}
			//f = T.types_first2[idx];
			idx++;
			}
		else {
			//f = -1;
			}
		}

	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard "
				"solving second system "
				<< cnt_second_system << " which is " << T.D2->m
				<< " x " << T.D2->n << endl;
		cout << "variable blocks:" << endl;
		cout << "i : r : col_classes_len[COL][r] : types_first2[i] : "
			"types_len[r]" << endl;
		int f, l;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first2[i];
			l = T.types_len[r];
			cout << i << " : " << r << " : " << setw(3)
				<< col_classes_len[COL][r] << " : " << setw(3)
				<< f << " : " << setw(3) << l << endl;
			}
		}

	if (f_omit) {
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"before T.solve_second_system_omit" << endl;
		}
		T.solve_second_system_omit(verbose_level, 
			col_classes_len[COL], 
			line_types, nb_line_types, distributions, nb_distributions, 
			omit);
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"after T.solve_second_system_omit" << endl;
		}
		}
	else {
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"before T.solve_second_system_with_help" << endl;
		}
		T.solve_second_system_with_help(verbose_level, 
			f_use_mckay_solver, f_once, 
			col_classes_len[COL], f_scale, scaling, 
			line_types, nb_line_types, distributions, nb_distributions, 
			cnt_second_system, Sol);
		if (f_v) {
			cout << "tdo_scheme::refine_cols_hard "
					"after T.solve_second_system_with_help" << endl;
		}
		}

	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard "
				"second system " << cnt_second_system
			<< " found " << nb_distributions << " distributions." << endl;
		}

#if 0
	// ATTENTION: this is from a specific problem of CHEON !!!!

	cout << "ATTENTION, we are running specific code "
			"for a problem of Cheon" << endl;
	int cnt, h, x0;
	
	cnt = 0;
	for (h = 0; h < nb_distributions; h++) {
		x0 = distributions[h * nb_line_types + 0];
		if (x0 == 12) {
			for (j = 0; j < nb_line_types; j++) {
				distributions[cnt * nb_line_types + j] =
						distributions[h * nb_line_types + j];
				}
			cnt++;
			}
		if (x0 > 12) {
			cout << "x0 > 12, something is wrong" << endl;
			exit(1);
			}
		}
	cout << "CHEON: we found " << cnt << " refinements with x0=12" << endl;
	nb_distributions = cnt;

	// ATTENTION
#endif

	cnt_second_system++;
	}
	if (f_v) {
		cout << "tdo_scheme::refine_cols_hard after closing T." << endl;
		}
	return TRUE;
}

void tdo_scheme::column_refinement_L1_L2(
	partitionstack &P, int f_omit, int omit,
	int &L1, int &L2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l1, l2, omit2, i;
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW]; // the finer scheme

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += P.cellSize[i];
			}
		}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "tdo_scheme::column_refinement_L1_L2 "
				"l1 = " << l1
			<< " l2=" << l2 << " L1=" << L1 << " L2=" << L2 << endl;
		}
}

int tdo_scheme::tdo_columns_setup_first_system(int verbose_level, 
	tdo_data &T, int r, partitionstack &P, 
	int f_omit, int omit, 
	int *&line_types, int &nb_line_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, f, l, I, J, rr, R, S, a, a2, s, /*l1, l2,*/ L1, L2;
	int h, u, d, d2, o, e, p, eqn_number, nb_vars, nb_eqns;
	
	// create all partitions which are refined line types

	if (!f_omit)
		omit = 0;

	if (f_v) {
		cout << "tdo_scheme::tdo_columns_setup_first_system "
				"r=" << r << endl;
		if (f_omit) {
			cout << "omit=" << omit << endl;
			}
		}
		
	R = nb_col_classes[COL];
	//l1 = nb_row_classes[COL];
	//l2 = nb_row_classes[ROW]; // the finer scheme

	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_vars = L2;
	nb_eqns = L1 + 1 + (R - 1);
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
	
	// the m equalities that come from the fact that the n e w type
	// is a partition of the old type.
	
	// we are in the r-th column class (r is given)
	
	for (I = 0; I < L1; I++) {
		f = P.startCell[I];
		l = P.cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->Aij(I, J) = 1;
			a = the_row_scheme[J * R + r];
			if (a == 0)
				T.D1->x_max[J] = 0;
			else
				T.D1->x_max[J] = row_classes_len[ROW][J];
			}
		s = the_col_scheme[I * R + r];
		T.D1->RHS[I] = s;
		T.D1->type[I] = t_EQ;
		S += s;
		}
	
	eqn_number = L1;
	
	for (i = 0; i < L2; i++) {
		a = minus_one_if_positive(the_row_scheme[i * R + r]);
		T.D1->Aij(eqn_number, i) = a;
		}
	T.D1->RHS[eqn_number] = col_classes_len[ROW][r] - 1;
			// the -1 was missing!!!
	T.D1->type[eqn_number] = t_LE;
	eqn_number++;
	
	for (j = 0; j < R; j++) {
		if (j == r)
			continue;
		for (i = 0; i < L2; i++) {
			a = the_row_scheme[i * R + j];
			T.D1->Aij(eqn_number, i) = a;
			}
		T.D1->RHS[eqn_number] = col_classes_len[ROW][j];
		T.D1->type[eqn_number] = t_LE;
		eqn_number++;
		}
	T.D1->m = eqn_number;


	// try to reduce the upper bounds:
		
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		u = T.types_first[rr];
		//cout << "u=" << u << endl;
		for (j = 0; j < nb_vars; j++) {
			//cout << "j=" << j << endl;
			if (T.D1->x_max[j] == 0)
				continue;
			d = row_classes_len[ROW][j];
			p = col_classes_len[COL][rr];
			d2 = binomial2(d);
			a = line_types[u * nb_vars + j];
			a2 = binomial2(a);
			o = d2 - a2 * p;
			if (o < 0) {
				if (f_vv) {
					cout << "only one type, but no solution because of "
						"joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			e = largest_binomial2_below(o);
			T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
			}
		}

		
	T.D1->f_has_sum = TRUE;
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;

	T.D1->eliminate_zero_rows_quick(verbose_level);
		
	if (f_v) {
		cout << "tdo_scheme::tdo_columns_setup_first_system "
				"r=" << r << " finished" << endl;
		}
	if (f_vv) {
		T.D1->print();
		}
	return TRUE;
}	

int tdo_scheme::tdo_columns_setup_second_system(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit,  
	int f_use_packing_numbers, 
	int *&line_types, int &nb_line_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, len, r, I, J, L1, L2, Nb_eqns, Nb_vars;

	if (f_v) {
		cout << "tdo_scheme::tdo_columns_setup_second_system" << endl;
		cout << "f_use_packing_numbers="
				<< f_use_packing_numbers << endl;
		}
		
	int nb_eqns_joining, nb_eqns_counting;
	int nb_eqns_upper_bound, nb_eqns_used;
	int l2;
	
	l2 = nb_row_classes[ROW]; // the finer scheme

	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	nb_eqns_joining = L2 + binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_upper_bound = 0;
	if (f_use_packing_numbers) {
		for (i = 0; i < l2; i++) {
			len = row_classes_len[ROW][i];
			if (len > 2) {
				nb_eqns_upper_bound += len - 2;
				}
			}
		}
	
	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_upper_bound;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
		}

	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "tdo_scheme::tdo_columns_setup_second_system "
				"opening second system with "
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
		}
	if (f_vv) {
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_upper_bound=" << nb_eqns_upper_bound << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
		cout << "i : r = T.multiple_types[i] : T.types_first2[i] "
				": T.types_len[r]" << endl;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			cout << i << " : " << r << " : " << T.types_first2[i]
				<< " : " << T.types_len[r] << endl;
			}
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;

	if (!tdo_columns_setup_second_system_eqns_joining(verbose_level, 
		T, P, 
		f_omit, omit, 
		line_types, nb_line_types,
		0 /*eqn_start*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	tdo_columns_setup_second_system_eqns_counting(verbose_level, 
		T, P, 
		f_omit, omit, 
		line_types, nb_line_types,
		nb_eqns_joining /* eqn_start */);
	if (f_use_packing_numbers) {
		if (!tdo_columns_setup_second_system_eqns_upper_bound(
			verbose_level,
			T, P, 
			f_omit, omit, 
			line_types, nb_line_types,
			nb_eqns_joining + nb_eqns_counting /* eqn_start */,
			nb_eqns_used)) {
			if (f_v) {
				T.D2->print();
				}
			return FALSE;
			}
		}
	
	
	
	T.D2->eliminate_zero_rows_quick(verbose_level);
	
	
	if (f_vv) {	
		cout << "tdo_scheme::tdo_columns_setup_second_system, "
				"The second system is" << endl;
		T.D2->print();
		}
	if (f_v) {
		cout << "tdo_scheme::tdo_columns_setup_second_system "
				"finished" << endl;
		}
	return TRUE;
}

int tdo_scheme::tdo_columns_setup_second_system_eqns_joining(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, 
	int *line_types, int nb_line_types,
	int eqn_start)
{
	int f_v = (verbose_level >= 1);
	int l2, L1, L2, i, r, f, l, j, c;
	int J, I, I1, I2, a, b, ab, a2, k, h, rr, p, u;
	char label[100];
	
	l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	for (I = 0; I < L2; I++) {
		sprintf(label, "J_{%d}", I + 1);
		T.D2->init_eqn_label(eqn_start + I, label);
		}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = ij2k(I1, I2, L2);
			sprintf(label, "J_{%d,%d}", I1 + 1, I2 + 1);
			T.D2->init_eqn_label(eqn_start + L2 + k, label);
			}
		}
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = line_types[c * L2 + I];
				a2 = binomial2(a);
				T.D2->Aij(eqn_start + I, J) = a2;
				}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = ij2k(I1, I2, L2);
					a = line_types[c * L2 + I1];
					b = line_types[c * L2 + I2];
					ab = a * b;
					T.D2->Aij(eqn_start + L2 + k, J) = ab;
					}
				}
			}
		}

	// prepare RHS:
	
	for (I = 0; I < L2; I++) {
		a = row_classes_len[ROW][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_start + I] = a2;
		}
	for (I1 = 0; I1 < L2; I1++) {
		a = row_classes_len[ROW][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = row_classes_len[ROW][I2];
			k = ij2k(I1, I2, L2);
			T.D2->RHS[eqn_start + l2 + k] = a * b;
			}
		}
	
	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = col_classes_len[COL][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = line_types[u * L2 + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_start + I] -= a2 * p;
			if (T.D2->RHS[eqn_start + I] < 0) {
				if (f_v) {
					cout << "tdo_columns_setup_second_system_eqns_"
							"joining: RHS is negative, no solution for "
							"the distribution" << endl;
					}
				return FALSE;
				}
			}
		for (I1 = 0; I1 < L2; I1++) {
			a = line_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = line_types[u * L2 + I2];
				k = ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_start + L2 + k] -= ab;
				if (T.D2->RHS[eqn_start + L2 + k] < 0) {
					if (f_v) {
						cout << "tdo_columns_setup_second_system_eqns_"
								"joining: RHS is negative, no solution for "
								"the distribution" << endl;
						}
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

void tdo_scheme::tdo_columns_setup_second_system_eqns_counting(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, 
	int *line_types, int nb_line_types,
	int eqn_start)
{
	int /*l2,*/ L1, L2, i, r, f, l, j, c, J, I, a, b, S, s;
	char label[100];

	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				sprintf(label, "F_{%d,%d}", r + 1, I + 1);
				T.D2->init_eqn_label(eqn_start + i * (L2 + 1) + I, label);
				}
			}
		sprintf(label, "F_{%d}", r + 1);
		T.D2->init_eqn_label(eqn_start + i * (L2 + 1) + L2, label);
		}

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = line_types[c * L2 + I];
				T.D2->Aij(eqn_start + i * (L2 + 1) + I, J) = a;
				}
			T.D2->Aij(eqn_start + i * (L2 + 1) + L2, J) = 1;
			}
		}
	
	// set upper bound x_max:
	
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		s = col_classes_len[ROW][r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			T.D2->x_max[J] = s;
			}
		}

	// prepare RHS:
	
	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		for (I = 0; I < L2; I++) {
			a = the_row_scheme[I * nb_col_classes[ROW] + r];
			b = row_classes_len[ROW][I];
			T.D2->RHS[eqn_start + i * (L2 + 1) + I] = a * b;
			}
		s = col_classes_len[ROW][r];
		T.D2->RHS[eqn_start + i * (L2 + 1) + L2] = s;
		S += s;
		}

	T.D2->f_has_sum = TRUE;
	T.D2->sum = S;
	T.D2->f_x_max = TRUE;
}

int tdo_scheme::tdo_columns_setup_second_system_eqns_upper_bound(
	int verbose_level,
	tdo_data &T, partitionstack &P, 
	int f_omit, int omit, 
	int *line_types, int nb_line_types,
	int eqn_start, int &nb_eqns_used)
{
	int f_v = (verbose_level >= 1);
	int nb_eqns_packing;
	int /*l2,*/ L1, L2, i, r, f, l, j, c, J, I;
	int k, h, rr, p, u, a, len, f_used;
	char label[100];
	
	nb_eqns_packing = 0;
	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	for (I = 0; I < L2; I++) {
		len = row_classes_len[ROW][I];
		if (len <= 2)
			continue;
		for (k = 3; k <= len; k++) {
			f_used = FALSE;
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * L2 + I];
					if (a < k)
						continue;
					f_used = TRUE;
					T.D2->Aij(eqn_start + nb_eqns_packing, J) = 1;
					}
				} // next i
			if (f_used) {
				int bound;
				
				bound = TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = col_classes_len[ROW][rr];
					u = T.types_first[rr];
					a = line_types[u * L2 + I];
					if (a < k)
						continue;
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_scheme::tdo_columns_setup_"
									"second_system_eqns_upper_bound "
									"RHS < 0" << endl;
							}
						return FALSE;
						}
					}
				sprintf(label, "P_{%d,%d} \\,\\mbox{using}\\, P(%d,%d)=%d",
						I + 1, k, len, k, bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				nb_eqns_packing++;
				}
			} // next k
		}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_columns_setup_second_system_eqns_upper_bound "
				"nb_eqns_used = " << nb_eqns_used << endl;
		}
	return TRUE;
}

// #############################################################################
// TDO parameter refinement for 3-designs - row refinement
// #############################################################################


int tdo_scheme::td3_refine_rows(int verbose_level, int f_once,
	int lambda3, int block_size,
	int *&point_types, int &nb_point_types, int &point_type_len,  
	int *&distributions, int &nb_distributions)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int R, /*l1,*/ l2, r;
	int nb_eqns, nb_vars = 0;
	int point_types_allocated;
	partitionstack P;
	int lambda2;
	int nb_points;
	int nb_sol;
	tdo_data T;

	if (f_v) {
		cout << "tdo_scheme::td3_refine_rows" << endl;
		}
	nb_points = m;
	lambda2 = lambda3 * (nb_points - 2) / (block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points
				<< " lambda2 = " << lambda2 << endl;
		}
	if ((block_size - 2) * lambda2 != lambda3 * (nb_points - 2)) {
		cout << "parameters are wrong" << endl;
		exit(1);
		}

	get_column_split_partition(verbose_level, P);
	
	R = nb_row_classes[ROW];
	//l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	
	T.allocate(R);
	
	T.types_first[0] = 0;
	
	point_types_allocated = 100;
	nb_point_types = 0;
	point_types = NEW_int(point_types_allocated * l2);
	point_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_vvv) {
			cout << "r=" << r << endl;
			}
		if (!td3_rows_setup_first_system(verbose_level - 1, 
			lambda3, block_size, lambda2, 
			T, r, P, 
			nb_vars, nb_eqns, 
			point_types, nb_point_types)) {
			FREE_int(point_types);
			return FALSE;
			}
		
		nb_sol = T.solve_first_system(verbose_level - 1, 
			point_types, nb_point_types, point_types_allocated);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol
				<< " refined point types" << endl;
			}
		if (nb_sol == 0) {
			FREE_int(point_types);
			return FALSE;
			}
		
		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;
		
		if (nb_sol == 1) {
			if (f_vv) {
				cout << "only one solution in block r=" << r << endl;
				}
			T.only_one_type[T.nb_only_one_type++] = r;
			}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
			}
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;
		
		} // next r
	
	
	// now we compute the distributions:
	//
	int Nb_vars, Nb_eqns;

	if (!td3_rows_setup_second_system(verbose_level, 
		lambda3, block_size, lambda2, 
		T, 
		nb_vars, Nb_vars, Nb_eqns, 
		point_types, nb_point_types)) {
		FREE_int(point_types);
		return FALSE;
		}
	
	if (Nb_vars == 0) {
		int h, r, u;
		
		distributions = NEW_int(1 * nb_point_types);
		nb_distributions = 0;
		for (h = 0; h < T.nb_only_one_type; h++) {
			r = T.only_one_type[h];
			u = T.types_first[r];
			//cout << "only one type, r=" << r << " u=" << u
			//<< " row_classes_len[ROW][r]="
			//<< row_classes_len[ROW][r] << endl;
			distributions[nb_distributions * nb_point_types + u] = 
				row_classes_len[ROW][r];
			}
		nb_distributions++;
		return TRUE;
		}

	int f_scale = FALSE;
	int scaling = 0;
	
	T.solve_second_system(verbose_level - 1, FALSE /* f_use_mckay */,f_once, 
		row_classes_len[ROW], f_scale, scaling, 
		point_types, nb_point_types, distributions, nb_distributions);


	if (f_v) {
		cout << "tdo_scheme::td3_refine_rows "
				"found " << nb_distributions
				<< " distributions." << endl;
		}
	return TRUE;
}

int tdo_scheme::td3_rows_setup_first_system(int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, int r, partitionstack &P,
	int &nb_vars, int &nb_eqns,
	int *&point_types, int &nb_point_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, j, R, l1, l2, r2, r3, S, I, J, f, l, s;
	int eqn_offset, eqn_cnt;
	
	if (f_v) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << endl;
		}
		

	// create all partitions which are refined
	// point types of points in block r

	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];
		
	nb_vars = l2;
	nb_eqns = R + R + (R - 1) + (((R - 1) * (R - 2)) >> 1) + l1;
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
	for (I = 0; I < nb_eqns; I++) 
		T.D1->RHS[I] = 9999;
	
	// pair joinings
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			// connections within the same row-partition
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] =
					minus_one_if_positive(the_col_scheme[r2 * l2 + J]);
				}
			T.D1->RHS[r2] = (row_classes_len[ROW][r2] - 1) * lambda2;
			}
		else {
			// connections to the point from different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] = the_col_scheme[r2 * l2 + J];
				}
			T.D1->RHS[r2] = row_classes_len[ROW][r2] * lambda2;
			}
		}
	if (f_vv) {
		cout << "r=" << r << " after pair joining, the system is" << endl;
		T.D1->print();
		}

	// triple joinings
	eqn_offset = R;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			// connections to pairs within the same row-partition
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + r2) * nb_vars + J] =
					binomial2(minus_one_if_positive(
							the_col_scheme[r2 * l2 + J]));
				}
			T.D1->RHS[eqn_offset + r2] =
				binomial2((row_classes_len[ROW][r2] - 1)) * lambda3;
			}
		else {
			// connections to pairs with one
			// in the same and one in the other part
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + r2) * nb_vars + J] =
					minus_one_if_positive(the_col_scheme[r * l2 + J])
						* the_col_scheme[r2 * l2 + J];
				}
			T.D1->RHS[eqn_offset + r2] =
				(row_classes_len[ROW][r] - 1) *
					row_classes_len[ROW][r2] * lambda3;
			}
		}
	if (f_vv) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << " after triple joining, the system is" << endl;
		T.D1->print();
		}
	
	eqn_offset += R;
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r)
			continue;
		// connections to pairs from one different row-partition
		for (J = 0; J < nb_vars; J++) {
			T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] =
				binomial2(the_col_scheme[r2 * l2 + J]);
			}
		T.D1->RHS[eqn_offset + eqn_cnt] = binomial2(
				row_classes_len[ROW][r2]) * lambda3;
		eqn_cnt++;
		}
	if (f_vv) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << " after connections to pairs "
				"from one different row-partition, the system is" << endl;
		T.D1->print();
		}

	eqn_offset += (R - 1);
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r)
			continue;
		for (r3 = r2 + 1; r3 < R; r3++) {
			if (r3 == r)
				continue;
			// connections to pairs from two different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] =
					the_col_scheme[r2 * l2 + J] * the_col_scheme[r3 * l2 + J];
				}
			T.D1->RHS[eqn_offset + eqn_cnt] =
				row_classes_len[ROW][r2] * row_classes_len[ROW][r3] * lambda3;
			eqn_cnt++;
			}
		}
	eqn_offset += eqn_cnt;
	if (f_vv) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << " after connections to pairs from two "
				"different row-partitions, the system is" << endl;
		T.D1->print();
		}
	
	S = 0;
	for (i = 0; i < l1; i++) {
		s = the_row_scheme[r * l1 + i];
		if (f_vvv) {
			cout << "r=" << r << " i=" << i << " s=" << s << endl;
			}
		T.D1->RHS[eqn_offset + i] = s;
		S += s;
		f = P.startCell[i];
		l = P.cellSize[i];
		if (f_vvv) {
			cout << "f=" << f << " l=" << l << endl;
			}
			
		for (j = 0; j < l; j++) {
			T.D1->A[(eqn_offset + i) * nb_vars + f + j] = 1;
			T.D1->x_max[f + j] = s;
			}
		}
	if (f_vv) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << " after adding extra equations, "
				"the system is" << endl;
		T.D1->print();
		}
	
	T.D1->f_has_sum = TRUE;
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
		
	if (f_vv) {
		cout << "tdo_scheme::td3_rows_setup_first_system "
				"r=" << r << " the system is" << endl;
		T.D1->print();
		}
	return TRUE;
}		

int tdo_scheme::td3_rows_setup_second_system(int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, 
	int nb_vars, int &Nb_vars, int &Nb_eqns, 
	int *&point_types, int &nb_point_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int l2, i, r, I, J, nb_eqns_counting;
	int S;
	
	if (f_v) {
		cout << "tdo_scheme::td3_rows_setup_second_system" << endl;
	}
	l2 = nb_col_classes[COL];
	
	nb_eqns_counting = T.nb_multiple_types * (l2 + 1);
	Nb_eqns = nb_eqns_counting;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
		}


	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "td3_rows_setup_second_system: "
			"opening second system with "
			<< Nb_eqns << " equations and "
			<< Nb_vars << " variables" << endl;
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;
	for (I = 0; I < Nb_eqns; I++) 
		T.D2->RHS[I] = 9999;


	if (!td3_rows_counting_flags(verbose_level, 
		lambda3, block_size, lambda2, S, 
		T, 
		nb_vars, Nb_vars, 
		point_types, nb_point_types, 0)) {
		return FALSE;
		}
	

	T.D2->f_has_sum = TRUE;
	T.D2->sum = S;
	

	if (f_vv) {	
		cout << "The second system is" << endl;
	
		T.D2->print();
		}
	
	if (f_v) {
		cout << "tdo_scheme::td3_rows_setup_second_system "
				"done" << endl;
	}
	return TRUE;

}

int tdo_scheme::td3_rows_counting_flags(int verbose_level,
	int lambda3, int block_size, int lambda2, int &S,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&point_types, int &nb_point_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = nb_col_classes[COL];
	
	if (f_v) {
		cout << "td3_rows_counting_flags: eqn_offset=" << eqn_offset 
			<< " nb_multiple_types=" << T.nb_multiple_types << endl;
		}
	// counting flags, a block diagonal system with 
	// nb_multiple_types * (l2 + 1) equations
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (I = 0; I < l2; I++) {
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = point_types[c * nb_vars + I];
				T.D2->A[(eqn_offset + i * (l2 + 1) + I) * Nb_vars + J] = a;
				}
			a = the_col_scheme[r * nb_col_classes[COL] + I];
			b = col_classes_len[COL][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = point_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;
				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_rows_counting_flags: RHS["
							"nb_eqns_joining + i * (l2 + 1) + I] "
							"is negative, no solution for the "
							"distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			} // next I
		} // next i


	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		// one extra equation for the sum
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			// counting: extra row of ones
			T.D2->A[(eqn_offset + i * (l2 + 1) + l2) * Nb_vars + J] = 1;
			}
		
		s = row_classes_len[COL][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
		}
	if (f_vvv) {
		cout << "td3_rows_counting_flags, the system is" << endl;
		T.D2->print();
		}

	
	return TRUE;
}

// #############################################################################
// TDO parameter refinement for 3-designs - column refinement
// #############################################################################



int tdo_scheme::td3_refine_columns(int verbose_level, int f_once,
	int lambda3, int block_size, int f_scale, int scaling,
	int *&line_types, int &nb_line_types, int &line_type_len,  
	int *&distributions, int &nb_distributions)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int R, /*l1,*/ l2, r, nb_eqns, nb_vars = 0;
	int line_types_allocated;
	partitionstack P;
	int lambda2;
	int nb_points;
	int nb_sol;
	tdo_data T;

	if (f_v) {
		cout << "td3_refine_columns" << endl;
		}
	nb_points = m;
	lambda2 = lambda3 * (nb_points - 2) / (block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points
			<< " lambda2 = " << lambda2 << endl;
		}
	if ((block_size - 2) * lambda2 != lambda3 * (nb_points - 2)) {
		cout << "parameter are wrong" << endl;
		exit(1);
		}

	get_row_split_partition(verbose_level, P);
	
	R = nb_col_classes[COL];
	//l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
	
	T.allocate(R);
	
	T.types_first[0] = 0;
	
	line_types_allocated = 100;
	nb_line_types = 0;
	line_types = NEW_int(line_types_allocated * l2);
	line_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_vvv) {
			cout << "r=" << r << endl;
			}
		if (!td3_columns_setup_first_system(verbose_level - 1, 
			lambda3, block_size, lambda2, 
			T, r, P, 
			nb_vars, nb_eqns, 
			line_types, nb_line_types)) {
			FREE_int(line_types);
			return FALSE;
			}
		
		nb_sol = T.solve_first_system(verbose_level - 1, 
			line_types, nb_line_types, line_types_allocated);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol
				<< " refine line types" << endl;
			}
		if (nb_sol == 0) {
			FREE_int(line_types);
			return FALSE;
			}
		
		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;
		
		if (nb_sol == 1) {
			if (f_vv) {
				cout << "only one solution in block r=" << r << endl;
				}
			T.only_one_type[T.nb_only_one_type++] = r;
			}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
			}
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;
		
		} // next r
	
	
	// now we compute the distributions:
	//
	int Nb_vars, Nb_eqns;

	if (!td3_columns_setup_second_system(verbose_level, 
		lambda3, block_size, lambda2, f_scale, scaling, 
		T, 
		nb_vars, Nb_vars, Nb_eqns,
		line_types, nb_line_types)) {
		FREE_int(line_types);
		return FALSE;
		}

	T.solve_second_system(verbose_level - 1,
		FALSE /* f_use_mckay */, f_once,
		col_classes_len[COL], f_scale, scaling, 
		line_types, nb_line_types,
		distributions, nb_distributions);


	if (f_v) {
		cout << "td3_refine_columns: found "
			<< nb_distributions << " distributions." << endl;
		}
	return TRUE;
}

int tdo_scheme::td3_columns_setup_first_system(int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, int r, partitionstack &P,
	int &nb_vars,int &nb_eqns,
	int *&line_types, int &nb_line_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int j, R, l1, l2, S, I, J, f, l, a, a2;
	int s, d, d2, d3, o, h, rr, p, u, a3, e;
	
	if (f_v) {
		cout << "td3_columns_setup_first_system r=" << r << endl;
		}
		

	// create all partitions which are refined line types

	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
		
	nb_vars = l2; // P.n
	nb_eqns = l1; // = P.ht
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
			
	for (I = 0; I < nb_eqns; I++) {
		f = P.startCell[I];
		l = P.cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->A[I * nb_vars + J] = 1;
			a = the_row_scheme[J * R + r];
			if (a == 0)
				T.D1->x_max[J] = 0;
			else
				T.D1->x_max[J] = row_classes_len[ROW][J];
			}
		s = the_col_scheme[I * R + r];
		T.D1->RHS[I] = s;
		S += s;
		}

	// try to reduce the upper bounds:
		
	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0)
			continue;
		d = row_classes_len[ROW][j];
		d2 = binomial2(d) * lambda2;
		o = d2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];

			u = T.types_first[rr];
			//cout << "u=" << u << endl;
				
			a = line_types[u * nb_vars + j];
			a2 = binomial2(a);
			o -= a2 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "only one type, but no solution because "
						"of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			}
		e = largest_binomial2_below(o);
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
		}
	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0)
			continue;
		d = row_classes_len[ROW][j];
		d3 = binomial3(d) * lambda3;
		o = d3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
				u = T.types_first[rr];
			//cout << "u=" << u << endl;
			
			a = line_types[u * nb_vars + j];
			a3 = binomial3(a);
			o -= a3 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "only one type, but no solution because "
						"of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			}
		e = largest_binomial3_below(o);
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
		}
		
	T.D1->f_has_sum = TRUE;
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
	if (f_vv) {
		cout << "r=" << r << " the system is" << endl;
		T.D1->print();
		}
	return TRUE;
}		


int tdo_scheme::td3_columns_setup_second_system(
	int verbose_level,
	int lambda3, int block_size, int lambda2,
	int f_scale, int scaling,
	tdo_data &T, 
	int nb_vars, int &Nb_vars, int &Nb_eqns, 
	int *&line_types, int &nb_line_types)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int l2, i, r, I, J, a;
	int S;
	int nb_eqns_joining, nb_eqns_joining_pairs;
	int nb_eqns_joining_triples, nb_eqns_counting;

	l2 = nb_row_classes[ROW];
	
	nb_eqns_joining_triples = l2 + l2 * (l2 - 1) + binomial3(l2);
		// l2 times: triples within a given class
		// l2 * (l2 - 1) times (ordered pairs from an l2 set): 
		//     triples with 2 in a given class, 1 in another given class
		// binomial3(l2) triples from different classes
	nb_eqns_joining_pairs = l2 + binomial2(l2);
		// l2 times: pairs within a given class
		// binomial2(l2) pairs from different classes
	nb_eqns_joining = nb_eqns_joining_triples + nb_eqns_joining_pairs;
	nb_eqns_counting = T.nb_multiple_types * (l2 + 1);
	Nb_eqns = nb_eqns_joining + nb_eqns_counting;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
		}

	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "td3_columns_setup_second_system: "
			"opening second system with "
			<< Nb_eqns << " equations and "
			<< Nb_vars << " variables" << endl;
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;
	for (I = 0; I < Nb_eqns; I++) 
		T.D2->RHS[I] = 9999;


	if (!td3_columns_triples_same_class(verbose_level, 
		lambda3, block_size, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, 0)) {
		return FALSE;
		}
	
	if (!td3_columns_pairs_same_class(verbose_level, 
		lambda3, block_size, lambda2, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining_triples)) {
		return FALSE;
		}
	
	if (!td3_columns_counting_flags(verbose_level, 
		lambda3, block_size, lambda2, S, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda2_joining_pairs_from_different_classes(
		verbose_level,
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining_triples + l2)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda3_joining_triples_2_1(verbose_level, 
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, l2)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda3_joining_triples_1_1_1(verbose_level, 
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, l2 + l2 * (l2 - 1))) {
		return FALSE;
		}
	
	if (f_scale) {
		if (S % scaling) {
			cout << "cannot scale by " << scaling
				<< " b/c S=" << S << endl;
			exit(1);
			}
		S /= scaling;
		for (I = 0; I < Nb_eqns; I++) {
			a = T.D2->RHS[I];
			if (a % scaling) {
				if (a % scaling) {
					cout << "cannot scale by " << scaling
						<< " b/c RHS[" << I << "]=" << a << endl;
					}
				exit(1);
				}
			a /= scaling;
			T.D2->RHS[I] = a;
			}
#if 0
		for (I = 0; I < Nb_eqns; I++) 
			for (J = 0; J < Nb_vars; J++) 
				T.D2->A[I * Nb_vars + J] *= scaling;
#endif
		}

	

	T.D2->f_has_sum = TRUE;
	T.D2->sum = S;
	

	if (f_vv) {	
		cout << "The second system is" << endl;
	
		T.D2->print();
		}
	
	return TRUE;

}


int tdo_scheme::td3_columns_triples_same_class(int verbose_level,
	int lambda3, int block_size,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, a3, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_triples_same_class: "
				"eqn_offset=" << eqn_offset << endl;
		}
	// triples from the same class:
	for (I = 0; I < l2; I++) {
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first[r];
			l = T.types_len[r];
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				a3 = binomial3(a);
				// joining triples from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a3;
				}
			}
		a = row_classes_len[ROW][I];
		a3 = binomial3(a);
		T.D2->RHS[eqn_offset + I] = a3 * lambda3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a3 = binomial3(a);
			T.D2->RHS[eqn_offset + I] -= a3 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[I] is negative, "
						"no solution for the distribution" << endl;
					}
				return FALSE;
				}
			}
		}
	if (f_vvv) {
		cout << "triples from the same class, the system is" << endl;
		T.D2->print();
		}
	return TRUE;
}

int tdo_scheme::td3_columns_pairs_same_class(int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, a2, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_pairs_same_class: "
			"eqn_offset=" << eqn_offset << endl;
		}
	// pairs from the same class:
	for (I = 0; I < l2; I++) {
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first[r];
			l = T.types_len[r];
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				a2 = binomial2(a);
				// joining pairs from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a2;
				}
			}
		a = row_classes_len[ROW][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2 * lambda2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[eqn_offset + I] "
						"is negative, no solution for the "
						"distribution" << endl;
					}
				return FALSE;
				}
			}
		}
	if (f_vvv) {
		cout << "pairs from the same class, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

int tdo_scheme::td3_columns_counting_flags(int verbose_level,
	int lambda3, int block_size, int lambda2, int &S,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_counting_flags: "
				"eqn_offset=" << eqn_offset << endl;
		}
	// counting flags, a block diagonal system with 
	// nb_multiple_types * (l2 + 1) equations
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (I = 0; I < l2; I++) {
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				T.D2->A[(eqn_offset + i * (l2 + 1) + I) * Nb_vars + J] = a;
				}
			a = the_row_scheme[I * nb_col_classes[ROW] + r];
			b = row_classes_len[ROW][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;
				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_columns_counting_flags: "
							"RHS[nb_eqns_joining + i * (l2 + 1) + I] "
							"is negative, no solution for the "
							"distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			} // next I
		} // next i


	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		// one extra equation for the sum
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			// counting: extra row of ones
			T.D2->A[(eqn_offset + i * (l2 + 1) + l2) * Nb_vars + J] = 1;
			}
		
		s = col_classes_len[ROW][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
		}
	if (f_vvv) {
		cout << "td3_columns_counting_flags, the system is" << endl;
		T.D2->print();
		}

	
	return TRUE;
}

int tdo_scheme::td3_columns_lambda2_joining_pairs_from_different_classes(
	int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I1, I2, i, r, f, l, j, c, J, a, b, ab, k, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda2_joining_pairs_from_different_"
			"classes: eqn_offset=" << eqn_offset << endl;
		}
	// lambda2: joining pairs from different classes
	for (I1 = 0; I1 < l2; I1++) {
		for (I2 = I1 + 1; I2 < l2; I2++) {
			k = ij2k(I1, I2, l2);
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * nb_vars + I1];
					b = line_types[c * nb_vars + I2];
					ab = a * b;
					// joining pairs from different classes:
					T.D2->A[(eqn_offset + k) * Nb_vars + J] = ab;
					}
				}
			a = row_classes_len[ROW][I1];
			b = row_classes_len[ROW][I2];
			T.D2->RHS[eqn_offset + k] = a * b * lambda2;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[eqn_offset + k] -= a * b * p;
				if (T.D2->RHS[eqn_offset + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda2_joining_pairs_"
							"from_different_classes: RHS[eqn_offset + k] "
							"is negative, no solution for the "
							"distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda2_joining_pairs_from_different_"
				"classes, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

int tdo_scheme::td3_columns_lambda3_joining_triples_2_1(
	int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I1, I2, i, r, f, l, j, c, J, a, a2, ab, b, k, rr, p, u, l2, h;
	int length_first, length_first2, length_second;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda3_joining_triples_2_1: "
			"eqn_offset=" << eqn_offset << endl;
		}
	// lambda3: joining triples with two in the first
	// class and one in the second class
	for (I1 = 0; I1 < l2; I1++) {
		length_first = row_classes_len[ROW][I1];
		length_first2 = binomial2(length_first);
		for (I2 = 0; I2 < l2; I2++) {
			if (I2 == I1)
				continue;
			length_second = row_classes_len[ROW][I2];
			k = ordered_pair_rank(I1, I2, l2);
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * nb_vars + I1];
					b = line_types[c * nb_vars + I2];
					ab = binomial2(a) * b;
					T.D2->A[(l2 + k) * Nb_vars + J] = ab;
					}
				}
			T.D2->RHS[l2 + k] = length_first2 * length_second * lambda3;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				a2 = binomial2(a);
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[l2 + k] -= a2 * b * p;
				if (T.D2->RHS[l2 + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda3_joining_triples_2_1: "
							"RHS[l2 + k] is negative, no solution for "
							"the distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_2_1, "
				"the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

int tdo_scheme::td3_columns_lambda3_joining_triples_1_1_1(
	int verbose_level,
	int lambda3, int block_size, int lambda2,
	tdo_data &T, 
	int nb_vars, int Nb_vars, 
	int *&line_types, int &nb_line_types, int eqn_offset)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I1, I2, I3, i, r, f, l, j, c, J, a, b, k, rr, p, u, l2, h, g;
	int length_first, length_second, length_third;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda3_joining_triples_1_1_1: "
			"eqn_offset=" << eqn_offset << endl;
		}
	// lambda3: joining triples with all in different classes
	for (I1 = 0; I1 < l2; I1++) {
		length_first = row_classes_len[ROW][I1];
		for (I2 = I1 + 1; I2 < l2; I2++) {
			length_second = row_classes_len[ROW][I2];
			for (I3 = I2 + 1; I3 < l2; I3++) {
				length_third = row_classes_len[ROW][I3];
				k = ijk_rank(I1, I2, I3, l2);
				for (i = 0; i < T.nb_multiple_types; i++) {
					r = T.multiple_types[i];
					f = T.types_first[r];
					l = T.types_len[r];
					for (j = 0; j < l; j++) {
						c = f + j;
						J = T.types_first2[i] + j;
						a = line_types[c * nb_vars + I1];
						b = line_types[c * nb_vars + I2];
						g = line_types[c * nb_vars + I3];
						T.D2->A[(l2 + l2 * (l2 - 1) + k) *
								Nb_vars + J] = a * b * g;
						}
					}
				T.D2->RHS[l2 + l2 * (l2 - 1) + k] = length_first *
						length_second * length_third * lambda3;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = col_classes_len[COL][rr];
					u = T.types_first[rr];
					a = line_types[u * nb_vars + I1];
					b = line_types[u * nb_vars + I2];
					g = line_types[u * nb_vars + I3];
					T.D2->RHS[l2 + l2 * (l2 - 1) + k] -= a * b * g * p;
					if (T.D2->RHS[l2 + l2 * (l2 - 1) + k] < 0) {
						if (f_v) {
							cout << "td3_columns_lambda3_joining_triples_"
								"1_1_1: RHS[l2 + l2 * (l2 - 1) + k] is "
								"negative, no solution for the "
								"distribution" << endl;
							}
						return FALSE;
						}
					} // next h
				}
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_1_1_1, "
				"the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

}
