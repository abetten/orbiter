// tdo_scheme.C
//
// Anton Betten 8/27/07

#include "galois.h"
#include "incidence.h"

tdo_scheme::tdo_scheme()
{
	INT i;
	
	part = NULL;
	entries = NULL;
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
	P = NULL;
}

tdo_scheme::~tdo_scheme()
{
	INT i;
	
	if (part) {
		FREE_INT(part);
		part = NULL;
		}
	if (entries) {
		FREE_INT(entries);
		entries = NULL;
		}
	for (i = 0; i < NUMBER_OF_SCHEMES; i++) {
		free_partition(i);
		}
	if (the_row_scheme) {
		FREE_INT(the_row_scheme);
		the_row_scheme = NULL;
		}
	if (the_col_scheme) {
		FREE_INT(the_col_scheme);
		the_col_scheme = NULL;
		}
	if (the_extra_row_scheme) {
		FREE_INT(the_extra_row_scheme);
		the_extra_row_scheme = NULL;
		}
	if (the_extra_col_scheme) {
		FREE_INT(the_extra_col_scheme);
		the_extra_col_scheme = NULL;
		}
	if (the_row_scheme_cur) {
		FREE_INT(the_row_scheme_cur);
		the_row_scheme_cur = NULL;
		}
	if (the_col_scheme_cur) {
		FREE_INT(the_col_scheme_cur);
		the_col_scheme_cur = NULL;
		}
	if (the_extra_row_scheme_cur) {
		FREE_INT(the_extra_row_scheme_cur);
		the_extra_row_scheme_cur = NULL;
		}
	if (the_extra_col_scheme_cur) {
		FREE_INT(the_extra_col_scheme_cur);
		the_extra_col_scheme_cur = NULL;
		}
	if (P) {
		delete P;
		P = NULL;
		}
}

void tdo_scheme::init_part_and_entries(INT *Part, INT *Entries, INT verbose_level)
{
	INT i;
	INT f_v = (verbose_level >= 1);
	
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
		FREE_INT(part);
		}
	if (entries) {
		FREE_INT(entries);
		}
	part = NEW_INT(part_length + 1);
	for (i = 0; i <= part_length; i++) {
		part[i] = Part[i];
		}
	entries = NEW_INT(4 * nb_entries + 1);
	for (i = 0; i <= 4 * nb_entries; i++) {
		entries[i] = Entries[i];
		}
}

void tdo_scheme::init_part_and_entries_INT(INT *Part, INT *Entries, INT verbose_level)
{
	INT i;
	INT f_v = (verbose_level >= 1);
	
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
		FREE_INT(part);
		}
	if (entries) {
		FREE_INT(entries);
		}
	part = NEW_INT(part_length + 1);
	for (i = 0; i <= part_length; i++) {
		part[i] = Part[i];
		}
	entries = NEW_INT(4 * nb_entries + 1);
	for (i = 0; i <= 4 * nb_entries; i++) {
		entries[i] = Entries[i];
		}
}

void tdo_scheme::init_TDO(INT *Part, INT *Entries,
	INT Row_level, INT Col_level, INT Extra_row_level, INT Extra_col_level,
	INT Lambda_level, INT verbose_level)
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
		FREE_INT(the_row_scheme_cur);
		the_row_scheme_cur = NULL;
		}
	if (the_col_scheme_cur) {
		FREE_INT(the_col_scheme_cur);
		the_col_scheme_cur = NULL;
		}
	if (the_extra_row_scheme_cur) {
		FREE_INT(the_extra_row_scheme_cur);
		the_extra_row_scheme_cur = NULL;
		}
	if (the_extra_col_scheme_cur) {
		FREE_INT(the_extra_col_scheme_cur);
		the_extra_col_scheme_cur = NULL;
		}
}

void tdo_scheme::init_partition_stack(INT verbose_level)
{
	INT k, at, f, c, l, i;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
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
		INT_vec_print(cout, part, part_length + 1);
		cout << endl;
		}
	
	P = new partitionstack;
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
				FREE_INT(the_row_scheme);
				the_row_scheme = NULL;
				}
			the_row_scheme = NEW_INT(l * l);
			for (i = 0; i < l * l; i++) {
				the_row_scheme[i] = -1;
				}
			get_partition(ROW, l, verbose_level - 3);
			get_row_or_col_scheme(ROW, l, verbose_level - 3);
			}
			
		if (P->ht == col_level) {
			l = P->ht;
			if (the_col_scheme) {
				FREE_INT(the_col_scheme);
				the_col_scheme = NULL;
				}
			the_col_scheme = NEW_INT(l * l);
			for (i = 0; i < l * l; i++) {
				the_col_scheme[i] = -1;
				}
			get_partition(COL, l, verbose_level - 3);	
			get_row_or_col_scheme(COL, l, verbose_level - 3);
			}
			
		if (P->ht == extra_row_level) {
			l = P->ht;
			if (the_extra_row_scheme) {
				FREE_INT(the_extra_row_scheme);
				the_extra_row_scheme = NULL;
				}
			the_extra_row_scheme = NEW_INT(l * l);
			for (i = 0; i < l * l; i++) {
				the_extra_row_scheme[i] = -1;
				}
			get_partition(EXTRA_ROW, l, verbose_level - 3);
			get_row_or_col_scheme(EXTRA_ROW, l, verbose_level - 3);	
			}
			
		if (P->ht == extra_col_level) {
			l = P->ht;
			if (the_extra_col_scheme) {
				FREE_INT(the_extra_col_scheme);
				the_extra_col_scheme = NULL;
				}
			the_extra_col_scheme = NEW_INT(l * l);
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
		FREE_INT(the_row_scheme);
		the_row_scheme = NULL;
		}
	if (the_col_scheme) {
		FREE_INT(the_col_scheme);
		the_col_scheme = NULL;
		}
	if (the_extra_row_scheme) {
		FREE_INT(the_extra_row_scheme);
		the_extra_row_scheme = NULL;
		}
	if (the_extra_col_scheme) {
		FREE_INT(the_extra_col_scheme);
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

void tdo_scheme::get_partition(INT h, INT l, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 10);
	INT i;
	
	if (f_v) {
		cout << "tdo_scheme::get_partition h=" << h << " l=" << l << " m=" << m << " n=" << n << endl;
		}
	if (l < 0) {
		cout << "tdo_scheme::get_partition l is negative" << endl;
		exit(1);
		}
	free_partition(h);
	row_classes[h] = NEW_INT(l);
	col_classes[h] = NEW_INT(l);
	row_class_index[h] = NEW_INT(l);
	col_class_index[h] = NEW_INT(l);
	row_classes_first[h] = NEW_INT(l);
	row_classes_len[h] = NEW_INT(l);
	col_classes_first[h] = NEW_INT(l);
	col_classes_len[h] = NEW_INT(l);
	row_class_no[h] = NEW_INT(m);
	col_class_no[h] = NEW_INT(n);

	for (i = 0; i < l; i++) {
		row_class_index[h][i] = -1;
		col_class_index[h][i] = -1;
		}
			
	P->get_row_and_col_classes(row_classes[h], nb_row_classes[h],
		col_classes[h], nb_col_classes[h], verbose_level - 1);
				
	for (i = 0; i < nb_row_classes[h]; i++) {
		row_class_index[h][row_classes[h][i]] = i;
		if (f_vv) {
			cout << "row_class_index[h][" << row_classes[h][i] << "] = " << row_class_index[h][row_classes[h][i]] << endl;
			}
		}
	for (i = 0; i < nb_col_classes[h]; i++) {
		col_class_index[h][col_classes[h][i]] = i;
		if (f_vv) {
			cout << "col_class_index[h][" << col_classes[h][i] << "] = " << col_class_index[h][col_classes[h][i]] << endl;
			}
		}
	if (f_vv) {
		cout << "nb_row_classes[h]=" << nb_row_classes[h] << endl;
		cout << "nb_col_classes[h]=" << nb_col_classes[h] << endl;
		}
}

void tdo_scheme::free_partition(INT i)
{
		if (row_classes[i]) {
			FREE_INT(row_classes[i]);
			row_classes[i] = NULL;
			}
		if (col_classes[i]) {
			FREE_INT(col_classes[i]);
			col_classes[i] = NULL;
			}
		if (row_class_index[i]) {
			FREE_INT(row_class_index[i]);
			row_class_index[i] = NULL;
			}
		if (col_class_index[i]) {
			FREE_INT(col_class_index[i]);
			col_class_index[i] = NULL;
			}
		if (row_classes_first[i]) {
			FREE_INT(row_classes_first[i]);
			row_classes_first[i] = NULL;
			}
		if (row_classes_len[i]) {
			FREE_INT(row_classes_len[i]);
			row_classes_len[i] = NULL;
			}
		if (row_class_no[i]) {
			FREE_INT(row_class_no[i]);
			row_class_no[i] = NULL;
			}
		if (col_classes_first[i]) {
			FREE_INT(col_classes_first[i]);
			col_classes_first[i] = NULL;
			}
		if (col_classes_len[i]) {
			FREE_INT(col_classes_len[i]);
			col_classes_len[i] = NULL;
			}
		if (col_class_no[i]) {
			FREE_INT(col_class_no[i]);
			col_class_no[i] = NULL;
			}
}

void tdo_scheme::complete_partition_info(INT h, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 5);
	INT f, i, j, c1, S, k;
	
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
				cout << "row_class_no[h][" << f + k << "]=" << row_class_no[h][f + k] << endl;
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
				cout << "col_class_no[h][" << f + k << "]=" << col_class_no[h][f + k] << endl;
				}
			}
		f += S;
		}
}

void tdo_scheme::get_row_or_col_scheme(INT h, INT l, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, d, c1, c2, s1, s2, v;
	
	if (f_v) {
		cout << "tdo_scheme::get_row_or_col_scheme" << endl;
		}
	for (i = 0; i < nb_entries; i++) {
		d = entries[i * 4 + 0];
		c1 = entries[i * 4 + 1];
		c2 = entries[i * 4 + 2];
		v = entries[i * 4 + 3];
		if (d == l) {
			//cout << "entry " << i << " : " << d << " " << c1 << " " << c2 << " " << v << endl;
			if (h == ROW && row_class_index[h][c1] >= 0) { // row scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_row_scheme[" << s1 << " * " << nb_col_classes[h] << " + " << s2 << "] = " << v << endl;
				the_row_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == COL && col_class_index[h][c1] >= 0) { // col scheme
				s1 = row_class_index[h][c2];
				s2 = col_class_index[h][c1];
				//cout << "the_col_scheme[" << s1 << " * " << nb_col_classes[h] << " + " << s2 << "] = " << v << endl;
				the_col_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == EXTRA_ROW && row_class_index[h][c1] >= 0) { // col scheme
				s1 = row_class_index[h][c1];
				s2 = col_class_index[h][c2];
				//cout << "the_extra_row_scheme[" << s1 << " * " << nb_col_classes[h] << " + " << s2 << "] = " << v << endl;
				the_extra_row_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			else if (h == EXTRA_COL && col_class_index[h][c1] >= 0) { // col scheme
				s1 = row_class_index[h][c2];
				s2 = col_class_index[h][c1];
				//cout << "EXTRA_COL:" << endl;
				//cout << "c1=" << c1 << endl;
				//cout << "c2=" << c2 << endl;
				//cout << "s1=" << s1 << endl;
				//cout << "s2=" << s2 << endl;
				//cout << "the_extra_col_scheme[" << s1 << " * " << nb_col_classes[h] << " + " << s2 << "] = " << v << endl;
				the_extra_col_scheme[s1 * nb_col_classes[h] + s2] = v;
				}
			//print_row_test_data();
			} // if
		} // next i
	if (h == ROW) {
		if (the_row_scheme_cur) {
			FREE_INT(the_row_scheme_cur);
			the_row_scheme_cur = NULL;
			}
		the_row_scheme_cur = NEW_INT(m * nb_col_classes[h]);
		for (i = 0; i < m; i++) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				the_row_scheme_cur[i * nb_col_classes[h] + j] = 0;
				}
			}
		//print_row_test_data();
		}
	if (h == COL) {
		if (the_col_scheme_cur) {
			FREE_INT(the_col_scheme_cur);
			the_col_scheme_cur = NULL;
			}
		the_col_scheme_cur = NEW_INT(n * nb_row_classes[h]);
		for (i = 0; i < n; i++) {
			for (j = 0; j < nb_row_classes[h]; j++) {
				the_col_scheme_cur[i * nb_row_classes[h] + j] = 0;
				}
			}
		}
	if (h == EXTRA_ROW) {
		if (the_extra_row_scheme_cur) {
			FREE_INT(the_extra_row_scheme_cur);
			the_extra_row_scheme_cur = NULL;
			}
		the_extra_row_scheme_cur = NEW_INT(m * nb_col_classes[h]);
		for (i = 0; i < m; i++) {
			for (j = 0; j < nb_col_classes[h]; j++) {
				the_extra_row_scheme_cur[i * nb_col_classes[h] + j] = 0;
				}
			}
		}
	if (h == EXTRA_COL) {
		if (the_extra_col_scheme_cur) {
			FREE_INT(the_extra_col_scheme_cur);
			the_extra_col_scheme_cur = NULL;
			}
		the_extra_col_scheme_cur = NEW_INT(n * nb_row_classes[h]);
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

void tdo_scheme::get_column_split_partition(INT verbose_level, partitionstack &P)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;
	
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
			cout << "i=" << i << " ci=" << ci << " j1=" << j1 << " cc=" << cc << endl;
			}
		P.subset_size = 0;
		for (h = 0; h < l; h++) {
			j = P.pointList[f + h];
			cj = col_classes[COL][j];
			if (FALSE) {
				cout << "j=" << j << " cj=" << cj << endl;
				}
			if (!tdo_scheme::P->is_descendant_of_at_level(cj, ci, level[ROW], FALSE)) {
				if (FALSE) {
					cout << j << "/" << cj << " is not a descendant of " << i << "/" << ci << endl;
					}
				P.subset[P.subset_size++] = j;
				}
			}
		if (FALSE) {
			cout << "non descendants of " << i << "/" << ci << " : ";
			INT_set_print(cout, P.subset, P.subset_size);
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

void tdo_scheme::get_row_split_partition(INT verbose_level, partitionstack &P)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT i, j, h, j1, cc, f, l, ci, cj, l1, l2, R;
	
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
			cout << "i=" << i << " ci=" << ci << " j1=" << j1 << " cc=" << cc << endl;
			}
		P.subset_size = 0;
		for (h = 0; h < l; h++) {
			j = P.pointList[f + h];
			cj = row_classes[ROW][j];
			if (FALSE) {
				cout << "j=" << j << " cj=" << cj << endl;
				}
			if (!tdo_scheme::P->is_descendant_of_at_level(cj, ci, level[COL], FALSE)) {
				if (FALSE) {
					cout << j << "/" << cj << " is not a descendant of " << i << "/" << ci << endl;
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
			INT_set_print(cout, P.subset, P.subset_size);
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

void tdo_scheme::print_scheme(INT h, INT f_v)
{
	INT i, j, c1, c2, a = 0;
	
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
	cout << "is " << nb_row_classes[h] << " x " << nb_col_classes[h] << endl;
	cout << "          | ";
	for (j = 0; j < nb_col_classes[h]; j++) {
		c2 = col_classes[h][j];
		cout << setw(3) << col_classes_len[h][j] << "_{" << setw(3) << c2 << "}";
		}
	cout << endl;
	cout << "============";
	for (j = 0; j < nb_col_classes[h]; j++) {
		cout << "=========";
		}
	cout << endl;
	for (i = 0; i < nb_row_classes[h]; i++) {
		c1 = row_classes[h][i];
		cout << setw(3) << row_classes_len[h][i] << "_{" << setw(3) << c1 << "} | ";
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
			cout << i << " : " << row_classes_first[h][i] << " : " << row_classes_len[h][i] << endl;
			}
		cout << "class_no:" << endl;
		for (i = 0; i < m; i++) {
			cout << i << " : " << row_class_no[h][i] << endl;
			}
		cout << "col_classes first / len:" << endl;
		for (i = 0; i < nb_col_classes[h]; i++) {
			cout << i << " : " << col_classes_first[h][i] << " : " << col_classes_len[h][i] << endl;
			}
		cout << "col_class_no:" << endl;
		for (i = 0; i < n; i++) {
			cout << i << " : " << col_class_no[h][i] << endl;
			}
		}
}

void tdo_scheme::print_scheme_tex(ostream &ost, INT h)
{
	print_scheme_tex_fancy(ost, h, FALSE, NULL);
}

void tdo_scheme::print_scheme_tex_fancy(ostream &ost, INT h, INT f_label, BYTE *label)
{
	INT i, j, a = 0, n, m, c1, c2;
	
	n = nb_row_classes[h];
	m = nb_col_classes[h];
	ost << "$$" << endl;
	ost << "\\begin{array}{r|*{" << m << "}{r}}" << endl;
	if (f_label) {
		ost << "\\multicolumn{" << m + 1 << "}{c}{\\mbox{" << label << "}}\\\\" << endl;
		}
	if (h == ROW || h == EXTRA_ROW)
		ost << "\\rightarrow";
	else if (h == COL || h == EXTRA_COL)
		ost << "\\downarrow";
	else if (h == LAMBDA)
		ost << "\\lambda";
	for (j = 0; j < m; j++) {
		c2 = col_classes[h][j];
		ost << " & " << setw(3) << col_classes_len[h][j] << "_{" << setw(3) << c2 << "}";
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

void tdo_scheme::compute_whether_first_inc_must_be_moved(INT *f_first_inc_must_be_moved, INT verbose_level)
{
	INT i, j, ii, fi, fii, fj, row_cell0, row_cell, col_cell, a, b, c;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "tdo_scheme::compute_whether_first_inc_must_be_moved" << endl;
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
			cout << "considering whether incidence in block " << i << "," << j << " must be moved" << endl;
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
						cout << "i=" << i << " ii=" << ii << " different COL fuse, hence it must not be moved" << endl;
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
						cout << "ii=" << ii << " seeing non zero entry " << b << ", hence it must be moved" << endl;
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
						cout << "i=" << i << " ii=" << ii << " different EXTRACOL fuse, hence it must not be moved" << endl;
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
						cout << "ii=" << ii << " seeing non zero entry " << b << ", hence it must be moved" << endl;
						}
					f_first_inc_must_be_moved[i] = TRUE;
					break;
					}
				} // next ii
			}
		
		}
}

INT tdo_scheme::count_nb_inc_from_row_scheme(INT verbose_level)
{
	INT i, j, a, b = 0, nb_inc;
	INT f_v = (verbose_level > 1);
	
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

INT tdo_scheme::count_nb_inc_from_extra_row_scheme(INT verbose_level)
{
	INT i, j, a, b = 0, nb_inc;
	INT f_v = (verbose_level > 1);
	
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


