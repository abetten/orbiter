// spreadsheet.C
// 
// Anton Betten
// July 18, 2012
//
// moved to GALOIS: March 15, 2013
//

#include "galois.h"

INT strcmp_with_or_without(BYTE *p, BYTE *q);

spreadsheet::spreadsheet()
{
	null();
}

spreadsheet::~spreadsheet()
{
	freeself();
}

void spreadsheet::null()
{
	tokens = NULL;
	line_start = NULL;
	line_size = NULL;
	Table = NULL;
}

void spreadsheet::freeself()
{
	INT i;
	
	if (tokens) {
		for (i = 0; i < nb_tokens; i++) {
			FREE_BYTE(tokens[i]);
			}
		FREE_PBYTE(tokens);
		}
	if (line_start) {
		FREE_INT(line_start);
		}
	if (line_size) {
		FREE_INT(line_size);
		}
	if (Table) {
		FREE_INT(Table);
		}
	null();
}

void spreadsheet::init_set_of_sets(set_of_sets *S, INT f_make_heading)
{
	INT s, i, j, a, h, len, offset = 0;
	BYTE str[1000];

	s = S->largest_set_size();
	if (f_make_heading) {
		spreadsheet::nb_rows = S->nb_sets + 1;
		offset = 1;
		}
	else {
		spreadsheet::nb_rows = S->nb_sets;
		offset = 0;
		}
	spreadsheet::nb_cols = s + 1;
	Table = NEW_INT(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = -1;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_PBYTE(nb_tokens + 1);

	h = 0;
	if (f_make_heading) {
		for (j = 0; j < s + 1; j++) {
			sprintf(str, "C%ld", j);
			len = strlen(str);
			tokens[h] = NEW_BYTE(len + 1);
			strcpy(tokens[h], str);
			Table[0 * nb_cols + j] = h;
			h++;
			}
		}
	for (i = 0; i < S->nb_sets; i++) {

		sprintf(str, "%ld", S->Set_size[i]);
		len = strlen(str);
		tokens[h] = NEW_BYTE(len + 1);
		strcpy(tokens[h], str);
		Table[(i + offset) * nb_cols + 0] = h;
		h++;

		for (j = 0; j < S->Set_size[i]; j++) {
			a = S->Sets[i][j];
			
			sprintf(str, "%ld", a);
			len = strlen(str);
			tokens[h] = NEW_BYTE(len + 1);
			strcpy(tokens[h], str);
			Table[(i + offset) * nb_cols + 1 + j] = h;
			h++;

			}
		}
}

void spreadsheet::init_INT_matrix(INT nb_rows, INT nb_cols, INT *A)
{
	INT i, len, a;
	BYTE str[1000];
	
	spreadsheet::nb_rows = nb_rows;
	spreadsheet::nb_cols = nb_cols;
	Table = NEW_INT(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = -1;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_PBYTE(nb_tokens + 1);
	for (i = 0; i < nb_tokens; i++) {
		a = A[i];
		sprintf(str, "%ld", a);
		len = strlen(str);
		tokens[i] = NEW_BYTE(len + 1);
		strcpy(tokens[i], str);
		Table[i] = i;
		}
}

void spreadsheet::init_empty_table(INT nb_rows, INT nb_cols)
{
	INT i;
	
	spreadsheet::nb_rows = nb_rows;
	spreadsheet::nb_cols = nb_cols;
	Table = NEW_INT(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = i;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_PBYTE(nb_tokens);
	for (i = 0; i < nb_tokens; i++) {
		tokens[i] = NULL;
		}
}

void spreadsheet::fill_entry_with_text(INT row_idx, INT col_idx, const BYTE *text)
{
	INT l, t;
	
	t = Table[row_idx * nb_cols + col_idx];
	if (tokens[t]) {
		//cout << "fill_column_with_text before FREE_BYTE i=" << i << " col_idx=" << col_idx << " t=" << t << endl;
		FREE_BYTE(tokens[t]);
		}
	l = strlen(text);
	tokens[t] = NEW_BYTE(l + 1);
	strcpy(tokens[t], text);
}

void spreadsheet::fill_column_with_text(INT col_idx, const BYTE **text, const BYTE *heading)
{
	INT i, l, t;
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_text before FREE_BYTE i=" << i << " col_idx=" << col_idx << " t=" << t << endl;
			FREE_BYTE(tokens[t]);
			}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], heading);
			}
		else {
			l = strlen(text[i - 1]);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], text[i - 1]);
			}
		}
}

void spreadsheet::fill_column_with_INT(INT col_idx, INT *data, const BYTE *heading)
{
	INT i, l, t;
	BYTE str[1000];
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_INT before FREE_BYTE i=" << i << " col_idx=" << col_idx << " t=" << t << endl;
			FREE_BYTE(tokens[t]);
			}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], heading);
			}
		else {
			sprintf(str, "%ld", data[i - 1]);
			l = strlen(str);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], str);
			}
		}
}

void spreadsheet::fill_column_with_row_index(INT col_idx, const BYTE *heading)
{
	INT i, l, t;
	BYTE str[1000];
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_row_index before FREE_BYTE i=" << i << " col_idx=" << col_idx << " t=" << t << endl;
			FREE_BYTE(tokens[t]);
			}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], heading);
			}
		else {
			sprintf(str, "%ld", i - 1);
			l = strlen(str);
			tokens[t] = NEW_BYTE(l + 1);
			strcpy(tokens[t], str);
			}
		}
}

void spreadsheet::add_token(BYTE *label)
{
	BYTE **tokens2;
	INT i, j, len;

	tokens2 = NEW_PBYTE(nb_tokens + 1);
	for (i = 0; i < nb_tokens; i++) {
		tokens2[i] = tokens[i];
		}
	len = strlen(label);
	tokens2[nb_tokens] = NEW_BYTE(len + 1);
	for (i = 0, j = 0; i < len; i++) {
		if ((INT)label[i] < 0) {
			cout << "spreadsheet::add_token negative character " << (INT) label[i] << endl;
			}
		else {
			tokens2[nb_tokens][j++] = label[i];
			}
		}
	tokens2[nb_tokens][j++] = 0;
	//strcpy(tokens2[nb_tokens], label);
	FREE_PBYTE(tokens);
	tokens = tokens2;
	nb_tokens++;
}

void spreadsheet::save(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);


	{
	ofstream f(fname);
	print_table(f, FALSE);
	f << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
}

void spreadsheet::read_spreadsheet(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT i;

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet reading file " << fname << " of size " << file_size(fname) << endl;
		}
	

	
	tokenize(fname, tokens, nb_tokens, verbose_level /* - 2*/);

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet read file with " << nb_tokens << " tokens" << endl;

		if (f_vv) {
			for (i = 0; i < nb_tokens; i++) {
				cout << setw(6) << i << " : '" << tokens[i] << "'" << endl;
				}
			}
		}



	find_rows(verbose_level - 2);

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet Found " << nb_lines << " lines" << endl;
		}
	
	if (f_vv) {
		{
		INT f, l, j;
	
		for (i = 0; i < nb_lines; i++) {
			f = line_start[i];
			l = line_size[i];
			cout << "Line " << i << " : ";
			for (j = 0; j < l; j++) {
				cout << "'" << tokens[f + j] << "'";
				if (j < l - 1) {
					cout << ", ";
					}
				}
			cout << endl;
			}
		}
		}

	INT j;
	
	nb_rows = nb_lines;
	nb_cols = line_size[0];
	Table = NEW_INT(nb_rows * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = -1;
			}
		}
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < MINIMUM(nb_cols, line_size[i]); j++) {
			Table[i * nb_cols + j] = line_start[i] + j;
			}
		}

	if (f_vv) {
		cout << "spreadsheet::read_spreadsheet" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << "row " << i << " column " << j << " entry '" << tokens[Table[i * nb_cols + j]] << "'" << endl;
				}
			}
		}
	
	if (f_v) {
		cout << "spreadsheet::read_spreadsheet reading file " << fname << " of size " << file_size(fname) << " done" << endl;
		}
	
}

void spreadsheet::print_table(ostream &ost, INT f_enclose_in_parentheses)
{
	INT i;
	
	//cout << "Table:" << endl;
	for (i = 0; i < nb_rows; i++) {
		print_table_row(i, f_enclose_in_parentheses, ost);
		}
}

void spreadsheet::print_table_latex_all_columns(ostream &ost, INT f_enclose_in_parentheses)
{
	INT i, j;
	INT *f_column_select;

	f_column_select = NEW_INT(nb_cols);
	for (j = 0; j < nb_cols; j++) {
		f_column_select[j] = TRUE;
		}
	
	//cout << "Table:" << endl;
	ost << "\\begin{tabular}{|c|";
	for (j = 0; j < nb_cols; j++) {
		ost << "c|";
		}
	ost << "}" << endl;
	for (i = 0; i < nb_rows; i++) {
		print_table_row_latex(i, f_column_select, f_enclose_in_parentheses, ost);
		}
	ost << "\\end{tabular}" << endl;

	FREE_INT(f_column_select);
}

void spreadsheet::print_table_latex(ostream &ost, INT *f_column_select, INT f_enclose_in_parentheses)
{
	INT i, j;
	
	//cout << "Table:" << endl;
	ost << "\\begin{tabular}{|";
	for (j = 0; j < nb_cols; j++) {
		if (f_column_select[j]) {
			ost << "p{3cm}|";
			}
		}
	ost << "}" << endl;
	for (i = 0; i < nb_rows; i++) {
		print_table_row_latex(i, f_column_select, f_enclose_in_parentheses, ost);
		}
	ost << "\\end{tabular}" << endl;
}

void spreadsheet::print_table_row(INT row, INT f_enclose_in_parentheses, ostream &ost)
{
	INT j, t; //, h;
	INT f_enclose;
	
	//cout << "Row " << row << " : ";
	for (j = 0; j < nb_cols; j++) {
		t = Table[row * nb_cols + j];
		if (t >= 0) {
#if 0
			if (row == 0 && j == 0) {
				cout << "printing token '" << tokens[t] << "'" << endl;
				for (h = 0; h < 10; h++) {
					cout << h << " : " << (INT) tokens[t][h] << endl;
					}
				}
#endif
			if (tokens[t][0] == '\"') {
				f_enclose = FALSE;
				}
			else {
				f_enclose = TRUE;
				}
			if (f_enclose) {
				ost << "\"";
				}
			if (tokens[t] == NULL) {
				cout << "spreadsheet::print_table_row token[t] == NULL, t = " << t << endl;
				}
			else {
				ost << tokens[t];
				}
			if (f_enclose) {
				ost << "\"";
				}
			}
		if (j < nb_cols - 1) {
			ost << ",";
			}
		}
	ost << endl;
}

void spreadsheet::print_table_row_latex(INT row, INT *f_column_select, INT f_enclose_in_parentheses, ostream &ost)
{
	INT j, t, l; //, h;
	INT f_first = TRUE;
	
	//cout << "Row " << row << " : ";
	//ost << row;
	for (j = 0; j < nb_cols; j++) {
		if (f_column_select[j]) {
			if (f_first) {
				f_first = FALSE;
				}
			else {
				ost << " & ";
				}
			t = Table[row * nb_cols + j];
			if (t >= 0) {
	#if 0
				if (row == 0 && j == 0) {
					cout << "printing token '" << tokens[t] << "'" << endl;
					for (h = 0; h < 10; h++) {
						cout << h << " : " << (INT) tokens[t][h] << endl;
						}
					}
	#endif
				if (f_enclose_in_parentheses) {
					ost << "\"";
					}
				if (tokens[t][0] == '"') {
					tokens[t][0] = ' ';
					}
				l = strlen(tokens[t]);
				if (tokens[t][l - 1] == '"') {
					tokens[t][l - 1] = ' ';
					}
				ost << tokens[t];
				if (f_enclose_in_parentheses) {
					ost << "\"";
					}
				}
			}
		}
	ost << "\\\\" << endl;
}

void spreadsheet::print_table_row_detailed(INT row, ostream &ost)
{
	INT j, t;
	
	ost << "Row " << row << " of the table is:" << endl;
	for (j = 0; j < nb_cols; j++) {
		ost << "Column " << j << " / " << nb_cols << " : ";
		t = Table[row * nb_cols + j];
		if (t >= 0) {
			ost << tokens[t];
			}
		if (j < nb_cols - 1) {
			ost << ",";
			}
		ost << endl;
		}
}

void spreadsheet::print_table_with_row_selection(INT *f_selected, ostream &ost)
{
	INT i;
	
	//cout << "Table:" << endl;
	for (i = 0; i < nb_rows; i++) {
		if (!f_selected[i]) {
			continue;
			}
		print_table_row(i, FALSE, ost);
		}
}

void spreadsheet::print_table_sorted(ostream &ost, const BYTE *sort_by)
{
	INT i, t, ii;
	INT idx;
	INT *perm;
	BYTE **labels;
	
	idx = find_by_column(sort_by);
	perm = NEW_INT(nb_rows - 1);
	labels = NEW_PBYTE(nb_rows - 1);
	for (i = 0; i < nb_rows - 1; i++) {
		perm[i] = i;
		t = Table[(i + 1) * nb_cols + idx];
		if (t >= 0) {
			if (tokens[t][0] == '"') {
				labels[i] = NEW_BYTE(strlen(tokens[t]) + 1);
				strcpy(labels[i], tokens[t] + 1);
				}
			else {
				labels[i] = NEW_BYTE(strlen(tokens[t]) + 1);
				strcpy(labels[i], tokens[t]);
				}
			}
		else {
			labels[i] = NEW_BYTE(1);
			labels[i][0] = 0;
			}
		}
	
	quicksort_array_with_perm(nb_rows - 1, (void **) labels, perm, 
		compare_strings, NULL /*void *data*/);

	
	//cout << "Table:" << endl;
	for (i = 0; i < nb_rows; i++) {
		if (i == 0) {
			ii = 0;
			}
		else {
			ii = perm[i - 1] + 1;
			}
		print_table_row(ii, FALSE, ost);
		}
}


void spreadsheet::add_column_with_constant_value(BYTE *label, BYTE *value)
{
	INT i;

	reallocate_table();
	add_token(label);
	Table[0 * nb_cols + nb_cols - 1] = nb_tokens - 1;
	for (i = 1; i < nb_rows; i++) {
		add_token(value);
		Table[i * nb_cols + nb_cols - 1] = nb_tokens - 1;
		}
	
}

void spreadsheet::reallocate_table()
{
	INT i, j;
	INT *Table2;

	Table2 = NEW_INT(nb_rows * (nb_cols + 1));
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table2[i * (nb_cols + 1) + j] = Table[i * nb_cols + j];
			}
		Table2[i * (nb_cols + 1) + nb_cols] = -1;
		}
	FREE_INT(Table);
	Table = Table2;
	nb_cols++;
}

void spreadsheet::reallocate_table_add_row()
{
	INT i, j;
	INT *Table2;

	Table2 = NEW_INT((nb_rows + 1) * nb_cols);
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table2[i * nb_cols + j] = Table[i * nb_cols + j];
			}
		}
	for (j = 0; j < nb_cols; j++) {
		Table2[nb_rows * nb_cols + j] = -1;
		}
	FREE_INT(Table);
	Table = Table2;
	nb_rows++;
}

INT spreadsheet::find_by_column(const BYTE *join_by)
{
	INT j, t, c; //, h;
	
	for (j = 0; j < nb_cols; j++) {
		t = Table[0 * nb_cols + j];
		if (t >= 0) {
			c = strncmp(tokens[t], join_by, strlen(join_by));
#if 0
			cout << "comparing '" << tokens[t] << "' with '" << join_by << "' yields " << c << endl;
			for (h = 0; h < (INT)strlen(join_by); h++) {
				cout << h << " : " << tokens[t][h] << " : " << join_by[h] << endl;
				}
#endif
			if (c == 0) {
				return j;
				}
			}
		}
	// in case we don't find it, maybe it is because the labels are all encapsulated in \" signs
	BYTE join_by_in_quotes[1000];

	sprintf(join_by_in_quotes, "\"%s",join_by);
	for (j = 0; j < nb_cols; j++) {
		t = Table[0 * nb_cols + j];
		if (t >= 0) {
			c = strncmp(tokens[t], join_by_in_quotes, strlen(join_by_in_quotes));
#if 0
			cout << "comparing '" << tokens[t] << "' with '" << join_by << "' yields " << c << endl;
			for (h = 0; h < (INT)strlen(join_by); h++) {
				cout << h << " : " << (INT) tokens[t][h] << " : " << (INT) join_by[h] << endl;
				}
#endif
			if (c == 0) {
				return j;
				}
			}
		}
	cout << "by column not found, join_by='" << join_by << "'" << endl;
	cout << "The first row of the table is:" << endl;
	print_table_row_detailed(0, cout);
	//print_table(cout);
	//cout << "by column not found" << endl;
	exit(1);
}

void spreadsheet::tokenize(const BYTE *fname, 
	BYTE **&tokens, INT &nb_tokens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE buf[BUFSIZE];
	BYTE *p_buf;
	BYTE str[BUFSIZE];
	INT i, r;

	if (f_v) {
		cout << "spreadsheet::tokenize file=" << fname << endl;
		cout << "spreadsheet::tokenize verbose_level=" << verbose_level << endl;
		}
	{
	ifstream fp(fname);
	i = 0;
	while (TRUE) {
	if (fp.eof()) {
		break;
		}
	fp.getline(buf, BUFSIZE, '\n');
	if (f_vv) {
		cout << "Line read :'" << buf << "'" << endl;
		}
	p_buf = buf;
	if (strncmp(buf, "END", 3) == 0) {
		break;
		}

#if 0
	// delete negative characters:
	INT len = strlen(buf);
	for (i = 0, j = 0; i < len; i++) {
		if ((INT) buf[i] >= 0) {
			buf[j++] = buf[i];
			}
		else {
			cout << "spreadsheet::tokenize skipping negative character" << endl;
			}
		}
	buf[j] = 0;
#endif

	//i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		r = s_scan_token_comma_separated(&p_buf, str);

		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '" << str << "'" << endl;
			}
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		i++;
		}
	i++; // End of line
	}
	}
	nb_tokens = i;


	//f_vv = TRUE;


	tokens = NEW_PBYTE(nb_tokens);
	{
	ifstream fp(fname);
	i = 0;
	while (TRUE) {
	if (fp.eof()) {
		break;
		}
	fp.getline(buf, BUFSIZE, '\n');
	p_buf = buf;
	if (strncmp(buf, "END", 3) == 0) {
		break;
		}
	if (f_vv) {
		cout << "read line '" << p_buf << "'" << " i=" << i << endl;
		}

#if 0
	// delete negative characters:
	INT len = strlen(buf);
	for (i = 0, j = 0; i < len; i++) {
		if ((INT) buf[i] >= 0) {
			buf[j++] = buf[i];
			}
		else {
			cout << "spreadsheet::tokenize skipping negative character" << endl;
			}
		}
	buf[j] = 0;
#endif

	//i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		r = s_scan_token_comma_separated(&p_buf, str);
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		tokens[i] = NEW_BYTE(strlen(str) + 1);
		strcpy(tokens[i], str);
		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '" << tokens[i] << "'" << endl;
			}
		i++;
		}

#if 1
	sprintf(str, "END_OF_LINE");
	tokens[i] = NEW_BYTE(strlen(str) + 1);
	strcpy(tokens[i], str);
	if (f_vv) {
		cout << "Token " << setw(6) << i << " is '" << tokens[i] << "'" << endl;
		}
	i++;
#endif

	}
	}
}

void spreadsheet::remove_quotes(INT verbose_level)
{
	INT i, j, h, l, t;
	
	for (i = 1; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			t = Table[i * nb_cols + j];
			if (t < 0) {
				continue;
				}
			if (tokens[t][0] == '"') {
				l = strlen(tokens[t]);
				for (h = 1; h < l; h++) {
					tokens[t][h - 1] = tokens[t][h];
					}
				tokens[t][l - 1] = 0;
				}
			l = strlen(tokens[t]);
			if (l && tokens[t][l - 1] == '"') {
				tokens[t][l - 1] = 0;
				}
			}
		}
}

void spreadsheet::remove_rows(const BYTE *drop_column, const BYTE *drop_label, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, h, t, idx, nbr, f_delete;

	if (f_v) {
		cout << "spreadsheet::remove_rows" << endl;
		}
	nbr = nb_rows;
	idx = find_by_column(drop_column);
	cout << "drop column is " << idx << endl;
	cout << "drop label is " << drop_label << endl;
	h = 1;
	for (i = 1; i < nb_rows; i++) {
		t = Table[i * nb_cols + idx];
		if (t >= 0 && strcmp(tokens[t], drop_label) == 0) {
			f_delete = TRUE;
			}
		else {
			f_delete = FALSE;
			}
		if (!f_delete) {
			for (j = 0; j < nb_cols; j++) {
				Table[h * nb_cols + j] = Table[i * nb_cols + j];
				}
			h++;
			}
		}
	nb_rows = h;
	if (f_v) {
		cout << "spreadsheet::remove_rows, removed " << nbr - nb_rows << " rows" << endl;
		}
}

void spreadsheet::remove_rows_where_field_is_empty(const BYTE *drop_column, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, h, t, idx, nbr, f_delete;

	if (f_v) {
		cout << "spreadsheet::remove_rows_where_field_is_empty" << endl;
		}
	nbr = nb_rows;
	idx = find_by_column(drop_column);
	cout << "drop column is " << idx << endl;
	h = 1;
	for (i = 1; i < nb_rows; i++) {
		t = Table[i * nb_cols + idx];
		if (t == -1) {
			f_delete = TRUE;
			}
		else if (t >= 0 && strlen(tokens[t]) == 0) {
			f_delete = TRUE;
			}
		else {
			f_delete = FALSE;
			}
		if (!f_delete) {
			for (j = 0; j < nb_cols; j++) {
				Table[h * nb_cols + j] = Table[i * nb_cols + j];
				}
			h++;
			}
		}
	nb_rows = h;
	if (f_v) {
		cout << "spreadsheet::remove_rows_where_field_is_empty, removed " << nbr - nb_rows << " rows" << endl;
		}
}

void spreadsheet::find_rows(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, cnt;

	if (f_v) {
		cout << "find_rows" << endl;
		}
	cnt = 0;

	for (i = 0; i < nb_tokens; i++) {
		if (strcmp(tokens[i], "END_OF_LINE") == 0) {
			cnt++;
			}
		}
	nb_lines = cnt;
	line_start = NEW_INT(nb_lines + 1);
	line_size = NEW_INT(nb_lines);

	cnt = 0;

	line_start[cnt] = 0;
	for (i = 0; i < nb_tokens; i++) {
		if (f_vv) {
			cout << "cnt=" << cnt << " i=" << i << " tokens[i]=" << tokens[i] << endl;
			}
		if (strcmp(tokens[i], "END_OF_LINE") == 0) {
			line_size[cnt] = i - line_start[cnt];
			if (f_v) {
				cout << "end of line" << endl;
				}
			cnt++;
			line_start[cnt] = i + 1;
			}
		}
}

BYTE *spreadsheet::get_string(INT i, INT j)
{
	INT t;
	BYTE str[1000];
	BYTE *s;
	
	t = Table[i * nb_cols + j];
	//cout << "t=" << t << endl;
	if (t == -1) {
		s = NEW_BYTE(1);
		strcpy(s, "");
		}
	else {
		if (strlen(tokens[t]) >= 2 && tokens[t][0] == '"') {
			strcpy(str, tokens[t] + 1);
			str[strlen(str) - 1] = 0;
			}
		else {
			strcpy(str, tokens[t]);
			}

		s = NEW_BYTE(strlen(str) + 1);
		strcpy(s, str);
		}
	return s;
}

INT spreadsheet::get_INT(INT i, INT j)
{
	BYTE *p;
	INT a;

	p = get_string(i, j);
	a = atoi(p);
	return a;
}

double spreadsheet::get_double(INT i, INT j)
{
	BYTE *p;
	double a;

	p = get_string(i, j);
	a = atof(p);
	return a;
}

void spreadsheet::join_with(spreadsheet *S2, INT by1, INT by2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 3);
	//INT by1, by2;
	INT j1, j2, t1, t2;
	INT i1, i2;
	BYTE *label2;
	INT tt1, tt2;
	INT f_need_to_add;


	if (f_v) {
		cout << "spreadsheet::join_with" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	//by1 = find_by_column(join_by);
	//by2 = S2->find_by_column(join_by);

	if (f_vv) {
		cout << "by1=" << by1 << " by2=" << by2 << endl;
		}



	for (i2 = 1; i2 < S2->nb_rows; i2++) {
		BYTE *T2;
		t2 = S2->Table[i2 * S2->nb_cols + by2];
		if (t2 == -1) {
			continue;
			}
		T2 = S2->tokens[t2];
		if (strlen(T2) == 0) {
			continue;
			}
		for (i1 = 1; i1 < nb_rows; i1++) {
			if (Table[i1 * nb_cols + by1] == -1) {
				continue;
				}
			//cout << "i1=" << i1 << " label=" << tokens[Table[i1 * nb_cols + by1]] << endl;
			if (strcmp_with_or_without(tokens[Table[i1 * nb_cols + by1]], T2) == 0) {
				break;
				}
			}
		if (i1 == nb_rows) {
			cout << "adding a row corresponding to " << T2 << endl;
			reallocate_table_add_row();
			add_token(T2);
			Table[i1 * nb_cols + by1] = nb_tokens - 1;
			}
		}


	for (j2 = 0; j2 < S2->nb_cols; j2++) {
		if (f_vv) {
			cout << "j2=" << j2 << endl;
			}
		if (j2 == by2) {
			continue;
			}
		t2 = S2->Table[j2];
		if (t2 == -1) {
			continue;
			}
		if (f_vv) {
			cout << "joining column " << S2->tokens[t2] << endl;
			}

		for (j1 = 0; j1 < nb_cols; j1++) {
			if (j1 == by1) {
				continue;
				}
			t1 = Table[j1];
			if (t1 == -1) {
				continue;
				}
			if (strcmp_with_or_without(tokens[t1], S2->tokens[t2]) == 0) {
				break;
				}
			}
		if (j1 == nb_cols) {
			// reallocate Table
			cout << "reallocating table" << endl;
			reallocate_table();
			cout << "reallocating table done" << endl;
			add_token(S2->tokens[t2]);
			Table[0 * nb_cols + j1] = nb_tokens - 1;
			cout << "added token " << S2->tokens[t2] << " as a column heading" << endl;
 			}
		t1 = Table[j1];

		if (f_vv) {
			cout << "joining columns " << tokens[t1] << " and " << S2->tokens[t2] << endl;
			}
		
		for (i2 = 1; i2 < S2->nb_rows; i2++) {
			if (f_v3) {
				cout << "i2=" << i2 << endl;
				}
			tt2 = S2->Table[i2 * S2->nb_cols + j2];
			if (f_v3) {
				cout << "tt2=" << tt2 << endl;
				}
			if (tt2 == -1) {
				continue;
				}
			if (S2->Table[i2 * S2->nb_cols + by2] == -1) {
				continue;
				}
			label2 = S2->tokens[S2->Table[i2 * S2->nb_cols + by2]];
			if (f_v3) {
				cout << "label2='" << label2 << "'" << endl;
				}
			for (i1 = 1; i1 < nb_rows; i1++) {
				if (Table[i1 * nb_cols + by1] == -1) {
					continue;
					}
				//cout << "i1=" << i1 << " label=" << tokens[Table[i1 * nb_cols + by1]] << endl;
				if (strcmp_with_or_without(tokens[Table[i1 * nb_cols + by1]], label2) == 0) {
					break;
					}
				}
			if (i1 == nb_rows) {
				cout << "entry " << label2 << " not found in first table" << endl;
				exit(1);
				//reallocate_table_add_row();
				//Table[i1 * nb_cols + by1] = S2->Table[i2 * S2->nb_cols + by2];
				//exit(1);
				}
			else {
				cout << "label2 " << label2 << " found in row " << i1 << " in first table" << endl;
				}
			tt1 = Table[i1 * nb_cols + j1];
			f_need_to_add = TRUE;
			if (tt1 >= 0) {
				if (f_v3) {
					cout << "i1=" << i1 << " i2=" << i2 << " we have " << tokens[tt1] << " vs " << S2->tokens[tt2] << endl;
					}
				if (strcmp_with_or_without(tokens[tt1], S2->tokens[tt2]) == 0) {
					f_need_to_add = FALSE;
					}
				}
			if (f_v3) {
				cout << "f_need_to_add=" << f_need_to_add << endl;
				}
			if (f_need_to_add) {
				if (f_v3) {
					cout << "adding token " << S2->tokens[tt2] << endl;
					}
				add_token(S2->tokens[tt2]);
				Table[i1 * nb_cols + j1] = nb_tokens - 1;
				if (f_v3) {
					cout << "added token " << S2->tokens[tt2] << " check: " << tokens[Table[i1 * nb_cols + j1]] << endl;
					}
				}
			else {
				if (f_v3) {
					cout << "no need to add" << endl;
					}
				}
			} // next i2
		}
	if (f_v) {
		cout << "spreadsheet::join_with done" << endl;
		}
}

void spreadsheet::patch_with(spreadsheet *S2, BYTE *join_by)
{
	INT by1;
	INT t0, t1, t2, t3;
	INT i1, i2;
	INT what_idx;
	INT nb_patch = 0;


	by1 = find_by_column(join_by);

	cout << "spreadsheet::patch_with by1=" << by1 << endl;
	cout << "spreadsheet::patch_with S2->nb_rows=" << S2->nb_rows << endl;



	for (i2 = 1; i2 < S2->nb_rows; i2++) {
		BYTE *what;
		BYTE *who;
		BYTE *patch_value;
		t0 = S2->Table[i2 * S2->nb_cols + 0];
		t1 = S2->Table[i2 * S2->nb_cols + 1];
		t2 = S2->Table[i2 * S2->nb_cols + 2];
		t3 = S2->Table[i2 * S2->nb_cols + 3];
		if (t0 == -1) {
			continue;
			}
		what = S2->tokens[t0];
		if (strlen(what) == 0) {
			continue;
			}
		if (strcmp(what, "-1") == 0) {
			break;
			}
		who = S2->tokens[t1];
		if (strlen(who) == 0) {
			continue;
			}
		patch_value = S2->tokens[t3];

		for (i1 = 1; i1 < nb_rows; i1++) {
			if (Table[i1 * nb_cols + by1] == -1) {
				continue;
				}
			//cout << "i1=" << i1 << " label=" << tokens[Table[i1 * nb_cols + by1]] << endl;
			if (strcmp(tokens[Table[i1 * nb_cols + by1]], who) == 0) {
				break;
				}
			}
		if (i1 == nb_rows) {
			cout << "spreadsheet::patch_with Did not find " << who << " in first table" << endl;
			}
		else {
			what_idx = find_by_column(what);
			add_token(patch_value);
			Table[i1 * nb_cols + what_idx] = nb_tokens - 1;
			cout << "patch " << nb_patch << " applied, " << who << " now has " << patch_value << " in " << what << endl;
			nb_patch++;
			}
		}
	cout << "spreadsheet::patch_with applied " << nb_patch << " patches" << endl;

}



//



INT my_atoi(BYTE *str)
{
	INT a;
	if (strlen(str)) {
		sscanf(str, "%ld", &a); // don't use atoi, it fails for large numbers.
		return a;
		}
	return 0;
}

INT compare_strings(void *a, void *b, void *data)
{
	BYTE *A = (BYTE *) a;
	BYTE *B = (BYTE *) b;
	return strcmp(A, B);
}

INT strcmp_with_or_without(BYTE *p, BYTE *q)
{
	BYTE *str1;
	BYTE *str2;
	INT ret;

	if (p[0] == '"') {
		str1 = NEW_BYTE(strlen(p) + 1);
		strcpy(str1, p);
		}
	else {
		str1 = NEW_BYTE(strlen(p) + 3);
		strcpy(str1, "\"");
		strcpy(str1 + strlen(str1), p);
		strcpy(str1 + strlen(str1), "\"");
		}
	if (q[0] == '"') {
		str2 = NEW_BYTE(strlen(q) + 1);
		strcpy(str2, q);
		}
	else {
		str2 = NEW_BYTE(strlen(q) + 3);
		strcpy(str2, "\"");
		strcpy(str2 + strlen(str2), q);
		strcpy(str2 + strlen(str2), "\"");
		}
	ret = strcmp(str1, str2);
	FREE_BYTE(str1);
	FREE_BYTE(str2);
	return ret;
}



