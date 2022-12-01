// spreadsheet.cpp
// 
// Anton Betten
// July 18, 2012
//
// moved to GALOIS: March 15, 2013
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


spreadsheet::spreadsheet()
{
	tokens = NULL;
	nb_tokens = 0;

	line_start = NULL;
	line_size = NULL;
	nb_lines = 0;

	nb_rows = nb_cols = 0;
	Table = NULL;
}

spreadsheet::~spreadsheet()
{
	int i;
	
	if (tokens) {
		for (i = 0; i < nb_tokens; i++) {
			FREE_char(tokens[i]);
			}
		FREE_pchar(tokens);
		}
	if (line_start) {
		FREE_int(line_start);
		}
	if (line_size) {
		FREE_int(line_size);
		}
	if (Table) {
		FREE_int(Table);
		}
}

void spreadsheet::init_set_of_sets(set_of_sets *S, int f_make_heading)
{
	int s, i, j, a, h, len, offset = 0;
	char str[1000];

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
	Table = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = -1;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_pchar(nb_tokens + 1);

	h = 0;
	if (f_make_heading) {
		for (j = 0; j < s + 1; j++) {
			snprintf(str, 1000, "C%d", j);
			len = strlen(str);
			tokens[h] = NEW_char(len + 1);
			strcpy(tokens[h], str);
			Table[0 * nb_cols + j] = h;
			h++;
			}
		}
	for (i = 0; i < S->nb_sets; i++) {

		snprintf(str, 1000, "%ld", S->Set_size[i]);
		len = strlen(str);
		tokens[h] = NEW_char(len + 1);
		strcpy(tokens[h], str);
		Table[(i + offset) * nb_cols + 0] = h;
		h++;

		for (j = 0; j < S->Set_size[i]; j++) {
			a = S->Sets[i][j];
			
			snprintf(str, 1000, "%d", a);
			len = strlen(str);
			tokens[h] = NEW_char(len + 1);
			strcpy(tokens[h], str);
			Table[(i + offset) * nb_cols + 1 + j] = h;
			h++;

			}
		}
}

void spreadsheet::init_int_matrix(int nb_rows, int nb_cols, int *A)
{
	int i, len, a;
	char str[1000];
	
	spreadsheet::nb_rows = nb_rows;
	spreadsheet::nb_cols = nb_cols;
	Table = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = -1;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_pchar(nb_tokens + 1);
	for (i = 0; i < nb_tokens; i++) {
		a = A[i];
		snprintf(str, 1000, "%d", a);
		len = strlen(str);
		tokens[i] = NEW_char(len + 1);
		strcpy(tokens[i], str);
		Table[i] = i;
		}
}

void spreadsheet::init_empty_table(int nb_rows, int nb_cols)
{
	int i;
	
	spreadsheet::nb_rows = nb_rows;
	spreadsheet::nb_cols = nb_cols;
	Table = NEW_int(nb_rows * nb_cols);
	for (i = 0; i < nb_rows * nb_cols; i++) {
		Table[i] = i;
		}
	nb_tokens = nb_rows * nb_cols;
	tokens = NEW_pchar(nb_tokens);
	for (i = 0; i < nb_tokens; i++) {
		tokens[i] = NULL;
		}
}

void spreadsheet::fill_entry_with_text(int row_idx,
		int col_idx, const char *text)
{
	int l, t;
	
	t = Table[row_idx * nb_cols + col_idx];
	if (tokens[t]) {
		//cout << "fill_column_with_text before FREE_char i="
		//<< i << " col_idx=" << col_idx << " t=" << t << endl;
		FREE_char(tokens[t]);
		}
	l = strlen(text);
	tokens[t] = NEW_char(l + 1);
	strcpy(tokens[t], text);
}

void spreadsheet::fill_entry_with_text(int row_idx,
		int col_idx, std::string &text)
{
	int l, t;

	t = Table[row_idx * nb_cols + col_idx];
	if (tokens[t]) {
		//cout << "fill_column_with_text before FREE_char i="
		//<< i << " col_idx=" << col_idx << " t=" << t << endl;
		FREE_char(tokens[t]);
		}
	l = text.size();
	tokens[t] = NEW_char(l + 1);
	strcpy(tokens[t], text.c_str());
}

void spreadsheet::set_entry_lint(int row_idx,
		int col_idx, long int val)
{
	int l, t;
	char str[1000];

	snprintf(str, sizeof(str), "%ld", val);

	t = Table[row_idx * nb_cols + col_idx];
	if (tokens[t]) {
		//cout << "fill_column_with_text before FREE_char i="
		//<< i << " col_idx=" << col_idx << " t=" << t << endl;
		FREE_char(tokens[t]);
		}
	l = strlen(str);
	tokens[t] = NEW_char(l + 1);
	strcpy(tokens[t], str);
}


void spreadsheet::fill_column_with_text(int col_idx,
		std::string *text, const char *heading)
{
	int i, l, t;
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_text before FREE_char i="
			//<< i << " col_idx=" << col_idx << " t=" << t << endl;
			FREE_char(tokens[t]);
			}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], heading);
			}
		else {
			l = text[i - 1].length();
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], text[i - 1].c_str());
			}
		}
}

void spreadsheet::fill_column_with_int(int col_idx,
		int *data, const char *heading)
{
	int i, l, t;
	char str[1000];
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_int before FREE_char i=" << i
			//<< " col_idx=" << col_idx << " t=" << t << endl;
			FREE_char(tokens[t]);
			}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], heading);
			}
		else {
			snprintf(str, 1000, "%d", data[i - 1]);
			l = strlen(str);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], str);
			}
		}
}

void spreadsheet::fill_column_with_lint(int col_idx,
		long int *data, const char *heading)
{
	int i, l, t;
	char str[1000];

	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_int before FREE_char i=" << i
			//<< " col_idx=" << col_idx << " t=" << t << endl;
			FREE_char(tokens[t]);
		}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], heading);
		}
		else {
			snprintf(str, sizeof(str), "%ld", data[i - 1]);
			l = strlen(str);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], str);
		}
	}
}

void spreadsheet::fill_column_with_row_index(
		int col_idx, const char *heading)
{
	int i, l, t;
	char str[1000];
	
	for (i = 0; i < nb_rows; i++) {
		t = Table[i * nb_cols + col_idx];
		if (tokens[t]) {
			//cout << "fill_column_with_row_index before FREE_char i="
			//<< i << " col_idx=" << col_idx << " t=" << t << endl;
			FREE_char(tokens[t]);
		}
		if (i == 0) {
			l = strlen(heading);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], heading);
		}
		else {
			snprintf(str, 1000, "%d", i - 1);
			l = strlen(str);
			tokens[t] = NEW_char(l + 1);
			strcpy(tokens[t], str);
		}
	}
}

void spreadsheet::add_token(const char *label)
{
	char **tokens2;
	int i, j, len;

	tokens2 = NEW_pchar(nb_tokens + 1);
	for (i = 0; i < nb_tokens; i++) {
		tokens2[i] = tokens[i];
	}
	len = strlen(label);
	tokens2[nb_tokens] = NEW_char(len + 1);
	for (i = 0, j = 0; i < len; i++) {
		if ((int)label[i] < 0) {
			cout << "spreadsheet::add_token negative character "
					<< (int) label[i] << endl;
		}
		else {
			tokens2[nb_tokens][j++] = label[i];
		}
	}
	tokens2[nb_tokens][j++] = 0;
	//strcpy(tokens2[nb_tokens], label);
	FREE_pchar(tokens);
	tokens = tokens2;
	nb_tokens++;
}

void spreadsheet::save(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;


	{
	ofstream f(fname);
	print_table(f, FALSE);
	f << "END" << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
}

void spreadsheet::read_spreadsheet(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet reading file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
		}
	

	
	if (f_v) {
		cout << "spreadsheet::read_spreadsheet before tokenize" << endl;
	}
	tokenize(fname, tokens, nb_tokens, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "spreadsheet::read_spreadsheet after tokenize" << endl;
	}

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet read file with "
				<< nb_tokens << " tokens" << endl;

		if (f_vv) {
			for (i = 0; i < nb_tokens; i++) {
				cout << setw(6) << i << " : '" << tokens[i] << "'" << endl;
				}
			}
		}



	if (f_v) {
		cout << "spreadsheet::read_spreadsheet before find_rows" << endl;
	}
	find_rows(0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "spreadsheet::read_spreadsheet Found "
				<< nb_lines << " lines" << endl;
		}
	
	if (FALSE) {
		{
		int f, l, j;
	
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

	int j;
	
	nb_rows = nb_lines;
	nb_cols = line_size[0];
	Table = NEW_int(nb_rows * nb_cols);
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

	if (FALSE) {
		cout << "spreadsheet::read_spreadsheet" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << "row " << i << " column " << j << " entry '"
						<< tokens[Table[i * nb_cols + j]] << "'" << endl;
				}
			}
		}
	
	if (f_v) {
		cout << "spreadsheet::read_spreadsheet reading file "
				<< fname << " of size " << Fio.file_size(fname)
				<< " done" << endl;
		}
	
}

void spreadsheet::print_table(ostream &ost, int f_enclose_in_parentheses)
{
	int i;
	
	//cout << "Table:" << endl;
	for (i = 0; i < nb_rows; i++) {
		print_table_row(i, f_enclose_in_parentheses, ost);
		}
}

void spreadsheet::print_table_latex_all_columns(
		ostream &ost, int f_enclose_in_parentheses)
{
	int i, j;
	int *f_column_select;

	f_column_select = NEW_int(nb_cols);
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
		print_table_row_latex(i,
				f_column_select, f_enclose_in_parentheses, ost);
		}
	ost << "\\end{tabular}" << endl;

	FREE_int(f_column_select);
}

void spreadsheet::print_table_latex(ostream &ost,
		int *f_column_select, int f_enclose_in_parentheses,
		int nb_lines_per_table)
{
	int I, i, j;
	int nb_r;
	

	nb_r = nb_rows - 1; // take away one because of header

	//cout << "Table:" << endl;
	for (I = 0; I < (nb_r + nb_lines_per_table - 1) / nb_lines_per_table; I++) {
		ost << "\\begin{tabular}[t]{|";
		for (j = 0; j < nb_cols; j++) {
			if (f_column_select[j]) {
				ost << "r|";
				//ost << "p{3cm}|";
			}
		}
		ost << "}" << endl;
		ost << "\\hline" << endl;

		print_table_row_latex(0,
				f_column_select,
				f_enclose_in_parentheses,
				ost);
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;


		for (i = 0; i < nb_lines_per_table; i++) {
			if (1 + I * nb_lines_per_table + i < nb_rows) {
				print_table_row_latex(1 + I * nb_lines_per_table + i,
						f_column_select,
						f_enclose_in_parentheses,
						ost);
				ost << "\\hline" << endl;
			}
		}
		ost << "\\end{tabular}" << endl;
	}
}

void spreadsheet::print_table_row(int row,
		int f_enclose_in_parentheses, ostream &ost)
{
	int j, t; //, h;
	int f_enclose;
	
	//cout << "Row " << row << " : ";
	for (j = 0; j < nb_cols; j++) {
		t = Table[row * nb_cols + j];
		if (t >= 0) {
#if 0
			if (row == 0 && j == 0) {
				cout << "printing token '" << tokens[t] << "'" << endl;
				for (h = 0; h < 10; h++) {
					cout << h << " : " << (int) tokens[t][h] << endl;
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
				cout << "spreadsheet::print_table_row token[t] == NULL, "
						"t = " << t << endl;
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

void spreadsheet::print_table_row_latex(int row,
		int *f_column_select, int f_enclose_in_parentheses,
		ostream &ost)
{
	int j, t, l; //, h;
	int f_first = TRUE;
	
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
						cout << h << " : " << (int) tokens[t][h] << endl;
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

void spreadsheet::print_table_row_detailed(int row, ostream &ost)
{
	int j, t;
	
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

void spreadsheet::print_table_row_with_column_selection(int row,
		int f_enclose_in_parentheses,
		int *Col_selection, int nb_cols_selected,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int col, h;
	//int f_enclose;

	if (f_v) {
		cout << "spreadsheet::print_table_row_with_column_selection" << endl;
	}
	//cout << "Row " << row << " : ";
	for (h = 0; h < nb_cols_selected; h++) {
		col = Col_selection[h];

		std::string str;

		get_string(str, row, col);

		if (f_v) {
			cout << "row " << row << " col " << col << " : " << str << endl;
		}

#if 0
		t = Table[row * nb_cols + j];
		if (t >= 0) {
#if 0
			if (row == 0 && j == 0) {
				cout << "printing token '" << tokens[t] << "'" << endl;
				for (h = 0; h < 10; h++) {
					cout << h << " : " << (int) tokens[t][h] << endl;
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
				cout << "spreadsheet::print_table_row_with_column_selection token[t] == NULL, "
						"t = " << t << endl;
			}
			else {
				ost << tokens[t];
			}
			if (f_enclose) {
				ost << "\"";
			}
		}
#endif
		ost << str;
		if (h < nb_cols_selected - 1) {
			ost << ",";
		}
	}
	ost << endl;
}


void spreadsheet::print_table_with_row_selection(
		int *f_selected, ostream &ost)
{
	int i;
	
	//cout << "Table:" << endl;
	for (i = 0; i < nb_rows; i++) {
		if (!f_selected[i]) {
			continue;
			}
		print_table_row(i, FALSE, ost);
		}
}

void spreadsheet::print_table_sorted(ostream &ost,
		const char *sort_by)
{
	int i, t, ii;
	int idx;
	int *perm;
	char **labels;
	sorting Sorting;
	
	idx = find_by_column(sort_by);
	perm = NEW_int(nb_rows - 1);
	labels = NEW_pchar(nb_rows - 1);
	for (i = 0; i < nb_rows - 1; i++) {
		perm[i] = i;
		t = Table[(i + 1) * nb_cols + idx];
		if (t >= 0) {
			if (tokens[t][0] == '"') {
				labels[i] = NEW_char(strlen(tokens[t]) + 1);
				strcpy(labels[i], tokens[t] + 1);
				}
			else {
				labels[i] = NEW_char(strlen(tokens[t]) + 1);
				strcpy(labels[i], tokens[t]);
				}
			}
		else {
			labels[i] = NEW_char(1);
			labels[i][0] = 0;
			}
		}
	
	Sorting.quicksort_array_with_perm(nb_rows - 1, (void **) labels, perm,
		string_tools_compare_strings, NULL /*void *data*/);

	
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


void spreadsheet::add_column_with_constant_value(const char *label, char *value)
{
	int i;

	reallocate_table();
	add_token(label);
	Table[0 * nb_cols + nb_cols - 1] = nb_tokens - 1;
	for (i = 1; i < nb_rows; i++) {
		add_token(value);
		Table[i * nb_cols + nb_cols - 1] = nb_tokens - 1;
		}
	
}

void spreadsheet::add_column_with_int(const char *label, int *Value)
{
	int i;
	char str[1000];

	reallocate_table();
	add_token(label);
	Table[0 * nb_cols + nb_cols - 1] = nb_tokens - 1;
	for (i = 1; i < nb_rows; i++) {
		snprintf(str, 1000, "%d", Value[i - 1]);
		add_token(str);
		Table[i * nb_cols + nb_cols - 1] = nb_tokens - 1;
		}

}
void spreadsheet::add_column_with_text(const char *label, char **Value)
{
	int i;

	reallocate_table();
	add_token(label);
	Table[0 * nb_cols + nb_cols - 1] = nb_tokens - 1;
	for (i = 1; i < nb_rows; i++) {
		add_token(Value[i - 1]);
		Table[i * nb_cols + nb_cols - 1] = nb_tokens - 1;
		}

}

void spreadsheet::reallocate_table()
{
	int i, j;
	int *Table2;

	Table2 = NEW_int(nb_rows * (nb_cols + 1));
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table2[i * (nb_cols + 1) + j] = Table[i * nb_cols + j];
			}
		Table2[i * (nb_cols + 1) + nb_cols] = -1;
		}
	FREE_int(Table);
	Table = Table2;
	nb_cols++;
}

void spreadsheet::reallocate_table_add_row()
{
	int i, j;
	int *Table2;

	Table2 = NEW_int((nb_rows + 1) * nb_cols);
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table2[i * nb_cols + j] = Table[i * nb_cols + j];
			}
		}
	for (j = 0; j < nb_cols; j++) {
		Table2[nb_rows * nb_cols + j] = -1;
		}
	FREE_int(Table);
	Table = Table2;
	nb_rows++;
}

int spreadsheet::find_column(std::string &column_label)
{
	return find_by_column(column_label.c_str());
}

int spreadsheet::find_by_column(const char *join_by)
{
	int j, t, c; //, h;
	
	for (j = 0; j < nb_cols; j++) {
		t = Table[0 * nb_cols + j];
		if (t >= 0) {
			c = strncmp(tokens[t], join_by, strlen(join_by));
#if 0
			cout << "comparing '" << tokens[t] << "' with '"
					<< join_by << "' yields " << c << endl;
			for (h = 0; h < (int)strlen(join_by); h++) {
				cout << h << " : " << tokens[t][h] << " : "
						<< join_by[h] << endl;
				}
#endif
			if (c == 0) {
				return j;
				}
			}
		}
	// in case we don't find it, maybe it is because the labels
	//are all encapsulated in \" signs
	char join_by_in_quotes[1000];

	snprintf(join_by_in_quotes, 1000, "\"%s",join_by);
	for (j = 0; j < nb_cols; j++) {
		t = Table[0 * nb_cols + j];
		if (t >= 0) {
			c = strncmp(tokens[t], join_by_in_quotes,
					strlen(join_by_in_quotes));
#if 0
			cout << "comparing '" << tokens[t] << "' with '"
					<< join_by << "' yields " << c << endl;
			for (h = 0; h < (int)strlen(join_by); h++) {
				cout << h << " : " << (int) tokens[t][h] << " : "
						<< (int) join_by[h] << endl;
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

void spreadsheet::tokenize(std::string &fname,
	char **&tokens, int &nb_tokens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char *buf;
	const char *p_buf;
	char *str;
	int sz;
	int line_cnt, i; //, r;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "spreadsheet::tokenize file=" << fname << endl;
		cout << "spreadsheet::tokenize verbose_level="
				<< verbose_level << endl;
		}

	sz = Fio.file_size(fname);
	if (f_v) {
		cout << "spreadsheet::tokenize file size " << sz << endl;
	}

	buf = NEW_char(sz + 1);
	if (f_v) {
		cout << "spreadsheet::tokenize after NEW_char 1" << endl;
	}

	str = NEW_char(sz + 1);
	if (f_v) {
		cout << "spreadsheet::tokenize after NEW_char 2" << endl;
	}

	{
		string_tools ST;
		ifstream fp(fname);
		i = 0;
		line_cnt = 0;
		while (TRUE) {
			line_cnt++;
			if (fp.eof()) {
				if (f_vv) {
					cout << "eof" << endl;
				}
				break;
			}
			if (f_vv) {
				cout << "before fp.getline" << endl;
			}
			fp.getline(buf, sz, '\n');
			if (f_vv) {
				cout << "Line read :'" << buf << "'" << endl;
			}
			if (strlen(buf) == 0) {
				break;
			}
			p_buf = buf;
			if (strncmp(buf, "END", 3) == 0) {
				break;
			}

	#if 0
			// delete negative characters:
			int len = strlen(buf);
			for (i = 0, j = 0; i < len; i++) {
				if ((int) buf[i] >= 0) {
					buf[j++] = buf[i];
					}
				else {
					cout << "spreadsheet::tokenize skipping "
							"negative character" << endl;
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
				if (!ST.s_scan_token_comma_separated(&p_buf, str, verbose_level - 1)) {
					break;
				}

				if (f_vv) {
					cout << "Line " << line_cnt << ", token " << setw(6) << i << " is '"
							<< str << "'" << endl;
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


	tokens = NEW_pchar(nb_tokens);
	{
		string_tools ST;
		ifstream fp(fname);
		i = 0;
		while (TRUE) {
			if (fp.eof()) {
				break;
			}
			fp.getline(buf, sz, '\n');
			p_buf = buf;
			if (strncmp(buf, "END", 3) == 0) {
				break;
			}
			if (f_vv) {
				cout << "read line '" << p_buf << "'" << " i=" << i << endl;
			}

		#if 0
			// delete negative characters:
			int len = strlen(buf);
			for (i = 0, j = 0; i < len; i++) {
				if ((int) buf[i] >= 0) {
					buf[j++] = buf[i];
				}
				else {
					cout << "spreadsheet::tokenize skipping "
							"negative character" << endl;
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
				if (!ST.s_scan_token_comma_separated(&p_buf, str, verbose_level - 1)) {
					break;
				}
		#if 0
				if (strcmp(str, ",") == 0) {
					continue;
				}
		#endif
				tokens[i] = NEW_char(strlen(str) + 1);
				strcpy(tokens[i], str);
				if (f_vv) {
					cout << "Token " << setw(6) << i << " is '"
							<< tokens[i] << "'" << endl;
				}
				i++;
			}

		#if 1
			snprintf(str, sz, "END_OF_LINE");
			tokens[i] = NEW_char(strlen(str) + 1);
			strcpy(tokens[i], str);
			if (f_vv) {
				cout << "Token " << setw(6) << i << " is '"
						<< tokens[i] << "'" << endl;
			}
			i++;
		#endif

		}
	}
	FREE_char(buf);
	FREE_char(str);
}

void spreadsheet::remove_quotes(int verbose_level)
{
	int i, j, h, l, t;
	
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

void spreadsheet::remove_rows(const char *drop_column,
		const char *drop_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, h, t, idx, nbr, f_delete;

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
		cout << "spreadsheet::remove_rows, removed "
				<< nbr - nb_rows << " rows" << endl;
		}
}

void spreadsheet::remove_rows_where_field_is_empty(
		const char *drop_column, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, h, t, idx, nbr, f_delete;

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
		cout << "spreadsheet::remove_rows_where_field_is_empty, "
				"removed " << nbr - nb_rows << " rows" << endl;
		}
}

void spreadsheet::find_rows(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, cnt;

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
	line_start = NEW_int(nb_lines + 1);
	line_size = NEW_int(nb_lines);

	cnt = 0;

	line_start[cnt] = 0;
	for (i = 0; i < nb_tokens; i++) {
		if (f_vv) {
			cout << "cnt=" << cnt << " i=" << i
					<< " tokens[i]=" << tokens[i] << endl;
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

void spreadsheet::get_value_double_or_NA(int i, int j,
		double &val, int &f_NA)
{
	string str;
	string_tools ST;

	get_string(str, i, j);
	cout << "spreadsheet::get_value_double_or_NA str=" << str << endl;
	if (ST.stringcmp(str, "NA") == 0) {
		val = 0;
		f_NA = TRUE;
	}
	else {
		val = get_double(i, j);
		f_NA = FALSE;
	}
}

void spreadsheet::get_string(std::string &str, int i, int j)
{
	int t; // l
	//char *str;
	//char *s;
	
	t = Table[i * nb_cols + j];
	//cout << "t=" << t << endl;
	if (t == -1) {
		str.assign("");
		//s = NEW_char(1);
		//strcpy(s, "");
		}
	else {
		//str = NEW_char(strlen(tokens[t]) + 1);
		//l = strlen(tokens[t]);
		str.assign(tokens[t]);
#if 0
		if (l >= 2 && tokens[t][0] == '"' && tokens[t][l - 1] == '"') {
			tokens[t][l - 1] = 0;
			str.assign(tokens[t] + 1);

			//strcpy(str, tokens[t] + 1);
			//str[strlen(str) - 1] = 0;
			}
		else {
			str.assign(tokens[t]);
			//strcpy(str, tokens[t]);
			}
#endif

		//s = NEW_char(strlen(str) + 1);
		//strcpy(s, str);
		//FREE_char(str);
		}
	//return s;
}

long int spreadsheet::get_lint(int i, int j)
{
	string str;
	long int a;
	string_tools ST;

	get_string(str, i, j);

	a = ST.strtolint(str);

	return a;
}

double spreadsheet::get_double(int i, int j)
{
	string str;
	double a;
	string_tools ST;

	get_string(str, i, j);
	a = ST.strtof(str);
	return a;
}

void spreadsheet::join_with(spreadsheet *S2,
		int by1, int by2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	//int by1, by2;
	int j1, j2, t1, t2;
	int i1, i2;
	char *label2;
	int tt1, tt2;
	int f_need_to_add;
	string_tools ST;


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
		char *T2;
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
			//cout << "i1=" << i1 << " label="
			//<< tokens[Table[i1 * nb_cols + by1]] << endl;
			if (ST.strcmp_with_or_without(
					tokens[Table[i1 * nb_cols + by1]], T2) == 0) {
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
			if (ST.strcmp_with_or_without(tokens[t1], S2->tokens[t2]) == 0) {
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
			cout << "added token " << S2->tokens[t2]
				<< " as a column heading" << endl;
 			}
		t1 = Table[j1];

		if (f_vv) {
			cout << "joining columns " << tokens[t1] << " and "
					<< S2->tokens[t2] << endl;
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
				//cout << "i1=" << i1 << " label="
				//<< tokens[Table[i1 * nb_cols + by1]] << endl;
				if (ST.strcmp_with_or_without(
						tokens[Table[i1 * nb_cols + by1]], label2) == 0) {
					break;
					}
				}
			if (i1 == nb_rows) {
				cout << "entry " << label2 << " not found in "
						"first table" << endl;
				exit(1);
				//reallocate_table_add_row();
				//Table[i1 * nb_cols + by1] =
				//S2->Table[i2 * S2->nb_cols + by2];
				//exit(1);
				}
			else {
				cout << "label2 " << label2 << " found in row "
						<< i1 << " in first table" << endl;
				}
			tt1 = Table[i1 * nb_cols + j1];
			f_need_to_add = TRUE;
			if (tt1 >= 0) {
				if (f_v3) {
					cout << "i1=" << i1 << " i2=" << i2 << " we have "
							<< tokens[tt1] << " vs "
							<< S2->tokens[tt2] << endl;
					}
				if (ST.strcmp_with_or_without(tokens[tt1],
						S2->tokens[tt2]) == 0) {
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
					cout << "added token " << S2->tokens[tt2]
						<< " check: " << tokens[Table[i1 * nb_cols + j1]]
						<< endl;
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

void spreadsheet::patch_with(spreadsheet *S2, char *join_by)
{
	int by1;
	int t0, t1, /*t2,*/ t3;
	int i1, i2;
	int what_idx;
	int nb_patch = 0;


	by1 = find_by_column(join_by);

	cout << "spreadsheet::patch_with by1=" << by1 << endl;
	cout << "spreadsheet::patch_with S2->nb_rows=" << S2->nb_rows << endl;



	for (i2 = 1; i2 < S2->nb_rows; i2++) {
		char *what;
		char *who;
		char *patch_value;
		t0 = S2->Table[i2 * S2->nb_cols + 0];
		t1 = S2->Table[i2 * S2->nb_cols + 1];
		//t2 = S2->Table[i2 * S2->nb_cols + 2];
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
			//cout << "i1=" << i1 << " label="
			//<< tokens[Table[i1 * nb_cols + by1]] << endl;
			if (strcmp(tokens[Table[i1 * nb_cols + by1]], who) == 0) {
				break;
				}
			}
		if (i1 == nb_rows) {
			cout << "spreadsheet::patch_with Did not find " << who
					<< " in first table" << endl;
			}
		else {
			what_idx = find_by_column(what);
			add_token(patch_value);
			Table[i1 * nb_cols + what_idx] = nb_tokens - 1;
			cout << "patch " << nb_patch << " applied, " << who
					<< " now has " << patch_value << " in " << what << endl;
			nb_patch++;
			}
		}
	cout << "spreadsheet::patch_with applied " << nb_patch
			<< " patches" << endl;

}



//





}}}


