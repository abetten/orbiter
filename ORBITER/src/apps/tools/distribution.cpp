// distribution.C
// 
// Anton Betten
// June 7, 2018
//
// 
//
//

#include "orbiter.h"

// global data:

INT t0; // the system time when the program started

 
int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_file = FALSE;
	const BYTE *fname;
	const BYTE *column;
	INT f_secondary = FALSE;
	const BYTE *secondary_column;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			column = argv[++i];
			cout << "-file " << fname << " " << column << endl;
			}
		else if (strcmp(argv[i], "-secondary") == 0) {
			f_secondary = TRUE;
			secondary_column = argv[++i];
			cout << "-secondary " << secondary_column << endl;
			}
		}
	if (f_file == FALSE) {
		cout << "Please specify -file <fname> <column>" << endl;
		exit(1);
		}

	spreadsheet *S;

	INT idx, len, j, t;
	INT *Data;
	
	S = NEW_OBJECT(spreadsheet);
	


	cout << "Reading " << fname << endl;
	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	cout << "Read file " << fname << endl;
	//S->print_table(cout, FALSE /* f_enclose_in_parentheses */);
	cout << "S->nb_rows=" << S->nb_rows << endl;

	idx = S->find_by_column(column);
	Data = NEW_INT(S->nb_rows);
	for (i = 1; i < S->nb_rows; i++) {
		j = i - 1;
		t = S->Table[i * S->nb_cols + idx];
		if (t == -1) {
			Data[j] = 0;
			continue;
			}
		if (S->tokens[t][0] == '"') {
			Data[j] = my_atoi(S->tokens[t] + 1);
			}
		else {
			Data[j] = my_atoi(S->tokens[t]);
			}
		}
	len = S->nb_rows - 1;


	classify C;

	C.init(Data, len, FALSE, 0);
	cout << "Data:" << endl;
	C.print(FALSE /* f_backwards */);
	cout << endl;
	C.print_naked_tex(cout, FALSE /* f_backwards */);
	cout << endl;
	cout << "Number of values = " << len << endl;
	cout << endl;

	if (f_secondary) {
		INT idx2;
		INT f, l, j, pos;
		INT *Data2;
		classify *C2;
		
		C2 = NEW_OBJECTS(classify, C.nb_types);
		idx2 = S->find_by_column(secondary_column);
		cout << "secondary distribution for column "
				<< secondary_column << " idx2=" << idx2 << endl;
		for (i = 0; i < C.nb_types; i++) {
			cout << "type " << i << " / " << C.nb_types << ":" << endl;
			f = C.type_first[i];
			l = C.type_len[i];
			Data2 = NEW_INT(l);
			for (j = 0; j < l; j++) {
				pos = C.sorting_perm_inv[f + j];
				//Data2[j] = Data[pos];
				Data2[j] = S->get_INT(pos + 1, idx2);
				}

			C2[i].init(Data2, l, FALSE, 0);
			cout << "Data2:" << endl;
			C2[i].print(FALSE /* f_backwards */);
			cout << endl;
			C2[i].print_naked_tex(cout, FALSE /* f_backwards */);
			cout << endl;
			cout << "Number of values = " << l << endl;
			cout << endl;
			

			FREE_INT(Data2);
			
			}

		for (i = 0; i < C.nb_types; i++) {
			cout << i << " & ";
			f = C.type_first[i];
			l = C.type_len[i];
			pos = C.sorting_perm_inv[f + 0];
			cout << Data[pos] << " & ";
			C2[i].print_naked_tex(cout, FALSE /* f_backwards */);
			cout << "\\\\" << endl;
			}
		
		}

	FREE_OBJECT(S);
}


