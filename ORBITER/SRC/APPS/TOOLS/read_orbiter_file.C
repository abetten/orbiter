// read_orbiter_file.C
// 
// Anton Betten
// June 7, 2018
//
// 
//

#include "orbiter.h"
#include "discreta.h"

#define MY_BUFSIZE ONE_MILLION

int main(int argc, char **argv)
{
	INT i, j;
	INT verbose_level = 0;
	INT f_file = FALSE;
	const BYTE *file_name = NULL;
	INT f_save = FALSE;
	const BYTE *file_name_save = NULL;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			file_name = argv[++i];
			cout << "-file " << file_name << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			file_name_save = argv[++i];
			cout << "-save " << file_name_save << endl;
			}
		}
	if (!f_file) {
		cout << "please use -file <file>" << endl;
		exit(1);
		}
	

	INT nb_orbits;
	BYTE **data;
	INT *Set_sizes;
	INT **Sets;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii;
	INT *Casenumbers;

	INT *Ago;
	
#if 0
	//INT **sets;
	//INT *set_sizes;
	cout << "before read_and_parse_data_file " << fname << endl;
	read_and_parse_data_file(fname, Nb_rows[i], 
		data, sets, set_sizes, 
		verbose_level);
#endif

	cout << "before try_to_read_file" << endl;
	if (try_to_read_file(file_name, nb_orbits, data, 0 /* verbose_level */)) {
		cout << "read file " << file_name << " nb_cases = " << nb_orbits << endl;
		}
	else {
		cout << "couldn't read file " << file_name << endl;
		exit(1);
		}

	cout << "before parse_sets" << endl;
	parse_sets(nb_orbits, data, FALSE /* f_casenumbers */, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level);
	cout << "after parse_sets, scanning Ago[i]" << endl;
	Ago = NEW_INT(nb_orbits);
	for (j = 0; j < nb_orbits; j++) {
		Ago[j] = atoi(Ago_ascii[j]);
		}
	cout << "after scanning Ago" << endl;

	

	if (nb_orbits == 0) {
		cout << "nb_orbits == 0" << endl;
		exit(1);
		}

	

	INT level;
	PBYTE *Text_level;
	PBYTE *Text_node;
	PBYTE *Text_orbit_reps;
	PBYTE *Text_stab_order;
	BYTE str[10000];

	level = Set_sizes[0];

	cout << "level=" << level << endl;

	Text_level = NEW_PBYTE(nb_orbits);
	Text_node = NEW_PBYTE(nb_orbits);
	Text_orbit_reps = NEW_PBYTE(nb_orbits);
	Text_stab_order = NEW_PBYTE(nb_orbits);

	for (i = 0; i < nb_orbits; i++) {
		sprintf(str, "%ld", level);
		Text_level[i] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_level[i], str);

		sprintf(str, "%ld", i);
		Text_node[i] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_node[i], str);

		INT_vec_print_to_str(str, Sets[i], level);
		Text_orbit_reps[i] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_orbit_reps[i], str);
		
		Text_stab_order[i] = NEW_BYTE(strlen(Ago_ascii[i]) + 1);
		strcpy(Text_stab_order[i], Ago_ascii[i]);
		
		}

	spreadsheet *Sp;
	
	Sp = new spreadsheet;
	Sp->init_empty_table(nb_orbits + 1, 5);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const BYTE **) Text_level, "Level");
	Sp->fill_column_with_text(2, (const BYTE **) Text_node, "Node");
	Sp->fill_column_with_text(3, (const BYTE **) Text_orbit_reps, "Orbit rep");
	Sp->fill_column_with_text(4, (const BYTE **) Text_stab_order, "Stab order");

	if (f_save) {
		cout << "before Sp->save " << file_name_save << endl;
		Sp->save(file_name_save, verbose_level);
		cout << "after Sp->save " << file_name_save << endl;
		}

	for (i = 0; i < nb_orbits; i++) {
		FREE_BYTE(Text_level[i]);
		}
	FREE_PBYTE(Text_level);
	for (i = 0; i < nb_orbits; i++) {
		FREE_BYTE(Text_node[i]);
		}
	FREE_PBYTE(Text_node);
	for (i = 0; i < nb_orbits; i++) {
		FREE_BYTE(Text_orbit_reps[i]);
		}
	FREE_PBYTE(Text_orbit_reps);
	for (i = 0; i < nb_orbits; i++) {
		FREE_BYTE(Text_stab_order[i]);
		}

}

