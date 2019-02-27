// read_orbiter_file.C
// 
// Anton Betten
// June 7, 2018
//
// 
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


#define MY_BUFSIZE ONE_MILLION

int main(int argc, char **argv)
{
	int i, j;
	int verbose_level = 0;
	int f_file = FALSE;
	const char *file_name = NULL;
	int f_save = FALSE;
	const char *file_name_save = NULL;

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
	

#if 0
	int nb_orbits;
	char **data;
	int *Set_sizes;
	int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;

	int *Ago;
	
#if 0
	//int **sets;
	//int *set_sizes;
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
#else
	orbiter_data_file *ODF;
	int *Ago;
	char fname[1000];
	char candidates_fname[1000];
	int f_has_candidates = FALSE;
	int level;

	sprintf(fname, "%s", file_name);
	sprintf(candidates_fname, "%s_candidates.bin", fname);
	if (file_size(candidates_fname) > 0) {
		f_has_candidates = TRUE;
	}
	ODF = NEW_OBJECT(orbiter_data_file);
	ODF->load(fname, verbose_level);
	if (ODF->nb_cases == 0) {
		cout << "The file is empty" << endl;
		exit(1);
	}
	level = ODF->set_sizes[0];
	cout << "found " << ODF->nb_cases << " orbits at level " << level << endl;

#endif
	cout << "after parse_sets, scanning Ago[i]" << endl;
	Ago = NEW_int(ODF->nb_cases);
	for (j = 0; j < ODF->nb_cases; j++) {
		Ago[j] = atoi(ODF->Ago_ascii[j]);
		}
	cout << "after scanning Ago" << endl;

	

	if (ODF->nb_cases == 0) {
		cout << "ODF->nb_cases == 0" << endl;
		exit(1);
		}

	int nb_orbits = ODF->nb_cases;

	//int level;
	pchar *Text_level;
	pchar *Text_node;
	pchar *Text_orbit_reps;
	pchar *Text_stab_order;
	char str[10000];

	//level = Set_sizes[0];

	cout << "level=" << level << endl;

	Text_level = NEW_pchar(nb_orbits);
	Text_node = NEW_pchar(nb_orbits);
	Text_orbit_reps = NEW_pchar(nb_orbits);
	Text_stab_order = NEW_pchar(nb_orbits);

	for (i = 0; i < nb_orbits; i++) {
		sprintf(str, "%d", level);
		Text_level[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_level[i], str);

		sprintf(str, "%d", i);
		Text_node[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_node[i], str);

		int_vec_print_to_str(str, ODF->sets[i], level);
		Text_orbit_reps[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_orbit_reps[i], str);
		
		Text_stab_order[i] = NEW_char(strlen(ODF->Ago_ascii[i]) + 1);
		strcpy(Text_stab_order[i], ODF->Ago_ascii[i]);
		
		}

	spreadsheet *Sp;
	
	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_empty_table(nb_orbits + 1, 5);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const char **) Text_level, "Level");
	Sp->fill_column_with_text(2, (const char **) Text_node, "Node");
	Sp->fill_column_with_text(3, (const char **) Text_orbit_reps, "Orbit rep");
	Sp->fill_column_with_text(4, (const char **) Text_stab_order, "Stab order");

	if (f_save) {
		cout << "before Sp->save " << file_name_save << endl;
		Sp->save(file_name_save, verbose_level);
		cout << "after Sp->save " << file_name_save << endl;
		}

	for (i = 0; i < nb_orbits; i++) {
		FREE_char(Text_level[i]);
		}
	FREE_pchar(Text_level);
	for (i = 0; i < nb_orbits; i++) {
		FREE_char(Text_node[i]);
		}
	FREE_pchar(Text_node);
	for (i = 0; i < nb_orbits; i++) {
		FREE_char(Text_orbit_reps[i]);
		}
	FREE_pchar(Text_orbit_reps);
	for (i = 0; i < nb_orbits; i++) {
		FREE_char(Text_stab_order[i]);
		}

}

