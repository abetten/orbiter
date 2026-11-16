// orbiter_data_file.cpp
// 
// Anton Betten
// July 30, 2018
//
//
// 
// pulled out of ovoid: Jul 30, 2018
//

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {


orbiter_data_file::orbiter_data_file()
{
	Record_birth();
	nb_cases = 0;

	Orbit_reps = NULL;

	//sets = NULL;
	//set_sizes = NULL;

	Ago_text = NULL;
	Aut_text = NULL;
	Casenumbers = 0;
}


orbiter_data_file::~orbiter_data_file()
{
	Record_death();
	//int i;
	
	delete [] Ago_text;
	delete [] Aut_text;

#if 0
	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
	}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);
#endif

	FREE_OBJECT(Orbit_reps);

	//FREE_int(set_sizes);
	FREE_int(Casenumbers);
}

void orbiter_data_file::load(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char **data;
	int i;
	file_io Fio;


	if (f_v) {
		cout << "orbiter_data_file::load "
				"loading file " << fname << endl;
	}


	if (!Fio.try_to_read_file(fname,
			nb_cases, data, verbose_level)) {
		cout << "orbiter_data_file::load couldn't read file " 
			<< fname << endl;
		exit(1);
	}
	
	if (f_v) {
		cout << "file read containing " << nb_cases << " cases" << endl;
	}

#if 1
	char **Ago_ascii;
	char **Aut_ascii;

	if (f_v) {
		cout << "read_and_parse_data_file: "
				"parsing sets" << endl;
	}

	long int **sets;
	int *set_sizes;

	
	Fio.parse_sets(
			nb_cases, data,
		false /*f_casenumbers */,
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 2);

	if (f_v) {
		cout << "read_and_parse_data_file: "
				"parsing sets done" << endl;
	}


	int underlying_set_size = INT_MAX;

	Orbit_reps = NEW_OBJECT(data_structures::set_of_sets);

	Orbit_reps->init_simple(underlying_set_size,
			nb_cases, verbose_level);


	for (i = 0; i < nb_cases; i++) {
		Orbit_reps->Sets[i] = sets[i];
		Orbit_reps->Set_size[i] = set_sizes[i];
	}

	FREE_plint(sets);
	FREE_int(set_sizes);

	Ago_text = new std::string [nb_cases];
	Aut_text = new std::string [nb_cases];

	for (i = 0; i < nb_cases; i++) {
		Ago_text[i] = Ago_ascii[i];
	}

	for (i = 0; i < nb_cases; i++) {
		Aut_text[i] = Aut_ascii[i];
	}

	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
	}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);

	for (i = 0; i < nb_cases; i++) {
		FREE_char(data[i]);
	}
	FREE_pchar(data);
#endif
	
	if (f_v) {
		cout << "orbiter_data_file::load "
				"done nb_cases = " << nb_cases << endl;
	}
}

void orbiter_data_file::load_orbiter_file_csv_format(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "orbiter_data_file::load_orbiter_file_csv_format "
				"loading file " << fname << endl;
	}

	file_io Fio;
	std::string *col_label;
	std::string *Table;
	int m, n;


	if (f_v) {
		cout << "orbiter_data_file::load_orbiter_file_csv_format "
				"before Fio.Csv_file_support->read_table_of_strings_and_drop_quotes " << endl;
	}
	Fio.Csv_file_support->read_table_of_strings_and_drop_quotes(
			fname, col_label,
			Table, m, n,
			verbose_level);
	if (f_v) {
		cout << "orbiter_data_file::load_orbiter_file_csv_format "
				"after Fio.Csv_file_support->read_table_of_strings_and_drop_quotes " << endl;
	}

	other::data_structures::string_tools String;

	std::string str_to_find;
	int idx_line, idx_node, idx_level, idx_orbit_idx, idx_ago, idx_rep, idx_orbit_length, idx_sv_length, idx_gens;

	str_to_find = "Line";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_line);

	str_to_find = "Node";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_node);

	str_to_find = "Level";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_level);

	str_to_find = "OrbitIdx";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_orbit_idx);

	str_to_find = "OrbitRep";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_rep);

	str_to_find = "StabOrder";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_ago);

	str_to_find = "OrbitLength";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_orbit_length);

	str_to_find = "SV_length";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_sv_length);

	str_to_find = "StabGensCoded";
	String.find_string_in_array(
			col_label, n /* nb_strings */,
			str_to_find, idx_gens);

	//Row,Line,Node,Level,OrbitIdx,OrbitRep,StabOrder,OrbitLength,SV_length,NA,StabGensCoded

	nb_cases = m;
	int i;


	// scan the sets. They can be of different sizes:


	int underlying_set_size = INT_MAX;

	Orbit_reps = NEW_OBJECT(data_structures::set_of_sets);

	Orbit_reps->init_simple(underlying_set_size,
			nb_cases, verbose_level);




	//sets = NEW_plint(nb_cases);
	//set_sizes = NEW_int(nb_cases);

	for (i = 0; i < nb_cases; i++) {

		int sz;

		Lint_vec_scan(Table[i * n + idx_rep], Orbit_reps->Sets[i], sz);

		Orbit_reps->Set_size[i] = sz;
	}

	//sets = NULL;
	//set_sizes = NULL;

	Ago_text = new std::string [nb_cases];
	Aut_text = new std::string [nb_cases];

	for (i = 0; i < nb_cases; i++) {
		Ago_text[i] = Table[i * n + idx_ago];
	}
	for (i = 0; i < nb_cases; i++) {
		Aut_text[i] = Table[i * n + idx_gens];
	}

	Casenumbers = 0;

	delete [] Table;
	delete [] col_label;

	if (f_v) {
		cout << "orbiter_data_file::load_orbiter_file_csv_format "
				"done" << endl;
	}
}

int orbiter_data_file::search(
		long int *search_set, int search_set_sz,
		int &idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "orbiter_data_file::search" << endl;
	}


	int ret;

	ret = Orbit_reps->search(
			search_set, search_set_sz,
			idx,
			verbose_level - 1);

	if (f_v) {
		cout << "orbiter_data_file::search done" << endl;
	}
	return ret;
}

}}}}


