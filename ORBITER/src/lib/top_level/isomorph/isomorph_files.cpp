// isomorph_files.cpp
// 
// Anton Betten
// started 2007
//
// moved here from global.cpp: Nov 1, 2009
// renamed isomorph_files.cpp from global_solution.cpp
//
// 
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {

#define MY_BUFSIZE 1000000

void isomorph::init_solutions(int **Solutions, int *Nb_sol,
	int verbose_level)
// Solutions[nb_starter], Nb_sol[nb_starter]
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	sorting Sorting;

	if (f_v) {
		cout << "isomorph::init_solutions "
				"nb_starter = " << nb_starter << endl;
		}
	solution_first = NEW_int(nb_starter + 1);
	solution_len = NEW_int(nb_starter);
	N = 0;
	for (i = 0; i < nb_starter; i++) {
		solution_first[i] = 0;
		solution_len[i] = Nb_sol[i];
		N += solution_len[i];
		}
	if (f_v) {
		cout << "isomorph::init_solutions N = " << N << endl;
		}
	solution_first[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		solution_first[i + 1] = solution_first[i] + solution_len[i];
		}
	if (solution_first[nb_starter] != N) {
		cout << "isomorph::init_solutions "
				"solution_first[nb_starter] != N" << endl;
		exit(1);
		}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "isomorph::init_solutions "
				"after init_starter_number" << endl;
		}

	write_solution_first_and_len();
	if (f_v) {
		cout << "isomorph::init_solutions "
				"after write_solution_first_and_len" << endl;
		}


	setup_and_create_solution_database(0/*verbose_level - 1*/);
	
	int h;
	int no = 0;
	int print_mod = 1000;

	id_to_datref = NEW_int(N);
	id_to_hash = NEW_int(N);
	hash_vs_id_hash = NEW_int(N);
	hash_vs_id_id = NEW_int(N);

	if (f_v) {
		cout << "isomorph::init_solutions "
				"before add_solutions_to_database" << endl;
		}

	for (h = 0; h < nb_starter; h++) {
		if (solution_len[h]) {
			add_solutions_to_database(Solutions[h], 
				h, solution_len[h], N, print_mod, no, 
				verbose_level);
			}
		}

	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "isomorph::init_solutions "
				"written hash and datref file" << endl;
		cout << "isomorph::init_solutions "
				"sorting hash_vs_id_hash" << endl;
		}
	{
		classify C;

		C.init(hash_vs_id_hash, N, TRUE, 0);
		cout << "isomorph::init_solutions "
				"Classification of hash values:" << endl;
		C.print(FALSE /*f_backwards*/);
	}
	Sorting.int_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "isomorph::init_solutions "
				"after sorting hash_vs_id_hash" << endl;
		}

	close_solution_database(0 /*verbose_level - 1*/);



	if (f_v) {
		cout << "isomorph::init_solutions done" << endl;
		}
}

void isomorph::count_solutions_from_clique_finder_case_by_case(
		int nb_files, long int *list_of_cases, const char **fname,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h;
	file_io Fio;

	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder_case_by_case "
				"nb_starter = " << nb_starter << " nb_files=" << nb_files << endl;
		}
	solution_first = NEW_int(nb_starter + 1);
	solution_len = NEW_int(nb_starter);

	int_vec_zero(solution_len, nb_starter);
	N = 0;
	for (i = 0; i < nb_files; i++) {
		int nb_solutions;
		
		Fio.count_number_of_solutions_in_file(fname[i],
			nb_solutions, 
			verbose_level - 2);

		if (f_vv) {
			cout << "isomorph::count_solutions_from_clique_finder_"
					"case_by_case file " << i << " / " << nb_files << " = "
					<< fname[i] << " read, nb_solutions="
					<< nb_solutions << endl;
			}

		h = list_of_cases[i];
		solution_len[h] = nb_solutions;

		N += nb_solutions;
		}
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder_"
				"case_by_case done counting solutions, "
				"total number of solutions = " << N << endl;
		cout << "h : solution_len[h]" << endl;
		for (h = 0; h < nb_starter; h++) {
			cout << h << " : " << solution_len[h] << endl;
			}
		}
	solution_first[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		solution_first[i + 1] = solution_first[i] + solution_len[i];
		}
	if (solution_first[nb_starter] != N) {
		cout << "isomorph::count_solutions_from_clique_finder_"
				"case_by_case solution_first[nb_starter] != N" << endl;
		exit(1);
		}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder_"
				"case_by_case after init_starter_number" << endl;
		}

	write_solution_first_and_len();
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder_"
				"case_by_case after write_solution_first_and_len" << endl;
		}
}



void isomorph::count_solutions_from_clique_finder(
		int nb_files, const char **fname, int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h, c, n;
	file_io Fio;

	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder "
				"nb_starter = " << nb_starter << " nb_files="
				<< nb_files << endl;
		}
	solution_first = NEW_int(nb_starter + 1);
	solution_len = NEW_int(nb_starter);
	for (i = 0; i < nb_starter; i++) {
		solution_first[i] = 0;
		solution_len[i] = 0;
		}
	N = 0;
	for (i = 0; i < nb_files; i++) {
		int *nb_solutions;
		int *case_nb;
		int nb_cases;
		
		Fio.count_number_of_solutions_in_file_by_case(fname[i],
			nb_solutions, case_nb, nb_cases, 
			verbose_level - 2);

		if (f_vv) {
			cout << "isomorph::count_solutions_from_clique_finder "
					"file " << i << " / " << nb_files << " = " << fname[i]
					<< " read, nb_cases=" << nb_cases << endl;
			}

		for (h = 0; h < nb_cases; h++) {
			c = case_nb[h];
			n = nb_solutions[h];
			solution_len[c] = n;
			N += n;
			}
		FREE_int(nb_solutions);
		FREE_int(case_nb);
		}
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder "
				"done counting solutions, total number of "
				"solutions = " << N << endl;
		cout << "h : solution_len[h]" << endl;
		for (h = 0; h < nb_starter; h++) {
			cout << h << " : " << solution_len[h] << endl;
			}
		}
	solution_first[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		solution_first[i + 1] = solution_first[i] + solution_len[i];
		}
	if (solution_first[nb_starter] != N) {
		cout << "isomorph::count_solutions_from_clique_finder "
				"solution_first[nb_starter] != N" << endl;
		exit(1);
		}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder "
				"after init_starter_number" << endl;
		}

	write_solution_first_and_len();
	if (f_v) {
		cout << "isomorph::count_solutions_from_clique_finder "
				"after write_solution_first_and_len" << endl;
		}
}



void isomorph::read_solutions_from_clique_finder_case_by_case(
		int nb_files, long int *list_of_cases, const char **fname,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
// Called after count_solutions_from_clique_finder
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	sorting Sorting;
	file_io Fio;

	int i, no = 0;
	int print_mod = 1000;

	
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder_"
				"case_by_case nb_files=" << nb_files
				<< " N=" << N << endl;
		}
	

	setup_and_create_solution_database(0/*verbose_level - 1*/);

	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder_"
				"case_by_case after setup_and_create_solution_"
				"database" << endl;
		}

	id_to_datref = NEW_int(N);
	id_to_hash = NEW_int(N);
	hash_vs_id_hash = NEW_int(N);
	hash_vs_id_id = NEW_int(N);

	for (i = 0; i < nb_files; i++) {

		if (f_vv) {
			cout << "isomorph::read_solutions_from_clique_finder_"
					"case_by_case, file " << i << " / " << nb_files
					<< " which is " << fname[i] << endl;
			}
		int nb_solutions;
		int *Solutions;
		
		Fio.read_solutions_from_file(fname[i],
			nb_solutions, 
			Solutions, size /* solution_size */, 
			verbose_level - 2);

		if (f_vv) {
			cout << "isomorph::read_solutions_from_clique_finder_"
					"case_by_case file " << fname[i] << " number of "
							"solutions read: " << nb_solutions << endl;
			}

		add_solutions_to_database(Solutions, 
			list_of_cases[i], nb_solutions, N, print_mod, no, 
			verbose_level);


		FREE_int(Solutions);
		}

	
	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder_case_by_case "
				"written hash and datref file" << endl;
		cout << "isomorph::read_solutions_from_clique_finder_case_by_case "
				"sorting hash_vs_id_hash" << endl;
		}
	{
		classify C;

		C.init(hash_vs_id_hash, N, TRUE, 0);
		cout << "isomorph::read_solutions_from_clique_finder_case_by_case "
				"Classification of hash values:" << endl;
		C.print(FALSE /*f_backwards*/);
	}
	Sorting.int_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder_case_by_case "
				"after sorting hash_vs_id_hash" << endl;
		}

	close_solution_database(0 /*verbose_level - 1*/);


	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder_case_by_case "
				"done" << endl;
		}
}


void isomorph::read_solutions_from_clique_finder(
		int nb_files, const char **fname,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
// Called after count_solutions_from_clique_finder
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int i, no = 0;
	//int *data;
	int print_mod = 1000;
	sorting Sorting;
	file_io Fio;

	
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder "
				"nb_files=" << nb_files << " N=" << N << endl;
		}
	

	setup_and_create_solution_database(0/*verbose_level - 1*/);

	//data = NEW_int(size + 1);

	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder "
				"after setup_and_create_solution_database" << endl;
		}

	id_to_datref = NEW_int(N);
	id_to_hash = NEW_int(N);
	hash_vs_id_hash = NEW_int(N);
	hash_vs_id_id = NEW_int(N);

	for (i = 0; i < nb_files; i++) {

		if (f_vv) {
			cout << "isomorph::read_solutions_from_clique_finder "
					"file " << fname[i] << endl;
			}
		int *nb_solutions;
		int *case_nb;
		int nb_cases;
		int **Solutions;
		int the_case, h; //, u, v;
		int nb_solutions_total;
		char fname_summary[1000];
		char extension[1000];
		
		strcpy(fname_summary, fname[i]);
		chop_off_extension_if_present(fname_summary, extension);
		strcat(fname_summary, "_summary.csv");

		Fio.count_number_of_solutions_in_file_by_case(fname[i],
			nb_solutions, case_nb, nb_cases, 
			verbose_level - 2);

		
		Fio.int_vecs_write_csv(case_nb, nb_solutions, nb_cases,
			fname_summary, "Case_nb", "Nb_sol");
		
		nb_solutions_total = 0;
		for (h = 0; h < nb_cases; h++) {
			nb_solutions_total += nb_solutions[h];
			}

		Fio.read_solutions_from_file_by_case(fname[i],
			nb_solutions, case_nb, nb_cases, 
			Solutions, size /* solution_size */, 
			verbose_level - 2);

		if (f_vv) {
			cout << "isomorph::read_solutions_from_clique_finder "
					"file " << fname[i] << " read solutions" << endl;
			}

		for (h = 0; h < nb_cases; h++) {
			the_case = case_nb[h];

			add_solutions_to_database(Solutions[h], 
				the_case, nb_solutions[h], nb_solutions_total,
				print_mod, no,
				verbose_level);

			}

		FREE_int(nb_solutions);
		FREE_int(case_nb);
		for (h = 0; h < nb_cases; h++) {
			FREE_int(Solutions[h]);
			}
		FREE_pint(Solutions);
		if (f_vv) {
			cout << "isomorph::read_solutions_from_clique_finder "
					"file " << fname[i] << " done" << endl;
			}
		}

	
	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder "
				"written hash and datref file" << endl;
		cout << "isomorph::read_solutions_from_clique_finder "
				"sorting hash_vs_id_hash" << endl;
		}
	{
		classify C;

		C.init(hash_vs_id_hash, N, TRUE, 0);
		cout << "isomorph::read_solutions_from_clique_finder "
				"Classification of hash values:" << endl;
		C.print(FALSE /*f_backwards*/);
	}
	Sorting.int_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder "
				"after sorting hash_vs_id_hash" << endl;
		}

	close_solution_database(0 /*verbose_level - 1*/);

	//FREE_int(data);

	if (f_v) {
		cout << "isomorph::read_solutions_from_clique_finder done" << endl;
		}
}

void isomorph::add_solutions_to_database(int *Solutions, 
	int the_case, int nb_solutions, int nb_solutions_total,
	int print_mod, int &no,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int u, v;
	int *data;
	
	if (f_v) {
		cout << "isomorph::add_solutions_to_database "
				"case " << the_case << endl;
		}
	data = NEW_int(size + 1);
	for (u = 0; u < nb_solutions; u++) {

		uint_4 datref;
		int hs, id;
				
		data[0] = the_case;
		for (v = 0; v < size; v++) {
			data[1 + v] = Solutions[u * size + v];
			}
		id = solution_first[data[0]] + u;

		hs = int_vec_hash_after_sorting(data + 1, size);
		if (f_vvv) {
			cout << "isomorph::add_solutions_to_database "
					"case " << the_case << " u=" << u << " id=" << id
					<< " hs=" << hs << " no=" << no << endl;
			}

			
		add_solution_to_database(data, 
			u, id, no, nb_solutions_total, hs, datref,
			print_mod, verbose_level - 2);
			// in isomorph_database.cpp
			
		id_to_datref[id] = datref;
		id_to_hash[id] = hs;
		hash_vs_id_hash[id] = hs;
		hash_vs_id_id[id] = id;
				
		no++;
		}
	FREE_int(data);
	if (f_v) {
		cout << "isomorph::add_solutions_to_database "
				"case " << the_case << " done, added "
				<< nb_solutions << " solutions; "
						"n e w database length is " << no << endl;
		}
}


void isomorph::build_up_database(int nb_files,
	const char **fname,
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	int i, nb_total = 0, j, a, nb = 0, prev = 0, id = 0, h;
	char *p_buf;
	int data[1000];
	char buf[MY_BUFSIZE];
	int print_mod = 1000;
	uint_4 datref;
	int nb_fail = 0;
	sorting Sorting;
	file_io Fio;

	
	if (f_v) {
		cout << "isomorph::build_up_database "
				"nb_files=" << nb_files << " N=" << N << endl;
		}
	

	setup_and_create_solution_database(verbose_level - 1);


	if (f_v) {
		cout << "isomorph::build_up_database "
				"after setup_and_create_solution_database" << endl;
		}

	id_to_datref = NEW_int(N);
	id_to_hash = NEW_int(N);
	hash_vs_id_hash = NEW_int(N);
	hash_vs_id_id = NEW_int(N);

	for (i = 0; i < nb_files; i++) {

		ifstream f(fname[i]);
		if (f_v) {
			cout << "isomorph::build_up_database "
					"reading file " << fname[i] << " of size "
					<< Fio.file_size(fname[i]) << endl;
			}

		while (TRUE) {

			if (f.eof()) {
				break;
				}

#if 0
			{
			string S;
			int l;
			getline(f, S);
			l = S.length();
			if (f_vvv) {
				cout << "isomorph::build_up_database "
						"read line of length " << l << " : " << S << endl;
				}
			for (j = 0; j < l; j++) {
				buf[j] = S[j];
				}
			buf[l] = 0;
			}
#else
			{
			f.getline(buf, MY_BUFSIZE, '\n');
			}
#endif
			if (f_vvv) {
				cout << "isomorph::build_up_database "
						"read: " << buf << endl;
				}

			p_buf = buf;

			
			s_scan_int(&p_buf, &a);
			
			data[0] = a; // starter number
			
			if (data[0] != prev) {
				prev = data[0];
				nb = 0;
				}
			if (a == -1) {
				break;
				}

			for (j = 0; j < size; j++) {
				s_scan_int(&p_buf, &a);
				data[j + 1] = a;
				}


			if (f_has_final_test_function) {
				if (!(*final_test_function)(data + 1, size,
						final_test_data, verbose_level - 1)) {
					if (f_vvv) {
						cout << "isomorph::build_up_database "
								"solution fails the final test, "
								"skipping" << endl;
						}
					nb_fail++;
					continue;
					}
				else {
					cout << nb_total << " : " << data[0] << " : ";
					int_vec_print(cout, data + 1, size);
					cout << endl;
					}
				}

			id = solution_first[data[0]] + nb;
			

			h = int_vec_hash_after_sorting(data + 1, size);
			
			add_solution_to_database(data, 
				nb, id, nb_total, N, h, datref, print_mod,
				verbose_level - 3);
			
			id_to_datref[id] = datref;
			id_to_hash[id] = h;
			hash_vs_id_hash[id] = h;
			hash_vs_id_id[id] = id;
			
			nb_total++; // number of solutions total
			nb++; // number of solutions within the starter case
			} // end while
		if (f_v) {
			cout << "isomorph::build_up_database "
					"finished reading file " << fname[i]
					<< " nb=" << nb << " nb_total=" << nb_total << endl;
			}
		}	// next i
	

	if (f_v) {
		cout << "isomorph::build_up_database finished "
				"number of solutions total = " << nb_total
				<< " nb_fail = " << nb_fail << endl;
		}


	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "isomorph::build_up_database "
				"written hash and datref file" << endl;
		cout << "isomorph::build_up_database "
				"sorting hash_vs_id_hash" << endl;
		}
	{
		classify C;

		C.init(hash_vs_id_hash, N, TRUE, 0);
		cout << "isomorph::build_up_database "
				"Classification of hash values:" << endl;
		C.print(FALSE /*f_backwards*/);
	}
	Sorting.int_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "isomorph::build_up_database "
				"after sorting hash_vs_id_hash" << endl;
		}

	close_solution_database(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::build_up_database done" << endl;
		}
}


void isomorph::init_cases_from_file_modulus_and_build_up_database(
	int modulus, int level, 
	int f_collated, int base_split, 
	int f_get_statistics, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char **fname;
	int i;
	char file_name[1000];
	
	if (f_v) {
		cout << "isomorph::init_cases_from_file_modulus_"
				"and_build_up_database modulus = " << modulus << endl;
		}
	fname = NEW_pchar(modulus);
	if (f_v) {
		cout << "creating file names" << endl;
		}
	for (i = 0; i < modulus; i++) {
		if (f_collated) {
			sprintf(file_name, "collated_%s_%d_%d_%d_%d.txt",
					prefix, level, base_split, i, modulus);
			}
		else {
			sprintf(file_name, "extend_%s_%d_%d_%d.txt",
					prefix, level, i, modulus);
			}
		//sprintf(file_name, "extend_BLT_41_lvl_%d_%d_42_%d_1024.txt",
		//level, level, i);
		fname[i] = NEW_char(strlen(file_name) + 1);
		strcpy(fname[i], file_name);
		}
	if (f_vv) {
		for (i = 0; i < modulus; i++) {
			cout << i << " : " << fname[i] << endl;
			}
		}
	count_solutions(modulus,
		(const char **) fname,
		f_get_statistics,
		f_has_final_test_function,
		final_test_function,
		final_test_data,
		verbose_level);

	//registry_dump_sorted();
	//registry_dump_sorted_by_size();

	// now we know N, the number of solutions
	
	build_up_database(modulus, (const char **) fname, 
		f_has_final_test_function,
		final_test_function,
		final_test_data,
		verbose_level);
	
	if (f_v) {
		cout << "deleting file names" << endl;
		}
	for (i = 0; i < modulus; i++) {
		FREE_char(fname[i]);
		}
	FREE_pchar(fname);
}

void isomorph::init_cases_from_file_mixed_modulus_and_build_up_database(
	int nb_Mod, int *Mod_r, int *Mod_split, int *Mod_base_split, 
	int level, int f_get_statistics, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char **fname;
	int i, r, s, bs, nb_files, j, k, h, u;
	char file_name[1000];
	
	if (f_v) {
		cout << "isomorph::init_cases_from_file_mixed_"
				"modulus_and_build_up_database" << endl;
		cout << "r : split : base_split" << endl;
		for (i = 0; i < nb_Mod; i++) {
			cout << Mod_r[i] << " : " << Mod_split[i] << " : "
					<< Mod_base_split[i] << endl;
			}
		cout << "computing the number of files" << endl;
		}
	nb_files = 0;
	for (i = 0; i < nb_Mod; i++) {
		r = Mod_r[i];
		s = Mod_split[i];
		bs = Mod_base_split[i];
		nb_files += bs / s;
		}
	if (f_v) {
		cout << "number of files is " << nb_files << endl;
		}
	
	fname = NEW_pchar(nb_files);
	if (f_v) {
		cout << "creating file names" << endl;
		}
	j = 0;
	for (i = 0; i < nb_Mod; i++) {
		r = Mod_r[i];
		s = Mod_split[i];
		bs = Mod_base_split[i];
		k = bs / s;
		for (h = 0; h < k; h++) {
			u = h * s + r;
			sprintf(file_name, "extend_%s_%d_%d_%d.txt",
					prefix, level, u, bs);
			//sprintf(file_name, "extend_BLT_41_lvl_%d_%d_42_%d_1024.txt",
			// level, level, i);
			fname[j] = NEW_char(strlen(file_name) + 1);
			strcpy(fname[j], file_name);
			j++;
			}
		}
	if (j != nb_files) {
		cout << "isomorph::init_cases_from_file_mixed_modulus_"
				"and_build_up_database j != nb_files" << endl;
		exit(1);
		}
	if (f_vv) {
		for (i = 0; i < nb_files; i++) {
			cout << i << " : " << fname[i] << endl;
			}
		}
	
	count_solutions(nb_files,
		(const char **) fname,
		f_get_statistics,
		f_has_final_test_function,
		final_test_function,
		final_test_data,
		verbose_level);
	
	// now we know N, the number of solutions
	
	
	build_up_database(nb_files, (const char **) fname, 
		f_has_final_test_function,
		final_test_function,
		final_test_data,
		verbose_level);
	
	if (f_v) {
		cout << "deleting file names" << endl;
		}
	for (i = 0; i < nb_files; i++) {
		FREE_char(fname[i]);
		}
	FREE_pchar(fname);
}

void isomorph::count_solutions(int nb_files, const char **fname, 
	int f_get_statistics, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int total_days, total_hours, total_minutes;
	
	if (nb_starter == 0) {
		cout << "isomorph::count_solutions "
				"nb_starter == 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "isomorph::count_solutions "
				"nb_starter = " << nb_starter << endl;
		}
	solution_first = NEW_int(nb_starter + 1);
	solution_len = NEW_int(nb_starter);
	for (i = 0; i < nb_starter; i++) {
		solution_first[i] = 0;
		solution_len[i] = 0;
		}
	stats_nb_backtrack = NEW_int(nb_starter);
	stats_nb_backtrack_decision = NEW_int(nb_starter);
	stats_graph_size = NEW_int(nb_starter);
	stats_time = NEW_int(nb_starter);
	
	for (i = 0; i < nb_starter; i++) {
		stats_nb_backtrack[i] = -1;
		stats_nb_backtrack_decision[i] = -1;
		stats_graph_size[i] = -1;
		stats_time[i] = -1;
		}
	
	count_solutions2(nb_files, fname, 
		total_days, total_hours, total_minutes, 
		f_has_final_test_function,
		final_test_function,
		final_test_data,
		verbose_level);
	if (f_v) {
		cout << "isomorph::count_solutions "
				"after count_solutions2" << endl;
		cout << "case_len: ";
		int_vec_print(cout, solution_len, nb_starter);
		cout << endl;
		}
	cout << "total computing time for the search : ";
	cout << total_days << "-" << total_hours << ":"
			<< total_minutes << ":" << 0;
	cout << endl;

	cout << "case : nb of solutions" << endl;
	solution_first[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		solution_first[i + 1] = solution_first[i] + solution_len[i];
		if (solution_len[i]) {
			cout << i << " : " << solution_len[i] << endl;
			}
		}
	N = solution_first[nb_starter];
	if (f_v) {
		cout << "isomorph::count_solutions N=" << N << endl;
		}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "isomorph::count_solutions "
				"after init_starter_number" << endl;
		}

	write_solution_first_and_len();
	if (f_v) {
		cout << "isomorph::count_solutions "
				"after write_solution_first_and_len" << endl;
		}
	
	if (f_get_statistics) {
		get_statistics(nb_files, fname, verbose_level);
		write_statistics();
		evaluate_statistics(verbose_level);
		}
}

void isomorph::get_statistics(int nb_files,
		const char **fname, int verbose_level)
{
	int i, the_case, nb_sol, nb_backtrack, nb_backtrack_decision;
	int nb_points, dt[5], dt_total;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	char fname_summary[1000];
	file_io Fio;

	if (f_v) {
		cout << "get_statistics: reading "
				<< nb_files << " files" << endl;
		}
#if 0
	for (i = 0; i < nb_files; i++) {
		cout << fname[i] << endl;
		}
#endif

	
	for (i = 0; i < nb_files; i++) {


		strcpy(fname_summary, fname[i]);
		if (strcmp(fname_summary + strlen(fname_summary) - 4, ".txt")) {
			cout << "get_statistics: file name does not end in .txt" << endl;
			return;
			}
		strcpy(fname_summary + strlen(fname_summary) - 4, ".summary");

		ifstream fp(fname_summary);
		
		if (f_v) {
			cout << "file " << i << " / " << nb_files
					<< ", reading file " << fname_summary
					<< " of size " << Fio.file_size(fname[i]) << endl;
			}
		if (Fio.file_size(fname_summary) <= 0) {
			cout << "problems reading file " << fname_summary << endl;
			return;
			}
		while (TRUE) {
			fp >> the_case;
			if (the_case == -1)
				break;
			fp >> nb_sol;
			fp >> nb_backtrack;
			fp >> nb_backtrack_decision;
			fp >> nb_points;
			fp >> dt[0];
			fp >> dt[1];
			fp >> dt[2];
			fp >> dt[3];
			fp >> dt[4];
			fp >> dt_total;
			stats_nb_backtrack[the_case] = nb_backtrack;
			stats_nb_backtrack_decision[the_case] = nb_backtrack_decision;
			stats_graph_size[the_case] = nb_points;
			stats_time[the_case] = dt_total;
			}
		} // next i
	
}

void isomorph::write_statistics()
{
	file_io Fio;

	{
	ofstream f(fname_statistics);
	int i;
	
	f << nb_starter << endl;
	for (i = 0; i < nb_starter; i++) {
		f << setw(7) << i << " " 
			<< setw(4) << stats_nb_backtrack[i] 
			<< setw(4) << stats_nb_backtrack_decision[i] 
			<< setw(4) << stats_graph_size[i] 
			<< setw(4) << stats_time[i] << endl;
		}
	f << "-1" << endl;
	}
	cout << "written file " << fname_statistics << " of size "
			<< Fio.file_size(fname_statistics) << endl;
}

void isomorph::evaluate_statistics(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_backtrack_max;
	int nb_backtrack_min;
	int graph_size_max;
	int graph_size_min;
	int time_max;
	int time_min;
	longinteger_object a, b, c, a1, b1, c1, d, n, q1, q2, q3, r1, r2, r3;
	longinteger_domain D;
	
	nb_backtrack_max = nb_backtrack_min = stats_nb_backtrack[0];
	graph_size_max = graph_size_min = stats_graph_size[0];
	time_max = time_min = stats_time[0];
	
	a.create(0);
	b.create(0);
	c.create(0);
	for (i = 0; i < nb_starter; i++) {
		nb_backtrack_max =
				MAXIMUM(nb_backtrack_max, stats_nb_backtrack[i]);
		nb_backtrack_min =
				MINIMUM(nb_backtrack_min, stats_nb_backtrack[i]);
		graph_size_max =
				MAXIMUM(graph_size_max, stats_graph_size[i]);
		graph_size_min =
				MINIMUM(graph_size_min, stats_graph_size[i]);
		time_max =
				MAXIMUM(time_max, stats_time[i]);
		time_min =
				MINIMUM(time_min, stats_time[i]);
		a1.create(stats_nb_backtrack[i]);
		b1.create(stats_graph_size[i]);
		c1.create(stats_time[i]);
		D.add(a, a1, d);
		d.assign_to(a);
		D.add(b, b1, d);
		d.assign_to(b);
		D.add(c, c1, d);
		d.assign_to(c);
		}
	if (f_v) {
		cout << "evaluate_statistics" << endl;
		cout << "nb_backtrack_max=" << nb_backtrack_max << endl;
		cout << "nb_backtrack_min=" << nb_backtrack_min << endl;
		cout << "graph_size_max=" << graph_size_max << endl;
		cout << "graph_size_min=" << graph_size_min << endl;
		cout << "time_max=" << time_max << endl;
		cout << "time_min=" << time_min << endl;
		cout << "sum nb_backtrack = " << a << endl;
		cout << "sum graph_size = " << b << endl;
		cout << "sum time = " << c << endl;
		n.create(nb_starter);
		D.integral_division(a, n, q1, r1, 0);
		D.integral_division(b, n, q2, r2, 0);
		D.integral_division(c, n, q3, r3, 0);
		cout << "average nb_backtrack = " << q1 << endl;
		cout << "average graph_size = " << q2 << endl;
		cout << "average time = " << q3 << endl;
		}
}




void isomorph::count_solutions2(int nb_files, const char **fname, 
	int &total_days, int &total_hours, int &total_minutes, 
	int f_has_final_test_function, 
	int (*final_test_function)(int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data, 
	int verbose_level)
// also fills the array case_len[] with the
// number of solutions per starter
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, no, l, j, a, nb, prev;
	char *p_buf;
	char buf[MY_BUFSIZE];
	int data[1000];
	int *nb_sol_per_file;
	Vector v;
	char str[1000];
	int nb_fail = 0;
	file_io Fio;

	if (f_v) {
		cout << "count_solutions2: reading " << nb_files << " files" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		if (f_vv) {
			for (i = 0; i < nb_files; i++) {
				cout << fname[i] << endl;
				}
			}
		}
	total_days = 0;
	total_hours = 0;
	total_minutes = 0;

	nb_sol_per_file = NEW_int(nb_files);
	
	for (i = 0; i < nb_files; i++) {

		nb_sol_per_file[i] = 0;

		ifstream f(fname[i]);
		cout << "reading file " << fname[i] << " of size "
				<< Fio.file_size(fname[i]) << endl;
		if (Fio.file_size(fname[i]) <= 0) {
			cout << "problems reading file " << fname[i] << endl;
			exit(1);
			}
		no = 0;
		nb = 0;
		prev = -1;
		while (TRUE) {

			if (f.eof()) {
				break;
				}
			{
			string S;
			getline(f, S);
			l = S.length();
			//cout << "read line of length " << l << " : " << S << endl;
			for (j = 0; j < l; j++) {
				buf[j] = S[j];
				}
			buf[l] = 0;
			}
			if (FALSE) {
				cout << "line " << no << " read: " << buf << endl;
				}
			
			p_buf = buf;

			
			s_scan_int(&p_buf, &a);
			
			data[0] = a; // case number

			
			if (data[0] == -1) {
				solution_len[prev] = nb;
				if (f_vvv) {
					cout << "isomorph::count_solutions2 "
							"solution_len[" << prev << "]=" << nb << endl;
					}
				break;
				}


			for (j = 0; j < size; j++) {
				s_scan_int(&p_buf, &a);
				data[j + 1] = a;
				}


			if (f_has_final_test_function) {
				if (!(*final_test_function)(data + 1, size,
						final_test_data, verbose_level - 1)) {
					if (f_vvv) {
						cout << "isomorph::count_solutions2 "
								"solution fails the final test, "
								"skipping" << endl;
						}
					nb_fail++;
					continue;
					}
				}


			if (data[0] != prev) {
				if (prev != -1) {
					solution_len[prev] = nb;
					if (f_vvv) {
						cout << "isomorph::count_solutions "
								"solution_len[" << prev << "]="
								<< nb << endl;
						}
					}
				prev = data[0];
				nb = 0;
				}
			nb_sol_per_file[i]++;
			no++;
			nb++;
			} // while
		
		cout << "file " << fname[i] << " has " << nb_sol_per_file[i]
			<< " lines and " << nb << " solutions and " << nb_fail
			<< " false positives" << endl;

		s_scan_token_arbitrary(&p_buf, str);
		cout << "file " << fname[i] << " time " << str << endl;
	
		} // next i
	
	FREE_int(nb_sol_per_file);
	if (f_v) {
		cout << "isomorph::count_solutions2 done" << endl;
		}
}



void isomorph::write_solution_first_and_len()
{
	ofstream f(fname_case_len);
	int i;
	
	f << N << " " << nb_starter << endl;
	for (i = 0; i < nb_starter; i++) {
		f << setw(4) << i << " " << setw(4) << solution_first[i]
			<< " " << setw(4) << solution_len[i] << endl;
		}
	f << "-1" << endl;
}

void isomorph::read_solution_first_and_len()
{
	sorting Sorting;
	file_io Fio;

	cout << "isomorph::read_solution_first_and_len "
			"reading from file "
		<< fname_case_len << " of size "
		<< Fio.file_size(fname_case_len) << endl;
	
	ifstream f(fname_case_len);
	int i, a;
	
	f >> N >> nb_starter;
	solution_first = NEW_int(nb_starter + 1);
	solution_len = NEW_int(nb_starter + 1);
	for (i = 0; i < nb_starter; i++) {
		f >> a;
		f >> solution_first[i];
		f >> solution_len[i];
		}
	solution_first[nb_starter] =
			solution_first[nb_starter - 1] +
			solution_len[nb_starter - 1];
	f >> a;
	if (a != -1) {
		cout << "problem in read_solution_first_and_len" << endl;
		exit(1);
		}
	cout << "isomorph::read_solution_first_and_len:" << endl;
	Sorting.int_vec_print_classified(cout, solution_len, nb_starter);
	cout << endl;
	cout << "isomorph::read_solution_first_and_len done" << endl; 
}

void isomorph::write_starter_nb_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;


	if (f_v) {
		cout << "isomorph::write_starter_nb_orbits" << endl;
		}

	Fio.int_vec_write_csv(flag_orbit_fst, nb_starter,
			fname_orbits_of_stabilizer_csv, "Stab_orbits");

	cout << "isomorph::write_starter_nb_orbits Written file "
			<< fname_orbits_of_stabilizer_csv << " of size "
			<< Fio.file_size(fname_orbits_of_stabilizer_csv) << endl;
	
	if (f_v) {
		cout << "isomorph::write_starter_nb_orbits done" << endl;
		}
}

void isomorph::read_starter_nb_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;


	if (f_v) {
		cout << "isomorph::read_starter_nb_orbits" << endl;
		}
	int *M;
	int m, n, i;

	cout << "isomorph::read_starter_nb_orbits Reading file "
			<< fname_orbits_of_stabilizer_csv << " of size "
			<< Fio.file_size(fname_orbits_of_stabilizer_csv) << endl;

	Fio.int_matrix_read_csv(fname_orbits_of_stabilizer_csv,
			M, m, n, verbose_level);
	
	if (m != nb_starter) {
		cout << "isomorph::read_starter_nb_orbits "
				"m != nb_starter" << endl;
		exit(1);
		}
	if (n != 1) {
		cout << "isomorph::read_starter_nb_orbits "
				"n != 1" << endl;
		exit(1);
		}

	flag_orbit_fst = NEW_int(nb_starter + 1);
	flag_orbit_len = NEW_int(nb_starter);
	flag_orbit_fst[0] = 0;
	for (i = 0; i < m; i++) {
		flag_orbit_len[i] = M[i];
		flag_orbit_fst[i + 1] =
				flag_orbit_fst[i] + flag_orbit_len[i];
		}
	
	FREE_int(M);
	
	if (f_v) {
		cout << "isomorph::read_starter_nb_orbits done" << endl;
		}
}


void isomorph::write_hash_and_datref_file(int verbose_level)
// Writes the file 'fname_hash_and_datref'
// containing id_to_hash[] and id_to_datref[]
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "isomorph::write_hash_and_datref_file" << endl;
		}
	{
	ofstream f(fname_hash_and_datref);
	int i;
	
	f << N << endl;
	for (i = 0; i < N; i++) {
		f << setw(3) << i << " " 
			<< setw(3) << id_to_hash[i] << " " 
			<< setw(3) << id_to_datref[i] << endl;
		}
	f << -1 << endl;
	}
	if (f_v) {
		cout << "isomorph::write_hash_and_datref_file finished" << endl;
		cout << "isomorph::write_hash_and_datref_file written file "
				<< fname_hash_and_datref << " of size "
				<< Fio.file_size(fname_hash_and_datref) << endl;
		}
}

void isomorph::read_hash_and_datref_file(int verbose_level)
// Reads the file 'fname_hash_and_datref'
// containing id_to_hash[] and id_to_datref[]
// Also initializes hash_vs_id_hash and hash_vs_id_id
// Called from init_solution
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	sorting Sorting;
	
	if (f_v) {
		cout << "isomorph::read_hash_and_datref_file" << endl;
		}
	ifstream f(fname_hash_and_datref);
	int id, a, h, d, N1;
	f >> N1;
	if (N1 != N) {
		cout << "isomorph::read_hash_and_datref_file "
				"N1 != N" << endl;
		cout << "N=" << N << endl;
		cout << "N1=" << N1 << endl;
		exit(1);
		}

	id_to_datref = NEW_int(N);
	id_to_hash = NEW_int(N);
	hash_vs_id_hash = NEW_int(N);
	hash_vs_id_id = NEW_int(N);

	for (id = 0; id < N; id++) {
		f >> a >> h >> d;
		if (a != id) {
			cout << "isomorph::read_hash_and_datref_file "
					"a != id" << endl;
			exit(1);
			}
		id_to_hash[id] = h;
		id_to_datref[id] = d;
		hash_vs_id_hash[id] = h;
		hash_vs_id_id[id] = id;
		}
	f >> a;
	if (a != -1) {
		cout << "isomorph::read_hash_and_datref_file "
				"EOF marker missing" << endl;
		exit(1);
		}
	Sorting.int_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);

	if (f_vv) {
		cout << "isomorph::read_hash_and_datref_file" << endl;
		print_hash_vs_id();
	}

	if (f_v) {
		cout << "isomorph::read_hash_and_datref_file done" << endl;
		}
}

void isomorph::print_hash_vs_id()
{
	int i;

	cout << "isomorph::print_hash_vs_id" << endl;
	cout << "i : hash_vs_id_hash[i] : hash_vs_id_id[i]" << endl;
	for (i = 0; i < N; i++) {
		cout << i << " : " << hash_vs_id_hash[i]
			<< " : " << hash_vs_id_id[i] << endl;
	}
}

void isomorph::write_orbit_data(int verbose_level)
// Writes the file 'fname_staborbits'
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "write_orbit_data" << endl;
		}
	{
	ofstream f(fname_staborbits);
	int i;
	
	f << nb_orbits << " " << N << endl;
	for (i = 0; i < nb_orbits; i++) {
		f << setw(3) << i << " " 
			<< setw(3) << orbit_fst[i] << " " 
			<< setw(3) << orbit_len[i] << endl;
		}
	for (i = 0; i < N; i++) {
		f << setw(3) << i << " " 
			<< setw(3) << orbit_number[i] << " "
			<< setw(3) << orbit_perm[i] << " "
			<< setw(3) << schreier_vector[i] << " "
			<< setw(3) << schreier_prev[i] << " "
			<< endl;
		}
	f << "-1" << endl;
	}
	if (f_v) {
		cout << "write_orbit_data finished" << endl;
		cout << "written file " << fname_staborbits << " of size "
				<< Fio.file_size(fname_staborbits) << endl;
		}
}

void isomorph::read_orbit_data(int verbose_level)
// Reads from the file 'fname_staborbits'
// Reads nb_orbits, N, 
// orbit_fst[nb_orbits + 1]
// orbit_len[nb_orbits]
// orbit_number[N]
// orbit_perm[N]
// schreier_vector[N]
// schreier_prev[N]
// and computed orbit_perm_inv[N]
{
	int f_v = (verbose_level >= 1);
	combinatorics_domain Combi;
	
	ifstream f(fname_staborbits);
	int i, a;
	
	if (f_v) {
		cout << "read_orbit_data" << endl;
		}
	f >> nb_orbits >> N;
	if (f_v) {
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "N=" << N << endl;
		}
	
	orbit_fst = NEW_int(nb_orbits + 1);
	orbit_len = NEW_int(nb_orbits);
	orbit_number = NEW_int(N);
	orbit_perm = NEW_int(N);
	orbit_perm_inv = NEW_int(N);
	schreier_vector = NEW_int(N);
	schreier_prev = NEW_int(N);
	
	for (i = 0; i < nb_orbits; i++) {
		f >> a;
		f >> orbit_fst[i];
		f >> orbit_len[i];
		}
	for (i = 0; i < N; i++) {
		f >> a;
		f >> orbit_number[i];
		f >> orbit_perm[i];
		f >> schreier_vector[i];
		f >> schreier_prev[i];
		}
	orbit_fst[nb_orbits] = N;
	Combi.perm_inverse(orbit_perm, orbit_perm_inv, N);
	f >> a;
	if (a != -1) {
		cout << "problem in read_orbit_data" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "read_orbit_data finished" << endl;
		}
}

void isomorph::print_isomorphism_types(int f_select,
		int select_first, int select_len,
		int verbose_level)
// Calls print_set_function (if available)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, id, first, c;
	longinteger_object go;
	
	if (f_v) {
		cout << "isomorph::print_isomorphism_types" << endl;
		if (f_select) {
			cout << "printing " << select_first << " / "
					<< select_len << endl;
			}
		}
	cout << "we found " << Reps->count << " isomorphism types" << endl;
	cout << "i : orbit_no : id of orbit representative (solution) : "
			"prefix case number" << endl;
	for (i = 0; i < Reps->count; i++) {
		j = Reps->rep[i];
		first = orbit_fst[j];
		c = starter_number[first];
		id = orbit_perm[first];		
		cout << "isomorphism type " << i << " : " << j << " : "
				<< id << " : " << c;
		if (Reps->stab[i]) {
			Reps->stab[i]->group_order(go);
			cout << " stabilizer order " << go << endl;
			}
		else {
			cout << endl;
			}
		}

	long int data[1000];

	setup_and_open_solution_database(verbose_level - 1);

	if (!f_select) {
		select_first = 0;
		select_len = Reps->count;
		}
	for (h = 0; h < select_len; h++) {
		
		i = select_first + h;
		j = Reps->rep[i];
		id = orbit_perm[orbit_fst[j]];		
		load_solution(id, data);
		cout << "isomorphism type " << i << " : " << j << " : "
				<< id << " : ";
		lint_vec_print(cout, data, size);
		cout << endl;
#if 0
		for (j = 0; j < size; j++) {
			O->unrank_point(O->v2, 1, data[j]);
			int_vec_print(cout, O->v2, algebraic_dimension);
			if (j < size - 1) {
				cout << ", ";
				}
			cout << endl;
			}
#endif
		sims *Stab;
		
		Stab = Reps->stab[i];

		if (f_v) {
			cout << "isomorph::print_isomorphism_types computing "
					"induced action on the set (in data)" << endl;
			}
		induced_action_on_set(Stab, data, verbose_level);

		if (f_v) {
			longinteger_object go;
			
			AA->group_order(go);
			cout << "action " << AA->label << " computed, "
					"group order is " << go << endl;
			}

		schreier Orb;
		longinteger_object go;
		
		AA->compute_all_point_orbits(Orb, Stab->gens, verbose_level - 2);
		cout << "Computed all orbits on the set, found "
				<< Orb.nb_orbits << " orbits" << endl;
		cout << "orbit lengths: ";
		int_vec_print(cout, Orb.orbit_len, Orb.nb_orbits);
		cout << endl;
	
		if (print_set_function) {
			if (f_v) {
				cout << "isomorph::print_isomorphism_types "
						"calling print_set_function, "
						"iso_cnt=" << i + 1 << endl;
				}
			(*print_set_function)(this, i + 1, Stab,
					Orb, data, print_set_data, verbose_level);
			if (f_v) {
				cout << "isomorph::print_isomorphism_types "
						"after print_set_function, "
						"iso_cnt=" << i + 1 << endl;
				}
			}
		}
	close_solution_database(verbose_level - 1);
}

void isomorph::induced_action_on_set_and_kernel(
	ostream &file, action *A,
	sims *Stab, int size, long int *set, int verbose_level)
// Used in isomorph_BLT
{
	int f_v = (verbose_level >= 1);
	action *AAA;
	//sims K;
	longinteger_object go, ko;
	int i;
	int *Elt1;
	
	Elt1 = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "isomorph::induced_action_on_set_and_kernel "
				"calling induced_action_by_restriction" << endl;
		cout << "set: ";
		lint_vec_print(cout, set, size);
		cout << endl;
		}

	int f_induce_action = TRUE;
	
	AAA = gen->Poset->A2->create_induced_action_by_restriction(
		Stab,
		size, set,
		f_induce_action,
		verbose_level - 1);

	if (f_v) {
		cout << "isomorph::induced_action_on_set_and_kernel: "
				"after induced_action_by_restriction" << endl;
		}

	AAA->group_order(go);
	
	file << endl << "\\bigskip" << endl
			<< "The induced group has order "
			<< go << " and is generated by:" << endl << endl;
	AAA->group_order(go);
	for (i = 0; i < Stab->gens.len; i++) {
		int f_do_it_anyway_even_for_big_degree= TRUE; 
		int f_print_cycles_of_length_one = TRUE;
		
		file << "$g_{" << setw(2) << i + 1 << "} = $";
		AAA->element_print_as_permutation_with_offset(
			Stab->gens.ith(i), file, 1,
			f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
		file << "\\\\" << endl;
		}
	if (go.compare_with_int(10) < 0) {
		file << "group order is small, so we list "
				"all elements\\\\" << endl;
		for (i = 0; i < go.as_lint(); i++) {
			int f_do_it_anyway_even_for_big_degree = TRUE; 
			int f_print_cycles_of_length_one = TRUE;
			
			file << "$a_{" << setw(2) << i + 1 << "} = $";
			Stab->element_unrank_lint(i, Elt1);
			AAA->element_print_as_permutation_with_offset(Elt1, file, 1,
				f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one, 
				0 /* verbose_level */);
			file << "\\\\" << endl;
			}
		file << "and now the elements themselves:" << endl;
		for (i = 0; i < go.as_lint(); i++) {

			Stab->element_unrank_lint(i, Elt1);

			int *fp, n;
		
			fp = NEW_int(A->degree);
			n = A->find_fixed_points(Elt1, fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			FREE_int(fp);


			file << "$a_{" << setw(2) << i + 1 << "} = $" << endl;
			file << "$";
			AAA->element_print_latex(Elt1, file);
			file << "$ with " << n << " fixed points\\\\" << endl;
			}
		}

	if (AAA->Kernel) {
		if (f_v) {
			cout << "isomorph::induced_action_on_set_and_kernel: "
					"printing kernel generators" << endl;
			}
	AAA->Kernel->group_order(ko);
	file << "Kernel has order " << ko
			<< " and is generated by:\\\\" << endl;
	for (i = 0; i < AAA->Kernel->gens.len; i++) {
		file << "$$ b_{" << setw(2) << i + 1 << "} = " << endl;
		A->element_print_latex(AAA->Kernel->gens.ith(i), file);
		file << "$$" << endl;
		//file << "$b_{" << setw(2) << i + 1 << "} = $" << endl;
		//A->element_print_as_permutation_with_offset(K.gens.ith(i), file, 1);
		file << "\\\\" << endl;
		}
	
	if (!ko.is_one()) {
		schreier Orb;
		isomorph::A->compute_all_point_orbits(Orb,
				AAA->Kernel->gens, verbose_level - 2);
		int *val, *mult, len;
	
		file << "The kernel has $" << Orb.nb_orbits
				<< "$ orbits on the quadric.\\\\" << endl;
		int_vec_distribution(Orb.orbit_len, Orb.nb_orbits,
				val, mult, len);
		file << "The orbit length are $[";
		for (i = len - 1; i >= 0; i--) {
			file << val[i];
			if (mult[i] > 1) {
				file << "^{" << mult[i] << "}";
				}
			if (i)
				file << ", ";
			}
		file << "]$\\\\" << endl;
		
#if 0
		int min_length, min_idx;
		
		min_idx = -1;
		for (i = 0; i < Orb.nb_orbits; i++) {
			if (Orb.orbit_len[i] == 1)
				continue;
			if (min_idx == -1) {
				min_idx = i;
				min_length = Orb.orbit_len[i];
				continue;
				}
			if (Orb.orbit_len[i] < min_length) {
				min_idx = i;
				min_length = Orb.orbit_len[i];
				}
			}
		if (min_idx >= 0) {
			induced_action_on_orbit(file, AAA.Kernel->A,
					AAA.Kernel, Orb, min_idx, verbose_level);
			}
#endif

		FREE_int(val);
		FREE_int(mult);
		}
	} // if (AAA.Kernel)

	
	file << "\\bigskip" << endl << endl;
	FREE_int(Elt1);
	FREE_OBJECT(AAA);
	
}


void isomorph::handle_event_files(int nb_event_files,
		const char **event_file_name, int verbose_level)
{
	int i;
	
	Reps->count = 0;
	for (i = 0; i < nb_event_files; i++) {
		read_event_file(event_file_name[i], verbose_level);
		}
	cout << "after reading " << nb_event_files
			<< " event files, isomorph_cnt = "
			<< Reps->count << endl;
	
}

void isomorph::read_event_file(const char *event_file_name,
		int verbose_level)
{
	int i;
	int nb_completed_cases, *completed_cases;
	
	completed_cases = NEW_int(10000);
	event_file_completed_cases(event_file_name,
			nb_completed_cases, completed_cases, verbose_level);
	cout << "file " << event_file_name << " holds "
			<< nb_completed_cases << " completed cases: ";
	int_vec_print(cout, completed_cases, nb_completed_cases);
	cout << endl;
	for (i = 0; i < nb_completed_cases; i++) {
		event_file_read_case(event_file_name,
				completed_cases[i], verbose_level);
		}
	Reps->count = MAXIMUM(Reps->count,
			completed_cases[nb_completed_cases - 1] + 1);
}

#define MY_BUFSIZE 1000000

void isomorph::skip_through_event_file(
		ifstream &f, int verbose_level)
{
	char buf[MY_BUFSIZE];
	char token[1000];
	int l, j, case_no;
	char *p_buf;

	cout << "isomorph::skip_through_event_file" << endl;

	while (TRUE) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
		if (strncmp(buf, "-1", 2) == 0) {
			return;
			}
		*fp_event_out << buf << endl;
			
		p_buf = buf;
		if (strncmp(buf, "BEGIN", 5) == 0) {
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_int(&p_buf, &case_no);
			cout << "located isomorphism type "
					<< case_no << " in event file" << endl;
			cout << "buf=" << buf << endl;
			for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
				if (Reps->fusion[orbit_no] == -2) {
					break;
					}
				}
			cout << "it belongs to orbit_no " << orbit_no << endl;
			*fp_event_out << "O " << orbit_no << endl;
			Reps->fusion[orbit_no] = orbit_no;
			skip_through_event_file1(f, case_no,
					orbit_no, verbose_level);
			Reps->count++;
			}

		}
	cout << "isomorph::skip_through_event_file done" << endl;
}

void isomorph::skip_through_event_file1(ifstream &f,
		int case_no, int orbit_no, int verbose_level)
{
	int l, j, from_orbit, to_orbit, rank_subset;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];


	while (TRUE) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
			
		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			cout << "isomorphism type " << case_no
					<< " has been read from event file" << endl;
			Reps->calc_fusion_statistics();
			Reps->print_fusion_statistics();
			*fp_event_out << buf << endl;
			return;
			}
		s_scan_token(&p_buf, token);
		if (strcmp(token, "F") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_int(&p_buf, &to_orbit);

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read F)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			Reps->fusion[to_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "A") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_token(&p_buf, token); // group order

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read A)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "AF") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_int(&p_buf, &to_orbit);
			s_scan_token(&p_buf, token); // group order

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read AF)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "O") == 0) {
			// do not print buf
			}
		else {
			*fp_event_out << buf << endl;
			}

		}
}


void isomorph::event_file_completed_cases(
	const char *event_file_name,
	int &nb_completed_cases, int *completed_cases,
	int verbose_level)
{
	int l, j, a;
	char *p_buf;
	char token[1000];
	ifstream f(event_file_name);
	char buf[MY_BUFSIZE];
	
	nb_completed_cases = 0;
	while (TRUE) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
			
		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_int(&p_buf, &a);
			cout << "isomorphism type " << a
					<< " has been completed" << endl;
			completed_cases[nb_completed_cases++] = a;
			}

		}
}

void isomorph::event_file_read_case(
		const char *event_file_name, int case_no,
		int verbose_level)
{
	int l, j, a;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	ifstream f(event_file_name);
	
	while (TRUE) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
			
		p_buf = buf;
		if (strncmp(buf, "BEGIN", 5) == 0) {
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_token(&p_buf, token);
			s_scan_int(&p_buf, &a);
			if (a == case_no) {
				cout << "located isomorphism type " << a
						<< " in event file" << endl;
				event_file_read_case1(f, case_no, verbose_level);
				return;
				}
			}

		}
	cout << "did not find case " << case_no << " in event file "
			<< event_file_name << endl;
	exit(1);
}

void isomorph::event_file_read_case1(ifstream &f,
		int case_no, int verbose_level)
{
	int l, j, from_orbit, to_orbit, rank_subset;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];


	while (TRUE) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
			
		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			cout << "isomorphism type " << case_no
					<< " has been read from event file" << endl;
			Reps->calc_fusion_statistics();
			Reps->print_fusion_statistics();
			return;
			}
		s_scan_token(&p_buf, token);
		if (strcmp(token, "F") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_int(&p_buf, &to_orbit);

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			Reps->fusion[to_orbit] = from_orbit;
			}
		else if (strcmp(token, "A") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_token(&p_buf, token); // group order

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			}
		else if (strcmp(token, "AF") == 0) {
			s_scan_int(&p_buf, &from_orbit);
			s_scan_int(&p_buf, &rank_subset);
			s_scan_int(&p_buf, &to_orbit);
			s_scan_token(&p_buf, token); // group order

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			}

		}
}

#define MY_BUFSIZE 1000000

int isomorph::next_subset_play_back(int &subset_rank,
	ifstream *play_back_file,
	int &f_eof, int verbose_level)
{
	int f_v = (verbose_level >= 3);
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	int rank;
	combinatorics_domain Combi;
		
	f_eof = FALSE;
	if (play_back_file->eof()) {
		cout << "end of file reached" << endl;
		f_eof = TRUE;
		return FALSE;
		}
	play_back_file->getline(buf, MY_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		cout << "isomorph::next_subset_play_back "
				"reached an empty line" << endl;
		exit(1);
		}
	if (strncmp(buf, "BEGIN", 5) == 0) {
		cout << "BEGIN reached" << endl;
		play_back_file->getline(buf, MY_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			cout << "empty line reached" << endl;
			exit(1);
			}
		}
	if (strncmp(buf, "-1", 2) == 0) {
		cout << "end of file marker -1 reached" << endl;
		f_eof = TRUE;
		return FALSE;
		}
	if (strncmp(buf, "END-EOF", 7) == 0) {
		cout << "END-EOF reached" << endl;
		f_eof = TRUE;
		return FALSE;
		}
	if (strncmp(buf, "END", 3) == 0) {
		cout << "END reached" << endl;
		return FALSE;
		}
	if (f_v) {
		cout << "parsing: " << buf << endl;
		}
	p_buf = buf;
	s_scan_token(&p_buf, token);
	s_scan_int(&p_buf, &rank);
	s_scan_int(&p_buf, &rank);
	if (f_v) {
		cout << "rank = " << rank << endl;
		cout << "subset_rank = " << subset_rank << endl;
		}
	if (rank == subset_rank) {
		if (f_v) {
			cout << "rank is equal to subset_rank, "
					"so we proceed" << endl;
			}
		}
	else {

#if 0
		if (rank < subset_rank) {
			cout << "rank is less than subset_rank, "
					"something is wrong" << endl;
			exit(1);
			}
#endif
		Combi.unrank_k_subset(rank, subset, size, level);
		subset_rank = Combi.rank_k_subset(subset, size, level);
		if (f_v) {
			cout << "moved to set " << subset_rank << endl;
			}
		}
	return TRUE;
}

void isomorph::read_everything_including_classification(
		const char *prefix_classify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "isomorph::read_everything_including_classification" << endl;
		}
	

	if (f_v) {
		cout << "isomorph::read_everything_including_classification before read_data_files_for_starter" << endl;
	}
	read_data_files_for_starter(level,
			prefix_classify, verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after read_data_files_for_starter" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification before init_solution" << endl;
	}
	init_solution(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after init_solution" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification before load_table_of_solutions" << endl;
	}
	load_table_of_solutions(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after load_table_of_solutions" << endl;
	}
	
	if (f_v) {
		cout << "isomorph::read_everything_including_classification before read_orbit_data" << endl;
	}
	read_orbit_data(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after read_orbit_data" << endl;
	}

	depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph::read_everything_including_classification before gen->recreate_schreier_vectors_up_to_level" << endl;
	}
	gen->recreate_schreier_vectors_up_to_level(level - 1,
			verbose_level);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after gen->recreate_schreier_vectors_up_to_level" << endl;
	}

	
	if (f_v) {
		for (i = 0; i <= level + 1; i++) {
			cout << "gen->first_oracle_node_at_level[" << i
					<< "]=" << gen->first_poset_orbit_node_at_level[i] << endl;
			}
		cout << "depth_completed=" << depth_completed << endl;
		}


	if (f_v) {
		cout << "isomorph::read_everything_including_classification before iso_test_init" << endl;
	}
	iso_test_init(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after iso_test_init" << endl;
	}

	//int f_implicit_fusion = FALSE;
	
	gen->f_allowed_to_show_group_elements = FALSE;
	
	if (f_v) {
		cout << "isomorph::read_everything_including_classification before read_starter_nb_orbits" << endl;
	}
	read_starter_nb_orbits(verbose_level); // added Oct 30, 2014 
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after read_starter_nb_orbits" << endl;
	}
	
	if (f_v) {
		cout << "isomorph::read_everything_including_classification before Reps->load" << endl;
	}
	Reps->load(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification after Reps->load" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_"
				"classification done" << endl;
		}
}

}}


