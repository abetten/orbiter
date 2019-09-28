/*
 * file_io.cpp
 *
 *  Created on: Apr 21, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;

#include <cstdio>
#include <sys/types.h>
#ifdef SYSTEMUNIX
#include <unistd.h>
#endif
#include <fcntl.h>



namespace orbiter {
namespace foundations {

#define MY_OWN_BUFSIZE 1000000

file_io::file_io()
{

}

file_io::~file_io()
{

}

void file_io::concatenate_files(const char *fname_in_mask, int N,
	const char *fname_out, const char *EOF_marker, int f_title_line,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	char *buf;
	int h, cnt;

	if (f_v) {
		cout << "concatenate_files " << fname_in_mask
			<< " N=" << N << " fname_out=" << fname_out << endl;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);

	{
	ofstream fp_out(fname_out);
	for (h = 0; h < N; h++) {
		sprintf(fname, fname_in_mask, h);
		if (file_size(fname) < 0) {
			cout << "concatenate_files input file does not exist" << endl;
			exit(1);
			}

			{
			ifstream fp(fname);

			cnt = 0;
			while (TRUE) {
				if (fp.eof()) {
					cout << "Encountered End-of-file without having seem EOF "
							"marker, perhaps the file is corrupt. "
							"I was trying to read the file " << fname << endl;
					//exit(1);
					break;
					}

				fp.getline(buf, MY_OWN_BUFSIZE, '\n');
				cout << "Read: " << buf << endl;
				if (strncmp(buf, EOF_marker, strlen(EOF_marker)) == 0) {
					break;
					}
				if (f_title_line) {
					if (h == 0 || cnt) {
						fp_out << buf << endl;
						}
					}
				else {
					fp_out << buf << endl;
					}
				cnt++;
				}
			}
		} // next h
	fp_out << EOF_marker << endl;
	}
	cout << "Written file " << fname_out << " of size "
		<< file_size(fname_out) << endl;
	FREE_char(buf);
	if (f_v) {
		cout << "concatenate_files done" << endl;
		}

}

void file_io::poset_classification_read_candidates_of_orbit(
	const char *fname, int orbit_at_level,
	int *&candidates, int &nb_candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb, cand_first, i;


	if (f_v) {
		cout << "poset_classification_read_candidates_of_orbit" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		}

	if (file_size(fname) <= 0) {
		cout << "poset_classification_read_candidates_of_orbit file "
				<< fname << " does not exist" << endl;
		exit(1);
		}

#if 0
	FILE *fp;
	fp = fopen(fname, "rb");

	nb = fread_int4(fp);
	if (orbit_at_level >= nb) {
		cout << "poset_classification_read_candidates_of_orbit "
				"orbit_at_level >= nb" << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		cout << "nb=" << nb << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "seeking position "
				<< (1 + orbit_at_level * 2) * sizeof(int_4) << endl;
		}
	fseek(fp, (1 + orbit_at_level * 2) * sizeof(int_4), SEEK_SET);
	nb_candidates = fread_int4(fp);
	if (f_vv) {
		cout << "nb_candidates=" << nb_candidates << endl;
		}
	cand_first = fread_int4(fp);
	if (f_v) {
		cout << "cand_first=" << cand_first << endl;
		}
	candidates = NEW_int(nb_candidates);
	fseek(fp, (1 + nb * 2 + cand_first) * sizeof(int_4), SEEK_SET);
	for (i = 0; i < nb_candidates; i++) {
		candidates[i] = fread_int4(fp);
		}
	fclose(fp);
#else
	{
		ifstream fp(fname, ios::binary);
		fp.read((char *) &nb, sizeof(int));
		if (orbit_at_level >= nb) {
			cout << "poset_classification_read_candidates_of_orbit "
					"orbit_at_level >= nb" << endl;
			cout << "orbit_at_level=" << orbit_at_level << endl;
			cout << "nb=" << nb << endl;
			exit(1);
			}
		if (f_vv) {
			cout << "seeking position "
					<< (1 + orbit_at_level * 2) * sizeof(int) << endl;
			}
		fp.seekg((1 + orbit_at_level * 2) * sizeof(int), ios::beg);
		fp.read((char *) &nb_candidates, sizeof(int));
		if (f_vv) {
			cout << "nb_candidates=" << nb_candidates << endl;
			}
		fp.read((char *) &cand_first, sizeof(int));
		if (f_v) {
			cout << "cand_first=" << cand_first << endl;
			}
		candidates = NEW_int(nb_candidates);
		fp.seekg((1 + nb * 2 + cand_first) * sizeof(int), ios::beg);
		for (i = 0; i < nb_candidates; i++) {
			fp.read((char *) &candidates[i], sizeof(int));

		}

	}
#endif
	if (f_v) {
		cout << "poset_classification_read_candidates_of_orbit "
				"done" << endl;
		}
}


void file_io::read_candidates_for_one_orbit_from_file(char *prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		int *S,
		void (*early_test_func_callback)(int *S, int len,
			int *candidates, int nb_candidates,
			int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		int *&candidates,
		int &nb_candidates,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, orbit_idx;
	int *candidates1 = NULL;
	int nb_candidates1;

	if (f_v) {
		cout << "read_candidates_for_one_orbit_from_file" << endl;
		cout << "level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " level_of_candidates_file="
				<< level_of_candidates_file << endl;
	}

	orbit_idx = find_orbit_index_in_data_file(prefix,
			level_of_candidates_file, S,
			verbose_level);

	if (f_v) {
		cout << "read_candidates_for_one_orbit_from_file "
				"orbit_idx=" << orbit_idx << endl;
	}

	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files "
				"before generator_read_candidates_of_orbit" << endl;
		}
	char fname2[1000];
	sprintf(fname2, "%s_lvl_%d_candidates.bin",
			prefix, level_of_candidates_file);
	poset_classification_read_candidates_of_orbit(
		fname2, orbit_idx,
		candidates1, nb_candidates1, verbose_level - 1);


	for (h = level_of_candidates_file; h < level; h++) {

		int *candidates2;
		int nb_candidates2;

		if (f_v) {
			cout << "read_orbit_rep_and_candidates_from_files_"
					"and_process testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
			}
		candidates2 = NEW_int(nb_candidates1);

		(*early_test_func_callback)(S, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, 0 /*verbose_level - 1*/);

		if (f_v) {
			cout << "read_orbit_rep_and_candidates_from_files_"
					"and_process number of candidates at level "
					<< h + 1 << " reduced from " << nb_candidates1
					<< " to " << nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
			}

		int_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_int(candidates2);
		}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "read_candidates_for_one_orbit_from_file done" << endl;
	}
}



int file_io::find_orbit_index_in_data_file(const char *prefix,
		int level_of_candidates_file, int *starter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	int orbit_idx;

	if (f_v) {
		cout << "find_orbit_index_in_data_file" << endl;
	}

	sprintf(fname, "%s_lvl_%d", prefix, level_of_candidates_file);

	if (file_size(fname) <= 0) {
		cout << "find_orbit_index_in_data_file file "
				<< fname << " does not exist" << endl;
		exit(1);
		}
	ifstream f(fname);
	int a, i, cnt;
	int *S;
	char buf[MY_OWN_BUFSIZE];
	int len, str_len;
	char *p_buf;

	S = NEW_int(level_of_candidates_file);

	cnt = 0;
	f.getline(buf, MY_OWN_BUFSIZE, '\n'); // skip the first line

	orbit_idx = 0;

	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, MY_OWN_BUFSIZE, '\n');
		//cout << "Read line " << cnt << "='" << buf << "'" << endl;
		str_len = strlen(buf);
		if (str_len == 0) {
			cout << "read_orbit_rep_and_candidates_from_files "
					"str_len == 0" << endl;
			exit(1);
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;
		s_scan_int(&p_buf, &a);
		if (a == -1) {
			break;
			}
		len = a;
		if (a != level_of_candidates_file) {
			cout << "a != level_of_candidates_file" << endl;
			cout << "a=" << a << endl;
			cout << "level_of_candidates_file="
					<< level_of_candidates_file << endl;
			exit(1);
			}
		for (i = 0; i < len; i++) {
			s_scan_int(&p_buf, &S[i]);
			}
		for (i = 0; i < level_of_candidates_file; i++) {
			if (S[i] != starter[i]) {
				break;
				}
			}
		if (i == level_of_candidates_file) {
			// We found the representative that matches the prefix:
			orbit_idx = cnt;
			break;
			}
		else {
			cnt++;
			}
		}
	FREE_int(S);
	if (f_v) {
		cout << "find_orbit_index_in_data_file done" << endl;
	}
	return orbit_idx;
}


void file_io::write_exact_cover_problem_to_file(int *Inc,
		int nb_rows, int nb_cols, const char *fname)
{
	int i, j, d;

	{
	ofstream fp(fname);
	fp << nb_rows << " " << nb_cols << endl;
	for (i = 0; i < nb_rows; i++) {
		d = 0;
		for (j = 0; j < nb_cols; j++) {
			if (Inc[i * nb_cols + j]) {
				d++;
				}
			}
		fp << d;
		for (j = 0; j < nb_cols; j++) {
			if (Inc[i * nb_cols + j]) {
				fp << " " << j;
				}
			}
		fp << endl;
		}
	}
	cout << "write_exact_cover_problem_to_file written file "
		<< fname << " of size " << file_size(fname) << endl;
}

#define BUFSIZE_READ_SOLUTION_FILE ONE_MILLION

void file_io::read_solution_file(char *fname,
	int *Inc, int nb_rows, int nb_cols,
	int *&Solutions, int &sol_length, int &nb_sol,
	int verbose_level)
// sol_length must be constant
{
	int f_v = (verbose_level >= 1);
	int nb, nb_max, i, j, a, nb_sol1;
	int *x, *y;

	if (f_v) {
		cout << "file_io::read_solution_file" << endl;
		}
	x = NEW_int(nb_cols);
	y = NEW_int(nb_rows);
	if (f_v) {
		cout << "file_io::read_solution_file reading file " << fname
			<< " of size " << file_size(fname) << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "file_io::read_solution_file "
				"There is something wrong with the file "
			<< fname << endl;
		exit(1);
		}
	char *buf;
	char *p_buf;
	buf = NEW_char(BUFSIZE_READ_SOLUTION_FILE);
	nb_sol = 0;
	nb_max = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, BUFSIZE_READ_SOLUTION_FILE, '\n');
			p_buf = buf;
			if (strlen(buf)) {
				for (j = 0; j < nb_cols; j++) {
					x[j] = 0;
					}
				s_scan_int(&p_buf, &nb);
				if (nb_sol == 0) {
					nb_max = nb;
					}
				else {
					if (nb != nb_max) {
						cout << "file_io::read_solution_file "
								"solutions have different length" << endl;
						exit(1);
						}
					}
				//cout << "buf='" << buf << "' nb=" << nb << endl;

				for (i = 0; i < nb_rows; i++) {
					y[i] = 0;
					}
				for (i = 0; i < nb_rows; i++) {
					for (j = 0; j < nb_cols; j++) {
						y[i] += Inc[i * nb_cols + j] * x[j];
						}
					}
				for (i = 0; i < nb_rows; i++) {
					if (y[i] != 1) {
						cout << "file_io::read_solution_file "
								"Not a solution!" << endl;
						int_vec_print_fully(cout, y, nb_rows);
						cout << endl;
						exit(1);
						}
					}
				nb_sol++;
				}
			}
	}
	if (f_v) {
		cout << "file_io::read_solution_file: Counted " << nb_sol
			<< " solutions in " << fname
			<< " starting to read now." << endl;
		}
	sol_length = nb_max;
	Solutions = NEW_int(nb_sol * sol_length);
	nb_sol1 = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, BUFSIZE_READ_SOLUTION_FILE, '\n');
			p_buf = buf;
			if (strlen(buf)) {
				for (j = 0; j < nb_cols; j++) {
					x[j] = 0;
					}
				s_scan_int(&p_buf, &nb);
				//cout << "buf='" << buf << "' nb=" << nb << endl;

				for (i = 0; i < sol_length; i++) {
					s_scan_int(&p_buf, &a);
					Solutions[nb_sol1 * sol_length + i] = a;
					}
				nb_sol1++;
				}
			}
	}
	if (f_v) {
		cout << "file_io::read_solution_file: Read " << nb_sol
			<< " solutions from file " << fname << endl;
		}
	FREE_int(x);
	FREE_int(y);
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::read_solution_file done" << endl;
		}
}

void file_io::count_number_of_solutions_in_file_and_get_solution_size(
	const char *fname,
	int &nb_solutions, int &solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	int s;

	if (f_v) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
		}

	nb_solutions = 0;
	if (file_size(fname) < 0) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);



	solution_size = -1;
	{
	ifstream fp(fname);
	char *p_buf;


	while (TRUE) {
		if (fp.eof()) {
			cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size "
					"eof, break" << endl;
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size "
					"empty line" << endl;
			exit(1);
			}

		p_buf = buf;
		s_scan_int(&p_buf, &s);
		if (solution_size == -1) {
			solution_size = s;
		}
		else {
			if (solution_size != s) {
				cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size "
						"solution_size is not constant" << endl;
				exit(1);
			}
		}

		if (strncmp(buf, "-1", 2) == 0) {
			break;
			}
		nb_solutions++;
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
		}
}

void file_io::count_number_of_solutions_in_file(const char *fname,
	int &nb_solutions,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;

	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
		}

	nb_solutions = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);



	{
	ifstream fp(fname);


	while (TRUE) {
		if (fp.eof()) {
			cout << "count_number_of_solutions_in_file "
					"eof, break" << endl;
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "count_number_of_solutions_in_file "
					"empty line" << endl;
			exit(1);
			}

		if (strncmp(buf, "-1", 2) == 0) {
			break;
			}
		nb_solutions++;
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
		}
}

void file_io::count_number_of_solutions_in_file_by_case(const char *fname,
	int *&nb_solutions, int *&case_nb, int &nb_cases,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	//int nb_sol;
	int N = 1000;
	int i;
	int the_case;
	int the_case_count = 0;

	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case "
			<< fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
		}

	nb_solutions = NEW_int(N);
	case_nb = NEW_int(N);
	nb_cases = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file_by_case file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);



	{
	ifstream fp(fname);


	//nb_sol = 0;
	the_case = -1;
	while (TRUE) {
		if (fp.eof()) {
			cout << "count_number_of_solutions_in_file_by_case "
					"eof, break" << endl;
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "count_number_of_solutions_in_file_by_case "
					"empty line, break" << endl;
			break;
			}

		if (strncmp(buf, "# start case", 12) == 0) {
			the_case = atoi(buf + 13);
			the_case_count = 0;
			cout << "count_number_of_solutions_in_file_by_case "
					"read start case " << the_case << endl;
			}
		else if (strncmp(buf, "# end case", 10) == 0) {
			if (nb_cases == N) {
				int *nb_solutions1;
				int *case_nb1;

				nb_solutions1 = NEW_int(N + 1000);
				case_nb1 = NEW_int(N + 1000);
				for (i = 0; i < N; i++) {
					nb_solutions1[i] = nb_solutions[i];
					case_nb1[i] = case_nb[i];
					}
				FREE_int(nb_solutions);
				FREE_int(case_nb);
				nb_solutions = nb_solutions1;
				case_nb = case_nb1;
				N += 1000;
				}
			nb_solutions[nb_cases] = the_case_count;
			case_nb[nb_cases] = the_case;
			nb_cases++;
			//cout << "count_number_of_solutions_in_file_by_case "
			//"read end case " << the_case << endl;
			the_case = -1;
			}
		else {
			if (the_case >= 0) {
				the_case_count++;
				}
			}

		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case "
			<< fname << endl;
		cout << "nb_cases = " << nb_cases << endl;
		}
}


void file_io::read_solutions_from_file_and_get_solution_size(const char *fname,
	int &nb_solutions, int *&Solutions, int &solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "read_solutions_from_file_and_get_solution_size" << endl;
		cout << "read_solutions_from_file_and_get_solution_size trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
	}

	if (file_size(fname) < 0) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size "
				"the file " << fname << " does not exist" << endl;
		return;
		}

	count_number_of_solutions_in_file_and_get_solution_size(fname,
		nb_solutions, solution_size,
		verbose_level - 2);

	if (f_v) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size, reading "
			<< nb_solutions << " solutions of size " << solution_size << endl;
	}

	Solutions = NEW_int(nb_solutions * solution_size);

	char *buf;
	char *p_buf;
	int i, a, nb_sol;

	buf = NEW_char(MY_OWN_BUFSIZE);
	nb_sol = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, MY_OWN_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
				}
			if (a != solution_size) {
				cout << "file_io::read_solutions_from_file_and_get_solution_size "
						"a != solution_size" << endl;
				exit(1);
				}
			for (i = 0; i < solution_size; i++) {
				s_scan_int(&p_buf, &a);
				Solutions[nb_sol * solution_size + i] = a;
				}
			nb_sol++;
			}
	}
	if (nb_sol != nb_solutions) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size "
				"nb_sol != nb_solutions" << endl;
		exit(1);
		}
	FREE_char(buf);

	if (f_v) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size" << endl;
	}
}


void file_io::read_solutions_from_file(const char *fname,
	int &nb_solutions, int *&Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int i, a, nb_sol;

	if (f_v) {
		cout << "read_solutions_from_file" << endl;
		cout << "read_solutions_from_file trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file solution_size="
			<< solution_size << endl;
		}

	if (file_size(fname) < 0) {
		cout << "file_io::read_solutions_from_file the file " << fname << " does not exist" << endl;
		return;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);

	count_number_of_solutions_in_file(fname,
		nb_solutions,
		verbose_level - 2);
	if (f_v) {
		cout << "read_solutions_from_file, reading "
			<< nb_solutions << " solutions" << endl;
		}



	Solutions = NEW_int(nb_solutions * solution_size);

	nb_sol = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, MY_OWN_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
				}
			if (a != solution_size) {
				cout << "read_solutions_from_file "
						"a != solution_size" << endl;
				exit(1);
				}
			for (i = 0; i < solution_size; i++) {
				s_scan_int(&p_buf, &a);
				Solutions[nb_sol * solution_size + i] = a;
				}
			nb_sol++;
			}
	}
	if (nb_sol != nb_solutions) {
		cout << "read_solutions_from_file "
				"nb_sol != nb_solutions" << endl;
		exit(1);
		}
	FREE_char(buf);
	if (f_v) {
		cout << "read_solutions_from_file done" << endl;
		}
}

void file_io::read_solutions_from_file_by_case(const char *fname,
	int *nb_solutions, int *case_nb, int nb_cases,
	int **&Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	//int nb_sol;
	int i;
	int nb_case1;
	int the_case;
	int the_case_count = 0;

	if (f_v) {
		cout << "read_solutions_from_file_by_case" << endl;
		cout << "read_solutions_from_file_by_case trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file_by_case solution_size="
			<< solution_size << endl;
		}

	if (file_size(fname) < 0) {
		return;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);

	Solutions = NEW_pint(nb_cases);

	{
	ifstream fp(fname);


	//nb_sol = 0;
	nb_case1 = 0;
	the_case = -1;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "read_solutions_from_file_by_case "
					"empty line, break" << endl;
			break;
			}

		if (strncmp(buf, "# start case", 12) == 0) {
			the_case = atoi(buf + 13);
			the_case_count = 0;
			if (the_case != case_nb[nb_case1]) {
				cout << "read_solutions_from_file_by_case "
						"the_case != case_nb[nb_case1]" << endl;
				exit(1);
				}
			Solutions[nb_case1] =
					NEW_int(nb_solutions[nb_case1] * solution_size);
			cout << "read_solutions_from_file_by_case "
					"read start case " << the_case << endl;
			}
		else if (strncmp(buf, "# end case", 10) == 0) {
			if (the_case_count != nb_solutions[nb_case1]) {
				cout << "read_solutions_from_file_by_case "
						"the_case_count != nb_solutions[nb_case1]" << endl;
				exit(1);
				}
			cout << "read_solutions_from_file_by_case "
					"read end case " << the_case << endl;
			nb_case1++;
			the_case = -1;
			}
		else {
			if (the_case >= 0) {
				char *p_buf;
				int sz, a;

				//cout << "read_solutions_from_file_by_case "
				//"reading solution " << the_case_count
				//<< " for case " << the_case << endl;
				p_buf = buf;
				s_scan_int(&p_buf, &sz);
				if (sz != solution_size) {
					cout << "read_solutions_from_file_by_case "
							"sz != solution_size" << endl;
					exit(1);
					}
				for (i = 0; i < sz; i++) {
					s_scan_int(&p_buf, &a);
					Solutions[nb_case1][the_case_count * solution_size + i] = a;
					}
				the_case_count++;
				}
			}

		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "read_solutions_from_file_by_case done" << endl;
		}
}

void file_io::copy_file_to_ostream(ostream &ost, char *fname)
{
	//char buf[MY_OWN_BUFSIZE];

	{
	ifstream fp(fname);

#if 0
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');

#if 0
		// check for comment line:
		if (buf[0] == '#')
			continue;
#endif

		ost << buf << endl;
		}
#endif
	while (TRUE) {
		char c;
		fp.get(c);
		if (fp.eof()) {
			break;
			}
		ost << c;
		}
	}

}

void file_io::int_vec_write_csv(int *v, int len,
	const char *fname, const char *label)
{
	int i;

	{
	ofstream f(fname);

	f << "Case," << label << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v[i] << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_vecs_write_csv(int *v1, int *v2, int len,
	const char *fname, const char *label1, const char *label2)
{
	int i;

	{
	ofstream f(fname);

	f << "Case," << label1 << "," << label2 << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v1[i] << "," << v2[i] << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_vecs3_write_csv(int *v1, int *v2, int *v3, int len,
	const char *fname,
	const char *label1, const char *label2, const char *label3)
{
	int i;

	{
	ofstream f(fname);

	f << "Case," << label1 << "," << label2 << "," << label3 << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v1[i] << "," << v2[i] << "," << v3[i] << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_vec_array_write_csv(int nb_vecs, int **Vec, int len,
	const char *fname, const char **column_label)
{
	int i, j;

	cout << "int_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
	cout << "column labels:" << endl;
	for (j = 0; j < nb_vecs; j++) {
		cout << j << " : " << column_label[j] << endl;
		}

	{
	ofstream f(fname);

	f << "Row";
	for (j = 0; j < nb_vecs; j++) {
		f << "," << column_label[j];
		}
	f << endl;
	for (i = 0; i < len; i++) {
		f << i;
		for (j = 0; j < nb_vecs; j++) {
			f << "," << Vec[j][i];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_matrix_write_csv(const char *fname, int *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);

	f << "Row";
	for (j = 0; j < n; j++) {
		f << ",C" << j;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void file_io::lint_matrix_write_csv(const char *fname, long int *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);

	f << "Row";
	for (j = 0; j < n; j++) {
		f << ",C" << j;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void file_io::double_matrix_write_csv(
		const char *fname, double *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);

	f << "Row";
	for (j = 0; j < n; j++) {
		f << ",C" << j;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_matrix_write_csv_with_labels(const char *fname,
	int *M, int m, int n, const char **column_label)
{
	int i, j;

	{
	ofstream f(fname);

	f << "Row";
	for (j = 0; j < n; j++) {
		f << "," << column_label[j];
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void file_io::int_matrix_read_csv(const char *fname,
	int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "int_matrix_read_csv reading file " << fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "int_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = my_atoi(S.get_string(i + 1, j + 1));
			M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "int_matrix_read_csv done" << endl;
		}

}

void file_io::lint_matrix_read_csv(const char *fname,
	long int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;

	if (f_v) {
		cout << "lint_matrix_read_csv reading file " << fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "int_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = NEW_lint(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = my_atol(S.get_string(i + 1, j + 1));
			M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "lint_matrix_read_csv done" << endl;
		}

}

void file_io::double_matrix_read_csv(const char *fname,
	double *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "double_matrix_read_csv reading file "
			<< fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "double_matrix_read_csv file " << fname
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;
	double d;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = new double [m * n];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			sscanf(S.get_string(i + 1, j + 1), "%lf", &d);
			M[i * n + j] = d;
			}
		}
	}
	if (f_v) {
		cout << "double_matrix_read_csv done" << endl;
		}

}

void file_io::int_matrix_write_text(const char *fname, int *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);

	f << m << " " << n << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f << M[i * n + j] << " ";
			}
		f << endl;
		}
	}
}

void file_io::int_matrix_read_text(const char *fname, int *&M, int &m, int &n)
{
	int i, j;

	if (file_size(fname) <= 0) {
		cout << "int_matrix_read_text The file "
			<< fname << " does not exist" << endl;
		exit(1);
		}
	{
	ifstream f(fname);

	f >> m >> n;
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f >> M[i * n + j];
			}
		}
	}
}

void file_io::parse_sets(int nb_cases, char **data, int f_casenumbers,
	int *&Set_sizes, int **&Sets,
	char **&Ago_ascii, char **&Aut_ascii,
	int *&Casenumbers,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, casenumber;
	char *ago_ascii, *aut_ascii;
	char *p_buf;

	if (f_v) {
		cout << "parse_sets f_casenumbers=" << f_casenumbers
			<< " nb_cases = " << nb_cases << endl;
		}

	ago_ascii = NEW_char(MY_OWN_BUFSIZE);
	aut_ascii = NEW_char(MY_OWN_BUFSIZE);

	Set_sizes = NEW_int(nb_cases);
	Sets = NEW_pint(nb_cases);
	Ago_ascii = NEW_pchar(nb_cases);
	Aut_ascii = NEW_pchar(nb_cases);
	Casenumbers = NEW_int(nb_cases);

	for (h = 0; h < nb_cases; h++) {

		//cout << h << " : ";
		//cout << " : " << data[h] << endl;

		p_buf = data[h];
		if (f_casenumbers) {
			s_scan_int(&p_buf, &casenumber);
			}
		else {
			casenumber = h;
			}

		parse_line(p_buf, Set_sizes[h], Sets[h],
			ago_ascii, aut_ascii);

		Casenumbers[h] = casenumber;

		Ago_ascii[h] = NEW_char(strlen(ago_ascii) + 1);
		strcpy(Ago_ascii[h], ago_ascii);

		Aut_ascii[h] = NEW_char(strlen(aut_ascii) + 1);
		strcpy(Aut_ascii[h], aut_ascii);

#if 0
		cout << h << " : ";
		print_set(cout, len, sets[h]);
		cout << " : " << data[h] << endl;
#endif

		if (f_vv && ((h % 1000000) == 0)) {
			cout << h << " : " << Casenumbers[h]
				<< " : " << data[h] << endl;
			}
		}


	FREE_char(ago_ascii);
	FREE_char(aut_ascii);
}

void file_io::parse_line(char *line, int &len,
	int *&set, char *ago_ascii, char *aut_ascii)
{
	int i;
	char *p_buf;

	//cout << "parse_line: " << line << endl;
	p_buf = line;
	s_scan_int(&p_buf, &len);
	//cout << "parsing data of length " << len << endl;
	set = NEW_int(len);
	for (i = 0; i < len; i++) {
		s_scan_int(&p_buf, &set[i]);
		}
	s_scan_token(&p_buf, ago_ascii);
	if (strcmp(ago_ascii, "1") == 0) {
		aut_ascii[0] = 0;
		}
	else {
		s_scan_token(&p_buf, aut_ascii);
		}
}


int file_io::count_number_of_orbits_in_file(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf, *p_buf;
	int nb_sol, len;
	int ret;

	if (f_v) {
		cout << "count_number_of_orbits_in_file " << fname << endl;
		cout << "count_number_of_orbits_in_file "
				"trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_orbits_in_file "
				"file size is -1" << endl;
		return -1;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);



	{
	ifstream fp(fname);


	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}

		//cout << "count_number_of_orbits_in_file "
		//"reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			cout << "count_number_of_orbits_in_file "
					"reading an empty line" << endl;
			break;
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;
		s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "count_number_of_orbits_in_file "
						"found a complete file with " << nb_sol
						<< " solutions" << endl;
				}
			break;
			}
		else {
			if (FALSE) {
				cout << "count_number_of_orbits_in_file "
						"found a set of size " << len << endl;
				}
			}
		nb_sol++;
		}
	}
	ret = nb_sol;
//finish:

	FREE_char(buf);

	return ret;
}

int file_io::count_number_of_lines_in_file(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	int nb_lines;

	if (f_v) {
		cout << "count_number_of_lines_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_lines_in_file file size is -1" << endl;
		return 0;
		}

	buf = NEW_char(MY_OWN_BUFSIZE);



	{
	ifstream fp(fname);


	nb_lines = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}

		//cout << "count_number_of_lines_in_file "
		// "reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		nb_lines++;
		}
	}
	FREE_char(buf);

	return nb_lines;
}

int file_io::try_to_read_file(const char *fname,
	int &nb_cases, char **&data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int n1;
	char *buf, *p_buf;
	int nb_sol, len, a;

	if (f_v) {
		cout << "try_to_read_file trying to read file " << fname
			<< " of size " << file_size(fname) << endl;
		}
	buf = NEW_char(MY_OWN_BUFSIZE);


	if (file_size(fname) <= 0)
		goto return_false;

	{
	ifstream fp(fname);

#if 0
	if (fp.eof()) {
		goto return_false;
		}
	fp.getline(buf, MY_OWN_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		goto return_false;
		}
	sscanf(buf + 1, "%d", &n1);
	cout << "n1=" << n1;
	if (n1 != n) {
		cout << "try_to_read_file() n1 != n" << endl;
		exit(1);
		}
#endif

	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			goto return_false;
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;
		s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "found a complete file with "
					<< nb_sol << " solutions" << endl;
				}
			break;
			}
		nb_sol++;
		}
	}
	nb_cases = nb_sol;
	data = NEW_pchar(nb_cases);
	{
	ifstream fp(fname);

#if 0
	if (fp.eof()) {
		goto return_false;
		}
	fp.getline(buf, MY_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		goto return_false;
		}
	sscanf(buf + 1, "%d", &n1);
	if (n1 != n) {
		cout << "try_to_read_file() n1 != n" << endl;
		exit(1);
		}
#endif

	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_OWN_BUFSIZE, '\n');
		len = strlen(buf);
		if (len == 0) {
			goto return_false;
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;
		s_scan_int(&p_buf, &a);
		if (a == -1) {
			if (f_v) {
				cout << "read " << nb_sol
					<< " solutions" << endl;
				}
			break;
			}


		data[nb_sol] = NEW_char(len + 1);
		strcpy(data[nb_sol], buf);

		//cout << nb_sol << " : " << data[nb_sol] << endl;

		nb_sol++;
		}
	}

	FREE_char(buf);
	return TRUE;

return_false:
	FREE_char(buf);
	return FALSE;
}

void file_io::read_and_parse_data_file(
	const char *fname, int &nb_cases,
	char **&data, int **&sets, int *&set_sizes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "read_and_parse_data_file: reading file "
			<< fname << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level)) {
		if (f_vv) {
			cout << "file read containing " << nb_cases
				<< " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file couldn't read file "
			<< fname << endl;
		exit(1);
		}

#if 0
	for (i = 0; i < nb_cases; i++) {
		cout << i << " : " << data[i] << endl;
		}
#endif


	if (f_v) {
		cout << "read_and_parse_data_file: parsing sets" << endl;
		}
	//parse_sets(nb_cases, data, set_sizes, sets);

	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i;

	parse_sets(nb_cases, data, FALSE /*f_casenumbers */,
		set_sizes, sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level - 2*/);

	FREE_int(Casenumbers);

	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}

	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
		}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);
	if (f_v) {
		cout << "read_and_parse_data_file done" << endl;
		}

}

void file_io::parse_sets_and_check_sizes_easy(int len, int nb_cases,
	char **data, int **&sets)
{
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int *set_sizes;
	int i;

	parse_sets(nb_cases, data, FALSE /*f_casenumbers */,
		set_sizes, sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level - 2*/);
	for (i = 0; i < nb_cases; i++) {
		if (set_sizes[i] != len) {
			cout << "parse_sets_and_check_sizes_easy "
					"set_sizes[i] != len" << endl;
			exit(1);
			}
		}


	FREE_int(set_sizes);
	FREE_int(Casenumbers);

#if 1
	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}
#endif

	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
		}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);

}

void file_io::free_data_fancy(int nb_cases,
	int *Set_sizes, int **Sets,
	char **Ago_ascii, char **Aut_ascii,
	int *Casenumbers)
// Frees only those pointers that are not NULL
{
	int i;

	if (Ago_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Ago_ascii[i]);
			}
		FREE_pchar(Ago_ascii);
		}
	if (Aut_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Aut_ascii[i]);
			}
		FREE_pchar(Aut_ascii);
		}
	if (Sets) {
		for (i = 0; i < nb_cases; i++) {
			FREE_int(Sets[i]);
			}
		FREE_pint(Sets);
		}
	if (Set_sizes) {
		FREE_int(Set_sizes);
		}
	if (Casenumbers) {
		FREE_int(Casenumbers);
		}
}

void file_io::read_and_parse_data_file_fancy(
	const char *fname,
	int f_casenumbers,
	int &nb_cases,
	int *&Set_sizes, int **&Sets,
	char **&Ago_ascii,
	char **&Aut_ascii,
	int *&Casenumbers,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char **data;
	int i;

	if (f_v) {
		cout << "read_and_parse_data_file_fancy "
				"reading file "
			<< fname << endl;
		}
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy "
				"before try_to_read_file" << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level - 1)) {
		if (f_vv) {
			cout << "read_and_parse_data_file_fancy "
					"file read containing "
				<< nb_cases << " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file_fancy "
				"couldn't read file fname="
			<< fname << endl;
		exit(1);
		}

#if 0
	if (f_vv) {
		cout << "after try_to_read_file" << endl;
		for (i = 0; i < nb_cases; i++) {
			cout << i << " : " << data[i] << endl;
			}
		}
#endif


	if (f_vv) {
		cout << "read_and_parse_data_file_fancy "
				"parsing sets" << endl;
		}
	parse_sets(nb_cases, data, f_casenumbers,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 2);

	if (f_vv) {
		cout << "read_and_parse_data_file_fancy "
				"freeing temporary data" << endl;
		}
	for (i = 0; i < nb_cases; i++) {
		FREE_char(data[i]);
		}
	FREE_pchar(data);
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy done" << endl;
		}
}

void file_io::read_set_from_file(const char *fname,
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;

	if (f_v) {
		cout << "read_set_from_file opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
		}
	ifstream f(fname);

	f >> set_size;
	if (f_v) {
		cout << "read_set_from_file allocating set of size "
			<< set_size << endl;
		}
	the_set = NEW_int(set_size);

	if (f_v) {
		cout << "read_set_from_file reading set of size "
			<< set_size << endl;
		}
	for (i = 0; i < set_size; i++) {
		f >> a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i
			//<< "-th number is " << a << endl;
			//}
		if (a == -1)
			break;
		the_set[i] = a;
		}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void file_io::write_set_to_file(const char *fname,
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "write_set_to_file opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);

	f << set_size << endl;

	for (i = 0; i < set_size; i++) {
#if 0
		if (i && ((i % 10) == 0)) {
			f << endl;
			}
#endif
		f << the_set[i] << " ";
		}
	f << endl << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
		}
}

void file_io::read_set_from_file_lint(const char *fname,
	long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	long int a;

	if (f_v) {
		cout << "read_set_from_file_lint opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
		}
	ifstream f(fname);

	f >> set_size;
	if (f_v) {
		cout << "read_set_from_file_lint allocating set of size "
			<< set_size << endl;
		}
	the_set = NEW_lint(set_size);

	if (f_v) {
		cout << "read_set_from_file_lint reading set of size "
			<< set_size << endl;
		}
	for (i = 0; i < set_size; i++) {
		f >> a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i
			//<< "-th number is " << a << endl;
			//}
		if (a == -1)
			break;
		the_set[i] = a;
		}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		lint_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void file_io::write_set_to_file_lint(const char *fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "write_set_to_file_lint opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);

	f << set_size << endl;

	for (i = 0; i < set_size; i++) {
#if 0
		if (i && ((i % 10) == 0)) {
			f << endl;
			}
#endif
		f << the_set[i] << " ";
		}
	f << endl << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
		}
}

void file_io::read_set_from_file_int4(const char *fname,
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, b;
	int_4 a;

	if (f_v) {
		cout << "read_set_from_file_int4 opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
		}
	ifstream f(fname, ios::binary);

	f.read((char *) &a, sizeof(int_4));
	set_size = a;
	the_set = NEW_int(set_size);

	for (i = 0; i < set_size; i++) {
		f.read((char *) &a, sizeof(int_4));
		b = a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i
			//<< "-th number is " << a << endl;
			//}
		if (b == -1)
			break;
		the_set[i] = b;
		}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void file_io::read_set_from_file_int8(const char *fname,
	long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	long int b;
	int_8 a;

	if (f_v) {
		cout << "read_set_from_file_int8 opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
		}
	ifstream f(fname, ios::binary);

	f.read((char *) &a, sizeof(int_8));
	set_size = a;
	the_set = NEW_lint(set_size);

	for (i = 0; i < set_size; i++) {
		f.read((char *) &a, sizeof(int_8));
		b = a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i
			//<< "-th number is " << a << endl;
			//}
		if (b == -1)
			break;
		the_set[i] = b;
		}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		lint_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void file_io::write_set_to_file_as_int4(const char *fname,
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int_4 a;
	int b;

	if (f_v) {
		cout << "write_set_to_file_as_int4 opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname, ios::binary);


	a = (int_4) set_size;
	f.write((char *) &a, sizeof(int_4));
	b = a;
	if (b != set_size) {
		cout << "write_set_to_file_as_int4 "
				"data loss regarding set_size" << endl;
		cout << "set_size=" << set_size << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		exit(1);
		}
	for (i = 0; i < set_size; i++) {
		a = (int_4) the_set[i];
		f.write((char *) &a, sizeof(int_4));
		b = a;
		if (b != the_set[i]) {
			cout << "write_set_to_file_as_int4 data loss" << endl;
			cout << "i=" << i << endl;
			cout << "the_set[i]=" << the_set[i] << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	}
	if (f_v) {
		cout << "Written file " << fname
			<< " of size " << file_size(fname) << endl;
		}
}

void file_io::write_set_to_file_as_int8(const char *fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int_8 a, b;

	if (f_v) {
		cout << "write_set_to_file_as_int8 opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname, ios::binary);


	a = (int_8) set_size;
	f.write((char *) &a, sizeof(int_8));
	b = a;
	if (b != set_size) {
		cout << "write_set_to_file_as_int8 "
				"data loss regarding set_size" << endl;
		cout << "set_size=" << set_size << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		exit(1);
		}
	for (i = 0; i < set_size; i++) {
		a = (int_8) the_set[i];
		f.write((char *) &a, sizeof(int_8));
		b = a;
		if (b != the_set[i]) {
			cout << "write_set_to_file_as_int8 data loss" << endl;
			cout << "i=" << i << endl;
			cout << "the_set[i]=" << the_set[i] << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	}
	if (f_v) {
		cout << "Written file " << fname
			<< " of size " << file_size(fname) << endl;
		}
}

void file_io::read_k_th_set_from_file(const char *fname, int k,
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, h;

	if (f_v) {
		cout << "read_k_th_set_from_file opening file "
			<< fname << " of size " << file_size(fname)
			<< " for reading" << endl;
		}
	ifstream f(fname);

	f >> set_size;
	the_set = NEW_int(set_size);

	for (h = 0; h <= k; h++) {
		for (i = 0; i < set_size; i++) {
			f >> a;
			if (f_v) {
				cout << "read_k_th_set_from_file: h="
					<< h << " the " << i
					<< "-th number is " << a << endl;
				}
			//if (a == -1)
				//break;
			the_set[i] = a;
			}
		}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}


void file_io::write_incidence_matrix_to_file(char *fname,
	int *Inc, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_inc;

	if (f_v) {
		cout << "write_incidence_matrix_to_file opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);

	nb_inc = 0;
	for (i = 0; i < m * n; i++) {
		if (Inc[i]) {
			nb_inc++;
			}
		}
	f << m << " " << n << " " << nb_inc << endl;

	for (i = 0; i < m * n; i++) {
		if (Inc[i]) {
			f << i << " ";
			}
		}
	f << " 0" << endl; // no group order

	f << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
		}
}

#define READ_INCIDENCE_BUFSIZE 1000000

void file_io::read_incidence_matrix_from_inc_file(int *&M, int &m, int &n,
	char *inc_file_name, int inc_file_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char buf[READ_INCIDENCE_BUFSIZE];
	char *p_buf;
	int *X = NULL;


	if (f_v) {
		cout << "read_incidence_matrix_from_inc_file "
			<< inc_file_name << " no " << inc_file_idx << endl;
		}
	{
	ifstream f(inc_file_name);

	if (f.eof()) {
		exit(1);
		}
	f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		exit(1);
		}
	sscanf(buf, "%d %d %d", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_int(nb_inc);
	cnt = 0;
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			continue;
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;

		s_scan_int(&p_buf, &a);
		if (f_vv) {
			//cout << cnt << " : " << a << " ";
			}
		if (a == -1) {
			cout << "\nread_incidence_matrix_from_inc_file: "
					"found a complete file with "
				<< cnt << " solutions" << endl;
			break;
			}
		X[0] = a;

		//cout << "reading " << nb_inc << " incidences" << endl;
		for (h = 1; h < nb_inc; h++) {
			s_scan_int(&p_buf, &a);
			if (a < 0 || a >= m * n) {
				cout << "attention, read " << a
					<< " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		if (cnt == inc_file_idx) {
			M = NEW_int(m * n);
			for (h = 0; h < m * n; h++) {
				M[h] = 0;
				}
			for (h = 0; h < nb_inc; h++) {
				M[X[h]] = 1;
				}
			if (f_vv) {
				cout << "read_incidence_matrix_from_inc_file: "
						"found the following incidence matrix:" << endl;
				print_integer_matrix_width(cout,
					M, m, n, n, 1);
				}
			break;
			}
		cnt++;
		}
	}
	FREE_int(X);
}

int file_io::inc_file_get_number_of_geometries(
	char *inc_file_name, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char buf[READ_INCIDENCE_BUFSIZE];
	char *p_buf;
	int *X = NULL;
	int m, n;


	if (f_v) {
		cout << "inc_file_get_number_of_geometries "
			<< inc_file_name << endl;
		}
	{
	ifstream f(inc_file_name);

	if (f.eof()) {
		exit(1);
		}
	f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		exit(1);
		}
	sscanf(buf, "%d %d %d", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_int(nb_inc);
	cnt = 0;
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			continue;
			}

		// check for comment line:
		if (buf[0] == '#')
			continue;

		p_buf = buf;

		s_scan_int(&p_buf, &a);
		if (f_vv) {
			//cout << cnt << " : " << a << " ";
			}
		if (a == -1) {
			cout << "\nread_incidence_matrix_from_inc_file: "
					"found a complete file with " << cnt
					<< " solutions" << endl;
			break;
			}
		X[0] = a;

		//cout << "reading " << nb_inc << " incidences" << endl;
		for (h = 1; h < nb_inc; h++) {
			s_scan_int(&p_buf, &a);
			if (a < 0 || a >= m * n) {
				cout << "attention, read " << a
					<< " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		cnt++;
		}
	}
	FREE_int(X);
	return cnt;
}

long int file_io::file_size(const char *name)
{
	//cout << "file_size fname=" << name << endl;
#ifdef SYSTEMUNIX
	int handle;
	long int size;

	//cout << "Unix mode" << endl;
	handle = open(name, O_RDWR/*mode*/);
	size = lseek(handle, 0L, SEEK_END);
	close(handle);
	return size;
#endif
#ifdef SYSTEMMAC
	int handle;
	long int size;

	//cout << "Macintosh mode" << endl;
	handle = open(name, O_RDONLY);
		/* THINK C Unix Lib */
	size = lseek(handle, 0L, SEEK_END);
		/* THINK C Unix Lib */
	close(handle);
	return size;
#endif
#ifdef SYSTEMWINDOWS

	//cout << "Windows mode" << endl;

	int handle = _open(name, _O_RDONLY);
	int size   = _lseek(handle, 0, SEEK_END);
	close (handle);
	return size;
#endif
}

void file_io::delete_file(const char *fname)
{
	char str[1000];

	sprintf(str, "rm %s", fname);
	system(str);
}

void file_io::fwrite_int4(FILE *fp, int a)
{
	int_4 I;

	I = (int_4) a;
	fwrite(&I, 1 /* size */, 4 /* items */, fp);
}

int_4 file_io::fread_int4(FILE *fp)
{
	int_4 I;

	fread(&I, 1 /* size */, 4 /* items */, fp);
	return I;
}

void file_io::fwrite_uchars(FILE *fp, uchar *p, int len)
{
	fwrite(p, 1 /* size */, len /* items */, fp);
}

void file_io::fread_uchars(FILE *fp, uchar *p, int len)
{
	fread(p, 1 /* size */, len /* items */, fp);
}

void file_io::read_numbers_from_file(const char *fname,
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	double d;

	if (f_v) {
		cout << "read_numbers_from_file opening file " << fname
				<< " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);

	set_size = 1000;
	the_set = NEW_int(set_size);

	for (i = 0; TRUE; i++) {
		if (f.eof()) {
			break;
			}
		f >> d;
		a = (int) d;
		if (f_vv) {
			cout << "read_set_from_file: the " << i
					<< "-th number is " << d << " which becomes "
					<< a << endl;
			}
		if (a == -1)
			break;
		the_set[i] = a;
		if (i >= set_size) {
			cout << "i >= set_size" << endl;
			exit(1);
			}
		}
	set_size = i;
	if (f_v) {
		cout << "read a set of size " << set_size
				<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void file_io::read_ascii_set_of_sets_constant_size(
		const char *fname_ascii,
		int *&Sets, int &nb_sets, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "file_io::read_ascii_set_of_sets_constant_size "
				"reading ascii file " << fname_ascii << endl;
		}
	sorting Sorting;
	int N;
	int i;

	N = count_number_of_lines_in_file(fname_ascii,
			0 /* verbose_level */);


	{
		if (f_v) {
			cout << "file_io::read_ascii_set_of_sets_constant_size "
					"Reading file " << fname_ascii << " of size "
					<< file_size(fname_ascii) << ":" << endl;
		}
		ifstream fp(fname_ascii);

		int nb;




		nb_sets = 0;
		while (TRUE) {
			fp >> nb;
			if (nb == -1) {
				break;
			}

			if (f_v) {
				cout << "file_io::read_ascii_set_of_sets_constant_size "
						"set " << nb_sets << ":";
			}

			if (nb_sets == 0) {
				set_size = nb;
				Sets = NEW_int(N * set_size);
			}
			else {
				if (nb != set_size) {
					cout << "file_io::read_ascii_set_of_sets_constant_size "
							"nb != set_size" << endl;
					exit(1);
				}
			}
			for (i = 0; i < set_size; i++) {
				fp >> Sets[nb_sets * set_size + i];
			}

			Sorting.int_vec_heapsort(Sets + nb_sets * set_size, set_size);

			if (f_v) {
				cout << "file_io::read_ascii_set_of_sets_constant_size "
						"set " << nb_sets << " / " << N << " is ";
				int_vec_print(cout, Sets + nb_sets * set_size, set_size);
				cout << endl;
			}
			nb_sets++;
		}
	}
	if (f_v) {
		cout << "file_io::read_ascii_set_of_sets_constant_size "
				"We found " << nb_sets << " sets" << endl;
	}

#if 0
	cout << "writing spreads to file " << fname_spreads << endl;
	Fio.int_matrix_write_csv(fname_spreads, Spreads, nb_spreads,
			P->spread_size);

	cout << "Written file " << fname_spreads << " of size "
			<< Fio.file_size(fname_spreads) << endl;
	FREE_int(Spreads);
#endif
	if (f_v) {
		cout << "file_io::read_ascii_set_of_sets_constant_size "
				"reading ascii file " << fname_ascii << " done" << endl;
		}
}



}}


