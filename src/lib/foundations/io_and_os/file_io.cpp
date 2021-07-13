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
	int &cnt_total,
	vector<int> missing_idx,
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

	//missing_idx = NEW_int(N);
	buf = NEW_char(MY_OWN_BUFSIZE);
	cnt_total = 0;
	{
		ofstream fp_out(fname_out);
		for (h = 0; h < N; h++) {
			sprintf(fname, fname_in_mask, h);
			if (file_size(fname) < 0) {
				cout << "concatenate_files input file does not exist: " << fname << " skipping" << endl;
				//missing_idx[nb_missing++] = h;
				missing_idx.push_back(h);
			}
			else {
				ifstream fp(fname);

				cnt = 0;
				while (TRUE) {
					if (fp.eof()) {
						cout << "Encountered End-of-file without having seem EOF "
								"marker, perhaps the file is corrupt. "
								"I was trying to read the file " << fname << endl;
						missing_idx.push_back(h);
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
				cnt_total += cnt;
			}
		} // next h
		fp_out << EOF_marker << " " << cnt_total << endl;
	}
	cout << "Written file " << fname_out << " of size "
		<< file_size(fname_out) << endl;
	FREE_char(buf);
	cout << "There are " << missing_idx.size() << " missing files, they are:" << endl;
	for (h = 0; h < (int) missing_idx.size(); h++) {
		sprintf(fname, fname_in_mask, missing_idx[h]);
		cout << h << " : " << missing_idx[h] << " : " << fname << endl;
	}

	if (f_v) {
		cout << "concatenate_files done" << endl;
	}
}

void file_io::concatenate_files_into(const char *fname_in_mask, int N,
	ofstream &fp_out, const char *EOF_marker, int f_title_line,
	int &cnt_total,
	vector<int> &missing_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	char *buf;
	int h, cnt;

	if (f_v) {
		cout << "file_io::concatenate_files_into " << fname_in_mask
			<< " N=" << N << " into an open file" << endl;
	}

	//missing_idx = NEW_int(N);
	buf = NEW_char(MY_OWN_BUFSIZE);
	cnt_total = 0;
	{
		//ofstream fp_out(fname_out);
		for (h = 0; h < N; h++) {
			sprintf(fname, fname_in_mask, h);

			fp_out << "# start of file " << fname << endl;

			if (file_size(fname) < 0) {
				cout << "file_io::concatenate_files_into "
						"input file does not exist: " << fname << " skipping" << endl;
				//missing_idx[nb_missing++] = h;
				missing_idx.push_back(h);
			}
			else {
				ifstream fp(fname);

				cnt = 0;
				while (TRUE) {
					if (fp.eof()) {
						cout << "Encountered End-of-file without having seem EOF "
								"marker, perhaps the file is corrupt. "
								"I was trying to read the file " << fname << endl;
						missing_idx.push_back(h);
						break;
					}

					fp.getline(buf, MY_OWN_BUFSIZE, '\n');
					//cout << "Read: " << buf << endl;
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
				cnt_total += cnt;
			}
			fp_out << "# end of file " << fname << endl;
		} // next h
		//fp_out << EOF_marker << " " << cnt_total << endl;
	}
	FREE_char(buf);
	cout << "There are " << missing_idx.size() << " missing files, they are:" << endl;
	for (h = 0; h < (int) missing_idx.size(); h++) {
		sprintf(fname, fname_in_mask, missing_idx[h]);
		cout << h << " : " << missing_idx[h] << " : " << fname << endl;
	}

	if (f_v) {
		cout << "file_io::concatenate_files_into done" << endl;
	}
}

void file_io::poset_classification_read_candidates_of_orbit(
	std::string &fname, int orbit_at_level,
	long int *&candidates, int &nb_candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb, cand_first, i;


	if (f_v) {
		cout << "file_io::poset_classification_read_candidates_of_orbit" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
	}

	if (file_size(fname) <= 0) {
		cout << "file_io::poset_classification_read_candidates_of_orbit file "
				<< fname << " does not exist" << endl;
		exit(1);
	}

	{
		ifstream fp(fname, ios::binary);
		fp.read((char *) &nb, sizeof(int));
		if (orbit_at_level >= nb) {
			cout << "file_io::poset_classification_read_candidates_of_orbit "
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
		candidates = NEW_lint(nb_candidates);

		int *candidates0;
		candidates0 = NEW_int(nb_candidates);
		fp.seekg((1 + nb * 2 + cand_first) * sizeof(int), ios::beg);
		for (i = 0; i < nb_candidates; i++) {
			fp.read((char *) &candidates0[i], sizeof(int));
			candidates[i] = candidates0[i];
		}
		FREE_int(candidates0);

	}

	if (f_v) {
		cout << "file_io::poset_classification_read_candidates_of_orbit done" << endl;
	}
}


void file_io::read_candidates_for_one_orbit_from_file(std::string &prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		long int *S,
		void (*early_test_func_callback)(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		long int *&candidates,
		int &nb_candidates,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, orbit_idx;
	long int *candidates1 = NULL;
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
	string fname2;

	char str[1000];
	fname2.assign(prefix);
	sprintf(str, "_lvl_%d_candidates.bin", level_of_candidates_file);
	fname2.append(str);

	poset_classification_read_candidates_of_orbit(
		fname2, orbit_idx,
		candidates1, nb_candidates1, verbose_level - 1);


	for (h = level_of_candidates_file; h < level; h++) {

		long int *candidates2;
		int nb_candidates2;

		if (f_v) {
			cout << "read_orbit_rep_and_candidates_from_files_"
					"and_process testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
		}
		candidates2 = NEW_lint(nb_candidates1);

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

		Orbiter->Lint_vec.copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_lint(candidates2);
	}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "read_candidates_for_one_orbit_from_file done" << endl;
	}
}



int file_io::find_orbit_index_in_data_file(std::string &prefix,
		int level_of_candidates_file, long int *starter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	char str[1000];
	int orbit_idx;

	if (f_v) {
		cout << "find_orbit_index_in_data_file" << endl;
	}

	fname.assign(prefix);
	sprintf(str, "_lvl_%d", level_of_candidates_file);
	fname.append(str);

	if (file_size(fname) <= 0) {
		cout << "find_orbit_index_in_data_file file "
				<< fname << " does not exist" << endl;
		exit(1);
	}
	ifstream f(fname);
	string_tools ST;
	int a, i, cnt;
	long int *S;
	char buf[MY_OWN_BUFSIZE];
	int len, str_len;
	char *p_buf;

	S = NEW_lint(level_of_candidates_file);

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
		ST.s_scan_int(&p_buf, &a);
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
			ST.s_scan_lint(&p_buf, &S[i]);
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
	FREE_lint(S);
	if (f_v) {
		cout << "find_orbit_index_in_data_file done" << endl;
	}
	return orbit_idx;
}


void file_io::write_exact_cover_problem_to_file(int *Inc,
		int nb_rows, int nb_cols, std::string &fname)
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

void file_io::read_solution_file(std::string &fname,
	int *Inc, int nb_rows, int nb_cols,
	int *&Solutions, int &sol_length, int &nb_sol,
	int verbose_level)
// sol_length must be constant
{
	int f_v = (verbose_level >= 1);
	int nb, nb_max, i, j, a, nb_sol1;
	int *x, *y;
	string_tools ST;

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
				if (buf[0] == '#') {
					continue;
				}
				for (j = 0; j < nb_cols; j++) {
					x[j] = 0;
				}
				ST.s_scan_int(&p_buf, &nb);
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
						Orbiter->Int_vec.print_fully(cout, y, nb_rows);
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
				ST.s_scan_int(&p_buf, &nb);
				//cout << "buf='" << buf << "' nb=" << nb << endl;

				for (i = 0; i < sol_length; i++) {
					ST.s_scan_int(&p_buf, &a);
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
		std::string &fname,
	int &nb_solutions, int &solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	string_tools ST;
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
		int line_number = 1;


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
						"line " << line_number << " empty line" << endl;
				exit(1);
			}
			if (buf[0] == '#') {
				line_number++;
				continue;
			}

			p_buf = buf;
			ST.s_scan_int(&p_buf, &s);
			if (solution_size == -1) {
				solution_size = s;
			}
			else {
				if (s != -1) {
					if (solution_size != s) {
						cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size "
								"solution_size is not constant" << endl;
						cout << "solution_size=" << solution_size << endl;
						cout << "s=" << s << endl;
						cout << "line " << line_number << endl;
						exit(1);
					}
				}
			}

			if (strncmp(buf, "-1", 2) == 0) {
				break;
			}
			nb_solutions++;
			line_number++;
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
	}
}

void file_io::count_number_of_solutions_in_file(std::string &fname,
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
			if (buf[0] == '#') {
				continue;
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

void file_io::count_number_of_solutions_in_file_by_case(std::string &fname,
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


void file_io::read_solutions_from_file_and_get_solution_size(std::string &fname,
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

	string_tools ST;
	char *buf;
	char *p_buf;
	int i, a, nb_sol;

	buf = NEW_char(MY_OWN_BUFSIZE);
	nb_sol = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, MY_OWN_BUFSIZE, '\n');
			if (strlen(buf) && buf[0] == '#') {
				continue;
			}
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			ST.s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
			}
			if (a != solution_size) {
				cout << "file_io::read_solutions_from_file_and_get_solution_size "
						"a != solution_size" << endl;
				exit(1);
			}
			for (i = 0; i < solution_size; i++) {
				ST.s_scan_int(&p_buf, &a);
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


void file_io::read_solutions_from_file(std::string &fname,
	int &nb_solutions, int *&Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int i, a, nb_sol;
	string_tools ST;

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
			ST.s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
			}
			if (a != solution_size) {
				cout << "read_solutions_from_file "
						"a != solution_size" << endl;
				exit(1);
			}
			for (i = 0; i < solution_size; i++) {
				ST.s_scan_int(&p_buf, &a);
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


void file_io::read_solutions_from_file_size_is_known(std::string &fname,
	std::vector<std::vector<int> > &Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	vector<int> one_solution;
	string_tools ST;
	int i, a;

	if (f_v) {
		cout << "read_solutions_from_file_size_is_known" << endl;
		cout << "read_solutions_from_file_size_is_known trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file_size_is_known solution_size="
			<< solution_size << endl;
	}

	if (file_size(fname.c_str()) < 0) {
		cout << "file_io::read_solutions_from_file_size_is_known the file " << fname << " does not exist" << endl;
		return;
	}

	buf = NEW_char(MY_OWN_BUFSIZE);

	one_solution.resize(solution_size);

	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, MY_OWN_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			ST.s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
			}

			one_solution[0] = a;
			for (i = 1; i < solution_size; i++) {
				ST.s_scan_int(&p_buf, &a);
				one_solution[i] = a;
			}
			Solutions.push_back(one_solution);
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "read_solutions_from_file_size_is_known done" << endl;
	}
}


void file_io::read_solutions_from_file_by_case(std::string &fname,
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
	string_tools ST;

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
					ST.s_scan_int(&p_buf, &sz);
					if (sz != solution_size) {
						cout << "read_solutions_from_file_by_case "
								"sz != solution_size" << endl;
						exit(1);
					}
					for (i = 0; i < sz; i++) {
						ST.s_scan_int(&p_buf, &a);
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

void file_io::copy_file_to_ostream(ostream &ost, const char *fname)
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
	std::string &fname, const char *label)
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

void file_io::lint_vec_write_csv(long int *v, int len,
		std::string &fname, const char *label)
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
		std::string &fname, const char *label1, const char *label2)
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
		std::string &fname,
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
		std::string &fname, const char **column_label)
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

void file_io::lint_vec_array_write_csv(int nb_vecs, long int **Vec, int len,
		std::string &fname, const char **column_label)
{
	int i, j;

	cout << "lint_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
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

void file_io::int_matrix_write_csv(std::string &fname, int *M, int m, int n)
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

void file_io::lint_matrix_write_csv(std::string &fname, long int *M, int m, int n)
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

void file_io::lint_matrix_write_csv_override_headers(std::string &fname, std::string *headers, long int *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "Row";
		for (j = 0; j < n; j++) {
			f << "," << headers[j];
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
		std::string &fname, double *M, int m, int n)
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

void file_io::int_matrix_write_csv_with_labels(std::string &fname,
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

void file_io::lint_matrix_write_csv_with_labels(std::string &fname,
	long int *M, int m, int n, const char **column_label)
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

void file_io::int_matrix_read_csv(std::string &fname,
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
		cout << "The spreadsheet has " << S.nb_cols << " columns" << endl;
		M = NEW_int(m * n);
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				char *p;
				p = S.get_string(i + 1, j + 1);
				a = my_atoi(p);
				FREE_char(p);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "int_matrix_read_csv done" << endl;
	}
}

void file_io::lint_matrix_read_csv(std::string &fname,
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
				char *p;

				p = S.get_string(i + 1, j + 1);
				a = my_atol(p);
				FREE_char(p);
				M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "lint_matrix_read_csv done" << endl;
	}

}

void file_io::double_matrix_read_csv(std::string &fname,
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
				char *p;
				p = S.get_string(i + 1, j + 1);
				sscanf(p, "%lf", &d);
				FREE_char(p);
				M[i * n + j] = d;
			}
		}
	}
	if (f_v) {
		cout << "double_matrix_read_csv done" << endl;
	}
}

void file_io::int_matrix_write_cas_friendly(std::string &fname, int *M, int m, int n)
{
	int i, j;

	{
		ofstream f(fname);

		f << "[";
		for (i = 0; i < m; i++) {
			f << "[";
			for (j = 0; j < n; j++) {
				f << M[i * n + j];
				if (j < n - 1) {
					f << ", ";
				}
			}
			f << "]";
			if (i < m - 1) {
				f << ", " << endl;
			}
		}
		f << "]" << endl;
	}
}

void file_io::int_matrix_write_text(std::string &fname, int *M, int m, int n)
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

void file_io::lint_matrix_write_text(std::string &fname, long int *M, int m, int n)
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

void file_io::int_matrix_read_text(std::string &fname, int *&M, int &m, int &n)
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
	int *&Set_sizes, long int **&Sets,
	char **&Ago_ascii, char **&Aut_ascii,
	int *&Casenumbers,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, casenumber;
	char *ago_ascii, *aut_ascii;
	char *p_buf;
	string_tools ST;

	if (f_v) {
		cout << "parse_sets f_casenumbers=" << f_casenumbers
			<< " nb_cases = " << nb_cases << endl;
	}

	ago_ascii = NEW_char(MY_OWN_BUFSIZE);
	aut_ascii = NEW_char(MY_OWN_BUFSIZE);

	Set_sizes = NEW_int(nb_cases);
	Sets = NEW_plint(nb_cases);
	Ago_ascii = NEW_pchar(nb_cases);
	Aut_ascii = NEW_pchar(nb_cases);
	Casenumbers = NEW_int(nb_cases);

	for (h = 0; h < nb_cases; h++) {

		//cout << h << " : ";
		//cout << " : " << data[h] << endl;

		p_buf = data[h];
		if (f_casenumbers) {
			ST.s_scan_int(&p_buf, &casenumber);
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
	long int *&set, char *ago_ascii, char *aut_ascii)
{
	int i;
	char *p_buf;
	string_tools ST;

	//cout << "parse_line: " << line << endl;
	p_buf = line;
	ST.s_scan_int(&p_buf, &len);
	//cout << "parsing data of length " << len << endl;
	set = NEW_lint(len);
	for (i = 0; i < len; i++) {
		ST.s_scan_lint(&p_buf, &set[i]);
	}
	ST.s_scan_token(&p_buf, ago_ascii);
	if (strcmp(ago_ascii, "1") == 0) {
		aut_ascii[0] = 0;
	}
	else {
		ST.s_scan_token(&p_buf, aut_ascii);
	}
}


int file_io::count_number_of_orbits_in_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf, *p_buf;
	int nb_sol, len;
	int ret;
	string_tools ST;

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
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;
			ST.s_scan_int(&p_buf, &len);
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

int file_io::count_number_of_lines_in_file(std::string &fname, int verbose_level)
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

int file_io::try_to_read_file(std::string &fname,
	int &nb_cases, char **&data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int n1;
	char *buf, *p_buf;
	int nb_sol, len, a;
	string_tools ST;

	if (f_v) {
		cout << "try_to_read_file trying to read file " << fname
			<< " of size " << file_size(fname) << endl;
	}
	buf = NEW_char(MY_OWN_BUFSIZE);


	if (file_size(fname.c_str()) <= 0) {
		goto return_false;
	}

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
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;
			ST.s_scan_int(&p_buf, &len);
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
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;
			ST.s_scan_int(&p_buf, &a);
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
	std::string &fname, int &nb_cases,
	char **&data, long int **&sets, int *&set_sizes,
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
	char **data, long int **&sets)
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
	int *Set_sizes, long int **Sets,
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
			FREE_lint(Sets[i]);
		}
		FREE_plint(Sets);
	}
	if (Set_sizes) {
		FREE_int(Set_sizes);
	}
	if (Casenumbers) {
		FREE_int(Casenumbers);
	}
}

void file_io::read_and_parse_data_file_fancy(
		std::string &fname,
	int f_casenumbers,
	int &nb_cases,
	int *&Set_sizes, long int **&Sets,
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

void file_io::read_set_from_file(std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
// if the file is empty, set_size cannot be determined and is set to 0
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;

	if (f_v) {
		cout << "file_io::read_set_from_file opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
	}
	ifstream f(fname);

	f >> set_size;

	if (set_size == -1) {
		if (f_v) {
			cout << "file_io::read_set_from_file the file is empty, "
					"set_size cannot be determined" << endl;
		}
		set_size = 0;
	}
	else {
		if (f_v) {
			cout << "file_io::read_set_from_file allocating set of size "
				<< set_size << endl;
		}
		the_set = NEW_lint(set_size);

		if (f_v) {
			cout << "file_io::read_set_from_file reading set of size "
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
			cout << "file_io::read_set_from_file read a set of size " << set_size
				<< " from file " << fname << endl;
		}
		if (f_vv) {
			cout << "file_io::read_set_from_file the set is:" << endl;
			Orbiter->Lint_vec.print(cout, the_set, set_size);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "file_io::read_set_from_file done" << endl;
	}
}

void file_io::write_set_to_file(std::string &fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "write_set_to_file opening file "
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);

	f << set_size << " ";

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

void file_io::read_set_from_file_lint(std::string &fname,
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
		if (a == -1) {
			break;
		}
		the_set[i] = a;
	}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "the set is:" << endl;
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::write_set_to_file_lint(std::string &fname,
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

void file_io::read_set_from_file_int4(std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
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
	the_set = NEW_lint(set_size);

	for (i = 0; i < set_size; i++) {
		f.read((char *) &a, sizeof(int_4));
		b = a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i
			//<< "-th number is " << a << endl;
			//}
		if (b == -1) {
			break;
		}
		the_set[i] = b;
	}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "the set is:" << endl;
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::read_set_from_file_int8(std::string &fname,
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
		if (b == -1) {
			break;
		}
		the_set[i] = b;
	}
	if (f_v) {
		cout << "read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "the set is:" << endl;
		Orbiter->Lint_vec.print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::write_set_to_file_as_int4(std::string &fname,
	long int *the_set, int set_size, int verbose_level)
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

void file_io::write_set_to_file_as_int8(std::string &fname,
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

void file_io::read_k_th_set_from_file(std::string &fname, int k,
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
		Orbiter->Int_vec.print(cout, the_set, set_size);
		cout << endl;
	}
}


void file_io::write_incidence_matrix_to_file(std::string &fname,
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
		std::string &inc_file_name, int inc_file_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char buf[READ_INCIDENCE_BUFSIZE];
	char *p_buf;
	int *X = NULL;
	string_tools ST;


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
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;

			ST.s_scan_int(&p_buf, &a);
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
				ST.s_scan_int(&p_buf, &a);
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
					Orbiter->Int_vec.print_integer_matrix_width(cout,
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
	string_tools ST;


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
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;

			ST.s_scan_int(&p_buf, &a);
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
				ST.s_scan_int(&p_buf, &a);
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

long int file_io::file_size(std::string &fname)
{
	return file_size(fname.c_str());
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

	cout << "file_io::file_size SYSTEMWINDOWS but not SYSTEMUNIX" << endl;
	exit(1);

	//int handle = _open(name, _O_RDONLY);
	//int size   = _lseek(handle, 0, SEEK_END);
	//close (handle);
	//return size;
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

void file_io::read_numbers_from_file(std::string &fname,
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
		if (a == -1) {
			break;
		}
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
		Orbiter->Int_vec.print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::read_ascii_set_of_sets_constant_size(
		std::string &fname_ascii,
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
				Orbiter->Int_vec.print(cout, Sets + nb_sets * set_size, set_size);
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

void file_io::write_decomposition_stack(char *fname, int m, int n, int *v, int *b, int *aij, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::write_widor" << endl;
	}
	{
		ofstream f(fname);
		int i, j;


		f << "<HTDO type=pt ptanz=" << m << " btanz=" << n << " fuse=simple>" << endl;
		f << "        ";
		for (j = 0; j < n; j++) {
			f << setw(8) << b[j] << " ";
			}
		f << endl;
		for (i = 0; i < m; i++) {
			f << setw(8) << v[i];
			for (j = 0; j < n; j++) {
				f << setw(8) << aij[i * n + j] << " ";
				}
			f << endl;
			}
		f << endl;
		for (i = 0; i < m; i++) {
			f << setw(3) << 1;
			}
		f << endl;
		f << "</HTDO>" << endl;
	}

	if (f_v) {
		cout << "file_io::write_decomposition_stack done" << endl;
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
	}
}

void file_io::create_file(create_file_description *Descr, int verbose_level)
{
	file_io Fio;
	int j;

	if (Descr->f_read_cases) {

		cout << "Descr->f_read_cases" << endl;
		string fname;
		char str[1000];
		//int *Cases;
		int nb_cases;
		int n, c;

		cout << "reading file " << Descr->read_cases_fname << endl;


		spreadsheet S;

		S.read_spreadsheet(Descr->read_cases_fname, 0/*verbose_level - 1*/);

		nb_cases = S.nb_rows;
		n = S.nb_cols;

#if 0
		Fio.int_matrix_read_csv(Descr->read_cases_fname,
				Cases, nb_cases, n, 0 /* verbose_level */);
#endif

		cout << "nb_cases = " << nb_cases << endl;
		cout << "n = " << n << endl;
		if (n != 1) {
			cout << "read cases, n != 1" << endl;
			exit(1);
		}
#if 0
		cout << "We found " << nb_cases << " cases to do:" << endl;
		int_vec_print(cout, Cases, nb_cases);
		cout << endl;
#endif

		const char *log_fname = "log_file.txt";
		const char *log_mask = "\tsbatch job%03d";
		{
		ofstream fp_log(log_fname);

		for (c = 0; c < nb_cases; c++) {

			//i = Cases[c];
			sprintf(str, Descr->file_mask.c_str(), c);
			fname.assign(str);


			{
				ofstream fp(fname);

				for (j = 0; j < Descr->nb_lines; j++) {
					if (Descr->f_line_numeric[j]) {
						sprintf(str, Descr->lines[j].c_str(), c);
					}
					else {
						sprintf(str, Descr->lines[j].c_str(), S.get_string(c, 0));
					}
					fix_escape_characters(str);
					fp << str << endl;
				}
			}
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;

			char log_entry[1000];

			sprintf(log_entry, log_mask, c);
			fp_log << log_entry << endl;
			}
		}
		cout << "Written file " << log_fname << " of size "
				<< Fio.file_size(log_fname) << endl;
	}
	else if (Descr->f_read_cases_text) {
		cout << "read_cases_text" << endl;

		if (!Descr->f_N) {
			cout << "please use option -N <N>" << endl;
			exit(1);
		}
		if (!Descr->f_command) {
			cout << "please use option -command <command>" << endl;
			exit(1);
		}

		cout << "Reading file " << Descr->read_cases_fname << endl;

		spreadsheet *S;
		int row;

		S = NEW_OBJECT(spreadsheet);
		S->read_spreadsheet(Descr->read_cases_fname, 0 /*verbose_level*/);

		cout << "Read spreadsheet with " << S->nb_rows << " rows" << endl;

		//S->print_table(cout, FALSE /* f_enclose_in_parentheses */);
		for (row = 0; row < MINIMUM(10, S->nb_rows); row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row,
					FALSE /* f_enclose_in_parentheses */, cout);
		}
		cout << "..." << endl;
		for (row = MAXIMUM(S->nb_rows - 10, 0); row < S->nb_rows; row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row,
					FALSE /* f_enclose_in_parentheses */, cout);
		}



		create_files_list_of_cases(S, Descr, verbose_level);

	}
	else if (Descr->f_N) {
		create_files(Descr, verbose_level);
	}

}

void file_io::fix_escape_characters(char *str)
{
	int i, j, l;

	l = strlen(str);
	for (i = 0, j = 0; i < l; i++, j++) {
		if (str[i] == '\\' && i < l - 1 && str[i + 1] == 't') {
			str[j] = '\t';
			i++;
		}
		else if (str[i] == '\\' && i < l - 1 && str[i + 1] == 'D') {
			str[j] = '$';
			i++;
		}
		else if (str[i] == '\\' && i < l - 1 && str[i + 1] == 'B') {
			str[j] = '\\';
			i++;
		}
		else if (str[i] == '\\' && i < l - 1 && str[i + 1] == 'n') {
			str[j] = '\n';
			i++;
		}
		else {
			str[j] = str[i];
		}
	}
	str[j] = 0;
}

void file_io::create_files(create_file_description *Descr,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	file_io Fio;

	string fname;
	char str[1000];
	int r;

	if (f_v) {
		cout << "file_io::create_files" << endl;
	}

	const char *makefile_fname = "makefile_submit";
	{
	ofstream fp_makefile(makefile_fname);

	for (i = 0; i < Descr->N; i++) {

		sprintf(str, Descr->file_mask.c_str(), i);
		fname.assign(str);

		fp_makefile << "\tsbatch " << fname << endl;
		{
			ofstream fp(fname);

			for (j = 0; j < Descr->nb_lines; j++) {


				cout << "mask='" << Descr->lines[j].c_str() << "'" << endl;
				sprintf(str, Descr->lines[j].c_str(), i, i, i, i, i, i, i, i);


				fix_escape_characters(str);
				cout << "str='" << str << "'" << endl;
				fp << str << endl;
			}
			if (Descr->f_repeat) {
				if (Descr->f_split) {
					for (r = 0; r < Descr->split_m; r++) {
						for (j = 0; j < Descr->repeat_N; j++) {
							if ((j % Descr->split_m) == r) {
								sprintf(str, Descr->repeat_mask.c_str(), j);
								fix_escape_characters(str);
								fp << str << endl;
							}
						}
						fp << endl;
					}
				}
				else {
					int c;

					sprintf(str, Descr->repeat_mask.c_str(), Descr->repeat_N);
					fix_escape_characters(str);
					fp << str << endl;
					if (!Descr->f_command) {
						cout << "please use option -command when using -repeat" << endl;
						exit(1);
					}
					for (j = 0; j < Descr->repeat_N; j++) {
						c = Descr->repeat_start + j * Descr->repeat_increment;
						sprintf(str, Descr->command.c_str(), c, c, c, c);
						fix_escape_characters(str);
						fp << str << endl;
					}
				}
				for (j = 0; j < Descr->nb_final_lines; j++) {
					fp << Descr->final_lines[j] << endl;
				}
			}
		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		}

	}
	cout << "Written file " << makefile_fname << " of size "
			<< Fio.file_size(makefile_fname) << endl;


	if (f_v) {
		cout << "file_io::create_files done" << endl;
	}
}

void file_io::create_files_list_of_cases(spreadsheet *S,
		create_file_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	string fname;
	char str[1000];
	file_io Fio;

	if (f_v) {
		cout << "file_io::create_files_list_of_cases" << endl;
	}

	int nb_cases = S->nb_rows - 1;
	cout << "nb_cases=" << nb_cases << endl;


	const char *makefile_fname = "makefile_submit";
	const char *fname_submit_script = "submit_jobs.sh";
	{
		ofstream fp_makefile(makefile_fname);
		ofstream fp_submit_script(fname_submit_script);

		fp_submit_script << "#!/bin/bash" << endl;
		for (i = 0; i < Descr->N; i++) {

			sprintf(str, Descr->file_mask.c_str(), i);
			fname.assign(str);

			fp_makefile << "\tsbatch " << fname << endl;
			fp_submit_script << "sbatch " << fname << endl;
			{
				ofstream fp(fname);

				for (j = 0; j < Descr->nb_lines; j++) {
					sprintf(str, Descr->lines[j].c_str(), i, i, i, i, i, i, i, i);
					fix_escape_characters(str);
					fp << str << endl;
				}

				if (Descr->f_tasks) {
					char str[1000];
					int t;
					//int NT;

					sprintf(str, Descr->tasks_line.c_str(), Descr->nb_tasks);
					fp << str << endl;
					//NT = Descr->N * Descr->nb_tasks;
					for (t = 0; t < Descr->nb_tasks; t++) {
						sprintf(str, Descr->command.c_str(), i, t, i, t);
						fix_escape_characters(str);
						fp << str; // << " \\" << endl;
						for (j = 0; j < nb_cases; j++) {
							if ((j % Descr->N) != i) {
								continue;
							}
							if (((j - i) / Descr->N) % Descr->nb_tasks != t) {
								continue;
							}
							char *entry;
							//int case_number;

							//case_number = S->get_int(j + 1, Descr->read_cases_column_of_case);
							entry = S->get_string(j + 1, Descr->read_cases_column_of_fname);
							fp << /* case_number << " " <<*/ entry;

							if (j < nb_cases - Descr->N) {
								fp << ", "; // << endl;
							}
							else {
								fp << ")\"\\" << endl;
							}
						}
						fp << " & " << endl;
						//fp << "\t\t" << -1 << " &" << endl;
					}
				} // if
				else {
					sprintf(str, Descr->command.c_str(), i);
					fix_escape_characters(str);
					fp << str << " \\" << endl;
					//fp << command << " \\" << endl;
					for (j = 0; j < nb_cases; j++) {
						if ((j % Descr->N) != i) {
							continue;
						}
						char *entry;
						//int case_number;

						//case_number = S->get_int(j + 1, Descr->read_cases_column_of_case);
						entry = S->get_string(j + 1, Descr->read_cases_column_of_fname);
						fp <<  "\t\t" /*<< case_number << " "*/ << entry << " \\" << endl;
#if 0
						if (j < nb_cases - N) {
							fp << ", "; // << endl;
						}
						else {
							fp << ")\"\\" << endl;
						}
#endif
					}
					fp << " & " << endl;
					//fp << "\t\t" << -1 << " &" << endl;
				} // else

				for (j = 0; j < Descr->nb_final_lines; j++) {
					sprintf(str, Descr->final_lines[j].c_str(), i, i, i, i, i, i, i, i);
					fix_escape_characters(str);
					fp << str << endl;
				} // next j

			} // close fp(fname)

			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;

		} // next i

	}
	cout << "Written file " << makefile_fname << " of size "
			<< Fio.file_size(makefile_fname) << endl;

	const char *mask_submit_script_piecewise = "submit_jobs_%d.sh";
	char fname_submit_piecewise[1000];
	char cmd[2000];
	int h;
	int N1 = 128;

	for (h = 0; h < Descr->N / N1; h++) {
		sprintf(fname_submit_piecewise, mask_submit_script_piecewise, h * N1);
		{
			ofstream fp_submit_script(fname_submit_piecewise);

			fp_submit_script << "#!/bin/bash" << endl;
			for (i = 0; i < N1; i++) {

				sprintf(str, Descr->file_mask.c_str(), h * N1 + i);
				fname.assign(str);

				fp_submit_script << "sbatch " << fname;
				if (i < N1 - 1) {
					fp_submit_script << "; ";
				}
				else {
					fp_submit_script << endl;
				}
			}
		}
		cout << "Written file " << fname_submit_piecewise << " of size "
			<< Fio.file_size(fname_submit_piecewise) << endl;
		sprintf(cmd, "chmod +x %s", fname_submit_piecewise);
		system(cmd);
	}
	if (f_v) {
		cout << "file_io::create_files_list_of_cases done" << endl;
	}
}

int file_io::number_of_vertices_in_colored_graph(std::string &fname, int verbose_level)
{
	colored_graph CG;

	CG.load(fname, verbose_level);

	return CG.nb_points;
}

void file_io::do_csv_file_select_rows(std::string &fname,
		std::string &rows_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_select_rows" << endl;
	}
	int *Rows;
	int nb_rows;
	string_tools ST;

	Orbiter->Int_vec.scan(rows_text, Rows, nb_rows);

	spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;



	string fname_out;

	fname_out.assign(fname);
	ST.chop_off_extension(fname_out);
	fname_out.append("_select.csv");

	{
		ofstream ost(fname_out);
		ost << "Row,";
		S.print_table_row(0, FALSE, ost);
		for (i = 0; i < nb_rows; i++) {
			ost << i << ",";
			S.print_table_row(Rows[i] + 1, FALSE, ost);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;

	FREE_int(Rows);

	if (f_v) {
		cout << "file_io::do_csv_file_select_rows done" << endl;
	}
}

void file_io::do_csv_file_split_rows_modulo(std::string &fname,
		int split_modulo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_split_rows_modulo" << endl;
	}
	string_tools ST;
	spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i, I;





	for (I = 0; I < split_modulo; I++) {

		string fname_out;
		char str[1000];

		fname_out.assign(fname);
		ST.chop_off_extension(fname_out);
		sprintf(str, "_split_%d_mod_%d.csv", I, split_modulo);
		fname_out.append(str);

		{
			ofstream ost(fname_out);
			S.print_table_row(0, FALSE, ost);
			for (i = 0; i < S.nb_rows - 1; i++) {
				if ((i % split_modulo) != I) {
					continue;
				}
				//ost << i << ",";
				S.print_table_row(i + 1, FALSE, ost);
				}
			ost << "END" << endl;
		}
		cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;
	}

	if (f_v) {
		cout << "file_io::do_csv_file_split_rows_modulo done" << endl;
	}
}

void file_io::do_csv_file_select_cols(std::string &fname,
		std::string &cols_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_select_cols" << endl;
	}
	int *Cols;
	int nb_cols;
	string_tools ST;

	Orbiter->Int_vec.scan(cols_text, Cols, nb_cols);

	spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;
	int nb_rows;

	nb_rows = S.nb_rows;
	if (f_v) {
		cout << "file_io::do_csv_file_select_cols nb_rows=" << nb_rows << endl;
	}


	string fname_out;

	fname_out.assign(fname);
	ST.chop_off_extension(fname_out);
	fname_out.append("_select.csv");

	{
		ofstream ost(fname_out);
		ost << "Row,";
		S.print_table_row_with_column_selection(0, FALSE, Cols, nb_cols, ost);
		for (i = 0; i < nb_rows - 1; i++) {
			ost << i << ",";
			S.print_table_row_with_column_selection(i + 1, FALSE,
					Cols, nb_cols, ost);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;

	fname_out.assign(fname);
	ST.chop_off_extension(fname_out);
	fname_out.append("_special.csv");

	{
		ofstream ost(fname_out);
		//ost << "Row,";
		//S.print_table_row_with_column_selection(0, FALSE, Cols, nb_cols, ost);
		for (i = 0; i < nb_rows - 1; i++) {
			ost << "Orb" << i << "=";
			S.print_table_row_with_column_selection(i + 1, FALSE,
					Cols, nb_cols, ost);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;

	FREE_int(Cols);
	if (f_v) {
		cout << "file_io::do_csv_file_select_cols done" << endl;
	}
}



void file_io::do_csv_file_select_rows_and_cols(std::string &fname,
		std::string &rows_text, std::string &cols_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_select_rows_and_cols" << endl;
	}
	int *Rows;
	int nb_rows;
	int *Cols;
	int nb_cols;
	string_tools ST;

	Orbiter->Int_vec.scan(rows_text, Rows, nb_rows);

	Orbiter->Int_vec.scan(cols_text, Cols, nb_cols);

	spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int i;



	string fname_out;

	fname_out.assign(fname);
	ST.chop_off_extension(fname_out);
	fname_out.append("_select.csv");

	{
		ofstream ost(fname_out);
		ost << "Row,";
		S.print_table_row_with_column_selection(0, FALSE, Cols, nb_cols, ost);
		for (i = 0; i < nb_rows; i++) {
			ost << i << ",";
			S.print_table_row_with_column_selection(Rows[i] + 1, FALSE,
					Cols, nb_cols, ost);
			}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;


	FREE_int(Rows);
	FREE_int(Cols);
	if (f_v) {
		cout << "file_io::do_csv_file_select_rows done" << endl;
	}
}

void file_io::do_csv_file_extract_column_to_txt(
		std::string &csv_fname, std::string &col_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_extract_column_to_txt" << endl;
	}
	string fname;
	string_tools ST;

	spreadsheet *S;
	int identifier_column;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(csv_fname, 0 /*verbose_level*/);
	cout << "Table " << csv_fname << " has been read" << endl;


	identifier_column = S->find_column(col_label);


	fname.assign(csv_fname);
	ST.replace_extension_with(fname, "_");
	fname.append(col_label);
	fname.append(".txt");




	{
		ofstream ost(fname);

		int i, j;

		for (i = 1; i < S->nb_rows; i++) {
			string entry;
			long int *v;
			int sz;

			S->get_string_entry(entry, i, identifier_column);
			Orbiter->Lint_vec.scan(entry, v, sz);
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << v[j];
			}
			ost << endl;
			FREE_lint(v);
		}
		ost << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	}

	if (f_v) {
		cout << "file_io::do_csv_file_extract_column_to_txt done" << endl;
	}
}



void file_io::do_csv_file_sort_each_row(
		std::string &csv_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_sort_each_row" << endl;
	}
	int *M;
	int m, n;
	sorting Sorting;
	int i;
	string fname;
	string_tools ST;

	int_matrix_read_csv(csv_fname, M, m, n, verbose_level);
	for (i = 0; i < m; i++) {
		Sorting.int_vec_heapsort(M + i * n, n);
	}
	fname.assign(csv_fname);
	ST.replace_extension_with(fname, "_sorted.csv");

	int_matrix_write_csv(fname, M, m, n);

	if (f_v) {
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	}

	if (f_v) {
		cout << "file_io::do_csv_file_sort_each_row done" << endl;
	}
}

void file_io::do_csv_file_join(
		std::vector<std::string> &csv_file_join_fname,
		std::vector<std::string> &csv_file_join_identifier, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_join" << endl;
	}

	int nb_files;
	int i;

	nb_files = csv_file_join_fname.size();

	spreadsheet *S;
	int *identifier_column;

	S = new spreadsheet[nb_files];
	identifier_column = NEW_int(nb_files);

	for (i = 0; i < nb_files; i++) {
		cout << "Reading table " << csv_file_join_fname[i] << endl;
		S[i].read_spreadsheet(csv_file_join_fname[i], 0 /*verbose_level*/);
		cout << "Table " << csv_file_join_fname[i] << " has been read" << endl;
#if 0
		if (i == 0) {
			cout << "The first table is:" << endl;
			S[0].print_table(cout, FALSE);
			}
#endif
		if (FALSE) {
			cout << "The " << i << "th table is:" << endl;
			S[i].print_table(cout, FALSE);
			}


		}

#if 0
	cout << "adding " << nb_with << " -with entries" << endl;
	for (i = 0; i < nb_with; i++) {
		S[with_table[i]].add_column_with_constant_value(with_label[i], with_value[i]);
		}
#endif

	for (i = 0; i < nb_files; i++) {
		identifier_column[i] = S[i].find_by_column(csv_file_join_identifier[i].c_str());
		cout << "Table " << csv_file_join_fname[i] << ", identifier " << identifier_column[i] << endl;
		}

#if 0
	for (i = 0; i < nb_files; i++) {
		by_column[i] = S[i].find_by_column(join_by);
		cout << "File " << fname[i] << " by_column[" << i << "]=" << by_column[i] << endl;
		}
#endif

	cout << "joining " << nb_files << " files" << endl;
	for (i = 1; i < nb_files; i++) {
		cout << "Joining table " << 0 << " = " << csv_file_join_fname[0] << " with table " << i << " = " << csv_file_join_fname[i] << endl;
		S[0].join_with(S + i, identifier_column[0], identifier_column[i], verbose_level - 2);
		cout << "joining " << csv_file_join_fname[0] << " with table " << csv_file_join_fname[i] << " done" << endl;
#if 0
		cout << "After join, the table is:" << endl;
		S[0].print_table(cout, FALSE);
#endif
		}






#if 0
	if (f_drop) {
		S[0].remove_rows(drop_column, drop_label, verbose_level);
		}
#endif


	string save_fname;
	string_tools ST;

	save_fname.assign(csv_file_join_fname[0]);
	ST.chop_off_extension(save_fname);
	save_fname.append("_joined.csv");

	{
		ofstream f(save_fname);
		S[0].print_table(f, FALSE);
		f << "END" << endl;
	}
	cout << "Written file " << save_fname << " of size " << file_size(save_fname) << endl;


	if (f_v) {
		cout << "file_io::do_csv_file_join done" << endl;
	}
}

void file_io::do_csv_file_concatenate(
		std::vector<std::string> &fname_in, std::string &fname_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_concatenate" << endl;
	}

	int nb_files;
	int i;

	nb_files = fname_in.size();

	spreadsheet *S;
	int *identifier_column;

	S = new spreadsheet[nb_files];
	identifier_column = NEW_int(nb_files);

	for (i = 0; i < nb_files; i++) {
		cout << "Reading table " << fname_in[i] << endl;
		S[i].read_spreadsheet(fname_in[i], 0 /*verbose_level*/);
		cout << "Table " << fname_in[i] << " has been read" << endl;

		if (FALSE) {
			cout << "The " << i << "-th table is:" << endl;
			S[i].print_table(cout, FALSE);
		}


	}

	{
		ofstream ost(fname_out);
		int j;
		int f_enclose_in_parentheses = FALSE;

		S[0].print_table_row(0, f_enclose_in_parentheses, ost);
		for (i = 0; i < nb_files; i++) {
			//S[i].print_table(ost, FALSE);
			for (j = 1; j < S[i].nb_rows; j++) {
				S[i].print_table_row(j, f_enclose_in_parentheses, ost);
			}
		}
		ost << "END" << endl;
	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;


	if (f_v) {
		cout << "file_io::do_csv_file_concatenate done" << endl;
	}
}

void file_io::do_csv_file_latex(std::string &fname,
		int f_produce_latex_header,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::do_csv_file_latex" << endl;
	}
	spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	if (f_v) {
		cout << "file_io::do_csv_file_latex S.nb_rows = " << S.nb_rows << endl;
		cout << "file_io::do_csv_file_latex S.nb_cols = " << S.nb_cols << endl;
	}


	string fname_out;
	string_tools ST;

	fname_out.assign(fname);
	ST.chop_off_extension(fname_out);
	fname_out.append(".tex");

	{
		ofstream ost(fname_out);
		latex_interface L;

		//S.print_table_latex_all_columns(ost, FALSE /* f_enclose_in_parentheses */);

		int *f_column_select;
		int j;

		f_column_select = NEW_int(S.nb_cols);
		for (j = 0; j < S.nb_cols; j++) {
			f_column_select[j] = TRUE;
		}
		f_column_select[0] = FALSE;


		if (f_produce_latex_header) {
			//L.head_easy(ost);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				"File", "Orbiter",
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				FALSE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extras_for_preamble */);
		}

		S.print_table_latex(ost,
				f_column_select, FALSE /* f_enclose_in_parentheses */);

		FREE_int(f_column_select);

		if (f_produce_latex_header) {
			L.foot(ost);
		}

	}
	cout << "Written file " << fname_out << " of size " << file_size(fname_out) << endl;


	if (f_v) {
		cout << "file_io::do_csv_file_select_rows done" << endl;
	}
}

void file_io::read_solutions_and_tally(std::string &fname, int sz, int verbose_level)
{
	int nb_solutions;
	int solution_size = sz;
	int *Sol;
	int i, j;

	std::vector<std::vector<int> > Solutions;


	read_solutions_from_file_size_is_known(fname,
			Solutions, solution_size,
			verbose_level);

	nb_solutions = Solutions.size();

	Sol = NEW_int(nb_solutions * solution_size);
	for (i = 0; i < nb_solutions; i++) {
		for (j = 0; j < solution_size; j++) {
			Sol[i * solution_size + j] = Solutions[i][j];
		}
	}


	cout << "nb_solutions = " << nb_solutions << endl;

	tally T;

	T.init(Sol, nb_solutions * solution_size, TRUE, 0);
	cout << "tally:" << endl;
	T.print(TRUE);
	cout << endl;


	int *Pts;
	int nb_pts;
	int multiplicity = 4;

	T.get_data_by_multiplicity(
			Pts, nb_pts, multiplicity, verbose_level);

	cout << "multiplicity " << multiplicity << " number of pts = " << nb_pts << endl;
	Orbiter->Int_vec.print(cout, Pts, nb_pts);
	cout << endl;


	FREE_int(Sol);

}

}}


