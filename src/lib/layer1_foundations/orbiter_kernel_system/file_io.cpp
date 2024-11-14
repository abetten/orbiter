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
namespace layer1_foundations {
namespace orbiter_kernel_system {





file_io::file_io()
{
	Csv_file_support = NEW_OBJECT(csv_file_support);
	Csv_file_support->init(this);
}

file_io::~file_io()
{
	if (Csv_file_support) {
		FREE_OBJECT(Csv_file_support);
		Csv_file_support = NULL;
	}
}


// ToDo: please get rid of all: char *buf; fp.getline(buf, sz, '\n');
// and replace by: string s; getline(f, s);




void file_io::concatenate_files(
		std::string &fname_in_mask, int N,
		std::string &fname_out, std::string &EOF_marker,
		int f_title_line,
	int &cnt_total,
	vector<int> missing_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, cnt;

	if (f_v) {
		cout << "file_io::concatenate_files " << fname_in_mask
			<< " N=" << N << " fname_out=" << fname_out << endl;
	}

	data_structures::string_tools ST;

	//missing_idx = NEW_int(N);
	cnt_total = 0;
	{
		ofstream fp_out(fname_out);
		for (h = 0; h < N; h++) {

			string fname;

			//snprintf(fname, sizeof(fname), fname_in_mask.c_str(), h);

			fname = ST.printf_d(fname_in_mask, h);



			long int sz;

			sz = file_size(fname);
			if (sz < 0) {
				cout << "file_io::concatenate_files "
						"input file does not exist: "
						<< fname << " skipping" << endl;
				//missing_idx[nb_missing++] = h;
				missing_idx.push_back(h);
			}
			else {

				char *buf;

				buf = NEW_char(sz + 1);

				ifstream fp(fname);

				cnt = 0;
				while (true) {
					if (fp.eof()) {
						cout << "file_io::concatenate_files "
								"Encountered End-of-file without having seen EOF "
								"marker, perhaps the file is corrupt. "
								"I was trying to read the file " << fname << endl;
						missing_idx.push_back(h);
						break;
					}

					fp.getline(buf, sz, '\n'); // ToDo
					cout << "Read: " << buf << endl;
					if (strncmp(buf, EOF_marker.c_str(), strlen(EOF_marker.c_str())) == 0) {
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
				FREE_char(buf);
			}
		} // next h
		fp_out << EOF_marker << " " << cnt_total << endl;
	}
	cout << "file_io::concatenate_files "
			"Written file " << fname_out << " of size "
		<< file_size(fname_out) << endl;
	cout << "file_io::concatenate_files "
			"There are " << missing_idx.size()
			<< " missing files, they are:" << endl;

	for (h = 0; h < (int) missing_idx.size(); h++) {

		string fname;

		//snprintf(fname, sizeof(fname), fname_in_mask.c_str(), missing_idx[h]);

		fname = ST.printf_d(fname_in_mask, missing_idx[h]);

		cout << h << " : " << missing_idx[h] << " : " << fname << endl;
	}

	if (f_v) {
		cout << "file_io::concatenate_files done" << endl;
	}
}

void file_io::concatenate_files_into(
		std::string &fname_in_mask, int N,
	ofstream &fp_out, std::string &EOF_marker,
	int f_title_line,
	int &cnt_total,
	vector<int> &missing_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, cnt;

	if (f_v) {
		cout << "file_io::concatenate_files_into " << fname_in_mask
			<< " N=" << N << " into an open file" << endl;
	}

	data_structures::string_tools ST;

	//missing_idx = NEW_int(N);
	cnt_total = 0;
	{
		//ofstream fp_out(fname_out);
		for (h = 0; h < N; h++) {

			string fname;

			//snprintf(fname, sizeof(fname), fname_in_mask.c_str(), h);
			fname = ST.printf_d(fname_in_mask, h);

			fp_out << "# start of file " << fname << endl;

			long int sz;
			char *buf;

			sz = file_size(fname);
			if (sz < 0) {
				cout << "file_io::concatenate_files_into "
						"input file does not exist: "
						<< fname << " skipping" << endl;
				//missing_idx[nb_missing++] = h;
				missing_idx.push_back(h);
			}
			else {
				buf = NEW_char(sz + 1);
				ifstream fp(fname);

				cnt = 0;
				while (true) {
					if (fp.eof()) {
						cout << "file_io::concatenate_files_into "
								"Encountered End-of-file without having seen EOF "
								"marker, perhaps the file is corrupt. "
								"I was trying to read the file " << fname << endl;
						missing_idx.push_back(h);
						break;
					}

					fp.getline(buf, sz, '\n'); // ToDo
					//cout << "Read: " << buf << endl;
					if (strncmp(buf, EOF_marker.c_str(), strlen(EOF_marker.c_str())) == 0) {
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
				FREE_char(buf);
			}
			fp_out << "# end of file " << fname << endl;
		} // next h
		//fp_out << EOF_marker << " " << cnt_total << endl;
	}
	cout << "There are " << missing_idx.size()
			<< " missing files, they are:" << endl;

	for (h = 0; h < (int) missing_idx.size(); h++) {

		string fname;

		//snprintf(fname, sizeof(fname), fname_in_mask.c_str(), missing_idx[h]);
		fname = ST.printf_d(fname_in_mask, missing_idx[h]);

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


void file_io::read_candidates_for_one_orbit_from_file(
		std::string &prefix,
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
		cout << "file_io::read_candidates_for_one_orbit_from_file" << endl;
		cout << "level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " level_of_candidates_file="
				<< level_of_candidates_file << endl;
	}

	orbit_idx = find_orbit_index_in_data_file(prefix,
			level_of_candidates_file, S,
			verbose_level);

	if (f_v) {
		cout << "file_io::read_candidates_for_one_orbit_from_file "
				"orbit_idx=" << orbit_idx << endl;
	}

	if (f_v) {
		cout << "file_io::read_candidates_for_one_orbit_from_file "
				"before generator_read_candidates_of_orbit" << endl;
	}
	string fname2;

	fname2 = prefix + "_lvl_" + std::to_string(level_of_candidates_file);

	poset_classification_read_candidates_of_orbit(
		fname2, orbit_idx,
		candidates1, nb_candidates1, verbose_level - 1);


	for (h = level_of_candidates_file; h < level; h++) {

		long int *candidates2;
		int nb_candidates2;

		if (f_v) {
			cout << "file_io::read_candidates_for_one_orbit_from_file"
					"and_process testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
		}
		candidates2 = NEW_lint(nb_candidates1);

		(*early_test_func_callback)(S, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, 0 /*verbose_level - 1*/);

		if (f_v) {
			cout << "file_io::read_candidates_for_one_orbit_from_file"
					"and_process number of candidates at level "
					<< h + 1 << " reduced from " << nb_candidates1
					<< " to " << nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
		}

		Lint_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_lint(candidates2);
	}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "file_io::read_candidates_for_one_orbit_from_file done" << endl;
	}
}



int file_io::find_orbit_index_in_data_file(
		std::string &prefix,
		int level_of_candidates_file, long int *starter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	int orbit_idx;

	if (f_v) {
		cout << "file_io::find_orbit_index_in_data_file" << endl;
	}

	fname = prefix + "_lvl_" + std::to_string(level_of_candidates_file);

	long int sz;

	sz = file_size(fname);
	if (sz <= 0) {
		cout << "file_io::find_orbit_index_in_data_file "
				"file " << fname << " does not exist" << endl;
		exit(1);
	}
	ifstream f(fname);
	data_structures::string_tools ST;
	int a, i, cnt;
	long int *S;
	char *buf;
	int len, str_len;
	char *p_buf;

	buf = NEW_char(sz + 1);
	S = NEW_lint(level_of_candidates_file);

	cnt = 0;
	f.getline(buf, sz, '\n'); // skip the first line  // ToDo

	orbit_idx = 0;

	while (true) {
		if (f.eof()) {
			break;
		}
		f.getline(buf, sz, '\n'); // ToDo
		//cout << "Read line " << cnt << "='" << buf << "'" << endl;
		str_len = strlen(buf);
		if (str_len == 0) {
			cout << "file_io::find_orbit_index_in_data_file "
					"str_len == 0" << endl;
			exit(1);
		}

		// check for comment line:
		if (buf[0] == '#') {
			continue;
		}

		p_buf = buf;
		ST.s_scan_int(&p_buf, &a);
		if (a == -1) {
			break;
		}
		len = a;
		if (a != level_of_candidates_file) {
			cout << "file_io::find_orbit_index_in_data_file "
					"a != level_of_candidates_file" << endl;
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
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::find_orbit_index_in_data_file done" << endl;
	}
	return orbit_idx;
}


void file_io::write_exact_cover_problem_to_file(
		int *Inc,
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
	cout << "file_io::write_exact_cover_problem_to_file written file "
		<< fname << " of size " << file_size(fname) << endl;
}

void file_io::read_solution_file(
		std::string &fname,
	int *Inc, int nb_rows, int nb_cols,
	int *&Solutions, int &sol_length, int &nb_sol,
	int verbose_level)
// sol_length must be constant
{
	int f_v = (verbose_level >= 1);
	int nb, nb_max, i, j, a, nb_sol1;
	int *x, *y;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::read_solution_file" << endl;
	}
	x = NEW_int(nb_cols);
	y = NEW_int(nb_rows);
	if (f_v) {
		cout << "file_io::read_solution_file reading file " << fname
			<< " of size " << file_size(fname) << endl;
	}

	long int sz;

	sz = file_size(fname);
	if (sz <= 0) {
		cout << "file_io::read_solution_file "
				"There is something wrong with the file "
			<< fname << endl;
		exit(1);
	}
	char *buf;
	char *p_buf;
	buf = NEW_char(sz + 1);
	nb_sol = 0;
	nb_max = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, sz, '\n'); // ToDo
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
						Int_vec_print_fully(cout, y, nb_rows);
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
			f.getline(buf, sz, '\n'); // ToDo
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
	data_structures::string_tools ST;
	int s;

	if (f_v) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
	}

	long int sz;
	nb_solutions = 0;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
	}

	buf = NEW_char(sz + 1);



	solution_size = -1;
	{
		ifstream fp(fname);
		char *p_buf;
		int line_number = 1;


		while (true) {
			if (fp.eof()) {
				cout << "file_io::count_number_of_solutions_in_file_and_get_solution_size "
						"eof, break" << endl;
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
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

void file_io::count_number_of_solutions_in_file(
		std::string &fname,
	int &nb_solutions,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;

	if (f_v) {
		cout << "file_io::count_number_of_solutions_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
	}

	long int sz;

	nb_solutions = 0;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::count_number_of_solutions_in_file file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
	}

	buf = NEW_char(sz + 1);



	{

		ifstream fp(fname);


		while (true) {
			if (fp.eof()) {
				cout << "file_io::count_number_of_solutions_in_file "
						"eof, break" << endl;
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
			//cout << "read line '" << buf << "'" << endl;
			if (strlen(buf) == 0) {
				cout << "file_io::count_number_of_solutions_in_file "
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
		cout << "file_io::count_number_of_solutions_in_file " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
	}
}

void file_io::count_number_of_solutions_in_file_by_case(
		std::string &fname,
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
		cout << "file_io::count_number_of_solutions_in_file_by_case "
			<< fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
	}

	nb_solutions = NEW_int(N);
	case_nb = NEW_int(N);
	nb_cases = 0;

	long int sz;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::count_number_of_solutions_in_file_by_case file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
	}

	buf = NEW_char(sz + 1);



	{
		ifstream fp(fname);


		//nb_sol = 0;
		the_case = -1;
		while (true) {
			if (fp.eof()) {
				cout << "file_io::count_number_of_solutions_in_file_by_case "
						"eof, break" << endl;
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
			//cout << "read line '" << buf << "'" << endl;
			if (strlen(buf) == 0) {
				cout << "file_io::count_number_of_solutions_in_file_by_case "
						"empty line, break" << endl;
				break;
			}

			if (strncmp(buf, "# start case", 12) == 0) {
				the_case = atoi(buf + 13);
				the_case_count = 0;
				cout << "file_io::count_number_of_solutions_in_file_by_case "
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
		cout << "file_io::count_number_of_solutions_in_file_by_case "
			<< fname << endl;
		cout << "nb_cases = " << nb_cases << endl;
	}
}


void file_io::read_solutions_from_file_and_get_solution_size(
		std::string &fname,
	int &nb_solutions, long int *&Solutions, int &solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size" << endl;
		cout << "file_io::read_solutions_from_file_and_get_solution_size trying to read file "
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

	Solutions = NEW_lint(nb_solutions * solution_size);

	data_structures::string_tools ST;
	char *buf;
	char *p_buf;
	int i, nb_sol;
	long int a;
	long int sz;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::read_solutions_from_file_and_get_solution_size file "
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
	}

	buf = NEW_char(sz + 1);
	nb_sol = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, sz, '\n'); // ToDo
			if (strlen(buf) && buf[0] == '#') {
				continue;
			}
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			ST.s_scan_lint(&p_buf, &a);

			if (a == -1) {
				break;
			}
			if (a != solution_size) {
				cout << "file_io::read_solutions_from_file_and_get_solution_size "
						"a != solution_size" << endl;
				exit(1);
			}
			for (i = 0; i < solution_size; i++) {
				ST.s_scan_lint(&p_buf, &a);
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


void file_io::read_solutions_from_file(
		std::string &fname,
	int &nb_solutions, long int *&Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int i, nb_sol;
	long int a;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::read_solutions_from_file" << endl;
		cout << "file_io::read_solutions_from_file trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file solution_size="
			<< solution_size << endl;
	}

	long int sz;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::file_io::read_solutions_from_file "
				"the file " << fname << " does not exist" << endl;
		return;
	}

	buf = NEW_char(sz + 1);

	count_number_of_solutions_in_file(fname,
		nb_solutions,
		verbose_level - 2);
	if (f_v) {
		cout << "file_io::read_solutions_from_file, reading "
			<< nb_solutions << " solutions" << endl;
	}



	Solutions = NEW_lint(nb_solutions * solution_size);

	nb_sol = 0;
	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, sz, '\n'); // ToDo
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			ST.s_scan_lint(&p_buf, &a);

			if (a == -1) {
				break;
			}
			if (a != solution_size) {
				cout << "file_io::read_solutions_from_file "
						"a != solution_size" << endl;
				exit(1);
			}
			for (i = 0; i < solution_size; i++) {
				ST.s_scan_lint(&p_buf, &a);
				Solutions[nb_sol * solution_size + i] = a;
			}
			nb_sol++;
		}
	}
	if (nb_sol != nb_solutions) {
		cout << "file_io::read_solutions_from_file "
				"nb_sol != nb_solutions" << endl;
		exit(1);
	}
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::read_solutions_from_file done" << endl;
	}
}


void file_io::read_solutions_from_file_size_is_known(
		std::string &fname,
	std::vector<std::vector<long int> > &Solutions,
	int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	vector<long int> one_solution;
	data_structures::string_tools ST;
	int i;
	long int a;

	if (f_v) {
		cout << "read_solutions_from_file_size_is_known" << endl;
		cout << "read_solutions_from_file_size_is_known trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file_size_is_known solution_size="
			<< solution_size << endl;
	}

	long int sz;

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::read_solutions_from_file_size_is_known "
				"the file " << fname << " does not exist" << endl;
		return;
	}

	buf = NEW_char(sz + 1);

	one_solution.resize(solution_size);

	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, sz, '\n'); // ToDo
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			ST.s_scan_lint(&p_buf, &a);

			if (a == -1) {
				break;
			}

			one_solution[0] = a;
			for (i = 1; i < solution_size; i++) {
				ST.s_scan_lint(&p_buf, &a);
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


void file_io::read_solutions_from_file_by_case(
		std::string &fname,
	int *nb_solutions, int *case_nb, int nb_cases,
	long int **&Solutions, int solution_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	//int nb_sol;
	int i;
	int nb_case1;
	int the_case;
	int the_case_count = 0;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::read_solutions_from_file_by_case" << endl;
		cout << "file_io::read_solutions_from_file_by_case "
				"trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
		cout << "file_io::read_solutions_from_file_by_case "
				"solution_size="
			<< solution_size << endl;
	}

	long int sz;

	sz = file_size(fname);
	if (sz < 0) {
		return;
	}

	buf = NEW_char(sz + 1);

	Solutions = NEW_plint(nb_cases);

	{
		ifstream fp(fname);


		//nb_sol = 0;
		nb_case1 = 0;
		the_case = -1;
		while (true) {
			if (fp.eof()) {
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
			//cout << "read line '" << buf << "'" << endl;
			if (strlen(buf) == 0) {
				cout << "file_io::read_solutions_from_file_by_case "
						"empty line, break" << endl;
				break;
			}

			if (strncmp(buf, "# start case", 12) == 0) {
				the_case = atoi(buf + 13);
				the_case_count = 0;
				if (the_case != case_nb[nb_case1]) {
					cout << "file_io::read_solutions_from_file_by_case "
							"the_case != case_nb[nb_case1]" << endl;
					exit(1);
				}
				Solutions[nb_case1] =
						NEW_lint(nb_solutions[nb_case1] * solution_size);
				cout << "file_io::read_solutions_from_file_by_case "
						"read start case " << the_case << endl;
			}
			else if (strncmp(buf, "# end case", 10) == 0) {
				if (the_case_count != nb_solutions[nb_case1]) {
					cout << "file_io::read_solutions_from_file_by_case "
							"the_case_count != nb_solutions[nb_case1]" << endl;
					exit(1);
				}
				cout << "file_io::read_solutions_from_file_by_case "
						"read end case " << the_case << endl;
				nb_case1++;
				the_case = -1;
			}
			else {
				if (the_case >= 0) {
					char *p_buf;
					long int sz;
					long int a;

					//cout << "read_solutions_from_file_by_case "
					//"reading solution " << the_case_count
					//<< " for case " << the_case << endl;
					p_buf = buf;
					ST.s_scan_lint(&p_buf, &sz);
					if (sz != solution_size) {
						cout << "file_io::read_solutions_from_file_by_case "
								"sz != solution_size" << endl;
						exit(1);
					}
					for (i = 0; i < sz; i++) {
						ST.s_scan_lint(&p_buf, &a);
						Solutions[nb_case1][the_case_count * solution_size + i] = a;
					}
					the_case_count++;
				}
			}
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::read_solutions_from_file_by_case done" << endl;
	}
}

void file_io::copy_file_to_ostream(
		std::ostream &ost, std::string &fname)
{

	{
		ifstream fp(fname);

#if 0
		while (true) {
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
		while (true) {
			char c;
			fp.get(c);
			if (fp.eof()) {
				break;
			}
			ost << c;
		}
	}

}





void file_io::int_matrix_write_cas_friendly(
		std::string &fname,
		int *M, int m, int n)
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

void file_io::int_matrix_write_text(
		std::string &fname,
		int *M, int m, int n)
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

void file_io::lint_matrix_write_text(
		std::string &fname,
		long int *M, int m, int n)
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

void file_io::int_matrix_read_text(
		std::string &fname,
		int *&M, int &m, int &n)
{
	int i, j;

	if (file_size(fname) <= 0) {
		cout << "file_io::int_matrix_read_text The file "
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


// ToDo please eliminate this:

#define MY_OWN_BUFSIZE 1000000


void file_io::parse_sets(
		int nb_cases, char **data, int f_casenumbers,
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
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::parse_sets "
				"f_casenumbers=" << f_casenumbers
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

		parse_line(
				p_buf, Set_sizes[h], Sets[h],
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

void file_io::parse_line(
		char *line, int &len,
	long int *&set, char *ago_ascii, char *aut_ascii)
{
	int i;
	char *p_buf;
	data_structures::string_tools ST;

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
	long int sz;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::count_number_of_orbits_in_file " << fname << endl;
		cout << "file_io::count_number_of_orbits_in_file "
				"trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
	}

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::count_number_of_orbits_in_file "
				"file size is -1" << endl;
		return -1;
	}

	buf = NEW_char(sz + 1);



	{
		ifstream fp(fname);


		nb_sol = 0;
		while (true) {
			if (fp.eof()) {
				break;
			}

			//cout << "count_number_of_orbits_in_file "
			//"reading line, nb_sol = " << nb_sol << endl;
			fp.getline(buf, sz, '\n'); // ToDo
			if (strlen(buf) == 0) {
				cout << "file_io::count_number_of_orbits_in_file "
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
					cout << "file_io::count_number_of_orbits_in_file "
							"found a complete file with " << nb_sol
							<< " solutions" << endl;
				}
				break;
			}
			else {
				if (false) {
					cout << "file_io::count_number_of_orbits_in_file "
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

int file_io::count_number_of_lines_in_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	int nb_lines;
	long int sz;

	if (f_v) {
		cout << "file_io::count_number_of_lines_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
	}

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::count_number_of_lines_in_file "
				"file size is -1" << endl;
		return 0;
	}

	buf = NEW_char(sz + 1);



	{
		ifstream fp(fname);


		nb_lines = 0;
		while (true) {
			if (fp.eof()) {
				break;
			}

			//cout << "count_number_of_lines_in_file "
			// "reading line, nb_sol = " << nb_sol << endl;
			fp.getline(buf, sz, '\n'); // ToDo
			nb_lines++;
		}
	}
	FREE_char(buf);

	return nb_lines;
}

int file_io::try_to_read_file(
		std::string &fname,
	int &nb_cases, char **&data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int n1;
	int nb_sol, len, a;
	long int sz;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "file_io::try_to_read_file "
				"trying to read file " << fname
			<< " of size " << file_size(fname) << endl;
	}


	sz = file_size(fname);
	if (sz <= 0) {
		goto return_false;
	}

	{
	char *buf, *p_buf;
	buf = NEW_char(sz + 1);

	{
		ifstream fp(fname);

#if 0
		if (fp.eof()) {
			goto return_false;
		}
		fp.getline(buf, sz, '\n');
		if (strlen(buf) == 0) {
			goto return_false;
		}
		sscanf(buf + 1, "%d", &n1);
		cout << "n1=" << n1;
		if (n1 != n) {
			cout << "file_io::try_to_read_file n1 != n" << endl;
			exit(1);
		}
#endif

		nb_sol = 0;
		while (true) {
			if (fp.eof()) {
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
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
					cout << "file_io::try_to_read_file "
							"found a complete file with "
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
		fp.getline(buf, sz, '\n');
		if (strlen(buf) == 0) {
			goto return_false;
		}
		sscanf(buf + 1, "%d", &n1);
		if (n1 != n) {
			cout << "file_io::try_to_read_file n1 != n" << endl;
			exit(1);
		}
#endif

		nb_sol = 0;
		while (true) {
			if (fp.eof()) {
				break;
			}
			fp.getline(buf, sz, '\n'); // ToDo
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
	}
	return true;

return_false:
	return false;
}

void file_io::read_and_parse_data_file(
	std::string &fname, int &nb_cases,
	char **&data, long int **&sets, int *&set_sizes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "file_io::read_and_parse_data_file "
				"reading file "
			<< fname << endl;
	}
	if (try_to_read_file(fname, nb_cases, data, verbose_level)) {
		if (f_vv) {
			cout << "file_io::read_and_parse_data_file "
					"file read containing " << nb_cases
				<< " cases" << endl;
		}
	}
	else {
		cout << "file_io::read_and_parse_data_file "
				"couldn't read file "
			<< fname << endl;
		exit(1);
	}

#if 0
	for (i = 0; i < nb_cases; i++) {
		cout << i << " : " << data[i] << endl;
	}
#endif


	if (f_v) {
		cout << "file_io::read_and_parse_data_file "
				"parsing sets" << endl;
	}
	//parse_sets(nb_cases, data, set_sizes, sets);

	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i;

	parse_sets(
			nb_cases, data, false /*f_casenumbers */,
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
		cout << "file_io::read_and_parse_data_file done" << endl;
	}
}

void file_io::parse_sets_and_check_sizes_easy(
		int len, int nb_cases,
	char **data, long int **&sets)
{
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int *set_sizes;
	int i;

	parse_sets(
			nb_cases, data, false /*f_casenumbers */,
		set_sizes, sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level - 2*/);
	for (i = 0; i < nb_cases; i++) {
		if (set_sizes[i] != len) {
			cout << "file_io::parse_sets_and_check_sizes_easy "
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

void file_io::free_data_fancy(
		int nb_cases,
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
		cout << "file_io::read_and_parse_data_file_fancy "
				"reading file "
			<< fname << endl;
	}
	if (f_vv) {
		cout << "file_io::read_and_parse_data_file_fancy "
				"before try_to_read_file" << endl;
	}
	if (try_to_read_file(
			fname, nb_cases, data, verbose_level - 1)) {
		if (f_vv) {
			cout << "file_io::read_and_parse_data_file_fancy "
					"file read containing "
				<< nb_cases << " cases" << endl;
		}
	}
	else {
		cout << "file_io::read_and_parse_data_file_fancy "
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
		cout << "file_io::read_and_parse_data_file_fancy "
				"parsing sets" << endl;
	}
	parse_sets(
			nb_cases, data, f_casenumbers,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 2);

	if (f_vv) {
		cout << "file_io::read_and_parse_data_file_fancy "
				"freeing temporary data" << endl;
	}
	for (i = 0; i < nb_cases; i++) {
		FREE_char(data[i]);
	}
	FREE_pchar(data);
	if (f_vv) {
		cout << "file_io::read_and_parse_data_file_fancy done" << endl;
	}
}

void file_io::read_set_from_file(
		std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
// if the file is empty,
// set_size cannot be determined and is set to 0
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;

	if (f_v) {
		cout << "file_io::read_set_from_file "
				"opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
	}
	ifstream f(fname);

	f >> set_size;

	if (set_size == -1) {
		if (f_v) {
			cout << "file_io::read_set_from_file "
					"the file is empty, "
					"set_size cannot be determined" << endl;
		}
		set_size = 0;
	}
	else {
		if (f_v) {
			cout << "file_io::read_set_from_file "
					"allocating set of size "
				<< set_size << endl;
		}
		the_set = NEW_lint(set_size);

		if (f_v) {
			cout << "file_io::read_set_from_file "
					"reading set of size "
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
			cout << "file_io::read_set_from_file "
					"read a set of size " << set_size
				<< " from file " << fname << endl;
		}
		if (f_vv) {
			cout << "file_io::read_set_from_file "
					"the set is:" << endl;
			Lint_vec_print(cout, the_set, set_size);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "file_io::read_set_from_file done" << endl;
	}
}

void file_io::write_set_to_file(
		std::string &fname,
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

void file_io::read_set_from_file_lint(
		std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	long int a;

	if (f_v) {
		cout << "file_io::read_set_from_file_lint opening file " << fname
			<< " of size " << file_size(fname)
			<< " for reading" << endl;
	}
	ifstream f(fname);

	f >> set_size;
	if (f_v) {
		cout << "file_io::read_set_from_file_lint allocating set of size "
			<< set_size << endl;
	}
	the_set = NEW_lint(set_size);

	if (f_v) {
		cout << "file_io::read_set_from_file_lint reading set of size "
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
		cout << "file_io::read_set_from_file_lint "
				"read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "file_io::read_set_from_file_lint "
				"the set is:" << endl;
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::write_set_to_file_lint(
		std::string &fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "file_io::write_set_to_file_lint opening file "
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
		cout << "file_io::write_set_to_file_lint "
				"Written file " << fname << " of size "
			<< file_size(fname) << endl;
	}
}

void file_io::read_set_from_file_int4(
		std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, b;
	int_4 a;

	if (f_v) {
		cout << "file_io::read_set_from_file_int4 opening file " << fname
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
		cout << "file_io::read_set_from_file_int4 "
				"read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "file_io::read_set_from_file_int4 "
				"the set is:" << endl;
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::read_set_from_file_int8(
		std::string &fname,
	long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	long int b;
	int_8 a;

	if (f_v) {
		cout << "file_io::read_set_from_file_int8 "
				"opening file " << fname
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
		cout << "file_io::read_set_from_file_int8 "
				"read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "file_io::read_set_from_file_int8 "
				"the set is:" << endl;
		Lint_vec_print(cout, the_set, set_size);
		cout << endl;
	}
}

void file_io::write_set_to_file_as_int4(
		std::string &fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int_4 a;
	int b;

	if (f_v) {
		cout << "file_io::write_set_to_file_as_int4 opening file "
			<< fname << " for writing" << endl;
	}
	{
		ofstream f(fname, ios::binary);


		a = (int_4) set_size;
		f.write((char *) &a, sizeof(int_4));
		b = a;
		if (b != set_size) {
			cout << "file_io::write_set_to_file_as_int4 "
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
				cout << "file_io::write_set_to_file_as_int4 data loss" << endl;
				cout << "i=" << i << endl;
				cout << "the_set[i]=" << the_set[i] << endl;
				cout << "a=" << a << endl;
				cout << "b=" << b << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "file_io::write_set_to_file_as_int4 "
				"Written file " << fname
			<< " of size " << file_size(fname) << endl;
	}
}

void file_io::write_set_to_file_as_int8(
		std::string &fname,
	long int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int_8 a, b;

	if (f_v) {
		cout << "file_io::write_set_to_file_as_int8 opening file "
			<< fname << " for writing" << endl;
	}
	{
		ofstream f(fname, ios::binary);


		a = (int_8) set_size;
		f.write((char *) &a, sizeof(int_8));
		b = a;
		if (b != set_size) {
			cout << "file_io::write_set_to_file_as_int8 "
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
				cout << "file_io::write_set_to_file_as_int8 data loss" << endl;
				cout << "i=" << i << endl;
				cout << "the_set[i]=" << the_set[i] << endl;
				cout << "a=" << a << endl;
				cout << "b=" << b << endl;
				exit(1);
			}
		}
	}
	if (f_v) {
		cout << "file_io::write_set_to_file_as_int8 "
				"Written file " << fname
			<< " of size " << file_size(fname) << endl;
	}
}

void file_io::read_k_th_set_from_file(
		std::string &fname, int k,
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, h;

	if (f_v) {
		cout << "file_io::read_k_th_set_from_file opening file "
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
				cout << "file_io::read_k_th_set_from_file: h="
					<< h << " the " << i
					<< "-th number is " << a << endl;
			}
			//if (a == -1)
				//break;
			the_set[i] = a;
		}
	}
	if (f_v) {
		cout << "file_io::read_k_th_set_from_file "
				"read a set of size " << set_size
			<< " from file " << fname << endl;
	}
	if (f_vv) {
		cout << "file_io::read_k_th_set_from_file "
				"the set is:" << endl;
		Int_vec_print(cout, the_set, set_size);
		cout << endl;
	}
}


void file_io::write_incidence_matrix_to_file(
		std::string &fname,
	int *Inc, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_inc;

	if (f_v) {
		cout << "file_io::write_incidence_matrix_to_file "
				"opening file "
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
		cout << "file_io::write_incidence_matrix_to_file "
				"Written file " << fname << " of size "
			<< file_size(fname) << endl;
	}
}

void file_io::read_incidence_matrix_from_inc_file(
		int *&M, int &m, int &n,
		std::string &inc_file_name, int inc_file_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char *buf;
	char *p_buf;
	int *X = NULL;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "file_io::read_incidence_matrix_from_inc_file "
			<< inc_file_name << " no " << inc_file_idx << endl;
	}

	file_io Fio;
	int sz;

	sz = Fio.file_size(inc_file_name);

	buf = NEW_char(sz);

	{
		ifstream f(inc_file_name);

		if (f.eof()) {
			exit(1);
		}
		f.getline(buf, sz, '\n'); // ToDo
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
		while (true) {
			if (f.eof()) {
				break;
			}
			f.getline(buf, sz, '\n'); // ToDo
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
				cout << "\nfile_io::read_incidence_matrix_from_inc_file: "
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
					cout << "file_io::read_incidence_matrix_from_inc_file: "
							"found the following incidence matrix:" << endl;
					Int_vec_print_integer_matrix_width(cout,
						M, m, n, n, 1);
				}
				break;
			}
			cnt++;
		}
	}
	FREE_int(X);
	FREE_char(buf);
}


void file_io::read_incidence_file(
		std::vector<std::vector<int> > &Geos,
		int &m, int &n, int &nb_flags,
		std::string &inc_file_name, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, h, cnt;
	char *buf;
	char *p_buf;
	int *X = NULL;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "file_io::read_incidence_file " << inc_file_name << endl;
	}

	file_io Fio;
	int sz;

	sz = Fio.file_size(inc_file_name);

	if (f_v) {
		cout << "file_io::read_incidence_file "
				"file size = " << sz << endl;
	}

	buf = NEW_char(sz);

	{
		ifstream f(inc_file_name);

		if (f.eof()) {
			exit(1);
		}
		f.getline(buf, sz, '\n'); // ToDo
		if (strlen(buf) == 0) {
			exit(1);
		}
		sscanf(buf, "%d %d %d", &m, &n, &nb_flags);
		if (f_vv) {
			cout << "m=" << m;
			cout << " n=" << n;
			cout << " nb_flags=" << nb_flags << endl;
		}
		X = NEW_int(nb_flags);
		cnt = 0;
		while (true) {
			if (f.eof()) {
				break;
			}
			f.getline(buf, sz, '\n'); // ToDo
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
				cout << "file_io::read_incidence_file: "
						"found a complete file with "
					<< cnt << " solutions" << endl;
				break;
			}
			X[0] = a;

			//cout << "reading " << nb_inc << " incidences" << endl;
			for (h = 1; h < nb_flags; h++) {
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

			vector<int> v;

			for (h = 0; h < nb_flags; h++) {
				v.push_back(X[h]);
			}
			Geos.push_back(v);
			cnt++;
		}
	}
	FREE_int(X);
	FREE_char(buf);
}


void file_io::read_incidence_by_row_ranks_file(
		std::vector<std::vector<int> > &Geos,
		int &m, int &n, int &r,
		std::string &inc_file_name, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, h, cnt;
	char *buf;
	char *p_buf;
	int *X = NULL;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "file_io::read_incidence_by_row_ranks_file " << inc_file_name << endl;
	}

	file_io Fio;
	int file_sz;

	file_sz = Fio.file_size(inc_file_name);

	buf = NEW_char(file_sz);

	{
		ifstream f(inc_file_name);

		if (f.eof()) {
			exit(1);
		}
		f.getline(buf, file_sz, '\n'); // ToDo
		if (strlen(buf) == 0) {
			exit(1);
		}
		sscanf(buf, "%d %d %d", &m, &n, &r);
		if (f_vv) {
			cout << "m=" << m;
			cout << " n=" << n;
			cout << " r=" << r << endl;
		}
		X = NEW_int(m);
		int *Row;
		combinatorics::combinatorics_domain Combi;

		Row = NEW_int(n);
		cnt = 0;
		while (true) {
			if (f.eof()) {
				break;
			}
			f.getline(buf, file_sz, '\n'); // ToDo
			if (strlen(buf) == 0) {
				continue;
			}


			cout << "read line: " << buf << endl;

			// check for comment line:
			if (buf[0] == '#') {
				continue;
			}

			p_buf = buf;

			ST.s_scan_int(&p_buf, &a);
			if (f_vv) {
				cout << cnt << " : " << a << " ";
			}
			if (a == -1) {
				cout << "file_io::read_incidence_file: "
						"found a complete file with "
					<< cnt << " solutions" << endl;
				break;
			}

			cout << "reading row consisting of "
					<< m << " entries" << endl;
			for (h = 0; h < m; h++) {
				if (h == 0) {
					;
				}
				else {
					ST.s_scan_int(&p_buf, &a);
				}
				X[h] = a;
				//M[a] = 1;
			}
			//f >> a; // skip aut group order

			vector<int> v;
			int u;

			for (h = 0; h < m; h++) {
				Combi.unrank_k_subset(X[h], Row, n, r);
				cout << "row " << h << " / " << m << " : " << X[h] << " : ";
				Int_vec_print(cout, Row, r);
				cout << endl;
				for (u = 0; u < r; u++) {
					v.push_back(h * n + Row[u]);
				}
			}
			cout << "geo " << cnt << " has " << v.size() << " incidences: ";
			for (h = 0; h < v.size(); h++) {
				cout << v[h];
				if (h < v.size() - 1) {
					cout << ",";
				}
			}
			cout << endl;
			Geos.push_back(v);
			cnt++;
		}
		FREE_int(Row);
		FREE_int(X);
	}
	FREE_char(buf);
}



int file_io::inc_file_get_number_of_geometries(
	std::string &inc_file_name, int verbose_level)
// this function is not used
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char *buf;
	char *p_buf;
	int *X = NULL;
	int m, n;
	data_structures::string_tools ST;


	if (f_v) {
		cout << "file_io::inc_file_get_number_of_geometries "
			<< inc_file_name << endl;
	}

	file_io Fio;
	int sz;

	sz = Fio.file_size(inc_file_name);

	buf = NEW_char(sz);

	{
		ifstream f(inc_file_name);

		if (f.eof()) {
			exit(1);
		}
		f.getline(buf, sz, '\n'); // ToDo
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
		while (true) {
			if (f.eof()) {
				break;
			}
			f.getline(buf, sz, '\n'); // ToDo
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
				cout << "\nfile_io::read_incidence_matrix_from_inc_file: "
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
	FREE_char(buf);
	return cnt;
}

long int file_io::file_size(
		std::string &fname)
{
	return file_size(fname.c_str());
}

long int file_io::file_size(
		const char *name)
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

void file_io::delete_file(
		std::string &fname)
{
	string cmd;

	cmd = "rm " + fname;
	system(cmd.c_str());
}

void file_io::fwrite_int4(
		FILE *fp, int a)
{
	int_4 I;

	I = (int_4) a;
	fwrite(&I, 1 /* size */, 4 /* items */, fp);
}

int_4 file_io::fread_int4(
		FILE *fp)
{
	int_4 I;

	fread(&I, 1 /* size */, 4 /* items */, fp);
	return I;
}

void file_io::fwrite_uchars(
		FILE *fp, unsigned char *p, int len)
{
	fwrite(p, 1 /* size */, len /* items */, fp);
}

void file_io::fread_uchars(
		FILE *fp, unsigned char *p, int len)
{
	fread(p, 1 /* size */, len /* items */, fp);
}


void file_io::read_ascii_set_of_sets_constant_size(
		std::string &fname_ascii,
		int *&Sets, int &nb_sets, int &set_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "file_io::read_ascii_set_of_sets_constant_size "
				"reading ascii file " << fname_ascii << endl;
	}
	data_structures::sorting Sorting;
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
		while (true) {
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

			Sorting.int_vec_heapsort(
					Sets + nb_sets * set_size, set_size);

			if (f_v) {
				cout << "file_io::read_ascii_set_of_sets_constant_size "
						"set " << nb_sets << " / " << N << " is ";
				Int_vec_print(cout, Sets + nb_sets * set_size, set_size);
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

void file_io::write_decomposition_stack(
		std::string &fname, int m, int n,
		int *v, int *b, int *aij, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::write_decomposition_stack" << endl;
	}
	{
		ofstream f(fname);
		int i, j;


		f << "<HTDO type=pt ptanz=" << m
				<< " btanz=" << n << " fuse=simple>" << endl;
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
		cout << "written file " << fname
				<< " of size " << file_size(fname) << endl;
	}
}

void file_io::create_files_direct(
		std::string &fname_mask,
		std::string &content_mask,
		std::vector<std::string> &labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::create_files_direct" << endl;
	}

	data_structures::string_tools ST;
	int j;
	std::string fname;

	for (j = 0; j < labels.size(); j++) {

		fname = ST.printf_s(
			fname_mask, labels[j]);

		{
			ofstream fp(fname);

			string text;

			text = ST.printf_s(
					content_mask, labels[j]);

			ST.fix_escape_characters(text);

			fp << text << endl;
		}

		if (f_v) {
			cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		}


	}

	if (f_v) {
		cout << "file_io::create_files_direct done" << endl;
	}
}

void file_io::create_file(
		create_file_description *Descr, int verbose_level)
{
	//file_io Fio;
	int j;

	if (Descr->f_read_cases) {

		cout << "Descr->f_read_cases" << endl;
		string fname;
		//char str[1000];
		//int *Cases;
		int nb_cases;
		int n, c;
		data_structures::string_tools ST;

		cout << "reading file " << Descr->read_cases_fname << endl;


		data_structures::spreadsheet S;

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
			//snprintf(str, sizeof(str), Descr->file_mask.c_str(), c);
			//fname.assign(str);

			fname = ST.printf_d(Descr->file_mask, c);


			{
				ofstream fp(fname);

				string line;

				for (j = 0; j < Descr->nb_lines; j++) {
					if (Descr->f_line_numeric[j]) {
						//snprintf(str, sizeof(str), Descr->lines[j].c_str(), c);
						line = ST.printf_d(Descr->lines[j], c);

					}
					else {
						string s;
						char str[1000];

						S.get_string(s, c, 0);

						snprintf(str, sizeof(str),
								Descr->lines[j].c_str(), s.c_str());
						line = str;
					}
					ST.fix_escape_characters(line);
					fp << line << endl;
				}
			}
			cout << "Written file " << fname << " of size "
					<< file_size(fname) << endl;

			char log_entry[1000];

			snprintf(log_entry, sizeof(log_entry), log_mask, c);
			fp_log << log_entry << endl;
			}
		}
		cout << "Written file " << log_fname << " of size "
				<< file_size(log_fname) << endl;
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

		data_structures::spreadsheet *S;
		int row;

		S = NEW_OBJECT(data_structures::spreadsheet);
		S->read_spreadsheet(
				Descr->read_cases_fname, 0 /*verbose_level*/);

		cout << "Read spreadsheet with " << S->nb_rows << " rows" << endl;

		//S->print_table(cout, false /* f_enclose_in_parentheses */);
		for (row = 0; row < MINIMUM(10, S->nb_rows); row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row,
					false /* f_enclose_in_parentheses */, cout);
		}
		cout << "..." << endl;
		for (row = MAXIMUM(S->nb_rows - 10, 0); row < S->nb_rows; row++) {
			cout << "row " << row << " : ";
			S->print_table_row(row,
					false /* f_enclose_in_parentheses */, cout);
		}



		create_files_list_of_cases(S, Descr, verbose_level);

	}
	else if (Descr->f_N) {
		create_files(Descr, verbose_level);
	}

}

void file_io::create_files(
		create_file_description *Descr,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	file_io Fio;
	data_structures::string_tools ST;

	string fname;
	int r;

	if (f_v) {
		cout << "file_io::create_files" << endl;
	}

	const char *makefile_fname = "makefile_submit";
	{
	ofstream fp_makefile(makefile_fname);

	for (i = 0; i < Descr->N; i++) {

		fname = ST.printf_d(Descr->file_mask, i);

		fp_makefile << "\tsbatch " << fname << endl;
		{
			ofstream fp(fname);

			for (j = 0; j < Descr->nb_lines; j++) {


				{
					cout << "mask='" << Descr->lines[j].c_str() << "'" << endl;
					char str[1000];

					snprintf(str, sizeof(str),
							Descr->lines[j].c_str(), i, i, i, i, i, i, i, i);

					string s;

					s = str;

					ST.fix_escape_characters(s);
					cout << "str='" << s << "'" << endl;
					fp << s << endl;
				}
			}
			if (Descr->f_repeat) {
				if (Descr->f_split) {
					for (r = 0; r < Descr->split_m; r++) {
						for (j = 0; j < Descr->repeat_N; j++) {
							if ((j % Descr->split_m) == r) {

								string s;

								s = ST.printf_d(Descr->repeat_mask, i);
								ST.fix_escape_characters(s);
								fp << s << endl;

							}
						}
						fp << endl;
					}
				}
				else {

					{
						string s;

						s = ST.printf_d(Descr->repeat_mask, Descr->repeat_N);

						ST.fix_escape_characters(s);
						fp << s << endl;
					}
					if (!Descr->f_command) {
						cout << "please use option -command when using -repeat" << endl;
						exit(1);
					}
					for (j = 0; j < Descr->repeat_N; j++) {

						int c;
						c = Descr->repeat_start + j * Descr->repeat_increment;

						string s;
						char str[1000];

						snprintf(str, sizeof(str),
								Descr->command.c_str(), c, c, c, c);

						s = str;

						ST.fix_escape_characters(s);
						fp << s << endl;
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

void file_io::create_files_list_of_cases(
		data_structures::spreadsheet *S,
		create_file_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	string fname;
	file_io Fio;
	data_structures::string_tools ST;

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

			fname = ST.printf_d(Descr->file_mask, i);

			fp_makefile << "\tsbatch " << fname << endl;
			fp_submit_script << "sbatch " << fname << endl;
			{
				ofstream fp(fname);

				for (j = 0; j < Descr->nb_lines; j++) {
					char str[1000];
					string s;

					snprintf(str, sizeof(str),
							Descr->lines[j].c_str(), i, i, i, i, i, i, i, i);
					s = str;
					ST.fix_escape_characters(s);
					fp << s << endl;
				}

				if (Descr->f_tasks) {
					int t;
					//int NT;

					{
						string s;

						s = ST.printf_d(Descr->tasks_line, Descr->nb_tasks);
						fp << s << endl;
					}
					//NT = Descr->N * Descr->nb_tasks;
					for (t = 0; t < Descr->nb_tasks; t++) {

						char str[1000];
						snprintf(str, sizeof(str),
								Descr->command.c_str(), i, t, i, t);

						string s;

						s = str;
						ST.fix_escape_characters(s);
						fp << s; // << " \\" << endl;

						for (j = 0; j < nb_cases; j++) {
							if ((j % Descr->N) != i) {
								continue;
							}
							if (((j - i) / Descr->N) % Descr->nb_tasks != t) {
								continue;
							}
							string entry;
							//int case_number;

							//case_number = S->get_int(j + 1, Descr->read_cases_column_of_case);
							S->get_string(entry, j + 1, Descr->read_cases_column_of_fname);
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


					{
						string s;

						s = ST.printf_d(Descr->command, i);
						ST.fix_escape_characters(s);
						fp << s << " \\" << endl;
					}

					//fp << command << " \\" << endl;
					for (j = 0; j < nb_cases; j++) {
						if ((j % Descr->N) != i) {
							continue;
						}
						string entry;
						//int case_number;

						//case_number = S->get_int(j + 1, Descr->read_cases_column_of_case);
						S->get_string(entry, j + 1, Descr->read_cases_column_of_fname);
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

					char str[1000];

					snprintf(str, sizeof(str),
							Descr->final_lines[j].c_str(),
							i, i, i, i, i, i, i, i);

					string s;

					s = str;
					ST.fix_escape_characters(s);
					fp << s << endl;
				} // next j

			} // close fp(fname)

			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;

		} // next i

	}
	cout << "Written file " << makefile_fname << " of size "
			<< Fio.file_size(makefile_fname) << endl;

	string mask_submit_script_piecewise;

	mask_submit_script_piecewise = "submit_jobs_%d.sh";
	string fname_submit_piecewise;
	int h;
	int N1 = 128;

	for (h = 0; h < Descr->N / N1; h++) {

		fname_submit_piecewise = ST.printf_d(mask_submit_script_piecewise, h * N1);

		{
			ofstream fp_submit_script(fname_submit_piecewise);

			fp_submit_script << "#!/bin/bash" << endl;
			for (i = 0; i < N1; i++) {

				fname = ST.printf_d(Descr->file_mask, h * N1 + i);

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

		string cmd;

		cmd = "chmod +x " + fname_submit_piecewise;
		system(cmd.c_str());

	}
	if (f_v) {
		cout << "file_io::create_files_list_of_cases done" << endl;
	}
}

int file_io::number_of_vertices_in_colored_graph(
		std::string &fname, int verbose_level)
{
	graph_theory::colored_graph CG;

	CG.load(fname, verbose_level);

	return CG.nb_points;
}

void file_io::read_solutions_and_tally(
		std::string &fname, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::read_solutions_and_tally" << endl;
	}

	int nb_solutions;
	int solution_size = sz;
	long int *Sol;
	int i, j;

	std::vector<std::vector<long int> > Solutions;


	read_solutions_from_file_size_is_known(fname,
			Solutions, solution_size,
			verbose_level);

	nb_solutions = Solutions.size();

	Sol = NEW_lint(nb_solutions * solution_size);
	for (i = 0; i < nb_solutions; i++) {
		for (j = 0; j < solution_size; j++) {
			Sol[i * solution_size + j] = Solutions[i][j];
		}
	}


	cout << "nb_solutions = " << nb_solutions << endl;

	data_structures::tally T;

	T.init_lint(Sol, nb_solutions * solution_size, true, 0);
	cout << "tally:" << endl;
	T.print(true);
	cout << endl;


	int *Pts;
	int nb_pts;
	int multiplicity = 4;

	T.get_data_by_multiplicity(
			Pts, nb_pts, multiplicity, verbose_level);

	cout << "multiplicity " << multiplicity
			<< " number of pts = " << nb_pts << endl;
	Int_vec_print(cout, Pts, nb_pts);
	cout << endl;


	FREE_lint(Sol);
	if (f_v) {
		cout << "file_io::read_solutions_and_tally done" << endl;
	}


}


void file_io::extract_from_makefile(
		std::string &fname,
		std::string &label,
		int f_tail, std::string &tail,
		std::vector<std::string> &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	int nb_lines;
	long int sz;

	if (f_v) {
		cout << "file_io::extract_from_makefile " << fname << endl;
		cout << "trying to read file " << fname << " of size "
			<< file_size(fname) << endl;
	}

	sz = file_size(fname);
	if (sz < 0) {
		cout << "file_io::extract_from_makefile file size is -1" << endl;
		exit(1);
	}


	buf = NEW_char(sz + 1);



	{
		ifstream fp(fname);
		int f_found;
		int f_has_been_found = false;

		nb_lines = 0;
		while (true) {
			if (fp.eof()) {
				break;
			}

			//cout << "count_number_of_lines_in_file "
			// "reading line, nb_sol = " << nb_sol << endl;
			fp.getline(buf, sz, '\n'); // ToDo

			f_found = false;

			if (strncmp(buf, label.c_str(), label.length()) == 0) {
				f_found = true;
			}
			if (f_found && f_tail) {
				if (strncmp(buf + label.length(),
						tail.c_str(), tail.length()) != 0) {
					f_found = false;
				}
			}

			if (f_found) {
				f_has_been_found = true;
				if (f_v) {
					cout << "file_io::extract_from_makefile "
							"found label " << label
							<< " at line " << nb_lines << endl;
				}
				string s;

				s.assign(buf);
				text.push_back(s);
				while (true) {
					if (fp.eof()) {
						break;
					}
					fp.getline(buf, sz, '\n'); // ToDo
					if (strlen(buf) == 0) {
						break;
					}
					s.assign(buf);
					text.push_back(s);
				}
				break;
			}
			nb_lines++;
		}
		if (!f_has_been_found) {
			cout << "label not be found: " << label << endl;
			while (true) {
				;
			}
		}
	}
	FREE_char(buf);

	if (f_v) {
		cout << "file_io::extract_from_makefile done" << endl;
	}
}



void file_io::count_solutions_in_list_of_files(
		int nb_files, std::string *fname,
		int *List_of_cases, int *&Nb_sol_per_file,
		int solution_size,
		int f_has_final_test_function,
		int (*final_test_function)(long int *data, int sz,
				void *final_test_data, int verbose_level),
		void *final_test_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, l, j, line_number;
	long int *data;
	int nb_fail = 0;
	data_structures::string_tools ST;



	if (f_v) {
		cout << "file_io::count_solutions_in_list_of_files: "
				"reading " << nb_files << " files" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		if (f_vv) {
			for (i = 0; i < nb_files; i++) {
				cout << fname[i] << endl;
			}
		}
	}

	data = NEW_lint(solution_size);
	Nb_sol_per_file = NEW_int(nb_files);

	for (i = 0; i < nb_files; i++) {

		int the_case;

		the_case = List_of_cases[i];
		Nb_sol_per_file[i] = 0;

		{
			ifstream f(fname[i]);
			char *buf;
			long int a;
			char *p_buf;

			if (f_v) {
				cout << "reading file " << fname[i]
					<< " the_case = " << the_case << " of size "
					<< file_size(fname[i]) << endl;
			}
			if (file_size(fname[i]) <= 0) {
				cout << "file " << fname[i] << " does not exist" << endl;
				exit(1);
			}
			line_number = 0;
			while (true) {

				if (f.eof()) {
					break;
				}
				{
					string S;
					getline(f, S);

					l = S.length();


					buf = NEW_char(l + 1);

					//cout << "read line of length " << l << " : " << S << endl;
					for (j = 0; j < l; j++) {
						buf[j] = S[j];
					}
					buf[l] = 0;
				}
				if (false) {
					cout << "line " << line_number << " read: " << buf << endl;
				}

				p_buf = buf;


				ST.s_scan_lint(&p_buf, &a);

				// size of the set

				if (a == -1) {
					break;
				}

				for (j = 0; j < solution_size; j++) {
					ST.s_scan_lint(&p_buf, &a);
					data[j] = a;
				}


				if (f_has_final_test_function) {
					if (!(*final_test_function)(data, solution_size,
							final_test_data, verbose_level - 1)) {
						if (f_vvv) {
							cout << "file_io::count_solutions_in_list_of_files "
									"solution fails the final test, "
									"skipping" << endl;
						}
						nb_fail++;
						continue;
					}
				}

				FREE_char(buf);
				Nb_sol_per_file[i]++;

			}

		} // while

		if (f_v) {
			cout << "file " << fname[i] << " has " << Nb_sol_per_file[i]
				<< " solutions and " << nb_fail
				<< " false positives" << endl;
		}

	} // next i

	FREE_lint(data);

	if (f_v) {
		cout << "file_io::count_solutions_in_list_of_files done" << endl;
	}
}


void file_io::read_file_as_array_of_strings(
		std::string &fname,
	std::vector<std::string> &Lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::read_file_as_array_of_string " << fname << endl;
	}


	{

		ifstream fp(fname);

		string str;

		while (true) {
			if (fp.eof()) {
				break;
			}
			getline(fp, str);
			Lines.push_back(str);
		}
	}


	if (f_v) {
		cout << "file_io::read_file_as_array_of_string "
				"nb_lines = " << Lines.size() << endl;
	}

	if (f_v) {
		cout << "file_io::read_file_as_array_of_string done" << endl;
	}
}

void file_io::serialize_file_names(
	std::string &fname_list_of_file,
	std::string &output_mask,
	int &nb_files,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::serialize_file_names "
				<< fname_list_of_file << " to " << output_mask << endl;
	}


	data_structures::string_tools ST;

	vector<std::string> Lines;
	int i;

	read_file_as_array_of_strings(
			fname_list_of_file,
		Lines,
		verbose_level);


	nb_files = Lines.size();

	if (f_v) {
		cout << "interface_toolkit::worker "
				"serialize_file_names_output_mask = "
				<< output_mask << endl;
	}

	for (i = 0; i < nb_files; i++) {

		string str, cmd;

		cout << "i=" << i << " / " << nb_files << " fname=" << Lines[i] << endl;

		str = ST.printf_d(
				output_mask, i);

		cmd = "mv " + Lines[i] + " " + str;


		cout << "executing : " << cmd << endl;
		system(cmd.c_str());

	}

	if (f_v) {
		cout << "file_io::serialize_file_names done" << endl;
	}
}

void file_io::read_error_pattern_from_output_file(
		std::string &fname,
		int nb_lines,
		std::vector<std::vector<int> > &Error1,
		std::vector<std::vector<int> > &Error2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	data_structures::string_tools ST;
	int h, i, j, l, idx;
	char c;
	int a;
	long int N;

	if (f_v) {
		cout << "file_io::read_error_pattern_from_output_file" << endl;
		cout << "file_io::read_error_pattern_from_output_file trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
	}

	N = file_size(fname);
	if (N < 0) {
		cout << "file_io::read_error_pattern_from_output_file "
				"the file " << fname << " does not exist" << endl;
		return;
	}

	buf = NEW_char(N + 1);

	{
		ifstream f(fname);

		while (!f.eof()) {
			f.getline(buf, N + 1, '\n'); // ToDo
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			if (strncmp(buf, "undetected error code", 21) == 0) {


				sscanf(buf + 21, "%d", &idx);


				vector<int> error;
				for (l = 0; l < nb_lines; l++) {
					f.getline(buf, N + 1, '\n'); // ToDo
					for (h = 0; h < 2; h++) {
						for (i = 0; i < 8; i++) {
							for (j = 0; j < 2; j++) {
								c = buf[12 + h * 25 + i * 3 + j];
								if (c >= '0' && c <= '9') {
									a = (int) (c - '0');
								}
								else {
									a = 10 + (int) (c - 'a');
								}
								error.push_back(a);
							}
						}

					}
				}
				if (idx == 1) {
					Error1.push_back(error);
				}
				else {
					Error2.push_back(error);
				}

			}

		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "file_io::read_error_pattern_from_output_file done" << endl;
	}
}


void file_io::read_gedcom_file(
		std::string &fname,
		std::vector<std::vector<std::string> > &Data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;
	long int N;

	if (f_v) {
		cout << "file_io::read_gedcom_file" << endl;
		cout << "file_io::read_gedcom_file trying to read file "
			<< fname << " of size " << file_size(fname) << endl;
	}

	N = file_size(fname);
	if (N < 0) {
		cout << "file_io::read_gedcom_file "
				"the file " << fname << " does not exist" << endl;
		return;
	}

	{
		ifstream f(fname);

		int line_cnt;

		line_cnt = 0;
		while (!f.eof()) {
			string s;
			std::string part1;
			std::string part2;
			std::string part3;
			std::vector<std::string> data;
			getline(f, s);

			//l = s.length();
			line_cnt++;



			int len;

			len = s.length();
			if (len < 1) {
				continue;
			}
			std::size_t found;

			found = s.find(' ');
			if (found == std::string::npos) {

				cout << "parse error in line " << line_cnt << " : " << s << endl;
				exit(1);
			}
			part1 = s.substr (0, found);
			std::string rest = s.substr (found + 1, len - found - 1);



			found = rest.find(' ');
			if (found == std::string::npos) {

				part2 = rest;
			}
			else {
				part2 = rest.substr (0, found);
				part3 = rest.substr (found + 1, len - found - 1);
			}
			data.push_back(part1);
			data.push_back(part2);
			data.push_back(part3);
			Data.push_back(data);

		}
	}
	if (f_v) {
		cout << "file_io::read_gedcom_file done" << endl;
	}
}

void file_io::write_solutions_as_index_set(
		std::string &fname_solutions,
		int *Sol, int nb_sol, int width, int sum,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::write_solutions_as_index_set" << endl;
	}

	{
		ofstream fp(fname_solutions);
		int i, j;
		int a, cnt;

		for (i = 0; i < nb_sol; i++) {
			fp << sum;
			cnt = 0;
			for (j = 0; j < width; j++) {
				a = Sol[i * width + j];
				if (a) {
					cnt++;
					fp << " " << j;
				}
			}
			if (cnt != sum) {
				cout << "file_io::write_solutions_as_index_set cnt != sum" << endl;
				exit(1);
			}
			fp << endl;
		}
		fp << -1 << " " << nb_sol << endl;
	}

	if (f_v) {
		cout << "Written file " << fname_solutions << " of size "
				<< file_size(fname_solutions) << endl;
	}

	if (f_v) {
		cout << "file_io::write_solutions_as_index_set done" << endl;
	}
}

int file_io::count_number_of_data_lines_in_spreadsheet(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_io::count_number_of_data_lines_in_spreadsheet" << endl;
	}

	data_structures::spreadsheet S;
	int nb_data;

	if (f_v) {
		cout << "file_io::count_number_of_data_lines_in_spreadsheet fname=" << fname << endl;
	}
	S.read_spreadsheet(fname, 0 /*verbose_level*/);

	nb_data = S.nb_rows - 1;

	return nb_data;

}

void file_io::read_graph_dimacs_format(
		std::string &fname,
		int &nb_V, std::vector<std::vector<int>> &Edges,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int sz;

	if (f_v) {
		cout << "file_io::read_graph_dimacs_format "
				"fname = " << fname << endl;
	}

	sz = file_size(fname);

	if (sz <= 0) {
		cout << "file_io::read_graph_dimacs_format The file "
			<< fname << " does not exist" << endl;
		exit(1);
	}


	std::vector<std::string> Lines;

	read_file_as_array_of_strings(
			fname,
			Lines,
			verbose_level);

	int nb_E;

	if (Lines.size() < 1) {
		cout << "file_io::read_graph_dimacs_format the file is empty" << endl;
		exit(1);
	}
	sscanf(Lines[0].c_str(), "p edge %d %d", &nb_V, &nb_E);
	if (f_v) {
		cout << "file_io::read_graph_dimacs_format a graph on "
				<< nb_V << " vertices with " << nb_E << " edges" << endl;
	}
	if (Lines.size() < 1 + nb_E) {
		cout << "file_io::read_graph_dimacs_format the file is too short" << endl;
		exit(1);
	}

	int i, a, b;

	for (i = 0; i < nb_E; i++) {


		sscanf(Lines[1 + i].c_str(), "e %d %d", &a, &b);

		vector<int> v;

		v.push_back(a - 1);
		v.push_back(b - 1);

		Edges.push_back(v);
	}


	if (f_v) {
		cout << "file_io::read_graph_dimacs_format done" << endl;
	}
}

void file_io::read_graph_Brouwer_format(
		std::string &fname,
		int &nb_V, int *&Adj,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int sz;

	if (f_v) {
		cout << "file_io::read_graph_Brouwer_format "
				"fname = " << fname << endl;
	}

	sz = file_size(fname);

	if (sz <= 0) {
		cout << "file_io::read_graph_Brouwer_format The file "
			<< fname << " does not exist" << endl;
		exit(1);
	}


	std::vector<std::string> Lines;

	read_file_as_array_of_strings(
			fname,
			Lines,
			verbose_level);

	if (Lines.size() < 1) {
		cout << "file_io::read_graph_Brouwer_format the file is empty" << endl;
		exit(1);
	}
	sscanf(Lines[0].c_str(), "n%d", &nb_V);
	if (f_v) {
		cout << "file_io::read_graph_Brouwer_format a graph on "
				<< nb_V << " vertices" << endl;
	}
	if (Lines.size() < 1 + nb_V) {
		cout << "file_io::read_graph_Brouwer_format the file is too short" << endl;
		exit(1);
	}

	Adj = NEW_int(nb_V * nb_V);
	Int_vec_zero(Adj, nb_V * nb_V);

	int i, j, h;

	data_structures::string_tools ST;

	for (i = 0; i < nb_V; i++) {

		string str;

		str = Lines[1 + i];


		std::vector<std::string> adj_list;

		ST.parse_comma_separated_list(
				str,
				adj_list,
				0 /*verbose_level */);

		if (f_v) {
			cout << "file_io::read_graph_Brouwer_format neighbors of vertex " << i << " : ";
			for (h = 0; h < adj_list.size(); h++) {
				sscanf(adj_list[h].c_str(), "%d", &j);
				cout << j << " ";
			}
			cout << endl;
		}

		for (h = 0; h < adj_list.size(); h++) {
			sscanf(adj_list[h].c_str(), "%d", &j);
			Adj[i * nb_V + j] = 1;
			Adj[j * nb_V + i] = 1;
		}

	}


	if (f_v) {
		cout << "file_io::read_graph_Brouwer_format done" << endl;
	}
}



}}}



