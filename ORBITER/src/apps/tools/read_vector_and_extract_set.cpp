/*
 * read_vector_and_extract_set.cpp
 *
 *  Created on: Aug 21, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


using namespace orbiter;


#define MY_BUFSIZE ONE_MILLION

int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_file = FALSE;
	const char *file_name = NULL;
	int f_extract = FALSE;
	long int extract_value = 0;
	const char *extract_fname = NULL;
	int f_extract_values = FALSE;
	int extract_value_from = 0;
	int extract_value_to = 0;
	const char *extract_fname_mask = NULL;

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
		else if (strcmp(argv[i], "-extract") == 0) {
			f_extract = TRUE;
			extract_value = atoi(argv[++i]);
			extract_fname = argv[++i];
			cout << "-extract " << extract_value << " " << extract_fname << endl;
		}
		else if (strcmp(argv[i], "-extract_values") == 0) {
			f_extract_values = TRUE;
			extract_value_from = atoi(argv[++i]);
			extract_value_to = atoi(argv[++i]);
			extract_fname_mask = argv[++i];
			cout << "-extract_values " << extract_value_from << " " << extract_value_to << " " << extract_fname_mask << endl;
		}
	}
	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
	}
	{
		long int d;
		long int i;
		char *Data;

		cout << "reading TR from file " << file_name << endl;
		{
			ifstream fp(file_name, ios::binary);


			fp.read((char *) &d, sizeof(long int));

			Data = NEW_char(d);
			for (i = 0; i < d; i++) {
				fp.read((char *) &Data [i], sizeof(char));
			}
		}

		if (f_extract) {
			long int cnt;
			cnt = 0;
			for (i = 0; i < d; i++) {
				if (Data[i] == extract_value) {
					cnt++;
				}
			}
			long int *Set;
			long int sz;

			Set = NEW_lint(cnt);
			sz = 0;
			for (i = 0; i < d; i++) {
				if (Data[i] == extract_value) {
					Set[sz++] = i;
				}
			}


			file_io Fio;

			Fio.lint_matrix_write_csv(extract_fname, Set, sz, 1);


		}
		if (f_extract_values) {
			int extract_value;

			for (extract_value = extract_value_from; extract_value <= extract_value_to; extract_value++) {

				cout << "extracting value " << extract_value << endl;

				long int cnt;
				cnt = 0;
				for (i = 0; i < d; i++) {
					if (Data[i] == extract_value) {
						cnt++;
					}
				}
				cout << "We found " << cnt << " entries of value " << extract_value << endl;


				uint32_t *Set;
				uint32_t sz;

				Set = (uint32_t *) NEW_int(cnt);
				sz = 0;
				for (i = 0; i < d; i++) {
					if (Data[i] == extract_value) {
						Set[sz++] = i;
					}
				}

				cout << "We extracted all " << sz << " entries of value " << extract_value << endl;


				char fname[1000];

				sprintf(fname, extract_fname_mask, extract_value);
				cout << "We will write the file " << fname << endl;

				{
					ofstream fp(fname, ios::binary);

					fp.write((char *) &sz, sizeof(uint32_t));
					for (i = 0; i < sz; i++) {
						fp.write((char *) &Set[i], sizeof(uint32_t));
					}
				}
				cout << "We are done writing the file " << fname << endl;
				FREE_int((int *) Set);
			}

		}


	}


}


