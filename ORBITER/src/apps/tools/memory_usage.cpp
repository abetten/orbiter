// memory_usage.cpp
//
// Anton Betten
// August 27, 2018
//
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;

	int f_file_mask = FALSE;
	int range_first = 0;
	int range_len = 0;
	const char *fname_mask;

	int nb_extra_files = 0;
	const char *extra_files[1000];


	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file_mask") == 0) {
			f_file_mask = TRUE;
			range_first = atoi(argv[++i]);
			range_len = atoi(argv[++i]);
			fname_mask = argv[++i];
			cout << "-file_mask " << range_first << " " << range_len
					<< " " << fname_mask << endl;
			}
		else if (strcmp(argv[i], "-extra_file") == 0) {
			extra_files[nb_extra_files] = argv[++i];
			cout << "-extra_file " << extra_files[nb_extra_files] << endl;
			nb_extra_files++;
			}
		}

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);


	int nb_files = 0;
	int h;

	if (f_file_mask) {
		nb_files += range_len;
	}
	nb_files += nb_extra_files;


	char fname[1000];
	char **fnames;
	fnames = NEW_pchar(nb_files);
	h = 0;
	if (f_file_mask) {
		for (i = 0; i < range_len; i++) {
			sprintf(fname, fname_mask, range_first + i);
			fnames[h] = NEW_char(strlen(fname) + 1);
			strcpy(fnames[h], fname);
			cout << "created file name " << h << " as " << fnames[h] << endl;
			h++;
		}
	}
	for (i = 0; i < nb_extra_files; i++) {
		fnames[h] = NEW_char(strlen(extra_files[i]) + 1);
		strcpy(fnames[h], extra_files[i]);
		cout << "created file name " << h << " as " << fnames[h] << endl;
		h++;
	}
	if (h != nb_files) {
		cout << "h != nb_files" << endl;
		exit(1);
	}

	int idx;
	mem_object_registry **M;

	M = (mem_object_registry **) NEW_pvoid(nb_files);
	for (idx = 0; idx < nb_files; idx++) {

		cout << "file " << idx << " / " << nb_files << " is "
				<< fnames[idx] << ":" << endl;
		spreadsheet *S;
		char *p;
		int N;
		long int a;

		S = NEW_OBJECT(spreadsheet);

		cout << "Reading table " << fnames[idx] << endl;
		if (file_size(fnames[idx]) <= 0) {
			cout << "error: the file " << fnames[idx]
				<< " does not exist" << endl;
			exit(1);
		}
		S->read_spreadsheet(fnames[idx], 0 /*verbose_level*/);
		cout << "Table " << fnames[idx] << " has been read" << endl;

		N = S->nb_rows - 1;


		M[idx] = NEW_OBJECT(mem_object_registry);
		M[idx]->allocate(N, verbose_level);
		for (i = 0; i < N; i++) {

			p = S->get_string(i + 1, 1);
			//sscanf(p, "%d", &a);
			std::stringstream ss;
			ss << std::hex << p + 2;
			ss >> a;
			M[idx]->entries[i].pointer = (void *) a;

			M[idx]->entries[i].time_stamp = S->get_int(i + 1, 2);

			M[idx]->entries[i].set_type_from_string(S->get_string(i + 1, 3));
			M[idx]->entries[i].object_n = S->get_int(i + 1, 4);
			M[idx]->entries[i].object_size_of = S->get_int(i + 1, 5);

			p = S->get_string(i + 1, 6);
			M[idx]->entries[i].extra_type_info = NEW_char(strlen(p) + 1);
			strcpy((char *) M[idx]->entries[i].extra_type_info, p);

			p = S->get_string(i + 1, 7);
			M[idx]->entries[i].source_file = NEW_char(strlen(p) + 1);
			strcpy((char *) M[idx]->entries[i].source_file, p);

			M[idx]->entries[i].source_line = S->get_int(i + 1, 8);
			}
		M[idx]->nb_entries_used = N;
		//M[idx]->dump();

		cout << "sorting by size:" << endl;
		M[idx]->sort_by_size(verbose_level);

		strcpy(fname, fnames[idx]);
		chop_off_extension(fname);
		strcat(fname, "_by_size.csv");
		cout << "writing file " << fname << endl;
		M[idx]->dump_to_csv_file(fname);


		cout << "sorting by type:" << endl;
		M[idx]->sort_by_type(verbose_level);

		strcpy(fname, fnames[idx]);
		chop_off_extension(fname);
		strcat(fname, "_by_type.csv");
		cout << "writing file " << fname << endl;
		M[idx]->dump_to_csv_file(fname);


		cout << "file " << idx << " / " << nb_files << " is "
			<< fnames[idx] << ":" << endl;
		cout << "usage by location:" << endl;
		M[idx]->sort_by_location_and_get_frequency(verbose_level);
	}



}


