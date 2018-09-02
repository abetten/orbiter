// make_BLT_set.C
//
// Anton Betten
//
// started:  May 9, 2011




#include "orbiter.h"

void print_usage();



void print_usage()
{
	cout << "usage: make_BLT_set.out [options] <q> <k>" << endl;
	cout << "Write the file BLT_<q>_<k>.txt containing the k-th BLT set in Q(4,q)" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <k>" << endl;
	cout << "   verbose level k" << endl;
}

int main(int argc, char **argv)
{
	INT t0 = os_ticks();
	INT verbose_level = 0;
	INT i;
	INT q, k;

	if (argc <= 2) {
		print_usage();
		exit(1);
		}
	for (i = 1; i < argc - 2; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}

		}
	q = atoi(argv[argc - 2]);
	k = atoi(argv[argc - 1]);

	INT f_v = (verbose_level >= 1);

	INT *BLT;
	char fname[1000];

	BLT = BLT_representative(q, k);
	sprintf(fname, "BLT_%ld_%ld.txt", q, k);
	write_set_to_file(fname, BLT, q + 1, verbose_level - 1);
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	the_end(t0);
}

