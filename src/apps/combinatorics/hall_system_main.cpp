// hall_system_main.cpp
//
// Anton Betten
// 12/17/2018
//
//
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;

	int f_n = FALSE;
	int n = 0;
	int f_depth = FALSE;
	int depth = 0;
	os_interface Os;

	t0 = Os.os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-mem") == 0) {
			f_memory_debug = TRUE;
			cout << "-mem" << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_depth) {
		cout << "please use option -depth <depth>" << endl;
		exit(1);
		}

	hall_system_classify *H;

	H = NEW_OBJECT(hall_system_classify);

	H->init(argc, argv, n, depth, verbose_level);

	FREE_OBJECT(H);


	the_end(t0);

	if (f_memory_debug) {
		global_mem_object_registry.sort_by_location(verbose_level);
		global_mem_object_registry.dump_to_csv_file("hall_system_classify_leftover_memory.csv");
	}
	//the_end_quietly(t0);
}



