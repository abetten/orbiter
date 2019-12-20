// hadamard.cpp
// 
// Anton Betten
// December 9, 2014
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;




// global data:

int t0; // the system time when the program started


int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int verbose_level_clique = 0;
	int f_n = FALSE;
	int n = 0;
	int f_draw = FALSE;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-v_clique") == 0) {
			verbose_level_clique = atoi(argv[++i]);
			cout << "-v_clique " << verbose_level_clique << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			cout << "-draw " << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n> to specify n" << endl;
		exit(1);
		}

	hadamard_classify H;

	H.init(n, f_draw, verbose_level, verbose_level_clique);

	the_end(t0);

}

