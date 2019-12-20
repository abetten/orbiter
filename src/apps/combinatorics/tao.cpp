// tao.cpp
// 
// Anton Betten
// May 21, 2017
//
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


void test_heisenberg(int n, int q, int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_heisenberg = FALSE;
	int n, q;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-heisenberg") == 0) {
			f_heisenberg = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-heisenberg " << n << " " << q << endl;
			}
		}
	
	if (f_heisenberg) {
		test_heisenberg(n, q, verbose_level);
		}
	the_end(t0);
}





void test_heisenberg(int n, int q, int verbose_level)
{
	difference_set_in_heisenberg_group *DS;
	finite_field *F;

	DS = NEW_OBJECT(difference_set_in_heisenberg_group);
	

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);


	DS->init(n, F, verbose_level);



	if (n != 2) {
		cout << "from now on, we are expecting that n=2" << endl;
		exit(1);
		}
	if (q != 3) {
		cout << "from now on, we are expecting that q=3" << endl;
		exit(1);
		}

	DS->do_n2q3(verbose_level);

	
}



