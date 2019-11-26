// burnside.cpp
//
// Anton Betten
// February 26, 2015
//

#include "orbiter.h"


using namespace std;



using namespace orbiter;
using namespace orbiter::top_level;

int t0;





int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_n = FALSE;
	int n = 0;
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
		}
	


	if (!f_n) {
		cout << "please specify -n <n>" << endl;
		exit(1);
		}

	character_table_burnside *CTB;

	CTB = NEW_OBJECT(character_table_burnside);

	CTB->do_it(n, verbose_level);

	FREE_OBJECT(CTB);
	
}



