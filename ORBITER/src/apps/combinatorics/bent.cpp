/*
 * bent.cpp
 *
 *  Created on: Oct 16, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;

int main(int argc, char **argv);



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;

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
		cout << "please use -n <n>" << endl;
		exit(1);
	}

	bent_function_classify *BFC;

	BFC = NEW_OBJECT(bent_function_classify);

	BFC->init(n, verbose_level);

	BFC->search(verbose_level);

	FREE_OBJECT(BFC);
}


