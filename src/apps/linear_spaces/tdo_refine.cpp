// tdo_refine.cpp
//
// Anton Betten
//
// started:  Dec 26 2006

#include "orbiter.h"

using namespace std;


using namespace orbiter;


//#define MY_BUFSIZE 1000000

const char *version = "tdo_refine 7/31 2008";



void print_usage()
{
	cout << "usage: tdo_refine.out [options] <tdo-file>" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <verbose_level> " << endl;
	cout << "   Specify the amount of text output" << endl;
	cout << "   the higher verbose_level is, the more output" << endl;
	cout << "   verbose_level = 0 means no output" << endl;
	cout << "-lambda3 <lambda3> <k>" << endl;
	cout << "   do a 3-design parameter refinement" << endl;
	cout << "   lambda3 is the number of blocks on 3 points" << endl;
	cout << "   k is the (constant) block size" << endl;
	cout << "-scale <n>" << endl;
	cout << "   when doing the refinement," << endl;
	cout << "   consider only refinements where each class in" << endl;
	cout << "   the refinement has size a multiple of n" << endl;
	cout << "-select <label>" << endl;
	cout << "   select the TDO whose label is <label>" << endl;
	cout << "-range <f> <l>" << endl;
	cout << "   select the TDO in interval [f..f+l-1]" << endl;
	cout << "   where counting starts from 1 (!!!)" << endl;
	cout << "-solution <n> <file>" << endl;
	cout << "   Read the solutions to system <n> from <file>" << endl;
	cout << "   rather than trying to compute them" << endl;
	cout << "   This option can appear repeatedly" << endl;
	cout << "-o1 <n>" << endl;
	cout << "   omit the last n blocks from refinement (1st system - types)" << endl;
	cout << "-o2 <n>" << endl;
	cout << "   omit the last n blocks from refinement (2nd system - distribution)" << endl;
	cout << "-D1_upper_bound_x0 <n>" << endl;
	cout << "   upper bound <n> for x[0] in the first system (column refinement only!)" << endl;
	cout << "-reverse" << endl;
	cout << "   reverseordering of refinements" << endl;
	cout << "-reverse_inverse" << endl;
	cout << "   reverse ordering of refinements increasing" << endl;
	cout << "-nopacking" << endl;
	cout << "   Do not use inequalities based on packing numbers" << endl;
	cout << "-dual_is_linear_space" << endl;
	cout << "   Dual is a linear space, too (affect refine rows)" << endl;
	cout << "-once" << endl;
	cout << "   When refining, only find the first refinement" << endl;
	cout << "-mckay" << endl;
	cout << "   Use McKay solver for solving the second system" << endl;
}


int main(int argc, char **argv)
{
	tdo_refinement *G;
	int verbose_level = 0;
	int i;
	os_interface Os;

	cout << version << endl;

	if (argc <= 1) {
		print_usage();
		exit(1);
	}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
	}

	G = NEW_OBJECT(tdo_refinement);
	

	G->init(verbose_level);
	G->read_arguments(argc, argv);
	G->main_loop(verbose_level);
	
	cout << "time: ";
	Os.time_check(cout, G->t0);
	cout << endl;
	FREE_OBJECT(G);
}

