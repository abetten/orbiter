// ovoid.h
// 
// Anton Betten
// May 16, 2011
//
//
// 
//
//

using namespace orbiter;


typedef  int * pint;





// global data and global functions:

extern int t0; // the system time when the program started

void usage(int argc, const char **argv);
int check_conditions(int len, int *S, void *data, int verbose_level);
void callback_print_set(std::ostream &ost, int len, int *S, void *data);
//int callback_check_conditions(int len, int *S, void *data, int verbose_level);

