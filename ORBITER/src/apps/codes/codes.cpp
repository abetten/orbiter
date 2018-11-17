// codes.C
//
// Anton Betten
// December 30, 2003

#include "codes.h"

// global data:

int t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	
	{
	code_generator cg;
	
	cout << argv[0] << endl;
	cg.init(argc, argv);

	cg.main(cg.verbose_level);
	
	cout << "nb_calls_to_finite_field_init=" << nb_calls_to_finite_field_init << endl;
	
	}
	the_end(t0);
	//the_end_quietly(t0);
}

