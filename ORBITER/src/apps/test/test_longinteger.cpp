// test_longinteger.C
//
// Anton Betten
// August 28, 2018

#include "orbiter.h"

using namespace orbiter;



int main(int argc, char **argv)
{
	int t0 = os_ticks();
	int i;
	int verbose_level = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}


	f_memory_debug = TRUE;

	{
	longinteger_domain D;

	longinteger_object A, B, G, U, V;

	A.create(10);
	B.create(23);

	cout << "before D.extended_gcd" << endl;
	D.extended_gcd(A, B,
			G, U, V, verbose_level);
	cout << "after D.extended_gcd" << endl;

	}

	global_mem_object_registry.manual_dump_with_file_name("memory_dump_at_end.csv");


	time_check(cout, t0);
	cout << endl;
}

