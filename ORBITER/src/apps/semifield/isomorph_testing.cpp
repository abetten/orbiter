/*
 * isomorph_testing.cpp
 *
 *  Created on: May 8, 2019
 *      Author: betten
 *
 *  originally created on March 7, 2018
 *
 */


#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;



// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	semifield_classify_with_substructure SCWS;


	t0 = os_ticks();

	SCWS.read_arguments(argc, argv, verbose_level);

	if (!SCWS.f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}


	SCWS.init(verbose_level);
		// recovers the classification of the substructures

	SCWS.read_data(verbose_level);
		// reads the files
		// which contain the liftings of the substructures

	SCWS.Sub->compute_orbits(verbose_level);
		// computes the orbits in all cases where needed

	SCWS.Sub->compute_flag_orbits(verbose_level);


	if (SCWS.f_load_classification) {
		SCWS.load_classification(verbose_level);
		SCWS.load_flag_orbits(verbose_level);
	}
	else {
		// this is the most time consuming step:
		SCWS.classify_semifields(verbose_level);

		// saves the classification and the flag orbits
		// to file afterwards
	}


	SCWS.identify_semifield(verbose_level);


	SCWS.identify_semifields_from_file(
				verbose_level);


	SCWS.latex_report(
				verbose_level);


	SCWS.generate_source_code(verbose_level);



#if 0
	cout << "before freeing Gr" << endl;
	FREE_OBJECT(Sub.Gr);
	cout << "before freeing transporter1" << endl;
	FREE_int(Sub.transporter1);
	FREE_int(Sub.transporter2);
	FREE_int(Sub.transporter3);
	cout << "before freeing Basis1" << endl;
	FREE_int(Sub.Basis1);
	FREE_int(Sub.Basis2);
	FREE_int(Sub.B);
	cout << "before freeing Flag_orbits" << endl;
	FREE_OBJECT(Sub.Flag_orbits);

	cout << "before freeing L3" << endl;
	FREE_OBJECT(Sub.L3);
	cout << "before freeing L2" << endl;
	FREE_OBJECT(L2);
	cout << "before freeing SC" << endl;
	FREE_OBJECT(Sub.SC);
	cout << "before freeing F" << endl;
	FREE_OBJECT(F);
	cout << "before leaving scope" << endl;
	}
	cout << "after leaving scope" << endl;

#endif



	the_end(t0);
}



