/*
 * isomorph_testing.cpp
 *
 *  Created on: Oct 26, 2024
 *      Author: betten
 */


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


	time_check(cout, t0);
	cout << "before SCWS.init" << endl;

	SCWS.init(verbose_level);
		// recovers the classification of the substructures
		// calls Sub->L3->init_level_three

	time_check(cout, t0);
	cout << "before SCWS.init" << endl;



	//L3->compute_level_three(verbose_level);





	if (SCWS.f_decomposition_matrix_level_3) {

		cout << "decomposition matrix at level 3" << endl;


		time_check(cout, t0);
		cout << " : ";
		cout << "before L3->recover_level_three_from_file" << endl;

		SCWS.Sub->L3->recover_level_three_from_file(
				TRUE /* f_read_flag_orbits */,
				verbose_level);

		time_check(cout, t0);
		cout << " : ";
		cout << "after L3->recover_level_three_from_file" << endl;




		time_check(cout, t0);
		cout << " : ";
		cout << "before SCWS.Sub->init" << endl;

		SCWS.Sub->init();
			// allocates the arrays and matrices

		time_check(cout, t0);
		cout << " : ";
		cout << "after SCWS.Sub->init" << endl;

		time_check(cout, t0);
		cout << " : ";
		cout << "before SCWS.Sub->load_flag_orbits" << endl;

		SCWS.Sub->Flag_orbits = NEW_OBJECT(flag_orbits);
		SCWS.load_flag_orbits(verbose_level);

		time_check(cout, t0);
		cout << " : ";
		cout << "after SCWS.Sub->load_flag_orbits" << endl;

		time_check(cout, t0);
		cout << " : ";
		cout << "before SCWS.Sub->load_classification" << endl;

		SCWS.load_classification(verbose_level);

		time_check(cout, t0);
		cout << " : ";
		cout << "after SCWS.Sub->load_classification" << endl;


		SCWS.decomposition(verbose_level);
	}
	else {

		time_check(cout, t0);
		cout << " : ";
		cout << "before L3->recover_level_three_downstep" << endl;
		SCWS.Sub->L3->recover_level_three_downstep(verbose_level);

		time_check(cout, t0);
		cout << " : ";
		cout << "after L3->recover_level_three_downstep" << endl;

		cout << "before L3->recover_level_three_from_file" << endl;
		SCWS.Sub->L3->recover_level_three_from_file(
				TRUE /* f_read_flag_orbits */, verbose_level);
		cout << "after L3->recover_level_three_from_file" << endl;


		SCWS.read_data(verbose_level);
			// reads the files
			// which contain the liftings of the substructures

		SCWS.Sub->compute_orbits(verbose_level);
			// computes the orbits in all cases where needed

		SCWS.Sub->compute_flag_orbits(verbose_level);
			// initializes Fo_first and Flag_orbits

		SCWS.Sub->init();
			// allocated the arrays and matrices

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


		if (SCWS.f_report) {
			SCWS.latex_report(verbose_level);
		}

		SCWS.generate_source_code(verbose_level);
	}

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




