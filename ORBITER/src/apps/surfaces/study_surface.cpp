// study_surface.cpp
// 
// Anton Betten
// September 12, 2016
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


int main(int argc, const char **argv);






int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int f_q = TRUE;
	int q = 0;
	int f_nb = FALSE;
	int nb = 0;
	


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-nb") == 0) {
			f_nb = TRUE;
			nb = atoi(argv[++i]);
			cout << "-nb " << nb << endl;
			}
		}

	//int f_v = (verbose_level >= 1);
	

	if (f_q && f_nb) {
		surface_study *study;

		study = NEW_OBJECT(surface_study);

		cout << "before study->init" << endl;
		study->init(q, nb, verbose_level);
		cout << "after study->init" << endl;

		cout << "before study->study_intersection_points" << endl;
		study->study_intersection_points(verbose_level);
		cout << "after study->study_intersection_points" << endl;

		cout << "before study->study_line_orbits" << endl;
		study->study_line_orbits(verbose_level);
		cout << "after study->study_line_orbits" << endl;
		
		cout << "before study->study_group" << endl;
		study->study_group(verbose_level);
		cout << "after study->study_group" << endl;
		
		cout << "before study->study_orbits_on_lines" << endl;
		study->study_orbits_on_lines(verbose_level);
		cout << "after study->study_orbits_on_lines" << endl;
		
		cout << "before study->study_find_eckardt_points" << endl;
		study->study_find_eckardt_points(verbose_level);
		cout << "after study->study_find_eckardt_points" << endl;
	
#if 0
		if (study->nb_Eckardt_pts == 6) {
			cout << "before study->study_surface_with_6_eckardt_points" << endl;
			study->study_surface_with_6_eckardt_points(verbose_level);
			cout << "after study->study_surface_with_6_eckardt_points" << endl;
			}
#endif
		
		}

}


