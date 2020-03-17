// create_object.cpp
// 
// Anton Betten
// June 25, 2011
//
// 
// Creates combinatorial object that can be described as a set of integers.
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;





// global data:

int t0; // the system time when the program started




int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_description = FALSE;
	combinatorial_object_description *Descr;
	int f_save = FALSE;

	os_interface Os;


	t0 = Os.os_ticks();



 	

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-object") == 0) {
			f_description = TRUE;
			Descr = NEW_OBJECT(combinatorial_object_description);
			i += Descr->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-object" << endl;
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			cout << "-save " << endl;
		}
#if 0
		else if (strcmp(argv[i], "-transform") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = FALSE;
			cout << "-transform " << transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
		else if (strcmp(argv[i], "-transform_inverse") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = TRUE;
			cout << "-transform_inverse "
					<< transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
#endif
	}
	
	if (!Descr->f_q) {
		cout << "please specify the field order "
				"using the option -q <q>" << endl;
		exit(1);
	}


	combinatorial_object_create *COC;
	//int j;

	COC = NEW_OBJECT(combinatorial_object_create);

	cout << "before COC->init" << endl;
	COC->init(Descr, verbose_level);
	cout << "after COC->init" << endl;
	


	cout << "we created a set of " << COC->nb_pts << " points, called " << COC->fname << endl;



	cout << "list of points:" << endl;

	cout << COC->nb_pts << endl;
	for (i = 0; i < COC->nb_pts; i++) {
		cout << COC->Pts[i] << " ";
		}
	cout << endl;


	if (f_save) {
		file_io Fio;
		char fname[1000];

		sprintf(fname, "%s", COC->fname);

		cout << "and we will write to the file " << fname << endl;
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_OBJECT(COC);


	the_end(t0);
}



