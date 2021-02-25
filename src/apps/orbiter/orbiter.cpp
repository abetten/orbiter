// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;

int build_number =
#include "../../../build_number"
;



int main(int argc, const char **argv)
{

	//cout << "orbiter.out main" << endl;

	orbiter_top_level_session Top_level_session;
	int i;



	The_Orbiter_top_level_session = &Top_level_session;


	// setup:


	cout << "Welcome to Orbiter!  Your build number is " << build_number << "." << endl;
	cout << "A user's guide is available here: " << endl;
	cout << "https://www.math.colostate.edu/~betten/orbiter/users_guide.pdf" << endl;
	cout << "The sources are available here: " << endl;
	cout << "https://github.com/abetten/orbiter" << endl;
	cout << "An example makefile with many commands from the user's guide is here: " << endl;
	cout << "https://github.com/abetten/orbiter/tree/master/examples/users_guide/makefile" << endl;
#ifdef SYSTEMUNIX
	cout << "SYSTEMUNIX is defined" << endl;
#endif
#ifdef SYSTEMWINDOWS
	cout << "SYSTEMWINDOWS is defined" << endl;
#endif
#ifdef SYSTEM_IS_MACINTOSH
	cout << "SYSTEM_IS_MACINTOSH is defined" << endl;
#endif
	cout << "sizeof(int)=" << sizeof(int) << endl;
	cout << "sizeof(long int)=" << sizeof(long int) << endl;


	std::string *Argv;

	convert_arguments(argc, argv, Argv);


	i = Top_level_session.startup_and_read_arguments(argc, Argv, 1);



	int verbose_level;

	verbose_level = Top_level_session.Orbiter_session->verbose_level;

	int f_v = (verbose_level > 1);

	if (f_v) {
		cout << "main, before Top_level_session.handle_everything" << endl;
	}

	Top_level_session.handle_everything(argc, Argv, i, verbose_level);

	if (f_v) {
		cout << "main, after Top_level_session.handle_everything" << endl;
	}

	cout << "Orbiter session finished." << endl;
	cout << "User time: ";
	the_end(Top_level_session.Orbiter_session->t0);

}



