// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::layer5_applications;

int build_number =
#include "../../../build_number"
;

//! This is the Orbiter front-end. It creates an object of type user_interface::orbiter_top_level_session and executes the command line


std::string do_orbiter_session(
		int argc, const char **argv, int verbose_level);


int main(
		int argc, const char **argv)
{

	//cout << "orbiter.out main" << endl;


	int verbose_level = 1;

	//int f_v = (verbose_level >= 1);

	string delta_t;


	delta_t = do_orbiter_session(argc, argv, verbose_level);


	cout << "The Orbiter session is finished." << endl;
	cout << "User time: " << delta_t << endl;

}

std::string do_orbiter_session(
		int argc, const char **argv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int t0;


	cout << "Welcome to Orbiter! Your build number is " << build_number + 1 << "." << endl;

	if (f_v) {
		cout << "do_orbiter_session" << endl;
	}
	user_interface::orbiter_top_level_session Top_level_session;


	user_interface::The_Orbiter_top_level_session = &Top_level_session;


	if (f_v) {
		cout << "do_orbiter_session "
				"before Top_level_session.execute_command_line" << endl;
	}
	Top_level_session.execute_command_line(
			argc, argv, verbose_level);
	if (f_v) {
		cout << "do_orbiter_session "
				"after Top_level_session.execute_command_line" << endl;
	}


	orbiter_kernel_system::os_interface Os;
	string str;

	t0 = Top_level_session.Orbiter_session->t0;

	str = Os.stringify_time_difference(t0);

	return str;
}




