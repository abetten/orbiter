// process.cpp
// 
// Anton Betten
// July 28, 2011
//
// 
// Performs some task for each input set.
// The input sets are defined using the data_input_stream class.
// An output file is generated.

#include "orbiter.h"

using namespace std;



using namespace orbiter;




// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;

	int f_job = FALSE;
	projective_space_job_description *Job;
	

	t0 = os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-job") == 0) {
			f_job = TRUE;

			Job = NEW_OBJECT(projective_space_job_description);

			i += Job->read_arguments(argc - i,
				argv + i + 1, verbose_level);
			cout << "-job " << endl;
		}
	}
	cout << "process.cpp finished reading arguments" << endl;

	if (!f_job) {
		cout << "please use option -job ... -end" << endl;
		exit(1);
	}
	if (!Job->f_q) {
		cout << "please use option -q <q> within the job description" << endl;
		exit(1);
	}
	if (!Job->f_n) {
		cout << "please use option -n <n> to specify the projective dimension  within the job description" << endl;
		exit(1);
	}
	if (!Job->f_fname_base_out) {
		cout << "please use option -fname_base_out <fname_base_out> within the job description" << endl;
		exit(1);
	}


	
	Job->perform_job(verbose_level);




	the_end(t0);
}



