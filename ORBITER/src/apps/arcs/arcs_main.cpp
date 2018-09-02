// arcs_main.C
// 
// Anton Betten
//
// previous version Dec 6, 2004
// revised June 19, 2006
// revised Aug 17, 2008
//
// Searches for arcs in desarguesian projective planes
//
//

#include "orbiter.h"

// global data:

INT t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	
	{
	arc_generator *Gen;
	finite_field *F;

	
	Gen = NEW_OBJECT(arc_generator);

	cout << argv[0] << endl;
	cout << "before Gen->read_arguments" << endl;
	Gen->read_arguments(argc, argv);
	

	cout << "before creating the finite field" << endl;
	F = NEW_OBJECT(finite_field);

	if (Gen->f_poly) {
		F->init_override_polynomial(Gen->q,
				Gen->poly, 0 /*verbose_level*/);
		}
	else {
		F->init(Gen->q, 0);
		}


	cout << "before Gen->init" << endl;
	Gen->init(F, 
		Gen->ECA->input_prefix, 
		Gen->ECA->base_fname,
		Gen->ECA->starter_size, 
		argc, argv, 
		Gen->verbose_level);
	


	cout << "before Gen->main" << endl;
	Gen->main(Gen->verbose_level);

		if (Gen->f_starter) {
				cout << "preparing level spreadsheet" << endl;
				{
				spreadsheet *Sp;
				Gen->gen->make_spreadsheet_of_level_info(
						Sp, Gen->ECA->starter_size);
				char fname_csv[1000];
				sprintf(fname_csv, "arcs_%ld_%ld_level.csv",
						Gen->q, Gen->ECA->starter_size);
				Sp->save(fname_csv, Gen->verbose_level);
				delete Sp;
				}
				cout << "preparing orbit spreadsheet" << endl;
				{
				spreadsheet *Sp;
				Gen->gen->make_spreadsheet_of_orbit_reps(
						Sp, Gen->ECA->starter_size);
				char fname_csv[1000];
				sprintf(fname_csv, "arcs_%ld_%ld.csv",
						Gen->q, Gen->ECA->starter_size);
				Sp->save(fname_csv, Gen->verbose_level);
				delete Sp;
				}
				cout << "preparing orbit spreadsheet done" << endl;
			}
	
	FREE_OBJECT(Gen);
	FREE_OBJECT(F);
	
	}
	cout << "Memory usage = " << os_memory_usage()
			<<  " Time = " << delta_time(t0)
			<< " tps = " << os_ticks_per_second() << endl;
	the_end(t0);
	//the_end_quietly(t0);
}



