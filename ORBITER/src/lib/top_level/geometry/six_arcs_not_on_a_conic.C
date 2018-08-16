// six_arcs_not_on_a_conic.C
// 
// Anton Betten
//
// March 6, 2018
//


#include "orbiter.h"

six_arcs_not_on_a_conic::six_arcs_not_on_a_conic()
{
	null();
}

six_arcs_not_on_a_conic::~six_arcs_not_on_a_conic()
{
	freeself();
}

void six_arcs_not_on_a_conic::null()
{
	F = NULL;
	P2 = NULL;
	Gen = NULL;
	Not_on_conic_idx = NULL;
}

void six_arcs_not_on_a_conic::freeself()
{
	if (Gen) {
		delete Gen;
		}
	if (Not_on_conic_idx) {
		FREE_INT(Not_on_conic_idx);
		}
	null();
}

void six_arcs_not_on_a_conic::init(finite_field *F, projective_space *P2, 
	int argc, const char **argv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT level = 6;

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init" << endl;
		}

	six_arcs_not_on_a_conic::F = F;
	six_arcs_not_on_a_conic::P2 = P2;
	
	sprintf(base_fname, "arcs_%ld", F->q);

	
	Gen = new arc_generator;

	Gen->f_poly = FALSE;

	Gen->d = 2; // we will classify two-arcs


	Gen->ECA = new exact_cover_arguments;
	Gen->IA = new isomorph_arguments;

	Gen->ECA->f_has_solution_prefix = TRUE;
	Gen->ECA->solution_prefix = "";

	Gen->ECA->f_has_base_fname = TRUE;
	Gen->ECA->base_fname = base_fname;
	
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init before Gen->init" << endl;
		}
	Gen->init(F, 
		"" /* Gen->ECA->input_prefix */, 
		"" /* Gen->ECA->base_fname */,
		6 /* Gen->ECA->starter_size */, 
		argc, argv, 
		verbose_level - 2);
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init after Gen->init" << endl;
		}



	//cout << "before Gen->main" << endl;
	//Gen->main(Gen->verbose_level);

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"Classifying 6-arcs for q=" << F->q << endl;
		}
	
	Gen->compute_starter(verbose_level - 1);


	nb_orbits = Gen->gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"We found " << nb_orbits << " isomorphism types "
						"of 6-arcs" << endl;
		}


	
	INT Arc6[6];
	INT h, j;
	

	nb_arcs_not_on_conic = 0;

	Not_on_conic_idx = NEW_INT(nb_orbits);	
	
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"testing the arcs" << endl;
		}

	for (h = 0; h < nb_orbits; h++) {

		if (f_v) {
			cout << "six_arcs_not_on_a_conic::init "
					"testing arc " << h << " / " << nb_orbits << endl;
			}

		
		Gen->gen->get_set_by_level(level, h, Arc6);
		
		
		INT **Pts_on_conic;
		INT *nb_pts_on_conic;
		INT len1;


		
		if (f_v) {
			cout << "six_arcs_not_on_a_conic::init "
					"computing conic intersections:" << endl;
			}
		P2->conic_type(
			Arc6, 6, 
			Pts_on_conic, nb_pts_on_conic, len1, 
			0 /*verbose_level*/);
		if (f_v) {
			cout << "The arc intersects " << len1
					<< " conics in 6 or more points. " << endl;
			}

		if (len1 == 0) {
			Not_on_conic_idx[nb_arcs_not_on_conic++] = h;
			}

		for (j = 0; j < len1; j++) {
			FREE_INT(Pts_on_conic[j]);
			}
		FREE_PINT(Pts_on_conic);
		FREE_INT(nb_pts_on_conic);
		}

	if (f_v) {
		cout << "We found " << nb_arcs_not_on_conic << " isomorphism types "
				"of 6-arcs not on a conic, out of a total of "
				<< nb_orbits << " isomorphism types of arcs" << endl;
		}
	



	if (f_v) {
		cout << "six_arcs_not_on_a_conic::done" << endl;
		}

}


void six_arcs_not_on_a_conic::report_latex(ostream &ost)
{
	INT h;
	
	ost << "\\subsection*{Classification of 6-arcs not on a conic "
			"in $\\PG(2," << F->q << ")$}" << endl;
	
	longinteger_object go;
	longinteger_domain D;
	{
	Gen->A->Strong_gens->group_order(go);

	ost << "The order of the group $" << Gen->A->label_tex << "$ is ";
	go.print_not_scientific(ost);
	ost << endl;

	ost << "\\bigskip" << endl << endl;
	}

	longinteger_object ol, Ol;
	Ol.create(0);
	for (h = 0; h < nb_arcs_not_on_conic; h++) {
		set_and_stabilizer *R;

		R = Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Not_on_conic_idx[h] /* orbit_at_level */,
				0 /* verbose_level */);
		Gen->gen->orbit_length(
				Not_on_conic_idx[h] /* node */,
				6 /* level */, ol);
		D.add_in_place(Ol, ol);
		
		
		ost << "$" << h << " / " << nb_arcs_not_on_conic
				<< "$ Arc $" << Not_on_conic_idx[h] << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		delete R;
	}
	ost << "The overall number of 6-arcs not on a conic "
			"in $\\PG(2," << F->q << ")$ is: " << Ol << "\\\\" << endl;
}



