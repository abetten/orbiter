// six_arcs_not_on_a_conic.cpp
// 
// Anton Betten
//
// March 6, 2018
//


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

six_arcs_not_on_a_conic::six_arcs_not_on_a_conic()
{
	P2 = NULL;
	Descr = NULL;
	Gen = NULL;
	nb_orbits = 0;
	Not_on_conic_idx = NULL;
	nb_arcs_not_on_conic = 0;
	//null();
}


six_arcs_not_on_a_conic::~six_arcs_not_on_a_conic()
{
	freeself();
}

void six_arcs_not_on_a_conic::null()
{
}

void six_arcs_not_on_a_conic::freeself()
{
	if (Gen) {
		FREE_OBJECT(Gen);
	}
	if (Not_on_conic_idx) {
		FREE_int(Not_on_conic_idx);
	}
	null();
}

void six_arcs_not_on_a_conic::init(
	arc_generator_description *Descr,
	action *A,
	projective_space *P2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int level = 6;

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init" << endl;
	}

	six_arcs_not_on_a_conic::P2 = P2;
	six_arcs_not_on_a_conic::Descr = Descr;
	

	

	Descr->f_target_size = TRUE;
	Descr->target_size = 6;
	Descr->f_d = TRUE;
	Descr->d = 2;
	Descr->f_n = TRUE;
	Descr->n = 3;
	Descr->f_conic_test = TRUE;





	
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"before Gen->init" << endl;
	}

	Gen = NEW_OBJECT(arc_generator);


	Gen->init(
			Descr,
			A, A->Strong_gens,
			//GTA,
			//F,
			//A, A->Strong_gens,
			//6 /* Gen->ECA->starter_size */,
			//TRUE /* f_conic_test */,
			//Control,
			verbose_level - 2);



	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"after Gen->init" << endl;
	}



	//cout << "before Gen->main" << endl;
	//Gen->main(Gen->verbose_level);

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"Classifying 6-arcs for q=" << Descr->F->q << endl;
		cout << "six_arcs_not_on_a_conic::init before Gen->compute_starter" << endl;
	}
	
	Gen->compute_starter(verbose_level - 1);


	nb_orbits = Gen->gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"We found " << nb_orbits << " isomorphism types "
				"of 6-arcs" << endl;
	}


	
	long int Arc6[6];
	int h, j;
	

	nb_arcs_not_on_conic = 0;

	Not_on_conic_idx = NEW_int(nb_orbits);	
	
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
		
		
		long int **Pts_on_conic;
		int *nb_pts_on_conic;
		int len1;


		
		if (f_v) {
			cout << "six_arcs_not_on_a_conic::init "
					"computing conic intersections:" << endl;
		}
		P2->conic_type(
			Arc6, 6, 
			Pts_on_conic, nb_pts_on_conic, len1, 
			verbose_level - 2);
		if (f_v) {
			cout << "The arc intersects " << len1
					<< " conics in 6 or more points. " << endl;
		}

		if (len1 == 0) {
			Not_on_conic_idx[nb_arcs_not_on_conic++] = h;
		}

		for (j = 0; j < len1; j++) {
			FREE_lint(Pts_on_conic[j]);
		}
		FREE_plint(Pts_on_conic);
		FREE_int(nb_pts_on_conic);
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


void six_arcs_not_on_a_conic::recognize(long int *arc6, int *transporter,
		int &orbit_not_on_conic_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_at_level;
	sorting Sorting;

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::recognize" << endl;
	}

	Gen->gen->identify(arc6, 6,
		transporter,
		orbit_at_level,
		0 /*verbose_level */);


	if (!Sorting.int_vec_search(Not_on_conic_idx,
		nb_arcs_not_on_conic, orbit_at_level,
		orbit_not_on_conic_idx)) {
		cout << "six_arcs_not_on_a_conic::recognize could not find orbit" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::recognize done" << endl;
	}
}


void six_arcs_not_on_a_conic::report_latex(ostream &ost)
{
	int h;
	
	ost << "\\subsection*{Classification of 6-arcs not on a conic "
			"in $\\PG(2," << Descr->F->q << ")$}" << endl;
	
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
	Ol.create(0, __FILE__, __LINE__);
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

		FREE_OBJECT(R);
	}
	ost << "The overall number of 6-arcs not on a conic "
			"in $\\PG(2," << Descr->F->q << ")$ is: " << Ol << "\\\\" << endl;
}

void six_arcs_not_on_a_conic::report_specific_arc_basic(ostream &ost, int arc_idx)
{
	set_and_stabilizer *The_arc;
	longinteger_object go;

	The_arc = Gen->gen->get_set_and_stabilizer(
			6 /* level */,
			Not_on_conic_idx[arc_idx],
			0 /* verbose_level */);

	The_arc->Strong_gens->group_order(go);

	ost << "Arc " << arc_idx << " / " << nb_arcs_not_on_conic << " is: ";
	ost << "$$" << endl;
	//int_vec_print(fp, Arc6, 6);
	//ost << "\{";
	The_arc->print_set_tex(ost);
	//ost << "\}_{" << go << "}";
	ost << "$$" << endl;

	//P2->F->display_table_of_projective_points(ost, The_arc->data, 6, 3);


	//ost << "The arc-stabilizer is the following group:\\\\" << endl;
	//The_arc->Strong_gens->print_generators_tex(ost);

	FREE_OBJECT(The_arc);


}

void six_arcs_not_on_a_conic::report_specific_arc(ostream &ost, int arc_idx)
{
	set_and_stabilizer *The_arc;
	longinteger_object go;

	The_arc = Gen->gen->get_set_and_stabilizer(
			6 /* level */,
			Not_on_conic_idx[arc_idx],
			0 /* verbose_level */);

	The_arc->Strong_gens->group_order(go);

	ost << "Arc " << arc_idx << " / " << nb_arcs_not_on_conic << " is: ";
	ost << "$$" << endl;
	//int_vec_print(fp, Arc6, 6);
	//ost << "\{";
	The_arc->print_set_tex(ost);
	//ost << "\}_{" << go << "}";
	ost << "$$" << endl;

	P2->F->display_table_of_projective_points(ost, The_arc->data, 6, 3);


	//ost << "The arc-stabilizer is the following group:\\\\" << endl;
	//The_arc->Strong_gens->print_generators_tex(ost);

	FREE_OBJECT(The_arc);


}


}}



