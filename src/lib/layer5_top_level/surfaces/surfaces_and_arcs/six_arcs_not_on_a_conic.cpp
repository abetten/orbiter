// six_arcs_not_on_a_conic.cpp
// 
// Anton Betten
//
// March 6, 2018
//


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


six_arcs_not_on_a_conic::six_arcs_not_on_a_conic()
{
	Record_birth();
	Descr = NULL;
	PA = NULL;
	Gen = NULL;
	nb_orbits = 0;
	Not_on_conic_idx = NULL;
	nb_arcs_not_on_conic = 0;
}


six_arcs_not_on_a_conic::~six_arcs_not_on_a_conic()
{
	Record_death();
	if (Gen) {
		FREE_OBJECT(Gen);
	}
	if (Not_on_conic_idx) {
		FREE_int(Not_on_conic_idx);
	}
}

void six_arcs_not_on_a_conic::init(
		apps_geometry::arc_generator_description *Descr,
		projective_geometry::projective_space_with_action *PA,
	int f_test_nb_Eckardt_points, int nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int level = 6;

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init" << endl;
	}

	if (PA->n != 2) {
		cout << "six_arcs_not_on_a_conic::init PA->n != 2" << endl;
		exit(1);
	}

	six_arcs_not_on_a_conic::Descr = Descr;
	six_arcs_not_on_a_conic::PA = PA;
	

	

	Descr->f_target_size = true;
	Descr->target_size = 6;
	Descr->f_d = true;
	Descr->d = 2;
	Descr->f_conic_test = true;
	Descr->f_test_nb_Eckardt_points = f_test_nb_Eckardt_points;
	Descr->nb_E = nb_E;




	
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"before Gen->init" << endl;
	}

	Gen = NEW_OBJECT(apps_geometry::arc_generator);


	Gen->init(
			Descr,
			PA,
			PA->A->Strong_gens,
			verbose_level - 2);



	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"after Gen->init" << endl;
	}



	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"Classifying 6-arcs for q=" << PA->F->q << endl;
		cout << "six_arcs_not_on_a_conic::init before Gen->compute_starter" << endl;
	}
	
	Gen->compute_starter(verbose_level - 1);

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::init "
				"Classifying 6-arcs for q=" << PA->F->q << endl;
		cout << "six_arcs_not_on_a_conic::init after Gen->compute_starter" << endl;
	}


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

		if (f_v && (h % 10000) == 0) {
			cout << "six_arcs_not_on_a_conic::init "
					"testing arc " << h << " / " << nb_orbits << endl;
		}

		
		Gen->gen->get_set_by_level(level, h, Arc6);
		
		
		long int **Pts_on_conic;
		int **Conic_eqn;
		int *nb_pts_on_conic;
		int len1;

		if (f_v && (h % 10000) == 0) {
			cout << "six_arcs_not_on_a_conic::init "
					"testing arc " << h << " / " << nb_orbits << " : ";
			Lint_vec_print(cout, Arc6, 6);
			cout << endl;
		}


		
		if (f_v && (h % 10000) == 0) {
			cout << "six_arcs_not_on_a_conic::init "
					"computing conic intersections:" << endl;
		}
		PA->P->Plane->conic_type(
			Arc6, 6, 
			6 /* threshold */,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len1,
			0 /*verbose_level - 2*/);
		if (f_v && (h % 10000) == 0) {
			cout << "The arc intersects " << len1
					<< " conics in 6 or more points. " << endl;
		}

		if (len1 == 0) {
			Not_on_conic_idx[nb_arcs_not_on_conic++] = h;
		}

		for (j = 0; j < len1; j++) {
			FREE_lint(Pts_on_conic[j]);
			FREE_int(Conic_eqn[j]);
		}
		FREE_plint(Pts_on_conic);
		FREE_pint(Conic_eqn);
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


void six_arcs_not_on_a_conic::recognize(
		long int *arc6, int *transporter,
		int &orbit_not_on_conic_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_at_level;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "six_arcs_not_on_a_conic::recognize" << endl;
	}

	Gen->gen->get_Orbit_tracer()->identify(
			arc6, 6,
		transporter,
		orbit_at_level,
		0 /*verbose_level */);


	if (!Sorting.int_vec_search(
			Not_on_conic_idx,
		nb_arcs_not_on_conic, orbit_at_level,
		orbit_not_on_conic_idx)) {
		cout << "six_arcs_not_on_a_conic::recognize could not find orbit" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "six_arcs_not_on_a_conic::recognize done" << endl;
	}
}


void six_arcs_not_on_a_conic::report_latex(
		std::ostream &ost)
{
	int h;
	
	ost << "\\subsection*{Classification of 6-arcs not on a conic "
			"in $\\PG(2," << PA->F->q << ")$}" << endl;
	
	algebra::ring_theory::longinteger_object go;
	algebra::ring_theory::longinteger_domain D;
	{
		PA->A->Strong_gens->group_order(go);

		ost << "The order of the group $" << PA->A->label_tex << "$ is ";
		go.print_not_scientific(ost);
		ost << endl;

		ost << "\\bigskip" << endl << endl;
	}

	algebra::ring_theory::longinteger_object ol, Ol;
	Ol.create(0);
	for (h = 0; h < nb_arcs_not_on_conic; h++) {
		data_structures_groups::set_and_stabilizer *R;

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
			"in $\\PG(2," << PA->F->q << ")$ is: " << Ol << "\\\\" << endl;
}

void six_arcs_not_on_a_conic::report_specific_arc_basic(
		std::ostream &ost, int arc_idx)
{
	data_structures_groups::set_and_stabilizer *The_arc;
	algebra::ring_theory::longinteger_object go;

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

void six_arcs_not_on_a_conic::report_specific_arc(
		std::ostream &ost, int arc_idx)
{
	data_structures_groups::set_and_stabilizer *The_arc;
	algebra::ring_theory::longinteger_object go;

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

	PA->F->Io->display_table_of_projective_points(ost, The_arc->data, 6, 3);


	//ost << "The arc-stabilizer is the following group:\\\\" << endl;
	//The_arc->Strong_gens->print_generators_tex(ost);

	FREE_OBJECT(The_arc);


}


}}}}




