// surface_classify_wedge.cpp
// 
// Anton Betten
// September 2, 2016
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {


surface_classify_wedge::surface_classify_wedge()
{
	Record_birth();
	PA = NULL;
	F = NULL;
	q = 0;

	//std::string fname_base;

	A = NULL;
	A2 = NULL;


	Surf = NULL;
	Surf_A = NULL;
	
	Elt0 = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;

	Five_p1 = NULL;

	Classify_double_sixes = NULL;

	Flag_orbits = NULL;
	Surfaces = NULL;
	
	Surface_repository = NULL;

}

surface_classify_wedge::~surface_classify_wedge()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge" << endl;
	}
	if (Elt0) {
		FREE_int(Elt0);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}

	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"before FREE_OBJECTS(Five_p1)" << endl;
	}
	if (Five_p1) {
		FREE_OBJECT(Five_p1);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"after FREE_OBJECTS(Five_p1)" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"before FREE_OBJECTS(Classify_double_sixes)" << endl;
	}
	if (Classify_double_sixes) {
		FREE_OBJECT(Classify_double_sixes);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"after FREE_OBJECTS(Classify_double_sixes)" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"before FREE_OBJECTS(Flag_orbits)" << endl;
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"after FREE_OBJECTS(Flag_orbits)" << endl;
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"before FREE_OBJECTS(Surfaces)" << endl;
	}
	if (Surfaces) {
		FREE_OBJECT(Surfaces);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge "
				"after FREE_OBJECTS(Surfaces)" << endl;
	}


	if (Surface_repository) {
		FREE_OBJECT(Surface_repository);
	}

	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge done" << endl;
	}
}

void surface_classify_wedge::init(
		projective_geometry::projective_space_with_action *PA,
	poset_classification::poset_classification_control
		*Control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	
	if (f_v) {
		cout << "surface_classify_wedge::init" << endl;
	}
	surface_classify_wedge::PA = PA;
	Surf_A = PA->Surf_A;
	Surf = Surf_A->Surf;
	F = PA->F;
	q = F->q;

	fname_base = "surface_q" + std::to_string(q);

	
	
	A = PA->A;
	A2 = PA->A_on_lines;


	
	Elt0 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	Five_p1 = NEW_OBJECT(classify_five_plus_one);

	if (f_v) {
		cout << "surface_classify_wedge::init "
				"before Five_p1->init" << endl;
	}
	Five_p1->init(PA, Control, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::init "
				"after Five_p1->init" << endl;
	}


	Classify_double_sixes = NEW_OBJECT(classify_double_sixes);

	if (f_v) {
		cout << "surface_classify_wedge::init "
				"before Classify_double_sixes->init" << endl;
	}
	Classify_double_sixes->init(Five_p1, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::init "
				"after Classify_double_sixes->init" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::init done" << endl;
	}
}

void surface_classify_wedge::do_classify_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::do_classify_double_sixes" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (test_if_double_sixes_have_been_computed_already()) {
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"before read_double_sixes" << endl;
		}
		read_double_sixes(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"after read_double_sixes" << endl;
		}
	}

	else {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"before Five_p1->classify_partial_ovoids" << endl;
		}
		Five_p1->classify_partial_ovoids(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"after Five_p1->classify_partial_ovoids" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"before Classify_double_sixes->classify" << endl;
		}
		Classify_double_sixes->classify(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"after Classify_double_sixes->classify" << endl;
		}



		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"before write_double_sixes" << endl;
		}
		write_double_sixes(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"after write_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"writing cheat sheet on double sixes" << endl;
		}
		create_report_double_sixes(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes "
					"writing cheat sheet on double sixes done" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::do_classify_double_sixes done" << endl;
	}
}

void surface_classify_wedge::do_classify_surfaces(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::do_classify_surfaces" << endl;
	}
	if (test_if_surfaces_have_been_computed_already()) {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces "
					"before read_surfaces" << endl;
		}
		read_surfaces(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces "
					"after read_surfaces" << endl;
		}

	}
	else {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces "
					"classifying surfaces" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces before "
					"classify_surfaces_from_double_sixes" << endl;
		}
		classify_surfaces_from_double_sixes(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces after "
					"classify_surfaces_from_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces "
					"before write_surfaces" << endl;
		}
		write_surfaces(verbose_level - 4);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces "
					"after write_surfaces" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::do_classify_surfaces done" << endl;
	}
}

void surface_classify_wedge::classify_surfaces_from_double_sixes(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"before downstep" << endl;
	}
	downstep(verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits
				<< " flag orbits out of "
				<< Classify_double_sixes->Double_sixes->nb_orbits
				<< " orbits of double sixes" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"before upstep" << endl;
	}
	upstep(verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"after upstep" << endl;
		cout << "we found " << Surfaces->nb_orbits
				<< " surfaces out from "
				<< Flag_orbits->nb_flag_orbits
				<< " double sixes" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"before post_process" << endl;
	}
	post_process(verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes "
				"after post_process" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes done" << endl;
	}
}

void surface_classify_wedge::post_process(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::post_process" << endl;
	}

	Surface_repository = NEW_OBJECT(surface_repository);

	if (f_v) {
		cout << "surface_classify_wedge::post_process "
				"before Surface_repository->init" << endl;
	}
	Surface_repository->init(this, verbose_level - 1);
	if (f_v) {
		cout << "surface_classify_wedge::post_process "
				"after Surface_repository->init" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::post_process done" << endl;
	}
}

void surface_classify_wedge::downstep(
		int verbose_level)
// from double sixes to cubic surfaces
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, nb_orbits, nb_flag_orbits;

	if (f_v) {
		cout << "surface_classify_wedge::downstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	nb_orbits = Classify_double_sixes->Double_sixes->nb_orbits;
	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(
			A,
			A2,
			nb_orbits /* nb_primary_orbits_lower */,
			27 /* pt_representation_sz */,
			nb_orbits /* nb_flag_orbits */,
			1 /* upper_bound_for_number_of_traces */, // ToDo
			NULL /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
			NULL /* void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)*/,
			NULL /* void *free_received_trace_data */,
			verbose_level - 3);

	if (f_v) {
		cout << "surface_classify_wedge::downstep "
				"initializing flag orbits" << endl;
	}

	nb_flag_orbits = 0;
	for (i = 0; i < nb_orbits; i++) {

		if (f_v) {
			cout << "surface_classify_wedge::downstep "
					"orbit " << i << " / " << nb_orbits << endl;
		}
		data_structures_groups::set_and_stabilizer *R;
		algebra::ring_theory::longinteger_object go;
		long int Lines[27];

		R = Classify_double_sixes->Double_sixes->get_set_and_stabilizer(
				i /* orbit_index */,
				0 /* verbose_level */);


		R->Strong_gens->group_order(go);

		Lint_vec_copy(R->data, Lines, 12);

		if (f_vv) {
			cout << "surface_classify_wedge::downstep "
					"before create_the_fifteen_other_lines" << endl;
		}

		Surf->create_the_fifteen_other_lines(
				Lines /* double_six */,
				Lines + 12 /* fifteen_other_lines */,
				0 /*verbose_level - 4*/);
		if (f_vv) {
			cout << "surface_classify_wedge::downstep "
					"after create_the_fifteen_other_lines" << endl;
		}


		if (f_vv) {
			cout << "surface_classify_wedge::downstep "
					"before Flag_orbit_node[].init" << endl;
		}

		Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(
			Flag_orbits,
			nb_flag_orbits /* flag_orbit_index */,
			i /* downstep_primary_orbit */,
			0 /* downstep_secondary_orbit */,
			1 /* downstep_orbit_len */,
			false /* f_long_orbit */,
			Lines /* int *pt_representation */,
			R->Strong_gens,
			0/*verbose_level - 2*/);

		if (f_vv) {
			cout << "surface_classify_wedge::downstep "
					"after Flag_orbit_node[].init" << endl;
		}

		R->Strong_gens = NULL;

		nb_flag_orbits++;


		FREE_OBJECT(R);
	}

	Flag_orbits->nb_flag_orbits = nb_flag_orbits;


	if (f_v) {
		cout << "surface_classify_wedge::downstep "
				"initializing flag orbits done" << endl;
	}
}

void surface_classify_wedge::upstep(
		int verbose_level)
// from double sixes to cubic surfaces
{
	int f_v = (verbose_level >= 1);
	int f, po, so, i, j;
	int *f_processed;
	int nb_processed;

	if (f_v) {
		cout << "surface_classify_wedge::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	Int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Surfaces = NEW_OBJECT(invariant_relations::classification_step);

	algebra::ring_theory::longinteger_object go;
	A->group_order(go);

	if (f_v) {
		cout << "surface_classify_wedge::upstep "
				"before Surfaces->init" << endl;
	}
	Surfaces->init(
			A, A2,
			Flag_orbits->nb_flag_orbits, 27, go,
			verbose_level - 3);
	if (f_v) {
		cout << "surface_classify_wedge::upstep "
				"after Surfaces->init" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::upstep "
				"number of flag orbits to be processed is "
				<< Flag_orbits->nb_flag_orbits << endl;
	}

	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		long int Lines[27];
		
		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) /
					(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "Defining another orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " from flag orbit " << f << " / "
					<< Flag_orbits->nb_flag_orbits
					<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;
		

		if (Flag_orbits->pt_representation_sz != 27) {
			cout << "Flag_orbits->pt_representation_sz != 27" << endl;
			exit(1);
		}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		Lint_vec_copy(Flag_orbits->Pt + f * 27, Lines, 27);




		data_structures_groups::vector_ge *coset_reps;
		int nb_coset_reps;
		
		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);

		coset_reps->init(
				Surf_A->A, 0/*verbose_level - 2*/);
		coset_reps->allocate(
				36, 0/*verbose_level - 2*/);


		groups::strong_generators *S;
		algebra::ring_theory::longinteger_object go;


		if (f_v) {
			cout << "Lines:";
			Lint_vec_print(cout, Lines, 27);
			cout << endl;
		}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy(verbose_level - 2);
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so << " go=" << go << endl;
		}

		nb_coset_reps = 0;
		for (i = 0; i < 36; i++) {
			
			if (f_v) {
				cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
						<< ", upstep i=" << i << " / 36" << endl;
			}
			int f2;

			long int double_six[12];


			for (j = 0; j < 12; j++) {
				double_six[j] = Lines[Surf->Schlaefli->Schlaefli_double_six->Double_six[i * 12 + j]];
			}
			if (f_v) {
				cout << "f=" << f << " / "
						<< Flag_orbits->nb_flag_orbits
						<< ", upstep i=" << i
						<< " / 36 double_six=";
				Lint_vec_print(cout, double_six, 12);
				cout << endl;
			}
			
			Classify_double_sixes->identify_double_six(
					double_six,
				Elt1 /* transporter */, f2,
				0/*verbose_level - 10*/);

			if (f_v) {
				cout << "f=" << f << " / "
						<< Flag_orbits->nb_flag_orbits
						<< ", upstep " << i
						<< " / 36, double six is "
								"isomorphic to orbit " << f2 << endl;
			}

			
			if (f2 == f) {
				if (f_v) {
					cout << "We found an automorphism of the surface:" << endl;
					//A->element_print_quick(Elt1, cout);
					//cout << endl;
				}
				A->Group_element->element_move(
						Elt1, coset_reps->ith(nb_coset_reps), 0);
				nb_coset_reps++;
			}
			else {
				if (f_v) {
					cout << "We are identifying flag orbit " << f2
							<< " with flag orbit " << f << endl;
				}
				if (!f_processed[f2]) {
					Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
						= Flag_orbits->nb_primary_orbits_upper;
					Flag_orbits->Flag_orbit_node[f2].f_fusion_node = true;
					Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
					Flag_orbits->Flag_orbit_node[f2].fusion_elt
						= NEW_int(A->elt_size_in_int);
					A->Group_element->element_invert(Elt1,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt, 0);
					f_processed[f2] = true;
					nb_processed++;
				}
				else {
					cout << "Flag orbit " << f2 << " has already been "
							"identified with flag orbit " << f << endl;
					if (Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
						cout << "Flag_orbits->Flag_orbit_node[f2]."
								"fusion_with != f" << endl;
						exit(1);
					}
				}
			}

		} // next i


		coset_reps->reallocate(
				nb_coset_reps, 0/*verbose_level - 2*/);

		groups::strong_generators *Aut_gens;

		{
			algebra::ring_theory::longinteger_object ago;

			if (f_v) {
				cout << "surface_classify_wedge::upstep "
						"Extending the group by a factor of "
						<< nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(groups::strong_generators);
			Aut_gens->init_group_extension(
					S, coset_reps,
					nb_coset_reps,
					0/*verbose_level - 2*/);

			Aut_gens->group_order(ago);


			if (f_v) {
				cout << "the double six has a stabilizer of order "
						<< ago << endl;
				//cout << "The double six stabilizer is:" << endl;
				//Aut_gens->print_generators_tex(cout);
			}
		}


		if (f_v) {
			cout << "surface_classify_wedge::upstep cubic surface orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " will be created " << endl;
		}


		Surfaces->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Surfaces,
			Flag_orbits->nb_primary_orbits_upper, 
			f,
			Aut_gens, Lines, NULL /* extra_data */,
			verbose_level - 4);

		if (f_v) {
			cout << "surface_classify_wedge::upstep cubic surface orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " has been created" << endl;
		}

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);
		
		f_processed[f] = true;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
	} // next f


	if (nb_processed != Flag_orbits->nb_flag_orbits) {
		cout << "nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = "
				<< Flag_orbits->nb_flag_orbits << endl;
		exit(1);
	}

	Surfaces->nb_orbits = Flag_orbits->nb_primary_orbits_upper;
	
	if (f_v) {
		cout << "We found " << Surfaces->nb_orbits
				<< " orbits of surfaces from "
				<< Flag_orbits->nb_flag_orbits
				<< " double sixes" << endl;
	}
	
	FREE_int(f_processed);


	if (f_v) {
		cout << "surface_classify_wedge::upstep done" << endl;
	}
}


void surface_classify_wedge::derived_arcs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int iso_type;
	int *Starter_configuration_idx;
	int nb_starter_conf;
	int c, orb, i;
	long int S[5];
	long int S2[7];
	long int K1[7];
	int w[6];
	int v[6];
	long int Arc[6];
	long int four_lines[4];
	long int trans12[2];
	int perp_sz;
	long int b5;

	if (f_v) {
		cout << "surface_classify_wedge::derived_arcs" << endl;
	}
	for (iso_type = 0; iso_type < Surfaces->nb_orbits; iso_type++) {
		if (f_v) {
			cout << "surface_classify_wedge::derived_arcs "
					"surface " << iso_type << " / "
					<< Surfaces->nb_orbits << ":" << endl;
		}

		starter_configurations_which_are_involved(
				iso_type,
			Starter_configuration_idx, nb_starter_conf,
			verbose_level);

		if (f_v) {
			cout << "surface_classify_wedge::derived_arcs "
					"There are " << nb_starter_conf
					<< " starter configurations which are involved: " << endl;
			Int_vec_print(cout,
					Starter_configuration_idx, nb_starter_conf);
			cout << endl;
		}

		for (c = 0; c < nb_starter_conf; c++) {
			orb = Starter_configuration_idx[c];
			//s = Starter_configuration_idx[c];
			//orb = Classify_double_sixes->Idx[s];

			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"configuration " << c << " / " << nb_starter_conf
						<< " is orbit " << orb << endl;
			}

			Five_p1->Five_plus_one->get_set_by_level(5, orb, S);

			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"starter configuration as neighbors: ";
				Lint_vec_print(cout, S, 5);
				cout << endl;
			}

			Lint_vec_apply(
					S,
					Five_p1->Linear_complex->Neighbor_to_line,
					S2, 5);

			S2[5] = Five_p1->Linear_complex->pt0_line;

			four_lines[0] = S2[0];
			four_lines[1] = S2[1];
			four_lines[2] = S2[2];
			four_lines[3] = S2[3];
			Surf->perp_of_four_lines(
					four_lines,
					trans12, perp_sz,
					0 /* verbose_level */);

			if (trans12[0] == Five_p1->Linear_complex->pt0_line) {
				b5 = trans12[1];
			}
			else if (trans12[1] == Five_p1->Linear_complex->pt0_line) {
				b5 = trans12[0];
			}
			else {
				cout << "surface_classify_wedge::derived_arcs "
						"something is wrong with the starter configuration" << endl;
				exit(1);
			}



			long int *lines;
			int nb_lines;
			long int lines_meet3[3];
			long int lines_skew3[3];

			lines_meet3[0] = S2[1]; // a_2
			lines_meet3[1] = S2[2]; // a_3
			lines_meet3[2] = S2[3]; // a_4
			lines_skew3[0] = S2[0]; // a_1
			lines_skew3[1] = b5;
			lines_skew3[2] = S2[5]; // b_6

			Surf->lines_meet3_and_skew3(lines_meet3,
					lines_skew3, lines, nb_lines,
					0 /* verbose_level */);
			//Surf->perp_of_three_lines(three_lines, perp, perp_sz, 0 /* verbose_level */);

			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"The lines which meet { a_2, a_3, a_4 } "
						"and are skew to { a_1, b_5, b_6 } are: ";
				Lint_vec_print(cout, lines, nb_lines);
				cout << endl;
				cout << "surface_classify_wedge::derived_arcs "
						"generator matrices:" << endl;
				Surf->Gr->print_set(lines, nb_lines);
			}

			FREE_lint(lines);

			lines_meet3[0] = S2[0]; // a_1
			lines_meet3[1] = S2[2]; // a_3
			lines_meet3[2] = S2[3]; // a_4
			lines_skew3[0] = S2[1]; // a_2
			lines_skew3[1] = b5;
			lines_skew3[2] = S2[5]; // b6

			Surf->lines_meet3_and_skew3(lines_meet3,
					lines_skew3, lines, nb_lines,
					0 /* verbose_level */);
			//Surf->perp_of_three_lines(three_lines, perp, perp_sz, 0 /* verbose_level */);

			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"The lines which meet { a_1, a_3, a_4 } "
						"and are skew to { a_2, b_5, b_6 } are: ";
				Lint_vec_print(cout, lines, nb_lines);
				cout << endl;
				cout << "surface_classify_wedge::derived_arcs "
						"generator matrices:" << endl;
				Surf->Gr->print_set(lines, nb_lines);
			}

			FREE_lint(lines);


			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"starter configuration as line ranks: ";
				Lint_vec_print(cout, S2, 6);
				cout << endl;
				cout << "b5=" << b5 << endl;
				cout << "surface_classify_wedge::derived_arcs "
						"generator matrices:" << endl;
				Surf->Gr->print_set(S2, 6);
				cout << "b5:" << endl;
				Surf->Gr->print_set(&b5, 1);
			}
			S2[6] = b5;

			for (int h = 0; h < 7; h++) {
				K1[h] = Surf->Klein->line_to_point_on_quadric(
						S2[h], 0/*verbose_level*/);
			}
			//int_vec_apply(S2, Surf->Klein->Line_to_point_on_quadric, K1, 7);
			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"starter configuration on the klein quadric: ";
				Lint_vec_print(cout, K1, 7);
				cout << endl;
				for (i = 0; i < 7; i++) {
					Surf->O->Hyperbolic_pair->unrank_point(
							w, 1, K1[i], 0 /* verbose_level*/);
					cout << i << " / " << 6 << " : ";
					Int_vec_print(cout, w, 6);
					cout << endl;
				}
			}

			Arc[0] = 1;
			Arc[1] = 2;
			for (i = 0; i < 4; i++) {
				Surf->O->Hyperbolic_pair->unrank_point(
						w, 1, K1[1 + i], 0 /* verbose_level*/);
				Int_vec_copy(w + 3, v, 3);
				F->Projective_space_basic->PG_element_rank_modified_lint(
						v, 1, 3, Arc[2 + i]);
			}
			if (f_v) {
				cout << "surface_classify_wedge::derived_arcs "
						"The associated arc is ";
				Lint_vec_print(cout, Arc, 6);
				cout << endl;
				for (i = 0; i < 6; i++) {
					F->Projective_space_basic->PG_element_unrank_modified_lint(
							v, 1, 3, Arc[i]);
					cout << i << " & " << Arc[i] << " & ";
					Int_vec_print(cout, v, 3);
					cout << " \\\\" << endl;
				}
			}

		}

		FREE_int(Starter_configuration_idx);
	}

	if (f_v) {
		cout << "surface_classify_wedge::derived_arcs done" << endl;
	}
}

void surface_classify_wedge::starter_configurations_which_are_involved(
		int iso_type,
		int *&Starter_configuration_idx, int &nb_starter_conf,
		int verbose_level)
// Determines the double sixes which are involved with the giben cubic surface.
{
	int f_v = (verbose_level >= 1);
	int /*k,*/ i, j, cnt, iso;
	//int nb_orbits;

	if (f_v) {
		cout << "surface_classify_wedge::starter_configurations_which_are_involved" << endl;
		}

	//k = 5;

	//nb_orbits = Classify_double_sixes->Five_plus_one->nb_orbits_at_level(k);
	cnt = 0;
	for (i = 0; i < Classify_double_sixes->nb; i++) {

		// loop over all 5+1 configurations which have rank 19 and are good

		j = Classify_double_sixes->Double_sixes->Orbit[i].orbit_index;
		iso = Surfaces->Orbit[j].orbit_index;
		//iso = is_isomorphic_to[i];
		if (iso == iso_type) {
			cnt++;
		}
	}
	nb_starter_conf = cnt;

	Starter_configuration_idx = NEW_int(nb_starter_conf);

	cnt = 0;
	for (i = 0; i < Classify_double_sixes->nb; i++) {
		j = Classify_double_sixes->Double_sixes->Orbit[i].orbit_index;
		iso = Surfaces->Orbit[j].orbit_index;
		//iso = is_isomorphic_to[i];
		if (iso == iso_type) {
			Starter_configuration_idx[cnt++] = i;
		}
	}

	if (f_v) {
		cout << "surface_classify_wedge::starter_configurations_which_are_involved" << endl;
	}
}







}}}}



