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

	Classify_double_sixes = NULL;

	Flag_orbits = NULL;
	Surfaces = NULL;
	

}

surface_classify_wedge::~surface_classify_wedge()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);


#if 0
	if (Surf) {
		FREE_OBJECT(Surf);
	}
	if (Surf_A) {
		FREE_OBJECT(Surf_A);
	}
#endif

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
		cout << "surface_classify_wedge::~surface_classify_wedge before FREE_OBJECTS(Flag_orbits)" << endl;
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge before FREE_OBJECTS(Surfaces)" << endl;
	}
	if (Surfaces) {
		FREE_OBJECT(Surfaces);
	}

	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge before FREE_OBJECTS(Classify_double_sixes)" << endl;
	}
	if (Classify_double_sixes) {
		FREE_OBJECT(Classify_double_sixes);
	}
	if (f_v) {
		cout << "surface_classify_wedge::~surface_classify_wedge done" << endl;
	}
}

void surface_classify_wedge::init(
		cubic_surfaces_in_general::surface_with_action *Surf_A,
	poset_classification::poset_classification_control *Control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	
	if (f_v) {
		cout << "surface_classify_wedge::init" << endl;
	}
	surface_classify_wedge::F = Surf_A->PA->F;
	surface_classify_wedge::Surf_A = Surf_A;
	surface_classify_wedge::Surf = Surf_A->Surf;
	q = F->q;

	fname_base.assign("surface_");
	char str[1000];

	snprintf(str, sizeof(str), "%d", q);
	fname_base.append(str);

	
	
	A = Surf_A->PA->A;
	A2 = Surf_A->PA->A_on_lines;


	
	Elt0 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	Classify_double_sixes = NEW_OBJECT(classify_double_sixes);

	if (f_v) {
		cout << "surface_classify_wedge::init "
				"before Classify_double_sixes->init" << endl;
	}
	Classify_double_sixes->init(Surf_A, Control, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::init "
				"after Classify_double_sixes->init" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::init done" << endl;
	}
}

void surface_classify_wedge::do_classify_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::do_classify_double_sixes" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (test_if_double_sixes_have_been_computed_already()) {
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes before "
					"read_double_sixes" << endl;
		}
		read_double_sixes(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes after "
					"read_double_sixes" << endl;
		}
	}

	else {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes before "
					"Classify_double_sixes->classify_partial_ovoids" << endl;
		}
		Classify_double_sixes->classify_partial_ovoids(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes after "
					"Classify_double_sixes->classify_partial_ovoids" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes before "
					"Classify_double_sixes->classify" << endl;
		}
		Classify_double_sixes->classify(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes after "
					"Classify_double_sixes->classify" << endl;
		}



		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes before "
					"write_double_sixes" << endl;
		}
		write_double_sixes(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes after "
					"write_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes writing cheat sheet "
					"on double sixes" << endl;
		}
		create_report_double_sixes(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_double_sixes writing cheat sheet on "
					"double sixes done" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::do_classify_double_sixes done" << endl;
	}
}

void surface_classify_wedge::do_classify_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::do_classify_surfaces" << endl;
	}
	if (test_if_surfaces_have_been_computed_already()) {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces before "
					"read_surfaces" << endl;
		}
		read_surfaces(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces after "
					"read_surfaces" << endl;
		}

	}
	else {

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces classifying surfaces" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces before "
					"SCW->classify_surfaces_from_double_sixes" << endl;
		}
		classify_surfaces_from_double_sixes(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces after "
					"SCW->classify_surfaces_from_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces before "
					"write_surfaces" << endl;
		}
		write_surfaces(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::do_classify_surfaces after "
					"write_surfaces" << endl;
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
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes before downstep" << endl;
	}
	downstep(verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits
				<< " flag orbits out of "
				<< Classify_double_sixes->Double_sixes->nb_orbits
				<< " orbits of double sixes" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes before upstep" << endl;
	}
	upstep(verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes after upstep" << endl;
		cout << "we found " << Surfaces->nb_orbits
				<< " surfaces out from "
				<< Flag_orbits->nb_flag_orbits
				<< " double sixes" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::classify_surfaces_from_double_sixes done" << endl;
	}
}

void surface_classify_wedge::downstep(int verbose_level)
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
		ring_theory::longinteger_object go;
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
			FALSE /* f_long_orbit */,
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

void surface_classify_wedge::upstep(int verbose_level)
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

	ring_theory::longinteger_object go;
	A->group_order(go);

	Surfaces->init(A, A2,
			Flag_orbits->nb_flag_orbits, 27, go,
			verbose_level - 3);


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
		coset_reps->init(Surf_A->A, 0/*verbose_level - 2*/);
		coset_reps->allocate(36, 0/*verbose_level - 2*/);


		groups::strong_generators *S;
		ring_theory::longinteger_object go;


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
				double_six[j] = Lines[Surf->Schlaefli->Double_six[i * 12 + j]];
			}
			if (f_v) {
				cout << "f=" << f << " / "
						<< Flag_orbits->nb_flag_orbits
						<< ", upstep i=" << i
						<< " / 36 double_six=";
				Lint_vec_print(cout, double_six, 12);
				cout << endl;
			}
			
			Classify_double_sixes->identify_double_six(double_six, 
				Elt1 /* transporter */, f2, 0/*verbose_level - 10*/);

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
				A->element_move(Elt1, coset_reps->ith(nb_coset_reps), 0);
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
					Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;
					Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
					Flag_orbits->Flag_orbit_node[f2].fusion_elt
						= NEW_int(A->elt_size_in_int);
					A->element_invert(Elt1,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt, 0);
					f_processed[f2] = TRUE;
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


		coset_reps->reallocate(nb_coset_reps, 0/*verbose_level - 2*/);

		groups::strong_generators *Aut_gens;

		{
			ring_theory::longinteger_object ago;

			if (f_v) {
				cout << "surface_classify_wedge::upstep "
						"Extending the "
						"group by a factor of " << nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(groups::strong_generators);
			Aut_gens->init_group_extension(S, coset_reps,
					nb_coset_reps, 0/*verbose_level - 2*/);

			Aut_gens->group_order(ago);


			if (f_v) {
				cout << "the double six has a stabilizer of order "
						<< ago << endl;
				cout << "The double six stabilizer is:" << endl;
				Aut_gens->print_generators_tex(cout);
			}
		}



		Surfaces->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Surfaces,
			Flag_orbits->nb_primary_orbits_upper, 
			Aut_gens, Lines, NULL /* extra_data */, verbose_level - 4);

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);
		
		f_processed[f] = TRUE;
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


void surface_classify_wedge::derived_arcs(int verbose_level)
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
			cout << "surface " << iso_type << " / "
					<< Surfaces->nb_orbits << ":" << endl;
		}

		starter_configurations_which_are_involved(iso_type,
			Starter_configuration_idx, nb_starter_conf, verbose_level);

		if (f_v) {
			cout << "There are " << nb_starter_conf
					<< " starter configurations which are involved: " << endl;
			Int_vec_print(cout, Starter_configuration_idx, nb_starter_conf);
			cout << endl;
		}

		for (c = 0; c < nb_starter_conf; c++) {
			orb = Starter_configuration_idx[c];
			//s = Starter_configuration_idx[c];
			//orb = Classify_double_sixes->Idx[s];

			if (f_v) {
				cout << "configuration " << c << " / " << nb_starter_conf
						<< " is orbit " << orb << endl;
			}

			Classify_double_sixes->Five_plus_one->get_set_by_level(5, orb, S);

			if (f_v) {
				cout << "starter configuration as neighbors: ";
				Lint_vec_print(cout, S, 5);
				cout << endl;
			}


			orbiter_kernel_system::Orbiter->Lint_vec->apply(S, Classify_double_sixes->Neighbor_to_line, S2, 5);
			S2[5] = Classify_double_sixes->pt0_line;

			four_lines[0] = S2[0];
			four_lines[1] = S2[1];
			four_lines[2] = S2[2];
			four_lines[3] = S2[3];
			Surf->perp_of_four_lines(four_lines,
					trans12, perp_sz, 0 /* verbose_level */);

			if (trans12[0] == Classify_double_sixes->pt0_line) {
				b5 = trans12[1];
			}
			else if (trans12[1] == Classify_double_sixes->pt0_line) {
				b5 = trans12[0];
			}
			else {
				cout << "something is wrong with the starter configuration" << endl;
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
				cout << "The lines which meet { a_2, a_3, a_4 } "
						"and are skew to { a_1, b_5, b_6 } are: ";
				Lint_vec_print(cout, lines, nb_lines);
				cout << endl;
				cout << "generator matrices:" << endl;
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
				cout << "The lines which meet { a_1, a_3, a_4 } "
						"and are skew to { a_2, b_5, b_6 } are: ";
				Lint_vec_print(cout, lines, nb_lines);
				cout << endl;
				cout << "generator matrices:" << endl;
				Surf->Gr->print_set(lines, nb_lines);
			}

			FREE_lint(lines);


			if (f_v) {
				cout << "starter configuration as line ranks: ";
				Lint_vec_print(cout, S2, 6);
				cout << endl;
				cout << "b5=" << b5 << endl;
				cout << "generator matrices:" << endl;
				Surf->Gr->print_set(S2, 6);
				cout << "b5:" << endl;
				Surf->Gr->print_set(&b5, 1);
			}
			S2[6] = b5;

			for (int h = 0; h < 7; h++) {
				K1[h] = Surf->Klein->line_to_point_on_quadric(S2[h], 0/*verbose_level*/);
			}
			//int_vec_apply(S2, Surf->Klein->Line_to_point_on_quadric, K1, 7);
			if (f_v) {
				cout << "starter configuration on the klein quadric: ";
				Lint_vec_print(cout, K1, 7);
				cout << endl;
				for (i = 0; i < 7; i++) {
					Surf->O->Hyperbolic_pair->unrank_point(w, 1, K1[i], 0 /* verbose_level*/);
					cout << i << " / " << 6 << " : ";
					Int_vec_print(cout, w, 6);
					cout << endl;
				}
			}

			Arc[0] = 1;
			Arc[1] = 2;
			for (i = 0; i < 4; i++) {
				Surf->O->Hyperbolic_pair->unrank_point(w, 1, K1[1 + i], 0 /* verbose_level*/);
				Int_vec_copy(w + 3, v, 3);
				F->PG_element_rank_modified_lint(v, 1, 3, Arc[2 + i]);
			}
			if (f_v) {
				cout << "The associated arc is ";
				Lint_vec_print(cout, Arc, 6);
				cout << endl;
				for (i = 0; i < 6; i++) {
					F->PG_element_unrank_modified_lint(v, 1, 3, Arc[i]);
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
		int iso_type, int *&Starter_configuration_idx, int &nb_starter_conf,
		int verbose_level)
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


void surface_classify_wedge::write_file(
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_classify_wedge::write_file" << endl;
	}
	fp.write((char *) &q, sizeof(int));

	Flag_orbits->write_file(fp, verbose_level);

	Surfaces->write_file(fp, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::write_file finished" << endl;
	}
}

void surface_classify_wedge::read_file(
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q1;
	
	if (f_v) {
		cout << "surface_classify_wedge::read_file" << endl;
	}
	fp.read((char *) &q1, sizeof(int));
	if (q1 != q) {
		cout << "surface_classify_wedge::read_file q1 != q" << endl;
		exit(1);
	}

	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->read_file(fp, A, A2, verbose_level);

	Surfaces = NEW_OBJECT(invariant_relations::classification_step);

	ring_theory::longinteger_object go;

	A->group_order(go);

	Surfaces->read_file(fp, A, A2, go, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::read_file finished" << endl;
	}
}




void surface_classify_wedge::identify_Eckardt_and_print_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	//int m;
	
	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q);
	Nb_lines = NEW_int(q);
	//Nb_E = NEW_int(q);
	for (i = 0; i < q; i++) {
		Iso_type[i] = -1;
		//Nb_E[i] = -1;
	}
	identify_Eckardt(Iso_type, Nb_lines, verbose_level);

#if 0
	m = q - 3;
	cout << "\\begin{array}{c|*{" << m << "}{c}}" << endl;
	for (i = 0; i < m; i++) {
		cout << " & " << i + 2;
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	//cout << "\\# E ";
	for (i = 0; i < m; i++) {
		cout << " & ";
		if (Nb_E[i + 2] == -1) {
			cout << "\\times ";
		}
		else {
			cout << Nb_E[i + 2];
		}
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\mbox{Iso} ";
	for (i = 0; i < m; i++) {
		cout << " & ";
		if (Nb_E[i + 2] == -1) {
			cout << "\\times ";
		}
		else {
			cout << Iso_type[i + 2];
		}
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;
#endif

	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt_and_print_table done" << endl;
	}
}

void surface_classify_wedge::identify_F13_and_print_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q);
	Nb_lines = NEW_int(q);
	//Nb_E = NEW_int(q);

	for (a = 0; a < q; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	identify_F13(Iso_type, Nb_lines, verbose_level);


	cout << "\\begin{array}{|c|c|c|c|}" << endl;
	cout << "\\hline" << endl;
	cout << "a & a & \\# lines & \\mbox{OCN} \\\\" << endl;
	cout << "\\hline" << endl;
	for (a = 1; a < q; a++) {
		cout << a << " & ";
		F->print_element(cout, a);
		cout << " & ";
		cout << Nb_lines[a] << " & ";
		//cout << Nb_E[a] << " & ";
		cout << Iso_type[a] << "\\\\" << endl;
	}
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;

	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13_and_print_table done" << endl;
	}
}

void surface_classify_wedge::identify_Bes_and_print_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, c;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q * q);
	Nb_lines = NEW_int(q * q);
	//Nb_E = NEW_int(q * q);

	for (a = 0; a < q * q; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	identify_Bes(Iso_type, Nb_lines, verbose_level);


	//cout << "\\begin{array}{|c|c|c|}" << endl;
	//cout << "\\hline" << endl;
	cout << "(a,c); \\# lines & \\mbox{OCN} \\\\" << endl;
	//cout << "\\hline" << endl;
	for (a = 2; a < q; a++) {
		for (c = 2; c < q; c++) {
			cout << "(" << a << "," << c << "); (";
			F->print_element(cout, a);
			cout << ", ";
			F->print_element(cout, c);
			cout << "); ";
			cout << Nb_lines[a * q + c] << "; ";
			//cout << Nb_E[a * q + c] << "; ";
			cout << Iso_type[a * q + c];
			cout << "\\\\" << endl;
		}
	}
	//cout << "\\hline" << endl;
	//cout << "\\end{array}" << endl;


	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes_and_print_table done" << endl;
	}
}


void surface_classify_wedge::identify_Eckardt(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, alpha, beta;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_Eckardt "
			"looping over all a:" << endl;
	b = 1;
	for (a = 2; a < q - 1; a++) {
		cout << "surface_classify_wedge::identify_Eckardt "
				"a = " << a << endl;


		algebraic_geometry::surface_object *SO;
		
		SO = Surf->create_Eckardt_surface(
				a, b,
				alpha, beta,
				verbose_level);

		identify_surface(SO->eqn,
			iso_type, Elt,
			verbose_level);

		cout << "surface_classify_wedge::identify_Eckardt "
			"a = " << a << " is isomorphic to iso_type "
			<< iso_type << ", an isomorphism is:" << endl;
		A->element_print_quick(Elt, cout);

		Iso_type[a] = iso_type;
		Nb_lines[a] = SO->nb_lines;
		//Nb_E[a] = nb_E;

		FREE_OBJECT(SO);
			
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt done" << endl;
	}
}

void surface_classify_wedge::identify_F13(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_F13 "
			"looping over all a:" << endl;
	for (a = 1; a < q; a++) {
		cout << "surface_classify_wedge::identify_F13 "
				"a = " << a << endl;

		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;

		algebraic_geometry::surface_object *SO;

		SO = Surf->create_surface_F13(a, verbose_level);


		identify_surface(SO->eqn,
			iso_type, Elt,
			verbose_level);

		cout << "surface_classify_wedge::identify_F13 "
			"a = " << a << " is isomorphic to iso_type "
			<< iso_type << ", an isomorphism is:" << endl;
		A->element_print_quick(Elt, cout);

		Iso_type[a] = iso_type;
		Nb_lines[a] = SO->nb_lines;
		//Nb_E[a] = nb_E;
		FREE_OBJECT(SO);

	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_F13 done" << endl;
	}
}

void surface_classify_wedge::identify_Bes(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, c;
	//int *coeff;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_Bes "
			"looping over all a:" << endl;

	for (i = 0; i < q * q; i++) {
		Iso_type[i] = -1;
		//Nb_E[i] = -1;
	}
	for (a = 2; a < q; a++) {
		cout << "surface_classify_wedge::identify_Bes "
				"a = " << a << endl;

		for (c = 2; c < q; c++) {
			cout << "surface_classify_wedge::identify_Bes "
					"a = " << a << " c = " << c << endl;

			Iso_type[a * q + c] = -1;
			Nb_lines[a * q + c] = -1;
			//Nb_E[a * q + c] = -1;

			algebraic_geometry::surface_object *SO;

			SO = Surf->create_surface_bes(a, c, verbose_level);

			cout << "surface_classify_wedge::identify_Bes "
					"nb_lines = " << SO->nb_lines << endl;

			identify_surface(SO->eqn,
				iso_type, Elt,
				verbose_level);

			cout << "surface_classify_wedge::identify_Bes "
				"a = " << a << " c = " << c << " is isomorphic to iso_type "
				<< iso_type << ", an isomorphism is:" << endl;
			A->element_print_quick(Elt, cout);

			Iso_type[a * q + c] = iso_type;
			Nb_lines[a * q + c] = SO->nb_lines;
			//Nb_E[a * q + c] = nb_E;
			FREE_OBJECT(SO);
		}
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes done" << endl;
	}
}



int surface_classify_wedge::isomorphism_test_pairwise(
		cubic_surfaces_in_general::surface_create *SC1,
		cubic_surfaces_in_general::surface_create *SC2,
	int &isomorphic_to1, int &isomorphic_to2,
	int *Elt_isomorphism_1to2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2, *Elt3;
	int ret;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise" << endl;
	}
	int *coeff1;
	int *coeff2;

	coeff1 = SC1->SO->eqn;
	coeff2 = SC2->SO->eqn;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	identify_surface(
		coeff1,
		isomorphic_to1, Elt1,
		verbose_level - 1);
	identify_surface(
		coeff2,
		isomorphic_to2, Elt2,
		verbose_level - 1);
	if (isomorphic_to1 != isomorphic_to2) {
		ret = FALSE;
		if (f_v) {
			cout << "surface_classify_wedge::isomorphism_test_pairwise "
					"not isomorphic" << endl;
		}
	}
	else {
		ret = TRUE;
		if (f_v) {
			cout << "surface_classify_wedge::isomorphism_test_pairwise "
					"they are isomorphic" << endl;
		}
		A->element_invert(Elt2, Elt3, 0);
		A->element_mult(Elt1, Elt3, Elt_isomorphism_1to2, 0);
		if (f_v) {
			cout << "an isomorphism from surface1 to surface2 is" << endl;
			A->element_print(Elt_isomorphism_1to2, cout);
		}
		groups::matrix_group *mtx;

		mtx = A->G.matrix_grp;

		if (f_v) {
			cout << "testing the isomorphism" << endl;
			A->element_print(Elt_isomorphism_1to2, cout);
			cout << "from: ";
			Int_vec_print(cout, coeff1, 20);
			cout << endl;
			cout << "to  : ";
			Int_vec_print(cout, coeff2, 20);
			cout << endl;
		}
		A->element_invert(Elt_isomorphism_1to2, Elt1, 0);
		if (f_v) {
			cout << "the inverse element is" << endl;
			A->element_print(Elt1, cout);
		}
		int coeff3[20];
		int coeff4[20];
		mtx->substitute_surface_equation(Elt1,
				coeff1, coeff3, Surf,
				verbose_level - 1);

		Int_vec_copy(coeff2, coeff4, 20);
		F->PG_element_normalize_from_front(
				coeff3, 1,
				Surf->nb_monomials);
		F->PG_element_normalize_from_front(
				coeff4, 1,
				Surf->nb_monomials);

		if (f_v) {
			cout << "after substitution, normalized" << endl;
			cout << "    : ";
			Int_vec_print(cout, coeff3, 20);
			cout << endl;
			cout << "coeff2, normalized" << endl;
			cout << "    : ";
			Int_vec_print(cout, coeff4, 20);
			cout << endl;
		}
		if (Sorting.int_vec_compare(coeff3, coeff4, 20)) {
			cout << "The surface equations are not equal. That is bad." << endl;
			exit(1);
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise done" << endl;
	}
	return ret;
}

void surface_classify_wedge::identify_surface(
	int *coeff_of_given_surface,
	int &isomorphic_to, int *Elt_isomorphism, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int line_idx, subset_idx;
	int double_six_orbit, iso_type, idx2;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface" << endl;
	}

	isomorphic_to = -1;

	int nb_points;
	//int nb_lines;

	if (f_v) {
		cout << "identifying the surface ";
		Int_vec_print(cout, coeff_of_given_surface,
			Surf->nb_monomials);
		cout << " = ";
		Surf->print_equation(cout, coeff_of_given_surface);
		cout << endl;
	}


	//Points = NEW_lint(Surf->P->N_points);

	// find all the points on the surface based on the equation:

	vector<long int> My_Points;
	int h;

	Surf->enumerate_points(coeff_of_given_surface, My_Points, 0/*verbose_level - 2*/);

	nb_points = My_Points.size();

	if (f_v) {
		cout << "The surface to be identified has "
				<< nb_points << " points" << endl;
	}

	// find all lines which are completely contained in the
	// set of points:

	geometry::geometry_global Geo;
	vector<long int> My_Lines;

	Geo.find_lines_which_are_contained(Surf->P,
			My_Points,
			My_Lines,
			0/*verbose_level - 2*/);

	// the lines are not arranged according to a double six

	if (f_v) {
		cout << "The surface has " << nb_points << " points and " << My_Lines.size() << " lines" << endl;
	}
	if (My_Lines.size() != 27 /*&& nb_lines != 21*/) {
		cout << "the input surface has " << My_Lines.size() << " lines" << endl;
		cout << "something is wrong with the input surface, skipping" << endl;
		cout << "Points:";
		orbiter_kernel_system::Orbiter->Lint_vec->print(cout, My_Points);
		cout << endl;
		cout << "Lines:";
		orbiter_kernel_system::Orbiter->Lint_vec->print(cout, My_Lines);
		cout << endl;

		return;
	}


	long int *Points;
	long int *Lines;

	Points = NEW_lint(nb_points);
	for (h = 0; h < nb_points; h++) {
		Points[h] = My_Points[h];
	}

	Lines = NEW_lint(27);

	for (h = 0; h < 27; h++) {
		Lines[h] = My_Lines[h];
	}

	int *Adj;


	Surf->compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines, 27 /* nb_lines */, 0 /* verbose_level */);


	data_structures::set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;

	line_intersections = NEW_OBJECT(data_structures::set_of_sets);

	line_intersections->init_from_adjacency_matrix(
		27 /* nb_lines*/, Adj,
		0 /* verbose_level */);

	Surf->list_starter_configurations(Lines, 27,
		line_intersections, Starter_Table, nb_starter,
		0/*verbose_level*/);

	long int S3[6];
	long int K1[6];
	long int W4[6];
	int l;
	int f;

	if (nb_starter == 0) {
		cout << "nb_starter == 0" << endl;
		exit(1);
	}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	Surf->create_starter_configuration(line_idx, subset_idx,
		line_intersections, Lines, S3, 0 /* verbose_level */);


	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"The starter configuration is S3=";
		Lint_vec_print(cout, S3, 6);
		cout << endl;
	}

	int i;
	for (i = 0; i < 6; i++) {
		K1[i] = Surf->Klein->line_to_point_on_quadric(S3[i], 0 /* verbose_level */);
	}
	//lint_vec_apply(S3, Surf->Klein->line_to_point_on_quadric, K1, 6);
		// transform the five lines plus transversal 
		// into points on the Klein quadric

	for (h = 0; h < 5; h++) {
		f = Surf->O->evaluate_bilinear_form_by_rank(K1[h], K1[5]);
		if (f) {
			cout << "surface_classify_wedge::identify_surface "
					"K1[" << h << "] and K1[5] are not collinear" << endl;
			exit(1);
		}
	}


	//Surf->line_to_wedge_vec(S3, W1, 5);
		// transform the five lines into wedge coordinates

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Classify_double_sixes->identify_five_plus_one" << endl;
	}
	Classify_double_sixes->identify_five_plus_one(
		S3 /* five_lines */,
		S3[5] /* transversal_line */,
		W4 /* int *five_lines_out_as_neighbors */,
		idx2 /* &orbit_index */,
		Elt2 /* transporter */,
		0/*verbose_level - 2*/);
	
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
			"The five plus one configuration lies in orbit "
			<< idx2 << endl;
		cout << "An isomorphism is given by:" << endl;
		A->element_print_quick(Elt2, cout);
	}

#if 0


	A->make_element_which_moves_a_line_in_PG3q(
			Surf->Gr, S3[5], Elt0, 0 /* verbose_level */);


	A2->map_a_set(W1, W2, 5, Elt0, 0 /* verbose_level */);

	int_vec_search_vec(Classify_double_sixes->Neighbors,
			Classify_double_sixes->nb_neighbors, W2, 5, W3);

	if (f_v) {
		cout << "down coset " << l << " / " << nb_starter
			<< " tracing the set ";
		int_vec_print(cout, W3, 5);
		cout << endl;
		}
	idx2 = Classify_double_sixes->gen->trace_set(
			W3, 5, 5, W4, Elt1, 0 /* verbose_level */);

	
	A->element_mult(Elt0, Elt1, Elt2, 0);
#endif


	if (!Sorting.int_vec_search(Classify_double_sixes->Po,
			Classify_double_sixes->Flag_orbits->nb_flag_orbits,
			idx2, f)) {
		cout << "cannot find orbit in Po" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"flag orbit = " << f << endl;
	}


	double_six_orbit =
		Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"double_six_orbit = "
				<< double_six_orbit << endl;
	}

	if (double_six_orbit < 0) {
		cout << "surface_classify_wedge::identify_surface "
				"double_six_orbit < 0, something is wrong" << endl;
		exit(1);
	}
	if (Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].f_fusion_node) {

		if (f_v) {
			cout << "surface_classify_wedge::identify_surface "
					"the flag orbit is a fusion node" << endl;
		}

		A->element_mult(Elt2,
			Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].fusion_elt,
			Elt3, 0);
	}
	else {

		if (f_v) {
			cout << "surface_classify_wedge::identify_surface "
					"the flag orbit is a definition node" << endl;
		}

		A->element_move(Elt2, Elt3, 0);
	}

	if (f_v) {
		cout << "An isomorphism is given by:" << endl;
		A->element_print_quick(Elt3, cout);
	}

	iso_type = Flag_orbits->Flag_orbit_node[double_six_orbit].upstep_primary_orbit;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"iso_type = " << iso_type << endl;
	}

	if (Flag_orbits->Flag_orbit_node[double_six_orbit].f_fusion_node) {
		A->element_mult(Elt3,
			Flag_orbits->Flag_orbit_node[double_six_orbit].fusion_elt,
			Elt_isomorphism, 0);
	}
	else {
		A->element_move(Elt3, Elt_isomorphism, 0);
	}

	//iso_type = is_isomorphic_to[orb2];
	//A->element_mult(Elt2, Isomorphisms->ith(orb2), Elt_isomorphism, 0);

	if (f_v) {
		cout << "The surface is isomorphic to surface " << iso_type << endl;
		cout << "An isomorphism is given by:" << endl;
		A->element_print_quick(Elt_isomorphism, cout);
	}
	isomorphic_to = iso_type;

	int *Elt_isomorphism_inv;

	Elt_isomorphism_inv = NEW_int(A->elt_size_in_int);
	A->element_invert(Elt_isomorphism, Elt_isomorphism_inv, 0);

	long int *image;

	image = NEW_lint(nb_points);
	A->map_a_set_and_reorder(Points, image,
			nb_points, Elt_isomorphism,
			0 /* verbose_level */);

	if (f_v) {
		cout << "The inverse isomorphism is given by:" << endl;
		A->element_print_quick(Elt_isomorphism_inv, cout);

		cout << "The image of the set of points is: ";
		Lint_vec_print(cout, image, nb_points);
		cout << endl;
	}

#if 0
	int i;
	for (i = 0; i < nb_points; i++) {
		if (image[i] != The_surface[isomorphic_to]->Surface[i]) {
			cout << "points disagree!" << endl;
			exit(1);
		}
	}
	cout << "the image set agrees with the point "
			"set of the chosen representative" << endl;
#endif

	FREE_lint(image);

	int *coeffs_transformed;

	coeffs_transformed = NEW_int(Surf->nb_monomials);
	



	int idx;
	long int Lines0[27];
	int eqn0[20];

	cout << "the surface in the list is = " << endl;
	idx = Surfaces->Orbit[isomorphic_to].orbit_index;
	Lint_vec_copy(Surfaces->Rep +
			idx * Surfaces->representation_sz,
			Lines0, 27);
	
	Surf->build_cubic_surface_from_lines(
			27, Lines0, eqn0,
			0 /* verbose_level*/);
	F->PG_element_normalize_from_front(eqn0, 1, Surf->nb_monomials);

	Int_vec_print(cout, eqn0, Surf->nb_monomials);
	//int_vec_print(cout,
	//The_surface[isomorphic_to]->coeff, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, eqn0);
	cout << endl;


	groups::matrix_group *mtx;

	mtx = A->G.matrix_grp;
	
	mtx->substitute_surface_equation(Elt_isomorphism_inv,
			coeff_of_given_surface, coeffs_transformed, Surf,
			verbose_level - 1);

#if 0
	cout << "coeffs_transformed = " << endl;
	int_vec_print(cout, coeffs_transformed, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeffs_transformed);
	cout << endl;
#endif

	F->PG_element_normalize_from_front(
			coeffs_transformed, 1,
			Surf->nb_monomials);

	cout << "the surface to be identified was " << endl;
	Int_vec_print(cout, coeff_of_given_surface, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeff_of_given_surface);
	cout << endl;


	cout << "coeffs_transformed (and normalized) = " << endl;
	Int_vec_print(cout, coeffs_transformed, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeffs_transformed);
	cout << endl;


	

	FREE_OBJECT(line_intersections);
	FREE_int(Starter_Table);
	FREE_int(Adj);
	FREE_lint(Points);
	FREE_lint(Lines);
	FREE_int(Elt_isomorphism_inv);
	FREE_int(coeffs_transformed);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface done" << endl;
	}
}

void surface_classify_wedge::latex_surfaces(
		ostream &ost, int f_with_stabilizers, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char str[1000];
	string title;

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces" << endl;
	}
	snprintf(str, sizeof(str), "Cubic Surfaces with 27 Lines in $\\PG(3,%d)$", q);
	title.assign(str);


	ost << "\\subsection*{The Group $\\PGGL(4," << q << ")$}" << endl;

	{
		ring_theory::longinteger_object go;
		A->Strong_gens->group_order(go);

		ost << "The order of the group is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}

#if 0
	Classify_double_sixes->print_five_plus_ones(ost);


	Classify_double_sixes->Double_sixes->print_latex(ost, title_ds);
#endif

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces before Surfaces->print_latex" << endl;
	}
	Surfaces->print_latex(ost, title, f_with_stabilizers,
			FALSE, NULL, NULL);
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces after Surfaces->print_latex" << endl;
	}


#if 1
	int orbit_index;
	
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"before loop over all surfaces" << endl;
	}
	for (orbit_index = 0; orbit_index < Surfaces->nb_orbits; orbit_index++) {
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"before report_surface, orbit_index = " << orbit_index << endl;
		}
		report_surface(ost, orbit_index, verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"after report_surface" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"after loop over all surfaces" << endl;
	}
#endif
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces done" << endl;
	}
}

void surface_classify_wedge::report_surface(
		ostream &ost, int orbit_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::set_and_stabilizer *SaS;
	long int Lines[27];
	int equation[20];

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"orbit_index = " << orbit_index << endl;
	}

	ost << endl; // << "\\clearpage" << endl << endl;
	ost << "\\section*{Surface $" << q << "\\#"
			<< orbit_index << "$}" << endl;


	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before Surfaces->get_set_and_stabilizer" << endl;
	}
	SaS = Surfaces->get_set_and_stabilizer(orbit_index,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"after Surfaces->get_set_and_stabilizer" << endl;
	}

	Lint_vec_copy(SaS->data, Lines, 27);
	
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before Surf->build_cubic_surface_from_lines" << endl;
		cout << "Surf->n = " << Surf->n << endl;
	}
	Surf->build_cubic_surface_from_lines(27,
			Lines, equation, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	F->PG_element_normalize_from_front(equation, 1, 20);


	//Surf->print_equation_wrapped(ost, equation);

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->init_with_27_lines" << endl;
	}
	algebraic_geometry::surface_object *SO;

	SO = NEW_OBJECT(algebraic_geometry::surface_object);
	SO->init_with_27_lines(Surf, Lines, equation,
			TRUE /*f_find_double_six_and_rearrange_lines*/,
			verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"after SO->init_with_27_lines" << endl;
	}


#if 0
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->enumerate_points" << endl;
	}
	SO->enumerate_points(verbose_level);
#endif

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->compute_properties" << endl;
	}
	SO->compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"after SO->compute_properties" << endl;
	}


	SO->SOP->print_equation(ost);


	cubic_surfaces_in_general::surface_object_with_action *SOA;

	SOA = NEW_OBJECT(cubic_surfaces_in_general::surface_object_with_action);

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SOA->init" << endl;
	}
	SOA->init_surface_object(Surf_A, SO, 
		SaS->Strong_gens, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"after SOA->init" << endl;
	}


	ring_theory::longinteger_object ago;
	SaS->Strong_gens->group_order(ago);
	ost << "The automorphism group of the surface has order " << ago << "\\\\" << endl;
	ost << "The automorphism group is the following group\\\\" << endl;

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SaS->Strong_gens->print_generators_tex" << endl;
	}
	SaS->Strong_gens->print_generators_tex(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_summary" << endl;
	}
	SO->SOP->print_summary(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;



	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_lines" << endl;
	}
	SO->SOP->print_lines(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;



	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_points" << endl;
		}
	SO->SOP->print_points(ost);


	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_Hesse_planes" << endl;
	}
	SO->SOP->print_Hesse_planes(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_tritangent_planes" << endl;
	}
	SO->SOP->print_tritangent_planes(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_axes" << endl;
	}
	SO->SOP->print_axes(ost);


	//New_clebsch->SO->print_planes_in_trihedral_pairs(fp);

#if 0
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SO->print_generalized_quadrangle" << endl;
	}
	SO->print_generalized_quadrangle(ost);

	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"before SOA->quartic" << endl;
	}
	SOA->quartic(ost,  verbose_level);
#endif

	FREE_OBJECT(SOA);
	FREE_OBJECT(SO);
	FREE_OBJECT(SaS);
	
	if (f_v) {
		cout << "surface_classify_wedge::report_surface "
				"orbit_index = " << orbit_index << " done" << endl;
	}
}

void surface_classify_wedge::generate_source_code(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	std::string fname;
	int orbit_index;
	int i, j;

	if (f_v) {
		cout << "surface_classify_wedge::generate_source_code" << endl;
	}
	fname.assign(fname_base);
	fname.append(".cpp");
	
	{
		ofstream f(fname.c_str());

		f << "static int " << fname_base.c_str() << "_nb_reps = "
				<< Surfaces->nb_orbits << ";" << endl;
		f << "static int " << fname_base.c_str() << "_size = "
				<< Surf->nb_monomials << ";" << endl;

	

		if (f_v) {
			cout << "surface_classify_wedge::generate_source_code "
					"preparing reps" << endl;
		}
		f << "// the equations:" << endl;
		f << "static int " << fname_base.c_str() << "_reps[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < Surfaces->nb_orbits;
				orbit_index++) {


			data_structures_groups::set_and_stabilizer *SaS;
			long int Lines[27];
			int equation[20];

			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			SaS = Surfaces->get_set_and_stabilizer(
					orbit_index, 0 /* verbose_level */);
			Lint_vec_copy(SaS->data, Lines, 27);

			Surf->build_cubic_surface_from_lines(27,
					Lines, equation, 0 /* verbose_level */);
			F->PG_element_normalize_from_front(equation, 1, 20);

			f << "\t";
			for (i = 0; i < Surf->nb_monomials; i++) {
				f << equation[i];
				f << ", ";
			}
			f << endl;

			FREE_OBJECT(SaS);
	
		}
		f << "};" << endl;



		if (f_v) {
			cout << "surface_classify_wedge::generate_source_code "
					"preparing stab_order" << endl;
		}
		f << "// the stabilizer orders:" << endl;
		f << "static const char *" << fname_base.c_str() << "_stab_order[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < Surfaces->nb_orbits;
				orbit_index++) {

			ring_theory::longinteger_object ago;

			Surfaces->Orbit[orbit_index].gens->group_order(ago);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;


		if (f_v) {
			cout << "surface_classify_wedge::generate_source_code "
					"preparing nb_E" << endl;
		}
		f << "// the number of Eckardt points:" << endl;
		f << "static int " << fname_base.c_str() << "_nb_E[] = { " << endl << "\t";
		for (orbit_index = 0;
				orbit_index < Surfaces->nb_orbits;
				orbit_index++) {
			data_structures_groups::set_and_stabilizer *SaS;
			long int Lines[27];
			int equation[27];
			long int *Pts;
			int nb_pts;
			data_structures::set_of_sets *pts_on_lines;
			int nb_E;

			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			SaS = Surfaces->get_set_and_stabilizer(
					orbit_index, 0 /* verbose_level */);
			Lint_vec_copy(SaS->data, Lines, 27);
			Surf->build_cubic_surface_from_lines(27,
					Lines, equation, 0 /* verbose_level */);
			F->PG_element_normalize_from_front(equation, 1, 20);

			Pts = NEW_lint(Surf->nb_pts_on_surface_with_27_lines);

			vector<long int> Points;
			int h;

			Surf->enumerate_points(equation, Points,
					0 /* verbose_level */);

			nb_pts = Points.size();
			Pts = NEW_lint(nb_pts);
			for (h = 0; h < nb_pts; h++) {
				Pts[h] = Points[h];
			}


			if (nb_pts != Surf->nb_pts_on_surface_with_27_lines) {
				cout << "surface_classify_wedge::generate_source_code "
						"nb_pts != Surf->nb_pts_on_surface_with_27_lines" << endl;
				exit(1);
			}

			int *f_is_on_line;

			Surf->compute_points_on_lines(Pts, nb_pts,
				Lines, 27 /*nb_lines*/,
				pts_on_lines,
				f_is_on_line,
				0/*verbose_level*/);

			FREE_int(f_is_on_line);

			nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);

			f << nb_E;
			if (orbit_index < Surfaces->nb_orbits - 1) {
				f << ", ";
			}
			if (((orbit_index + 1) % 10) == 0) {
				f << endl << "\t";
			}


			FREE_OBJECT(pts_on_lines);
			FREE_lint(Pts);
			FREE_OBJECT(SaS);
		}
		f << "};" << endl;


#if 0
		f << "static int " << prefix << "_single_six[] = { " << endl;
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			f << "\t" << The_surface[iso_type]->S2[5];
			for (j = 0; j < 5; j++) {
				f << ", ";
				f << The_surface[iso_type]->S2[j];
			}
			f << ", " << endl;
		}
		f << "};" << endl;
#endif

	
		if (f_v) {
			cout << "surface_classify_wedge::generate_source_code "
					"preparing Lines" << endl;
		}
		f << "// the lines in the order double six "
				"a_i, b_i and 15 more lines c_ij:" << endl;
		f << "static int " << fname_base.c_str() << "_Lines[] = { " << endl;


		for (orbit_index = 0;
				orbit_index < Surfaces->nb_orbits;
				orbit_index++) {


			data_structures_groups::set_and_stabilizer *SaS;
			long int Lines[27];

			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			SaS = Surfaces->get_set_and_stabilizer(
					orbit_index, 0 /* verbose_level */);
			Lint_vec_copy(SaS->data, Lines, 27);

			f << "\t";
			for (j = 0; j < 27; j++) {
				f << Lines[j];
				f << ", ";
			}
			f << endl;

			FREE_OBJECT(SaS);
		}
		f << "};" << endl;

		f << "static int " << fname_base.c_str() << "_make_element_size = "
				<< A->make_element_size << ";" << endl;

		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;

			stab_gens_first = NEW_int(Surfaces->nb_orbits);
			stab_gens_len = NEW_int(Surfaces->nb_orbits);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < Surfaces->nb_orbits;
					orbit_index++) {
				stab_gens_first[orbit_index] = fst;
				stab_gens_len[orbit_index] =
						Surfaces->Orbit[orbit_index].gens->gens->len;
				//stab_gens_len[orbit_index] =
				//The_surface[iso_type]->stab_gens->gens->len;
				fst += stab_gens_len[orbit_index];
			}

	
			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"preparing stab_gens_fst" << endl;
			}
			f << "static int " << fname_base.c_str() << "_stab_gens_fst[] = { ";
			for (orbit_index = 0;
					orbit_index < Surfaces->nb_orbits;
					orbit_index++) {
				f << stab_gens_first[orbit_index];
				if (orbit_index < Surfaces->nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"preparing stab_gens_len" << endl;
			}
			f << "static int " << fname_base.c_str() << "_stab_gens_len[] = { ";
			for (orbit_index = 0;
					orbit_index < Surfaces->nb_orbits;
					orbit_index++) {
				f << stab_gens_len[orbit_index];
				if (orbit_index < Surfaces->nb_orbits - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;


			if (f_v) {
				cout << "surface_classify_wedge::generate_source_code "
						"preparing stab_gens" << endl;
			}
			f << "static int " << fname_base.c_str() << "_stab_gens[] = {" << endl;
			for (orbit_index = 0;
					orbit_index < Surfaces->nb_orbits;
					orbit_index++) {
				int j;

				for (j = 0; j < stab_gens_len[orbit_index]; j++) {
					if (f_vv) {
						cout << "surface_classify_wedge::generate_source_code "
								"before extract_strong_generators_in_"
								"order generator " << j << " / "
								<< stab_gens_len[orbit_index] << endl;
					}
					f << "\t";
					A->element_print_for_make_element(
							Surfaces->Orbit[orbit_index].gens->gens->ith(j), f);
					//A->element_print_for_make_element(
					//The_surface[iso_type]->stab_gens->gens->ith(j), f);
					f << endl;
				}
			}
			f << "};" << endl;


			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname.c_str()) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::generate_source_code done" << endl;
	}
}


void surface_classify_wedge::generate_history(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::generate_history" << endl;
	}
	Classify_double_sixes->Five_plus_one->generate_history(5, verbose_level - 2);
	if (f_v) {
		cout << "surface_classify_wedge::generate_history done" << endl;
	}

}

int surface_classify_wedge::test_if_surfaces_have_been_computed_already()
{
	char fname[1000];
	orbiter_kernel_system::file_io Fio;
	int ret;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	if (Fio.file_size(fname) > 0) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}
	return ret;
}

void surface_classify_wedge::write_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->write_file" << endl;
		}
		write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces done" << endl;
	}
}

void surface_classify_wedge::read_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	cout << "Reading file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	{
		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->read_file" << endl;
			}
		read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->read_file" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces done" << endl;
	}
}

int surface_classify_wedge::test_if_double_sixes_have_been_computed_already()
{
	char fname[1000];
	orbiter_kernel_system::file_io Fio;
	int ret;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	if (Fio.file_size(fname) > 0) {
		ret = TRUE;
	}
	else {
		ret = FALSE;
	}
	return ret;
}

void surface_classify_wedge::write_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	{

	ofstream fp(fname);

	if (f_v) {
		cout << "surface_classify before "
				"SCW->Classify_double_sixes->write_file" << endl;
		}
	Classify_double_sixes->write_file(fp, verbose_level - 1);
	if (f_v) {
		cout << "surface_classify after "
				"SCW->Classify_double_sixes->write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes done" << endl;
	}
}

void surface_classify_wedge::read_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	if (f_v) {
		cout << "Reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
		Classify_double_sixes->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes done" << endl;
	}
}


void surface_classify_wedge::create_report(int f_with_stabilizers,
		graphics::layered_graph_draw_options *draw_options,
		poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report" << endl;
	}
	char str[1000];
	string fname, title, author, extra_praeamble;
	orbiter_kernel_system::file_io Fio;

	snprintf(str, 1000, "Cubic Surfaces with 27 Lines over GF(%d) ", q);
	title.assign(str);

	strcpy(str, "Orbiter");
	author.assign(str);

	snprintf(str, 1000, "Surfaces_q%d.tex", q);
	fname.assign(str);


		{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "surface_classify_wedge::create_report before report" << endl;
		}
		report(fp, f_with_stabilizers, draw_options, Opt, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify_wedge::create_report after report" << endl;
		}


		L.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void surface_classify_wedge::report(ostream &ost, int f_with_stabilizers,
		graphics::layered_graph_draw_options *draw_options,
		poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::report" << endl;
	}
	orbiter_kernel_system::latex_interface L;


#if 0
	ost << "\\section{The field of order " << LG->F->q << "}" << endl;
	ost << "\\noindent The field ${\\mathbb F}_{"
			<< LG->F->q
			<< "}$ :\\\\" << endl;
	LG->F->cheat_sheet(ost, verbose_level);
#endif

	if (f_v) {
		cout << "surface_classify_wedge::report before Classify_double_sixes->report" << endl;
	}
	Classify_double_sixes->report(ost, draw_options, Opt, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report after Classify_double_sixes->report" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->print_five_plus_ones" << endl;
	}
	Classify_double_sixes->print_five_plus_ones(ost);
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->print_five_plus_ones" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->Flag_orbits->print_latex" << endl;
	}

	{
		string title;

		title.assign("Flag orbits for double sixes");

		Classify_double_sixes->Flag_orbits->print_latex(ost, title, TRUE);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->Flag_orbits->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->Double_sixes->print_latex" << endl;
	}
	{
		string title;

		title.assign("Double Sixes");
		Classify_double_sixes->Double_sixes->print_latex(ost, title, TRUE,
				FALSE, NULL, NULL);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->Double_sixes->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report before Flag_orbits->print_latex" << endl;
	}
	{
		string title;

		title.assign("Flag orbits for double surfaces");

		Flag_orbits->print_latex(ost, title, TRUE);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report after Flag_orbits->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report before Surfaces->print_latex" << endl;
	}
	{
		string title;

		title.assign("Surfaces");
		Surfaces->print_latex(ost, title, TRUE,
				FALSE, NULL, NULL);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report after Surfaces->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report before latex_surfaces" << endl;
	}
	latex_surfaces(ost, f_with_stabilizers, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report after latex_surfaces" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report done" << endl;
	}
}

void surface_classify_wedge::create_report_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes" << endl;
	}


	char str[1000];
	string fname, title, author, extra_praeamble;

	snprintf(str, 1000, "Cheat Sheet on Double Sixes over GF(%d) ", q);
	title.assign(str);
	snprintf(str, 1000, "Double_sixes_q%d.tex", q);
	fname.assign(str);

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"before Classify_double_sixes->print_five_plus_ones" << endl;
		}
		Classify_double_sixes->print_five_plus_ones(fp);
		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"after Classify_double_sixes->print_five_plus_ones" << endl;
		}

		{
			string title;

			title.assign("Double Sixes");
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"before Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
			Classify_double_sixes->Double_sixes->print_latex(fp,
				title, FALSE /* f_with_stabilizers*/,
				FALSE, NULL, NULL);
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"after Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
		}

		L.foot(fp);
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes done" << endl;
	}
}

void surface_classify_wedge::test_isomorphism(
		cubic_surfaces_in_general::surface_create_description *Descr1,
		cubic_surfaces_in_general::surface_create_description *Descr2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism" << endl;
	}
	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism Descr1 = " << endl;
		Descr1->print();
		cout << "surface_classify_wedge::test_isomorphism Descr2 = " << endl;
		Descr2->print();
	}

	cubic_surfaces_in_general::surface_create *SC1;
	cubic_surfaces_in_general::surface_create *SC2;
	SC1 = NEW_OBJECT(cubic_surfaces_in_general::surface_create);
	SC2 = NEW_OBJECT(cubic_surfaces_in_general::surface_create);

	if (f_v) {
		cout << "before SC1->create_cubic_surface" << endl;
	}
	SC1->create_cubic_surface(Descr1, verbose_level);
	if (f_v) {
		cout << "after SC1->create_cubic_surface" << endl;
	}

	if (f_v) {
		cout << "before SC2->create_cubic_surface" << endl;
	}
	SC2->create_cubic_surface(Descr2, verbose_level);
	if (f_v) {
		cout << "after SC2->create_cubic_surface" << endl;
	}

	int isomorphic_to1;
	int isomorphic_to2;
	int *Elt_isomorphism_1to2;

	Elt_isomorphism_1to2 = NEW_int(A->elt_size_in_int);
	if (isomorphism_test_pairwise(
			SC1, SC2,
			isomorphic_to1, isomorphic_to2,
			Elt_isomorphism_1to2,
			verbose_level)) {

		if (f_v) {
			cout << "The surfaces are isomorphic, "
					"an isomorphism is given by" << endl;
			A->element_print(Elt_isomorphism_1to2, cout);
			cout << "The surfaces belongs to iso type "
					<< isomorphic_to1 << endl;
		}
	}
	else {
		if (f_v) {
			cout << "The surfaces are NOT isomorphic." << endl;
			cout << "surface 1 belongs to iso type "
					<< isomorphic_to1 << endl;
			cout << "surface 2 belongs to iso type "
					<< isomorphic_to2 << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism done" << endl;
	}
}




void surface_classify_wedge::recognition(
		cubic_surfaces_in_general::surface_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::recognition" << endl;
	}
	cubic_surfaces_in_general::surface_create *SC;
	groups::strong_generators *SG;
	groups::strong_generators *SG0;

	SC = NEW_OBJECT(cubic_surfaces_in_general::surface_create);

	if (f_v) {
		cout << "before SC->init" << endl;
	}
	SC->init(Descr, verbose_level);
	if (f_v) {
		cout << "after SC->init" << endl;
	}

	int isomorphic_to;
	int *Elt_isomorphism;

	Elt_isomorphism = NEW_int(A->elt_size_in_int);
	identify_surface(
		SC->SO->eqn,
		isomorphic_to, Elt_isomorphism,
		verbose_level);
	if (f_v) {
		cout << "surface belongs to iso type "
				<< isomorphic_to << endl;
	}
	SG = NEW_OBJECT(groups::strong_generators);
	SG0 = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "before SG->stabilizer_of_cubic_surface_from_catalogue" << endl;
		}
	SG->stabilizer_of_cubic_surface_from_catalogue(
		Surf_A->A,
		F, isomorphic_to,
		verbose_level);

	SG0->init_generators_for_the_conjugate_group_aGav(
			SG, Elt_isomorphism, verbose_level);
	ring_theory::longinteger_object go;

	SG0->group_order(go);
	if (f_v) {
		cout << "The full stabilizer has order " << go << endl;
		cout << "And is generated by" << endl;
		SG0->print_generators_tex(cout);
	}
	if (f_v) {
		cout << "surface_classify_wedge::recognition done" << endl;
	}
}


void surface_classify_wedge::sweep_Cayley(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::sweep_Cayley" << endl;
	}
	int isomorphic_to;
	int *Elt_isomorphism;

	Elt_isomorphism = NEW_int(A->elt_size_in_int);


	int k, l, m, n;
	int q4 = q * q * q * q;
	int *Table;
	int *Table_reverse;
	int nb_iso;
	int cnt = 0;
	int nb_identified = 0;
	int h;


	knowledge_base K;

	nb_iso = K.cubic_surface_nb_reps(q);

	Table = NEW_int(q4 * 5);
	Table_reverse = NEW_int(nb_iso * 5);

	for (h = 0; h < nb_iso; h++) {
		Table_reverse[h * 5 + 0] = -1;
		Table_reverse[h * 5 + 1] = -1;
		Table_reverse[h * 5 + 2] = -1;
		Table_reverse[h * 5 + 3] = -1;
		Table_reverse[h * 5 + 4] = -1;
	}


	for (k = 0; k < q; k++) {
		for (l = 1; l < q; l++) {
			for (m = 1; m < q; m++) {
				for (n = 1; n < q; n++) {


					cubic_surfaces_in_general::surface_create_description Descr;
					cubic_surfaces_in_general::surface_create *SC;

					SC = NEW_OBJECT(cubic_surfaces_in_general::surface_create);

					Descr.f_Cayley_form = TRUE;
					Descr.Cayley_form_k = k;
					Descr.Cayley_form_l = l;
					Descr.Cayley_form_m = m;
					Descr.Cayley_form_n = n;

					//Descr.f_q = TRUE;
					//Descr.q = q;

					if (f_v) {
						cout << "k=" << k << " l=" << l << " m=" << m << " n=" << n << " before SC->init" << endl;
					}
					SC->init(&Descr, 0 /*verbose_level*/);
					if (FALSE) {
						cout << "after SC->init" << endl;
					}

					if (SC->SO->nb_lines == 27) {

						identify_surface(
							SC->SO->eqn,
							isomorphic_to, Elt_isomorphism,
							verbose_level);
						if (f_v) {
							cout << "surface " << SC->label_txt << " belongs to iso type " << isomorphic_to << endl;
						}

						Table[cnt * 5 + 0] = k;
						Table[cnt * 5 + 1] = l;
						Table[cnt * 5 + 2] = m;
						Table[cnt * 5 + 3] = n;
						Table[cnt * 5 + 4] = isomorphic_to;

						if (Table_reverse[isomorphic_to * 5 + 0] == -1) {
							Table_reverse[isomorphic_to * 5 + 0] = cnt;
							Table_reverse[isomorphic_to * 5 + 1] = k;
							Table_reverse[isomorphic_to * 5 + 2] = l;
							Table_reverse[isomorphic_to * 5 + 3] = m;
							Table_reverse[isomorphic_to * 5 + 4] = n;
							nb_identified++;
						}
						cnt++;

					}

					FREE_OBJECT(SC);

					if (nb_identified == nb_iso) {
						break;
					}

				}
				if (nb_identified == nb_iso) {
					break;
				}
			}
			if (nb_identified == nb_iso) {
				break;
			}
		}
		if (nb_identified == nb_iso) {
			break;
		}
	}

	string fname;
	char str[1000];

	fname.assign("Cayley_q");
	snprintf(str, sizeof(str), "%d.csv", q);
	fname.append(str);
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Table, cnt, 5);

	fname.assign("Cayley_reverse_q");
	snprintf(str, sizeof(str), "%d.csv", q);
	fname.append(str);

	Fio.int_matrix_write_csv(fname, Table_reverse, nb_iso, 5);



	FREE_int(Elt_isomorphism);


	if (f_v) {
		cout << "surface_classify_wedge::sweep_Cayley done" << endl;
	}
}

void surface_classify_wedge::identify_general_abcd(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b, c, d;
	int a0, b0; //, c0, d0;
	int iso_type;
	int *Elt;
	int q2, q3, q4;

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd" << endl;
	}


	q2 = q * q;
	q3 = q2 * q;
	q4 = q3 * q;

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_general_abcd "
			"looping over all a:" << endl;

	for (i = 0; i < q4; i++) {
		Iso_type[i] = -1;
		Nb_lines[i] = -1;
		//Nb_E[i] = -1;
	}
	for (a = 1; a < q; a++) {
		cout << "surface_classify_wedge::identify_general_abcd "
				"a = " << a << endl;

		if (a == 0 || a == 1) {
			continue;
		}

		for (b = 1; b < q; b++) {
			cout << "surface_classify_wedge::identify_general_abcd "
					" b = " << b << endl;

			if (b == 0 || b == 1) {
				continue;
			}

			if (b == a) {
				continue;
			}

			if (EVEN(q)) {
				F->minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(a, b,
						a0, b0, verbose_level);

				cout << "a=" << a << " b=" << b << " a0=" << a0 << " b0=" << b0 << endl;

				if (a0 < a) {
					cout << "skipping" << endl;
					continue;
				}
				if (a0 == a && b0 < b) {
					cout << "skipping" << endl;
					continue;
				}
			}

			for (c = 1; c < q; c++) {
				cout << "surface_classify_wedge::identify_general_abcd "
						"a = " << a << " b = " << b << " c = " << c << endl;


				if (c == 0 || c == 1) {
					continue;
				}

				if (c == a) {
					continue;
				}



				for (d = 1; d < q; d++) {

					if (d == 0 || d == 1) {
						continue;
					}

					if (d == b) {
						continue;
					}

					if (d == c) {
						continue;
					}

#if 1
					// ToDo
					// warning: special case

					if (d != a) {
						continue;
					}
#endif

					cout << "surface_classify_wedge::identify_general_abcd "
							"a = " << a << " b = " << b << " c = " << c << " d = " << d << endl;

					int m1;


					m1 = F->negate(1);


#if 0
					// this is a test for having 6 Eckardt points:
					int b2, b2v, x1, x2;

					b2 = F->mult(b, b);
					//cout << "b2=" << b2 << endl;
					b2v = F->inverse(b2);
					//cout << "b2v=" << b2v << endl;

					x1 = F->add(F->mult(2, b), m1);
					x2 = F->mult(x1, b2v);

					if (c != x2) {
						cout << "skipping" << endl;
						continue;
					}
#endif



#if 0
					F->minimal_orbit_rep_under_stabilizer_of_frame(c, d,
							c0, d0, verbose_level);

					cout << "c=" << c << " d=" << d << " c0=" << c0 << " d0=" << d0 << endl;


					if (c0 < c) {
						cout << "skipping" << endl;
						continue;
					}
					if (c0 == c && d0 < d) {
						cout << "skipping" << endl;
						continue;
					}
#endif


					cout << "nonconical test" << endl;

					int admbc;
					//int m1;
					int a1, b1, c1, d1;
					int a1d1, b1c1;
					int ad, bc;
					int adb1c1, bca1d1;


					//m1 = F->negate(1);

					a1 = F->add(a, m1);
					b1 = F->add(b, m1);
					c1 = F->add(c, m1);
					d1 = F->add(d, m1);

					ad = F->mult(a, d);
					bc = F->mult(b, c);

					adb1c1 = F->mult3(ad, b1, c1);
					bca1d1 = F->mult3(bc, a1, d1);

					a1d1 = F->mult(a1, d1);
					b1c1 = F->mult(b1, c1);
					if (a1d1 == b1c1) {
						continue;
					}
					if (adb1c1 == bca1d1) {
						continue;
					}



					admbc = F->add(F->mult(a, d), F->negate(F->mult(b, c)));

					if (admbc == 0) {
						continue;
					}

					Iso_type[a * q3 + b * q2 + c * q + d] = -2;
					Nb_lines[a * q3 + b * q2 + c * q + d] = -1;
					//Nb_E[a * q3 + b * q2 + c * q + d] = -1;

					algebraic_geometry::surface_object *SO;

					SO = Surf->create_surface_general_abcd(a, b, c, d, verbose_level);


					identify_surface(SO->eqn,
						iso_type, Elt,
						verbose_level);

					cout << "surface_classify_wedge::identify_general_abcd "
							"a = " << a << " b = " << b << " c = " << c << " d = " << d
							<< " is isomorphic to iso_type "
						<< iso_type << ", an isomorphism is:" << endl;
					A->element_print_quick(Elt, cout);

					Iso_type[a * q3 + b * q2 + c * q + d] = iso_type;
					Nb_lines[a * q3 + b * q2 + c * q + d] = SO->nb_lines;
					//Nb_E[a * q3 + b * q2 + c * q + d] = nb_E;
				}
			}
		}
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd done" << endl;
	}
}

void surface_classify_wedge::identify_general_abcd_and_print_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int q4, q3, q2;

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table" << endl;
	}

	q2 = q * q;
	q3 = q2 * q;
	q4 = q3 * q;

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q4);
	Nb_lines = NEW_int(q4);
	//Nb_E = NEW_int(q4);

	for (a = 0; a < q4; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table before identify_general_abcd" << endl;
	}
	identify_general_abcd(Iso_type, Nb_lines, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table after identify_general_abcd" << endl;
	}


	//cout << "\\begin{array}{|c|c|c|}" << endl;
	//cout << "\\hline" << endl;
	cout << "(a,c); \\# lines & \\mbox{OCN} \\\\" << endl;
	//cout << "\\hline" << endl;
	for (a = 1; a < q; a++) {
		for (b = 1; b < q; b++) {
			for (c = 1; c < q; c++) {
				for (d = 1; d < q; d++) {
					cout << "$(" << a << "," << b << "," << c << "," << d << ")$; ";
#if 0
					cout << "$(";
						F->print_element(cout, a);
						cout << ", ";
						F->print_element(cout, b);
						cout << ", ";
						F->print_element(cout, c);
						cout << ", ";
						F->print_element(cout, d);
						cout << ")$; ";
#endif
					cout << Nb_lines[a * q3 + b * q2 + c * q + d] << "; ";
					//cout << Nb_E[a * q3 + b * q2 + c * q + d] << "; ";
					cout << Iso_type[a * q3 + b * q2 + c * q + d];
					cout << "\\\\" << endl;
				}
			}
		}
	}
	//cout << "\\hline" << endl;
	//cout << "\\end{array}" << endl;

	int *Table;
	int h = 0;
	int nb_lines, iso, nb_e;
	knowledge_base K;
	orbiter_kernel_system::file_io Fio;

	Table = NEW_int(q4 * 7);


	for (a = 1; a < q; a++) {
		for (b = 1; b < q; b++) {
			for (c = 1; c < q; c++) {
				for (d = 1; d < q; d++) {
					nb_lines = Nb_lines[a * q3 + b * q2 + c * q + d];
					iso = Iso_type[a * q3 + b * q2 + c * q + d];

					if (iso == -1) {
						continue;
					}

					if (iso >= 0) {
						nb_e = K.cubic_surface_nb_Eckardt_points(q, iso);
					}
					else {
						nb_e = -1;
					}

					Table[h * 7 + 0] = a;
					Table[h * 7 + 1] = b;
					Table[h * 7 + 2] = c;
					Table[h * 7 + 3] = d;
					Table[h * 7 + 4] = nb_lines;
					Table[h * 7 + 5] = iso;
					Table[h * 7 + 6] = nb_e;
					h++;
				}
			}
		}
	}

	char str[1000];
	string fname;

	snprintf(str, sizeof(str), "surface_recognize_abcd_q%d.csv", q);
	fname.assign(str);

	Fio.int_matrix_write_csv(fname, Table, h, 7);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table done" << endl;
	}
}



}}}}



