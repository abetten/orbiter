// classify_double_sixes.cpp
// 
// Anton Betten
//
// October 10, 2017
//
// based on surface_classify_wedge.cpp started September 2, 2016
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {



classify_double_sixes::classify_double_sixes()
{

	Five_p1 = NULL;

	Elt3 = NULL;

	len = 0;
	Idx = NULL;
	nb = 0;
	Po = NULL;


	Flag_orbits = NULL;

	Double_sixes = NULL;
}



classify_double_sixes::~classify_double_sixes()
{
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Idx) {
		FREE_int(Idx);
	}
	if (Po) {
		FREE_int(Po);
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (Double_sixes) {
		FREE_OBJECT(Double_sixes);
	}
}

void classify_double_sixes::init(
		classify_five_plus_one *Five_p1,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_double_sixes::init" << endl;
	}

	classify_double_sixes::Five_p1 = Five_p1;

	Elt3 = NEW_int(Five_p1->A->elt_size_in_int);

	if (f_v) {
		cout << "classify_double_sixes::init done" << endl;
	}
}






void classify_double_sixes::test_orbits(int verbose_level)
{
	//verbose_level += 2;
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);
	int i, r;
	long int S[5];
	long int S2[6];
	
	if (f_v) {
		cout << "classify_double_sixes::test_orbits" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	len = Five_p1->Five_plus_one->nb_orbits_at_level(5);

	if (f_v) {
		cout << "classify_double_sixes::test_orbits testing "
				<< len << " orbits of 5-sets of lines:" << endl;
	}
	nb = 0;
	Idx = NEW_int(len);
	for (i = 0; i < len; i++) {
		if ((i % 1000) == 0) {
			cout << "classify_double_sixes::test_orbits orbit "
				<< i << " / " << len << ":" << endl;
		}
		Five_p1->Five_plus_one->get_set_by_level(5, i, S);
		if (f_vv) {
			cout << "set: ";
			Lint_vec_print(cout, S, 5);
			cout << endl;
		}

		orbiter_kernel_system::Orbiter->Lint_vec->apply(
				S,
				Five_p1->Linear_complex->Neighbor_to_line,
				S2, 5);

		S2[5] = Five_p1->Linear_complex->pt0_line;

		if (f_vv) {
			cout << "5+1 lines = ";
			Lint_vec_print(cout, S2, 6);
			cout << endl;
		}

#if 1
		if (f_vv) {
			Five_p1->Surf->Gr->print_set(S2, 6);
		}
#endif

		r = Five_p1->Surf->rank_of_system(
				6, S2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "classify_double_sixes::test_orbits orbit "
					<< i << " / " << len
					<< " has rank = " << r << endl;
		}
		if (r == 19) {
			Idx[nb++] = i;
		}
	}

	if (f_v) {
		cout << "classify_double_sixes::test_orbits we found "
				<< nb << " / " << len
				<< " orbits where the rank is 19" << endl;
		cout << "Idx=";
		Int_vec_print(cout, Idx, nb);
		cout << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::test_orbits done" << endl;
	}
}


void classify_double_sixes::classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_double_sixes::classify" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify "
				"before downstep" << endl;
	}
	downstep(verbose_level - 2);
	if (f_v) {
		cout << "classify_double_sixes::classify "
				"after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits
				<< " flag orbits out of "
				<< Five_p1->Five_plus_one->nb_orbits_at_level(5)
				<< " orbits" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify "
				"before upstep" << endl;
	}
	upstep(verbose_level - 2);
	if (f_v) {
		cout << "classify_double_sixes::classify "
				"after upstep" << endl;
		cout << "we found " << Double_sixes->nb_orbits
				<< " double sixes out of "
				<< Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::classify done" << endl;
	}
}


void classify_double_sixes::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f, i, nb_orbits, nb_flag_orbits, c;

	if (f_v) {
		cout << "classify_double_sixes::downstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"before test_orbits" << endl;
	}
	test_orbits(0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"after test_orbits. Number of orbits = " << nb << endl;
	}
	if (false) {
		cout << "Idx=";
		Int_vec_print(cout, Idx, nb);
		cout << endl;
	}



	nb_orbits = Five_p1->Five_plus_one->nb_orbits_at_level(5);
	
	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(
			Five_p1->A,
			Five_p1->A2,
		nb_orbits, // nb_primary_orbits_lower,
		5 + 6 + 12, // pt_representation_sz,
		nb,
		1, // upper_bound_for_number_of_traces // ToDo
		NULL, // void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level)
		NULL, // void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)
		NULL, // void *free_received_trace_data
		verbose_level - 2);

	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"initializing flag orbits" << endl;
	}

	int f_process = false;
	int nb_100 = 1;

	if (nb > 1000) {
		f_process = true;
		nb_100 = nb / 100 + 1;
	}

	nb_flag_orbits = 0;
	for (f = 0; f < nb; f++) {

		i = Idx[f];
		if (f_v) {
			cout << "classify_double_sixes::downstep "
					"orbit " << f << " / " << nb
					<< " with rank = 19 is orbit "
					<< i << " / " << nb_orbits << endl;
		}
		if (f_process) {
			if ((f % nb_100) == 0) {
				cout << "classify_double_sixes::downstep orbit "
					<< i << " / " << nb_orbits
					<< ", progress at " << f / nb_100 << "%" << endl;
			}
		}

		data_structures_groups::set_and_stabilizer *R;
		ring_theory::longinteger_object ol;
		ring_theory::longinteger_object go;
		long int dataset[23];

		R = Five_p1->Five_plus_one->get_set_and_stabilizer(
				5 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);

		Five_p1->Five_plus_one->orbit_length(
				i /* node */, 5 /* level */, ol);

		R->Strong_gens->group_order(go);

		Lint_vec_copy(R->data, dataset, 5);

		orbiter_kernel_system::Orbiter->Lint_vec->apply(
				dataset,
				Five_p1->Linear_complex->Neighbor_to_line,
				dataset + 5,
				5);
		
		dataset[10] = Five_p1->Linear_complex->pt0_line;

		long int double_six[12];

		if (f_vv) {
			cout << "5+1 lines = ";
			Lint_vec_print(cout, dataset + 5, 6);
			cout << endl;
		}

		if (f_vv) {
			cout << "classify_double_sixes::downstep before "
					"five_plus_one_to_double_six" << endl;
		}
#if 0
		c = Five_p1->Surf_A->create_double_six_from_five_lines_with_a_common_transversal(
				dataset + 5,
				Five_p1->Linear_complex->pt0_line,
				double_six,
				0 /* verbose_level - 2*/);
#endif
		c = Five_p1->Surf_A->PA->P->Solid->five_plus_one_to_double_six(
				dataset + 5,
				Five_p1->Linear_complex->pt0_line,
				double_six,
				verbose_level - 2);

		if (f_vv) {
			cout << "classify_double_sixes::downstep after "
					"create_double_six_from_five_lines_with_a_common_transversal" << endl;
		}


		if (c) {

			if (f_vv) {
				cout << "The starter configuration is good, "
						"a double six has been computed:" << endl;
				Lint_matrix_print(double_six, 2, 6);
			}

			Lint_vec_copy(double_six, dataset + 11, 12);


			Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(
				Flag_orbits,
				nb_flag_orbits /* flag_orbit_index */,
				i /* downstep_primary_orbit */,
				0 /* downstep_secondary_orbit */,
				ol.as_int() /* downstep_orbit_len */,
				false /* f_long_orbit */,
				dataset /* int *pt_representation */,
				R->Strong_gens,
				verbose_level - 2);
			R->Strong_gens = NULL;

			if (f_vv) {
				cout << "orbit " << f << " / " << nb
					<< " with rank = 19 is orbit " << i
					<< " / " << nb_orbits << ", stab order "
					<< go << endl;
			}
			nb_flag_orbits++;
		}
		else {
			if (f_vv) {
				cout << "classify_double_sixes::downstep "
						"orbit " << f << " / " << nb
						<< " with rank = 19 does not yield a "
						"double six, skipping" << endl;
			}
		}


		FREE_OBJECT(R);
	}

	Flag_orbits->nb_flag_orbits = nb_flag_orbits;


	Po = NEW_int(nb_flag_orbits);
	for (f = 0; f < nb_flag_orbits; f++) {
		Po[f] = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_double_sixes::downstep we found "
			<< nb_flag_orbits << " flag orbits out of "
			<< nb_orbits << " orbits" << endl;
	}
	if (f_v) {
		cout << "classify_double_sixes::downstep "
				"initializing flag orbits done" << endl;
	}
}


void classify_double_sixes::upstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int i, j, h, k, i0;
	int f, po, so;
	int *f_processed;
	int nb_processed;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	Int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Double_sixes = NEW_OBJECT(invariant_relations::classification_step);

	ring_theory::longinteger_object go;
	Five_p1->A->group_order(go);

	Double_sixes->init(Five_p1->A,
			Five_p1->A2,
			Flag_orbits->nb_flag_orbits,
			12, go,
			verbose_level - 2);


	if (f_v) {
		cout << "flag orbit : downstep_primary_orbit" << endl;
		cout << "f : po" << endl;
		for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {
			po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
			cout << f << " : " << po << endl;
		}
	}
	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		long int dataset[23];
		
		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) /
				(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "classify_double_sixes::upstep "
				"Defining new orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< Flag_orbits->nb_flag_orbits
				<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit
			= Flag_orbits->nb_primary_orbits_upper;
		

		if (Flag_orbits->pt_representation_sz != 23) {
			cout << "Flag_orbits->pt_representation_sz != 23" << endl;
			exit(1);
		}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		Lint_vec_copy(Flag_orbits->Pt + f * 23, dataset, 23);




		data_structures_groups::vector_ge *coset_reps;
		int nb_coset_reps;
		
		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(Five_p1->Surf_A->A, verbose_level - 2);
		coset_reps->allocate(12, verbose_level - 2);


		groups::strong_generators *S;
		ring_theory::longinteger_object go;
		long int double_six[12];

		Lint_vec_copy(dataset + 11, double_six, 12);

		if (f_v) {
			cout << "double six:";
			Lint_vec_print(cout, double_six, 12);
			cout << endl;
		}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy(verbose_level - 2);
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so
					<< " go=" << go << endl;
		}

		nb_coset_reps = 0;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 6; j++) {
			
				if (f_v) {
					cout << "i=" << i << " j=" << j << endl;
				}
				long int transversal_line;
				long int five_lines[5];
				//int five_lines_in_wedge[5];
				long int five_lines_out_as_neighbors[5];
				int orbit_index;
				int f2;
				
				transversal_line = double_six[i * 6 + j];
				i0 = 1 - i;
				k = 0;
				for (h = 0; h < 6; h++) {
					if (h == j) {
						continue;
					}
					five_lines[k++] = double_six[i0 * 6 + h];
				}

				//int_vec_apply(five_lines,
				//Line_to_neighbor, five_lines_in_wedge, 5);
				
				if (f_v) {
					cout << "transversal_line = "
							<< transversal_line << " five_lines=";
					Lint_vec_print(cout, five_lines, 5);
					cout << endl;
				}
				Five_p1->identify_five_plus_one(
						five_lines,
						transversal_line,
					five_lines_out_as_neighbors,
					orbit_index,
					Elt3 /* transporter */,
					verbose_level - 4);

				if (f_v) {
					cout << "We found a transporter" << endl;
				}
				if (f_vv) {
					Five_p1->A->Group_element->element_print_quick(Elt3, cout);
				}

				if (!Sorting.int_vec_search(
						Po, Flag_orbits->nb_flag_orbits,
						orbit_index, f2)) {
					cout << "cannot find orbit " << orbit_index
							<< " in Po" << endl;
					cout << "Po=";
					Int_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
					cout << endl;
					exit(1);
				}

				if (Flag_orbits->Flag_orbit_node[f2].downstep_primary_orbit
						!= orbit_index) {
					cout << "Flag_orbits->Flag_orbit_node[f2].downstep_"
							"primary_orbit != orbit_index" << endl;
					exit(1);
				}





		
				if (f2 == f) {
					if (f_v) {
						cout << "We found an automorphism of "
								"the double six:" << endl;
					}
					if (f_vv) {
						Five_p1->A->Group_element->element_print_quick(Elt3, cout);
						cout << endl;
					}
					Five_p1->A->Group_element->element_move(
							Elt3, coset_reps->ith(nb_coset_reps), 0);
					nb_coset_reps++;
					//S->add_single_generator(Elt3,
					//2 /* group_index */, verbose_level - 2);
				}
				else {
					if (f_v) {
						cout << "We are identifying flag orbit "
								<< f2 << " with flag orbit " << f << endl;
					}
					if (!f_processed[f2]) {
						Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
							= Flag_orbits->nb_primary_orbits_upper;
						Flag_orbits->Flag_orbit_node[f2].f_fusion_node
							= true;
						Flag_orbits->Flag_orbit_node[f2].fusion_with
							= f;
						Flag_orbits->Flag_orbit_node[f2].fusion_elt
							= NEW_int(Five_p1->A->elt_size_in_int);
						Five_p1->A->Group_element->element_invert(
								Elt3,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
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
			} // next j
		} // next i


		coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

		groups::strong_generators *Aut_gens;

		{
			ring_theory::longinteger_object ago;

			if (f_v) {
				cout << "classify_double_sixes::upstep "
						"Extending the group by a factor of "
						<< nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(groups::strong_generators);
			Aut_gens->init_group_extension(S,
					coset_reps, nb_coset_reps,
					verbose_level - 4);
			if (f_v) {
				cout << "classify_double_sixes::upstep "
						"Aut_gens tl = ";
				Int_vec_print(cout,
						Aut_gens->tl, Aut_gens->A->base_len());
				cout << endl;
			}

			Aut_gens->group_order(ago);


			if (f_v) {
				cout << "the double six has a stabilizer of order "
						<< ago << endl;
			}
			if (f_vv) {
				cout << "The double six stabilizer is:" << endl;
				Aut_gens->print_generators_tex(cout);
			}
		}


		if (f_v) {
			cout << "classify_double_sixes::upstep double six orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " will be created" << endl;
		}

		Double_sixes->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Double_sixes,
			Flag_orbits->nb_primary_orbits_upper, 
			Aut_gens,
			dataset + 11,
			NULL /* extra_data */,
			verbose_level - 2);

		if (f_v) {
			cout << "classify_double_sixes::upstep double six orbit "
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

	Double_sixes->nb_orbits = Flag_orbits->nb_primary_orbits_upper;
	
	if (f_v) {
		cout << "We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of double sixes" << endl;
	}
	
	FREE_int(f_processed);


	if (f_v) {
		cout << "classify_double_sixes::upstep done" << endl;
	}
}


void classify_double_sixes::print_five_plus_ones(std::ostream &ost)
{
	int f, i, l;

	l = Five_p1->Five_plus_one->nb_orbits_at_level(5);

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Classification of $5+1$ Configurations "
			"in $\\PG(3," << Five_p1->q << ")$}" << endl;



	{
		ring_theory::longinteger_object go;
		Five_p1->A->Strong_gens->group_order(go);

		ost << "The order of the group is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}



	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object ol, Ol;
	Ol.create(0);

	ost << "The group has " 
		<< l 
		<< " orbits on five plus one configurations in $\\PG(3,"
		<< Five_p1->q << ").$" << endl << endl;

	ost << "Of these, " << nb << " impose 19 conditions."
			<< endl << endl;


	ost << "Of these, " << Flag_orbits->nb_flag_orbits
			<< " are associated with double sixes. "
				"They are:" << endl << endl;


	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		i = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;


		data_structures_groups::set_and_stabilizer *R;

		R = Five_p1->Five_plus_one->get_set_and_stabilizer(
				5 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		Five_p1->Five_plus_one->orbit_length(
				i /* node */,
				5 /* level */, ol);
		D.add_in_place(Ol, ol);
		
		ost << "$" << f << " / " << Flag_orbits->nb_flag_orbits
				<< "$ is orbit $" << i << " / " << l << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		FREE_OBJECT(R);
	}

	ost << "The overall number of five plus one configurations "
			"associated with double sixes in $\\PG(3," << Five_p1->q
			<< ")$ is: " << Ol << "\\\\" << endl;


	//Double_sixes->print_latex(ost, "Classification of Double Sixes");
}

void classify_double_sixes::identify_double_six(
		long int *double_six,
	int *transporter, int &orbit_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 4);
	int f, f2;
	int *Elt1;
	int *Elt2;
	long int transversal_line;
	long int five_lines[5];
	long int five_lines_out_as_neighbors[5];
	int po;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_double_sixes::identify_double_six" << endl;
	}
	Elt1 = NEW_int(Five_p1->A->elt_size_in_int);
	Elt2 = NEW_int(Five_p1->A->elt_size_in_int);
	
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six "
				"identifying the five lines a_1,...,a_5 "
				"with transversal b_6" << endl;
	}
	transversal_line = double_six[11];
	Lint_vec_copy(double_six, five_lines, 5);
	
	Five_p1->identify_five_plus_one(
			five_lines, transversal_line,
		five_lines_out_as_neighbors, po, 
		Elt1 /* transporter */,
		0 /* verbose_level */);

	if (f_vv) {
		cout << "po=" << po << endl;
		cout << "Elt1=" << endl;
		Five_p1->A->Group_element->element_print_quick(Elt1, cout);
	}

	
	if (!Sorting.int_vec_search(
			Po, Flag_orbits->nb_flag_orbits, po, f)) {
		cout << "classify_double_sixes::identify_double_six "
				"did not find po in Po" << endl;
		exit(1);
	}
	
	if (f_vv) {
		cout << "po=" << po << " f=" << f << endl;
	}

	if (Flag_orbits->Flag_orbit_node[f].f_fusion_node) {
		Five_p1->A->Group_element->element_mult(
				Elt1,
				Flag_orbits->Flag_orbit_node[f].fusion_elt,
				Elt2, 0);
		f2 = Flag_orbits->Flag_orbit_node[f].fusion_with;
		orbit_index =
				Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit;
	}
	else {
		f2 = -1;
		Five_p1->A->Group_element->element_move(Elt1, Elt2, 0);
		orbit_index = Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six "
				"f=" << f << " f2=" << f2 << " orbit_index="
				<< orbit_index << endl;
	}
	Five_p1->A->Group_element->element_move(Elt2, transporter, 0);
	if (f_vv) {
		cout << "transporter=" << endl;
		Five_p1->A->Group_element->element_print_quick(transporter, cout);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six done" << endl;
	}
}

void classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations(
		data_structures::spreadsheet *&Sp,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int nb_orbits;
	int i, k;
	int *Stab_order;
	int *Len;
	char **Transporter;
	string *Text;
	long int *rep;
	long int *lines;
	int *data;
	ring_theory::longinteger_object go;
	ring_theory::longinteger_object len;
	string fname_csv;
	char str[1000];


	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_"
				"of_fiveplusone_configurations" << endl;
	}
	snprintf(str, sizeof(str), "fiveplusone19_%d.csv", Five_p1->q);
	fname_csv.assign(str);

	k = 5;

	//nb_orbits = Five_plus_one->nb_orbits_at_level(k);
	rep = NEW_lint(k);
	lines = NEW_lint(k);
	Stab_order = NEW_int(nb);
	Len = NEW_int(nb);
	Transporter = NEW_pchar(nb);
	Text = new string [nb];
	data = NEW_int(Five_p1->A->make_element_size);

	for (i = 0; i < nb; i++) {

		Five_p1->Five_plus_one->get_set_by_level(k, Idx[i], rep);

		orbiter_kernel_system::Orbiter->Lint_vec->apply(
				rep, Five_p1->Linear_complex->Neighbor_to_line, lines, k);

		Five_p1->Five_plus_one->get_stabilizer_order(k, Idx[i], go);

		Five_p1->Five_plus_one->orbit_length(Idx[i], k, len);

		Stab_order[i] = go.as_int();

		Len[i] = len.as_int();
	}
	for (i = 0; i < nb; i++) {

		Five_p1->Five_plus_one->get_set_by_level(k, Idx[i], rep);

		orbiter_kernel_system::Orbiter->Lint_vec->apply(
				rep,
				Five_p1->Linear_complex->Neighbor_to_line,
				lines, k);

		Lint_vec_print_to_str(Text[i], lines, k);

	}

#if 0
	if (f_with_fusion) {
		for (i = 0; i < nb; i++) {
			if (Fusion[i] == -2) {
				str[0] = 0;
				strcat(str, "\"N/A\"");
			}
			else {
				A->element_code_for_make_element(transporter->ith(i), data);


				int_vec_print_to_str(str, data, A->make_element_size);

			}
			Transporter[i] = NEW_char(strlen(str) + 1);
			strcpy(Transporter[i], str);
		}
	}
#endif


	Sp = NEW_OBJECT(data_structures::spreadsheet);
#if 0
	if (f_with_fusion) {
		Sp->init_empty_table(nb + 1, 7);
	}
	else {
		Sp->init_empty_table(nb + 1, 5);
	}
#endif
	Sp->init_empty_table(nb + 1, 5);
	Sp->fill_column_with_row_index(0, "Orbit");
	Sp->fill_column_with_int(1, Idx, "Idx");
	Sp->fill_column_with_text(2, Text, "Rep");
	Sp->fill_column_with_int(3, Stab_order, "Stab_order");
	Sp->fill_column_with_int(4, Len, "Orbit_length");
#if 0
	if (f_with_fusion) {
		Sp->fill_column_with_int(5, Fusion, "Fusion");
		Sp->fill_column_with_text(6,
				(const char **) Transporter, "Transporter");
	}
#endif
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;

	FREE_lint(rep);
	FREE_lint(lines);
	FREE_int(Stab_order);
	FREE_int(Len);
	delete [] Text;
	for (i = 0; i < nb; i++) {
		FREE_char(Transporter[i]);
	}
	FREE_pchar(Transporter);
	FREE_int(data);
	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations "
				"done" << endl;
	}
}

void classify_double_sixes::write_file(
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classify_double_sixes::write_file" << endl;
	}
	fp.write((char *) &Five_p1->q, sizeof(int));
	fp.write((char *) &Five_p1->Linear_complex->nb_neighbors, sizeof(int));
	fp.write((char *) &len, sizeof(int));
	fp.write((char *) &nb, sizeof(int));
	fp.write((char *) &Flag_orbits->nb_flag_orbits, sizeof(int));

	for (i = 0; i < nb; i++) {
		fp.write((char *) &Idx[i], sizeof(int));
	}
	for (i = 0; i < Flag_orbits->nb_flag_orbits; i++) {
		fp.write((char *) &Po[i], sizeof(int));
	}


	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"before Five_plus_one->write_file" << endl;
	}
	Five_p1->Five_plus_one->write_file(fp,
			5 /* depth_completed */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"after Five_plus_one->write_file" << endl;
	}


	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"before Flag_orbits->write_file" << endl;
	}
	Flag_orbits->write_file(fp, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"after Flag_orbits->write_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"before Double_sixes->write_file" << endl;
	}
	Double_sixes->write_file(fp, 0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::write_file "
				"after Double_sixes->write_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::write_file finished" << endl;
	}
}

void classify_double_sixes::read_file(
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_flag_orbits;
	
	if (f_v) {
		cout << "classify_double_sixes::read_file" << endl;
	}
	fp.read((char *) &Five_p1->q, sizeof(int));
	fp.read((char *) &Five_p1->Linear_complex->nb_neighbors, sizeof(int));
	fp.read((char *) &len, sizeof(int));
	fp.read((char *) &nb, sizeof(int));
	fp.read((char *) &nb_flag_orbits, sizeof(int));

	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"q=" << Five_p1->q << endl;
		cout << "classify_double_sixes::read_file "
				"nb_neighbors=" << Five_p1->Linear_complex->nb_neighbors << endl;
		cout << "classify_double_sixes::read_file "
				"len=" << len << endl;
		cout << "classify_double_sixes::read_file "
				"nb=" << nb << endl;
		cout << "classify_double_sixes::read_file "
				"nb_flag_orbits=" << nb_flag_orbits << endl;
	}

	Idx = NEW_int(nb);
	for (i = 0; i < nb; i++) {
		fp.read((char *) &Idx[i], sizeof(int));
	}

	Po = NEW_int(nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {
		fp.read((char *) &Po[i], sizeof(int));
	}


	int depth_completed;

	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"before Five_plus_one->read_file" << endl;
	}
	Five_p1->Five_plus_one->read_file(
			fp, depth_completed,
			verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"after Five_plus_one->read_file" << endl;
	}
	if (depth_completed != 5) {
		cout << "classify_double_sixes::read_file "
				"depth_completed != 5" << endl;
		exit(1);
	}


	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	//Flag_orbits->A = A;
	//Flag_orbits->A2 = A;
	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"before Flag_orbits->read_file" << endl;
	}
	Flag_orbits->read_file(
			fp, Five_p1->A, Five_p1->A2,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"after Flag_orbits->read_file" << endl;
	}

	Double_sixes = NEW_OBJECT(invariant_relations::classification_step);
	//Double_sixes->A = A;
	//Double_sixes->A2 = A2;

	ring_theory::longinteger_object go;
	Five_p1->A->group_order(go);
	//A->group_order(Double_sixes->go);

	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"before Double_sixes->read_file" << endl;
	}
	Double_sixes->read_file(
			fp, Five_p1->A, Five_p1->A2, go,
			0/*verbose_level*/);
	if (f_v) {
		cout << "classify_double_sixes::read_file "
				"after Double_sixes->read_file" << endl;
	}

	if (f_v) {
		cout << "classify_double_sixes::read_file finished" << endl;
	}
}







}}}}




