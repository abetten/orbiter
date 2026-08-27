/*
 * orbits_on_polynomials.cpp
 *
 *  Created on: Nov 28, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


orbits_on_polynomials::orbits_on_polynomials()
{
	Record_birth();
	LG = NULL;
	degree_of_poly = 0;

	F = NULL;
	A = NULL;
	n = 0;
	// go;
	HPD = NULL;

	A2 = NULL;

	Elt1 = Elt2 = Elt3 = NULL;

	f_has_small_generating_set = false;
	generating_set_small = NULL;


	f_has_Sch = false;
	Sch = NULL;
	// full_go

	f_has_Orb = false;
	Orb = NULL;

	//fname_base
	//fname_csv
	//fname_report
	T = NULL;
	Nb_pts = NULL;

}

orbits_on_polynomials::~orbits_on_polynomials()
{
	Record_death();
	if (A2) {
		FREE_OBJECT(A2);
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
	if (f_has_small_generating_set) {
		FREE_OBJECT(generating_set_small);
		f_has_small_generating_set = false;
	}
	if (f_has_Sch) {
		FREE_OBJECT(Sch);
		f_has_Sch = false;
	}
	if (f_has_Orb) {
		FREE_OBJECT(Orb);
		f_has_Orb = false;
	}
	if (T) {
		FREE_OBJECT(T);
	}
}

void orbits_on_polynomials::init_Schreier(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}

	A->Strong_gens->group_order(target_go);


	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);



	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before compute_small_generating_set" << endl;
	}
	compute_small_generating_set(verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after compute_small_generating_set" << endl;
	}


	actions::action_global Action_global;

	Sch = NEW_OBJECT(groups::schreier);

	f_has_Sch = true;



	string fname_bitvector;

	fname_bitvector = fname_csv;

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before Action_global.all_point_orbits_Schreier_from_generators_first_next" << endl;
	}

	Action_global.all_point_orbits_Schreier_from_generators_first_next(
			A2,
			*Sch,
			generating_set_small,
			target_go,
			fname_bitvector,
			verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after Action_global.all_point_orbits_Schreier_from_generators_first_next" << endl;
	}



#if 0
	Action_global.all_point_orbits_from_strong_generators(
			A,
			*Sch,
			A->Strong_gens,
			verbose_level - 2);
#endif


#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}
	Sch = A->Strong_gens->compute_all_point_orbits_schreier(
			A2, print_interval, verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}
#endif



	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before Sch->write_orbit_summary" << endl;
	}
	Sch->write_orbit_summary(
			fname_csv,
			A /*default_action*/,
			go,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after Sch->write_orbit_summary" << endl;
	}



	A->group_order(full_go);
	T = NEW_OBJECT(data_structures_groups::orbit_transversal);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before T->init_from_schreier" << endl;
	}

	T->init_from_schreier(
			Sch,
			A,
			full_go,
			verbose_level);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after T->init_from_schreier" << endl;
	}



	Sch->Forest->print_orbit_reps(cout);


	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"before compute_points" << endl;
	}
	compute_points(verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier "
				"after compute_points" << endl;
	}



	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier done" << endl;
	}
}



void orbits_on_polynomials::init_Schreier_with_generators(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		data_structures_groups::vector_ge *generating_set,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;


	f_has_small_generating_set = true;
	generating_set_small = generating_set->duplicate(0 /* verbose_level */);


	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}

	A->Strong_gens->group_order(target_go);

	target_go_log2 = target_go.log2();
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators target_go_log2 = " << target_go_log2 << endl;
	}


	if (target_go_log2 > 31) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators target_go_log2 > 31" << endl;
		cout << "orbits_on_polynomials::init_Schreier_with_generators target_go_log2 = " << target_go_log2 << endl;
		exit(1);
	}

	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);

#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before compute_small_generating_set" << endl;
	}
	compute_small_generating_set(verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after compute_small_generating_set" << endl;
	}
#endif


	actions::action_global Action_global;

	Sch = NEW_OBJECT(groups::schreier);

	f_has_Sch = true;



	string fname_bitvector;

	fname_bitvector = fname_csv;

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before Action_global.all_point_orbits_Schreier_from_generators_first_next" << endl;
	}

	Action_global.all_point_orbits_Schreier_from_generators_first_next(
			A2,
			*Sch,
			generating_set_small,
			target_go,
			fname_bitvector,
			verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after Action_global.all_point_orbits_Schreier_from_generators_first_next" << endl;
	}



#if 0
	Action_global.all_point_orbits_from_strong_generators(
			A,
			*Sch,
			A->Strong_gens,
			verbose_level - 2);
#endif


#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}
	Sch = A->Strong_gens->compute_all_point_orbits_schreier(
			A2, print_interval, verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}
#endif



	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before Sch->write_orbit_summary" << endl;
	}
	Sch->write_orbit_summary(
			fname_csv,
			A /*default_action*/,
			go,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after Sch->write_orbit_summary" << endl;
	}



	A->group_order(full_go);
	T = NEW_OBJECT(data_structures_groups::orbit_transversal);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before T->init_from_schreier" << endl;
	}

	T->init_from_schreier(
			Sch,
			A,
			full_go,
			verbose_level);

	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after T->init_from_schreier" << endl;
	}



	Sch->Forest->print_orbit_reps(cout);


	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"before compute_points" << endl;
	}
	compute_points(verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators "
				"after compute_points" << endl;
	}



	if (f_v) {
		cout << "orbits_on_polynomials::init_Schreier_with_generators done" << endl;
	}
}




void orbits_on_polynomials::init_memory_efficient(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		data_structures_groups::vector_ge *generating_set,
		int print_interval,
		int verbose_level)
// Memory-efficient orbit computation using bitvector (~45 MB)
// instead of Schreier forest (~10 GB).
// Uses on-the-fly image computation (no caching).
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;


	f_has_small_generating_set = true;
	generating_set_small = generating_set->duplicate(0 /* verbose_level */);


	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}

	A->Strong_gens->group_order(target_go);

	target_go_log2 = target_go.log2();
	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient target_go_log2 = " << target_go_log2 << endl;
	}


	if (target_go_log2 > 31) {
		cout << "orbits_on_polynomials::init_memory_efficient target_go_log2 > 31" << endl;
		cout << "orbits_on_polynomials::init_memory_efficient target_go_log2 = " << target_go_log2 << endl;
		exit(1);
	}

	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"degree_of_poly = " << degree_of_poly << endl;
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


	// =========================================================
	// Memory-efficient BFS using bitvector
	// =========================================================

	long int degree = A2->degree;

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"degree = " << degree << endl;
		cout << "orbits_on_polynomials::init_memory_efficient "
				"bitvector size = " << (degree / 8 / 1024 / 1024) << " MB" << endl;
	}

	// Allocate bitvector for tracking visited points
	other::data_structures::bitvector *visited;
	visited = NEW_OBJECT(other::data_structures::bitvector);
	visited->allocate(degree);
	visited->zero();

	// Orbit data
	std::vector<long int> orbit_reps;
	std::vector<long int> orbit_lengths;

	long int nb_orbits_found = 0;
	long int total_covered = 0;
	int nb_gens = generating_set_small->len;

	other::orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"starting BFS with " << nb_gens << " generators" << endl;
	}

	for (long int pt = 0; pt < degree; pt++) {

		if (visited->s_i(pt)) {
			continue;
		}

		// New orbit starting at pt
		std::vector<long int> queue;
		queue.push_back(pt);
		visited->set_bit(pt);
		long int front = 0;

		while (front < (long int)queue.size()) {
			long int cur_pt = queue[front];
			front++;

			for (int g = 0; g < nb_gens; g++) {
				// On-the-fly image computation (no caching)
				long int next_pt = A2->Group_element->element_image_of(
					cur_pt, generating_set_small->ith(g), 0);

				if (!visited->s_i(next_pt)) {
					visited->set_bit(next_pt);
					queue.push_back(next_pt);
				}
			}
		}

		long int orbit_len = (long int)queue.size();
		orbit_reps.push_back(pt);
		orbit_lengths.push_back(orbit_len);
		nb_orbits_found++;
		total_covered += orbit_len;

		if ((nb_orbits_found % print_interval == 0) || orbit_len > 10000) {
			int t1 = Os.os_ticks();
			int elapsed = t1 - t0;
			cout << "orbits_on_polynomials::init_memory_efficient "
					"orbit " << nb_orbits_found
					<< " rep=" << pt
					<< " length=" << orbit_len
					<< " covered=" << total_covered
					<< "/" << degree
					<< " (" << (100.0 * total_covered / degree) << "%)"
					<< " elapsed=" << (elapsed / 1000) << "s"
					<< endl;
		}
	}

	int t1 = Os.os_ticks();
	int elapsed = t1 - t0;

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"found " << nb_orbits_found << " orbits" << endl;
		cout << "orbits_on_polynomials::init_memory_efficient "
				"total covered = " << total_covered << " / " << degree << endl;
		cout << "orbits_on_polynomials::init_memory_efficient "
				"elapsed time = " << (elapsed / 1000) << " seconds" << endl;
	}

	// Free bitvector
	FREE_OBJECT(visited);


	// =========================================================
	// Phase 2: Compute stabilizer generators using
	// orbit_of_equations for each orbit representative
	// =========================================================

	A->group_order(full_go);

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"Phase 2: computing stabilizer generators "
				"for " << nb_orbits_found << " orbits" << endl;
	}

	T = NEW_OBJECT(data_structures_groups::orbit_transversal);
	T->A = A;
	T->A2 = A2;
	T->nb_orbits = (int)nb_orbits_found;
	T->Reps = NEW_OBJECTS(data_structures_groups::set_and_stabilizer, T->nb_orbits);

	int *coeff;
	coeff = NEW_int(HPD->get_nb_monomials());

	int t2 = Os.os_ticks();

	long int full_go_lint = full_go.as_lint();

	for (int i = 0; i < (int)nb_orbits_found; i++) {

		// Build the orbit transversal entry
		long int *Set = NEW_lint(1);
		Set[0] = orbit_reps[i];

		groups::strong_generators *stab_gens;

		if (orbit_lengths[i] == full_go_lint) {
			// Trivial stabilizer: |Orbit| == |G| → |Stab| = 1
			// Skip orbit_of_equations entirely — just use identity
			stab_gens = NEW_OBJECT(groups::strong_generators);
			stab_gens->init(A);
			stab_gens->init_trivial_group(A, 0 /* verbose_level */);

		}
		else {
			// Non-trivial stabilizer: compute via orbit_of_equations
			HPD->unrank_coeff_vector(coeff, orbit_reps[i]);

			orbits_schreier::orbit_of_equations *OoE;
			OoE = NEW_OBJECT(orbits_schreier::orbit_of_equations);

			OoE->init(
					A, F,
					A2->G.OnHP,
					LG->Strong_gens,
					coeff,
					0 /* verbose_level */);

			stab_gens = OoE->stabilizer_orbit_rep(
					full_go,
					0 /* verbose_level */);

			FREE_OBJECT(OoE);
		}

		T->Reps[i].init_everything(
			A, A2,
			Set, 1 /* set_sz */,
			stab_gens, 0 /* verbose_level */);

		if ((i + 1) % 100 == 0 || i == (int)nb_orbits_found - 1) {
			int t3 = Os.os_ticks();
			int elapsed2 = t3 - t2;
			cout << "orbits_on_polynomials::init_memory_efficient "
					"Phase 2: orbit " << (i + 1)
					<< "/" << nb_orbits_found
					<< " elapsed=" << (elapsed2 / 1000) << "s"
					<< endl;
		}
	}

	FREE_int(coeff);

	int t_final = Os.os_ticks();
	int total_elapsed = t_final - t0;

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"Phase 2 done" << endl;
		cout << "orbits_on_polynomials::init_memory_efficient "
				"total elapsed time = " << (total_elapsed / 1000) << " seconds" << endl;
	}


	// Print orbit representatives
	cout << "Orbit representatives:" << endl;
	for (int i = 0; i < (int)nb_orbits_found; i++) {

		algebra::ring_theory::longinteger_object go_stab;
		T->Reps[i].Strong_gens->group_order(go_stab);

		cout << i << " : rep=" << orbit_reps[i]
			<< " len=" << orbit_lengths[i]
			<< " |Stab|=" << go_stab << endl;
	}


	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"before compute_points" << endl;
	}
	compute_points(verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"after compute_points" << endl;
	}


	// Write CSV summary
	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient "
				"writing CSV to " << fname_csv << endl;
	}
	{
		std::ofstream ost(fname_csv);
		ost << "OrbitIdx,Rep,OrbitLength,StabOrder" << endl;

		for (int i = 0; i < (int)nb_orbits_found; i++) {

			algebra::ring_theory::longinteger_object stab_order;
			T->Reps[i].Strong_gens->group_order(stab_order);

			ost << i << "," << orbit_reps[i] << ","
				<< orbit_lengths[i] << "," << stab_order << endl;
		}
	}

	// Mark as having a valid orbit transversal (T) so that
	// export functions (report, representatives_detailed) work
	f_has_Sch = true;

	if (f_v) {
		cout << "orbits_on_polynomials::init_memory_efficient done" << endl;
	}
}



void orbits_on_polynomials::init_bitvector_first(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		data_structures_groups::vector_ge *generating_set,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}

	f_has_small_generating_set = true;
	generating_set_small = generating_set->duplicate(0 /* verbose_level */);

	A->Strong_gens->group_order(target_go);

	target_go_log2 = target_go.log2();
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first target_go_log2 = " << target_go_log2 << endl;
	}


	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first "
				"before compute_small_generating_set" << endl;
	}
	compute_small_generating_set(verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first "
				"after compute_small_generating_set" << endl;
	}
#endif




	long int *set0;
	int set_sz = 1;

	set0 = NEW_lint(1);
	set0[0] = 0;




	other::orbiter_kernel_system::file_io Fio;
	orbits_schreier::orbit_of_sets *OrbOnSets;

	OrbOnSets = NEW_OBJECT(orbits_schreier::orbit_of_sets);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first "
				"before OrbOnSets->init" << endl;
	}
	OrbOnSets->init(
			A,
			A2,
			set0, set_sz,
			generating_set_small,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first "
				"after OrbOnSets->init" << endl;
	}


	int nb_rows = 1;
	int nb_cols = 3;
	std::string *Table;

	Table = new string[nb_rows * nb_cols];

	Table[0 * nb_cols + 0] = std::to_string(0);
	Table[0 * nb_cols + 1] = std::to_string(set0[0]);
	Table[0 * nb_cols + 2] = std::to_string(OrbOnSets->used_length);

	std::string fname_reps;
	string headings;

	headings = "Row,Rep,Length";

	fname_reps = fname_base  + "_orb" + std::to_string(0) + "_reps.csv";
	Fio.Csv_file_support->write_table_of_strings(
			fname_reps,
			nb_rows, nb_cols, Table,
			headings,
			verbose_level);



	{
		other::data_structures::bitvector *B;
		std::string fname;

		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_first "
					"before OrbOnSets->compute_bitvector" << endl;
		}
		B = OrbOnSets->compute_bitvector(
				verbose_level - 1);
		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_first "
					"after OrbOnSets->compute_bitvector" << endl;
		}



		fname = fname_base + "_orb" + std::to_string(0) + ".bitvector";
		B->write_file(fname, verbose_level - 1);

		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_first Written file "
					<< fname << " of size " << Fio.file_size(fname) << endl;
		}

		FREE_OBJECT(B);
	}



	FREE_lint(set0);
	FREE_OBJECT(OrbOnSets);
	delete [] Table;

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_first done" << endl;
	}
}




int orbits_on_polynomials::init_bitvector_continue(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		data_structures_groups::vector_ge *generating_set,
		int print_interval,
		int idx_of_last_orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue" << endl;
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"idx_of_last_orbit = " << idx_of_last_orbit << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue n = " << n << endl;
	}

	A->Strong_gens->group_order(target_go);

	target_go_log2 = target_go.log2();
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue target_go_log2 = " << target_go_log2 << endl;
	}


	f_has_small_generating_set = true;
	generating_set_small = generating_set->duplicate(0 /* verbose_level */);



	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"before compute_small_generating_set" << endl;
	}
	compute_small_generating_set(verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"after compute_small_generating_set" << endl;
	}
#endif


	while (true) {
		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_continue "
					"idx_of_last_orbit = " << idx_of_last_orbit << endl;
		}
		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_continue "
					"before complete_orbits" << endl;
		}
		if (complete_orbits(idx_of_last_orbit, verbose_level)) {
			cout << "orbits_on_polynomials::init_bitvector_continue "
					"the orbits have been computed completely" << endl;
			break;
		}
		if (f_v) {
			cout << "orbits_on_polynomials::init_bitvector_continue "
					"after complete_orbits" << endl;
		}
	}

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue done" << endl;
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"idx_of_last_orbit = " << idx_of_last_orbit << endl;
	}
	return true;
}


int orbits_on_polynomials::complete_orbits(
		int &idx_of_last_orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue" << endl;
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"idx_of_last_orbit = " << idx_of_last_orbit << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	std::string fname_reps_old;


	fname_reps_old = fname_base  + "_orb" + std::to_string(idx_of_last_orbit) + "_reps.csv";

	std::string *col_label;
	std::string *Table_old;
	int nb_rows_old, nb_cols_old;

	Fio.Csv_file_support->read_table_of_strings(
			fname_reps_old, col_label,
			Table_old, nb_rows_old, nb_cols_old,
			verbose_level);


	if (nb_cols_old != 3) {
		cout << "orbits_on_polynomials::init_bitvector_continue nb_cols_old != 3" << endl;
		exit(1);
	}

	string fname_bitvector_old;

	fname_bitvector_old = fname_base + "_orb" + std::to_string(idx_of_last_orbit) + ".bitvector";

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue reading file "
				<< fname_bitvector_old << " of size " << Fio.file_size(fname_bitvector_old) << endl;
	}



	other::data_structures::bitvector *B;
	//long int length;

	B = NEW_OBJECT(other::data_structures::bitvector);

	B->read_file(fname_bitvector_old, verbose_level);

	//length = B->get_length();




	long int cur_eqn_idx;

	cur_eqn_idx = B->get_first_entry_zero();

	if (cur_eqn_idx == -1) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"orbit computation is finished" << endl;

		FREE_OBJECT(B);
		delete [] Table_old;
		return true;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"orbit " << idx_of_last_orbit << " has new orbit representative = " << cur_eqn_idx << endl;
	}


	long int *set0;
	int set_sz = 1;

	set0 = NEW_lint(1);
	set0[0] = cur_eqn_idx;




	orbits_schreier::orbit_of_sets *OrbOnSets;

	OrbOnSets = NEW_OBJECT(orbits_schreier::orbit_of_sets);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"before OrbOnSets->init" << endl;
	}
	OrbOnSets->init(
			A,
			A2,
			set0, set_sz,
			generating_set_small,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"after OrbOnSets->init" << endl;
	}


	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"before OrbOnSets->add_to_existing_bitvector" << endl;
	}
	OrbOnSets->add_to_existing_bitvector(
			B,
			verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue "
				"after OrbOnSets->add_to_existing_bitvector" << endl;
	}


	int nb_rows = nb_rows_old + 1;
	int nb_cols = 3;
	std::string *Table;

	Table = new string[nb_rows * nb_cols];

	int i, j;

	for (i = 0; i < nb_rows_old; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = Table_old[i * nb_cols + j];
		}
	}

	idx_of_last_orbit++;

	Table[nb_rows_old * nb_cols + 0] = std::to_string(idx_of_last_orbit);
	Table[nb_rows_old * nb_cols + 1] = std::to_string(set0[0]);
	Table[nb_rows_old * nb_cols + 2] = std::to_string(OrbOnSets->used_length);

	std::string fname_reps;
	string headings;

	headings = "Row,Rep,Length";

	fname_reps = fname_base  + "_orb" + std::to_string(idx_of_last_orbit) + "_reps.csv";
	Fio.Csv_file_support->write_table_of_strings(
			fname_reps,
			nb_rows, nb_cols, Table,
			headings,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue Written file "
				<< fname_reps << " of size " << Fio.file_size(fname_reps) << endl;
	}

	string fname_bitvector_new;


	fname_bitvector_new = fname_base + "_orb" + std::to_string(idx_of_last_orbit) + ".bitvector";
	B->write_file(fname_bitvector_new, verbose_level - 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue Written file "
				<< fname_bitvector_new << " of size " << Fio.file_size(fname_bitvector_new) << endl;
	}

	FREE_OBJECT(B);


	string cmd1, cmd2;

	cmd1 = "rm " + fname_reps_old;
	cmd2 = "rm " + fname_bitvector_old;

	system(cmd1.c_str());
	system(cmd2.c_str());

	FREE_lint(set0);
	FREE_OBJECT(OrbOnSets);
	delete [] Table_old;
	delete [] Table;

	if (f_v) {
		cout << "orbits_on_polynomials::init_bitvector_continue done" << endl;
	}
	return false;

}

void orbits_on_polynomials::orbit_of_one_polynomial(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		algebra::expression_parser::symbolic_object_builder *Symbol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}


	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";




	if (Symbol->Formula_vector->len != 1) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial len != 1" << endl;
		exit(1);
	}


	int *eqn;
	int eqn_sz;

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"before get_multipoly" << endl;
	}
	Symbol->Formula_vector->V[0].get_multipoly(HPD,
			eqn, eqn_sz, verbose_level - 1);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"after get_multipoly" << endl;
	}
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"eqn = ";
		Int_vec_print(cout, eqn, eqn_sz);
		cout << endl;
	}




	// compute the orbit of the equation under the stabilizer of the set of points:


	f_has_Orb = true;

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"before Orb->init" << endl;
	}
	Orb->init(
			A, F,
			A2->G.OnHP,
		LG->Strong_gens /* A->Strong_gens*/, eqn,
		verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"after Orb->init" << endl;
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"found an orbit of length " << Orb->used_length << endl;
	}

	// who frees eqn?


	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial done" << endl;
	}
}


void orbits_on_polynomials::compute_points(
		int verbose_level)
// Points is a vector of vectors containing the orbits (as sets) one-by-one
{
	int *coeff;
	int i;

	coeff = NEW_int(HPD->get_nb_monomials());
	Nb_pts = NEW_int(T->nb_orbits);


	for (i = 0; i < T->nb_orbits; i++) {

		algebra::ring_theory::longinteger_object go;
		T->Reps[i].Strong_gens->group_order(go);

		cout << i << " : ";
		Lint_vec_print(cout, T->Reps[i].data, T->Reps[i].sz);
		cout << " : ";
		cout << go;

		cout << " : ";

		HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

		std::vector<long int> Pts;

		HPD->enumerate_points(coeff, Pts, verbose_level);

		Points.push_back(Pts);
		Nb_pts[i] = Pts.size();
	}
	FREE_int(coeff);

}


#if 0
void orbits_on_polynomials::compute_lines(
		int verbose_level)
// Points is a vector of vectors containing the orbits (as sets) one-by-one
{
	int *coeff;
	int i;

	coeff = NEW_int(HPD->get_nb_monomials());
	Nb_lines = NEW_int(T->nb_orbits);


	for (i = 0; i < T->nb_orbits; i++) {

		algebra::ring_theory::longinteger_object go;
		T->Reps[i].Strong_gens->group_order(go);

		cout << i << " : ";
		Lint_vec_print(cout, T->Reps[i].data, T->Reps[i].sz);
		cout << " : ";
		cout << go;

		cout << " : ";

		HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

		std::vector<long int> Pts;

		HPD->enumerate_points(coeff, Pts, verbose_level);

		Points.push_back(Pts);
		Nb_pts[i] = Pts.size();
	}
	FREE_int(coeff);

}

#endif



void orbits_on_polynomials::report(
		int verbose_level)
// used to create a projective_geometry::projective_space_with_action
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report" << endl;
	}
	cout << "orbit reps:" << endl;

	string title, author, extra_praeamble;

	fname_report = "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)+ "_q" + std::to_string(F->q) + ".tex";

	title = "Varieties of degree " + std::to_string(degree_of_poly)
			+ " in PG(" + std::to_string(n - 1)+ "," + std::to_string(F->q) + ")";

	author.assign("Orbiter");

	{
		ofstream ost(fname_report);
		other::l1_interfaces::latex_interface L;

		L.head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);

		ost << "\\small" << endl;
		ost << "\\arraycolsep=2pt" << endl;
		ost << "\\parindent=0pt" << endl;
		ost << "$q = " << F->q << "$\\\\" << endl;
		ost << "$n = " << n - 1 << "$\\\\" << endl;
		ost << "degree of poly $ = " << degree_of_poly << "$\\\\" << endl;

		ost << "\\clearpage" << endl << endl;


		// summary table:

		ost << "\\section*{The Varieties of degree $" << degree_of_poly
				<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, summary}" << endl;

#if 0
		T->print_table_latex(
				f,
				true /* f_has_callback */,
				polynomial_orbits_callback_print_function2,
				HPD /* callback_data */,
				true /* f_has_callback */,
				polynomial_orbits_callback_print_function,
				HPD /* callback_data */,
				verbose_level);
#else
		int *coeff;
		int i;

		coeff = NEW_int(HPD->get_nb_monomials());
		//Nb_pts = NEW_int(T->nb_orbits);


#if 0
		// compute the group of the surface:
		projective_geometry::projective_space_with_action *PA;
		int f_semilinear;
		number_theory::number_theory_domain NT;

		if (NT.is_prime(F->q)) {
			f_semilinear = false;
		}
		else {
			f_semilinear = true;
		}

		PA = NEW_OBJECT(projective_geometry::projective_space_with_action);

		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties before PA->init" << endl;
		}
		PA->init(
			F, n - 1 /*n*/, f_semilinear,
			true /* f_init_incidence_structure */,
			verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties after PA->init" << endl;
		}
#endif






		other::data_structures::tally T_nb_pts;
		int h, j, f, l, a;

		T_nb_pts.init(Nb_pts, T->nb_orbits, false, 0);

		for (h = T_nb_pts.nb_types - 1; h >= 0; h--) {

			f = T_nb_pts.type_first[h];
			l = T_nb_pts.type_len[h];
			a = T_nb_pts.data_sorted[f];

			ost << "\\subsection*{Objects with " << a << " Points}" << endl;

			ost << "There are " << l << " objects with " << a << " Points: \\\\" << endl;

			int *Idx;
			int len;

			T_nb_pts.get_class_by_value(Idx, len, a, 0 /*verbose_level*/);


			other::data_structures::sorting Sorting;

			Sorting.int_vec_heapsort(Idx, l);

			ost << "orbit : rep : go : poly : Pts\\\\" << endl;
			for (j = 0; j < l; j++) {

				//i = T_nb_pts.sorting_perm_inv[f + j];

				i = Idx[j];

				algebra::ring_theory::longinteger_object go;
				T->Reps[i].Strong_gens->group_order(go);

				// 1
				ost << i << " : ";

				// 2
				Lint_vec_print(ost, T->Reps[i].data, T->Reps[i].sz);
				ost << " : ";

				// 3
				ost << go;

				ost << " : ";

				HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

				//int nb_pts;

				//nb_pts = Nb_pts[i];

				//ost << nb_pts;
				//ost << " : ";


				// 4
				ost << T->Reps[i].data[0] << "=$";
				HPD->print_equation_tex(ost, coeff);
				//int_vec_print(f, coeff, HPD->get_nb_monomials());
				//cout << " = ";
				//HPD->print_equation_str(ost, coeff);

				//f << " & ";
				//Reps[i].Strong_gens->print_generators_tex(f);
				ost << "$";

				ost << " : ";

				//int u;
				//long int *set;
				//groups::strong_generators *Sg;

#if 0
				set = NEW_lint(nb_pts);
				for (u = 0; u < nb_pts; u++) {
					set[u] = Points[i][u];
				}
#endif

				// 5
				string s_points;

				s_points = Lint_vec_stl_stringify(Points[i]);

				ost << "\"" + s_points + "\"";


#if 0
				PA->compute_group_of_set(set, nb_pts,
						Sg,
						verbose_level);

				ost << " : go=";
				ring_theory::longinteger_object go1;
				Sg->group_order(go1);
				ost << go1;
#endif

				ost << "\\\\" << endl;

				//FREE_lint(set);
			}

			FREE_int(Idx);

		}
		//FREE_OBJECT(PA);

#endif

		FREE_int(coeff);



		L.foot(ost);

	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_report << " of size " << Fio.file_size(fname_report) << endl;
	if (f_v) {
		cout << "orbits_on_polynomials::report done" << endl;
	}

}


void orbits_on_polynomials::prepare_data(
		std::string &headings,
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbits_on_polynomials::prepare_data" << endl;
		cout << "orbits_on_polynomials::prepare_data verbose_level=" << verbose_level << endl;
	}
	int orbit_idx;

	nb_rows = T->nb_orbits;

	create_heading(headings, nb_cols);

	Table = new string[nb_rows * nb_cols];


	for (orbit_idx = 0; orbit_idx < nb_rows; orbit_idx++) {

		if (f_vv) {
			cout << "orbits_on_polynomials::prepare_data counter = " << orbit_idx
					<< " / " << nb_rows << endl;
		}


		std::vector<std::string> v;

		create_vector_of_strings(
				orbit_idx,
				v,
				verbose_level - 1);


		int j;

		for (j = 0; j < nb_cols; j++) {
			Table[orbit_idx * nb_cols + j] = v[j];
		}

	}
	if (f_v) {
		cout << "orbits_on_polynomials::prepare_data done" << endl;
	}
}



void orbits_on_polynomials::prepare_data_detailed(
		geometry::projective_geometry::projective_space *P,
		std::string &headings,
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbits_on_polynomials::prepare_data_detailed" << endl;
		cout << "orbits_on_polynomials::prepare_data_detailed verbose_level=" << verbose_level << endl;
	}
	int orbit_idx;

	nb_rows = T->nb_orbits;

	create_heading_detailed(headings, nb_cols);

	Table = new string[nb_rows * nb_cols];


	for (orbit_idx = 0; orbit_idx < nb_rows; orbit_idx++) {

		if (f_vv) {
			cout << "orbits_on_polynomials::prepare_data_detailed counter = " << orbit_idx
					<< " / " << nb_rows << endl;
		}


		std::vector<std::string> v;

		create_vector_of_strings_detailed(
				orbit_idx,
				P,
				v,
				verbose_level - 1);


		int j;

		for (j = 0; j < nb_cols; j++) {
			Table[orbit_idx * nb_cols + j] = v[j];
		}

	}
	if (f_v) {
		cout << "orbits_on_polynomials::prepare_data_detailed done" << endl;
	}
}





void orbits_on_polynomials::create_heading(
		std::string &heading, int &nb_cols)
{
	heading = "OrbIdx,Go,EqnCode,EqnVec,EqnAf,NbPts,Pts";
	nb_cols = 7;

}

void orbits_on_polynomials::create_heading_detailed(
		std::string &heading, int &nb_cols)
{
	heading = "OrbIdx,Go,EqnCode,EqnVec,EqnAf,NbPts,Pts,NbLines,Lines,NbSingularPts,SingPts";
	nb_cols = 11;

}

void orbits_on_polynomials::create_vector_of_strings(
		int orbit_idx,
		std::vector<std::string> &v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::create_vector_of_strings" << endl;
	}

	int *coeff;

	coeff = NEW_int(HPD->get_nb_monomials());


	int nb_cols;

	nb_cols = 7;

	v.resize(nb_cols);

	v[0] = std::to_string(orbit_idx);

	algebra::ring_theory::longinteger_object go;
	T->Reps[orbit_idx].Strong_gens->group_order(go);

	v[1] = go.stringify();

	v[2] = std::to_string(T->Reps[orbit_idx].data[0]);
	//v[2] = Lint_vec_stringify(T->Reps[orbit_idx].data, T->Reps[orbit_idx].sz);

	HPD->unrank_coeff_vector(coeff, T->Reps[orbit_idx].data[0]);

	v[3] = "\"" + Int_vec_stringify(coeff, HPD->get_nb_monomials()) + "\"";


	string s_eqn_af;

	s_eqn_af = HPD->stringify_equation(coeff, verbose_level);

	v[4] = "\"" + s_eqn_af + "\"";


	v[5] = std::to_string(Points[orbit_idx].size());

	string s_points;

	s_points = Lint_vec_stl_stringify(Points[orbit_idx]);

	v[6] = "\"" + s_points + "\"";



	FREE_int(coeff);


	if (f_v) {
		cout << "orbits_on_polynomials::create_vector_of_strings done" << endl;
	}
}



void orbits_on_polynomials::create_vector_of_strings_detailed(
		int orbit_idx,
		geometry::projective_geometry::projective_space *P,
		std::vector<std::string> &v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::create_vector_of_strings_detailed" << endl;
	}

	int *coeff;

	coeff = NEW_int(HPD->get_nb_monomials());


	int nb_cols;

	nb_cols = 11;

	v.resize(nb_cols);

	v[0] = std::to_string(orbit_idx);

	algebra::ring_theory::longinteger_object go;
	T->Reps[orbit_idx].Strong_gens->group_order(go);

	v[1] = go.stringify();

	v[2] = std::to_string(T->Reps[orbit_idx].data[0]);
	//v[2] = Lint_vec_stringify(T->Reps[orbit_idx].data, T->Reps[orbit_idx].sz);

	HPD->unrank_coeff_vector(coeff, T->Reps[orbit_idx].data[0]);

	v[3] = "\"" + Int_vec_stringify(coeff, HPD->get_nb_monomials()) + "\"";


	string s_eqn_af;

	s_eqn_af = HPD->stringify_equation(coeff, verbose_level);

	v[4] = "\"" + s_eqn_af + "\"";


	v[5] = std::to_string(Points[orbit_idx].size());

	string s_points;

	s_points = Lint_vec_stl_stringify(Points[orbit_idx]);

	v[6] = "\"" + s_points + "\"";



	{

		geometry::algebraic_geometry::variety_description *Descr;
		geometry::algebraic_geometry::variety_object *V;


		Descr = NEW_OBJECT(geometry::algebraic_geometry::variety_description);

#if 0
		int f_label_txt;
		std::string label_txt;

		int f_label_tex;
		std::string label_tex;

		int f_projective_space;
		std::string projective_space_label;

		// not to be documented:
		int f_projective_space_pointer;
		geometry::projective_geometry::projective_space *Projective_space_pointer;

		int f_ring;
		std::string ring_label;

		// not to be documented:
		int f_ring_pointer;
		algebra::ring_theory::homogeneous_polynomial_domain *Ring_pointer;

		int f_equation_in_algebraic_form;
		std::string equation_in_algebraic_form_text;

		int f_set_parameters;
		std::string set_parameters_label;
		std::string set_parameters_label_tex;
		std::string set_parameters_values;

		int f_equation_by_coefficients;
		std::string equation_by_coefficients_text;

		int f_equation_by_rank;
		std::string equation_by_rank_text;

		// unused:
		int f_second_equation_in_algebraic_form;
		std::string second_equation_in_algebraic_form_text;

		// unused:
		int f_second_equation_by_coefficients;
		std::string second_equation_by_coefficients_text;

		int f_points;
		std::string points_txt;

		int f_bitangents;
		std::string bitangents_txt;

		int f_compute_lines;

		std::vector<int> transformation_inverse;
		std::vector<std::string> transformations;

#endif

		Descr->f_projective_space_pointer = true;
		Descr->Projective_space_pointer = P;

		Descr->f_ring_pointer = true;
		Descr->Ring_pointer = HPD;

		Descr->f_compute_lines = true;

		Descr->f_equation_by_rank = true;
		Descr->equation_by_rank_text = std::to_string(T->Reps[orbit_idx].data[0]);


		V = NEW_OBJECT(geometry::algebraic_geometry::variety_object);


		if (f_v) {
			cout << "orbits_on_polynomials::create_vector_of_strings_detailed before V->init" << endl;
		}
		V->init(
				Descr,
				verbose_level);
		// Does not perform the transformations.
		// Called from variety_object_with_action::create_variety
		if (f_v) {
			cout << "orbits_on_polynomials::create_vector_of_strings_detailed after V->init" << endl;
		}

		v[7] = std::to_string(V->Line_sets->Set_size[0]);

		string s_lines;

		s_lines = Lint_vec_stringify(V->Line_sets->Sets[0], V->Line_sets->Set_size[0]);

		v[8] = "\"" + s_lines + "\"";


		if (f_v) {
			cout << "orbits_on_polynomials::create_vector_of_strings_detailed "
					"before V->compute_singular_points" << endl;
		}
		V->compute_singular_points(
				verbose_level - 3);
		if (f_v) {
			cout << "orbits_on_polynomials::create_vector_of_strings_detailed "
					"after V->compute_singular_points" << endl;
		}


		v[9] = std::to_string(V->Singular_points.size());


		string s_singular_points;

		s_singular_points = Lint_vec_stl_stringify(V->Singular_points);

		v[10] = "\"" + s_singular_points + "\"";


		FREE_OBJECT(V);
		FREE_OBJECT(Descr);

	}



	FREE_int(coeff);


	if (f_v) {
		cout << "orbits_on_polynomials::create_vector_of_strings_detailed done" << endl;
	}
}




void orbits_on_polynomials::report_detailed_list(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list" << endl;
	}
	// detailed listing:



	other::data_structures::tally T1;

	T1.init(Nb_pts, T->nb_orbits, false, 0);
	ost << "Distribution of the number of points: $";
	T1.print_bare_tex(ost, true);
	ost << "$\\\\" << endl;

#if 0
	ost << "\\section{The Varieties of degree $" << degree_of_poly
			<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, "
					"detailed listing}" << endl;
	{
		int fst, l, a, r;
		ring_theory::longinteger_object go, go1;
		ring_theory::longinteger_domain D;
		int *coeff;
		int *line_type;
		long int *Pts;
		int nb_pts;
		int *Kernel;
		int *v;
		int i;
		//int h, pt, orbit_idx;

		A->group_order(go);
		Pts = NEW_lint(P->Subspaces->N_points);
		coeff = NEW_int(HPD->get_nb_monomials());
		line_type = NEW_int(P->Subspaces->N_lines);
		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
		v = NEW_int(n);

		for (i = 0; i < Sch->nb_orbits; i++) {
			ost << "\\subsection*{Orbit " << i << " / "
					<< Sch->nb_orbits << "}" << endl;
			fst = Sch->orbit_first[i];
			l = Sch->orbit_len[i];

			D.integral_division_by_int(go, l, go1, r);
			a = Sch->orbit[fst];
			HPD->unrank_coeff_vector(coeff, a);


			vector<long int> Points;

			HPD->enumerate_points(coeff, Points, verbose_level);

			nb_pts = Points.size();
			Pts = NEW_lint(nb_pts);
			for (int u = 0; u < nb_pts; u++) {
				Pts[u] = Points[u];
			}

			ost << "stab order " << go1 << "\\\\" << endl;
			ost << "orbit length = " << l << "\\\\" << endl;
			ost << "orbit rep = " << a << "\\\\" << endl;
			ost << "number of points = " << nb_pts << "\\\\" << endl;

			ost << "$";
			Int_vec_print(ost, coeff, HPD->get_nb_monomials());
			ost << " = ";
			HPD->print_equation(ost, coeff);
			ost << "$\\\\" << endl;


			cout << "We found " << nb_pts << " points in the variety" << endl;
			cout << "They are : ";
			Lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			P->Reporting->print_set_numerical(cout, Pts, nb_pts);

			F->Io->display_table_of_projective_points(
					ost, Pts, nb_pts, n);

			P->Subspaces->line_intersection_type(Pts, nb_pts,
					line_type, 0 /* verbose_level */);

			ost << "The line type is: ";

			stringstream sstr;
			Int_vec_print_classified_str(sstr,
					line_type, P->Subspaces->N_lines,
					true /* f_backwards*/);
			string s = sstr.str();
			ost << "$" << s << "$\\\\" << endl;
			//int_vec_print_classified(line_type, HPD->P->N_lines);
			//cout << "after int_vec_print_classified" << endl;

			ost << "The stabilizer is generated by:" << endl;
			T->Reps[i].Strong_gens->print_generators_tex(ost);
		} // next i

		FREE_lint(Pts);
		FREE_int(coeff);
		FREE_int(line_type);
		FREE_int(Kernel);
		FREE_int(v);
		}
#endif

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list done" << endl;
	}
}


void orbits_on_polynomials::export_something(
		std::string &what, std::string &extra,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::export_something" << endl;
		//cout << "orbits_on_polynomials::export_something this = " << this << endl;
	}

	other::data_structures::string_tools ST;

	string fname_base;

	fname_base = "orbits_" + A2->label + "_" + what;
	if (f_v) {
		cout << "orbits_on_polynomials::export_something what = " << what << endl;
		cout << "orbits_on_polynomials::export_something fname_base = " << fname_base << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::export_something "
				"before export_something_worker" << endl;
	}
	export_something_worker(fname_base, what, extra, fname, verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::export_something "
				"after export_something_worker" << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::export_something done" << endl;
	}

}

void orbits_on_polynomials::export_something_worker(
		std::string &fname_base,
		std::string &what,
		std::string &extra,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::export_something_worker" << endl;
	}

	other::data_structures::string_tools ST;
	other::orbiter_kernel_system::file_io Fio;


	if (ST.stringcmp(what, "orbit") == 0) {

		if (f_v) {
			cout << "orbits_on_polynomials::export_something_worker what=orbit" << endl;
		}

		if (f_has_Sch) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Sch" << endl;
			}

			int data1;

			data1 = std::stoi(extra);

			fname = fname_base + "_orbit_" + std::to_string(data1) + ".csv";

			int orbit_idx = data1;
			std::vector<int> Orb;
			int *Pts;
			int i;

			Sch->Forest->get_orbit_in_order(Orb,
					orbit_idx, verbose_level);

			Pts = NEW_int(Orb.size());
			for (i = 0; i < Orb.size(); i++) {
				Pts[i] = Orb[i];
			}



			Fio.Csv_file_support->int_matrix_write_csv(
					fname, Pts, 1, Orb.size());

			FREE_int(Pts);
		}
		else if (f_has_Orb) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Orb" << endl;
			}

			int data1;

			data1 = std::stoi(extra);

			fname = fname_base + "_orbit_" + std::to_string(data1) + ".csv";

			std::string *Table;
			std::string *Headings;
			int nb_rows, nb_cols;

			Orb->get_table(
					Table, Headings,
					nb_rows, nb_cols,
					verbose_level);

			Fio.Csv_file_support->write_table_of_strings_with_col_headings(
					fname,
					nb_rows, nb_cols, Table,
					Headings,
					verbose_level);
		}
		else {
			cout << "orbits_on_polynomials::export_something_worker neither f_has_Sch nor f_has_Orb" << endl;
			exit(1);
		}

		cout << "orbits_on_polynomials::export_something_worker "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "representatives") == 0) {

		if (f_v) {
			cout << "orbits_on_polynomials::export_something_worker what=representatives" << endl;
		}

		if (f_has_Sch) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Sch" << endl;
			}

			std::string *Table;
			std::string Headings;
			int nb_rows, nb_cols;



			prepare_data(
					Headings,
					Table,
					nb_rows, nb_cols,
					verbose_level);

			fname = fname_base + ".csv";

			Fio.Csv_file_support->write_table_of_strings(
					fname,
					nb_rows, nb_cols, Table,
					Headings,
					verbose_level);

			other::data_structures::tally *Ago_dist;
			long int *Ago;


			Ago_dist = T->get_ago_distribution(
					Ago,
					verbose_level);

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker Ago_dist = ";
				Ago_dist->print_bare(true);
				cout << endl;


				Ago_dist->print_ago_sum_latex();
				cout << endl;

				Ago_dist->print_ago_sum();
				cout << endl;

			}


		}
		else if (f_has_Orb) {
			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Orb" << endl;
			}
			cout << "orbits_on_polynomials::export_something_worker not implemented for this type of orbit" << endl;
			exit(1);
		}
		else {
			cout << "orbits_on_polynomials::export_something_worker neither f_has_Sch nor f_has_Orb" << endl;
			exit(1);
		}

	}
	else if (ST.stringcmp(what, "representatives_detailed") == 0) {

		if (f_v) {
			cout << "orbits_on_polynomials::export_something_worker what=representatives_detailed" << endl;
		}

		if (f_has_Sch) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Sch" << endl;
			}

			std::string *Table;
			std::string Headings;
			int nb_rows, nb_cols;

			//geometry::projective_geometry::projective_space *P;

			layer5_applications::projective_geometry::projective_space_with_action *PA;


			PA = Get_projective_space(extra);

			prepare_data_detailed(
					PA->P,
					Headings,
					Table,
					nb_rows, nb_cols,
					verbose_level);

			fname = fname_base + ".csv";

			Fio.Csv_file_support->write_table_of_strings(
					fname,
					nb_rows, nb_cols, Table,
					Headings,
					verbose_level);

			other::data_structures::tally *Ago_dist;
			long int *Ago;


			Ago_dist = T->get_ago_distribution(
					Ago,
					verbose_level);

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker Ago_dist = ";
				Ago_dist->print_bare(true);
				cout << endl;


				Ago_dist->print_ago_sum_latex();
				cout << endl;

				Ago_dist->print_ago_sum();
				cout << endl;

			}


		}
		else if (f_has_Orb) {
			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Orb" << endl;
			}
			cout << "orbits_on_polynomials::export_something_worker not implemented for this type of orbit" << endl;
			exit(1);
		}
		else {
			cout << "orbits_on_polynomials::export_something_worker neither f_has_Sch nor f_has_Orb" << endl;
			exit(1);
		}
	}
	else {
		cout << "orbits_on_polynomials::export_something_worker "
				"unrecognized export target: " << what << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::export_something_worker done" << endl;
	}

}


void orbits_on_polynomials::compute_small_generating_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::compute_small_generating_set" << endl;
	}

	//algebra::ring_theory::longinteger_object target_go;



	if (f_v) {
		cout << "orbits_on_polynomials::compute_small_generating_set "
				"before A->find_small_generating_set" << endl;
	}
	A->find_small_generating_set(
			A->Strong_gens->gens,
			target_go,
			generating_set_small,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::compute_small_generating_set "
				"after A->find_small_generating_set" << endl;
	}

	f_has_small_generating_set = true;

	if (f_v) {
		cout << "orbits_on_polynomials::compute_small_generating_set done" << endl;
	}
}




}}}

