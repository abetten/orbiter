// incidence_structure.C
// 
// Anton Betten
//
// started 3/14/2012
// based on extra.C
// 
// 
//
//

#include "orbiter.h"

namespace orbiter {
namespace top_level {

void incidence_structure_compute_tda(partitionstack &S, 
	incidence_structure *Inc, 
	action *A, 
	int f_write_tda_files, 
	int f_include_group_order, 
	int f_pic, 
	int f_include_tda_scheme, 
	int verbose_level)
{
	incidence_structure_compute_TDA_general(S, Inc, 
		TRUE, A, NULL, NULL, 
		A->Strong_gens->gens, 
		//A->strong_generators, 
		f_write_tda_files, f_include_group_order, f_pic, f_include_tda_scheme, 
		verbose_level);
}

void incidence_structure_compute_TDA_general(partitionstack &S, 
	incidence_structure *Inc, 
	int f_combined_action, 
	action *A, action *A_on_points, action *A_on_lines, 
	vector_ge *generators, 
	int f_write_tda_files, 
	int f_include_group_order, 
	int f_pic, 
	int f_include_tda_scheme, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v4 = (verbose_level >= 4);
	//partitionstack S;
	//int N;
	//int ht0;
	//int hash;
	//int i;
	char fname[1000];
	char fname_pic[1000];
	char fname_scheme[1000];
	
	if (f_v) {
		cout << "TDA:" << endl;
		cout << "extra.C: incidence_structure_compute_TDA_general" << endl;
		}
	
	if (f_combined_action) {
		longinteger_object ago;
		A->group_order(ago);
		if (f_v) {
			cout << "The automorphism group of the incidence "
					"structure has order " << ago << endl;
			//cout << "f_has_strong_generators=" << A->f_has_strong_generators << endl;
			//cout << "A->degree=" << A->degree << endl;		
			}
		}

	//N = Inc->nb_points() + Inc->nb_lines();
	
	if (f_v) {
		cout << "extra.C: incidence_structure_compute_TDA_general "
				"initial partition:" << endl;
		S.print_classes_points_and_lines(cout);
		}
	//ht0 = S.ht;


	if (f_combined_action) {
		if (f_vv) {
			cout << "extra.C: incidence_structure_compute_TDA_general "
					"setting up schreier" << endl;
			}
		schreier *Sch;
		Sch = NEW_OBJECT(schreier);
		Sch->init(A);
		Sch->initialize_tables();
		Sch->init_generators(*generators);
		if (f_vv) {
			cout << "extra.C: incidence_structure_compute_TDA_general "
					"before compute_all_point_orbits" << endl;
			}
		Sch->compute_all_point_orbits(verbose_level + 3);
		
		if (f_v) {
			cout << "found " << Sch->nb_orbits << " orbits on points "
					"and lines" << endl;
			}
		S.split_by_orbit_partition(Sch->nb_orbits, 
			Sch->orbit_first, Sch->orbit_len, Sch->orbit,
			0 /* offset */, 
			verbose_level - 2);
		FREE_OBJECT(Sch);
		}
	else {
		schreier *Sch_points;
		schreier *Sch_lines;
		Sch_points = NEW_OBJECT(schreier);
		Sch_points->init(A_on_points);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*generators);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_points->nb_orbits
					<< " orbits on points" << endl;
			}
		Sch_lines = NEW_OBJECT(schreier);
		Sch_lines->init(A_on_lines);
		Sch_lines->initialize_tables();
		Sch_lines->init_generators(*generators);
		Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);
		
		if (f_v) {
			cout << "found " << Sch_lines->nb_orbits
					<< " orbits on lines" << endl;
			}
		S.split_by_orbit_partition(Sch_points->nb_orbits, 
			Sch_points->orbit_first, Sch_points->orbit_len,
			Sch_points->orbit,
			0 /* offset */, 
			verbose_level - 2);
		S.split_by_orbit_partition(Sch_lines->nb_orbits, 
			Sch_lines->orbit_first, Sch_lines->orbit_len,
			Sch_lines->orbit,
			Inc->nb_points() /* offset */, 
			verbose_level - 2);
		FREE_OBJECT(Sch_points);
		FREE_OBJECT(Sch_lines);
		}



	if (f_v) {
		cout << "extra.C: incidence_structure_compute_TDA_general "
				"the decomposition schemes:" << endl;
		//cout << S << endl;
		Inc->get_and_print_decomposition_schemes(S);
		Inc->get_and_print_decomposition_schemes_tex(S);
		S.print_classes_points_and_lines(cout);
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(cout,
				FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, S);
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(cout,
				FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, S);
		}


	if (f_write_tda_files) {
		sprintf(fname, "%s_tda.tex", Inc->label);
		sprintf(fname_pic, "%s_tda_pic.tex", Inc->label);
		sprintf(fname_scheme, "%s_tda_scheme.tex", Inc->label);
		{
		ofstream fp(fname);
		ofstream fp_pic(fname_pic);
		ofstream fp_scheme(fname_scheme);

		if (f_include_group_order || f_include_tda_scheme) {
			fp << "\\subsection*{The TDA at Height $"
					<< S.ht << "$}" << endl;
			fp << "$\\begin{array}{c}" << endl;
			if (f_pic) {
				fp << "\\input " << fname_pic << endl;
				Inc->latex_it(fp_pic, S);
				fp << "\\\\" << endl;
				}
			if (f_include_group_order) {
				longinteger_object ago;
				if (f_combined_action) {
					A->group_order(ago);
					}
				else {
					A_on_points->group_order(ago);
					}
				fp << ago << "\\\\" << endl;
				}
			if (f_include_tda_scheme) {
				fp << "\\input " << fname_scheme << endl;
				Inc->get_and_print_tactical_decomposition_scheme_tex(
					fp_scheme, FALSE /* f_enter_math */, S);
				}
			fp << "\\end{array}$" << endl;
			}
		else {
			Inc->latex_it(fp_pic, S);
			}
		}
		if (f_v) {
			cout << "written file " << fname << " of size "
					<< file_size(fname) << endl;
			cout << "written file " << fname_pic << " of size "
					<< file_size(fname_pic) << endl;
			cout << "written file " << fname_scheme << " of size "
					<< file_size(fname_scheme) << endl;
			}
		}


	int nb_V, nb_B;
	int *Vi, *Bj;
	int *R;
	int *X;
	incidence_structure *Inc2;

	Inc2 = NEW_OBJECT(incidence_structure);

	Inc->rearrange(Vi, nb_V, Bj, nb_B, R, X, S);
	
	Inc2->init_by_R_and_X(Inc->nb_points(),
			Inc->nb_lines(), R, X, Inc->max_r, verbose_level);
		
	if (f_write_tda_files) {
		sprintf(fname, "%s_tda.inc", Inc->label);
		Inc2->save_inc_file(fname);
		if (f_v) {
			cout << "written file " << fname << " of size "
					<< file_size(fname) << endl;
			}
		}

	FREE_OBJECT(Inc2);
	FREE_int(Vi);
	FREE_int(Bj);
	FREE_int(R);
	FREE_int(X);

	int f_labeled = TRUE;
	if (f_vv && FALSE) {
		Inc->print_partitioned(cout, S, f_labeled);
		}

#if 0
	if (f_v4) {
		for (i = 0; i < Sch->nb_orbits; i++) {
			f = Sch->orbit_first[i];
			l = Sch->orbit_len[i];
			if (f_v) {
				cout << "orbit " << i << " first=" << f
						<< " length=" << l << endl;
				}
			for (j = 0; j < l; j++) {
				Set[j] = Sch->orbit[f + j];
				}
			if (f_v) {
				cout << "orbit: ";
				int_vec_print(cout, Set, l);
				cout << endl;
				}
			sims *Stab;
			longinteger_object go;

			Sch->point_stabilizer(A, ago, Stab, i, verbose_level - 3);
			Stab->group_order(go);
		
			cout << "Orbit " << i << ", the stabilizer of point " << Set[0]
				<< " has order " << go << " and is generated by:" << endl;
			if (go.as_int() > 1) {
				Stab->print_generators();
				cout << "Orbit " << i << ", the stabilizer of point "
					<< Set[0] << " has order " << go
					<< " and is generated by:" << endl;
				Stab->print_generators_tex(cout);
				}
			else {
				cout << "The empty set" << endl;
				}

			FREE_OBJECT(Stab);
			}
		}
#endif


	//FREE_int(Set);
}


void incidence_structure_compute_TDO_TDA(incidence_structure *Inc, 
	int f_tda_files, 
	int f_tda_with_group_order, 
	int f_tda_with_scheme, 
	int f_pic, 
	int &TDO_ht, int &TDA_ht, 
	int verbose_level)
// called from INC_CAN/inc_select.C
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	partitionstack S;

	int N;

	if (f_v) {
		cout << "incidence_structure_compute_TDO_TDA" << endl;
		}
	N = Inc->nb_points() + Inc->nb_lines();
	
	S.allocate(N, 0);
	// split off the column class:
	S.subset_continguous(Inc->nb_points(), Inc->nb_lines());
	S.split_cell(0);
				
	int TDO_depth = N;


	if (f_vv) {
		cout << "before Inc->compute_TDO_safe" << endl;
		}
	Inc->compute_TDO_safe(S, TDO_depth, verbose_level - 3);
	TDO_ht = S.ht;


	action *A;
	longinteger_object ago;
	nauty_interface Nauty;

	
	if (f_vv) {
		cout << "before create_automorphism_group_of_"
				"incidence_structure" << endl;
		}
	A = Nauty.create_automorphism_group_of_incidence_structure(
			Inc, 
			verbose_level - 3);
	A->group_order(ago);


	if (f_vv) {
		cout << "before incidence_structure_compute_tda" << endl;
		}
	incidence_structure_compute_tda(S, Inc, 
		A, 
		f_tda_files, 
		f_tda_with_group_order, 
		f_pic, 
		f_tda_with_scheme, 
		verbose_level - 3);

	TDA_ht = S.ht;
	FREE_OBJECT(A);
}

int incidence_structure_find_blocking_set(
	incidence_structure *Inc, int input_no,
	int *blocking_set, int &blocking_set_size, 
	int blocking_set_starter_size, 
	int f_all_blocking_sets, 
	int f_blocking_set_size_desired, int blocking_set_size_desired, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_v4 = (verbose_level >= 4);
	int i, j;
	

	if (f_v) {
		cout << "incidence_structure_find_blocking_set "
				"input_no=" << input_no << endl;
		cout << "blocking_set_starter_size="
				<< blocking_set_starter_size << endl;
		cout << "f_all_blocking_sets = "
				<< f_all_blocking_sets << endl;
		cout << "f_blocking_set_size_desired = "
				<< f_blocking_set_size_desired << endl;
		if (f_blocking_set_size_desired) {
			cout << "blocking_set_size_desired = "
					<< blocking_set_size_desired << endl;
			}
		if ((input_no % 500) == 0) {
			cout << "incidence_structure_find_blocking_set "
					"input_no=" << input_no << endl;
			}
		}


	action *A;
	nauty_interface Nauty;

	A = Nauty.create_automorphism_group_of_incidence_structure(
		Inc, verbose_level);


	
	search_blocking_set SBS;

	SBS.init(Inc, A, verbose_level - 1);

	SBS.f_blocking_set_size_desired = f_blocking_set_size_desired;
	SBS.blocking_set_size_desired = blocking_set_size_desired;
	
	int depth = blocking_set_starter_size;
	int level;
	int f_OK;
	
	SBS.find_partial_blocking_sets(depth, verbose_level - 1);

	level = depth;
	if (f_vv) {
		int f, nb_orbits;
		
		cout << "after find_partial_blocking_sets" << endl;
		f = SBS.gen->first_poset_orbit_node_at_level[depth];
		nb_orbits = SBS.gen->first_poset_orbit_node_at_level[depth + 1] - f;
		cout << "incidence_structure_find_blocking_set: "
				"we found " << nb_orbits << " orbits on partial "
						"blocking sets of size " << depth << endl;
		}

	f_OK = SBS.test_level(level, verbose_level - 1);
	if (f_OK) {
		blocking_set_size = level;
		}
	else {
		SBS.search_for_blocking_set(input_no, level,
				f_all_blocking_sets, verbose_level - 1);

		if (SBS.nb_solutions) {
			f_OK = TRUE;
			blocking_set_size = SBS.solutions.front().size();
			for (j = 0; j < blocking_set_size; j++) {
				blocking_set[j] = SBS.solutions.front()[j];
				}
			}

		}
	if (f_v) {
		cout << "incidence_structure_find_blocking_set "
				"found blocking set of size " << blocking_set_size << ":";
		int_vec_print(cout, SBS.blocking_set, blocking_set_size);
		cout << endl;
		}
	for (i = 0; i < blocking_set_size; i++) {
		blocking_set[i] = SBS.blocking_set[i];
		}
	

	FREE_OBJECT(A);
	
	return f_OK;
}

}}


