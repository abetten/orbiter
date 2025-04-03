/*
 * semifield_lifting.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {



semifield_lifting::semifield_lifting()
{
	Record_birth();
	SC = NULL;
	L2 = NULL;
	Prev = NULL;

	n = k = k2 = 0;

	cur_level = 0;
	prev_level_nb_orbits = 0;

	f_prefix = false;
	//prefix = NULL;

	Prev_stabilizer_gens = NULL;
	Candidates = NULL;
	Nb_candidates = NULL;

	Downstep_nodes = NULL;
	nb_flag_orbits = 0;
	flag_orbit_first = NULL;
	flag_orbit_len = NULL;
	Flag_orbits = NULL;
	Gr = NULL;

	nb_orbits = 0;
	Po = NULL;
	So = NULL;
	Mo = NULL;
	Go = NULL;
	Pt = NULL;
	Stabilizer_gens = NULL;

	Matrix0 = Matrix1 = Matrix2 = NULL;
	window_in = NULL;

	ELT1 = ELT2 = ELT3 = NULL;
	basis_tmp = NULL;
	base_cols = NULL;
	M1 = NULL;
	Basis = NULL;
	Rational_normal_form = NULL;
	R1 = NULL;

}

semifield_lifting::~semifield_lifting()
{
	Record_death();
	if (flag_orbit_first) {
		FREE_int(flag_orbit_first);
	}
	if (flag_orbit_len) {
		FREE_int(flag_orbit_len);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	if (Matrix0) {
		FREE_int(Matrix0);
	}
	if (Matrix1) {
		FREE_int(Matrix1);
	}
	if (Matrix2) {
		FREE_int(Matrix2);
	}
	if (window_in) {
		FREE_int(window_in);
	}
	if (ELT1) {
		FREE_int(ELT1);
	}
	if (ELT2) {
		FREE_int(ELT2);
	}
	if (ELT3) {
		FREE_int(ELT3);
	}

	if (M1) {
		FREE_int(M1);
	}
	if (Basis) {
		FREE_int(Basis);
	}
	if (Rational_normal_form) {
		FREE_int(Rational_normal_form);
	}

	if (basis_tmp) {
		FREE_int(basis_tmp);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (R1) {
		FREE_OBJECT(R1);
	}
}

void semifield_lifting::init_level_three(
		semifield_level_two *L2,
		int f_prefix, std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int level = 3;

	if (f_v) {
		cout << "semifield_lifting::init_level_three" << endl;
	}
	semifield_lifting::L2 = L2;
	SC = L2->SC;
	n = SC->n;
	k = SC->k;
	k2 = SC->k2;
	cur_level = 3;
	prev_level_nb_orbits = L2->nb_orbits;
	Prev_stabilizer_gens = L2->Stabilizer_gens;
	semifield_lifting::f_prefix = f_prefix;
	semifield_lifting::prefix.assign(prefix);

	Gr = NEW_OBJECT(geometry::projective_geometry::grassmann);
	Gr->init(level, level - 1, SC->Mtx->GFq, 0/*verbose_level - 10*/);

	Matrix0 = NEW_int(k2);
	Matrix1 = NEW_int(k2);
	Matrix2 = NEW_int(k2);
	window_in = NEW_int(k2);

	actions::action *A;

	A = SC->A;

	ELT1 = NEW_int(A->elt_size_in_int);
	ELT2 = NEW_int(A->elt_size_in_int);
	ELT3 = NEW_int(A->elt_size_in_int);


	M1 = NEW_int(n * n);
	Basis = NEW_int(k * k);
	Rational_normal_form = NEW_int(k * k);
	basis_tmp = NEW_int(k * k2);
	base_cols = NEW_int(k2);

	R1 = NEW_OBJECT(algebra::linear_algebra::gl_class_rep);

	if (f_v) {
		cout << "semifield_lifting::init_level_three done" << endl;
	}
}


void semifield_lifting::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i; //, j, ext, idx;
	//long int a; //, b;


	if (f_v) {
		cout << "semifield_lifting::report" << endl;
	}

	{
		//const char *fname = "Reps_lvl_2.tex";
		//ofstream fp(fname);
		//latex_interface L;


		int *Mtx_Id;
		int *Mtx1;
		int *Mtx2;

		Mtx_Id = NEW_int(k2);
		Mtx1 = NEW_int(k2);
		Mtx2 = NEW_int(k2);

		SC->Mtx->GFq->Linear_algebra->identity_matrix(Mtx_Id, k);


		if (f_v) {
			cout << "semifield_lifting::report before flag orbits" << endl;
		}

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;



		ost << "\\section{Flag orbits at level 3}" << endl;

		ost << endl;

		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "\\mbox{L2 Orbit}  & \\mbox{Cand} & \\mbox{Flag orbits} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (i = 0; i < prev_level_nb_orbits; i++) {
			ost << i << " & " << L2->Nb_candidates[i] << " & "
					<< Downstep_nodes[i].Sch->Forest->nb_orbits << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;


		ost << endl;
		ost << "There are " << nb_flag_orbits << " flag orbits at level 3" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		ost << "\\begin{enumerate}[(1)]" << endl;

		for (i = 0; i < nb_flag_orbits; i++) {


			ost << "\\item" << endl;
			ost << i << " / " << nb_flag_orbits << ":\\\\" << endl;


			ost << "po=" << Flag_orbits[i].downstep_primary_orbit << ", ";
			ost << "so=" << Flag_orbits[i].downstep_secondary_orbit << ", ";
			ost << "pt=" << Flag_orbits[i].pt << ", ";
			ost << "ol=" << Flag_orbits[i].downstep_orbit_len << "\\\\" << endl;

			if (!Flag_orbits[i].f_fusion_node) {
				ost << "Defining node for orbit " << Flag_orbits[i].upstep_orbit << endl;
				//ost << "Flag orbit stabilizer has order " << Flag_orbits[i].gens->group_order_as_lint() << "\\\\" << endl;
				//Flag_orbits[i].gens->print_generators_tex(ost);
			}
			else {
				ost << "Fusion node for orbit " << Flag_orbits[i].fusion_with << endl;
			}


#if 0
			int downstep_primary_orbit;
			int downstep_secondary_orbit;
			int pt_local;
			long int pt;
			int downstep_orbit_len;
			int f_long_orbit;
			int upstep_orbit; // if !f_fusion_node
		 	int f_fusion_node;
			int fusion_with;
			int *fusion_elt;

			longinteger_object go;
			strong_generators *gens;
#endif

		}

		ost << "\\end{enumerate}" << endl;

		if (f_v) {
			cout << "semifield_lifting::report before Orbits at level 3" << endl;
		}
		ost << "\\section{Orbits at level 3}" << endl;

		ost << endl;
		ost << "There are " << nb_orbits << " orbits at level 3" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;


		ost << "\\begin{enumerate}[(1)]" << endl;
		for (i = 0; i < nb_orbits; i++) {


			ost << "\\item" << endl;
			algebra::ring_theory::longinteger_object go;
			int po;

			Stabilizer_gens[i].group_order(go);

			po = Po[i];
			ost << "go=" << go << ", ";
			ost << "po=" << Po[i] << ", ";
			ost << "so=" << So[i] << ", ";
			ost << "mo=" << Mo[i] << ", ";
			ost << "go=" << Go[i] << ", ";
			ost << "pt=" << Pt[i] << "\\\\" << endl;

			SC->matrix_unrank(Pt[i], Mtx2);

			int f_elements_exponential = false;

			SC->matrix_unrank(L2->Pt[po], Mtx1);


			string symbol_for_print;

			symbol_for_print.assign("\\alpha");

			ost << "$$" << endl;
			ost << "\\left\\{" << endl;
			ost << "\\left[" << endl;
			SC->Mtx->GFq->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx_Id, k, k);
			ost << "\\right]";
			ost << ",";
			ost << "\\left[" << endl;
			SC->Mtx->GFq->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx1, k, k);
			ost << "\\right]";
			ost << ",";
			ost << "\\left[" << endl;
			SC->Mtx->GFq->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx2, k, k);
			ost << "\\right]";
			ost << "\\right\\}" << endl;
			ost << "_{";
			ost << go << "}" << endl;
			ost << "$$" << endl;

			Stabilizer_gens[i].print_generators_tex(ost);

		}
		ost << "\\end{enumerate}" << endl;
		ost << endl;
		if (f_v) {
			cout << "semifield_lifting::report "
					"after Orbits at level 3" << endl;
		}


		FREE_int(Mtx_Id);
		FREE_int(Mtx1);
		FREE_int(Mtx2);
	}
	if (f_v) {
		cout << "semifield_lifting::report done" << endl;
	}
}


void semifield_lifting::recover_level_three_downstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_downstep" << endl;
	}
	if (f_v) {
		cout << "semifield_lifting::recover_level_three_downstep "
				"before downstep" << endl;
	}

	downstep(2, verbose_level);

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_downstep "
				"after downstep" << endl;
	}


	if (f_v) {
		cout << "semifield_lifting::recover_level_three_downstep "
				"after downstep" << endl;
	}


	if (f_v) {
		cout << "semifield_lifting::recover_level_three_downstep done" << endl;
	}
}

void semifield_lifting::recover_level_three_from_file(
		int f_read_flag_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file" << endl;
	}

#if 0
	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"before find_all_candidates" << endl;
	}

	find_all_candidates(2, verbose_level);

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"after find_all_candidates" << endl;
	}
#endif


	if (f_read_flag_orbits) {

		if (f_v) {
			cout << "semifield_lifting::recover_level_three_from_file "
					"before read_flag_orbits" << endl;
		}
		read_flag_orbits(verbose_level);
		if (f_v) {
			cout << "semifield_lifting::recover_level_three_from_file "
					"after read_flag_orbits" << endl;
		}
	}

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"before read_level_info_file" << endl;
	}
	read_level_info_file(verbose_level);
	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"after read_level_info_file" << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"before read_stabilizers" << endl;
	}
	read_stabilizers(verbose_level);
	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file "
				"after read_stabilizers" << endl;
	}



	if (f_v) {
		cout << "semifield_lifting::recover_level_three_from_file" << endl;
	}
}

void semifield_lifting::compute_level_three(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::compute_level_three" << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"before level_two_down" << endl;
	}

	level_two_down(verbose_level - 2);

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"after level_two_down" << endl;
	}
	if (f_v) {
		int orbit;
		cout << "semifield_lifting::level_two_down done, "
				"we found the following candidate sets:" << endl;
		cout << "Orbit : # candidates : # orbits" << endl;
		for (orbit = 0; orbit < prev_level_nb_orbits; orbit++) {
			cout << orbit << " : " << L2->Nb_candidates[orbit]
				<< " : " << Downstep_nodes[orbit].Sch->Forest->nb_orbits << endl;
		}
	}

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"before level_two_flag_orbits" << endl;
	}

	level_two_flag_orbits(verbose_level - 1);

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"after level_two_flag_orbits, nb_flag_orbits=" << nb_flag_orbits << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"before level_two_upstep" << endl;
	}

	level_two_upstep(verbose_level - 1);

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"after level_two_upstep, nb_orbits=" << nb_orbits << endl;
	}
	if (f_v) {
		int i;
		for (i = 0; i < nb_orbits; i++) {
			cout << i;
			cout << "," << Stabilizer_gens[i].group_order_as_lint() << ","
				<< Po[i] << "," << So[i] << "," << Mo[i] << "," << Pt[i] << endl;
		}
	}

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"before save_flag_orbits" << endl;
	}

	save_flag_orbits(verbose_level);

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"after save_flag_orbits" << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"before save_stabilizers" << endl;
	}
	save_stabilizers(verbose_level);
	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"after save_stabilizers" << endl;
	}


	if (f_v) {
		cout << "semifield_lifting::compute_level_three "
				"done" << endl;
	}
}

void semifield_lifting::level_two_down(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::level_two_down" << endl;
	}
	if (f_v) {
		cout << "semifield_lifting::level_two_down "
				"before downstep(2)" << endl;
	}



	downstep(2, verbose_level);

#if 0
	int Level_two_nb_orbits;
	long int **Level_two_Candidates;
	int *Level_two_Nb_candidates;

	Level_two_nb_orbits = L2->nb_orbits;
	Level_two_Candidates = L2->Candidates;
	Level_two_Nb_candidates = L2->Nb_candidates;

	//int **Candidates;
		// candidates for the generator matrix,
		// [nb_orbits]
	//int *Nb_candidates;
		// [nb_orbits]
#endif


	if (f_v) {
		cout << "semifield_lifting::level_two_down "
				"after downstep(2)" << endl;
	}
	if (f_v) {
		int orbit;
		cout << "semifield_lifting::level_two_down done, "
				"we found the following candidate sets:" << endl;
		cout << "Orbit : # candidates : # orbits" << endl;
		for (orbit = 0; orbit < prev_level_nb_orbits; orbit++) {
			cout << orbit << " : " << L2->Nb_candidates[orbit]
				<< " : " << Downstep_nodes[orbit].Sch->Forest->nb_orbits << endl;
		}
	}
	if (f_v) {
		cout << "semifield_lifting::level_two_down done" << endl;
	}
}

void semifield_lifting::level_two_flag_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::level_two_flag_orbits" << endl;
	}

	compute_flag_orbits(2 /* level */, verbose_level);

	if (f_v) {
		cout << "semifield_lifting::level_two_flag_orbits done" << endl;
	}
}

void semifield_lifting::level_two_upstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::level_two_upstep" << endl;
	}
	upstep(3 /* level */,
#if 0
		Level_two_nb_orbits /* old_nb_orbits */,
		Stabilizer_gens /* old_stabilizer_gens */,
		Level_two_down /* Down */,
		Level_two_middle /* M */,
		level_two_nb_middle_nodes /* nb_middle_nodes */,
		Level_three_po /* Po */,
		Level_three_so /* So */,
		Level_three_mo /* Mo */,
		Level_three_pt /* Pt */,
		Level_three_stabilizer_gens /* stabilizer_gens */,
		Level_three_nb_orbits /* new_nb_orbits */,
#endif
		verbose_level);


	if (f_v) {
		cout << "semifield_lifting::level_two_upstep done" << endl;
	}
}

void semifield_lifting::downstep(
	int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::downstep "
				"level = " << level << endl;
	}
	int orbit;


	if (f_v) {
		cout << "semifield_lifting::downstep "
				"level = " << level << " before find_all_candidates" << endl;
	}
	find_all_candidates(level, verbose_level);


	if (f_v) {
		int i;

		cout << "semifield_lifting::downstep "
				"level = " << level << " after find_all_candidates" << endl;
		cout << "i : Nb_candidates[i]" << endl;
		for (i = 0; i < prev_level_nb_orbits; i++) {
			cout << i << " : " << Nb_candidates[i] << endl;
		}
		cout << endl;
	}


	if (f_v) {
		cout << "semifield_lifting::downstep "
				"level = " << level
				<< " before processing all primary orbits" << endl;
	}
	Downstep_nodes = NEW_OBJECTS(semifield_downstep_node,
			prev_level_nb_orbits);

	int first_flag_orbit = 0;

	for (orbit = 0; orbit < prev_level_nb_orbits; orbit++) {
		if (f_v) {
			cout << "semifield_lifting::downstep "
					"processing orbit " << orbit << " / "
					<< prev_level_nb_orbits << endl;
		}

		Downstep_nodes[orbit].init(this, level, orbit,
			Candidates[orbit], Nb_candidates[orbit], first_flag_orbit,
			verbose_level - 2);

		first_flag_orbit += Downstep_nodes[orbit].Sch->Forest->nb_orbits;

		if (f_v) {
			cout << "semifield_lifting::downstep "
					"level = " << level
					<< " orbit " << orbit << " / "
					<< prev_level_nb_orbits << " : we found "
					<< Downstep_nodes[orbit].Sch->Forest->nb_orbits << " orbits" << endl;
		}

		//cout << "semifield_starter::downstep processing "
		//"orbit " << orbit << " / " << nb_orbits_at_level << " done" << endl;
	}
	if (f_v) {
		cout << "semifield_lifting::downstep "
				"level = " << level
				<< " after processing all primary orbits" << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::downstep level " << level << endl;
		cout << "orbit : candidates : number of down orbits" << endl;
		for (orbit = 0; orbit < prev_level_nb_orbits; orbit++) {
			cout << orbit << " : " << Nb_candidates[orbit]
				<< " : " << Downstep_nodes[orbit].Sch->Forest->nb_orbits << endl;
		}
	}
	if (f_v) {
		int *Nb_orbits;
		Nb_orbits = NEW_int(prev_level_nb_orbits);
		for (orbit = 0; orbit < prev_level_nb_orbits; orbit++) {
			Nb_orbits[orbit] = Downstep_nodes[orbit].Sch->Forest->nb_orbits;
		}
		other::data_structures::tally C;

		C.init(Nb_orbits, prev_level_nb_orbits, false, 0);
		cout << "semifield_lifting::downstep "
				"level " << level << " distribution of orbit lengths: ";
		C.print(true /* f_backwards */);
		FREE_int(Nb_orbits);
	}
	if (f_v) {
		cout << "semifield_lifting::downstep done" << endl;
	}
}


void semifield_lifting::compute_flag_orbits(
	int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_v4 = (verbose_level >= 4);
	int po;
	int so, f, pt_local, len;
	long int pt;
	algebra::ring_theory::longinteger_domain D;
	int *Mtx1;

	if (f_v) {
		cout << "semifield_lifting::compute_flag_orbits "
				"level = " << level
				<< " verbose_level = " << level
				<< endl;
	}

	//int nb_flag_orbits;

	//semifield_flag_orbit_node *Flag_orbits;



	//Elt1 = NEW_int(A_PGLk->elt_size_in_int);
	Mtx1 = NEW_int(k2);

	flag_orbit_first = NEW_int(prev_level_nb_orbits);
	flag_orbit_len = NEW_int(prev_level_nb_orbits);

	nb_flag_orbits = 0;
	for (po = 0; po < prev_level_nb_orbits; po++) {

		flag_orbit_first[po] = nb_flag_orbits;
		flag_orbit_len[po] = Downstep_nodes[po].Sch->Forest->nb_orbits;

		if (flag_orbit_first[po] != Downstep_nodes[po].first_flag_orbit) {
			cout << "semifield_lifting::compute_flag_orbits "
					"flag_orbit_first[po] != Downstep_nodes[po].first_flag_orbit" << endl;
			exit(1);
		}
		//Downstep_nodes[po].first_flag_orbit = nb_flag_orbits;
		nb_flag_orbits += flag_orbit_len[po];
	}

	if (f_v && prev_level_nb_orbits < 100) {
		cout << "semifield_lifting::compute_flag_orbits "
				"done with downstep at level " << level << ":" << endl;
		cout << "orbit : number of orbits" << endl;
		for (po = 0; po < prev_level_nb_orbits; po++) {
			cout << po << " : " << Downstep_nodes[po].Sch->Forest->nb_orbits << endl;
		}
	}
	if (f_v) {
		cout << "nb_flag_orbits = " << nb_flag_orbits << endl;
	}

	if (f_v) {
		cout << "allocating flag orbit nodes" << endl;
	}
	Flag_orbits = NEW_OBJECTS(semifield_flag_orbit_node, nb_flag_orbits);


	if (f_v) {
		cout << "looping over all " << prev_level_nb_orbits
				<< " primary orbits" << endl;
	}
	for (po = 0, f = 0; po < prev_level_nb_orbits; po++) {
		groups::schreier *S;
		algebra::ring_theory::longinteger_object go_prev;
		int go_prev_int;


		L2->Stabilizer_gens[po].group_order(go_prev);
		go_prev_int = go_prev.as_int();

		S = Downstep_nodes[po].Sch;

		if (f_v) {
			cout << "semifield_lifting::compute_flag_orbits "
					"at level " << level << ": orbit = " << po
					<< " / " << prev_level_nb_orbits
					<< " stabilizer order " << go_prev
					<< " nb_secondary_orbits = " << S->Forest->nb_orbits
					<< " flag orbit f = " << f << " / " << nb_flag_orbits
					<< endl;
		}

		for (so = 0; so < S->Forest->nb_orbits; so++, f++) {

			groups::sims *Stab;
			int r;
			int f_long_orbit;

			pt_local = S->Forest->orbit[S->Forest->orbit_first[so]];
			pt = Downstep_nodes[po].Candidates[pt_local];
			len = S->Forest->orbit_len[so];
			if (len == go_prev_int) {
				f_long_orbit = true;
			}
			else {
				f_long_orbit = false;
			}
			Flag_orbits[f].init(po, so,
					pt_local, pt, len, f_long_orbit,
					0 /*verbose_level*/);

			D.integral_division_by_int(go_prev, len, Flag_orbits[f].go, r);
			if ((f % 100) == 0) {
				cout << "flag orbit " << f << " / " << nb_flag_orbits
						<< " po = " << po << " / "
						<< prev_level_nb_orbits
						<< " so = " << so
						<< " / " << S->Forest->nb_orbits
						<< " pt_local=" << pt_local
						<< " / " << Downstep_nodes[po].nb_candidates
						<< " pt=" << pt
						<< " len=" << len
						<< " computing stabilizer of order "
						<< Flag_orbits[f].go << " f_long_orbit="
						<< f_long_orbit << endl;
				SC->matrix_unrank(pt, Mtx1);
				cout << "element " << pt << " is" << endl;
				Int_matrix_print(Mtx1, k, k);
			}

			if (!f_long_orbit) {
				if (false/*M[j].go.is_one()*/) {
					Flag_orbits[f].gens = NEW_OBJECT(groups::strong_generators);
					Flag_orbits[f].gens->init_trivial_group(SC->A, 0);
				}
				else {
					S->point_stabilizer(SC->A, go_prev,
						Stab, so /* orbit_no */, 0 /* verbose_level */);

					//if (f_vvv) {
						//cout << "orbit=" << orbit << " orbit=" << i
						// << " computing strong generators" << endl;
						//}
					Flag_orbits[f].gens = NEW_OBJECT(groups::strong_generators);
					Flag_orbits[f].gens->init_from_sims(Stab, 0 /* verbose_level */);


					FREE_OBJECT(Stab);
					if ((f % 100) == 0) {
						algebra::ring_theory::longinteger_object go;

						Flag_orbits[f].gens->group_order(go);
						cout << "The flag orbit stabilizer has order " << go << endl;
					}
				}
			}
			else {
				Flag_orbits[f].gens = NULL;
			}
		}
		if (f_v) {
			cout << "semifield_lifting::compute_flag_orbits "
					"at level " << level << ": orbit = " << po
					<< " / " << prev_level_nb_orbits
					<< " stabilizer order " << go_prev
					<< " nb_secondary_orbits = " << S->Forest->nb_orbits
					<< " done" << endl;
		}
	}

	if (f != nb_flag_orbits) {
		cout << "semifield_lifting::compute_flag_orbits "
				"f != nb_flag_orbits" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "semifield_lifting::compute_flag_orbits "
				"level " << level << " computing the distribution "
						"of stabilizer orders" << endl;
		int i;
		long int *Go;
		Go = NEW_lint(nb_flag_orbits);
		for (i = 0; i < nb_flag_orbits; i++) {
			Go[i] = Flag_orbits[i].group_order_as_int();
			//Go[i] = M[i].gens->group_order_as_int();
			//cout << i << " : " << Go[i] << endl;
		}
		other::data_structures::tally C;

		C.init_lint(Go, nb_flag_orbits, false, 0);
		cout << "semifield_lifting::compute_flag_orbits "
				"level " << level << " distribution of "
						"stabilizer orders of flag orbits is: ";
		C.print(true /* f_backwards */);
		FREE_lint(Go);
	}


	FREE_int(Mtx1);
	if (f_v) {
		cout << "semifield_lifting::compute_flag_orbits "
				"level " << level << " done" << endl;
	}
}

void semifield_lifting::upstep(
	int level,
	int verbose_level)
// level is the level that we want to classify
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 4);

	if (f_v) {
		cout << "semifield_lifting::upstep" << endl;
	}

	if (level != 3) {
		cout << "semifield_lifting::upstep level != 3" << endl;
		exit(1);
	}
	if (cur_level != 3) {
		cout << "semifield_lifting::upstep cur_level != 3" << endl;
		exit(1);
	}
	int f;
	int *transporter;
	int *Mtx;
	//int *pivots;
	int *base_change_matrix;
	int *changed_space;
	//int *changed_space_after_trace;
	long int *set;
	int i, N, h, po, so; //, pt_local;
	long int pt;
	//int trace_po, trace_so;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int /*fo,*/ class_idx;

	transporter = NEW_int(SC->A->elt_size_in_int);

	Mtx = NEW_int(level * k2);
	//pivots = NEW_int(level);
	base_change_matrix = NEW_int(level * level);
	changed_space = NEW_int(level * k2);
	//changed_space_after_trace = NEW_int(level * k2);
	set = NEW_lint(level);

	N = Combi.generalized_binomial(level, level - 1, SC->Mtx->GFq->q);


	Stabilizer_gens = NEW_OBJECTS(groups::strong_generators, nb_flag_orbits);
	Po = NEW_int(nb_flag_orbits);
	So = NEW_int(nb_flag_orbits);
	Mo = NEW_int(nb_flag_orbits);
	Go = NEW_lint(nb_flag_orbits);
	Pt = NEW_lint(nb_flag_orbits);

	nb_orbits = 0;
	for (f = 0; f < nb_flag_orbits; f++) {

		if (f_v) {
			cout << "Level " << level << ": flag orbit "
					<< f << " / " << nb_flag_orbits << endl;
		}

		if (Flag_orbits[f].f_fusion_node) {
			if (f_v) {
				cout << "Level " << level << ": skipping flag orbit "
						<< f << " / " << nb_flag_orbits
						<< " as it is a fusion node" << endl;
			}
			continue;
		}

		Flag_orbits[f].upstep_orbit = nb_orbits;

		SC->Mtx->GFq->Linear_algebra->identity_matrix(Mtx, k);

		po = Flag_orbits[f].downstep_primary_orbit;
		so = Flag_orbits[f].downstep_secondary_orbit;
		//pt_local = Flag_orbits[f].pt_local;
		pt = Flag_orbits[f].pt;

		Po[nb_orbits] = po;
		So[nb_orbits] = so;
		Mo[nb_orbits] = f;
		Go[nb_orbits] = 0;
		Pt[nb_orbits] = pt;

		if (f_v) {
			cout << "semifield_lifting::upstep Level "
					<< level << " flag orbit " << f << " / "
					<< nb_flag_orbits << ", before SC->matrix_unrank, pt = " << pt << endl;
		}

		// level two stuff:
		//fo = L2->Fo[po];
		class_idx = L2->So[po];
		SC->matrix_unrank(L2->class_rep_rank[class_idx], Mtx + 1 * k2);



		//get_pivots(level - 1, po, Mtx, pivots, verbose_level - 3);
		//int_vec_copy(SC->desired_pivots, pivots, level);
		SC->matrix_unrank(pt, Mtx + (level - 1) * k2);

		if (f_v) {
			cout << "semifield_lifting::upstep Level "
					<< level << " flag orbit " << f << " / "
					<< nb_flag_orbits << endl;
		}
		if (f_vv) {
			cout << "Mtx=" << endl;
			Int_matrix_print(Mtx, level, k2);
		}



		int **Aut;

		Aut = NEW_pint(N);

		upstep_loop_over_down_set(
			level, f, po, so, N,
			transporter, Mtx, //pivots,
			base_change_matrix, changed_space,
			//changed_space_after_trace,
			set,
			Aut,
			verbose_level - 1);


		int nb_aut_gens;

		nb_aut_gens = 0;
		for (h = 0; h < N; h++) {
			if (Aut[h]) {
				nb_aut_gens++;
			}
		}

		if (Flag_orbits[f].f_long_orbit) {
			Stabilizer_gens[nb_orbits].init_trivial_group(
					SC->A, 0 /* verbose_level */);
		}
		else {
			Stabilizer_gens[nb_orbits].init_copy(Flag_orbits[f].gens, 0);
		}

		data_structures_groups::vector_ge *coset_reps;

		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(SC->A, verbose_level - 2);
		coset_reps->allocate(nb_aut_gens, verbose_level - 2);
		i = 0;
		for (h = 0; h < N; h++) {
			if (Aut[h]) {
				SC->A->Group_element->element_move(Aut[h], coset_reps->ith(i), 0);
				i++;
			}
		}


		for (h = 0; h < N; h++) {
			if (Aut[h]) {
				FREE_int(Aut[h]);
			}
		}
		FREE_pint(Aut);

		if (f_v) {
			cout << "Level " << level << " orbit " << nb_orbits
					<< " flag orbit " << f << " = " << po << " / " << so
					<< " We are now extending the group by a factor of "
					<< nb_aut_gens << ":" << endl;
		}

		Stabilizer_gens[nb_orbits].add_generators(
				coset_reps, nb_aut_gens,
				0 /*verbose_level - 1*/);

		FREE_OBJECT(coset_reps);

		algebra::ring_theory::longinteger_object go;

		Stabilizer_gens[nb_orbits].group_order(go);
		Go[nb_orbits] = go.as_lint();
		if (f_v) {
			cout << "Level " << level << " orbit " << nb_orbits
					<< " The new group order is " << go << endl;
		}
		if (f_v && ((f & ((1 << 10) - 1)) == 0)) {
			cout << "Level " << level << ": flag orbit " << f << " / "
					<< nb_flag_orbits << " is linked to new orbit "
					<< Flag_orbits[f].upstep_orbit << endl;
		}



		nb_orbits++;
	}


	if (f_v) {
		cout << "semifield_lifting::upstep "
				"newly computed level " << level
				<< " distribution of stabilizer orders: ";
		print_stabilizer_orders();
	}
	if (f_v) {
		cout << "Level " << level << ", done with upstep, "
				"we found " << nb_orbits << " orbits" << endl;
	}



	if (f_v) {
		cout << "Level " << level << ", done with upstep, "
				"we found " << nb_orbits << " orbits, "
						"saving information" << endl;
	}

	write_level_info_file(verbose_level);


	FREE_int(transporter);
	FREE_int(Mtx);
	//FREE_int(pivots);
	FREE_int(base_change_matrix);
	FREE_int(changed_space);
	//FREE_int(changed_space_after_trace);
	FREE_lint(set);

	if (f_v) {
		cout << "semifield_lifting::upstep "
				"level " << level << " done" << endl;
	}
}



void semifield_lifting::upstep_loop_over_down_set(
	int level, int f, int po, int so, int N,
	int *transporter, int *Mtx, //int *pivots,
	int *base_change_matrix,
	int *changed_space,
	//int *changed_space_after_trace,
	long int *set,
	int **Aut,
	int verbose_level)
// level is the level that we want to classify
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	long int h;
	int trace_po, trace_so;

	if (f_v) {
		cout << "semifield_lifting::upstep_loop_over_down_set" << endl;
		cout << "semifield_lifting::upstep_loop_over_down_set Mtx:" << endl;
		Int_matrix_print(Mtx, level, k2);
	}


	for (h = 0; h < N; h++) {

		if (f_vv) {
			cout << "Level " << level << ": flag orbit "
					<< f << " / " << nb_flag_orbits
					<< " coset " << h << " / " << N << endl;
		}
		if (f_vvv) {
			cout << "semifield_lifting::upstep_loop_over_down_set "
					"before Gr->unrank_lint_here_and_extend_basis" << endl;
		}

		Gr->unrank_lint_here_and_extend_basis(
				base_change_matrix, h,
				0 /*verbose_level*/);
		if (f_vvv) {
			cout << "semifield_lifting::upstep_loop_over_down_set "
					"base_change_matrix=" << endl;
			Int_matrix_print(base_change_matrix, level, level);
		}
		if (f_vvv) {
			cout << "semifield_lifting::upstep_loop_over_down_set "
					"Mtx:" << endl;
			Int_matrix_print(Mtx, level, k2);
		}
		if (f_vvv) {
			cout << "semifield_lifting::upstep_loop_over_down_set "
					"before SC->F->mult_matrix_matrix" << endl;
		}
		SC->Mtx->GFq->Linear_algebra->mult_matrix_matrix(base_change_matrix,
				Mtx, changed_space, level, level, k2,
				0 /* verbose_level */);
		if (f_vvv) {
			cout << "Mtx:" << endl;
			Int_matrix_print(Mtx, level, k2);
			cout << "changed_space:" << endl;
			Int_matrix_print(changed_space, level, k2);
		}
		for (i = 0; i < level; i++) {
			if (f_vvv) {
				cout << "i=" << i << " / " << level << endl;
				Int_matrix_print(changed_space + i * k2, k, k);
			}
			set[i] = SC->matrix_rank(changed_space + i * k2);
		}
		if (f_vvv) {
			cout << "Level " << level << ": flag orbit "
					<< f << " / " << nb_flag_orbits
					<< " coset " << h << " / " << N << " set: ";
			Lint_vec_print(cout, set, level);
			cout << " before trace_very_general" << endl;
		}

		trace_very_general(
			changed_space,
			level,
			//changed_space_after_trace,
			transporter,
			trace_po, trace_so,
			verbose_level - 3);

		if (f_vv) {
			cout << "Level " << level << ": flag orbit "
					<< f << " / " << nb_flag_orbits
					<< " coset " << h << " / " << N << " after trace_very_general "
					<< " trace_po = " << trace_po
					<< " trace_so = " << trace_so << endl;
		}

		if (trace_po == po && trace_so == so) {
			if (f_vv) {
				cout << "Level " << level
						<< ", we found an automorphism" << endl;
			}
			Aut[h] = NEW_int(SC->A->elt_size_in_int);
			SC->A->Group_element->element_move(transporter, Aut[h], 0);
			//test_automorphism(Aut[h], level,
			//		changed_space, verbose_level - 3);
		}
		else {
			Aut[h] = NULL;
			int mo;

			mo = Downstep_nodes[trace_po].first_flag_orbit + trace_so;

			if (f_vv) {
				cout << "Level " << level << ": flag orbit "
						<< f << " / " << nb_flag_orbits
						<< " coset " << h << " / " << N << " we will install a "
						"fusion node from " << mo << "=" << trace_po
						<< "/" << trace_so << " to " << f
						<< "=" << po << "/" << so << endl;
			}
			// install a fusion node:

			if (Flag_orbits[mo].fusion_elt) {
				FREE_int(Flag_orbits[mo].fusion_elt);
			}
			Flag_orbits[mo].f_fusion_node = true;
			Flag_orbits[mo].fusion_with = f;
			Flag_orbits[mo].fusion_elt = NEW_int(SC->A->elt_size_in_int);
			SC->A->Group_element->element_invert(
					transporter,
					Flag_orbits[mo].fusion_elt, 0);

		}
	}


	if (f_v) {
		cout << "semifield_lifting::upstep_loop_over_down_set done" << endl;
	}

}

void semifield_lifting::find_all_candidates(
	int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::find_all_candidates = " << level << endl;
	}

	if (level == 2 && cur_level == 3) {
		if (f_v) {
			cout << "semifield_lifting::find_all_candidates "
					"before find_all_candidates_at_level_two" << endl;
		}
		L2->find_all_candidates_at_level_two(verbose_level);
		prev_level_nb_orbits = L2->nb_orbits;
		Candidates = L2->Candidates;
		Nb_candidates = L2->Nb_candidates;

	}
	else {
		cout << "semifield_lifting::find_all_candidates "
				"level = " << level << " nyi" << endl;
		exit(1);
	}

#if 0
	else if (level == 3) {
		if (f_v) {
			cout << "semifield_lifting::find_all_candidates "
					"before find_all_candidates_at_level_three" << endl;
		}
		find_all_candidates_at_level_three(verbose_level);
		Nb_candidates = Level_three_Nb_candidates;
	}
	else if (level == 4) {
		if (f_v) {
			cout << "semifield_lifting::find_all_candidates "
					"before find_all_candidates_at_level_four" << endl;
		}
		find_all_candidates_at_level_four(
				Level_four_Candidates, Level_four_Nb_candidates,
				verbose_level);
		Nb_candidates = Level_four_Nb_candidates;
	}
#endif

	if (f_v) {
		other::data_structures::tally C;

		C.init(Nb_candidates, prev_level_nb_orbits, false, 0);
		cout << "semifield_lifting::find_all_candidates "
				"level " << level
				<< " distribution of number of candidates: ";
		C.print(true /* f_backwards */);
	}

	if (f_v) {
		cout << "semifield_lifting::find_all_candidates "
				"level " << level << " done" << endl;
	}
}


void semifield_lifting::get_basis(
	int po3, int *basis,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int po, so, mo;
	long int pt;
	long int a;

	if (f_v) {
		cout << "semifield_lifting::get_basis "
				"po3 = " << po3 << endl;
	}

	SC->Mtx->GFq->Linear_algebra->identity_matrix(basis, k);

	po = Po[po3];
	so = So[po3];
	mo = Mo[po3];
	pt = Pt[po3];

	if (f_vv) {
		cout << "po=" << po << " so=" << so << " mo=" << mo
				<< " pt=" << pt << endl;
	}

#if 0
	ext = L2->up_orbit_rep[po];
	idx = L2->down_orbit_classes[ext * 2 + 0];
	a = L2->class_rep_rank[idx];
#else
	a = L2->Pt[po];
#endif

	SC->matrix_unrank(a, basis + 1 * k2);

	SC->matrix_unrank(pt, basis + 2 * k2);

#if 0
	pivots[0] = 0;
	pivots[1] = k;
	for (i = k - 1; i >= 2; i--) { // for (i = 2; i < k; i++)
		if (basis[2 * k2 + i * k + 0]) {
			pivots[2] = i * k;
			break;
		}
	}
	if (i == k) {
		cout << "Could not find pivot element" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "semifield_lifting::get_basis_and_pivots "
				"Basis:" << endl;
		int_matrix_print(basis, 3, k2);
		cout << "semifield_lifting::get_basis_and_pivots "
				"pivots: ";
		int_vec_print(cout, pivots, 3);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "semifield_lifting::get_basis_and_pivots "
				"po=" << po << " done" << endl;
	}
}



groups::strong_generators *semifield_lifting::get_stabilizer_generators(
	int level, int orbit_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_lifting::get_stabilizer_generators "
				"level = " << level << " orbit_idx=" << orbit_idx << endl;
	}

	if (level == 2 && cur_level == 3) {
		if (f_v) {
			cout << "semifield_lifting::get_stabilizer_generators "
					"before find_all_candidates_at_level_two" << endl;
		}
		if (orbit_idx >= L2->nb_orbits) {
			cout << "semifield_lifting::get_stabilizer_generators "
					"orbit_idx >= L2->nb_orbits" << endl;
			exit(1);
		}
		return &L2->Stabilizer_gens[orbit_idx];
	}
	cout << "semifield_lifting::get_stabilizer_generators "
			"level not yet implemented" << endl;
	exit(1);
}

int semifield_lifting::trace_to_level_three(
	int *input_basis, int basis_sz, int *transporter,
	int &trace_po,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int trace_po;
	int trace_so;
	int *Elt1;
	//int *basis_tmp;
	int ret;


	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three" << endl;
	}
	Elt1 = NEW_int(SC->A->elt_size_in_int);
	//basis_tmp = NEW_int(basis_sz * k2);

	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three "
				"before trace_very_general" << endl;
	}
	trace_very_general(
		input_basis, basis_sz, /*basis_tmp,*/ transporter,
		trace_po, trace_so,
		verbose_level);
	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three "
				"after trace_very_general "
				"trace_po=" << trace_po
				<< " trace_so=" << trace_so << endl;
	}

	if (f_vv) {
		cout << "semifield_lifting::trace_to_level_three "
				"reduced basis=" << endl;
		Int_matrix_print(input_basis, basis_sz, k2);
		cout << "Which is:" << endl;
		SC->basis_print(input_basis, basis_sz);
	}

	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three "
				"before trace_step_up" << endl;
	}
	ret = trace_step_up(
		trace_po, trace_so,
		input_basis, basis_sz, basis_tmp,
		transporter, Elt1,
		verbose_level);
	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three "
				"after trace_step_up" << endl;
	}
	if (ret == false) {
		cout << "semifield_lifting::trace_to_level_three "
				"trace_step_up return false" << endl;

	}

	FREE_int(Elt1);
	//FREE_int(basis_tmp);

	if (f_v) {
		cout << "semifield_lifting::trace_to_level_three "
				"done" << endl;
	}
	return ret;
}

int semifield_lifting::trace_step_up(
	int &po, int &so,
	int *changed_basis, int basis_sz, int *basis_tmp,
	int *transporter, int *ELT3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int fo, f0;
	int i, j;
	int ret = true;

	if (f_v) {
		cout << "semifield_lifting::trace_step_up po=" << po << " so=" << so << endl;
		cout << "semifield_lifting::trace_step_up "
				"Downstep_nodes[po].first_flag_orbit="
				<< Downstep_nodes[po].first_flag_orbit << endl;
	}
	fo = Downstep_nodes[po].first_flag_orbit + so;
	if (f_vv) {
		cout << "semifield_lifting::trace_step_up "
				"fo = " << fo << endl;
	}
	if (Flag_orbits[fo].f_fusion_node) {
		if (f_vv) {
			cout << "semifield_lifting::trace_step_up "
					"fusion node" << endl;
		}
		f0 = Flag_orbits[fo].fusion_with;
		SC->A->Group_element->element_mult(transporter,
				Flag_orbits[fo].fusion_elt,
				ELT3,
				0 /* verbose_level */);
		SC->A->Group_element->element_move(ELT3, transporter, 0);
		SC->apply_element_and_copy_back(Flag_orbits[fo].fusion_elt,
			changed_basis, basis_tmp,
			0, basis_sz, verbose_level);
		if (f_vv) {
			cout << "semifield_lifting::trace_step_up "
					"after fusion:" << endl;
			Int_matrix_print(changed_basis, basis_sz, k2);
			cout << "Which is:" << endl;
			SC->basis_print(changed_basis, basis_sz);
		}

	}
	else {
		f0 = fo;
	}
	if (f_vv) {
		cout << "semifield_lifting::trace_step_up "
				"f0 = " << f0 << endl;
	}
	po = Flag_orbits[f0].upstep_orbit;
	if (f_vv) {
		cout << "semifield_lifting::trace_step_up "
				"po = " << po << endl;
	}
	if (po == -1) {
		cout << "semifield_lifting::trace_step_up "
				"po == -1" << endl;
		exit(1);
	}

#if 0
	int *pivots;

	pivots = NEW_int(cur_level);
	get_pivots(3 /* level */,
			Flag_orbits[f0].upstep_orbit,
			pivots, changed_basis, verbose_level);

	if (f_vv) {
		cout << "semifield_lifting::trace_step_up "
				"pivots=";
		int_vec_print(cout, pivots, 3);
		cout << endl;
	}
#endif


	if (f_vv) {
		cout << "semifield_lifting::trace_step_up "
				"before Gauss_int_with_given_pivots" << endl;
		Int_matrix_print(changed_basis, basis_sz, k2);
	}
	if (!SC->Mtx->GFq->Linear_algebra->Gauss_int_with_given_pivots(
		changed_basis,
		false /* f_special */,
		true /* f_complete */,
		SC->desired_pivots,
		3 /* nb_pivots */,
		3, //basis_sz  m
		k2 /* n */,
		verbose_level)) {
		if (f_vv) {
			cout << "semifield_lifting::trace_step_up "
					"Gauss_int_with_given_pivots returns false, "
					"pivot cannot be found" << endl;
			Int_matrix_print(changed_basis, basis_sz, k2);
		}
		ret = false;
	}
	else {
		if (f_vv) {
			cout << "semifield_lifting::trace_step_up "
					"after Gauss_int_with_given_pivots:" << endl;
			Int_matrix_print(changed_basis, basis_sz, k2);
		}
		for (i = 0; i < 3; i++) {
			for (j = 3; j < basis_sz; j++) {
				SC->Mtx->GFq->Linear_algebra->Gauss_step(changed_basis + i * k2,
						changed_basis + j * k2, k2,
						SC->desired_pivots[i], 0 /*verbose_level*/);
			}
		}
		if (f_vv) {
			cout << "semifield_lifting::trace_step_up "
					"after reducing:" << endl;
			Int_matrix_print(changed_basis, basis_sz, k2);
			SC->basis_print(changed_basis, basis_sz);
		}
	}

	//FREE_int(pivots);
	if (f_v) {
		cout << "semifield_lifting::trace_step_up done" << endl;
	}
	return ret;
}



void semifield_lifting::trace_very_general(
	int *input_basis, int basis_sz,
	int *transporter,
	int &trace_po, int &trace_so,
	int verbose_level)
// input basis is input_basis of size basis_sz x k2
// there is a check if input_basis defines a semifield
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, j;
	actions::action *A;
	algebra::field_theory::finite_field *F;

	if (f_v) {
		cout << "semifield_lifting::trace_very_general" << endl;
	}
	if (cur_level != 3) {
		cout << "semifield_lifting::trace_very_general cur_level != 3" << endl;
		exit(1);
	}
	A = SC->A;
	F = SC->Mtx->GFq;
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"input basis:" << endl;
		SC->basis_print(input_basis, basis_sz);
	}
	if (!SC->test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "semifield_lifting::trace_very_general "
				"before trace_to_level_two" << endl;
	}
	trace_to_level_two(
		input_basis, basis_sz,
		/*basis_after_trace,*/ transporter,
		trace_po,
		verbose_level);
	if (f_v) {
		cout << "semifield_lifting::trace_very_general "
				"after trace_to_level_two" << endl;
		cout << "trace_po=" << trace_po << endl;
	}


	// Step 2 almost finished.
	// Next we need to compute the reduced coset representatives
	// for the remaining elements
	// w.r.t. the basis and the pivots in base_col

	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"we will now compute the reduced coset reps:" << endl;
	}

	base_cols[0] = 0;
	base_cols[1] = k;
	if (f_vvv) {
		cout << "semifield_lifting::trace_very_general base_cols=";
		Int_vec_print(cout, base_cols, 2);
		cout << endl;
	}
	for (i = 0; i < 2; i++) {
		for (j = 2; j < basis_sz; j++) {
			F->Linear_algebra->Gauss_step(input_basis + i * k2,
					input_basis + j * k2, k2, base_cols[i],
					0 /*verbose_level*/);
		}
	}
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"reduced basis=" << endl;
		Int_matrix_print(input_basis, basis_sz, k2);
		cout << "Which is:" << endl;
		SC->basis_print(input_basis, basis_sz);
	}
	if (!SC->test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial "
				"semifield condition" << endl;
		exit(1);
	}

	// Step 3:
	// Locate the third matrix, compute its rank,
	// and find the rank in the candidates array to compute the local point.
	// Then find the point in the schreier structure
	// and compute a coset representative.
	// This coset representative stabilizes the subspace which is
	// generated by the first two vectors, so when applying the mapping,
	// we can skip the first two vectors.

	long int a;
	int a_local, pos;

	a = SC->matrix_rank(input_basis + 2 * k2);
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"rank of row 2: a = " << a << endl;
	}

	a_local = Downstep_nodes[trace_po].find_point(a);
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"a_local = " << a_local << endl;
	}

	pos = Downstep_nodes[trace_po].Sch->Forest->orbit_inv[a_local];
	trace_so = Downstep_nodes[trace_po].Sch->Forest->orbit_number(a_local);
		// Level_two_down[trace_po].Sch->orbit_no[pos];

	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"trace_so = " << trace_so << endl;
	}

	Downstep_nodes[trace_po].Sch->Generators_and_images->coset_rep_inv(
			pos, 0 /* verbose_level */);
	A->Group_element->element_mult(transporter,
			Downstep_nodes[trace_po].Sch->Generators_and_images->cosetrep,
			ELT3,
			0 /* verbose_level */);
	A->Group_element->element_move(ELT3, transporter, 0);
	// apply cosetrep to base elements 2,...,basis_sz - 1:
	SC->apply_element_and_copy_back(
			Downstep_nodes[trace_po].Sch->Generators_and_images->cosetrep,
		input_basis, basis_tmp,
		2, basis_sz, verbose_level);
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"after transforming with cosetrep from "
				"secondary orbit (4):" << endl;
		SC->basis_print(input_basis, basis_sz);
	}
	base_cols[0] = 0;
	base_cols[1] = k;
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"base_cols=";
		Int_vec_print(cout, base_cols, 2);
		cout << endl;
	}
	for (i = 0; i < 2; i++) {
		for (j = 2; j < basis_sz; j++) {
			F->Linear_algebra->Gauss_step(input_basis + i * k2,
					input_basis + j * k2, k2,
					base_cols[i],
					0 /*verbose_level*/);
		}
	}
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"reduced basis(2)=" << endl;
		Int_matrix_print(input_basis, basis_sz, k2);
		cout << "Which is:" << endl;
		SC->basis_print(input_basis, basis_sz);
	}

#if 0
	if (!test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "semifield_lifting::trace_very_general done" << endl;
	}

}


void semifield_lifting::trace_to_level_two(
	int *input_basis, int basis_sz,
	int *transporter,
	int &trace_po,
	int verbose_level)
// input basis is input_basis of size basis_sz x k2
// there is a check if input_basis defines a semifield
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, j, idx, d, d0, c0, c1;
	actions::action *A;
	algebra::field_theory::finite_field *F;

	if (f_v) {
		cout << "semifield_lifting::trace_very_general" << endl;
	}
	if (cur_level != 3) {
		cout << "semifield_lifting::trace_very_general cur_level != 3" << endl;
		exit(1);
	}
	A = SC->A;
	F = SC->Mtx->GFq;
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"input basis:" << endl;
		SC->basis_print(input_basis, basis_sz);
	}
	if (!SC->test_partial_semifield(input_basis,
			basis_sz, 0 /* verbose_level */)) {
		cout << "does not satisfy the partial semifield condition" << endl;
		exit(1);
	}




	// Step 1:
	// trace the first matrix (which becomes the identity matrix):

	// create the n x n matrix which is a 2 x 2 block matrix
	// (A 0)
	// (0 I)
	// where A is input_basis
	// the resulting matrix will be put in transporter
	Int_vec_zero(M1, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			M1[i * n + j] = input_basis[i * k + j];
		}
	}
	for (i = k; i < n; i++) {
		M1[i * n + i] = 1;
	}
	A->Group_element->make_element(transporter, M1, 0);

	if (f_vvv) {
		cout << "transformation matrix transporter=" << endl;
		Int_matrix_print(transporter, n, n);
		cout << "transformation matrix M1=" << endl;
		Int_matrix_print(M1, n, n);
	}

	// apply transporter to elements 0,...,basis_sz - 1 of input_basis
	SC->apply_element_and_copy_back(transporter,
		input_basis, basis_tmp,
		0, basis_sz, verbose_level);
	if (f_v) {
		cout << "semifield_lifting::trace_very_general "
				"after transform (1):" << endl;
		SC->basis_print(input_basis, basis_sz);
	}
	if (!F->Linear_algebra->is_identity_matrix(input_basis, k)) {
		cout << "semifield_lifting::trace_very_general "
				"basis_tmp is not the identity matrix" << endl;
		exit(1);
	}

	// Now the first matrix has been transformed to the identity matrix.
	// The other matrices have been transformed as well.


	// Step 2:
	// Trace the second matrix using rational normal forms.
	// Do adjustment to get the right coset representative.
	// Apply fusion element if necessary

	L2->C->identify_matrix(
			input_basis + 1 * k2,
			R1,
			Basis, Rational_normal_form,
			0 /* verbose_level */);

	idx = L2->C->find_class_rep(L2->R,
			L2->nb_classes, R1,
			0 /* verbose_level */);
		// idx is the conjugacy class, which leads to the flag orbit

	d = L2->class_to_flag_orbit[idx];

		// d is the flag orbit associated to the conjugacy class.


	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"the second matrix belongs to conjugacy class "
				<< idx << " which is in down orbit " << d << endl;
	}

	L2->multiply_to_the_right(transporter,
			Basis, ELT2, ELT3,
			0 /* verbose_level */);

	// Applies the matrix diag(Basis, Basis) from the right to transporter
	// and puts the result into ELT3.
	// ELT2 = diag(Basis, Basis).


	A->Group_element->element_move(ELT3, transporter, 0);

	// apply ELT2 (i.e., Basis)
	// to input_basis elements 1, .., basis_sz - 1
	SC->apply_element_and_copy_back(ELT2,
		input_basis, basis_tmp,
		1, basis_sz, verbose_level);
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"after transform (2):" << endl;
		SC->basis_print(input_basis, basis_sz);
	}

	// now the second matrix is the flag orbit representative.
	// The other matrices have been transformed.
	//


	c0 = L2->flag_orbit_classes[d * 2 + 0];
	c1 = L2->flag_orbit_classes[d * 2 + 1];

	// { c0, c1 } is the two conjugacy classes
	// associated with flag orbit d (possibly c0 = c1).
	// At least one of them has to equal idx.
	// If it is the second, we have to apply one more transformation.

	if (c0 != idx && c1 == idx) {

		// if the conjugacy class is the second element in the orbit,
		// we need to apply class_rep_plus_I_Basis_inv[c0]:

		if (f_vv) {
			cout << "Adjusting" << endl;
		}
		L2->multiply_to_the_right(transporter,
				L2->class_rep_plus_I_Basis_inv[c0],
				ELT2, ELT3,
				0 /* verbose_level */);
		A->Group_element->element_move(ELT3, transporter, 0);

		// apply ELT2 to the basis elements 1,...,basis_sz - 1:
		SC->apply_element_and_copy_back(ELT2,
			input_basis, basis_tmp,
			1, basis_sz, verbose_level);
		if (f_vvv) {
			cout << "semifield_lifting::trace_very_general "
					"after transform because of adjustment:" << endl;
			SC->basis_print(input_basis, basis_sz);
		}
		// subtract the first matrix (the identity) off the second matrix
		// so that the second matrix has a zero in the top left position.
		for (i = 0; i < k2; i++) {
			input_basis[1 * k2 + i] = F->add(
					input_basis[1 * k2 + i],
					F->negate(input_basis[i]));
		}
		if (f_vvv) {
			cout << "semifield_lifting::trace_very_general "
					"after subtracting the identity:" << endl;
			SC->basis_print(input_basis, basis_sz);
		}
	}
	else {
		if (f_vv) {
			cout << "No adjustment needed" << endl;
		}
	}

	if (L2->f_Fusion[d]) {
		if (f_vv) {
			cout << "Applying fusion element" << endl;
		}
		if (L2->Fusion_elt[d] == NULL) {
			cout << "Fusion_elt[d] == NULL" << endl;
			exit(1);
		}
		d0 = L2->Fusion_idx[d];
		A->Group_element->element_mult(transporter, L2->Fusion_elt[d], ELT3, 0);
		//multiply_to_the_right(transporter, Fusion_elt[d],
		//ELT2, ELT3, 0 /* verbose_level */);
		A->Group_element->element_move(ELT3, transporter, 0);

		// apply Fusion_elt[d] to the basis elements 0,1,...,basis_sz - 1
		SC->apply_element_and_copy_back(
				L2->Fusion_elt[d],
				input_basis, basis_tmp,
				0, basis_sz,
				verbose_level);
		if (f_vvv) {
			cout << "semifield_lifting::trace_very_general "
					"after transform (3):" << endl;
			SC->basis_print(input_basis, basis_sz);
		}
		if (input_basis[0] == 0) {
			// add the second matrix to the first:
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->add(
						input_basis[j], input_basis[k2 + j]);
			}
		}
		// now, input_basis[0] != 0
		if (input_basis[0] == 0) {
			cout << "input_basis[0] == 0" << endl;
			exit(1);
		}
		if (input_basis[0] != 1) {
			int lambda;

			lambda = F->inverse(input_basis[0]);
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->mult(input_basis[j], lambda);
			}
		}
		if (input_basis[0] != 1) {

			cout << "input_basis[0] != 1" << endl;
			exit(1);
		}
		if (input_basis[k2]) {
			int lambda;
			lambda = F->negate(input_basis[k2]);
			for (j = 0; j < k2; j++) {
				input_basis[k2 + j] = F->add(
						input_basis[k2 + j],
						F->mult(input_basis[j], lambda));
			}
		}
		if (input_basis[k]) {
			int lambda;
			lambda = F->negate(input_basis[k]);
			for (j = 0; j < k2; j++) {
				input_basis[j] = F->add(
						input_basis[j],
						F->mult(input_basis[k2 + j], lambda));
			}
		}
		if (input_basis[k]) {
			cout << "input_basis[k] (should be zero by now)" << endl;
			exit(1);
		}
		if (f_vvv) {
			cout << "semifield_lifting::trace_very_general "
					"after gauss elimination:" << endl;
			SC->basis_print(input_basis, basis_sz);
		}
	}
	else {
		if (f_vv) {
			cout << "No fusion" << endl;
		}
		d0 = d;
	}
	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"d0 = " << d0 << endl;
	}
	if (L2->Fusion_elt[d0]) {
		cout << "Fusion_elt[d0]" << endl;
		exit(1);
	}



	trace_po = L2->Fusion_idx[d0];

	// trace_po = the level 2 orbit associated with the flag orbit d0


	if (f_vv) {
		cout << "semifield_lifting::trace_very_general "
				"trace_po = " << trace_po << endl;
	}





	if (f_v) {
		cout << "semifield_lifting::trace_very_general done" << endl;
	}
}


void semifield_lifting::deep_search(
	int orbit_r, int orbit_m,
	int f_out_path, std::string &out_path,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::deep_search "
				"orbit_r=" << orbit_r
				<< " orbit_m=" << orbit_m << endl;
	}


	L2->allocate_candidates_at_level_two(verbose_level);

	if (f_v) {
		cout << "semifield_lifting::deep_search "
				"before read_level_info_file" << endl;
	}
	read_level_info_file(verbose_level);
	if (f_v) {
		cout << "semifield_lifting::deep_search "
				"after read_level_info_file" << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::deep_search "
				"before deep_search_at_level_three" << endl;
	}

	int nb_sol;

	deep_search_at_level_three(orbit_r, orbit_m,
			f_out_path, out_path,
			nb_sol,
			verbose_level);
	if (f_v) {
		cout << "semifield_lifting::deep_search "
				"after deep_search_at_level_three "
				"nb_sol=" << nb_sol << endl;
	}

	if (f_v) {
		cout << "semifield_lifting::deep_search "
				" orbit_r=" << orbit_r
				<< " orbit_m=" << orbit_m << " done" << endl;
	}

}

void semifield_lifting::deep_search_at_level_three(
	int orbit_r, int orbit_m,
	int f_out_path, std::string &out_path,
	int &nb_sol,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit;
	int *Basis;
	//int *pivots;
	int i;
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_lifting::deep_search_at_level_three" << endl;
	}

	nb_sol = 0;
	make_fname_deep_search_slice_solutions(fname,
			f_out_path, out_path, orbit_r, orbit_m);


	{
		ofstream fp(fname);


		Basis = NEW_int(k * k2);
		//pivots = NEW_int(k2);


		if (f_v) {
			cout << "semifield_lifting::deep_search_at_level_three "
				"Level_three_nb_orbits = " << nb_orbits << endl;
		}

		for (orbit = 0; orbit < nb_orbits; orbit++) {

			if ((orbit % orbit_m) != orbit_r) {
				continue;
			}

			cout << "semifield_lifting::deep_search_at_level_three orbit "
				<< orbit << " / " << nb_orbits << ":" << endl;

			for (i = 0; i < 3; i++) {
				cout << "matrix " << i << ":" << endl;
				Int_matrix_print(Basis + i * k2, k, k);
			}
			cout << "pivots: ";
			Int_vec_print(cout, SC->desired_pivots, 3);
			cout << endl;


			if (k == 6) {
				deep_search_at_level_three_orbit(orbit,
					Basis, SC->desired_pivots, fp, nb_sol, verbose_level);
			}
#if 0
			else if (k == 5) {
				deep_search_at_level_three_orbit_depth2(orbit,
					Basis, pivots, fp, verbose_level);
			}
#endif

		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	make_fname_deep_search_slice_success(fname,
			f_out_path, out_path, orbit_r, orbit_m);
	{
		ofstream fp(fname);
		fp << "Case " << orbit_r << " mod " << orbit_m
				<< " is done" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	FREE_int(Basis);
	//FREE_int(pivots);
	if (f_v) {
		cout << "semifield_lifting::deep_search_at_level_three "
				"done" << endl;
	}
}

void semifield_lifting::print_stabilizer_orders()
{
	//long int *Go;
	//int i;
#if 0
	Go = NEW_lint(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		Go[i] = Stabilizer_gens[i].group_order_as_lint();
		}
#endif
	other::data_structures::tally C;

	C.init_lint(Go, nb_orbits, false, 0);
	cout << "distribution of stabilizer orders at level " << cur_level << " : ";
	C.print(true /* f_backwards */);
	//FREE_lint(Go);
}

void semifield_lifting::deep_search_at_level_three_orbit(
	int orbit, int *Basis, int *pivots,
	ofstream &fp,
	int &nb_sol,
	int verbose_level)
// deep search for levels three to six
// this function is called from
// semifield_starter::deep_search_at_level_three
// in semifield_starter_level_three.cpp
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int f_v4 = false; //(verbose_level >= 3);
	int po;
	long int a1, a2, a3;
	int cur_pivot_row;
	int u;
	other::data_structures::set_of_sets_lint *C3; // Level two candidates sorted by type
	other::data_structures::set_of_sets_lint *C4; // those that are compatible with A_3
	other::data_structures::set_of_sets_lint *C5; // those that are compatible with A_4
	other::data_structures::set_of_sets_lint *C6; // those that are compatible with A_5


	if (f_v) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				<< orbit << " / " << nb_orbits << endl;
	}

	cur_pivot_row = pivots[2] / k;
	if (f_vv) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				"cur_pivot_row = " << cur_pivot_row << endl;
	}

	if (cur_pivot_row != k - 1) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				"cur_pivot_row != k - 1" << endl;

		fp << "start orbit " << orbit << endl;
		fp << "finish orbit " << orbit << " " << nb_sol << endl;

		if (f_v) {
			cout << "semifield_lifting::deep_search_at_level_three_orbit "
					<< orbit << " / " << nb_orbits
					<< " skipped because pivot is not in the last row, "
							"nb_sol = " << nb_sol << endl;
		}

		return;
		//exit(1);
	}


	fp << "start orbit " << orbit << endl;

	level_three_get_a1_a2_a3(orbit, a1, a2, a3, verbose_level);

	po = Po[orbit];




	if (f_v) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				<< orbit << " / " << nb_orbits
				<< " reading candidates by type, po = " << po << endl;
	}
	L2->read_candidates_at_level_two_by_type(C3, po, verbose_level - 2);
		// semifield_starter_level_two.cpp
		// reads the files "C2_orbit%d_type%d_int4.bin"
		// this function allocates C3

	C4 = NEW_OBJECT(other::data_structures::set_of_sets_lint);
	C5 = NEW_OBJECT(other::data_structures::set_of_sets_lint);
	C6 = NEW_OBJECT(other::data_structures::set_of_sets_lint);

	long int underlying_set_size;
	int max_l = 0;
	long int *Tmp1;
	long int *Tmp2;
	algebra::number_theory::number_theory_domain NT;

	underlying_set_size = NT.i_power_j(SC->q, k2);

	C4->init_simple(underlying_set_size, NT.i_power_j(SC->q, k - 3),
			0 /*verbose_level - 2*/);
	C5->init_simple(underlying_set_size, NT.i_power_j(SC->q, k - 4),
			0 /*verbose_level - 2*/);
	C6->init_simple(underlying_set_size, NT.i_power_j(SC->q, k - 5),
			0 /*verbose_level - 2*/);

	for (u = 0; u < C3->nb_sets; u++) {
		max_l = MAXIMUM(max_l, C3->Set_size[u]);
	}
	Tmp1 = NEW_lint(max_l);
	Tmp2 = NEW_lint(max_l);


	for (u = 0; u < C4->nb_sets; u++) {
		C4->init_set(u, C3->Sets[2 * u], C3->Set_size[2 * u],
				0 /*verbose_level*/);
	}
	for (u = 0; u < C5->nb_sets; u++) {
		C5->init_set(u, C4->Sets[2 * u], C4->Set_size[2 * u],
				0 /*verbose_level*/);
	}
	for (u = 0; u < C6->nb_sets; u++) {
		C6->init_set(u, C5->Sets[2 * u], C5->Set_size[2 * u],
				0 /*verbose_level*/);
	}


	if (f_vv) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				"computing candidates C4" << endl;
	}

	candidate_testing(orbit,
		Basis + 2 * k2, k - 1, k - 2,
		C3, C4,
		Tmp1, Tmp2,
		verbose_level - 1);
		// in deep_search.cpp

	if (f_vv) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				"computing candidates C4 done" << endl;
	}



	int c4;
	long int a4;
	long int A4;
	int nb_sol0;


	for (c4 = 0; c4 < C4->Set_size[1]; c4++) {



		nb_sol0 = nb_sol;

		a4 = C4->Sets[1][c4];
		SC->matrix_unrank(a4, Basis + 3 * k2);

		// put the pivot element:
		Basis[3 * k2 + (k - 2) * k + 0] = 1;

		A4 = SC->matrix_rank(Basis + 3 * k2);

		if (f_v3) {
			cout << "Level 3, orbit " << orbit << " / "
					<< nb_orbits << " level 4 case "
					<< c4 << " / " << C4->Set_size[1] << " is matrix" << endl;
			Int_matrix_print(Basis + 3 * k2, k, k);
		}

		pivots[3] = (k - 2) * k;




		if (f_v4) {
			cout << "semifield_lifting::deep_search_at_level_three_orbit "
					"computing candidates C5" << endl;
		}

		if (!candidate_testing(orbit,
			Basis + 3 * k2, k - 2, k - 3,
			C4, C5,
			Tmp1, Tmp2,
			0 /*verbose_level - 1*/)) {
			goto done4;
		}

		if (f_v4) {
			cout << "semifield_lifting::deep_search_at_level_three_orbit "
					"computing candidates C5 done" << endl;
		}


		int c5;
		long int a5;
		long int A5;
		for (c5 = 0; c5 < C5->Set_size[1]; c5++) {


			a5 = C5->Sets[1][c5];
			SC->matrix_unrank(a5, Basis + 4 * k2);

			// put the pivot element:
			Basis[4 * k2 + (k - 3) * k + 0] = 1;

			A5 = SC->matrix_rank(Basis + 4 * k2);

			if (f_v3) {
				cout << "Level 3, orbit " << orbit << " / "
						<< nb_orbits << " level 4 case "
						<< c4 << " / " << C4->Set_size[1] << " level 5 case "
						<< c5 << " / " << C5->Set_size[1]
						<< " is matrix" << endl;
				Int_matrix_print(Basis + 4 * k2, k, k);
			}

			pivots[4] = (k - 3) * k;



			if (f_v4) {
				cout << "semifield_lifting::deep_search_at_level_"
						"three_orbit computing candidates C6" << endl;
			}

			if (!candidate_testing(orbit,
				Basis + 4 * k2, k - 3, k - 4,
				C5, C6,
				Tmp1, Tmp2,
				0 /*verbose_level - 1*/)) {
				continue;
			}

			if (f_v4) {
				cout << "semifield_lifting::deep_search_at_level_"
						"three_orbit computing candidates C6 done" << endl;
			}

			int c6;
			long int a6;
			long int A6;
			for (c6 = 0; c6 < C6->Set_size[1]; c6++) {


				a6 = C6->Sets[1][c6];
				SC->matrix_unrank(a6, Basis + 5 * k2);

				// put the pivot element:
				Basis[5 * k2 + (k - 4) * k + 0] = 1;

				A6 = SC->matrix_rank(Basis + 5 * k2);

				fp << "SOL " << orbit << " " << c4 << " " << c5 << " "
						<< c6 << " " << a1 << " " << a2 << " " << a3
						<< " " << A4 << " " << A5 << " " << A6 << endl;
				cout << "SEMIFIELD " << nb_sol << " : " << a1 << ", "
						<< a2 << ", " << a3 << ", " << A4 << ", "
						<< A5 << ", " << A6 << endl;

#if 0
				long int A[6];

				A[0] = a1;
				A[1] = a2;
				A[2] = a3;
				A[3] = A4;
				A[4] = A5;
				A[5] = A6;
#endif
				//compute_automorphism_group(6, 3, orbit, A, verbose_level);
				nb_sol++;

			}



		} // next c5


done4:

		if (f_v && ((c4 % 1000) == 0)) {
			cout << "Level 3, orbit " << orbit << " / "
					<< nb_orbits << ", L4 case " << c4
					<< " / " << C4->Set_size[1] << " done, "
					"yields nb_sol = " << nb_sol - nb_sol0
					<< " solutions, nb_sol = " << nb_sol << endl;
		}

	} // next c4




	fp << "finish orbit " << orbit << " " << nb_sol << endl;





	FREE_OBJECT(C3);
	FREE_OBJECT(C4);
	FREE_OBJECT(C5);
	FREE_OBJECT(C6);

	FREE_lint(Tmp1);
	FREE_lint(Tmp2);


	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_lifting::deep_search_at_level_three_orbit "
				<< orbit << " / " << nb_orbits << " done, "
						"nb_sol = " << nb_sol << endl;
	}
}

int semifield_lifting::candidate_testing(
	int orbit,
	int *last_mtx, int window_bottom, int window_size,
	other::data_structures::set_of_sets_lint *C_in,
	other::data_structures::set_of_sets_lint *C_out,
	long int *Tmp1, long int *Tmp2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u, v, w;
	int h, l;
	int Nb_sets_new;
	int /*window_bottom_new,*/ window_size_new;
	//int pivot_row;
	long int last_mtx_numeric;
	algebra::number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "semifield_lifting::candidate_testing" << endl;
	}

	last_mtx_numeric = SC->matrix_rank_without_first_column(last_mtx);


	//pivot_row = window_bottom;

	//window_bottom_new = pivot_row - 1;
	window_size_new = window_size - 1;

	Nb_sets_new = NT.i_power_j(SC->q, window_size_new);


	for (u = 1; u < Nb_sets_new; u++) {


		v = 2 * u;
		w = v + 1;
		l = C_in->Set_size[v];
		Gg.AG_element_unrank(SC->q, window_in, 1, window_size, v);
		if (f_v) {
			cout << "Level 3, Orbit " << orbit << " / "
					<< nb_orbits << ": testing "
					<< u << " / " << Nb_sets_new << " v=" << v
					<< " w=" << w << " testing " << l
					<< " points, pattern: ";
			Int_vec_print(cout, window_in, window_size);
			cout << endl;
		}


		int set_sz;

		for (h = 0; h < l; h++) {
			Tmp1[h] = C_in->Sets[v][h] ^ last_mtx_numeric;
		}

		Sorting.lint_vec_heapsort(Tmp1, l);

		Sorting.lint_vec_intersect_sorted_vectors(
				Tmp1, l,
				C_in->Sets[w], C_in->Set_size[w],
				Tmp2, set_sz);

		for (h = 0; h < set_sz; h++) {
			Tmp2[h] ^= last_mtx_numeric;
		}
		Sorting.lint_vec_heapsort(Tmp2, set_sz);
		for (h = 0; h < set_sz; h++) {
			C_out->Sets[u][h] = Tmp2[h];
		}
		C_out->Set_size[u] = set_sz;

		if (u && C_out->Set_size[u] == 0) {
			return false;
		}

	} // next u



	if (f_v) {
		cout << "semifield_lifting::candidate_testing "
				"done" << endl;
	}
	return true;
}

void semifield_lifting::level_three_get_a1_a2_a3(
	int po3, long int &a1, long int &a2, long int &a3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int po; // so, mo;
	long int pt, a;
	//int ext, idx;
	int *basis;

	if (f_v) {
		cout << "semifield_lifting::level_three_get_a1_a2_a3 "
				"po3 = " << po3 << endl;
	}
	po = Po[po3];
	//so = So[po3];
	//mo = Mo[po3];
	pt = Pt[po3];


#if 0
	ext = L2->up_orbit_rep[po];
	idx = L2->down_orbit_classes[ext * 2 + 0];
	a = L2->class_rep_rank[idx];
#else
	a = L2->Pt[po];
#endif

	basis = NEW_int(k2);

	SC->Mtx->GFq->Linear_algebra->identity_matrix(basis, k);

	a1 = SC->matrix_rank(basis);

	FREE_int(basis);

	a2 = a;
	a3 = pt;
	if (f_v) {
		cout << "semifield_lifting::level_three_get_a1_a2_a3 "
				" a1 = " << a1
				<< " a2 = " << a2
				<< " a3 = " << a3
				<< endl;
	}
}


void semifield_lifting::write_level_info_file(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_lifting::write_level_info_file "
				"level=" << cur_level << endl;
	}
	int i;
	int nb_vecs = 5;
	const char *column_label[] = {
		"Po",
		"So",
		"Mo",
		"Go",
		"Pt"
		};
	string fname;

	create_fname_level_info_file(fname);

	{
		ofstream f(fname);
		int j;

		f << "Row";
		for (j = 0; j < nb_vecs; j++) {
			f << "," << column_label[j];
			}
		f << endl;
		for (i = 0; i < nb_orbits; i++) {
			f << i;
			f << "," << Po[i]
				<< "," << So[i]
				<< "," << Mo[i]
				<< "," << Go[i]
				<< "," << Pt[i]
				<< endl;
			}
		f << "END" << endl;
	}

	cout << "Written file " << fname << " of size"
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "semifield_lifting::write_level_info_file done" << endl;
	}

}


void semifield_lifting::read_level_info_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	long int *M;
	int m, n, i;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_lifting::read_level_info_file" << endl;
		}
	create_fname_level_info_file(fname);

	cout << "semifield_lifting::read_level_info_file " << fname << endl;

	if (Fio.file_size(fname) <= 0) {
		cout << "semifield_lifting::read_level_info_file "
			"error trying to read the file " << fname << endl;
		exit(1);
		}

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname, M, m, n, 0 /* verbose_level */);
		// Row,Po,So,Mo,Go,Pt

	nb_orbits = m;

	Po = NEW_int(m);
	So = NEW_int(m);
	Mo = NEW_int(m);
	Go = NEW_lint(m);
	Pt = NEW_lint(m);


	for (i = 0; i < m; i++) {
		Po[i] = M[i * n + 1];
		So[i] = M[i * n + 2];
		Mo[i] = M[i * n + 3];
		Go[i] = M[i * n + 4];
		Pt[i] = M[i * n + 5];
		}

	FREE_lint(M);

	if (f_v) {
		cout << "semifield_lifting::read_level_info_file done" << endl;
		}
}

void semifield_lifting::save_flag_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::save_flag_orbits "
				"cur_level = " << cur_level << endl;
	}
	string fname;
	int i;
	other::orbiter_kernel_system::file_io Fio;

	make_fname_flag_orbits(fname);
	{
		ofstream fp(fname, ios::binary);

		fp.write((char *) &nb_flag_orbits, sizeof(int));
		for (i = 0; i < nb_flag_orbits; i++) {
			Flag_orbits[i].write_to_file_binary(this, fp, verbose_level - 1);
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "semifield_lifting::save_flag_orbits "
				"cur_level = " << cur_level << " done" << endl;
	}
}

void semifield_lifting::read_flag_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::read_flag_orbits "
				"cur_level = " << cur_level
				<< endl;
	}
	string fname;
	int i;
	other::orbiter_kernel_system::file_io Fio;


	make_fname_flag_orbits(fname);

	if (Fio.file_size(fname) <= 0) {
		cout << "semifield_lifting::read_flag_orbits "
				"file " << fname << " does not exist" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "semifield_lifting::read_flag_orbits "
				"reading file " << fname << endl;
	}

	{
		ifstream fp(fname, ios::binary);

		fp.read((char *) &nb_flag_orbits, sizeof(int));
		Flag_orbits = NEW_OBJECTS(semifield_flag_orbit_node, nb_flag_orbits);

		for (i = 0; i < nb_flag_orbits; i++) {
			if ((i & ((1 << 15) - 1)) == 0) {
				cout << "semifield_lifting::read_flag_orbits "
						<< i << " / " << nb_flag_orbits << endl;
			}
			Flag_orbits[i].read_from_file_binary(this, fp,
					0 /* verbose_level */);
		}
	}
	cout << "semifield_lifting::read_flag_orbits "
			"Read file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "semifield_lifting::read_flag_orbits "
				"cur_level = " << cur_level << " done, "
						"nb_flag_orbits=" << nb_flag_orbits << endl;
	}
}



void semifield_lifting::save_stabilizers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::save_stabilizers "
				"cur_level = " << cur_level << endl;
	}
	string fname;
	int i;
	other::orbiter_kernel_system::file_io Fio;

	make_fname_stabilizers(fname);
	{
		ofstream fp(fname, ios::binary);

		fp.write((char *) &nb_orbits, sizeof(int));
		for (i = 0; i < nb_orbits; i++) {
			Stabilizer_gens[i].write_to_file_binary(fp, verbose_level - 1);
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "semifield_lifting::save_stabilizers "
				"cur_level = " << cur_level << " done" << endl;
	}
}

void semifield_lifting::read_stabilizers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "semifield_lifting::read_stabilizers "
				"cur_level = " << cur_level << endl;
	}
	string fname;
	int i;
	other::orbiter_kernel_system::file_io Fio;


	make_fname_stabilizers(fname);

	if (Fio.file_size(fname) <= 0) {
		cout << "semifield_lifting::read_stabilizers "
				"file " << fname << " does not exist" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "semifield_lifting::read_stabilizers "
				"reading file " << fname << endl;
	}

	{
		ifstream fp(fname, ios::binary);

		fp.read((char *) &nb_orbits, sizeof(int));
		Stabilizer_gens = NEW_OBJECTS(groups::strong_generators, nb_orbits);

		for (i = 0; i < nb_orbits; i++) {
			if ((i & ((1 << 15) - 1)) == 0) {
				cout << "semifield_starter::read_stabilizers "
						<< i << " / " << nb_orbits << endl;
			}
			Stabilizer_gens[i].read_from_file_binary(SC->A, fp,
					0 /* verbose_level */);
		}
	}
	cout << "semifield_lifting::read_stabilizers "
			"Read file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "semifield_lifting::read_stabilizers "
				"cur_level = " << cur_level << " done" << endl;
	}
}

void semifield_lifting::make_file_name_schreier(
		std::string &fname,
		int level, int orbit_idx)
{
	if (f_prefix) {
		fname = prefix + "L" + std::to_string(level) + "_orbit" + std::to_string(orbit_idx) + "_schreier.csv";
	}
	else {
		fname = "L" + std::to_string(level) + "_orbit" + std::to_string(orbit_idx) + "_schreier.csv";
	}
}

void semifield_lifting::create_fname_level_info_file(
		std::string &fname)
{
	if (f_prefix) {
		fname = prefix + "L" + std::to_string(cur_level) + "_info.csv";
	}
	else {
		fname = "L" + std::to_string(cur_level) + "_info.csv";
	}
}

void semifield_lifting::make_fname_flag_orbits(
		std::string &fname)
{
	if (f_prefix) {
		fname = prefix + "L" + std::to_string(cur_level) + "_flag_orbits.bin";
	}
	else {
		fname = "L" + std::to_string(cur_level) + "_flag_orbits.bin";
	}
}

void semifield_lifting::make_fname_stabilizers(
		std::string &fname)
{
	if (f_prefix) {
		fname = prefix + "L" + std::to_string(cur_level) + "_flag_stabilizers.bin";
	}
	else {
		fname = "L" + std::to_string(cur_level) + "_flag_stabilizers.bin";
	}
}

void semifield_lifting::make_fname_deep_search_slice_solutions(
		std::string &fname,
		int f_out_path, std::string &out_path,
		int orbit_r, int orbit_m)
{
	if (f_out_path) {
		fname = prefix + "deep_slice" + std::to_string(orbit_r) + "_" + std::to_string(orbit_m) + "_sol.txt";
	}
	else {
		fname = "deep_slice" + std::to_string(orbit_r) + "_" + std::to_string(orbit_m) + "_sol.txt";
	}
}

void semifield_lifting::make_fname_deep_search_slice_success(
		std::string &fname,
		int f_out_path, std::string &out_path,
		int orbit_r, int orbit_m)
{
	if (f_out_path) {
		fname = prefix + "deep_slice" + std::to_string(orbit_r) + "_" + std::to_string(orbit_m) + "_suc.txt";
	}
	else {
		fname = "deep_slice" + std::to_string(orbit_r) + "_" + std::to_string(orbit_m) + "_suc.txt";
	}
}




}}}
