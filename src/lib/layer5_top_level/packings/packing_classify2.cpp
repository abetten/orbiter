// packing_classify2.cpp
// 
// Anton Betten
// Feb 6, 2013
//
// moved here from packing.cpp: Apr 25, 2016
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


void packing_classify::compute_klein_invariants(
		isomorph::isomorph *Iso,
		int f_split, int split_r, int split_m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int orbit, id;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_classify::compute_klein_invariants" << endl;
	}

	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
	
		if (f_split && (orbit % split_m) != split_r) {
			continue;
		}
		
		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Folding->Reps->count << endl;
		}
		
		string fname;
	
		klein_invariants_fname(fname, Iso->prefix_invariants, orbit);
		cout << "file size of " << fname << " is " << Fio.file_size(fname) << endl;
		if (Fio.file_size(fname) > 0) {
			if (f_v) {
				cout << "file " << fname << " exists, skipping" << endl;
			}
			continue;
		}
		id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[Iso->Folding->Reps->rep[orbit]]];
	
		Iso->Lifting->load_solution(id, the_packing, verbose_level - 1);
		if (f_vv) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ")" << endl;
			Lint_vec_print(cout, the_packing, Iso->size);
			cout << endl;
		}
		Spread_table_with_selection->Spread_tables->compute_list_of_lines_from_packing(list_of_lines,
				the_packing, size_of_packing, verbose_level - 2);
		if (f_v3) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ") list of lines:" << endl;
			Lint_matrix_print(list_of_lines,
					size_of_packing, spread_size);
			cout << endl;
		}

		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Folding->Reps->count
					<< " before compute_and_save_klein_invariants" << endl;
		}
		compute_and_save_klein_invariants(Iso->prefix_invariants,
			orbit, 
			list_of_lines,
			size_of_packing * spread_size,
			verbose_level - 2);
		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Folding->Reps->count
					<< " after compute_and_save_klein_invariants" << endl;
		}

	} // next orbit

	
	if (f_v) {
		cout << "packing_classify::compute_klein_invariants done" << endl;
	}
}

void packing_classify::klein_invariants_fname(
		std::string &fname,
		std::string &prefix, int iso_cnt)
{
	fname = prefix + std::to_string(iso_cnt) + "_klein_invariant.bin";
}

void packing_classify::compute_and_save_klein_invariants(
		std::string &prefix,
	int iso_cnt, 
	long int *data, int data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//ring_theory::longinteger_object *R;
	//long int **Pts_on_plane;
	//int *nb_pts_on_plane;
	//int nb_planes;
	int i, j;

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants" << endl;
	}
	
	if (data_size != size_of_packing * spread_size) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"data_size != size_of_packing * spread_size" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"before P3->Grass_lines->klein_correspondence" << endl;
	}
	P3->Subspaces->Grass_lines->klein_correspondence(P3, //P5,
		data, data_size, list_of_lines_klein_image, 0/*verbose_level*/);

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"after P3->Grass_lines->klein_correspondence" << endl;
	}


	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"before plane_intersection_type_slow" << endl;
	}

	int threshold = 3;


	geometry::intersection_type *Int_type;

	P5->plane_intersection_type(
			list_of_lines_klein_image, data_size, threshold,
		Int_type,
		verbose_level - 2);

#if 0
	P5->plane_intersection_type_slow(//Gr,
			list_of_lines_klein_image, data_size, threshold,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level /*- 3*/);
#endif

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"after plane_intersection_type_slow" << endl;
	}

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants: "
				"We found " << Int_type->len << " planes." << endl;
#if 1
		for (i = 0; i < Int_type->len; i++) {
			cout << setw(3) << i << " : " << Int_type->R[i]
				<< " : " << setw(5) << Int_type->nb_pts_on_plane[i] << " : ";
			Lint_vec_print(cout, Int_type->Pts_on_plane[i], Int_type->nb_pts_on_plane[i]);
			cout << endl;
		}
#endif
	}

	typed_objects::Vector v;

	v.m_l(3);
	v.m_ii(0, Int_type->len);
	v.s_i(1).change_to_vector();
	v.s_i(2).change_to_vector();

	v.s_i(1).as_vector().m_l(Int_type->len);
	v.s_i(2).as_vector().m_l(Int_type->len);
	for (i = 0; i < Int_type->len; i++) {
		v.s_i(1).as_vector().m_ii(i, Int_type->R[i].as_int());
		//v.s_i(1).as_vector().s_i(i).change_to_longinteger();
		//v.s_i(1).as_vector().s_i(i).as_longinteger().allocate(1, R[i].rep());
		v.s_i(2).as_vector().s_i(i).change_to_vector();
		v.s_i(2).as_vector().s_i(i).as_vector().m_l(Int_type->nb_pts_on_plane[i]);
		for (j = 0; j < Int_type->nb_pts_on_plane[i]; j++) {
			v.s_i(2).as_vector().s_i(i).as_vector().m_ii(j, Int_type->Pts_on_plane[i][j]);
		}
	}

	string fname;
	
	klein_invariants_fname(fname, prefix, iso_cnt);
	v.save_file(fname);

#if 0
	delete [] R;
	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);
#endif

	FREE_OBJECT(Int_type);


	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants done" << endl;
	}
}


void packing_classify::report(
		isomorph::isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_classify::report" << endl;
	}

	fname = "packing_report_q" + std::to_string(q) + ".tex";

	{
		ofstream f(fname);

		cout << "Writing file " << fname << " with "
				<< Iso->Folding->Reps->count << " spreads:" << endl;

		report_whole(Iso, f, verbose_level);

	} // close file f
	if (f_v) {
		cout << "packing_classify::report written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	


	if (f_v) {
		cout << "packing_classify::report done" << endl;
	}
}

void packing_classify::report_whole(
		isomorph::isomorph *Iso,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	report_title_page(Iso, ost, verbose_level);

	ost << "\\chapter{Summary}" << endl << endl;
	ost << "There are " << Iso->Folding->Reps->count
			<< " packings of PG$(3," << q << ")$." << endl << endl;




	invariants_packing *inv = NULL;
	


	inv = NEW_OBJECT(invariants_packing);


	if (f_v) {
		cout << "packing::report loading and "
				"computing invariants" << endl;
	}

	inv->init(Iso, this, verbose_level);


	if (f_v) {
		cout << "packing::report loading and "
				"computing invariants done" << endl;
	}

	
	data_structures::tally C_ago;

	
	C_ago.init(inv->Ago_int, Iso->Folding->Reps->count, false, 0);
	ost << "Classification by Ago: ";
	C_ago.print_bare_tex(ost, true /*f_backwards*/);
	ost << "\\\\" << endl;

	ost << "\\chapter{Invariants: Types of Packing}" << endl << endl;



	inv->make_table(Iso, ost, false, false, verbose_level);

	ost << "\\clearpage" << endl << endl;

	inv->make_table(Iso, ost, true, false, verbose_level);

	ost << "\\clearpage" << endl << endl;

	inv->make_table(Iso, ost, false, true, verbose_level);

	ost << "\\clearpage" << endl << endl;


	report_packings_by_ago(Iso, ost, inv, C_ago, verbose_level);


	
	report_extra_stuff(Iso, ost, verbose_level);
	

	l1_interfaces::latex_interface L;
	L.foot(ost);
	if (inv) {
		FREE_OBJECT(inv);
	}
}

void packing_classify::report_title_page(
		isomorph::isomorph *Iso,
		std::ostream &ost,
		int verbose_level)
{
	int f_book = true;
	int f_title = true;
	string title, author, extra_praeamble;

	int f_toc = true;
	int f_landscape = false;
	int f_12pt = false;
	int f_enlarged_page = true;
	int f_pagenumbers = true;
	l1_interfaces::latex_interface L;

	title = "The Packings of PG$(3," + std::to_string(q) + ")$";
	author = "Orbiter";

	L.head(ost, f_book, f_title,
		title, author, 
		f_toc, f_landscape, f_12pt,
		f_enlarged_page, f_pagenumbers,
		extra_praeamble /* extra_praeamble */);



}

void packing_classify::report_packings_by_ago(
		isomorph::isomorph *Iso,
		std::ostream &ost,
	invariants_packing *inv,
	data_structures::tally &C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "packing_classify::report_packings_by_ago" << endl;
	}

	ost << "\\chapter{The Packings of PG$(3," << q << ")$}"
			<< endl << endl;

	ost << "\\clearpage" << endl << endl;

	
	int u, v, a, cnt, fst, length, t, vv;
	int *set;

	cnt = 0;
	for (u = C_ago.nb_types - 1; u >= 0; u--) {
		fst = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[fst];


		ost << "\\section{Packings with a Group of Order "
				"$" << t << "$}" << endl;

		ost << "There are " << length << " packings with an "
				"automorphism group of order $" << t << "$.\\\\" << endl;

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		if (length > 100) {
			ost << "Too many packings to list.\\\\" << endl;
			continue;
		}
		
		set = NEW_int(length);
		
		for (v = 0; v < length; v++) {
			vv = fst + v;
			a = C_ago.sorting_perm_inv[vv];
			set[v] = a;
		}

		Sorting.int_vec_heapsort(set, length);

		for (v = 0; v < length; v++, cnt++) {

			int orbit;

			orbit = set[v];


	//for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
			report_isomorphism_type(Iso, ost,
				orbit, inv, verbose_level);
		
		//} // next orbit
		} // next v

		FREE_int(set);
	} // next u

	
	if (f_v) {
		cout << "packing_classify::report_packings_by_ago done" << endl;
	}
}


void packing_classify::report_isomorphism_type(
		isomorph::isomorph *Iso,
		std::ostream &ost,
	int orbit, invariants_packing *inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id, rep, first; //, c;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "packing_classify::report_isomorphism_type" << endl;
	}


	rep = Iso->Folding->Reps->rep[orbit];
	first = Iso->Lifting->flag_orbit_solution_first[rep];
	//c = Iso->starter_number[first];
	id = Iso->Lifting->orbit_perm[first];
	Iso->Lifting->load_solution(id, the_packing, verbose_level - 1);

	
	for (i = 0; i < Iso->size; i++) {
		dual_packing[i] = Spread_table_with_selection->Spread_tables->dual_spread_idx[the_packing[i]];
	}


	Spread_table_with_selection->Spread_tables->compute_list_of_lines_from_packing(list_of_lines,
			the_packing, size_of_packing, verbose_level - 2);


	ost << "\\subsection*{Isomorphism Type " << orbit << "}" << endl;
	ost << "\\bigskip" << endl;

	for (i = 0; i < Iso->size; i++) {
		spread_iso_type[i] = Spread_table_with_selection->Spread_tables->spread_iso_type[the_packing[i]];
	}

	ost << "spread : isotype : dualspread \\\\" << endl;
	for (i = 0; i < Iso->size; i++) {
		ost << the_packing[i];
		ost << " : ";
		ost << spread_iso_type[i];
		ost << " : ";
		ost << dual_packing[i];
		ost << "\\\\" << endl;
	}
	//ost << "\\\\" << endl;
	ost << "\\bigskip" << endl;

	data_structures::tally C_iso;

	C_iso.init_lint(spread_iso_type, Iso->size, false, 0);
	ost << "Classification by isomorphism type of spreads: ";
	C_iso.print_bare_tex(ost, false /*f_backwards*/);
	ost << "\\\\" << endl;
		
		
	int dual_idx;
#if 0
	int f_implicit_fusion = true;
	dual_idx = Iso->identify_database_is_open(
			dual_packing, f_implicit_fusion, verbose_level - 2);
#endif

	dual_idx = inv->Dual_idx[orbit];
	ost << "The dual packing belongs to isomorphism type "
			<< dual_idx << "\\\\" << endl;
	ost << "\\bigskip" << endl;

#if 0
	ost << "Stabilizer has order $";
	inv->Ago[orbit].print_not_scientific(ost);
	ost << "$. The group that is induced has order $";
	inv->Ago_induced[orbit].print_not_scientific(ost);
	ost << "$\\\\" << endl;
#endif


#if 0
	ost << "Plane type of Klein-image is $($ ";
	inv->Inv[orbit].C->print_bare_tex(ost, true /*f_backwards*/);
	ost << " $)$" << endl << endl;
	ost << "\\bigskip" << endl << endl;
#endif

	groups::sims *Stab;
		
	Stab = Iso->Folding->Reps->stab[orbit];

	Stab->group_order(go);
	ost << "Stabilizer has order $";
	go.print_not_scientific(ost);
	ost << "$\\\\" << endl;

	report_stabilizer(*Iso, ost, orbit, verbose_level);

	if (f_v) {
		cout << "packing_classify::report computing induced "
				"action on the set (in data)" << endl;
	}
	Iso->Folding->induced_action_on_set_basic(Stab, the_packing, verbose_level - 2);

	if (f_v) {
		ring_theory::longinteger_object go;
			
		Iso->Folding->AA->group_order(go);
		cout << "action " << Iso->Folding->AA->label << " computed, "
				"group order is " << go << endl;
	}

	report_stabilizer_in_action(*Iso, ost, orbit, verbose_level);

	if (go.as_int() > 2) {
		report_stabilizer_in_action_gap(*Iso, orbit, verbose_level);
		}

	groups::schreier Orb;
	//longinteger_object go;
		
	Iso->Folding->AA->compute_all_point_orbits(Orb,
			Stab->gens, verbose_level - 2);
	//cout << "Computed all orbits on the set, "
	//"found " << Orb.nb_orbits << " orbits" << endl;
	//cout << "orbit lengths: ";
	//int_vec_print(cout, Orb.orbit_len, Orb.nb_orbits);
	//cout << endl;

	data_structures::tally C;


	C.init(Orb.orbit_len, Orb.nb_orbits, false, 0);


	ost << "\\bigskip" << endl;

	ost << "There are $" << Orb.nb_orbits
			<< "$ orbits on the set.\\\\" << endl;
	ost << "The orbit type is $[$ ";
	C.print_bare_tex(ost, false /*f_backwards*/);
	ost << " $]$\\\\" << endl;
	ost << "\\bigskip" << endl;
	
	report_klein_invariants(Iso,
			ost, orbit, inv, verbose_level);
		

	report_packing_as_table(Iso,
			ost, orbit, inv, list_of_lines, verbose_level);



	if (f_v) {
		cout << "packing_classify::report_isomorphism_type done" << endl;
	}
}

void packing_classify::report_packing_as_table(
		isomorph::isomorph *Iso, std::ostream &ost,
	int orbit,
	invariants_packing *inv, long int *list_of_lines,
	int verbose_level)
{
	l1_interfaces::latex_interface L;

#if 1
	{
	int nb_points;
	int *the_spread;

	nb_points = T->SD->Grass->nb_points_covered(0 /*verbose_level*/);

	cout << "nb_points=" << nb_points << endl;
	the_spread = NEW_int(spread_size * nb_points);

	ost << "The lines of the packing are "
			"(each row corresponds to a spread):" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	L.lint_matrix_print_tex(ost, list_of_lines, size_of_packing, spread_size);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	if (T->Sing) {
		int i, j, a, b;
		ost << "$$" << endl;
		ost << "\\left[" << endl;
		ost << "\\begin{array}{*{" << spread_size << "}{c}}" << endl;
		for (i = 0; i < size_of_packing; i++) {
			for (j = 0; j < spread_size; j++) {
				a = list_of_lines[i * spread_size + j];
				b = T->Sing->line_orbit_inv[a];
				ost << T->Sing->line_orbit_label_tex[b];
				if (j < spread_size - 1) {
					ost << " & ";
					}
				}
			ost << "\\\\" << endl;
			}
		//int_matrix_print_tex(f,
		//list_of_lines, size_of_packing, spread_size);
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		ost << "$$" << endl;
		}
#if 0
	int u, v, a, j;
	for (u = 0; u < size_of_packing; u++) {
		f << "Spread $" << u << "$ is spread number $"
				<< packing[u] << "$.\\\\" << endl;
		f << "Lines of the spread are:" << endl;
		f << "$$" << endl;
		for (v = 0; v < spread_size; v++) {
			if (v && (v % 5) == 0) {
				f << "$$" << endl;
				f << "$$" << endl;
				}
			a = list_of_lines[u * spread_size + v];
			T->Grass->unrank_int(a, 0/*verbose_level - 4*/);
			f << "L_{" << a << "}=";
			f << "\\left[" << endl;
			f << "\\begin{array}{c}" << endl;
			for (i = 0; i < T->k; i++) {
				for (j = 0; j < T->n; j++) {
					f << T->Grass->M[i * T->n + j];
					}
				f << "\\\\" << endl;
				}
			f << "\\end{array}" << endl;
			f << "\\right]" << endl;
			} // next v
		f << "$$" << endl;
		for (v = 0; v < spread_size; v++) {
			a = list_of_lines[u * spread_size + v];
			T->Grass->unrank_int(a, 0/*verbose_level - 4*/);
			T->Grass->points_covered(
					the_spread + v * nb_points,
					0 /* verbose_level*/);
			}
		f << "The partition of the point set is:\\\\" << endl;
		f << "$$" << endl;
		f << "\\left[" << endl;
		int_matrix_print_tex(f,
				the_spread, spread_size, nb_points);
		f << "\\right]" << endl;
		f << "$$" << endl;
		} // next u
#endif
	FREE_int(the_spread);
	}
#endif
}

void packing_classify::report_klein_invariants(
		isomorph::isomorph *Iso, std::ostream &ost,
	int orbit, invariants_packing *inv,
	int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	orbiter_kernel_system::file_io Fio;

	// klein invariants:
	{
		string fname_klein;
		typed_objects::Vector V;
		
		klein_invariants_fname(fname_klein,
				Iso->prefix_invariants, orbit);
		if (Fio.file_size(fname_klein) > 0) {
			if (f_vv) {
				cout << "packing::report loading "
						"file " << fname_klein << endl;
				}
			V.load_file(fname_klein);
			inv->Inv[orbit].init_klein_invariants(
					V, verbose_level - 1);
			// free, so that we don't use that much memory:
			V.freeself();

			inv->Inv[orbit].compute_decomposition(verbose_level - 1);
			ost << "\\bigskip" << endl << endl;
			if (Fio.file_size(inv->Inv[orbit].fname_row_scheme) < 1000) {
				ost << "\\[" << endl;
				cout << "copying file "
						<< inv->Inv[orbit].fname_row_scheme
						<< " in" << endl;
				Fio.copy_file_to_ostream(ost,
						inv->Inv[orbit].fname_row_scheme);
				//f << "\\input "
				//<< inv->Inv[orbit].fname_row_scheme << endl;
				ost << "\\]" << endl;
				ost << "\\[" << endl;
				Fio.copy_file_to_ostream(ost,
						inv->Inv[orbit].fname_col_scheme);
				//ost << "\\input "
				//<< inv->Inv[orbit].fname_col_scheme << endl;
				ost << "\\]" << endl;
			}
			else {
				ost << "The TDO decomposition is "
						"too large to print\\\\" << endl;
			}
		}
	}
}

void packing_classify::report_stabilizer(
		isomorph::isomorph &Iso,
		std::ostream &ost, int orbit,
		int verbose_level)
{
	groups::sims *Stab;
	ring_theory::longinteger_object go;
	int i;

	Stab = Iso.Folding->Reps->stab[orbit];
	Stab->group_order(go);

	ost << "The stabilizer of order $" << go
			<< "$ is generated by:\\\\" << endl;
	ost << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		//f << "$$ g_{" << i + 1 << "}=" << endl;
		T->A->Group_element->element_print_latex(Stab->gens.ith(i), ost);
		//f << "$$" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
	}
	ost << "$$" << endl;
}

void packing_classify::report_stabilizer_in_action(
		isomorph::isomorph &Iso,
		std::ostream &ost, int orbit,
		int verbose_level)
{
	groups::sims *Stab;
	ring_theory::longinteger_object go;
	int i;

	Stab = Iso.Folding->Reps->stab[orbit];
	Stab->group_order(go);

	ost << "The stabilizer generators in their action "
			"on the spreads of the packing are:\\\\" << endl;
	//f << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = true;
		int f_print_cycles_of_length_one = false;

		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		ost << "$";
		Iso.Folding->AA->Group_element->element_print_as_permutation_with_offset(
			Stab->gens.ith(i), ost,
			offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one, 0 /* verbose_level */);
		ost << "$\\\\" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
	}
	//f << "$$" << endl;
}

void packing_classify::report_stabilizer_in_action_gap(
		isomorph::isomorph &Iso, int orbit,
		int verbose_level)
{
	groups::sims *Stab;
	ring_theory::longinteger_object go;
	int i;
	string fname;


	fname = "group_" + std::to_string(orbit) + ".g";
	{
		ofstream fp(fname);


		Stab = Iso.Folding->Reps->stab[orbit];
		Stab->group_order(go);

		//f << "The stabilizer generators in their action on "
		//"the spreads of the packing are:\\\\" << endl;
		//f << "$$" << endl;
		for (i = 0; i < Stab->gens.len; i++) {

			int offset = 1;
			int f_do_it_anyway_even_for_big_degree = true;
			int f_print_cycles_of_length_one = false;

			//int *fp, n, ord;

			//fp = NEW_int(A->degree);
			//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			//FREE_int(fp);

			//ord = A->element_order(Stab->gens.ith(i));

			fp << "g" << i + 1 << " := ";
			Iso.Folding->AA->Group_element->element_print_as_permutation_with_offset(
				Stab->gens.ith(i), fp,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one, 0 /* verbose_level */);
			fp << ";" << endl;
			//f << "of order $" << ord << "$ and with " << n
			//<< " fixed points" << endl;
		}
		fp << "G := Group([";
		for (i = 0; i < Stab->gens.len; i++) {
			fp << "g" << i + 1;
			if (i < Stab->gens.len - 1) {
				fp << ",";
			}
		}
		fp << "]);" << endl;
	}

}

void packing_classify::report_extra_stuff(
		isomorph::isomorph *Iso, std::ostream &ost,
		int verbose_level)
{
	ost << "\\chapter{The Field GF$(" << q << ")$}" << endl << endl;
	
	T->Mtx->GFq->Io->cheat_sheet(ost, verbose_level - 1);


	ost << "\\chapter{The Points and Lines of "
			"PG$(3," << q << ")$}" << endl << endl;

	//f << "\\clearpage" << endl << endl;

	{
		int nb_points;
		int nb_lines;
		int v[4];
		int i, j, u;
		combinatorics::combinatorics_domain Combi;

		nb_points = P3->Subspaces->N_points;
		nb_lines = Combi.generalized_binomial(4, 2, q);

		ost << "PG$(3," << q << ")$ has " << nb_points
				<< " points:\\\\" << endl;
		for (i = 0; i < nb_points; i++) {
			P3->unrank_point(v, i);
			ost << "$P_{" << i << "}=";
			Int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
		}
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
		ost << "PG$(3," << q << ")$ has " << nb_lines
				<< " lines:\\\\" << endl;
		for (u = 0; u < nb_lines; u++) {
			T->SD->Grass->unrank_lint(u, 0 /* verbose_level*/);
			ost << "$L_{" << u << "}=";
			ost << "\\left[" << endl;
			ost << "\\begin{array}{c}" << endl;
			for (i = 0; i < T->SD->k; i++) {
				for (j = 0; j < T->SD->n; j++) {
					ost << T->SD->Grass->M[i * T->SD->n + j];
				}
				ost << "\\\\" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "\\right]" << endl;
			ost << " = \\{" << endl;
			for (i = 0; i < P3->Subspaces->k; i++) {
				ost << P3->Subspaces->Implementation->Lines[u * P3->Subspaces->k + i];
				if (i < P3->Subspaces->k - 1) {
					ost << ", ";
				}
			}
			ost << "\\}" << endl;
			ost << "$\\\\" << endl;
		}

#if 1
		ost << "\\chapter{The Spreads of PG$(3," << q << ")$}"
				<< endl << endl;

		ost << "PG$(3," << q << ")$ has " << Spread_table_with_selection->Spread_tables->nb_spreads
				<< " labeled spreads\\\\" << endl;

		for (u = 0; u < Spread_table_with_selection->Spread_tables->nb_spreads; u++) {
			ost << "Spread " << u << " is $";
			Lint_vec_print_fully(ost,
					Spread_table_with_selection->Spread_tables->spread_table + u * spread_size, spread_size);
			ost << "$ isomorphism type "
					<< Spread_table_with_selection->Spread_tables->spread_iso_type[u] << "\\\\" << endl;

		}
#endif

	}
}

}}}

