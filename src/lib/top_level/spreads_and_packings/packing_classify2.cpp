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
namespace top_level {

void packing_classify::compute_klein_invariants(
		isomorph *Iso, int f_split, int split_r, int split_m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int orbit, id;
	file_io Fio;

	if (f_v) {
		cout << "packing_classify::compute_klein_invariants" << endl;
	}

	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
	
		if (f_split && (orbit % split_m) != split_r) {
			continue;
		}
		
		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Reps->count << endl;
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
		id = Iso->orbit_perm[Iso->orbit_fst[Iso->Reps->rep[orbit]]];
	
		Iso->load_solution(id, the_packing);
		if (f_vv) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ")" << endl;
			lint_vec_print(cout, the_packing, Iso->size);
			cout << endl;
		}
		Spread_table_with_selection->Spread_tables->compute_list_of_lines_from_packing(list_of_lines,
				the_packing, size_of_packing, verbose_level - 2);
		if (f_v3) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ") list of lines:" << endl;
			lint_matrix_print(list_of_lines,
					size_of_packing, spread_size);
			cout << endl;
		}

		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Reps->count
					<< " before compute_and_save_klein_invariants" << endl;
		}
		compute_and_save_klein_invariants(Iso->prefix_invariants,
			orbit, 
			list_of_lines,
			size_of_packing * spread_size,
			verbose_level - 2);
		if (f_v) {
			cout << "packing_classify::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Reps->count
					<< " after compute_and_save_klein_invariants" << endl;
		}

	} // next orbit

	
	if (f_v) {
		cout << "packing_classify::compute_klein_invariants done" << endl;
	}
}

void packing_classify::klein_invariants_fname(
		std::string &fname, std::string &prefix, int iso_cnt)
{
	fname.assign(prefix);
	char str[1000];
	sprintf(str, "%d_klein_invariant.bin", iso_cnt);
	fname.append(str);
}

void packing_classify::compute_and_save_klein_invariants(std::string &prefix,
	int iso_cnt, 
	long int *data, int data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
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
				"before P3->klein_correspondence" << endl;
	}
	P3->klein_correspondence(P5,
		data, data_size, list_of_lines_klein_image, 0/*verbose_level*/);




	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"after P3->klein_correspondence" << endl;
	}
	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants "
				"before plane_intersection_type_fast" << endl;
	}
	P5->plane_intersection_type_slow(Gr, list_of_lines_klein_image, data_size,
		R, Pts_on_plane, nb_pts_on_plane, nb_planes,
		verbose_level /*- 3*/);

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants: "
				"We found " << nb_planes << " planes." << endl;
#if 1
		for (i = 0; i < nb_planes; i++) {
			cout << setw(3) << i << " : " << R[i]
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			lint_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl;
		}
#endif
	}

	Vector v;

	v.m_l(3);
	v.m_ii(0, nb_planes);
	v.s_i(1).change_to_vector();
	v.s_i(2).change_to_vector();

	v.s_i(1).as_vector().m_l(nb_planes);
	v.s_i(2).as_vector().m_l(nb_planes);
	for (i = 0; i < nb_planes; i++) {
		v.s_i(1).as_vector().m_ii(i, R[i].as_int());
		//v.s_i(1).as_vector().s_i(i).change_to_longinteger();
		//v.s_i(1).as_vector().s_i(i).as_longinteger().allocate(1, R[i].rep());
		v.s_i(2).as_vector().s_i(i).change_to_vector();
		v.s_i(2).as_vector().s_i(i).as_vector().m_l(nb_pts_on_plane[i]);
		for (j = 0; j < nb_pts_on_plane[i]; j++) {
			v.s_i(2).as_vector().s_i(i).as_vector().m_ii(j, Pts_on_plane[i][j]);
		}
	}

	string fname;
	
	klein_invariants_fname(fname, prefix, iso_cnt);
	v.save_file(fname.c_str());

	delete [] R;
	for (i = 0; i < nb_planes; i++) {
		FREE_lint(Pts_on_plane[i]);
	}
	FREE_plint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);

	if (f_v) {
		cout << "packing_classify::compute_and_save_klein_invariants done" << endl;
	}
}


void packing_classify::report(isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	char fname[1000];
	file_io Fio;

	if (f_v) {
		cout << "packing_classify::report" << endl;
	}

	sprintf(fname, "packing_report_q%d.tex", (int)q);

	{
		ofstream f(fname);

		cout << "Writing file " << fname << " with "
				<< Iso->Reps->count << " spreads:" << endl;

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

void packing_classify::report_whole(isomorph *Iso,
		ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	report_title_page(Iso, ost, verbose_level);

	ost << "\\chapter{Summary}" << endl << endl;
	ost << "There are " << Iso->Reps->count
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

	
	tally C_ago;

	
	C_ago.init(inv->Ago_int, Iso->Reps->count, FALSE, 0);
	ost << "Classification by Ago: ";
	C_ago.print_naked_tex(ost, TRUE /*f_backwards*/);
	ost << "\\\\" << endl;

	ost << "\\chapter{Invariants: Types of Packing}" << endl << endl;



	inv->make_table(Iso, ost, FALSE, FALSE, verbose_level);

	ost << "\\clearpage" << endl << endl;

	inv->make_table(Iso, ost, TRUE, FALSE, verbose_level);

	ost << "\\clearpage" << endl << endl;

	inv->make_table(Iso, ost, FALSE, TRUE, verbose_level);

	ost << "\\clearpage" << endl << endl;


	report_packings_by_ago(Iso, ost, inv, C_ago, verbose_level);


	
	report_extra_stuff(Iso, ost, verbose_level);
	

	latex_interface L;
	L.foot(ost);
	if (inv) {
		FREE_OBJECT(inv);
	}
}

void packing_classify::report_title_page(
		isomorph *Iso, ostream &ost, int verbose_level)
{
	int f_book = TRUE;
	int f_title = TRUE;
	char title[1000];
	const char *author = "Orbiter";
	int f_toc = TRUE;
	int f_landscape = FALSE;
	int f_12pt = FALSE;
	int f_enlarged_page = TRUE;
	int f_pagenumbers = TRUE;
	latex_interface L;

	sprintf(title, "The Packings of PG$(%d,%d)$", (int)3, (int)q);
	L.head(ost, f_book, f_title,
		title, author, 
		f_toc, f_landscape, f_12pt,
		f_enlarged_page, f_pagenumbers,
		NULL /* extra_praeamble */);



}

void packing_classify::report_packings_by_ago(
	isomorph *Iso, ostream &ost,
	invariants_packing *inv, tally &C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;

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
	isomorph *Iso, ostream &ost,
	int orbit, invariants_packing *inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id, rep, first; //, c;
	longinteger_object go;

	if (f_v) {
		cout << "packing_classify::report_isomorphism_type" << endl;
	}


	rep = Iso->Reps->rep[orbit];
	first = Iso->orbit_fst[rep];
	//c = Iso->starter_number[first];
	id = Iso->orbit_perm[first];		
	Iso->load_solution(id, the_packing);

	
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

	tally C_iso;

	C_iso.init_lint(spread_iso_type, Iso->size, FALSE, 0);
	ost << "Classification by isomorphism type of spreads: ";
	C_iso.print_naked_tex(ost, FALSE /*f_backwards*/);
	ost << "\\\\" << endl;
		
		
	int dual_idx;
#if 0
	int f_implicit_fusion = TRUE;
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
	inv->Inv[orbit].C->print_naked_tex(ost, TRUE /*f_backwards*/);
	ost << " $)$" << endl << endl;
	ost << "\\bigskip" << endl << endl;
#endif

	sims *Stab;
		
	Stab = Iso->Reps->stab[orbit];

	Stab->group_order(go);
	ost << "Stabilizer has order $";
	go.print_not_scientific(ost);
	ost << "$\\\\" << endl;

	report_stabilizer(*Iso, ost, orbit, verbose_level);

	if (f_v) {
		cout << "packing_classify::report computing induced "
				"action on the set (in data)" << endl;
	}
	Iso->induced_action_on_set_basic(Stab, the_packing, verbose_level - 2);

	if (f_v) {
		longinteger_object go;
			
		Iso->AA->group_order(go);
		cout << "action " << Iso->AA->label << " computed, "
				"group order is " << go << endl;
	}

	report_stabilizer_in_action(*Iso, ost, orbit, verbose_level);

	if (go.as_int() > 2) {
		report_stabilizer_in_action_gap(*Iso, orbit, verbose_level);
		}

	schreier Orb;
	//longinteger_object go;
		
	Iso->AA->compute_all_point_orbits(Orb,
			Stab->gens, verbose_level - 2);
	//cout << "Computed all orbits on the set, "
	//"found " << Orb.nb_orbits << " orbits" << endl;
	//cout << "orbit lengths: ";
	//int_vec_print(cout, Orb.orbit_len, Orb.nb_orbits);
	//cout << endl;

	tally C;


	C.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);


	ost << "\\bigskip" << endl;

	ost << "There are $" << Orb.nb_orbits
			<< "$ orbits on the set.\\\\" << endl;
	ost << "The orbit type is $[$ ";
	C.print_naked_tex(ost, FALSE /*f_backwards*/);
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
	isomorph *Iso, ostream &ost,
	int orbit, invariants_packing *inv, long int *list_of_lines,
	int verbose_level)
{
	latex_interface L;

#if 1
	{
	int nb_points;
	int *the_spread;

	nb_points = T->Grass->nb_points_covered(0 /*verbose_level*/);

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
	isomorph *Iso, ostream &ost,
	int orbit, invariants_packing *inv,
	int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	file_io Fio;

	// klein invariants:
	{
		string fname_klein;
		Vector V;
		
		klein_invariants_fname(fname_klein,
				Iso->prefix_invariants, orbit);
		if (Fio.file_size(fname_klein) > 0) {
			if (f_vv) {
				cout << "packing::report loading "
						"file " << fname_klein << endl;
				}
			V.load_file(fname_klein.c_str());
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
						inv->Inv[orbit].fname_row_scheme.c_str());
				//f << "\\input "
				//<< inv->Inv[orbit].fname_row_scheme << endl;
				ost << "\\]" << endl;
				ost << "\\[" << endl;
				Fio.copy_file_to_ostream(ost,
						inv->Inv[orbit].fname_col_scheme.c_str());
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

void packing_classify::report_stabilizer(isomorph &Iso,
		ostream &ost, int orbit, int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;

	Stab = Iso.Reps->stab[orbit];
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
		T->A->element_print_latex(Stab->gens.ith(i), ost);
		//f << "$$" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
	}
	ost << "$$" << endl;
}

void packing_classify::report_stabilizer_in_action(
		isomorph &Iso, ostream &ost, int orbit,
		int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;

	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	ost << "The stabilizer generators in their action "
			"on the spreads of the packing are:\\\\" << endl;
	//f << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;

		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		ost << "$";
		Iso.AA->element_print_as_permutation_with_offset(
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
		isomorph &Iso, int orbit,
		int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;
	char fname[1000];


	sprintf(fname, "group_%d.g", orbit);
	{
		ofstream fp(fname);


		Stab = Iso.Reps->stab[orbit];
		Stab->group_order(go);

		//f << "The stabilizer generators in their action on "
		//"the spreads of the packing are:\\\\" << endl;
		//f << "$$" << endl;
		for (i = 0; i < Stab->gens.len; i++) {

			int offset = 1;
			int f_do_it_anyway_even_for_big_degree = TRUE;
			int f_print_cycles_of_length_one = FALSE;

			//int *fp, n, ord;

			//fp = NEW_int(A->degree);
			//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			//FREE_int(fp);

			//ord = A->element_order(Stab->gens.ith(i));

			fp << "g" << i + 1 << " := ";
			Iso.AA->element_print_as_permutation_with_offset(
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
		isomorph *Iso, ostream &ost,
		int verbose_level)
{
	ost << "\\chapter{The Field GF$(" << q << ")$}" << endl << endl;
	
	T->Mtx->GFq->cheat_sheet(ost, verbose_level - 1);


	ost << "\\chapter{The Points and Lines of "
			"PG$(3," << q << ")$}" << endl << endl;

	//f << "\\clearpage" << endl << endl;

	{
		int nb_points;
		int nb_lines;
		int v[4];
		int i, j, u;
		combinatorics_domain Combi;

		nb_points = P3->N_points;
		nb_lines = Combi.generalized_binomial(4, 2, q);

		ost << "PG$(3," << q << ")$ has " << nb_points
				<< " points:\\\\" << endl;
		for (i = 0; i < nb_points; i++) {
			P3->unrank_point(v, i);
			ost << "$P_{" << i << "}=";
			Orbiter->Int_vec.print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
		}
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
		ost << "PG$(3," << q << ")$ has " << nb_lines
				<< " lines:\\\\" << endl;
		for (u = 0; u < nb_lines; u++) {
			T->Grass->unrank_lint(u, 0 /* verbose_level*/);
			ost << "$L_{" << u << "}=";
			ost << "\\left[" << endl;
			ost << "\\begin{array}{c}" << endl;
			for (i = 0; i < T->k; i++) {
				for (j = 0; j < T->n; j++) {
					ost << T->Grass->M[i * T->n + j];
				}
				ost << "\\\\" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "\\right]" << endl;
			ost << " = \\{" << endl;
			for (i = 0; i < P3->k; i++) {
				ost << P3->Lines[u * P3->k + i];
				if (i < P3->k - 1) {
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
			lint_vec_print_fully(ost,
					Spread_table_with_selection->Spread_tables->spread_table + u * spread_size, spread_size);
			ost << "$ isomorphism type "
					<< Spread_table_with_selection->Spread_tables->spread_iso_type[u] << "\\\\" << endl;

		}
#endif

	}
}

}}

