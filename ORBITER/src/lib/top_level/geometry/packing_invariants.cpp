// packing_invariants.cpp
// 
// Anton Betten
// Feb 14, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {





packing_invariants::packing_invariants()
{
	null();
}

packing_invariants::~packing_invariants()
{
	freeself();
}

void packing_invariants::null()
{
	the_packing = NULL;
	list_of_lines = NULL;
	f_has_klein = FALSE;
}

void packing_invariants::freeself()
{
	int i;
	
	if (the_packing) {
		FREE_int(the_packing);
		}
	if (list_of_lines) {
		FREE_int(list_of_lines);
		}
	if (f_has_klein) {
		delete [] R;
		for (i = 0; i < nb_planes; i++) {
			FREE_int(Pts_on_plane[i]);
			}
		FREE_pint(Pts_on_plane);
		FREE_int(nb_pts_on_plane);
		}
	null();
}

void packing_invariants::init(packing *P, 
	char *prefix, char *prefix_tex, int iso_cnt, 
	int *the_packing, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "packing_invariants::init" << endl;
		}
	strcpy(packing_invariants::prefix, prefix);
	strcpy(packing_invariants::prefix_tex, prefix_tex);
	packing_invariants::P = P;
	packing_invariants::the_packing = NEW_int(P->size_of_packing);
	packing_invariants::iso_cnt = iso_cnt;
	for (i = 0; i < P->size_of_packing; i++) {
		packing_invariants::the_packing[i] = the_packing[i];
		}
	list_of_lines = NEW_int(P->size_of_packing * P->spread_size);
	P->compute_list_of_lines_from_packing(
			list_of_lines,
			the_packing,
			verbose_level - 2);
	if (f_vv) {
		cout << "list_of_lines:" << endl;
		int_matrix_print(list_of_lines,
				P->size_of_packing, P->spread_size);
		}
	f_has_klein = FALSE;

#if 0
	if (TRUE /*v.s_l() == 0*/) {
		//cout << "packing_invariants::init no Klein invariants, "
		// "skipping" << endl;
		nb_planes = -1;
		R = NULL;
		Pts_on_plane = NULL;
		nb_pts_on_plane = NULL;
		}
	else {
		}
#endif

	if (f_v) {
		cout << "packing_invariants::init done" << endl;
		}
}

void packing_invariants::init_klein_invariants(
		Vector &v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "packing_invariants::init_klein_invariants" << endl;
		}
	nb_planes = v.s_ii(0);
	R = new longinteger_object[nb_planes];
	Pts_on_plane = NEW_pint(nb_planes);
	nb_pts_on_plane = NEW_int(nb_planes);
	for (i = 0; i < nb_planes; i++) {
		R[i].create(v.s_i(1).as_vector().s_ii(i));
		}
	for (i = 0; i < nb_planes; i++) {
		nb_pts_on_plane[i] =
			v.s_i(2).as_vector().s_i(i).as_vector().s_l();
		Pts_on_plane[i] = NEW_int(nb_pts_on_plane[i]);
		for (j = 0; j < nb_pts_on_plane[i]; j++) {
			Pts_on_plane[i][j] =
				v.s_i(2).as_vector().s_i(i).as_vector().s_ii(j);
			}
		}
	f_has_klein = TRUE;
	if (f_v) {
		cout << "packing_invariants::init_klein_invariants done" << endl;
		}
}


void packing_invariants::compute_decomposition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "packing_invariants::compute_decomposition" << endl;
		}
	

	int f_second = FALSE;

	C = NEW_OBJECT(classify);
	C->init(nb_pts_on_plane, nb_planes, f_second, 0);
	if (f_v) {
		cout << "packing_invariants::compute_decomposition: "
				"plane-intersection type: ";
		C->print(FALSE /*f_backwards*/);
		C->print_naked_tex(cout, FALSE /*f_backwards*/);
		}
	
#if 0
	ost << "Plane type of Klein-image is $($ ";
	C->print_naked_tex(ost);
	ost << " $)$" << endl << endl;
	ost << "\\bigskip" << endl << endl;
#endif

	int a, b, h, f, l, m, u, uu, idx;
	
	m = 0;
	for (i = 0; i < C->nb_types; i++) {
		f = C->type_first[i];
		l = C->type_len[i];
		a = C->data_sorted[f];
		m = MAXIMUM(a, m);
		}

	nb_blocks = 0;
	for (i = 0; i < C->nb_types; i++) {
		f = C->type_first[i];
		l = C->type_len[i];
		a = C->data_sorted[f];
		if (TRUE /*a == m*/) {
			nb_blocks += l;
			}
		}
	if (f_v) {
		cout << "There are " << nb_blocks
				<< " interesting planes" << endl;
		}
	block_to_plane = NEW_int(nb_blocks);
	plane_to_block = NEW_int(nb_planes);


	nb_fake_points = P->Spread_tables->nb_spreads; //P->E->Reps->count;
	nb_fake_blocks = P->size_of_packing;
	total_nb_points =
			P->size_of_packing * P->spread_size
				+ nb_fake_points;
	total_nb_blocks = nb_blocks + nb_fake_blocks;

	Inc = NEW_int(total_nb_points * total_nb_blocks);
	for (i = 0; i < total_nb_points * total_nb_blocks; i++) {
		Inc[i] = 0;
		}
	for (i = 0; i < nb_planes; i++) {
		plane_to_block[i] = -1;
		}
	j = 0;
	for (h = 0; h < C->nb_types; h++) {
		f = C->type_first[h];
		l = C->type_len[h];
		a = C->data_sorted[f];
		if (TRUE /*a == m*/) {
			for (u = 0; u < l; u++) {
				a = C->data_sorted[f + u];
				idx = C->sorting_perm_inv[f + u];
				for (uu = 0; uu < a; uu++) {
					i = Pts_on_plane[idx][uu];
					Inc[i * total_nb_blocks + j] = 1;
					}
				block_to_plane[j] = idx;
				plane_to_block[idx] = j;
				j++;
				} // next u
			} // if
		} // next h
	for (h = 0; h < P->size_of_packing; h++) {
		for (u = 0; u < P->spread_size; u++) {
			i = h * P->spread_size + u;
			j = nb_blocks + h;
			Inc[i * total_nb_blocks + j] = 1;
			}
		}	
	for (h = 0; h < P->size_of_packing; h++) {
		a = the_packing[h];
		b = P->Spread_tables->spread_iso_type[a];
		i = P->size_of_packing * P->spread_size + b;
		j = nb_blocks + h;
		Inc[i * total_nb_blocks + j] = 1;
		}

	if (FALSE /*nb_blocks < 20*/) {

		cout << "we will draw an incidence picture" << endl;
		
		incidence_structure *I;
		partitionstack *Stack;
		
		I = NEW_OBJECT(incidence_structure);
		I->init_by_matrix(total_nb_points,
				total_nb_blocks, Inc,
				0 /* verbose_level */);
		Stack = NEW_OBJECT(partitionstack);
		Stack->allocate(total_nb_points + total_nb_blocks,
				0 /* verbose_level */);
		Stack->subset_continguous(total_nb_points, total_nb_blocks);
		Stack->split_cell(0 /* verbose_level */);
		Stack->subset_continguous(P->size_of_packing * P->spread_size,
				nb_fake_points);
		Stack->split_cell(0 /* verbose_level */);
		if (nb_fake_points >= 2) {

			// isolate the fake points:
			for (i = 1; i < nb_fake_points; i++) {
				Stack->subset_continguous(
						P->size_of_packing * P->spread_size + i,
						nb_fake_points - i);
				Stack->split_cell(0 /* verbose_level */);
				}
			}
		Stack->subset_continguous(total_nb_points + nb_blocks,
				nb_fake_blocks);
		Stack->split_cell(0 /* verbose_level */);
		Stack->sort_cells();

		sprintf(fname_incidence_pic, "%s%d_packing_planes.tex",
				prefix_tex, iso_cnt);
		{
			ofstream fp_pic(fname_incidence_pic);

			//ost << "\\input " << fname_incidence_pic << endl;
			I->latex_it(fp_pic, *Stack);
			//ost << "\\\\" << endl;
		}
		FREE_OBJECT(Stack);
		FREE_OBJECT(I);
		}

	// compute TDO:
	{
		int depth = INT_MAX;
		
		I = NEW_OBJECT(incidence_structure);
		I->init_by_matrix(total_nb_points,
				total_nb_blocks, Inc,
				0 /* verbose_level */);
		Stack = NEW_OBJECT(partitionstack);
		Stack->allocate(total_nb_points + total_nb_blocks,
				0 /* verbose_level */);
		Stack->subset_continguous(total_nb_points,
				total_nb_blocks);
		Stack->split_cell(0 /* verbose_level */);
		Stack->subset_continguous(P->size_of_packing * P->spread_size,
				nb_fake_points);
		Stack->split_cell(0 /* verbose_level */);
		if (nb_fake_points >= 2) {

			// isolate the fake points:
			for (i = 1; i < nb_fake_points; i++) {
				Stack->subset_continguous(
						P->size_of_packing * P->spread_size + i,
						nb_fake_points - i);
				Stack->split_cell(0 /* verbose_level */);
				}
			}
		Stack->subset_continguous(total_nb_points + nb_blocks,
				nb_fake_blocks);
		Stack->split_cell(0 /* verbose_level */);
		Stack->sort_cells();

		I->compute_TDO_safe(*Stack, depth, verbose_level - 2);
		

		sprintf(fname_row_scheme,
				"%s%d_packing_planes_row_scheme.tex",
				prefix_tex, iso_cnt);
		sprintf(fname_col_scheme,
				"%s%d_packing_planes_col_scheme.tex",
				prefix_tex, iso_cnt);
		{
		ofstream fp_row_scheme(fname_row_scheme);
		ofstream fp_col_scheme(fname_col_scheme);
		I->get_and_print_row_tactical_decomposition_scheme_tex(
			fp_row_scheme, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		I->get_and_print_column_tactical_decomposition_scheme_tex(
			fp_col_scheme, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		}


	}


	if (f_v) {
		cout << "packing_invariants::compute_decomposition done" << endl;
		}
}

}}

