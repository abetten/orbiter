/*
 * blt_set_invariants.cpp
 *
 *  Created on: Apr 7, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace orthogonal_geometry {




blt_set_invariants::blt_set_invariants()
{
	Record_birth();
	D = NULL;

	set_size = 0;
	the_set_in_orthogonal = NULL;
	the_set_in_PG = NULL;

	intersection_type = NULL;
	highest_intersection_number = 0;
	intersection_matrix = NULL;
	nb_planes = 0;

	f_has_interesting_planes = false;
	Sos = NULL;
	Sos2 = NULL;
	Sos3 = NULL;

	D2 = NULL;
	D3 = NULL;

	Sos2_idx = NULL;
	Sos3_idx = NULL;

}

blt_set_invariants::~blt_set_invariants()
{
	Record_death();
	if (the_set_in_orthogonal) {
		FREE_lint(the_set_in_orthogonal);
	}
	if (the_set_in_PG) {
		FREE_lint(the_set_in_PG);
	}
}

void blt_set_invariants::init(
		blt_set_domain *D, long int *the_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int v5[5];
	int i;

	if (f_v) {
		cout << "blt_set_invariants::init" << endl;
	}
	blt_set_invariants::D = D;
	set_size = D->q + 1;
	the_set_in_orthogonal = NEW_lint(set_size);
	the_set_in_PG = NEW_lint(set_size);
	Lint_vec_copy(the_set, the_set_in_orthogonal, set_size);

	for (i = 0; i < set_size; i++) {

		D->O->Hyperbolic_pair->unrank_point(
				v5, 1, the_set[i],
				0 /* verbose_level */);

		the_set_in_PG[i] = D->P->rank_point(v5);
	}

	if (f_v) {
		cout << "blt_set_invariants::init done" << endl;
	}
}

void blt_set_invariants::compute(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_invariants::compute" << endl;
	}

	algebra::ring_theory::longinteger_object *R;

	Sos = NEW_OBJECT(other::data_structures::set_of_sets);
	Sos2 = NEW_OBJECT(other::data_structures::set_of_sets);
	Sos3 = NEW_OBJECT(other::data_structures::set_of_sets);
	D2 = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);
	D3 = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);


	if (f_v) {
		cout << "blt_set_invariants::compute "
				"before P->plane_intersections G53" << endl;
	}
	f_has_interesting_planes = D->P->plane_intersections(
			D->G53,
			the_set_in_PG, set_size, R, *Sos,
			verbose_level - 1);
	if (f_v) {
		cout << "blt_set_invariants::compute "
				"after P->plane_intersections G53" << endl;
	}


	if (f_has_interesting_planes) {
		if (f_v) {
			cout << "blt_set_invariants::compute "
					"before intersection_matrix" << endl;
		}
		Sos->intersection_matrix(
			intersection_type, highest_intersection_number,
			intersection_matrix, nb_planes,
			verbose_level - 1);

		if (f_v) {
			cout << "blt_set_invariants::compute before "
					"extract_largest_sets" << endl;
		}
		Sos->extract_largest_sets(*Sos2,
				Sos2_idx, verbose_level - 1);

		if (f_v) {
			cout << "blt_set_invariants::compute before "
					"remove_sets_of_given_size" << endl;
		}
		Sos->remove_sets_of_given_size(3,
				*Sos3, Sos3_idx, verbose_level - 1);

		if (f_v) {
			cout << "blt_set_invariants::compute before "
					"Sos2->compute_tdo_decomposition" << endl;
		}
		Sos2->compute_tdo_decomposition(*D2, verbose_level - 1);


		D2->get_row_scheme(verbose_level);
		D2->get_col_scheme(verbose_level);
		if (Sos3->nb_sets) {
			if (f_v) {
				cout << "blt_set_invariants::compute before "
						"Sos3[h].compute_tdo_decomposition" << endl;
			}
			Sos3->compute_tdo_decomposition(*D3, verbose_level - 1);
			D3->get_row_scheme(verbose_level - 1);
			D3->get_col_scheme(verbose_level - 1);
		}
	#if 0
		P->plane_intersection_invariant(G,
			data2, set_size,
			intersection_type[h], highest_intersection_number[h],
			intersection_matrix[h], nb_planes[h],
			verbose_level);
	#endif
	}

	FREE_OBJECTS(R);

	if (f_v) {
		cout << "blt_set_invariants::compute done" << endl;
	}
}

void blt_set_invariants::latex(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "blt_set_invariants::latex" << endl;
	}



	int a, i, j;
	ost << "Plane intersection type is ";
	for (i = highest_intersection_number; i >= 0; i--) {

		a = intersection_type[i];
		if (a == 0) {
			continue;
		}
		ost << "$" << i;
		if (a > 9) {
			ost << "^{" << a << "}";
		}
		else if (a > 1) {
			ost << "^" << a;
		}
		ost << "$ ";
	}
	ost << "\\\\" << endl;
	ost << "Plane invariant is ";

	if (nb_planes < 10) {
		ost << "$$";
		ost << "\\left[" << endl;
		ost << "\\begin{array}{*{" << nb_planes << "}{c}}" << endl;
		for (i = 0; i < nb_planes; i++) {
			for (j = 0; j < nb_planes; j++) {
				ost << intersection_matrix[i * nb_planes + j];
				if (j < nb_planes - 1) {
					ost << " & ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		ost << "$$" << endl;
	}
	else {
		ost << "too big (" << nb_planes << " planes)\\\\" << endl;
	}

	int f_enter_math = false;
	int f_print_subscripts = true;

	ost << "$$" << endl;
	D2->Scheme->print_row_decomposition_tex(
		ost, f_enter_math, f_print_subscripts, verbose_level - 1);
	ost << "\\quad" << endl;
	D2->Scheme->print_column_decomposition_tex(
		ost, f_enter_math, f_print_subscripts, verbose_level - 1);
	ost << "$$" << endl;
	D2->Stack->print_classes_tex(ost);

	if (Sos3->nb_sets) {
		ost << "$$" << endl;

		D3->Scheme->print_row_decomposition_tex(
			ost, f_enter_math, f_print_subscripts, verbose_level - 1);
		ost << "$$" << endl;
		ost << "$$" << endl;
		D3->Scheme->print_column_decomposition_tex(
			ost, f_enter_math, f_print_subscripts, verbose_level - 1);
		ost << "$$" << endl;
		D3->Stack->print_classes_tex(ost);

		int t, fst_col, fst, len, u, a;

		fst_col = D3->Stack->startCell[1];
		for (t = 0; t < D3->Stack->ht; t++) {
			if (!D3->Stack->is_col_class(t)) {
				continue;
			}
			ost << "Column cell " << t << ":\\\\" << endl;
			len = D3->Stack->cellSize[t];
			fst = D3->Stack->startCell[t];
			int *Cell;
			Cell = NEW_int(len);
			for (u = 0; u < len; u++) {
				a = D3->Stack->pointList[fst + u] - fst_col;
				Cell[u] = a;
			}
			Sorting.int_vec_heapsort(Cell, len);
#if 0
			for (u = 0; u < len; u++) {
				a = Cell[u];
				b = Sos3_idx[h][a];
				f << a << " (rank = ";
				R[h][b].print_not_scientific(f);
				f << ") = ";
				G->unrank_longinteger(R[h][b], 0 /* verbose_level */);
				f << "$\\left[" << endl;
				f << "\\begin{array}{*{" << 5 << "}{c}}" << endl;
				for (i = 0; i < 3; i++) {
					for (j = 0; j < 5; j++) {
						c = G->M[i * 5 + j];
						f << c;
						if (j < 4) {
							f << "&";
						}
					}
					f << "\\\\" << endl;
				}
				f << "\\end{array}" << endl;
				f << "\\right]$\\\\" << endl;
			}
#endif
			FREE_int(Cell);
		}
	}

	int tt, u, v;
	tt = (set_size + 3) / 4;

	ost << "The points by ranks:\\\\" << endl;
	ost << "\\begin{center}" << endl;

	for (u = 0; u < 4; u++) {
		ost << "\\begin{tabular}[t]{|c|c|}" << endl;
		ost << "\\hline" << endl;
		ost << "$i$ & Rank \\\\" << endl;
		ost << "\\hline" << endl;
		for (i = 0; i < tt; i++) {
			v = u * tt + i;
			if (v < set_size) {
				ost << "$" << v << "$ & $" << the_set_in_orthogonal[v]
					<< "$ \\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{tabular}" << endl;
	}
	ost << "\\end{center}" << endl;

	ost << "The points:\\\\" << endl;
	int v5[5];
	for (i = 0; i < set_size; i++) {
		D->O->Hyperbolic_pair->unrank_point(
				v5, 1, the_set_in_orthogonal[i], 0 /* verbose_level */);
		//Grass->unrank_int(data[i], 0/*verbose_level - 4*/);
		if ((i % 4) == 0) {
			if (i) {
				ost << "$$" << endl;
			}
			ost << "$$" << endl;
		}
		//f << "\\left[" << endl;
		//f << "\\begin{array}{c}" << endl;
		ost << "P_{" << i /*data[i]*/ << "}=";
		Int_vec_print(ost, v5, 5);
#if 0
		for (u = 0; u < 5; u++) {
			for (v = 0; v < n; v++) {
				f << Grass->M[u * n + v];
			}
			ost << "\\\\" << endl;
		}
#endif
		//f << "\\end{array}" << endl;
		//f << "\\right]" << endl;
	}
	ost << "$$" << endl;


	if (f_v) {
		cout << "blt_set_invariants::latex done" << endl;
	}
}



}}}}


