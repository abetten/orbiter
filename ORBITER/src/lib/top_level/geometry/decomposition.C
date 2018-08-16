// decomposition.C
// 
// Anton Betten
// started October 21, 2010
//
//
// 
//
//

#include "orbiter.h"

#include <fstream>



void decomposition_projective_space(INT k, finite_field *F, 
	INT nb_subsets, INT *sz, INT **subsets, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	incidence_structure Inc;
	INT nb_pts, nb_lines;
	INT *Mtx;
	INT *part;
	INT i, j, level;
	projective_space PG;
		
	if (f_v) {
		cout << "decomposition_projective_space" << endl;
		}
	PG.init(k, F, 
		TRUE, 
		verbose_level - 2);
	nb_pts = PG.N_points;
	nb_lines = PG.N_lines;
	if (f_v) {
		cout << "m = N_points = " << nb_pts << endl;
		cout << "n = N_lines = " << nb_lines << endl;
		}
	part = NEW_INT(nb_subsets);
	Mtx = NEW_INT(nb_pts * nb_lines);
	if (PG.incidence_bitvec == NULL) {
		cout << "PG.incidence_bitvec == NULL" << endl;
		exit(1);
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_lines; j++) {
			Mtx[i * nb_lines + j] = PG.is_incident(i, j);
			}
		}
	Inc.init_by_matrix(nb_pts, nb_lines, Mtx, verbose_level - 2);




	partitionstack Stack;
	INT ht0, c, l;

	Stack.allocate(nb_pts + nb_lines, 0);
	Stack.subset_continguous(Inc.nb_points(), Inc.nb_lines());
	Stack.split_cell(0);



	for (level = 0; level < nb_subsets; level++) {


		ht0 = Stack.ht;

		if (sz[level]) {
			c = Stack.cellNumber[Stack.invPointList[subsets[level][0]]];
			l = Stack.cellSize[c];
			if (sz[level] < l) {
				Stack.split_cell(subsets[level], sz[level], 0);
				part[level] = Stack.ht - 1;
				}
			else {
				part[level] = c;
				}
			}
		else {
			part[level] = -1;
			}


		if (f_v) {
			cout << "decomposition_projective_space level " << level
					<< " : partition stack after splitting:" << endl;
			Stack.print(cout);
			cout << "i : part[i]" << endl;
			for (i = 0; i < nb_subsets; i++) {
				cout << setw(3) << i << " : " << setw(3) << part[i] << endl;
				}
			}


		//INT hash;
		INT TDO_depth;
		INT f_labeled = TRUE;


		TDO_depth = nb_pts + nb_lines;
	
		if (f_vv) {
			cout << "decomposition_projective_space before "
					"compute_TDO" << endl;
			}
		/*hash =*/ Inc.compute_TDO(Stack, ht0, TDO_depth,
				verbose_level + 2);
		if (f_vv) {
			cout << "decomposition_projective_space after "
					"compute_TDO" << endl;
			}

		if (FALSE) {
			Inc.print_partitioned(cout, Stack, f_labeled);
			}
		if (f_v) {
			Inc.get_and_print_decomposition_schemes(Stack);
			Stack.print_classes(cout);
			}


		}


	FREE_INT(part);
	FREE_INT(Mtx);
}

