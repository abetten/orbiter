/*
 * run_lifting.cpp
 *
 *  Created on: Mar 15, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;



int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;
	int f_k = FALSE;
	int k = 0;
	int f_sz_in = FALSE;
	int sz_in = 0;
	int f_sz_out = FALSE;
	int sz_out = 0;
	int f_slice = FALSE;
	int slice_r = 0;
	int slice_m = 0;
	int f_input_file = FALSE;
	const char *input_file_name = NULL;
	int f_output_prefix = FALSE;
	const char *output_prefix = NULL;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-sz_in") == 0) {
			f_sz_in = TRUE;
			sz_in = atoi(argv[++i]);
			cout << "-sz_in " << sz_in << endl;
			}
		else if (strcmp(argv[i], "-sz_out") == 0) {
			f_sz_out = TRUE;
			sz_out = atoi(argv[++i]);
			cout << "-sz_out " << sz_out << endl;
			}
		else if (strcmp(argv[i], "-slice") == 0) {
			f_slice = TRUE;
			slice_r = atoi(argv[++i]);
			slice_m = atoi(argv[++i]);
			cout << "-slice " << slice_r << " " << slice_m << endl;
			}
		else if (strcmp(argv[i], "-input_file") == 0) {
			f_input_file = TRUE;
			input_file_name = argv[++i];
			cout << "-input_file " << input_file_name << endl;
			}
		else if (strcmp(argv[i], "-output_prefix") == 0) {
			f_output_prefix = TRUE;
			output_prefix = argv[++i];
			cout << "-output_prefix " << output_prefix << endl;
			}
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "please use -k <k>" << endl;
		exit(1);
		}
	if (!f_sz_in) {
		cout << "please use -sz_in <sz_in>" << endl;
		exit(1);
		}
	if (!f_sz_out) {
		cout << "please use -sz_out <sz_out>" << endl;
		exit(1);
		}
	if (!f_input_file) {
		cout << "please use -input_file <input_file>" << endl;
		exit(1);
		}
	if (!f_output_prefix) {
		cout << "please use -output_prefix <output_prefix>" << endl;
		exit(1);
		}

	int h, j;
	file_io Fio;

	{
		ofstream fp("makefile_lifting");
		int nb_solutions;
		int *Solutions;
		int solution_size = sz_in;

		Fio.read_solutions_from_file(input_file_name,
			nb_solutions, Solutions, solution_size,
			verbose_level);

		cout << "found " << nb_solutions << " solutions" << endl;


				fp << "MY_PATH=~/DEV.19/GITHUB/orbiter" << endl;
				fp << "MY_PATH2=~/DEV.19/GITHUB/orbiter2" << endl;
				fp << "ORBITER=$(MY_PATH)/ORBITER" << endl;
				fp << "ORBITER2=$(MY_PATH2)/ORBITER2" << endl;
				fp << "SRC=$(ORBITER)/src" << endl;
				fp << "SRC2=$(ORBITER2)/src" << endl;


				fp << "ARCS_PATH=$(SRC)/apps/arcs" << endl;
				fp << "TOOLS_PATH=$(SRC)/apps/tools" << endl;
				fp << "PROJECTIVE_SPACE_PATH=$(SRC)/apps/projective_space" << endl;
				fp << "GROUPS_PATH=$(SRC)/apps/groups" << endl;
				fp << "SOLVER_PATH=$(SRC)/apps/solver" << endl;
				fp << "LINEAR_SPACES_PATH=$(SRC)/apps/linear_spaces" << endl;
				fp << "GRAPH_THEORY_PATH=$(SRC)/apps/graph_theory" << endl;

				fp << endl;
				fp << endl;


		fp << "all: ";
		for (h = 0; h < nb_solutions; h++) {
			fp << "lift_simeon_" << h << " ";
		}
		fp << ";" << endl;
		fp << endl;



		for (h = 0; h < nb_solutions; h++) {
			cout << "h=" << h << " / " << nb_solutions << endl;
			fp << "lift_simeon_" << h << ":" << endl;
			fp << "\t$(ARCS_PATH)/k_arc_lifting.out -v 2 -q 11 \\" << endl;
			fp << "\t-k " << k << " -sz " << sz_out << " \\" << endl;
			fp << "\t-arc \"";
			for (j = 0; j < solution_size; j++) {
				fp << Solutions[h * solution_size + j];
				if (j < solution_size - 1) {
					fp << ",";
				}
			}
			fp << "\" \\" << endl;
			fp << "\t-McKay \\" << endl;
			fp << "\t-dualize \\" << endl;
			fp << "\t-save_system simeon_" << h << ".diophant" << endl;
			fp << "\tmv solutions.txt solutions_" << h << ".txt" << endl;
			fp << "\t$(PROJECTIVE_SPACE_PATH)/canonical_form.out -v 2 -n 2 -q " << q << " \\" << endl;
			fp << "\t-input \\" << endl;
			fp << "\t-file_of_points solutions_" << h << ".txt \\" << endl;
			fp << "\t-end \\" << endl;
			fp << "\t-classify_nauty \\" << endl;
			fp << "\t-prefix " << output_prefix << "_" << h << " \\" << endl;
			fp << "\t-save " << output_prefix << "_" << h << " " << endl;
			fp << endl;
			fp << endl;
		} // next h
		fp << "merge:" << endl;
		fp << "\t$(PROJECTIVE_SPACE_PATH)/canonical_form.out -v 2 -n 2 -q " << q << " \\" << endl;
		fp << "\t-input \\" << endl;
		for (h = 0; h < nb_solutions; h++) {
			fp << "\t-file_of_points " << output_prefix << "_" << h << "_iso.txt \\" << endl;
		}
		fp << "\t-end \\" << endl;
		fp << "\t-classify_nauty \\" << endl;
		fp << "\t-prefix " << output_prefix << "_merged \\" << endl;
		fp << "\t-save " << output_prefix << "_merged " << endl;
		fp << endl;
		fp << endl;
		fp << "groups:" << endl;
		fp << "\t$(PROJECTIVE_SPACE_PATH)/canonical_form.out -v 2 -n 2 -q " << q << " \\" << endl;
		fp << "\t-input \\" << endl;
			fp << "\t-file_of_points " << output_prefix << "_merged_iso.txt \\" << endl;
		fp << "\t-end \\" << endl;
		fp << "\t-classify_nauty \\" << endl;
		fp << "\t-prefix " << output_prefix << "_merged \\" << endl;
		fp << "\t-latex " << output_prefix << "_report " << endl;
		fp << endl;
		fp << endl;
	}

#if 0
	lift_simeon_0:
		$(ARCS_PATH)/k_arc_lifting.out -v 2 -q 11 \
		-k 9 -sz 90 \
		-arc  \	"14,15,16,17,18,19,20,21,25,26,28,29,30,31,32,37,38,39,40,41,42,44,46,49,50,51,52,53,55,57,58,61,62,63,65,66,68,69,70,73,74,75,76,77,79,81,82,87,88,90,91,93,94,97,98,99,102,103,104,105,106,109,110	,112,113,114,115,116,118,121,123,124,125,126,128,129,130" \
		-McKay \
		-dualize \
		-save_system simeon_0.diophant
		mv solutions.txt solutions_0.txt

	# does not give (90,9)

	lift_simeon_31:
		$(ARCS_PATH)/k_arc_lifting.out -v 2 -q 11 \
		-k 9 -sz 90 \
		-arc 	\
		"13, 14 ,16 ,17 ,18 ,19 ,20 ,21 ,25, 26 ,28 ,29 ,30 ,31, 32, 33 ,37 ,38 ,39 ,40 ,41, 42, 44, 46, 49 51 52 53 54 55 57 58 61 62 63 65 66 68 69 70 73 74 75 76 77 79 81 82 87 88 91 92 93 94 97 98 99 102 103 104 105 106 109 110 113 114 115 116 118 123 124 125 126 127 128 129 130" \
		-McKay \
		-dualize \
		-save_system simeon_30.diophant
		mv solutions.txt solutions_30.txt

	lift_simeon_30_iso:
		$(PROJECTIVE_SPACE_PATH)/canonical_form.out -v 2 -n 2 -q 11 \
		-input \
		-file_of_points solutions_30.txt \
		-end \
		-classify_nauty \
		-prefix simeon_89_9_arcs \
		-save simeon_89_9_arcs \
		-latex
		pdflatex simeon_89_9_arcs_classification.tex
		mv simeon_89_9_arcs_iso.txt ./simeon_89_9_arcs_iso/simeon_30_89_9_arcs_iso.txt
		mv simeon_89_9_arcs_classification.pdf ./simeon_89_9_arcs_iso/simeon_30_89_9_arcs_classification.pdf
		open ./simeon_89_9_arcs_iso/simeon_30_89_9_arcs_classification.pdf
#endif


}




