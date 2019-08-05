// transpose.cpp
// 
// Anton Betten
// 1/2/13
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


#include "regular_ls.h"


// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_file_in = FALSE;
	const char *fname_in;
	int f_file_out = FALSE;
	const char *fname_out;
	int f_parameters = FALSE;
	int v, r, b, k;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file_in") == 0) {
			f_file_in = TRUE;
			fname_in = argv[++i];
			cout << "-file_in " << fname_in << endl;
			}
		else if (strcmp(argv[i], "-file_out") == 0) {
			f_file_out = TRUE;
			fname_out = argv[++i];
			cout << "-file_out " << fname_out << endl;
			}
		else if (strcmp(argv[i], "-parameters") == 0) {
			f_parameters = TRUE;
			v = atoi(argv[++i]);
			r = atoi(argv[++i]);
			b = atoi(argv[++i]);
			k = atoi(argv[++i]);
			cout << "-parameters " << v << " " << r
					<< " " << b << " " << k << endl;
			}

		}

#if 0
	{
		int w[3] = {20,22,23};
		int z[3];
		int rk;
		rk = rank_k_subset(w, 24, 3);
		cout << "The set ";
		int_vec_print(cout, w, 3);
		cout << " has rank " << rk << endl;
		unrank_k_subset(rk, z, 24, 3);
		cout << " and unranks to ";
		int_vec_print(cout, w, 3);
		cout << endl;
		//exit(1);
		
	}
#endif

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	if (!f_file_in) {
		cout << "Please use option -file_in <fname_in>" << endl;
		exit(1);
		}
	if (!f_parameters) {
		cout << "Please use option -parameters <v> <r> <b> <k>" << endl;
		exit(1);
		}
	int f_casenumbers = FALSE;
	int nb_cases;
	int *Set_sizes;
	int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int h, j, a, u, ii, jj;
	int *M1;
	int *M2;
	int *v1;
	int *v2;
	int *new_set;
	int **New_sets;
	int vb;
	int *Adj;
	int nb_points;
	int *points;
	combinatorics_domain Combi;
	file_io Fio;

	action *Ab;
	
	Fio.read_and_parse_data_file_fancy(fname_in,
		f_casenumbers, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 2);

	cout << "Read " << nb_cases << " orbits from file "
			<< fname_in << endl;

	points = NEW_int(b);
	nb_points = b;
	for (i = 0; i < b; i++) {
		points[i] = i;
		}
	vb = v + b;
	M1 = NEW_int(v * b);
	M2 = NEW_int(b * v);
	Adj = NEW_int(vb * vb);
	v1 = NEW_int(k);
	v2 = NEW_int(r);
	New_sets = NEW_pint(nb_cases);
	
	Ab = NEW_OBJECT(action);
	Ab->init_symmetric_group(b /* degree */, verbose_level);
	

	{
	ofstream fp(fname_out);

	fp << "# " << v << endl;


	for (h = 0; h < nb_cases; h++) {

		new_set = NEW_int(v);
		longinteger_object ago1, ago2, ago3, ago4;
		longinteger_domain D;

		ago1.create_from_base_10_string(Ago_ascii[h],
				0 /* verbose_level */);
		if (f_v) {
			cout << "orbit " << h << " / " << nb_cases
					<< " with ago = " << ago1 << ":" << endl;
			}
		if (Set_sizes[h] != b) {
			cout << "Set_sizes[h] != b" << endl;
			exit(1);
			}
		if (f_vv) {
			int_vec_print(cout, Sets[h], b);
			cout << endl;
			}
		for (i = 0; i < v * b; i++) {
			M1[i] = 0;
			}
		for (j = 0; j < b; j++) {
			a = Sets[h][j];
			Combi.unrank_k_subset(a, v1, v, k);
			if (f_vvv) {
				cout << a << " = ";
				int_vec_print(cout, v1, k);
				cout << endl;
				} 
			for (u = 0; u < k; u++) {
				i = v1[u];
				M1[i * b + j] = 1;
				}
			}
		if (f_vvv) {
			int_matrix_print(M1, v, b);
			}
		for (i = 0; i < v; i++) {
			u = 0;
			for (j = 0; j < b; j++) {
				if (M1[i * b + j]) {
					v2[u++] = j;
					}
				}
			if (u != r) {
				cout << "row " << i << " u != r" << endl;
				exit(1);
				}
			a = Combi.rank_k_subset(v2, b, r);
			new_set[i] = a;
			}
		if (f_vv) {
			cout << "Transposed:" << endl;
			int_vec_print(cout, new_set, v);
			cout << endl;
			}
		New_sets[h] = new_set;
		for (i = 0; i < v; i++) {
			for (j = 0; j < b; j++) {
				M2[j * v + i] = M1[i * b + j];
				}
			}
		if (f_vvv) {
			int_matrix_print(M2, b, v);
			}
		for (i = 0; i < vb * vb; i++) {
			Adj[i] = 0;
			}
		for (i = 0; i < vb; i++) {
			Adj[i * vb + i] = 0;
			}
		for (ii = 0; ii < b; ii++) {
			for (jj = 0; jj < v; jj++) {
				i = ii;
				j = b + jj;
				a = M2[ii * v + jj];
				Adj[i * vb + j] = a;
				Adj[j * vb + i] = a;
				}
			}
		int parts[2];
		int nb_parts = 2;

		parts[0] = b;
		parts[1] = v;

		action *At;
		if (f_vvv) {
			cout << "Adjacency matrix:" << endl;
			int_matrix_print(Adj, vb, vb);
			}
		int *labeling;

		labeling = NEW_int(vb);

		nauty_interface Nauty;

		At = Nauty.create_automorphism_group_of_graph_with_partition_and_labeling(
			vb, FALSE, NULL, Adj, 
			nb_parts, parts, 
			labeling, 
			verbose_level - 2);
		FREE_int(labeling);
		
		At->Sims->group_order(ago2);

		if (D.compare(ago1, ago2)) {
			cout << "Group orders differ: ago1=" << ago1
					<< " ago2=" << ago2 << endl;
			exit(1);
			}
		
		action *Ar;
		int f_induce_action = TRUE;
		
		//Ar = NEW_OBJECT(action);


		Ar = At->create_induced_action_by_restriction(
			At->Sims,
			nb_points, points,
			f_induce_action,
			verbose_level);
		Ar->Sims->group_order(ago3);
		if (D.compare(ago1, ago3)) {
			cout << "Group orders differ: ago1=" << ago1
					<< " ago3=" << ago3 << endl;
			exit(1);
			}

		fp << v << " ";
		for (i = 0; i < v; i++) {
			fp << New_sets[h][i] << " ";
			}


		{
		vector_ge SG;
		vector_ge SG2;
		int *tl;
		int *tl2;

		tl = NEW_int(Ar->base_len());
		tl2 = NEW_int(Ab->base_len());
		Ar->Sims->extract_strong_generators_in_order(SG, tl, verbose_level - 2);
		
		sims *Stab2;

		Stab2 = Ab->create_sims_from_generators_with_target_group_order(
			&SG, ago1, verbose_level);
		Stab2->extract_strong_generators_in_order(SG2, tl2, verbose_level - 2);
		cout << "tl2=";
		int_vec_print(cout, tl2, Ab->base_len());
		cout << endl;

		group G;

		G.init(Ab);
		G.init_strong_generators(SG2, tl2);
		G.schreier_sims(0);
		G.group_order(ago4);
		if (D.compare(ago1, ago4)) {
			cout << "Group orders differ: ago1=" << ago1
					<< " ago4=" << ago4 << endl;
			exit(1);
			}
		if (ago4.is_one()) {
			ago4.print_not_scientific(fp);
			fp << endl;
			//f << go << endl;
			}
		else {
			G.code_ascii(FALSE);
			ago4.print_not_scientific(fp);
			fp << " " << G.ascii_coding << endl;
			//f << go << " " << G.ascii_coding << endl;
			}
		FREE_int(tl);
		FREE_int(tl2);
		FREE_OBJECT(Stab2);
		}

		
		FREE_OBJECT(At);
		FREE_OBJECT(Ar);
		}

	fp << -1 << endl;
	}
	cout << "Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;
	
	the_end(t0);
	//the_end_quietly(t0);
}


