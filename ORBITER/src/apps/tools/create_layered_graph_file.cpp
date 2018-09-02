// create_layered_graph_file.C
// 
// Anton Betten
// July 2, 2016
//
//
// 
// creates a layered graph file from a text file
// which was created by DISCRETA/sgls2.C
// for an example, see the bottom of this file.
//
//

#include "orbiter.h"


void create_graph_from_file(layered_graph *&LG,
		INT f_grouping, double x_stretch, const char *fname);


int main(int argc, const char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_file = FALSE;
	const char *fname = NULL;
	INT f_grouping = FALSE;
	double x_stretch = 0.7;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-grouping") == 0) {
			f_grouping = TRUE;
			sscanf(argv[++i], "%lf", &x_stretch);
			cout << "-grouping" << x_stretch << endl;
			}
		}
	
	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}
	layered_graph *LG;

	create_graph_from_file(LG, f_grouping, x_stretch, fname);


	char fname_out[1000];

	sprintf(fname_out, "%s", fname);

	replace_extension_with(fname_out, ".layered_graph");
	
	
	LG->write_file(fname_out, 0 /*verbose_level*/);

	cout << "Written file " << fname_out << " of size "
			<< file_size(fname_out) << endl;

	delete LG;
}


void create_graph_from_file(layered_graph *&LG,
		INT f_grouping, double x_stretch, const char *fname)
{
	LG = NEW_OBJECT(layered_graph);
	INT nb_layer;
	INT *Nb;
	INT *Nb_orbits;
	INT **Orbit_length;
	INT i, l1, n1, l2, n2, nb_v = 0, c = 0, a;

	ifstream fp(fname);
	fp >> nb_layer;
	Nb = NEW_INT(nb_layer);
	Nb_orbits = NEW_INT(nb_layer);
	Orbit_length = NEW_PINT(nb_layer);
	nb_v = 0;
	for (i = 0; i < nb_layer; i++) {
		fp >> Nb[i];
		nb_v += Nb[i];
		}
	LG->add_data1(0, 0/*verbose_level*/);
	LG->init(nb_layer, Nb, "", 0);
	LG->place(0 /*verbose_level*/);


	for (l1 = 0; l1 < nb_layer; l1++) {
		for (n1 = 0; n1 < Nb[l1]; n1++) {
			fp >> a;

			char text[1000];

			sprintf(text, "%ld", a);
			LG->add_text(l1, n1, text, 0/*verbose_level*/);
			}
		}

	for (l1 = 0; l1 < nb_layer; l1++) {
		fp >> Nb_orbits[l1];
		Orbit_length[l1] = NEW_INT(Nb_orbits[l1]);
		for (i = 0; i < Nb_orbits[l1]; i++) {
			fp >> Orbit_length[l1][i];
			}
		}
	
	while (TRUE) {
		fp >> l1;
		if (l1 == -1) {
			break;
			}
		fp >> n1;
		fp >> l2;
		fp >> n2;
		LG->add_edge(l1, n1, l2, n2, 0 /*verbose_level*/);
		c++;
		}
	if (f_grouping) {
		LG->place_with_grouping(Orbit_length,
				Nb_orbits, x_stretch, 0 /*verbose_level*/);
		}
	cout << "created a graph with " << nb_v
			<< " vertices and " << c << " edges" << endl;
	
}

// example file created in DISCRETA/sgls2.C for the subgroup lattice of Sym(4):
#if 0
5
1 13 11 4 1 
1 2 2 2 2 2 2 3 3 3 3 2 2 2 4 4 4 6 6 6 6 4 4 4 4 8 8 8 12 24 
1 1 
3 6 4 3 
4 3 4 3 1 
2 3 1 
1 1 
0 0 1 0
0 0 1 1
0 0 1 2
0 0 1 3
0 0 1 4
0 0 1 5
0 0 1 6
0 0 1 7
0 0 1 8
0 0 1 9
0 0 1 10
0 0 1 11
0 0 1 12
1 0 2 0
1 0 2 3
1 0 2 4
1 1 2 1
1 1 2 3
1 1 2 5
1 2 2 2
1 2 2 3
1 2 2 6
1 3 2 0
1 3 2 5
1 3 2 6
1 4 2 2
1 4 2 4
1 4 2 5
1 5 2 1
1 5 2 4
1 5 2 6
1 6 2 3
1 6 3 3
1 7 2 5
1 7 3 3
1 8 2 6
1 8 3 3
1 9 2 4
1 9 3 3
1 10 2 0
1 10 2 7
1 10 2 10
1 11 2 2
1 11 2 8
1 11 2 10
1 12 2 1
1 12 2 9
1 12 2 10
2 0 3 0
2 1 3 1
2 2 3 2
2 3 4 0
2 4 4 0
2 5 4 0
2 6 4 0
2 7 3 0
2 8 3 2
2 9 3 1
2 10 3 0
2 10 3 1
2 10 3 2
2 10 3 3
3 0 4 0
3 1 4 0
3 2 4 0
3 3 4 0
-1
#endif


