// graph_classify_main.cpp
// 
// Anton Betten
// Nov 15 2007
//
//
// 
//
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started


void usage(int argc, const char **argv);


int main(int argc, const char **argv)
{
	os_interface Os;

	t0 = Os.os_ticks();
	
	if (argc <= 1) {
		usage(argc, argv);
		exit(1);
		}

	{
	graph_classify Gen;
	int schreier_depth = 10000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	int depth;
	int f_embedded = TRUE;
	int f_sideways = FALSE;

	
	Gen.init(argc, argv);

	int verbose_level = Gen.gen->verbose_level;

	depth = Gen.gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		Gen.gen->verbose_level);
	cout << "Gen.gen->main returns depth=" << depth << endl;

	if (Gen.f_tournament) {
		Gen.print_score_sequences(depth, verbose_level);
		}

	//Gen.gen->draw_poset(Gen.gen->fname_base, depth,
	//Gen.n /* data1 */, f_embedded, Gen.gen->verbose_level);

	if (Gen.f_draw_poset) {
		Gen.gen->draw_poset(Gen.gen->fname_base, depth, 
			Gen.n /* data1 */, f_embedded, f_sideways, 
			Gen.gen->verbose_level);
		}


	if (Gen.f_draw_full_poset) {
		//double x_stretch = 0.4;
		Gen.gen->draw_poset_full(Gen.gen->fname_base, depth, 
			Gen.n /* data1 */, f_embedded, f_sideways, 
			Gen.x_stretch, Gen.gen->verbose_level);
		}

	//Gen.gen->print_data_structure_tex(depth, Gen.gen->verbose_level);

	if (Gen.f_plesken) {
		latex_interface L;
		int *P;
		int N;
		Gen.gen->Plesken_matrix_up(depth, P, N, Gen.gen->verbose_level);
		cout << "Plesken matrix up:" << endl;
		L.int_matrix_print_tex(cout, P, N, N);

		FREE_int(P);
		Gen.gen->Plesken_matrix_down(depth, P, N, Gen.gen->verbose_level);
		cout << "Plesken matrix down:" << endl;
		L.int_matrix_print_tex(cout, P, N, N);

		FREE_int(P);
		}

	if (Gen.f_list) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;
		
		Gen.gen->list_all_orbits_at_level(Gen.gen->depth, 
			FALSE, NULL, NULL, 
			f_show_orbit_decomposition,
			f_show_stab, f_save_stab, f_show_whole_orbit);
		}

	if (Gen.f_list_all) {
		int f_show_orbit_decomposition = FALSE;
		int f_show_stab = FALSE;
		int f_save_stab = FALSE;
		int f_show_whole_orbit = FALSE;
		int j;
		
		for (j = 0; j <= Gen.gen->depth; j++) {
			Gen.gen->list_all_orbits_at_level(j, 
				FALSE, NULL, NULL, 
				f_show_orbit_decomposition,
				f_show_stab, f_save_stab, f_show_whole_orbit);
			}
		}

	if (Gen.f_draw_graphs) {
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;
		int level;

		for (level = 0; level <= Gen.gen->depth; level++) {
			Gen.draw_graphs(level, Gen.scale,
					xmax_in, ymax_in, xmax, ymax,
					Gen.f_embedded, Gen.f_sideways,
					Gen.gen->verbose_level);
			}
		}

	if (Gen.f_draw_graphs_at_level) {
		int xmax_in = 1000000;
		int ymax_in = 1000000;
		int xmax = 1000000;
		int ymax = 1000000;

		cout << "before Gen.draw_graphs" << endl;
		Gen.draw_graphs(Gen.level, Gen.scale,
				xmax_in, ymax_in, xmax, ymax,
				Gen.f_embedded, Gen.f_sideways,
				verbose_level);
		cout << "after Gen.draw_graphs" << endl;
		}

	if (Gen.f_draw_level_graph) {
		Gen.gen->draw_level_graph(Gen.gen->fname_base,
				Gen.gen->depth, Gen.n /* data1 */,
				Gen.level_graph_level,
				f_embedded, f_sideways,
				Gen.gen->verbose_level - 3);
		}

	if (Gen.f_test_multi_edge) {
		Gen.gen->test_for_multi_edge_in_classification_graph(
				depth, Gen.gen->verbose_level);
		}
	if (Gen.f_identify) {
		int *transporter;
		int orbit_at_level;
		
		transporter = NEW_int(Gen.gen->Poset->A->elt_size_in_int);
		
		Gen.gen->identify(Gen.identify_data, Gen.identify_data_sz,
				transporter, orbit_at_level, Gen.gen->verbose_level);

		FREE_int(transporter);
		}

	int N, F, level;
	
	N = 0;
	F = 0;
	for (level = 0; level <= Gen.gen->depth; level++) {
		N += Gen.gen->nb_orbits_at_level(level);
		}
	for (level = 0; level < Gen.gen->depth; level++) {
		F += Gen.gen->nb_flag_orbits_up_at_level(level);
		}
	cout << "N=" << N << endl;
	cout << "F=" << F << endl;
	} // clean up graph_generator
	
	the_end(t0);
	//the_end_quietly(t0);
}

void usage(int argc, const char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;

	cout << "-n <n>" << endl;
	cout << "   number of vertices is <n>" << endl;
	cout << "-regular <d>" << endl;
	cout << "   regular of degree <d>" << endl;
	cout << "-girth <g>" << endl;
	cout << "   girth <g>" << endl;
	cout << "-list" << endl;
	cout << "   list orbits after classification is complete" << endl;
	cout << "-tournament" << endl;
	cout << "   Classify tournaments instead" << endl;

	poset_classification gen;
	
	gen.usage();

}



