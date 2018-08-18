// graph.C
// 
// Anton Betten
// Nov 15 2007
//
//
// 
//
//

#include "orbiter.h"
#include "graph.h"

// global data:

INT t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	if (argc <= 1) {
		usage(argc, argv);
		exit(1);
		}

	{
	graph_generator Gen;
	INT schreier_depth = 10000;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	INT depth;
	INT f_embedded = TRUE;
	INT f_sideways = FALSE;

	
	Gen.init(argc, argv);

	INT verbose_level = Gen.gen->verbose_level;

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
		INT *P;
		INT N;
		Gen.gen->Plesken_matrix_up(depth, P, N, Gen.gen->verbose_level);
		cout << "Plesken matrix up:" << endl;
		INT_matrix_print_tex(cout, P, N, N);

		FREE_INT(P);
		Gen.gen->Plesken_matrix_down(depth, P, N, Gen.gen->verbose_level);
		cout << "Plesken matrix down:" << endl;
		INT_matrix_print_tex(cout, P, N, N);

		FREE_INT(P);
		}

	if (Gen.f_list) {
		INT f_show_orbit_decomposition = FALSE;
		INT f_show_stab = FALSE;
		INT f_save_stab = FALSE;
		INT f_show_whole_orbit = FALSE;
		
		Gen.gen->list_all_orbits_at_level(Gen.gen->depth, 
			FALSE, NULL, NULL, 
			f_show_orbit_decomposition,
			f_show_stab, f_save_stab, f_show_whole_orbit);
		}

	if (Gen.f_list_all) {
		INT f_show_orbit_decomposition = FALSE;
		INT f_show_stab = FALSE;
		INT f_save_stab = FALSE;
		INT f_show_whole_orbit = FALSE;
		INT j;
		
		for (j = 0; j <= Gen.gen->depth; j++) {
			Gen.gen->list_all_orbits_at_level(j, 
				FALSE, NULL, NULL, 
				f_show_orbit_decomposition,
				f_show_stab, f_save_stab, f_show_whole_orbit);
			}
		}

	if (Gen.f_draw_graphs) {
		INT xmax_in = 1000000;
		INT ymax_in = 1000000;
		INT xmax = 1000000;
		INT ymax = 1000000;
		INT level;

		for (level = 0; level <= Gen.gen->depth; level++) {
			Gen.draw_graphs(level, Gen.scale,
					xmax_in, ymax_in, xmax, ymax,
					Gen.f_embedded, Gen.f_sideways,
					Gen.gen->verbose_level);
			}
		}

	if (Gen.f_draw_graphs_at_level) {
		INT xmax_in = 1000000;
		INT ymax_in = 1000000;
		INT xmax = 1000000;
		INT ymax = 1000000;

		Gen.draw_graphs(Gen.level, Gen.scale,
				xmax_in, ymax_in, xmax, ymax,
				Gen.f_embedded, Gen.f_sideways,
				Gen.gen->verbose_level);
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
		INT *transporter;
		INT orbit_at_level;
		
		transporter = NEW_INT(Gen.gen->A->elt_size_in_INT);
		
		Gen.gen->identify(Gen.identify_data, Gen.identify_data_sz,
				transporter, orbit_at_level, Gen.gen->verbose_level);

		FREE_INT(transporter);
		}

	INT N, F, level;
	
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

	generator gen;
	
	gen.usage();

}



