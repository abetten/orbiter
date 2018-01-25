// translation_plane.h
// 
// Anton Betten
// November 19, 2009
//
//
// moved here: July 19, 2014
//
//


#if 0
// ##################################################################################################
// extending.C
// ##################################################################################################


class extending {
public:
	BYTE label[1000];
	INT the_case;
	translation_plane *T;
	
	
	group *Aut;
	
	
	INT *S; // [target_size]
	INT starter_size;
	INT target_size;
	
	schreier *ORB;
	INT Nb_good_orbits;
	INT *Good_orbits;
	

	INT nb_points;
	INT *points;
	INT adjacency_length;
	INT *adjacency; // nb_points choose 2
	
	ofstream *fp_out;
	ofstream *fp_summary;
	//BYTE fname_out[1000];
	//BYTE fname_summary[1000];
	INT case_no;

	INT dt[5];
	INT dt_total;
	

	INT N; // = i_power_j(q, k)
	INT nb_colors; // = N - (starter_size - 1);
	INT *open_colors;
		// [N]
	INT *open_colors_inv;
		// [N]
	INT *point_color;
		// [nb_points]

	INT *solution_point_color; // [target_size]
	


	// the following three arrays are allocated by extension_data::init_colors

	INT *f_color_satisfied; // maintained by extension_data::add_point/delete_point
	INT *color_chosen_at_depth; // maintained by extension_data::find_candidates
	INT *color_frequency; // maintained by extension_data::find_candidates
	
	clique_finder *CF;
	//clique_finder_interface *CFI;
	
	INT f_candidate_check_func;
	INT (*candidate_check_func)(INT len, INT *S, void *data, INT verbose_level);
	void *candidate_check_data;

	// added Dec. 22, 2011 for  find_candidates():
	INT f_choice_allocated;
	INT *remaining_colors; // [N]
	INT *remaining_color_idx; // [N]
	INT nb_remaining_colors;
	INT *live_points_by_color; // [N * nb_points]
	INT *f_live_points_by_color_active; // [N * nb_points]
	INT *nb_live_points_by_color; // [N]
	INT *nb_live_points_by_color_active; // [N]

	INT *buddy_table; // [nb_pts * N];
	INT *f_point_is_active; // [nb_points]
	INT nb_active_points;


extending();
~extending();
void null();
void free();
void init(BYTE *label, INT the_case, 
	translation_plane *T, 
	INT starter_size, INT *set, INT target_size, INT verbose_level);
void init_candidates(INT nb_candidates, INT *candidates, INT verbose_level);
void init_clique_finder(INT verbose_level);
void init_callbacks_and_CFI(INT verbose_level);
void do_search(INT verbose_level);
void extend(BYTE *label, INT the_case, 
	translation_plane *T, 
	INT starter_size, INT *set, INT target_size, 
	const BYTE *stab_ascii, INT &search_steps, INT &decision_steps, INT &nb_sol, 
	INT f_lexorder, 
	INT print_interval, 
	INT f_compute_points_only, ofstream *fp_points_out, 
	INT f_use_points, ifstream *fp_points_in, 
	INT f_has_candidates, INT nb_candidates, INT *candidates, 
	INT f_write_graph_file, 
	INT f_draw_graph, 
	INT f_write_tree, INT f_decision_nodes_only, 
	INT verbose_level);
INT setup(group *Aut, 
	INT f_lexorder, 
	INT f_use_points, ifstream *fp, 
	INT f_has_candidates, INT nb_candidates, INT *candidates, 
	INT verbose_level);
void compute_points(group *Aut, 
	INT f_lexorder, ofstream *fp, INT verbose_level);
void find_good_orbits(schreier &Orb, 
	INT *Good_orbits, INT &nb_good_orbits, 
	INT *&Pts, INT &nb_pts, INT f_lexorder, INT verbose_level);
void compute_adjacencies(INT verbose_level);
void clique_found(INT *current_clique, INT verbose_level);
void add_point(INT pt, 
	INT current_clique_size, INT *current_clique, 
	INT verbose_level);
void delete_point(INT pt, 
	INT current_clique_size, INT *current_clique, 
	INT verbose_level);
INT find_candidates(INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);
INT is_adjacent(INT pt1, INT pt2, INT verbose_level);
void write_graph(ofstream &ost, BYTE *label, 
	INT point_offset, INT f_point_labels);
void degree_test(INT nb_points, INT *points, INT *adjacency, 
	INT clique_size, INT *f_deleted);
void latex_degree_sequence(INT nb_points, INT *points, INT *adjacency);
void latex_adjacency_matrix(INT nb_points, INT *points, 
	INT *adjacency, INT *point_color, INT *f_deleted);
void after_reduction(clique_finder *CF, 
		INT depth, INT nb_live_points, INT verbose_level);
INT init_colors(INT verbose_level);
void allocate_color_arrays(INT verbose_level);
void allocate_choice();
void free_choice();
void print_point_colors();
INT find_candidates_sophisticated(INT current_clique_size, 
	INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);
};

void call_back_clique_found(clique_finder *CF, INT verbose_level);
void call_back_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
void call_back_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level);
INT call_back_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);
INT call_back_find_candidates_sophisticated(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level);
INT call_back_is_adjacent(clique_finder *CF, 
	INT pt1, INT pt2, INT verbose_level);
void call_back_after_reduction(clique_finder *CF, 
		INT depth, INT nb_live_points, INT verbose_level);
INT clique_setup(INT *starter, INT starter_size, 
	INT *point_set, INT point_set_size, 
	INT target_size, 
	void *data, 
	void *&data_local, INT verbose_level);
void clique_cleanup(void *data, void *data_local, INT verbose_level);

void translation_plane_do_extend(translation_plane *T, INT starter_size, 
		INT r, INT m, INT f_casenumbers, 
		INT f_lexorder,
		INT print_interval,  
		INT f_compute_points_only, INT f_use_points, 
		INT f_read_candidates_file, const BYTE *candidates_fname, 
		INT f_mem_dump_by_size, INT f_mem_dump, 
		INT f_write_graph_file, 
		INT f_draw_graph, 
		INT f_write_tree, INT f_decision_nodes_only, 
		INT verbose_level);
void translation_plane_extend_simple(translation_plane *T, INT starter_size, 
		INT *starter, 
		INT f_lexorder,
		INT f_write_graph_file, 
		INT f_draw_graph, 
		INT f_write_tree, INT f_decision_nodes_only, 
		INT verbose_level);
void translation_plane_init_clique(translation_plane *T, generator *gen, INT clique_level, INT verbose_level);
#endif


// in translation_plane_main.C:

extern INT t0;

void print_translation_plane(INT len, INT *S, void *data);


