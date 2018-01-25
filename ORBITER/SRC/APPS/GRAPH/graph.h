// graph.h
// 

typedef class graph_generator graph_generator;




// global data and global functions:

extern INT t0; // the system time when the program started

void usage(int argc, const char **argv);



class graph_generator {

public:

	generator *gen;

	action *A_base; // symmetric group on n vertices
	action *A_on_edges; // action on pairs
	
	INT f_n;	
	INT n; // number of vertices
	INT n2; // n choose 2

	INT *adjacency; // [n * n]
	
	//INT f_lex;
	
	INT f_regular;
	INT regularity;
	INT *degree_sequence; // [n]
	
	INT f_girth;
	INT girth;
	INT *neighbor; // [n]
	INT *neighbor_idx; // [n]
	INT *distance; // [n]

	INT f_list; // list whole orbits in the end
	INT f_list_all; // list whole orbits in the end
	INT f_draw_graphs;
	INT f_embedded;
	INT f_sideways;
	INT f_draw_graphs_at_level;
	INT level;
	double scale;
	INT f_x_stretch;
	double x_stretch;

	INT f_depth;
	INT depth;


	INT f_tournament;
	INT f_no_superking;

	INT f_draw_level_graph;
	INT level_graph_level;
	INT f_test_multi_edge;

	INT f_draw_poset;
	INT f_draw_full_poset;
	INT f_plesken;

	INT f_identify;
	INT identify_data[1000];
	INT identify_data_sz;
	

	

	graph_generator();
	~graph_generator();
	void read_arguments(int argc, const char **argv);
	void init(int argc, const char **argv);
	INT check_conditions(INT len, INT *S, INT verbose_level);
	INT check_conditions_tournament(INT len, INT *S, INT verbose_level);
	INT check_regularity(INT *S, INT len, INT verbose_level);
	INT compute_degree_sequence(INT *S, INT len);
	INT girth_check(INT *S, INT len, INT verbose_level);
	INT girth_test_vertex(INT *S, INT len, INT vertex, INT girth, INT verbose_level);
	void get_adjacency(INT *S, INT len, INT verbose_level);
	void print(INT *S, INT len);
	void print_score_sequences(INT level, INT verbose_level);
	void score_sequence(INT n, INT *set, INT sz, INT *score, INT verbose_level);
	void draw_graphs(INT level, double scale, INT xmax_in, INT ymax_in, INT xmax, INT ymax, INT f_embedded, INT f_sideways, INT verbose_level);

};


INT check_conditions(INT len, INT *S, void *data, INT verbose_level);
void print_set(INT len, INT *S, void *data);


