// graph.h
// 

typedef class graph_generator graph_generator;




// global data and global functions:

extern int t0; // the system time when the program started

void usage(int argc, const char **argv);

//! classification of graphs and tournaments



class graph_generator {

public:

	poset *Poset;
	poset_classification *gen;

	action *A_base; // symmetric group on n vertices
	action *A_on_edges; // action on pairs
	
	int f_n;	
	int n; // number of vertices
	int n2; // n choose 2

	int *adjacency; // [n * n]
	
	//int f_lex;
	
	int f_regular;
	int regularity;
	int *degree_sequence; // [n]
	
	int f_girth;
	int girth;
	int *neighbor; // [n]
	int *neighbor_idx; // [n]
	int *distance; // [n]

	int f_list; // list whole orbits in the end
	int f_list_all; // list whole orbits in the end
	int f_draw_graphs;
	int f_embedded;
	int f_sideways;
	int f_draw_graphs_at_level;
	int level;
	double scale;
	int f_x_stretch;
	double x_stretch;

	int f_depth;
	int depth;


	int f_tournament;
	int f_no_superking;

	int f_draw_level_graph;
	int level_graph_level;
	int f_test_multi_edge;

	int f_draw_poset;
	int f_draw_full_poset;
	int f_plesken;

	int f_identify;
	int identify_data[1000];
	int identify_data_sz;
	

	

	graph_generator();
	~graph_generator();
	void read_arguments(int argc, const char **argv);
	void init(int argc, const char **argv);
	int check_conditions(int len, int *S, int verbose_level);
	int check_conditions_tournament(int len, int *S, int verbose_level);
	int check_regularity(int *S, int len, int verbose_level);
	int compute_degree_sequence(int *S, int len);
	int girth_check(int *S, int len, int verbose_level);
	int girth_test_vertex(int *S, int len, int vertex, int girth, int verbose_level);
	void get_adjacency(int *S, int len, int verbose_level);
	void print(int *S, int len);
	void print_score_sequences(int level, int verbose_level);
	void score_sequence(int n, int *set, int sz, int *score, int verbose_level);
	void draw_graphs(int level, double scale, int xmax_in, int ymax_in, int xmax, int ymax, int f_embedded, int f_sideways, int verbose_level);

};


int check_conditions(int len, int *S, void *data, int verbose_level);
void print_set(int len, int *S, void *data);


