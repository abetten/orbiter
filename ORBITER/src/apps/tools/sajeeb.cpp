// sajeeb.C
// 
// Anton Betten
// Febryary 5, 2018
//
// 
//


#include "sajeeb.h"



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_list_of_cases = FALSE;
	const char *fname_list_of_cases = NULL;
	const char *fname_template = NULL;
	int f_output_file = FALSE;
	const char *output_file = NULL;
	

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-list_of_cases") == 0) {
			f_list_of_cases = TRUE;
			fname_list_of_cases = argv[++i];
			fname_template = argv[++i];
			cout << "-list_of_cases " << fname_list_of_cases << " " << fname_template << endl;
			}
		else if (strcmp(argv[i], "-output_file") == 0) {
			f_output_file = TRUE;
			output_file = argv[++i];
			cout << "-output_file " << output_file << endl;
			}
		}

	if (!f_list_of_cases) {
		cout << "please use -f_list_of_cases <f_list_of_cases>" << endl;
		exit(1);
		}
	if (!f_output_file) {
		cout << "please use -f_output_file <f_output_file>" << endl;
		exit(1);
		}

	int *list_of_cases;
	int nb_cases;
	char fname_sol[1000];
	char fname_stats[1000];
	
	if (f_output_file) {
		sprintf(fname_sol, "%s", output_file);
		sprintf(fname_stats, "%s", output_file);
		replace_extension_with(fname_stats, "_stats.csv");
		}
	else {
		sprintf(fname_sol, "solutions_%s", fname_list_of_cases);
		sprintf(fname_stats, "statistics_%s", fname_list_of_cases);
		replace_extension_with(fname_stats, ".csv");
		}
	read_set_from_file(fname_list_of_cases, list_of_cases, nb_cases, verbose_level);
	cout << "nb_cases=" << nb_cases << endl;

	colored_graph_all_cliques_list_of_cases(list_of_cases, nb_cases, FALSE /*f_output_solution_raw */, 
		fname_template, 
		fname_sol, fname_stats, 
		FALSE /*f_maxdepth*/, 0 /*maxdepth*/, 
		FALSE /*f_prefix*/, NULL /*prefix*/, 
		1000 /*print_interval */, 
		verbose_level);
	
	delete [] list_of_cases;
	cout << "all_rainbow_cliques.out written file " << fname_sol << " of size " << file_size(fname_sol) << endl;
	cout << "all_rainbow_cliques.out written file " << fname_stats << " of size " << file_size(fname_stats) << endl;

}

void read_set_from_file(const char *fname, int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	
	if (f_v) {
		cout << "read_set_from_file opening file " << fname << " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);
	
	f >> set_size;
	the_set = new int[set_size];
	
	for (i = 0; i < set_size; i++) {
		f >> a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i << "-th number is " << a << endl;
			//}
		if (a == -1)
			break;
		the_set[i] = a;
		}
	if (f_v) {
		cout << "read a set of size " << set_size << " from file " << fname << endl;
		}
#if 0
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
#endif
}

void replace_extension_with(char *p, const char *new_ext)
{
	int i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			p[i] = 0;
			break;
			}
		}
	strcat(p, new_ext);
}


#include <cstdio>
#include <sys/types.h>
#ifdef SYSTEMUNIX
#include <unistd.h>
#endif
#include <fcntl.h>



int file_size(const char *name)
{
	//cout << "file_size fname=" << name << endl;
#ifdef SYSTEMUNIX
	int handle, size;
	
	//cout << "Unix mode" << endl;
	handle = open(name, O_RDWR/*mode*/);
	size = lseek((int) handle, 0L, SEEK_END);
	close((int) handle);
	return(size);
#endif
#ifdef SYSTEMMAC
	int handle, size;
	
	//cout << "Macintosh mode" << endl;
	handle = open(name, O_RDONLY);
		/* THINK C Unix Lib */
	size = lseek(handle, 0L, SEEK_END);
		/* THINK C Unix Lib */
	close(handle);
	return(size);
#endif
#ifdef SYSTEMWINDOWS

	//cout << "Windows mode" << endl;

	int handle = _open (name,_O_RDONLY);
	int size   = _lseek (handle,0,SEEK_END);
	close (handle);
	return (size);
#endif
}

void int_vec_copy(int *from, int *to, int len)
{
	int i;
	int *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
		}
}

void int_vec_zero(int *v, int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 0;
		}
}

void int_vec_print(ostream &ost, int *v, int len)
{
	int i;
	
	if (len > 50) {
		ost << "( ";
		for (i = 0; i < 50; i++) {
			ost << v[i];
			if (i < len - 1)
				ost << ", ";
			}
		ost << "...";
		for (i = len - 3; i < len; i++) {
			ost << v[i];
			if (i < len - 1)
				ost << ", ";
			}
		ost << " )";
		}
	else {
		int_vec_print_fully(ost, v, len);
		}
}

void int_vec_print_fully(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " )";
}

void int_set_print(int *v, int len)
{
	int_set_print(cout, v, len);
}

void int_set_print(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " }";
}

void print_set(ostream &ost, int size, int *set)
{
	int i;
	
	ost << "{ ";
	for (i = 0; i < size; i++) {
		ost << set[i];
		if (i < size - 1)
			ost << ", ";
		}
	ost << " }";
}



void int_vec_swap_points(int *list, int *list_inv, int idx1, int idx2)
{
	int p1, p2;
	
	if (idx1 == idx2) {
		return;
		}
	p1 = list[idx1];
	p2 = list[idx2];
	list[idx1] = p2;
	list[idx2] = p1;
	list_inv[p1] = idx2;
	list_inv[p2] = idx1;
}





uchar *bitvector_allocate(int length)
{
	int l, i;
	uchar *p;

	l = (length + 7) >> 3;
	p = new uchar [l];
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

uchar *bitvector_allocate_and_coded_length(int length, int &coded_length)
{
	int l, i;
	uchar *p;

	l = (length + 7) >> 3;
	coded_length = l;
	p = new uchar [l];
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

void bitvector_m_ii(uchar *bitvec, int i, int a)
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
}

void bitvector_set_bit(uchar *bitvec, int i)
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	x |= mask;
}

int bitvector_s_i(uchar *bitvec, int i)
// returns 0 or 1
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (x & mask) {
		return 1;
		}
	else {
		return 0;
		}
}


int ij2k(int i, int j, int n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
		}
	if (i > j) {
		return ij2k(j, i, n);
		}
	else {
		return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
		}
}

void k2ij(int k, int & i, int & j, int n)
{
	int ii, k_save = k;
	
	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			i = ii;
			j = k + ii + 1;
			return;
			}
		k -= (n - ii - 1);
		}
	cout << "k2ij: k too large: k = " << k_save << " n = " << n << endl;
	exit(1);
}

void get_extension_if_present(const char *p, char *ext)
{
	int i, l = strlen(p);
	
	//cout << "get_extension_if_present " << p << " l=" << l << endl;
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			strcpy(ext, p + i);
			return;
			}
		}
}

void chop_off_extension_if_present(char *p, const char *ext)
{
	int l1 = strlen(p);
	int l2 = strlen(ext);
	
	if (l1 > l2 && strcmp(p + l1 - l2, ext) == 0) {
		p[l1 - l2] = 0;
		}
}

void fwrite_int4(FILE *fp, int a)
{
	int4 I;

	I = (int4) a;
	fwrite(&I, 1 /* size */, 4 /* items */, fp);
}

int4 fread_int4(FILE *fp)
{
	int4 I;

	fread(&I, 1 /* size */, 4 /* items */, fp);
	return I;
}

void fwrite_uchars(FILE *fp, uchar *p, int len)
{
	fwrite(p, 1 /* size */, len /* items */, fp);
}

void fread_uchars(FILE *fp, uchar *p, int len)
{
	fread(p, 1 /* size */, len /* items */, fp);
}



void colored_graph_all_cliques_list_of_cases(int *list_of_cases, int nb_cases, int f_output_solution_raw, 
	const char *fname_template, 
	const char *fname_sol, const char *fname_stats, 
	int f_maxdepth, int maxdepth, 
	int f_prefix, const char *prefix, 
	int print_interval, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c;
	int Search_steps = 0, Decision_steps = 0, Nb_sol = 0, Dt = 0;
	int search_steps, decision_steps, nb_sol, dt;
	char fname[1000];
	char fname_tmp[1000];

	{
	ofstream fp(fname_sol);
	ofstream fp_stats(fname_stats);
	
	fp_stats << "i,Case,Nb_sol,Nb_vertices,search_steps,decision_steps,dt" << endl;
	for (i = 0; i < nb_cases; i++) {
		colored_graph CG;

		c = list_of_cases[i];
		if (f_v) {
			cout << "colored_graph_all_cliques_list_of_cases case " << i << " / " << nb_cases << " which is " << c << endl;
			}
		sprintf(fname_tmp, fname_template, c);
		if (f_prefix) {
			sprintf(fname, "%s%s", prefix, fname_tmp);
			}
		else {
			strcpy(fname, fname_tmp);
			}
		CG.load(fname, verbose_level - 2);

		//CG.print();

		fp << "# start case " << c << endl;

		CG.all_rainbow_cliques(&fp, f_output_solution_raw, 
			f_maxdepth, maxdepth, 
			FALSE /* f_restrictions */, NULL /* restrictions */, 
			FALSE /* f_tree */, FALSE /* f_decision_nodes_only */, NULL /* fname_tree */,
			print_interval, 
			search_steps, decision_steps, nb_sol, dt, 
			verbose_level - 1);
		fp << "# end case " << c << " " << nb_sol << " " << search_steps 
				<< " " << decision_steps << " " << dt << endl;
		fp_stats << i << "," << c << "," << nb_sol << "," << CG.nb_points << "," << search_steps << "," << decision_steps << "," << dt << endl;
		Search_steps += search_steps;
		Decision_steps += decision_steps;
		Nb_sol += nb_sol;
		Dt += dt;
		
		}
	fp << -1 << " " << Nb_sol << " " << Search_steps 
				<< " " << Decision_steps << " " << Dt << endl;
	fp_stats << "END" << endl;
	}
	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_cases done Nb_sol=" << Nb_sol << endl;
		}
}


void call_back_clique_found_using_file_output(clique_finder *CF, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	//cout << "call_back_clique_found_using_file_output" << endl;
	
	file_output *FO = (file_output *) CF->call_back_clique_found_data;
	//clique_finder *CF = (clique_finder *) FO->user_data;

	FO->write_line(CF->target_depth, CF->current_clique, verbose_level);
}




colored_graph::colored_graph()
{
	null();
}

colored_graph::~colored_graph()
{
	freeself();
}

void colored_graph::null()
{
	user_data = NULL;
	points = NULL;
	point_color = NULL;
	f_ownership_of_bitvec = FALSE;
	bitvector_adjacency = NULL;
	f_has_list_of_edges = FALSE;
	nb_edges = 0;
	list_of_edges = NULL;
}

void colored_graph::freeself()
{
	//cout << "colored_graph::freeself" << endl;
	if (user_data) {
		delete [] user_data;
		}
	if (points) {
		delete [] points;
		}
	if (point_color) {
		delete [] point_color;
		}
	if (f_ownership_of_bitvec) {
		if (bitvector_adjacency) {
			delete [] bitvector_adjacency;
			}
		}
	if (list_of_edges) {
		delete [] list_of_edges;
		}
	null();
}

void colored_graph::compute_edges(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, nb, a;

	if (f_v) {
		cout << "colored_graph::compute_edges" << endl;
		}
	if (f_has_list_of_edges) {
		cout << "colored_graph::compute_edges f_has_list_of_edges" << endl;
		exit(1);
		}
	nb = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				nb++;
				}
			}
		}
	list_of_edges = new int [nb];
	nb_edges = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				a = ij2k(i, j, nb_points);
				list_of_edges[nb_edges++] = a;
				}
			}
		}
	if (nb_edges != nb) {
		cout << "colored_graph::compute_edges nb_edges != nb" << endl;
		exit(1);
		}

	f_has_list_of_edges = TRUE;
	if (f_v) {
		cout << "colored_graph::compute_edges done" << endl;
		}
}


int colored_graph::is_adjacent(int i, int j)
{
	if (i == j) {
		return FALSE;
		}
	if (i > j) {
		return is_adjacent(j, i);
		}
	int k;
	
	k = ij2k(i, j, nb_points);
	return bitvector_s_i(bitvector_adjacency, k);
}

void colored_graph::set_adjacency(int i, int j, int a)
{
	int k;
	k = ij2k(i, j, nb_points);
	bitvector_m_ii(bitvector_adjacency, k, a);
}

void colored_graph::print()
{
	int i;
	
	cout << "colored graph with " << nb_points << " points and " << nb_colors << " colors" << endl;

	cout << "i : points[i] : point_color[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " : " << points[i] << " : " << point_color[i] << endl;
		}
	
}

void colored_graph::init_with_point_labels(int nb_points, int nb_colors, 
	int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
	int *point_labels, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "colored_graph::init_with_point_labels" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		}
	init(nb_points, nb_colors, 
		colors, bitvec, f_ownership_of_bitvec, 
		verbose_level);
	int_vec_copy(point_labels, points, nb_points);
	if (f_v) {
		cout << "colored_graph::init_with_point_labels done" << endl;
		}
}

void colored_graph::init(int nb_points, int nb_colors, 
	int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "colored_graph::init" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		}
	colored_graph::nb_points = nb_points;
	colored_graph::nb_colors = nb_colors;
	
	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;

	user_data_size = 0;
	
	points = new int [nb_points];
	for (i = 0; i < nb_points; i++) {
		points[i] = i;
		}
	point_color = new int [nb_points];

	if (colors) {
		int_vec_copy(colors, point_color, nb_points);
		}
	else {
		int_vec_zero(point_color, nb_points);
		}
	
	colored_graph::f_ownership_of_bitvec = f_ownership_of_bitvec;
	bitvector_adjacency = bitvec;

	if (f_v) {
		cout << "colored_graph::init" << endl;
		}

}

void colored_graph::init_no_colors(int nb_points, uchar *bitvec, int f_ownership_of_bitvec, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
		}
	vertex_colors = new int[nb_points];
	int_vec_zero(vertex_colors, nb_points);

	init(nb_points, 1 /* nb_colors */, 
		vertex_colors, bitvec, f_ownership_of_bitvec, verbose_level);

	delete [] vertex_colors;
	if (f_v) {
		cout << "colored_graph::init_no_colors done" << endl;
		}
}

void colored_graph::init_adjacency(int nb_points, int nb_colors, 
	int *colors, int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int bitvector_length;
	uchar *bitvec;


	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		}
	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;
	bitvec = new uchar [bitvector_length];
	for (i = 0; i < bitvector_length; i++) {
		bitvec[i] = 0;
		}
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (Adj[i * nb_points + j]) {
				k = ij2k(i, j, nb_points);
				bitvector_m_ii(bitvec, k, 1);
				}
			}
		}
	init(nb_points, nb_colors, 
		colors, bitvec, TRUE /* f_ownership_of_bitvec */, 
		verbose_level);

	// do not free bitvec here

	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
		}

}

void colored_graph::init_adjacency_no_colors(int nb_points, int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
		}
	vertex_colors = new int[nb_points];
	int_vec_zero(vertex_colors, nb_points);

	init_adjacency(nb_points, 1 /* nb_colors */, 
		vertex_colors, Adj, verbose_level);

	delete [] vertex_colors;
	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors done" << endl;
		}
}

void colored_graph::init_user_data(int *data, int data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "colored_graph::init_user_data" << endl;
		}
	user_data_size = data_size;
	user_data = new int [data_size];
	for (i = 0; i < data_size; i++) {
		user_data[i] = data[i];
		}
	if (f_v) {
		cout << "colored_graph::init_user_data done" << endl;
		}
}

void colored_graph::load(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	FILE *fp;
	char ext[1000];
	int i;
	
	if (file_size(fname) <= 0) {
		cout << "colored_graph::load file is empty or does not exist" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "colored_graph::load Reading file " << fname << " of size " << file_size(fname) << endl;
		}


	get_extension_if_present(fname, ext);
	strcpy(fname_base, fname);
	chop_off_extension_if_present(fname_base, ext);
	if (f_v) {
		cout << "fname_base=" << fname_base << endl;
		}

	fp = fopen(fname, "rb");

	nb_points = fread_int4(fp);
	nb_colors = fread_int4(fp);


	L = (nb_points * (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;

	user_data_size = fread_int4(fp);
	user_data = new int[user_data_size];
	
	for (i = 0; i < user_data_size; i++) {
		user_data[i] = fread_int4(fp);
		}

	points = new int[nb_points];
	point_color = new int[nb_points];


	if (f_v) {
		cout << "colored_graph::load the graph has " << nb_points << " points and " << nb_colors << " colors" << endl;
		}
	
	for (i = 0; i < nb_points; i++) {
		points[i] = fread_int4(fp);
		point_color[i] = fread_int4(fp);
		if (point_color[i] >= nb_colors) {
			cout << "colored_graph::load" << endl;
			cout << "point_color[i] >= nb_colors" << endl;
			cout << "point_color[i]=" << point_color[i] << endl;
			cout << "i=" << i << endl;
			cout << "nb_colors=" << nb_colors << endl;
			exit(1);
			}
		}

#if 0
	cout << "colored_graph::load points=";
	int_vec_print(cout, points, nb_points);
	cout << endl;
#endif

	bitvector_adjacency = new uchar [bitvector_length];
	fread_uchars(fp, bitvector_adjacency, bitvector_length);


	fclose(fp);
	if (f_v) {
		cout << "colored_graph::load Read file " << fname << " of size " << file_size(fname) << endl;
		}
}

void colored_graph::all_cliques_of_size_k_ignore_colors(int target_depth, 
	int &nb_sol, int &decision_step_counter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	clique_finder *CF;
	int print_interval = 10000000;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors" << endl;
		}
	CF = new clique_finder;

	CF->init("", nb_points, 
		target_depth, 
		FALSE /* f_has_adj_list */, NULL /* int *adj_list_coded */, 
		TRUE /* f_has_bitvector */, bitvector_adjacency, 
		print_interval, 
		FALSE /* f_maxdepth */, 0 /* maxdepth */, 
		TRUE /* f_store_solutions */, 
		verbose_level - 1);

	CF->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	nb_sol = CF->nb_sol;
	decision_step_counter = CF->decision_step_counter;

	delete CF;
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors done" << endl;
		}
}

void colored_graph::all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(int target_depth, 
	const char *fname, 
	int f_restrictions, int *restrictions, 
	int &nb_sol, int &decision_step_counter, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	clique_finder *CF;
	int print_interval = 1000000;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file " << fname << endl;
		if (f_restrictions) {
			cout << "with restrictions: ";
			int_vec_print(cout, restrictions, 3);
			cout << endl;
			}
		}
	CF = new clique_finder;


	file_output *FO;
	FO = new file_output;
	FO->open(fname, CF, verbose_level);

	CF->call_back_clique_found = call_back_clique_found_using_file_output;
	CF->call_back_clique_found_data = FO;

	CF->init("", nb_points, 
		target_depth, 
		FALSE /* f_has_adj_list */, NULL /* int *adj_list_coded */, 
		TRUE /* f_has_bitvector */, bitvector_adjacency, 
		print_interval, 
		FALSE /* f_maxdepth */, 0 /* maxdepth */, 
		TRUE /* f_store_solutions */, 
		verbose_level - 1);

	if (f_restrictions) {
		if (f_v) {
			cout << "colored_graph::all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file before init_restrictions" << endl;
			}
		CF->init_restrictions(restrictions, verbose_level - 2);
		}



	CF->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	nb_sol = CF->nb_sol;
	decision_step_counter = CF->decision_step_counter;

	delete FO;
	delete CF;
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file done" << endl;
		}
}

void colored_graph::all_rainbow_cliques(ofstream *fp, int f_output_solution_raw, 
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R;

	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques" << endl;
		}
	R = new rainbow_cliques;
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques before R->search" << endl;
		}
	R->search(this, fp, f_output_solution_raw, 
		f_maxdepth, maxdepth, 
		f_restrictions, restrictions, 
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level - 1);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques after R->search" << endl;
		}
	delete R;
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques done" << endl;
		}
}

void colored_graph::all_rainbow_cliques_with_additional_test_function(ofstream *fp, int f_output_solution_raw, 
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int f_has_additional_test_function,
	void (*call_back_additional_test_function)(rainbow_cliques *R, void *user_data, 
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level), 
	int f_has_print_current_choice_function,
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level), 
	void *user_data, 
	int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R;

	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_test_function" << endl;
		}
	R = new rainbow_cliques;
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_test_function before R->search_with_additional_test_function" << endl;
		}
	R->search_with_additional_test_function(this, fp, f_output_solution_raw, 
		f_maxdepth, maxdepth, 
		f_restrictions, restrictions, 
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		f_has_additional_test_function,
		call_back_additional_test_function, 
		f_has_print_current_choice_function,
		call_back_print_current_choice, 
		user_data, 
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level - 1);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_test_function after R->search_with_additional_test_function" << endl;
		}
	delete R;
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_test_function done" << endl;
		}
}


void colored_graph::export_to_file(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "colored_graph::export_to_file" << endl;
		}
	{
		ofstream fp(fname);

		fp << "[" << endl;
		for (i = 0; i < nb_points; i++) {



			fp << "[";
			for (j = 0; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					fp << "1";
					}
				else {
					fp << "0";
					}
				if (j < nb_points - 1) {
					fp << ",";
					}
				}
			fp << "]";
			if (i < nb_points - 1) {
				fp << ", " << endl;
				}
			}
		fp << "];" << endl;

		


	}
	cout << "Written file " << fname << " of size " << file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_file" << endl;
		}
}


void colored_graph::early_test_func_for_clique_search(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int j, a, pt;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_clique_search checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		int_vec_copy(candidates, good_candidates, nb_candidates);
		return;
		}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		
		if (is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
			}
		} // next j
	
}

file_output::file_output()
{
	null();
}

file_output::~file_output()
{
	freeself();
}

void file_output::null()
{
	f_file_is_open = FALSE;
	fp = NULL;
}

void file_output::freeself()
{
	if (f_file_is_open) {
		close();
		}
	null();
}


void file_output::open(const char *fname, void *user_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "file_output::open" << endl;
		}
	strcpy(file_output::fname, fname);
	file_output::user_data = user_data;
	
	fp = new ofstream;
	fp->open(fname);
	f_file_is_open = TRUE;
	

	
	if (f_v) {
		cout << "file_output::open done" << endl;
		}
}

void file_output::close()
{
	*fp << "-1" << endl;
	delete fp;
	fp = NULL;
	f_file_is_open = FALSE;
}

void file_output::write_line(int nb, int *data, int verbose_level)
{
	int i;

	if (!f_file_is_open) {
		cout << "file_output::write_line file is not open" << endl;
		exit(1);
		}
	*fp << nb;
	for (i = 0; i < nb; i++) {
		*fp << " " << data[i];
		}
	*fp << endl;
}


#undef SUSPICOUS

void clique_finder::open_tree_file(const char *fname_base, int f_decision_nodes_only)
{
	f_write_tree = TRUE;
	clique_finder::f_decision_nodes_only = f_decision_nodes_only;
	sprintf(fname_tree, "%s.tree", fname_base);
	fp_tree = new ofstream;
	fp_tree->open(fname_tree);
}

void clique_finder::close_tree_file()
{
	*fp_tree << -1 << endl;
	fp_tree->close();
	delete fp_tree;
	cout << "written file " << fname_tree << " of size " << file_size(fname_tree) << endl;
}

void clique_finder::init(const char *label, int n, 
	int target_depth, 
	int f_has_adj_list, int *adj_list_coded, 
	int f_has_bitvector, uchar *bitvector_adjacency, 
	int print_interval, 
	int f_maxdepth, int maxdepth, 
	int f_store_solutions, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	strcpy(clique_finder::label, label);
	
	clique_finder::f_store_solutions = f_store_solutions;
	clique_finder::n = n;
	clique_finder::target_depth = target_depth;
	clique_finder::verbose_level = verbose_level;
	clique_finder::f_maxdepth = f_maxdepth;
	clique_finder::maxdepth = maxdepth;
	clique_finder::print_interval = print_interval;

	clique_finder::f_has_adj_list = f_has_adj_list;
	clique_finder::adj_list_coded = adj_list_coded;
	clique_finder::f_has_bitvector = f_has_bitvector;
	clique_finder::bitvector_adjacency = bitvector_adjacency;
	
	if (f_v) {
		cout << "clique_finder::init " << label << " n=" << n << " target_depth=" << target_depth << endl;
		cout << "f_has_adj_list=" << f_has_adj_list << endl;
		cout << "f_has_bitvector=" << f_has_bitvector << endl;
		}
	nb_sol = 0;

	pt_list = new int[n];
	if (f_v) {
		cout << "clique_finder::init pt_list allocated" << endl;
		}
	pt_list_inv = new int[n];
	if (f_v) {
		cout << "clique_finder::init pt_list_inv allocated" << endl;
		}
	nb_points = new int[target_depth + 1];
	candidates = new int[(target_depth + 1) * n];
	nb_candidates = new int[target_depth];
	current_choice = new int[target_depth];
	level_counter = new int[target_depth];
	f_level_mod = new int[target_depth];
	level_r = new int[target_depth];
	level_m = new int[target_depth];
	current_clique = new int[target_depth];

	int_vec_zero(level_counter, target_depth);
	int_vec_zero(f_level_mod, target_depth);
	int_vec_zero(level_r, target_depth);
	int_vec_zero(level_m, target_depth);


	for (i = 0; i < n; i++) {
		pt_list[i] = i;
		pt_list_inv[i] = i;
		}
	nb_points[0] = n;
	counter = 0;
	decision_step_counter = 0;

	//allocate_bitmatrix(verbose_level);

	if (f_v) {
		cout << "clique_finder::init finished" << endl;
		}
}

void clique_finder::allocate_bitmatrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, size;

	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix" << endl;
		}
	bitmatrix_N = (n + 7) >> 3; // 1 char = 8 bits = 2^3
	bitmatrix_m = n;
	bitmatrix_n = n;
	size = bitmatrix_m * bitmatrix_N;
	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix allocating BITMATRIX of size " << size << endl;
		}
	bitmatrix_adjacency = new uchar [size];
	f_has_bitmatrix = TRUE;

	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix adjacency matrix allocated" << endl;
		}


	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix initializing adjacency matrix:" << endl;
		}


	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			if (f_v && ((k % (1 << 19)) == 0)) {
				cout << k << " : i=" << i << " j=" << j << endl;
				}
			if (i == j) {
				//adjacency[i * n + j] = 0;
				m_iji(i, j, 0);
				}
			else {
				//adjacency[i * n + j] = -1;
				//adjacency[j * n + i] = -1;
				//k = ij2k(i, j, n);
				//adjacency[i * n + j] = adj_list_coded[k];
				//adjacency[j * n + i] = adj_list_coded[k];

				int aij = 0;

				if (f_has_adj_list) {
					aij = adj_list_coded[k];
					}
				else if (f_has_bitvector) {
					aij = bitvector_s_i(bitvector_adjacency, k);
					}
				m_iji(i, j, aij);
				m_iji(j, i, aij);
				k++;
				}
			}
		}
	if (f_v) {
		cout << "clique_finder::allocate_bitmatrix done" << endl;
		}
}

void clique_finder::init_restrictions(int *restrictions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, r, m;

	if (f_v) {
		cout << "clique_finder::init_restrictions" << endl;
		}
	for (h = 0; ; h++) {
		if (restrictions[h * 3] == -1) {
			break;
			}
		i = restrictions[h * 3 + 0];
		r = restrictions[h * 3 + 1];
		m = restrictions[h * 3 + 2];
		if (i >= target_depth) {
			cout << "clique_finder::init_restrictions i >= target_depth" << endl;
			exit(1);
			}
		f_level_mod[i] = TRUE;
		level_r[i] = r;
		level_m[i] = m;
		cout << "clique_finder::init_restrictions level " << i << " congruent " << r << " mod " << m << endl;
		}
	if (f_v) {
		cout << "clique_finder::init_restrictions done" << endl;
		}
}

clique_finder::clique_finder()
{
	null();
}

clique_finder::~clique_finder()
{
	free();
}

void clique_finder::null()
{
	f_maxdepth = FALSE;
	f_write_tree = FALSE;
	fp_tree = NULL;

	f_has_bitmatrix = FALSE;
	
	point_labels = NULL;
	point_is_suspicous = NULL;

	bitmatrix_adjacency = NULL;
	pt_list = NULL;
	pt_list_inv = NULL;
	nb_points = NULL;
	candidates = NULL;
	nb_candidates = NULL;
	current_choice = NULL;
	level_counter = NULL;
	f_level_mod = NULL;
	level_r = NULL;
	level_m = NULL;
	current_clique = NULL;
	
	call_back_clique_found = NULL;
	call_back_add_point = NULL;
	call_back_delete_point = NULL;
	call_back_find_candidates = NULL;
	call_back_is_adjacent = NULL;
	call_back_after_reduction = NULL;

	f_has_print_current_choice_function = FALSE;
	call_back_print_current_choice = NULL;
	print_current_choice_data = NULL;
	
	call_back_clique_found_data = NULL;
	
}

void clique_finder::free()
{
	if (point_labels) {
		delete [] point_labels;
		}
	if (point_is_suspicous) {
		delete [] point_is_suspicous;
		}

	if (bitmatrix_adjacency) {
		//delete [] adjacency;
		delete [] bitmatrix_adjacency;
		}
	if (pt_list) {
		delete [] pt_list;
		}
	if (pt_list_inv) {
		delete [] pt_list_inv;
		}
	if (nb_points) {
		delete [] nb_points;
		}
	if (candidates) {
		delete [] candidates;
		}
	if (nb_candidates) {
		delete [] nb_candidates;
		}
	if (current_choice) {
		delete [] current_choice;
		}
	if (level_counter) {
		delete [] level_counter;
		}
	if (f_level_mod) {
		delete [] f_level_mod;
		}
	if (level_r) {
		delete [] level_r;
		}
	if (level_m) {
		delete [] level_m;
		}
	if (current_clique) {
		delete [] current_clique;
		}
	null();
}

void clique_finder::init_point_labels(int *pt_labels)
{
	int i;
	point_labels = new int [n];
	for (i = 0; i < n; i++) {
		point_labels[i] = pt_labels[i];
		}
}

void clique_finder::print_set(int size, int *set)
{
	int i, a, b;
	
	cout << "(";
	for (i = 0; i < size; i++) {
		a = set[i];
		b = point_label(a);
		cout << b;
		if (i < size - 1)
			cout << ", ";
		}
	cout << ")";
}

void clique_finder::log_position_and_choice(int depth, int counter_save, int counter)
{
#if 0
	cout << label << " counter " << counter_save;
	if (counter != counter_save) {
		cout << "," << counter;
		}
	cout << " depth " << depth;
	//cout << " ";
	cout << " : " << current_choice[depth] << " / " << nb_candidates[depth];
#endif
	cout << "node " << counter << " at depth " << depth << " : ";
	log_choice(depth + 1);
	cout << " nb_sol=" << nb_sol << " ";
	if (FALSE) {
		cout << " clique ";
		int_set_print(cout, current_clique, depth);
		}
}

void clique_finder::log_position(int depth, int counter_save, int counter)
{
#if 0
	cout << label << " counter " << counter_save;
	if (counter != counter_save) {
		cout << "," << counter;
		}
	cout << " depth " << depth;
	cout << " nb_sol=" << nb_sol << " ";
#endif
	cout << "node " << counter << " at depth " << depth << " : ";
	log_choice(depth);
	if (FALSE) {
		cout << " clique ";
		int_set_print(cout, current_clique, depth);
		}
}

void clique_finder::log_choice(int depth)
{
	int i;

	cout << "choice ";
	for (i = 0; i < depth; i++) {
		cout << i << ": " << current_choice[i] << "/" << nb_candidates[i];
		if (i < depth - 1)
			cout << ", ";
		}
	cout << " ";
}

void clique_finder::swap_point(int idx1, int idx2)
{
	int_vec_swap_points(pt_list, pt_list_inv, idx1, idx2);
#if 0
	int p1, p2;
	
	if (idx1 == idx2) {
		return;
		}
	p1 = pt_list[idx1];
	p2 = pt_list[idx2];
	pt_list[idx1] = p2;
	pt_list[idx2] = p1;
	pt_list_inv[p1] = idx2;
	pt_list_inv[p2] = idx1;
#endif
}

int clique_finder::degree_of_point(int depth, int i, int nb_points)
{
	int pti, ptj, j, d;
	
	pti = pt_list[i];
	d = 0;
	for (j = 0; j < nb_points; j++) {
		if (j == i) {
			continue;
			}
		ptj = pt_list[j];
		if (is_adjacent(depth, pti, ptj)) {
			d++;
			}
		}
	return d;
}

#if 0
int clique_finder::degree_of_point_verbose(int i, int nb_points)
{
	int pti, ptj, j, d;
	
	pti = pt_list[i];
	d = 0;
	for (j = 0; j < nb_points; j++) {
		if (j == i)
			continue;
		ptj = pt_list[j];
		if (is_suspicous(pti) && is_suspicous(ptj)) {
			if (!is_adjacent(pti, ptj)) {
				cout << "the suspicous points " << pti << " and " << ptj << " are not adjacent" << endl;
				exit(1);
				}
			}
		if (is_adjacent(depth, pti, ptj))
			d++;
		}
	return d;
}
#endif

int clique_finder::is_suspicous(int i)
{
	if (point_is_suspicous == NULL)
		return FALSE;
	return point_is_suspicous[i];
}

int clique_finder::point_label(int i)
{
	if (point_labels)
		return point_labels[i];
	else
		return i;
}

int clique_finder::is_adjacent(int depth, int i, int j)
{
	int a;

	//a = adjacency[i * n + j];
	if (i == j) {
		return 0;
		}
	a = s_ij(i, j);
#if 0
	if (a == -1) {
		a = (*call_back_is_adjacent)(this, i, j, 0/* verbose_level */);
		adjacency[i * n + j] = a;
		adjacency[j * n + i] = a;
		}
#endif
	return a;
}

void clique_finder::write_entry_to_tree_file(int depth, int verbose_level)
{
	int i;
	
	if (!f_write_tree) {
		return;
		}

#if 0
		*fp_tree << "# " << depth << " ";
		for (i = 0; i < depth; i++) {
			*fp_tree << current_clique[i] << " ";
			}
		*fp_tree << endl;
#endif

	if (f_decision_nodes_only && nb_candidates[depth - 1] == 1) {
		return;
		}
	if (f_decision_nodes_only && depth == 0) {
		return;
		}
	if (f_decision_nodes_only) {
		int d;
	
		d = 0;
		for (i = 0; i < depth; i++) {
			if (nb_candidates[i] > 1) {
				d++;
				}
			}
		*fp_tree << d << " ";
		for (i = 0; i < depth; i++) {
			if (nb_candidates[i] > 1) {
				*fp_tree << current_clique[i] << " ";
				}
			}
		*fp_tree << endl;
		}
	else {
		*fp_tree << depth << " ";
		for (i = 0; i < depth; i++) {
			*fp_tree << current_clique[i] << " ";
			}
		*fp_tree << endl;
		}
}

void clique_finder::m_iji(int i, int j, int a)
{
	int m, n, N; //, jj, bit;
	//uchar mask;
	
	m = bitmatrix_m;
	n = bitmatrix_n;
	N = bitmatrix_N;
	if (i < 0 || i >= m) {
		cout << "clique_finder::m_iji() addressing error, i = " << i << ", m = " << m << endl;
		exit(1);		
		}
	if (j < 0 || j >= n) {
		cout << "clique_finder::m_iji() addressing error, j = " << j << ", n = " << n << endl;
		exit(1);		
		}

	bitvector_m_ii(bitmatrix_adjacency, i * n + j, a);

#if 0
	jj = j >> 3;
	bit = j & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitmatrix_adjacency[i * N + jj];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
#endif
}

int clique_finder::s_ij(int i, int j)
{
	int k, aij;
	
	if ( i < 0 || i >= n ) {
		cout << "clique_finder::s_ij() addressing error, i = " << i << ", n = " << n << endl;
		exit(1);	
		}
	if ( j < 0 || j >= n ) {
		cout << "clique_finder::s_ij() addressing error, j = " << j << ", n = " << n << endl;
		exit(1);	
		}
	if (i == j) {
		return 0;
		}

	if (f_has_bitmatrix) {
		return bitvector_s_i(bitmatrix_adjacency, i * n + j);
		}
	else if (f_has_adj_list) {
		k = ij2k(i, j, n);
		aij = adj_list_coded[k];
		return aij;
		}
	else if (f_has_bitvector) {
		k = ij2k(i, j, n);
		aij = bitvector_s_i(bitvector_adjacency, k);
		return aij;
		}
	else {
		cout << "clique_finder::s_ij we don't have a matrix" << endl;
		exit(1);
		}

#if 0
	//uchar mask;
	jj = j >> 3;
	bit = j & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitmatrix_adjacency[i * N + jj];
	if (x & mask)
		return 1;
	else
		return 0;
#endif
}


void clique_finder::backtrack_search(int depth, int verbose_level)
{
	int nb_old, i, nb_new;
	int pt1, pt2, pt, pass, f_go, counter_save;
	int my_verbose_level;
	
	counter++;
	counter_save = counter;

	if (depth && nb_candidates[depth - 1] > 1) {
		decision_step_counter++;
		}
	if ((counter & ((1 << 18) - 1)) == 0) {
		my_verbose_level = verbose_level + 1;
		}
	else {
		my_verbose_level = verbose_level;
		}
	int f_v = (my_verbose_level >= 1);
	int f_vv = (my_verbose_level >= 2);
	//int f_vvv = (my_verbose_level >= 3);

	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " starting" << endl;
		}
	write_entry_to_tree_file(depth, verbose_level);

	if (depth == target_depth) {
	
		// We found a clique:

		if (f_v) {
			cout << "clique_finder::backtrack_search depth == target_depth" << endl;
			}
		if (f_store_solutions) {
			//cout << "storing solution" << endl;
			vector<int> sol;
			int j;
			sol.resize(depth);
			for (j = 0; j < depth; j++) {
				sol[j] = (int) current_clique[j];
				}
			solutions.push_back(sol);
			
			}
		nb_sol++;
		
		//cout << "clique_finder::backtrack_search before call_back_clique_found" << endl;
		if (call_back_clique_found) {
			//cout << "calling call_back_clique_found" << endl;
			(*call_back_clique_found)(this, verbose_level);
			}
		//cout << "solution " << nb_sol << ", found a clique of size target_depth" << endl;
		//cout << "clique";
		//int_set_print(cout, current_clique, depth);
		//cout << " depth = " << depth << endl;
		//exit(1);

		return;
		}


	if (f_maxdepth && depth == maxdepth) {
		return;
		}
	if (depth == 0)
		nb_old = n;
	else
		nb_old = nb_points[depth - 1];

#if 0
	if (f_v || (counter % print_interval) == 0) {
		log_position(depth, counter_save, counter);
		cout << endl;
		}

	if (f_v && depth) {
		log_position(depth, counter_save, counter);
		cout << " : # active points from previous level is " << nb_old << endl;
		//cout << " : previous lvl_pt_list[" << depth - 1 << "] of size " << nb_old << " : " << endl;
		////int_vec_print(cout, lvl_pt_list[depth - 1], lvl_nb_points[depth - 1]);
		//print_point_set(depth, counter_save, counter, nb_old, lvl_pt_list[depth - 1]);
		//cout << endl;
		}
#endif

	// first pass:
	// if depth > 0 and we are not using call_back_find_candidates, 
	// we apply the lexicographical ordering test.
	// the points are required to be greater than the previous point in the clique.
	// this also eliminates the point 
	// that was added to the clique in the previous step from pt_list.
	
	if (f_v) {
		cout << "clique_finder::backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " first pass" << endl;
		}

	if (depth && call_back_find_candidates == NULL) {
		// if we don't have a find_candidates function,
		// then we use the lexicographical ordering.
		// The idea is that we may force the clique to be 
		// constructed in increasing order of its points.
		// Hence, now we can eliminate all points that 
		// are smaller than the most recently added clique point:
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (pt2 > pt1) {
				swap_point(nb_new, i);
				nb_new++;
				}
			}
		}
	else {
		nb_new = nb_old;
		}
	

	// second pass: find the points that are connected with the 
	// previously chosen clique point:
	
	nb_old = nb_new;	
	if (depth) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			// go over all points in the old list:
			
			pt2 = pt_list[i];
			if (is_adjacent(depth, pt1, pt2)) {

				// point is adjacent, so we keep that point

				swap_point(nb_new, i);
				nb_new++;
				}
			
			}
		}
	else {
		nb_new = nb_old;
		}
	

	pass = 2;
	
	if (f_vv) {
		log_position(depth, counter_save, counter);
		cout << " : pass 2: ";
		cout << endl;
		}


#if 0
	// higher passes: 
	// find the points that have sufficiently high degree:
	
	do {
		nb_old = nb_new;
		nb_new = 0;
		for (i = 0; i < nb_old; i++) {
			d = degree_of_point(i, nb_old);
			if (d >= target_depth - depth - 1) {
				swap_point(nb_new, i);
				nb_new++;
				}
			else {
				if (point_is_suspicous && 
					point_is_suspicous[pt_list[i]]) {
					log_position(depth, counter_save, counter);
					cout << " : pass " << pass 
						<< ": suspicous point " << point_label(pt_list[i]) << " eliminated, d=" << d 
						<< " is less than target_depth - depth - 1 = " << target_depth - depth - 1 << endl;;
					degree_of_point_verbose(i, nb_old);
					}
				}
			}
		pass++;

		if (f_vv) {
			log_position(depth, counter_save, counter);
			cout << " : pass " << pass << ": ";
			cout << endl;
			}

		} while (nb_new < nb_old);	
#endif

	nb_points[depth] = nb_new;


	
	if (f_v) {
		log_position(depth, counter_save, counter);
		cout << "after " << pass << " passes: nb_points = " << nb_new << endl;
		}
	

	if (call_back_after_reduction) {
		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " before call_back_after_reduction" << endl;
			}
		(*call_back_after_reduction)(this, depth, nb_points[depth], verbose_level);
		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " after call_back_after_reduction" << endl;
			}
		}


	{
	//int i; //, nb_old;

	if (call_back_find_candidates) {
		int reduced_nb_points;

		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " before call_back_find_candidates" << endl;
			}
		nb_candidates[depth] = (*call_back_find_candidates)(this, 
			depth, current_clique, 
			nb_points[depth], reduced_nb_points, pt_list, pt_list_inv, 
			candidates + depth * n, 
			0/*verbose_level*/);

		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " after call_back_find_candidates nb_candidates=" << nb_candidates[depth] << endl;
			}
		// The set of candidates is stored in 
		// candidates + depth * n.
		// The number of candidates is in nb_candidates[depth]


#ifdef SUSPICOUS
		if (f_vv) {
			if (point_is_suspicous) {
				cout << "candidate set of size " 
					<< nb_candidates[depth] << endl;
				cout << endl;
				}
			}
#endif
		nb_points[depth] = reduced_nb_points;
		}
	else {
		// If we don't have a find_candidates callback, 
		// we take all the points into consideration:

		int_vec_copy(pt_list, candidates + depth * n, nb_points[depth]);
		nb_candidates[depth] = nb_points[depth];
		}
	}


	// added Dec 2014:
	if (f_has_print_current_choice_function) {
		(*call_back_print_current_choice)(this, depth, print_current_choice_data, verbose_level);
		}
	
	// Now we are ready to go in the backtrack search.
	// We'll try each of the points in candidates one by one:

	for (current_choice[depth] = 0; current_choice[depth] < nb_candidates[depth]; current_choice[depth]++, level_counter[depth]++) {


		if (f_v) {
			log_position(depth, counter_save, counter);
			cout << " choice " << current_choice[depth] << " / " << nb_candidates[depth] << endl;
			}

		f_go = TRUE;  // Whether we want to go in the recursion.

		if (f_level_mod[depth]) {
			if ((level_counter[depth] % level_m[depth]) != level_r[depth]) {
				f_go = FALSE;
				}
			}


		pt = candidates[depth * n + current_choice[depth]];


		if (f_vv) {
			log_position_and_choice(depth, counter_save, counter);
			cout << endl;
			}



		// We add a point under consideration:
		
		current_clique[depth] = pt;

		if (call_back_add_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_add_point" << endl;
				}
			(*call_back_add_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_add_point" << endl;
				}
			}

		if (point_is_suspicous) {
			if (point_is_suspicous[pt]) {
				log_position(depth, counter_save, counter);
				cout << " : considering clique ";
				print_set(depth + 1, current_clique);
				//int_set_print(cout, current_clique, depth);
				cout << " depth = " << depth << " nb_old=" << nb_old << endl;
				f_go = TRUE;
				}
			else {
				f_go = FALSE;
				}
			}
	

		// and now, let's do the recursion:

		if (f_go) {
			backtrack_search(depth + 1, verbose_level);
			} // if (f_go)





		// We delete the point:

		if (call_back_delete_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_delete_point" << endl;
				}
			(*call_back_delete_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_delete_point" << endl;
				}
			}

		} // for current_choice[depth]


	
	if (f_v) {
		cout << "backtrack_search : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " done" << endl;
		}
}

int clique_finder::solve_decision_problem(int depth, int verbose_level)
// returns TRUE if we found a solution
{
	int nb_old, i, nb_new;
	int pt1, pt2, pt, pass, f_go, counter_save;
	int my_verbose_level;
	
	counter++;
	counter_save = counter;

	if (depth && nb_candidates[depth - 1] > 1) {
		decision_step_counter++;
		}
	if ((counter & ((1 << 17) - 1)) == 0) {
		my_verbose_level = verbose_level + 1;
		}
	else {
		my_verbose_level = verbose_level;
		}
	int f_v = (my_verbose_level >= 1);
	int f_vv = (my_verbose_level >= 2);
	//int f_vvv = (my_verbose_level >= 3);

	if (f_v) {
		cout << "solve_decision_problem : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " starting" << endl;
		}
	write_entry_to_tree_file(depth, verbose_level);

	if (depth == target_depth) {
		nb_sol++;
		//cout << "clique_finder::backtrack_search before call_back_clique_found" << endl;
		if (call_back_clique_found) {
			(*call_back_clique_found)(this, verbose_level);
			}
		//cout << "solution " << nb_sol << ", found a clique of size target_depth" << endl;
		//cout << "clique";
		//int_set_print(cout, current_clique, depth);
		//cout << " depth = " << depth << endl;
		//exit(1);

		return TRUE;
		}


	if (f_maxdepth && depth == maxdepth) {
		return FALSE;
		}
	if (depth == 0)
		nb_old = n;
	else
		nb_old = nb_points[depth - 1];

#if 0
	if (f_v || (counter % print_interval) == 0) {
		log_position(depth, counter_save, counter);
		cout << endl;
		}

	if (f_v && depth) {
		log_position(depth, counter_save, counter);
		cout << " : # active points from previous level is " << nb_old << endl;
		//cout << " : previous lvl_pt_list[" << depth - 1 << "] of size " << nb_old << " : " << endl;
		////int_vec_print(cout, lvl_pt_list[depth - 1], lvl_nb_points[depth - 1]);
		//print_point_set(depth, counter_save, counter, nb_old, lvl_pt_list[depth - 1]);
		//cout << endl;
		}
#endif

	// first pass:
	// if depth > 0 and we are not using call_back_find_candidates, 
	// we apply the lexicographical ordering test.
	// the points are required to be greater than the previous point in the clique.
	// this also eliminates the point 
	// that was added to the clique in the previous step from pt_list.
	
	if (depth && call_back_find_candidates == NULL) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (pt2 > pt1) {
				swap_point(nb_new, i);
				nb_new++;
				}
			}
		}
	else {
		nb_new = nb_old;
		}
	

	// second pass: find the points that are connected with the 
	// previously chosen clique point:
	
	nb_old = nb_new;	
	if (depth) {
		nb_new = 0;
		pt1 = current_clique[depth - 1];
		for (i = 0; i < nb_old; i++) {
			pt2 = pt_list[i];
			if (is_adjacent(depth, pt1, pt2)) {
				swap_point(nb_new, i);
				nb_new++;
				}
			}
		}
	else {
		nb_new = nb_old;
		}
	

	pass = 2;
	
	if (f_vv) {
		log_position(depth, counter_save, counter);
		cout << " : pass 2: ";
		cout << endl;
		}


#if 0
	// higher passes: 
	// find the points that have sufficiently high degree:
	
	do {
		nb_old = nb_new;
		nb_new = 0;
		for (i = 0; i < nb_old; i++) {
			d = degree_of_point(i, nb_old);
			if (d >= target_depth - depth - 1) {
				swap_point(nb_new, i);
				nb_new++;
				}
			else {
				if (point_is_suspicous && 
					point_is_suspicous[pt_list[i]]) {
					log_position(depth, counter_save, counter);
					cout << " : pass " << pass 
						<< ": suspicous point " << point_label(pt_list[i]) << " eliminated, d=" << d 
						<< " is less than target_depth - depth - 1 = " << target_depth - depth - 1 << endl;;
					degree_of_point_verbose(i, nb_old);
					}
				}
			}
		pass++;

		if (f_vv) {
			log_position(depth, counter_save, counter);
			cout << " : pass " << pass << ": ";
			cout << endl;
			}

		} while (nb_new < nb_old);	
#endif

	nb_points[depth] = nb_new;


	
	if (f_v) {
		log_position(depth, counter_save, counter);
		cout << "after " << pass << " passes: nb_points = " << nb_new << endl;
		}
	

	if (call_back_after_reduction) {
		(*call_back_after_reduction)(this, depth, nb_points[depth], verbose_level);
		}


	{
	int i; //, nb_old;

	if (call_back_find_candidates) {
		int reduced_nb_points;
		
		nb_candidates[depth] = (*call_back_find_candidates)(this, 
			depth, current_clique, 
			nb_points[depth], reduced_nb_points, 
			pt_list, pt_list_inv, 
			candidates + depth * n, 
			0/*verbose_level*/);
#ifdef SUSPICOUS
		if (f_vv) {
			if (point_is_suspicous) {
				cout << "candidate set of size " 
					<< nb_candidates[depth] << endl;
				cout << endl;
				}
			}
#endif
		nb_points[depth] = reduced_nb_points;
		}
	else {
		for (i = 0; i < nb_points[depth]; i++) {
			candidates[depth * n + i] = pt_list[i];
			}
		nb_candidates[depth] = nb_points[depth];
		}
	}




	for (current_choice[depth] = 0; current_choice[depth] < nb_candidates[depth]; current_choice[depth]++, level_counter[depth]++) {

		pt = candidates[depth * n + current_choice[depth]];

		f_go = TRUE;

		if (f_vv) {
			log_position_and_choice(depth, counter_save, counter);
			cout << endl;
			}
		// add a point
		
		current_clique[depth] = pt;

		if (call_back_add_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_add_point" << endl;
				}
			(*call_back_add_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_add_point" << endl;
				}
			}

		if (point_is_suspicous) {
			if (point_is_suspicous[pt]) {
				log_position(depth, counter_save, counter);
				cout << " : considering clique ";
				print_set(depth + 1, current_clique);
				//int_set_print(cout, current_clique, depth);
				cout << " depth = " << depth << " nb_old=" << nb_old << endl;
				f_go = TRUE;
				}
			else {
				f_go = FALSE;
				}
			}
	
		if (f_go) {
			if (solve_decision_problem(depth + 1, verbose_level)) {
				return TRUE;
				}
			} // if (f_go)

		// delete a point:

		if (call_back_delete_point) {
			if (FALSE /*f_v*/) {
				cout << "before call_back_delete_point" << endl;
				}
			(*call_back_delete_point)(this, depth, 
				current_clique, pt, 0/*verbose_level*/);
			if (FALSE /*f_v*/) {
				cout << "after call_back_delete_point" << endl;
				}
			}

		} // for current_choice[depth]


	
	if (f_v) {
		cout << "solve_decision_problem : ";
		log_position(depth, counter_save, counter);
		cout << " nb_sol=" << nb_sol << " done" << endl;
		}
	return FALSE;
}

void clique_finder::get_solutions(int *&Sol, int &nb_solutions, int &clique_sz, int verbose_level)
{
	int i, j;
	
	nb_solutions = nb_sol;
	//nb_sol = nb_sol;
	clique_sz = target_depth;
	Sol = new int [nb_sol * target_depth];
	for (i = 0; i < nb_sol; i++) {
		for (j = 0; j < target_depth; j++) {
			Sol[i * target_depth + j] = solutions.front()[j];
			}
		solutions.pop_front();
		}
}

void all_cliques_of_given_size(int *Adj, int nb_pts, int clique_sz, int *&Sol, int &nb_sol, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *adj_list_coded;
	int n2;
	int i, j, h;
	clique_finder *C;
	const char *label = "all_cliques_of_given_size";
	int print_interval = 1000;
	int f_maxdepth = FALSE;
	int maxdepth = 0;

	if (f_v) {
		cout << "all_cliques_of_given_size" << endl;
		}
	n2 = (nb_pts * (nb_pts - 1)) >> 1;
	adj_list_coded = new int [n2];
	h = 0;
	cout << "all_cliques_of_given_size: computing adj_list_coded" << endl;
	for (i = 0; i < nb_pts; i++) {
		for (j = i + 1; j < nb_pts; j++) {
			adj_list_coded[h++] = Adj[i * nb_pts + j];
			}
		}
	
	C = new clique_finder;
	
	if (f_v) {
		cout << "all_cliques_of_given_size: before C->init" << endl;
		}
	C->init(label, nb_pts, 
		clique_sz, 
		TRUE, adj_list_coded, 
		FALSE, NULL, 
		print_interval, 
		f_maxdepth, maxdepth, 
		TRUE /* f_store_solutions */, 
		verbose_level);

	C->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	if (f_v) {
		cout << "all_cliques_of_given_size done with search, we found " << C->nb_sol << " solutions" << endl;
		}

	int sz;
	C->get_solutions(Sol, nb_sol, sz, verbose_level);
	if (sz != clique_sz) {
		cout << "all_cliques_of_given_size sz != clique_sz" << endl;
		exit(1);
		}	
	delete C;
	delete [] adj_list_coded;
	if (f_v) {
		cout << "all_cliques_of_given_size done" << endl;
		}
}


rainbow_cliques::rainbow_cliques()
{
	null();
}

rainbow_cliques::~rainbow_cliques()
{
}

void rainbow_cliques::null()
{
	f_has_additional_test_function = FALSE;
}

void rainbow_cliques::freeself()
{
	null();
}

void rainbow_cliques::search(colored_graph *graph,
	ofstream *fp_sol, int f_output_solution_raw,
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int i;
	
	if (f_v) {
		cout << "rainbow_cliques::search" << endl;
		}

	search_with_additional_test_function(graph,
		fp_sol, f_output_solution_raw,
		f_maxdepth, maxdepth,
		f_restrictions, restrictions,
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		FALSE /* f_has_additional_test_function */,
		NULL, 
		FALSE /* f_has_print_current_choice_function */, 
		NULL, 
		NULL /* user_data */,
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level);
	
	if (f_v) {
		cout << "rainbow_cliques::search done" << endl;
		}
}

void rainbow_cliques::search_with_additional_test_function(
	colored_graph *graph, ofstream *fp_sol, int f_output_solution_raw, 
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions,
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int f_has_additional_test_function,
	void (*call_back_additional_test_function)(
		rainbow_cliques *R, void *user_data,
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level), 
	int f_has_print_current_choice_function,
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level), 
	void *user_data, 
	int &search_steps, int &decision_steps, int &nb_sol, int &dt, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_test_function" << endl;
		}

	rainbow_cliques::f_output_solution_raw = f_output_solution_raw;

	if (f_has_additional_test_function) {
		rainbow_cliques::f_has_additional_test_function = TRUE;
		rainbow_cliques::call_back_additional_test_function =
				call_back_additional_test_function;
		rainbow_cliques::user_data = user_data;
		}
	else {
		rainbow_cliques::f_has_additional_test_function = FALSE;
		}
	rainbow_cliques::graph = graph;
	rainbow_cliques::fp_sol = fp_sol;
	f_color_satisfied = new int[graph->nb_colors];
	color_chosen_at_depth = new int[graph->nb_colors];
	color_frequency = new int[graph->nb_colors];
	
	for (i = 0; i < graph->nb_colors; i++) {
		f_color_satisfied[i] = FALSE;
		}

	CF = new clique_finder;

	target_depth = graph->nb_colors;
	
	CF->init(graph->fname_base, graph->nb_points, 
		target_depth, 
		FALSE, NULL, 
		TRUE, graph->bitvector_adjacency, 
		print_interval, 
		f_maxdepth, maxdepth, 
		FALSE /* f_store_solutions */, 
		verbose_level - 2);

	CF->call_back_clique_found = call_back_colored_graph_clique_found;
	CF->call_back_add_point = call_back_colored_graph_add_point;
	CF->call_back_delete_point = call_back_colored_graph_delete_point;
	CF->call_back_find_candidates = call_back_colored_graph_find_candidates;
	CF->call_back_is_adjacent = NULL;

	//CF->call_back_after_reduction = call_back_after_reduction;
	CF->call_back_after_reduction = NULL;

	if (f_has_print_current_choice_function) {
		CF->f_has_print_current_choice_function = TRUE;
		CF->call_back_print_current_choice = call_back_print_current_choice;
		CF->print_current_choice_data = user_data;
		}
	
	CF->call_back_clique_found_data = this;
	
	
	if (f_restrictions) {
		if (f_v) {
			cout << "rainbow_cliques::search_with_additional_test_function "
					"before init_restrictions" << endl;
			}
		CF->init_restrictions(restrictions, verbose_level - 2);
		}

	if (f_tree) {
		CF->open_tree_file(fname_tree, f_decision_nodes_only);
		}
	
	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function "
				"now we start the rainbow clique finder process" << endl;
		}

#if 1

	CF->backtrack_search(0, 0 /*verbose_level*/);

#else
	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function "
				"before CF->backtrack_search_not_recursive" << endl;
		}
	CF->backtrack_search_not_recursive(verbose_level - 2);
	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function "
				"after CF->backtrack_search_not_recursive" << endl;
		}
#endif

	if (f_vv) {
		cout << "rainbow_cliques::search_with_additional_test_function "
				"done with finding all rainbow cliques" << endl;
		}

	if (f_v) {
		cout << "depth : level_counter" << endl;
		for (i = 0; i < CF->target_depth; i++) {
			cout << setw(3) << i << " : " << setw(6)
					<< CF->level_counter[i] << endl;
			}
		}

	if (f_tree) {
		CF->close_tree_file();
		}

	search_steps = CF->counter;
	decision_steps = CF->decision_step_counter;
	nb_sol = CF->nb_sol;
	


	delete CF;
	delete [] f_color_satisfied;
	delete [] color_chosen_at_depth;
	delete [] color_frequency;

	CF = NULL;
	f_color_satisfied = NULL;
	color_chosen_at_depth = NULL;
	color_frequency = NULL;

	if (f_v) {
		cout << "rainbow_cliques::search_with_additional_test_function "
				"done" << endl;
		}
	
}

int rainbow_cliques::find_candidates(
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c, i, j, c0, c0_freq, pt;
	
	if (f_v) {
		cout << "rainbow_cliques::find_candidates "
				"nb_pts = " << nb_pts << endl;
		}
	reduced_nb_pts = nb_pts;

	// determine the array color_frequency[].
	// color_frequency[i] is the frequency of points with color i 
	// in the list pt_list[]:

	int_vec_zero(color_frequency, graph->nb_colors);
	for (i = 0; i < nb_pts; i++) {
		pt = pt_list[i];
		if (pt >= graph->nb_points) {
			cout << "rainbow_cliques::find_candidates "
					"pt >= nb_points" << endl;
			exit(1);
			}
		c = graph->point_color[pt];
		if (c >= graph->nb_colors) {
			cout << "rainbow_cliques::find_candidates "
					"c >= nb_colors" << endl;
			exit(1);
			}
		color_frequency[c]++;
		}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates "
				"color_frequency: ";
		int_vec_print(cout, color_frequency, graph->nb_colors);
		cout << endl;
		}

	// Determine the color c0 with the minimal frequency:
	c0 = -1;
	c0_freq = 0;
	for (c = 0; c < graph->nb_colors; c++) {
		if (f_color_satisfied[c]) {
			if (color_frequency[c]) {
				cout << "rainbow_cliques::find_candidates "
						"satisfied color appears with positive "
						"frequency" << endl;
				cout << "current clique:";
				int_vec_print(cout, current_clique, current_clique_size);
				cout << endl;
				exit(1);
				}
			}
		else {
			if (color_frequency[c] == 0)
				return 0;
			if (c0 == -1) {
				c0 = c;
				c0_freq = color_frequency[c];
				}
			else {
				if (color_frequency[c] < c0_freq) {
					c0 = c;
					c0_freq = color_frequency[c];
					}
				}
			}
		}
	if (f_v) {
		cout << "rainbow_cliques::find_candidates minimal "
				"color is " << c0 << " with frequency "
				<< c0_freq << endl;
		}

	// And now we collect the points with color c0
	// in the array candidates:
	j = 0;
	for (i = 0; i < nb_pts; i++) {
		c = graph->point_color[pt_list[i]];
		if (c == c0) {
			candidates[j++] = pt_list[i];
			}
		}
	if (j != c0_freq) {
		cout << "rainbow_cliques::find_candidates "
				"j != c0_freq" << endl;
		exit(1);
		}

	// Mark color c0 as chosen:
	color_chosen_at_depth[current_clique_size] = c0;

	// we return the size of the candidate set:
	return c0_freq;
}

void rainbow_cliques::clique_found(
		int *current_clique, int verbose_level)
{
	int i;
	
	for (i = 0; i < target_depth; i++) {
		*fp_sol << current_clique[i] << " ";
		}
	*fp_sol << endl;
}

void rainbow_cliques::clique_found_record_in_original_labels(
		int *current_clique, int verbose_level)
{
	int i;
	
	*fp_sol << graph->user_data_size + target_depth << " ";
	for (i = 0; i < graph->user_data_size; i++) {
		*fp_sol << graph->user_data[i] << " ";
		}
	for (i = 0; i < target_depth; i++) {
		*fp_sol << graph->points[current_clique[i]] << " ";
		}
	*fp_sol << endl;
}


void call_back_colored_graph_clique_found(
		clique_finder *CF, int verbose_level)
{
	int f_v = FALSE; //(verbose_level >= 1);

	//cout << "call_back_colored_graph_clique_found" << endl;
	
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data;

	if (f_v) {
		int i, pt, c;
		
		cout << "call_back_colored_graph_clique_found clique";
		int_set_print(cout, CF->current_clique, CF->target_depth);
		cout << endl;
		for (i = 0; i < CF->target_depth; i++) {
			pt = CF->current_clique[i];
			c = R->graph->point_color[pt];
			cout << i << " : " << pt << " : " << c
					<< " : " << R->f_color_satisfied[c] << endl;
			}
		}
	if (R->f_output_solution_raw) {
		R->clique_found(CF->current_clique, verbose_level);
		}
	else {
		R->clique_found_record_in_original_labels(
				CF->current_clique, verbose_level);
		}
}

void call_back_colored_graph_add_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data;
	int c;
	
	c = R->graph->point_color[pt];
	if (R->f_color_satisfied[c]) {
		cout << "call_back_colored_graph_add_point "
				"color already satisfied" << endl;
		exit(1);
		}
	if (c != R->color_chosen_at_depth[current_clique_size]) {
		cout << "call_back_colored_graph_add_point "
				"c != color_chosen_at_depth[current_clique_size]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "call_back_colored_graph_add_point "
				"add_point " << pt << " at depth "
				<< current_clique_size << " color=" << c << endl;
		}
	R->f_color_satisfied[c] = TRUE;
}

void call_back_colored_graph_delete_point(clique_finder *CF, 
	int current_clique_size, int *current_clique, 
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data;
	int c;
	
	c = R->graph->point_color[pt];
	if (!R->f_color_satisfied[c]) {
		cout << "call_back_colored_graph_delete_point "
				"color not satisfied" << endl;
		exit(1);
		}
	R->f_color_satisfied[c] = FALSE;
	if (f_v) {
		cout << "call_back_colored_graph_delete_point "
				"delete_point " << pt << " at depth "
				<< current_clique_size << " color=" << c << endl;
		}
}

int call_back_colored_graph_find_candidates(
	clique_finder *CF,
	int current_clique_size, int *current_clique, 
	int nb_pts, int &reduced_nb_pts, 
	int *pt_list, int *pt_list_inv, 
	int *candidates, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R = (rainbow_cliques *)
			CF->call_back_clique_found_data;
	int ret;

	if (R->f_has_additional_test_function) {

		int tmp_nb_points;

		if (f_v) {
			cout << "call_back_colored_graph_find_candidates "
					"before call_back_additional_test_function" << endl;
			}
		(*R->call_back_additional_test_function)(R, R->user_data, 
			current_clique_size, current_clique, 
			nb_pts, tmp_nb_points, 
			pt_list, pt_list_inv, 
			verbose_level);

		nb_pts = tmp_nb_points;

		if (f_v) {
			cout << "call_back_colored_graph_find_candidates "
					"after call_back_additional_test_function "
					"nb_pts = " << nb_pts << endl;
			}

		}
	
	if (f_v) {
		cout << "call_back_colored_graph_find_candidates "
				"before R->find_candidates" << endl;
		}
	ret = R->find_candidates(
			current_clique_size, current_clique,
			nb_pts, reduced_nb_pts, 
			pt_list, pt_list_inv, 
			candidates, verbose_level);
	if (f_v) {
		cout << "call_back_colored_graph_find_candidates "
				"after R->find_candidates" << endl;
		}
	
	return ret;
}




