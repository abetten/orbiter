// poset_classification_io.cpp
//
// Anton Betten
//
// moved here from DISCRETA/snakesandladders.cpp
// December 27, 2008
// renamed from io.cpp to poset_classification_io.cpp Aug 24, 2011


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


void poset_classification::print_set_verbose(int node)
{
	Poo->get_node(node)->print_set_verbose(this);
}

void poset_classification::print_set_verbose(int level, int orbit)
{
	int node;

	node = Poo->first_node_at_level(level) + orbit;
	Poo->get_node(node)->print_set_verbose(this);
}

void poset_classification::print_set(int node)
{
	Poo->get_node(node)->print_set(this);
}

void poset_classification::print_set(int level, int orbit)
{
	int node;

	node = Poo->first_node_at_level(level) + orbit;
	Poo->get_node(node)->print_set(this);
}


void poset_classification::print_progress_by_extension(
		int size,
		int cur, int prev, int cur_ex,
		int nb_ext_cur, int nb_fuse_cur)
{
	double progress;


	progress = level_progress(size);

	print_level_info(size, prev);
	cout << " **** Upstep extension " << cur_ex << " / "
		<< Poo->node_get_nb_of_extensions(prev) << " with "
		<< nb_ext_cur << " n e w orbits and "
		<< nb_fuse_cur << " fusion nodes. We now have "
		<< cur - Poo->first_node_at_level(size)
		<< " nodes at level " << size;
		cout << ", ";
	print_progress(progress);
	print_progress_by_level(size);
}

void poset_classification::print_progress(
		int size,
		int cur, int prev,
		int nb_ext_cur, int nb_fuse_cur)
{
	double progress;


	progress = level_progress(size);

	print_level_info(size, prev);
	cout << " **** Upstep finished with "
		<< nb_ext_cur << " n e w orbits and "
		<< nb_fuse_cur << " fusion nodes. We now have "
		<< cur - Poo->first_node_at_level(size)
		<< " nodes at level " << size;
		cout << ", ";
	print_progress(progress);
	print_progress_by_level(size);
}

void poset_classification::print_progress(double progress)
{
	double progress0;
	long int progress1, progress2;

	progress0 = progress * 100.;
	progress2 = (long int) (progress0 * 100.);
	progress1 = progress2 / 100;
	progress2 = progress2 % 100;
	cout << "progress: " << progress1 << "."
			<< setw(2) << progress2 << " % " << endl;
}

void poset_classification::print_progress_by_level(int lvl)
{

	Poo->print_progress_by_level(lvl);
}

void poset_classification::print_orbit_numbers(int depth)
{
	int nb_nodes, j;

	nb_nodes = nb_orbits_at_level(depth);
	cout << "###########################################################"
			"#######################################" << endl;
	print_problem_label();
	cout << "Found " << nb_nodes << " orbits at depth " << depth << endl;
	for (j = 0; j <= depth; j++) {
		cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
	cout << "total: " << Poo->first_node_at_level(depth + 1) << endl;
	//gen->print_statistic_on_callbacks();
	compute_and_print_automorphism_group_orders(depth, cout);
}

void poset_classification::print()
{
	int nb_nodes, j;

	cout << "Poset classification:" << endl;
	print_problem_label();
	cout << "Action:" << endl;
	Poset->A->print_info();
	cout << "Action2:" << endl;
	Poset->A2->print_info();
	cout << "Group order:" << Poset->go << endl;
	cout << "Degree:" << Poset->A2->degree << endl;
	cout << "depth:" << depth << endl;
	nb_nodes = nb_orbits_at_level(depth);
	cout << "Found " << nb_nodes << " orbits at depth " << depth << endl;
	for (j = 0; j <= depth; j++) {
		cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
	cout << "total: " << Poo->first_node_at_level(depth + 1) << endl;
}

void poset_classification::print_statistic_on_callbacks_naked()
{
	cout << Poset->A->ptr->nb_times_image_of_called - nb_times_image_of_called0 << "/";
	cout << Poset->A->ptr->nb_times_mult_called - nb_times_mult_called0 << "/";
	cout << Poset->A->ptr->nb_times_invert_called - nb_times_invert_called0 << "/";
	cout << Poset->A->ptr->nb_times_retrieve_called - nb_times_retrieve_called0 << "/";
	cout << Poset->A->ptr->nb_times_store_called - nb_times_store_called0;
}

void poset_classification::print_statistic_on_callbacks()
{
	cout << "# of calls to image_of/mult/invert/retrieve/store: ";
	print_statistic_on_callbacks_naked();
	cout << endl;
}


void poset_classification::prepare_fname_data_file(std::string &fname,
		std::string &fname_base, int depth_completed)
{
	char str[1000];

	fname.assign(fname_base);
	sprintf(str, "_%d.data", depth_completed);
	fname.append(str);
}

void poset_classification::print_representatives_at_level(int lvl)
{
	int i, l;

	l = nb_orbits_at_level(lvl);
	cout << "The " << l << " representatives at level "
			<< lvl << " are:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " / " << l << " : ";
		Poo->get_node_ij(lvl, i)->print_set(this);
		cout << endl;
	}
}

void poset_classification::print_lex_rank(long int *set, int sz)
{
#if 0
	int r1, r2;
	int n;
	combinatorics_domain Combi;

	// ToDo
	n = Poset->A2->degree;
	r1 = Combi.rank_subset(set, sz, n);
	r2 = Combi.rank_k_subset(set, n, sz);

	cout << "lex rank = " << r1 << " lex rank as "
			<< sz << "-subset = " << r2;
#endif
}


void poset_classification::print_problem_label()
{
	if (problem_label[0]) {
		cout << problem_label << " ";
		}
}

void poset_classification::print_level_info(int prev_level, int prev)
{
	int t1, dt;
	orbiter_kernel_system::os_interface Os;

	t1 = Os.os_ticks();
	//cout << "poset_classification::print_level_info t0=" << t0 << endl;
	//cout << "poset_classification::print_level_info t1=" << t1 << endl;
	dt = t1 - t0;
	//cout << "poset_classification::print_level_info dt=" << dt << endl;

	cout << "Time ";
	Os.time_check_delta(cout, dt);
	print_problem_label();
	cout << " : Level " << prev_level << " Node " << prev << " = "
		<< prev - Poo->first_node_at_level(prev_level)
		<< " / "
		<< nb_orbits_at_level(prev_level)
		<< " : ";
}

void poset_classification::print_level_extension_info(
	int prev_level,
	int prev, int cur_extension)
{
	cout << "Level " << prev_level << " Node " << prev << " = "
		<< prev - Poo->first_node_at_level(prev_level)
		<< " / "
		<< nb_orbits_at_level(prev_level)
		<< " Extension " << cur_extension
		<< " / "
		<< Poo->node_get_nb_of_extensions(prev)
		<< " : ";
}

void poset_classification::print_level_extension_coset_info(
	int prev_level,
	int prev, int cur_extension, int coset, int nb_cosets)
{
	cout << "Level " << prev_level << " Node " << prev << " = "
		<< prev - Poo->first_node_at_level(prev_level)
		<< " / "
		<< nb_orbits_at_level(prev_level)
		<< " Extension " << cur_extension
		<< " / "
		<< Poo->node_get_nb_of_extensions(prev)
		<< " : "
		<< "Coset " << coset << " / " << nb_cosets << " : ";
}

void poset_classification::print_node(int node)
{
	cout << "poset_classification::print_node "
			"node " << node << ":" << endl;
	Poo->get_node(node)->print_node(this);
}

void poset_classification::print_extensions_at_level(
		ostream &ost, int lvl)
{
	int i, node;
	int fst, len;
	poset_orbit_node *O;

	ost << "extensions at level " << lvl << ":" << endl;
	fst = Poo->first_node_at_level(lvl);
	len = nb_orbits_at_level(lvl);
	ost << "there are " << len << " nodes at level " << lvl << ":" << endl;
	for (i = 0; i < len; i++) {
		node = fst + i;
		O = Poo->get_node(node);
		ost << "Node " << i << " / " << len << " = " << node << ":" << endl;
		O->print_extensions(ost);
	}
}

void poset_classification::print_fusion_nodes(int depth)
{
	int i, f, l, j, h;

	for (i = 0; i <= depth; i++) {
		f = Poo->first_node_at_level(i);
		l = nb_orbits_at_level(i);
		for (j = 0; j < l; j++) {
			poset_orbit_node *O;

			O = Poo->get_node(f + j);
			for (h = 0; h < O->get_nb_of_extensions(); h++) {
				extension *E = O->get_E(h);

				if (E->get_type() == EXTENSION_TYPE_FUSION) {
					cout << "fusion (" << f + j << "/" << h
							<< ") -> (" << E->get_data1() << "/"
							<< E->get_data2() << ")" << endl;
					}
				}
			}
		}
}


void poset_classification::read_data_file(int &depth_completed,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int size;
	int nb_group_elements;
	orbiter_kernel_system::memory_object *m;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "poset_classification::read_data_file "
				"fname = " << fname << endl;
		cout << "Poset->A->elt_size_in_int = "
				<< Poset->A->elt_size_in_int << endl;
		cout << "Poset->A->coded_elt_size_in_char = "
				<< Poset->A->coded_elt_size_in_char << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	size = Fio.file_size(fname);
	if (f_v) {
		cout << "file size = " << size << endl;
	}
	if (size == -1) {
		cout << "error: the file does not exist" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification::read_data_file "
				"before m->alloc" << endl;
	}

	m = NEW_OBJECT(orbiter_kernel_system::memory_object);
	m->alloc(size, 0);

	if (f_v) {
		cout << "poset_classification::read_data_file after m->alloc" << endl;
	}

	m->used_length = 0;


	if (f_v) {
		cout << "poset_classification::read_data_file "
				"Before reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	{
		ifstream fp(fname, ios::binary);

		fp.read((char *) m->data, size);
	}
	if (f_v) {
		cout << "poset_classification::read_data_file "
				"Read file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "poset_classification::read_data_file "
				"before read_memory_object" << endl;
	}
	Poo->read_memory_object(depth_completed, m,
			nb_group_elements, verbose_level - 2);
	if (f_v) {
		cout << "poset_classification::read_data_file "
				"after poset_classification_read_memory" << endl;
	}

	FREE_OBJECT(m);
	if (f_v) {
		cout << "poset_classification::read_data_file done" <<endl;
	}

}

void poset_classification::write_data_file(int depth_completed,
		std::string &fname_base, int verbose_level)
{
	orbiter_kernel_system::memory_object *m;
	int f_v = (verbose_level >= 1);
	string fname;
	int nb_group_elements;
	long int size0;
	int verbose_level1;
	orbiter_kernel_system::file_io Fio;

	prepare_fname_data_file(fname, fname_base, depth_completed);
	//sprintf(fname, "%s_%d.data", fname_base, depth_completed);

	if (f_v) {
		cout << "poset_classification::write_data_file "
				"fname = " << fname << endl;
		cout << "Poset->A->elt_size_in_int = "
				<< Poset->A->elt_size_in_int << endl;
		cout << "Poset->A->coded_elt_size_in_char = "
				<< Poset->A->coded_elt_size_in_char << endl;
	}
	size0 = Poo->calc_size_on_file(depth_completed, verbose_level);
	if (f_v) {
		cout << "size on file = " << size0 << endl;
	}

	verbose_level1 = verbose_level;
	if (size0 > 1000 * ONE_MILLION) {
		cout << "poset_classification::write_data_file "
				"file=" << fname << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big (> 1 GB)" << endl;
		verbose_level1 = 2;
	}

	m = NEW_OBJECT(orbiter_kernel_system::memory_object);
	//m->alloc(10, 0);
	m->alloc(size0, 0);
	m->used_length = 0;

	if (f_v) {
		cout << "poset_classification::write_data_file "
				"before write_memory_object" << endl;
	}
	Poo->write_memory_object(
			depth_completed, m,
			nb_group_elements, verbose_level1);
	if (f_v) {
		cout << "poset_classification::write_data_file "
				"after write_memory_object, written " << nb_group_elements
				<< " group elements" << endl;
		cout << "m->alloc_length=" << m->alloc_length << endl;
		cout << "m->used_length=" << m->used_length << endl;
		cout << "size0=" << size0 << endl;
	}
	if (m->used_length != size0) {
		cout << "poset_classification::write_data_file "
				"m->used_length != size0" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "poset_classification::write_data_file "
				"Writing data of size " << m->used_length
				<< " to file " << fname << endl;
	}
	{
		ofstream fp(fname, ios::binary);

		fp.write(m->data, m->used_length);
	}
	if (f_v) {
		cout << "poset_classification::write_data_file "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	FREE_OBJECT(m);

	if (f_v) {
		cout << "poset_classification::write_data_file "
			"finished written file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
		cout << " nb_group_elements=" << nb_group_elements << endl;
	}
}

void poset_classification::write_file(
		ofstream &fp,
		int depth_completed, int verbose_level)
{
	orbiter_kernel_system::memory_object *m;
	int f_v = (verbose_level >= 1);
	long int size0;
	int nb_group_elements = 0;


	if (f_v) {
		cout << "poset_classification::write_file "
				"depth_completed=" << depth_completed << endl;
	}
	size0 = Poo->calc_size_on_file(depth_completed, 0 /*verbose_level*/);
	if (f_v) {
		cout << "poset_classification::write_file size on file = " << size0 << endl;
	}

	if (size0 > 1000 * ONE_MILLION) {
		cout << "poset_classification::write_file" << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big (> 1 GB)" << endl;
	}

	m = NEW_OBJECT(orbiter_kernel_system::memory_object);
	m->alloc(10, 0);
	//m->alloc(size0, 0);
	m->used_length = 0;

	if (f_v) {
		cout << "poset_classification::write_file "
				"before write_memory_object" << endl;
	}
	Poo->write_memory_object(depth_completed, m,
			nb_group_elements, 0 /*verbose_level*/);
	if (TRUE) {
		cout << "poset_classification::write_file "
				"after write_memory_object" << endl;
		cout << "m->used_length=" << m->used_length << endl;
		cout << "m->alloc_length=" << m->alloc_length << endl;
	}
	if (m->used_length != size0) {
		cout << "poset_classification::write_file "
				"m->used_length != size0" << endl;
		cout << "m->used_length=" << m->used_length << endl;
		cout << "size0" << size0 << endl;
		exit(1);
	}

	long int size;
	size = m->used_length;

	if (size != size0) {
		cout << "poset_classification::write_file size != size0" << endl;
		cout << "poset_classification::write_file size = " << size << endl;
		cout << "poset_classification::write_file size0 = " << size0 << endl;
		exit(1);

	}

	if (f_v) {
		cout << "poset_classification::write_file "
				"before fp.write" << endl;
	}
	fp.write((char *) &depth_completed, sizeof(int));
	fp.write((char *) &size, sizeof(long int));
	fp.write(m->data, size);
	if (f_v) {
		cout << "poset_classification::write_file "
				"after fp.write" << endl;
	}

	FREE_OBJECT(m);

	if (f_v) {
		cout << "poset_classification::write_file done" << endl;
	}
}

void poset_classification::read_file(
		ifstream &fp,
		int &depth_completed, int verbose_level)
{
	orbiter_kernel_system::memory_object *m;
	int f_v = (verbose_level >= 1);
	long int size;
	int nb_group_elements;

	if (f_v) {
		cout << "poset_classification::read_file" << endl;
	}


	fp.read((char *) &depth_completed, sizeof(int));
	fp.read((char *) &size, sizeof(long int));

	if (f_v) {
		cout << "poset_classification::read_file "
				"size = " << size << endl;
	}

	m = NEW_OBJECT(orbiter_kernel_system::memory_object);

	m->alloc(size, 0);
	m->used_length = 0;

	fp.read(m->data, size);


	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "poset_classification::read_file "
				"before poset_classification_read_memory" << endl;
	}
	Poo->read_memory_object(depth_completed, m,
			nb_group_elements, verbose_level - 0);
	if (f_v) {
		cout << "poset_classification::read_file "
				"after poset_classification_read_memory" << endl;
	}

	FREE_OBJECT(m);

	if (f_v) {
		cout << "poset_classification::read_file done, "
				"depth_completed=" << depth_completed << endl;
	}
}

void poset_classification::housekeeping(int i,
		int f_write_files, int t0, int verbose_level)
{
	int j, nb_nodes;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);
	//int f_embedded = TRUE;
	
	if (f_v) {
		cout << "poset_classification::housekeeping "
				"level=" << i << endl;
		cout << "poset_classification::housekeeping "
				"verbose_level=" << verbose_level << endl;
		cout << "poset_classification::housekeeping "
				"problem_label_with_path=" << problem_label_with_path << endl;
	}
	nb_nodes = nb_orbits_at_level(i);
	if (f_v) {
		cout << "###################################################"
				"###############################################" << endl;
		print_problem_label();
		cout << "Found " << nb_nodes << " orbits at depth " << i << endl;

		if (f_v5) {
			cout << "orbits at level " << i << ":" << endl;
			print_representatives_at_level(i);
		}

		for (j = 0; j <= i; j++) {
			cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
		cout << "total: " << Poo->first_node_at_level(i + 1) << endl;



		//print_statistic_on_callbacks();
		compute_and_print_automorphism_group_orders(i, cout);
		//registry_dump_sorted();
		//registry_dump_sorted_by_size();
		//cout << "nb_times_trace=" << nb_times_trace << endl;
		//cout << "nb_times_trace_was_saved="
		// << nb_times_trace_was_saved << endl;
		//cout << "f_write_files=" << f_write_files << endl;
		int nb1, nb2;

		nb1 = Schreier_vector_handler->nb_calls_to_coset_rep_inv;
		nb2 = Schreier_vector_handler->nb_calls_to_coset_rep_inv_recursion;
		cout << "nb_calls_to_coset_rep_inv="
				<< nb1 << endl;
		cout << "nb_calls_to_coset_rep_inv_recursion="
				<< nb2 << endl;
		cout << "average word length=" <<
				(double) nb2 / (double) nb1 << endl;
	}
	if (Control->f_find_node_by_stabilizer_order) {
		find_node_by_stabilizer_order(i,
				Control->find_node_by_stabilizer_order, verbose_level);
	}
	if (f_vv) {
		if (nb_nodes < 1000) {
			int f_with_strong_generators = FALSE;
			int f_long_version = FALSE;
			Poo->write_lvl(cout, i, t0, f_with_strong_generators,
					f_long_version, verbose_level - 2);
		}
	}
	
	if (f_write_files) {
		string my_fname_base;
		
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"writing files" << endl;
		}




#if 1

		string fname_reps_csv;
		char str[1000];

		fname_reps_csv.assign(problem_label_with_path);
		sprintf(str, "_reps_lvl_%d", i);
		fname_reps_csv.append(str);
		fname_reps_csv.append(".csv");

		Poo->save_representatives_at_level_to_csv(fname_reps_csv, i, verbose_level);


		my_fname_base.assign(problem_label_with_path);
		my_fname_base.append("a");
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"my_fname_base=" << my_fname_base << endl;
			cout << "poset_classification_housekeeping "
					"before write_level_file_binary" << endl;
		}
		write_level_file_binary(i, my_fname_base,
				0/*verbose_level*/);
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"after write_level_file_binary" << endl;
		}
		if (i) {		
			my_fname_base.assign(problem_label_with_path);
			my_fname_base.append("b");
			if (f_v) {
				cout << "poset_classification_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "poset_classification_housekeeping "
						"before write_level_file_binary" << endl;
			}
			write_level_file_binary(i - 1, my_fname_base, 0/*verbose_level*/);
			if (f_v) {
				cout << "poset_classification_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "poset_classification_housekeeping "
						"before write_sv_level_file_binary" << endl;
			}
			write_sv_level_file_binary(i - 1, my_fname_base, 
				FALSE, 0, 0, 0 /*verbose_level*/);
		}
#endif
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"before write_lvl_file" << endl;
		}
		Poo->write_lvl_file(problem_label_with_path, i, t0,
				FALSE /* f_with_strong_generators */,
				FALSE /* f_long_version */, 0);
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"after write_lvl_file" << endl;
		}
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"before poset_classification::write_data_file" << endl;
		}
		poset_classification::write_data_file(i /* depth_completed */,
				problem_label_with_path, verbose_level);

		if (f_v) {
			cout << "poset_classification::housekeeping "
					"after poset_classification::write_data_file" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"not writing files" << endl;
		}
	}

#if 0
	if (Control->f_Log) {
		int verbose_level = 1;
		int f = first_poset_orbit_node_at_level[i];
		int len = nb_orbits_at_level(i);
		print_problem_label();
		cout << "There are " << len
				<< " nodes at level " << i << ":" << endl;
		for (j = 0; j < len; j++) {
			Log_nodes(f + j, i, cout, FALSE, verbose_level);
		}
	}

	if (Control->f_log && i == sz) {
		int verbose_level = 1;
		int ii;

		for (ii = 0; ii <= sz; ii++) {
			int f = first_poset_orbit_node_at_level[ii];
			int len = nb_orbits_at_level(ii);
			print_problem_label();
			cout << "There are " << len
					<< " nodes at level " << ii << ":" << endl;
			for (j = 0; j < len; j++) {
				Log_nodes(f + j, ii, cout, FALSE, verbose_level);
			}
		}
	}
#endif

	if (Control->f_T || (Control->f_t && i == sz)) {
		if (f_v) {
			cout << "poset_classification::housekeeping "
					"before write_treefile_and_draw_tree" << endl;
		}

		write_treefile(problem_label_with_path, i,
				Control->draw_options,
				//Control->xmax, Control->ymax, Control->radius, f_embedded,
				0 /*verbose_level - 1*/);
			// in poset_classification_draw.cpp

		if (f_v) {
			cout << "poset_classification::housekeeping "
					"after write_treefile_and_draw_tree" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"not writing tree" << endl;
		}
	}

	if (f_v) {
		cout << "poset_classification::housekeeping done" << endl;
	}
}

void poset_classification::housekeeping_no_data_file(int i,
		int t0, int verbose_level)
{
	int j;
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	//int f_embedded = TRUE;
	
	if (f_v) {
		cout << "poset_classification::"
				"housekeeping_no_data_file "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "######################################################"
				"############################################" << endl;
		cout << "depth " << i << " completed, found " 
			<< nb_orbits_at_level(i) << " orbits" << endl;


		if (f_v5) {
			cout << "orbits at level " << i << ":" << endl;
			print_representatives_at_level(i);
		}



		for (j = 0; j <= i; j++) {
			cout << j << " : " << nb_orbits_at_level(j)
					<< " orbits" << endl;
		}
		cout << "total: " << Poo->first_node_at_level(i + 1) << endl;
		compute_and_print_automorphism_group_orders(i, cout);
	}

	if (Control->f_W || (Control->f_w && i == sz)) {
#if 0
		char fname_base2[1000];
		
		sprintf(fname_base2, "%sa", fname_base);
		write_level_file_binary(i, fname_base2, 1/*verbose_level*/);
		if (i) {		
			sprintf(fname_base2, "%sb", fname_base);
			write_level_file_binary(i - 1, fname_base2, 1/*verbose_level*/);
			write_sv_level_file_binary(i - 1, 
				fname_base, FALSE, 0, 0, 1/*verbose_level*/);
		}
#endif

		Poo->write_lvl_file(problem_label_with_path, i, t0,
				FALSE /* f_with_strong_generators */,
				FALSE /* f_long_version */, 0);
		
		//poset_classification_write_data_file(gen,
		// i /* depth_completed */, gen->fname_base, 0);

	}

	if (Control->f_T || (Control->f_t && i == sz)) {
		write_treefile(problem_label_with_path, i,
				Control->draw_options,
				//Control->xmax, Control->ymax, Control->radius, f_embedded,
				verbose_level - 1);
	}
	if (f_v) {
		cout << "poset_classification::"
				"housekeeping_no_data_file done" << endl;
	}
}

void poset_classification::create_fname_sv_level_file_binary(std::string &fname,
		std::string &fname_base, int level)
{
	char str[1000];

	fname.assign(fname_base);
	sprintf(str, "_lvl_%d_sv.data", level);
	fname.append(str);

}

int poset_classification::test_sv_level_file_binary(
		int level, std::string &fname_base)
{
	string fname;
	orbiter_kernel_system::file_io Fio;
	
	create_fname_sv_level_file_binary(fname, fname_base, level);
	//sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	if (Fio.file_size(fname) >= 1) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void poset_classification::read_sv_level_file_binary(
	int level, std::string &fname_base,
	int f_split, int split_mod, int split_case, 
	int f_recreate_extensions, int f_dont_keep_sv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	orbiter_kernel_system::file_io Fio;
	
	create_fname_sv_level_file_binary(fname, fname_base, level);
	//sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::read_sv_level_file_binary "
				"reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	{
		ifstream fp(fname, ios::binary);

		Poo->read_sv_level_file_binary2(level, fp,
			f_split, split_mod, split_case,
			f_recreate_extensions, f_dont_keep_sv,
			verbose_level - 1);
	}

}

void poset_classification::write_sv_level_file_binary(
	int level, std::string &fname_base,
	int f_split, int split_mod, int split_case, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	orbiter_kernel_system::file_io Fio;

	create_fname_sv_level_file_binary(fname, fname_base, level);
	//sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary "
				"fname = " << fname << endl;
	}


	{
		ofstream fp(fname, ios::binary);

		Poo->write_sv_level_file_binary2(level, fp,
			f_split, split_mod, split_case,
			verbose_level);
	}

	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary "
			"finished written file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
	}
}

void poset_classification::read_level_file_binary(int level,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	char str[1000];
	int nb_group_elements;
	orbiter_kernel_system::file_io Fio;
	
	fname.assign(fname_base);
	sprintf(str, "_lvl_%d.data", level);
	fname.append(str);
	
	if (f_v) {
		cout << "poset_classification::read_level_file_binary "
				"reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (Fio.file_size(fname) < 0) {
		cout << "poset_classification::read_level_file_binary "
				"probems while reading file " << fname << endl;
		exit(1);
	}

	{
		ifstream fp(fname, ios::binary);

		Poo->read_level_file_binary2(level, fp,
				nb_group_elements, verbose_level);
	}

}

void poset_classification::write_level_file_binary(int level,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	char str[1000];
	int nb_group_elements;
	orbiter_kernel_system::file_io Fio;

	fname.assign(fname_base);
	sprintf(str, "_lvl_%d.data", level);
	fname.append(str);
	
	if (f_v) {
		cout << "poset_classification::write_level_file_binary "
				"fname = " << fname << endl;
	}



	{
		ofstream fp(fname, ios::binary);

		Poo->write_level_file_binary2(level, fp,
				nb_group_elements, verbose_level);
	}
	
	if (f_v) {
		cout << "poset_classification::write_level_file_binary "
				"finished written file "
			<< fname << " of size " << Fio.file_size(fname)
			<< " nb_group_elements=" << nb_group_elements << endl;
	}
}


void poset_classification::recover(
		std::string &recover_fname,
		int &depth_completed, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::recover "
				"recovering from file " << recover_fname << endl;
	}
	read_data_file(depth_completed, recover_fname, verbose_level);
	if (f_v) {
		cout << "poset_classification::recover "
				"recovering finished, "
				"depth_completed = " << depth_completed << endl;
	}
}

void poset_classification::make_fname_lvl_file_candidates(std::string &fname,
		std::string &fname_base, int lvl)
{
	char str[1000];

	fname.assign(fname_base);
	sprintf(str, "_lvl_%d_candidates.txt", lvl);
	fname.append(str);

}

void poset_classification::make_fname_lvl_file(std::string &fname,
		std::string &fname_base, int lvl)
{
	char str[1000];

	fname.assign(fname_base);
	sprintf(str, "_lvl_%d", lvl);
	fname.append(str);

}

void poset_classification::make_fname_lvl_reps_file(std::string &fname,
		std::string &fname_base, int lvl)
{
	char str[1000];

	fname.assign(fname_base);
	sprintf(str, "_lvl_%d_reps", lvl);
	fname.append(str);

}


#if 0
void poset_classification::Log_nodes(int cur, int depth,
		ostream &f, int f_recurse,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, next;
	poset_orbit_node *node = &root[cur];
		

	if (f_v) {
		cout << "Log_nodes cur=" << cur << endl;
		}
	if (f_base_case && cur < Base_case->size) {
		return; // !!!
		}
	if (f_v) {
		f << "Node " << cur << endl;
		f << "===============" << endl;
		node->log_current_node(this, depth, f,
				FALSE /* f_with_strong_generators */,
				verbose_level);
		//f << "the stabilizer has order ";
		//G.print_group_order(f);
		//f << endl;	
		
		f << "with " << node->nb_strong_generators
				<< " strong poset_classifications:" << endl;
		if (f_v) {
			cout << "Log_nodes cur=" << cur
					<< " printing strong poset_classifications" << endl;
			}
		for (i = 0; i < node->nb_strong_generators; i++) {
			Poset->A->element_retrieve(
					node->hdl_strong_generators[i], Elt1, 0);
			Poset->A->element_print_quick(Elt1, f);
			f << endl;
			if (Poset->A->degree < 100) {
				Poset->A->element_print_as_permutation(Elt1, f);
				f << endl;
				}
			}

		if (node->nb_strong_generators) {
			if (f_v) {
				cout << "Log_nodes cur=" << cur
						<< " printing tl" << endl;
				}
			f << "tl: ";
			int_vec_print(f, node->tl, Poset->A->base_len());
			f << endl;
			}
		
		if (f_v) {
			cout << "Log_nodes cur=" << cur
					<< " printing extensions" << endl;
			}
		node->print_extensions(f);
		f << endl;

		for (i = 0; i < node->get_nb_of_extensions(); i++) {
			if (node->get_E(i)->get_type() == EXTENSION_TYPE_FUSION) {
				f << "fusion node " << i << ":" << endl;
				Poset->A->element_retrieve(node->get_E(i)->get_data(), Elt1, 0);
				Poset->A->element_print_verbose(Elt1, f);
				f << endl;
				}
			}
		}
	else {
		//cout << "log_current_node node=" << node->node
		// << " prev=" << node->prev << endl;
		node->log_current_node(this, depth, f,
				FALSE /* f_with_strong_generators */, 0);
		}
	
	if (f_recurse) {
		//cout << "recursing into dependent nodes" << endl;
		for (i = 0; i < node->get_nb_of_extensions(); i++) {
			if (node->get_E(i)->get_type() == EXTENSION_TYPE_EXTENSION) {
				if (node->get_E(i)->get_data() >= 0) {
					next = node->get_E(i)->get_data();
					Log_nodes(next, depth + 1, f, TRUE, verbose_level);
					}
				}
			}
		}
}
#endif

void poset_classification::log_current_node(ostream &f, int size)
{
	//longinteger_object go;
	int i;
	

	f << size << " ";
	for (i = 0; i < size; i++) {
		f << set_S[i] << " ";
	}
	f << endl;

}



void poset_classification::make_spreadsheet_of_orbit_reps(
		data_structures::spreadsheet *&Sp, int max_depth)
{
	int Nb_orbits, nb_orbits, i, level, first;
	pchar *Text_level;
	pchar *Text_node;
	pchar *Text_orbit_reps;
	pchar *Text_stab_order;
	pchar *Text_orbit_length;
	pchar *Text_schreier_vector_length;
	ring_theory::longinteger_object stab_order, orbit_length;
	int schreier_vector_length;
	long int *rep;
	char str[1000];
	poset_orbit_node *O;

	Nb_orbits = 0;
	for (level = 0; level <= max_depth; level++) {
		Nb_orbits += nb_orbits_at_level(level);
	}

	rep = NEW_lint(max_depth);
	Text_level = NEW_pchar(Nb_orbits);
	Text_node = NEW_pchar(Nb_orbits);
	Text_orbit_reps = NEW_pchar(Nb_orbits);
	Text_stab_order = NEW_pchar(Nb_orbits);
	Text_orbit_length = NEW_pchar(Nb_orbits);
	Text_schreier_vector_length = NEW_pchar(Nb_orbits);

	first = 0;
	for (level = 0; level <= max_depth; level++) {
		first = Poo->first_node_at_level(level);
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {
			snprintf(str,1000,  "%d", level);
			Text_level[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_level[first + i], str);

			snprintf(str, 1000, "%d", i);
			Text_node[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_node[first + i], str);

			get_set_by_level(level, i, rep);
			Lint_vec_print_to_str(str, rep, level);
			Text_orbit_reps[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_orbit_reps[first + i], str);
			
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);
			stab_order.print_to_string(str);
			Text_stab_order[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_stab_order[first + i], str);
			
			orbit_length.print_to_string(str);
			Text_orbit_length[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_orbit_length[first + i], str);
			
			O = get_node_ij(level, i);
			if (O->has_Schreier_vector()) {
				schreier_vector_length = O->get_nb_of_live_points();
			}
			else {
				schreier_vector_length = 0;
			}
			snprintf(str, 1000, "%d", schreier_vector_length);
			Text_schreier_vector_length[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_schreier_vector_length[first + i], str);
			}
		}
	Sp = NEW_OBJECT(data_structures::spreadsheet);
	Sp->init_empty_table(Nb_orbits + 1, 7);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const char **) Text_level, "Level");
	Sp->fill_column_with_text(2, (const char **) Text_node, "Node");
	Sp->fill_column_with_text(3, (const char **) Text_orbit_reps, "Orbit rep");
	Sp->fill_column_with_text(4, (const char **) Text_stab_order, "Stab order");
	Sp->fill_column_with_text(5, (const char **) Text_orbit_length, "Orbit length");
	Sp->fill_column_with_text(6, (const char **) Text_schreier_vector_length, "Schreier vector length");

#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_lint(rep);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_level[i]);
	}
	FREE_pchar(Text_level);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_node[i]);
	}
	FREE_pchar(Text_node);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_orbit_reps[i]);
	}
	FREE_pchar(Text_orbit_reps);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_stab_order[i]);
	}
	FREE_pchar(Text_stab_order);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_orbit_length[i]);
	}
	FREE_pchar(Text_orbit_length);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_schreier_vector_length[i]);
	}
	FREE_pchar(Text_schreier_vector_length);
	
}

void poset_classification::make_spreadsheet_of_level_info(
		data_structures::spreadsheet *&Sp, int max_depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = FALSE; //(verbose_level >= 1);
	int nb_rows, Nb_orbits, nb_orbits, i, level;
	pchar *Text_label;
	pchar *Text_nb_orbits;
	pchar *Text_orbit_length_sum;
	pchar *Text_schreier_vector_length_sum;
	pchar *Text_binomial;
	ring_theory::longinteger_object stab_order, orbit_length,
		orbit_length_sum, orbit_length_total;
	ring_theory::longinteger_object a, a_total;
	ring_theory::longinteger_domain D;
	combinatorics::combinatorics_domain C;
	int schreier_vector_length_int;
	ring_theory::longinteger_object schreier_vector_length,
		schreier_vector_length_sum, schreier_vector_length_total;
	int *rep;
	char str[1000];
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_classification::"
				"make_spreadsheet_of_level_info" << endl;
	}
	nb_rows = max_depth + 2; // one extra row for totals
	rep = NEW_int(max_depth);
	Text_label = NEW_pchar(nb_rows);
	Text_nb_orbits = NEW_pchar(nb_rows);
	Text_orbit_length_sum = NEW_pchar(nb_rows);
	Text_schreier_vector_length_sum = NEW_pchar(nb_rows);
	Text_binomial = NEW_pchar(nb_rows);

	Nb_orbits = 0;
	orbit_length_total.create(0, __FILE__, __LINE__);
	schreier_vector_length_total.create(0, __FILE__, __LINE__);
	a_total.create(0, __FILE__, __LINE__);
	
	for (level = 0; level <= max_depth; level++) {

		if (f_v) {
			cout << "poset_classification::"
					"make_spreadsheet_of_level_info "
					"level = " << level << " / " << max_depth << endl;
		}
		nb_orbits = nb_orbits_at_level(level);


		snprintf(str, 1000, "%d", level);
		Text_label[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_label[level], str);

		snprintf(str, 1000, "%d", nb_orbits);
		Text_nb_orbits[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_nb_orbits[level], str);

		orbit_length_sum.create(0, __FILE__, __LINE__);
		schreier_vector_length_sum.create(0, __FILE__, __LINE__);

		for (i = 0; i < nb_orbits; i++) {
			
			if (FALSE) {
				cout << "poset_classification::"
						"make_spreadsheet_of_level_info "
						"level = " << level << " / " << max_depth
						<< " orbit " << i << " / " << nb_orbits << endl;
			}
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);

			D.add_in_place(orbit_length_sum, orbit_length);
			
			
			O = get_node_ij(level, i);

			if (O->has_Schreier_vector()) {
				schreier_vector_length_int = O->get_nb_of_live_points();


			}
			else {
				//cout << "node " << level << " / " << i
				//		<< " does not have a Schreier vector" << endl;
				schreier_vector_length_int = 1;
			}
			if (schreier_vector_length_int <= 0) {
				schreier_vector_length_int = 1;
				}
			schreier_vector_length.create(schreier_vector_length_int, __FILE__, __LINE__);

			if (schreier_vector_length_int >= 0) {
				D.add_in_place(schreier_vector_length_sum,
						schreier_vector_length);
			}

		}

		//cout << "poset_classification::make_spreadsheet_of_level_info
		// computing binomial coeffcient" << endl;
		C.binomial(a, Poset->A2->degree, level, FALSE);

		Nb_orbits += nb_orbits;
		D.add_in_place(orbit_length_total, orbit_length_sum);
		D.add_in_place(schreier_vector_length_total,
				schreier_vector_length_sum);
		D.add_in_place(a_total, a);

		orbit_length_sum.print_to_string(str);
		Text_orbit_length_sum[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_orbit_length_sum[level], str);

		schreier_vector_length_sum.print_to_string(str);
		Text_schreier_vector_length_sum[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_schreier_vector_length_sum[level], str);

		a.print_to_string(str);
		Text_binomial[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_binomial[level], str);

	}

	level = max_depth + 1;
	snprintf(str, 1000, "total");
	Text_label[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_label[level], str);

	snprintf(str, 1000, "%d", Nb_orbits);
	Text_nb_orbits[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_nb_orbits[level], str);

	orbit_length_total.print_to_string(str);
	Text_orbit_length_sum[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_orbit_length_sum[level], str);

	schreier_vector_length_total.print_to_string(str);
	Text_schreier_vector_length_sum[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_schreier_vector_length_sum[level], str);

	a_total.print_to_string(str);
	Text_binomial[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_binomial[level], str);


	Sp = NEW_OBJECT(data_structures::spreadsheet);
	Sp->init_empty_table(nb_rows + 1, 6);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const char **) Text_label, "Level");
	Sp->fill_column_with_text(2, (const char **) Text_nb_orbits, "Nb_orbits");
	Sp->fill_column_with_text(3, (const char **) Text_orbit_length_sum, "Orbit_length_sum");
	Sp->fill_column_with_text(4, (const char **) Text_schreier_vector_length_sum, "Schreier_vector_length_sum");
	Sp->fill_column_with_text(5, (const char **) Text_binomial, "Binomial");



#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_int(rep);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_label[i]);
	}
	FREE_pchar(Text_label);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_nb_orbits[i]);
	}
	FREE_pchar(Text_nb_orbits);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_orbit_length_sum[i]);
	}
	FREE_pchar(Text_orbit_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_schreier_vector_length_sum[i]);
	}
	FREE_pchar(Text_schreier_vector_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_binomial[i]);
	}
	FREE_pchar(Text_binomial);
	
}




void poset_classification::create_schreier_tree_fname_mask_base(
		std::string &fname_mask)
{

	fname_mask.assign(problem_label_with_path);
	fname_mask.append(Control->schreier_tree_prefix);
	fname_mask.append("schreier_tree_node_%d_%d");
}

void poset_classification::create_schreier_tree_fname_mask_base_tex(
		std::string &fname_mask)
{

	fname_mask.assign(problem_label_with_path);
	fname_mask.append(Control->schreier_tree_prefix);
	fname_mask.append("schreier_tree_node_%d.tex");
}

void poset_classification::create_shallow_schreier_tree_fname_mask_base(
		std::string &fname_mask)
{

	fname_mask.assign(problem_label_with_path);
	fname_mask.append(Control->schreier_tree_prefix);
	fname_mask.append("shallow_schreier_tree_node_%d_%d");

}

void poset_classification::create_shallow_schreier_tree_fname_mask(
		std::string &fname, int node)
{

	char str[1000];

	sprintf(str, "%d", node);
	fname.assign(problem_label_with_path);
	fname.append(Control->schreier_tree_prefix);
	fname.append("shallow_schreier_tree_node_");
	fname.append(str);
	fname.append("_%d");

}

void poset_classification::make_fname_candidates_file_default(
		char *fname2000, int level)
{
	snprintf(fname2000, 2000, "%s_lvl_%d_candidates.bin",
			problem_label_with_path.c_str(), level);
}


void poset_classification::wedge_product_export_magma(
		int n, int q, int vector_space_dimension,
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "poset_classification::wedge_product_export_magma" << endl;
	}

	//int level;
	long int *the_set;
	int *v;
	int a, i, j, h, fst, len, ii, jj;
	ring_theory::longinteger_object go;
	int *Elt;

	//level = depth_completed + 1;


	the_set = NEW_lint(level);
	v = NEW_int(vector_space_dimension);
	Elt = NEW_int(Poset->A->elt_size_in_int);

	fst = Poo->first_node_at_level(level);
	len = Poo->first_node_at_level(level + 1) - fst;
	if (f_v) {
		cout << "exporting to magma" << endl;
		cout << "fst=" << fst << " len=" << len << endl;
	}
	poset_orbit_node *O;
	char fname[1000];

	snprintf(fname, 1000, "Wedge_n%d_q%d_d%d.magma", n, q, level);

	{
		ofstream f(fname);

		f << "// file " << fname << endl;
		f << "n := " << n << ";" << endl;
		f << "q := " << q << ";" << endl;
		f << "d := " << level << ";" << endl;
		f << "n2 := " << vector_space_dimension << ";" << endl;
		f << "V := VectorSpace (GF (q), n2);" << endl;
		f << endl;
		f << "/* list of orbit reps */" << endl;
		f << "L := [" << endl;
		f << endl;

		for (i = 0; i < len; i++) {
			O = Poo->get_node(fst + i);

			f << "// orbit rep " << i << endl;
			f << "[" << endl;
			O->store_set_to(this, level - 1, the_set);
			for (j = 0; j < level; j++) {
				a = the_set[j];
				unrank_point(v, a);
				f << "[ ";
				for (h = 0; h < vector_space_dimension; h++) {
					f << v[h];
					if (h < vector_space_dimension - 1) {
						f << ", ";
					}
				}
				f << " ]";
				if (j < level - 1) {
					f << "," << endl;
				}
				else {
					f << "]" << endl;
				}
			}
			if (i < len - 1) {
				f << "," << endl << endl;
			}
			else {
				f << endl << "];" << endl << endl;
			}
		} // next i

		f << "// list of orbit lengths " << endl;
		f << "len := \[";

		for (i = 0; i < len; i++) {

			if ((i % 20) == 0) {
				f << endl;
				f << "// orbits " << i << " and following:" << endl;
			}

			orbit_length(i, level, go);
			f << go;
			if (i < len - 1) {
				f << ", ";
			}
		}
		f << "];" << endl << endl;


		f << "// subspaces of vector space " << endl;
		f << "L := [sub< V | L[i]>: i in [1..#L]];" << endl;

		f << "// stabilisers " << endl;
		f << "P := GL(n, q);" << endl;
		f << "E := ExteriorSquare (P);" << endl;


		f << "// base:" << endl;
		f << "BV := VectorSpace (GF (q), n);" << endl;
		f << "B := [ BV | " << endl;
		for (i = 0; i < Poset->A->base_len(); i++) {
			a = Poset->A->base_i(i);
			Poset->VS->F->PG_element_unrank_modified(v, 1, n, a);
			//(*Gen->unrank_point_func)(v, a, Gen->rank_point_data);
			f << "[ ";
			for (h = 0; h < n; h++) {
				f << v[h];
				if (h < n - 1) {
					f << ", ";
				}
			}
				if (i < Poset->A->base_len() - 1) {
					f << "], " << endl;
				}
			else {
				f << " ]" << endl;
			}
		}
		f << "];" << endl;
		f << endl;
		f << "P`Base := B;" << endl;

		f << "// list of stabilizer generators" << endl;
		f << "S := [" << endl;
		f << endl;

		for (i = 0; i < len; i++) {
			O = Poo->get_node(fst + i);

			std::vector<int> gen_hdl;
			std::vector<int> tl;

			O->get_strong_generators_handle(gen_hdl, verbose_level);

			O->get_tl(tl, this, verbose_level);

			f << "// orbit rep " << i << " has "
					<< gen_hdl.size() << " strong generators";
			if (gen_hdl.size()) {
				f << ", transversal lengths: ";
				//int_vec_print(f, O->tl, Poset->A->base_len());
				for (h = 0; h < tl.size(); h++) {
					f << tl[h];
				}
			}
			f << endl;
			f << "[" << endl;

			for (j = 0; j < gen_hdl.size(); j++) {

				Poset->A->element_retrieve(gen_hdl[j], Elt, 0);

				f << "[";
				//Gen->A->element_print_quick(Elt, f);
				for (ii = 0; ii < n; ii++) {
					f << "[";
					for (jj = 0; jj < n; jj++) {
						a = Elt[ii * n + jj];
						f << a;
						if (jj < n - 1) {
							f << ", ";
						}
						else {
							f << "]";
						}
					}
					if (ii < n - 1) {
						f << "," << endl;
					}
					else {
						f << "]";
					}
				}

				if (j < gen_hdl.size() - 1) {
					f << "," << endl;
				}
			}
			f << "]" << endl;
			if (i < len - 1) {
				f << "," << endl << endl;
			}
			else {
				f << endl << "];" << endl << endl;
			}
		} // next i

		f << endl << "T := [sub<GL(n, q) | [&cat (s): "
					 "s in S[i]]> : i in [1..#S]];"
					 << endl << endl;
	} // file f

	FREE_lint(the_set);
	FREE_int(v);
	FREE_int(Elt);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "poset_classification::wedge_product_export_magma "
				"done" << endl;
	}
}

void poset_classification::write_reps_csv(int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_reps_csv;
	char str[1000];

	if (f_v) {
		cout << "poset_classification::write_reps_csv" << endl;
	}
	fname_reps_csv.assign(problem_label_with_path);
	sprintf(str, "_reps_lvl_%d", lvl);
	fname_reps_csv.append(str);
	fname_reps_csv.append(".csv");

	Poo->save_representatives_at_level_to_csv(fname_reps_csv, lvl, verbose_level);

	if (f_v) {
		cout << "poset_classification::write_reps_csv done" << endl;
	}

}




}}}


