// flag_orbits.cpp
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace invariant_relations {

flag_orbits::flag_orbits()
{
	A = NULL;
	A2 = NULL;
	nb_primary_orbits_lower = 0;
	nb_primary_orbits_upper = 0;

	upper_bound_for_number_of_traces = 0;
	func_to_free_received_trace = NULL;
	func_latex_report_trace = NULL;
	free_received_trace_data = NULL;

	nb_flag_orbits = 0;
	Flag_orbit_node = NULL;
	pt_representation_sz = 0;
	Pt = NULL;
}

flag_orbits::~flag_orbits()
{
	freeself();
}

void flag_orbits::null()
{
}

void flag_orbits::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits::freeself" << endl;
	}
	if (Flag_orbit_node) {
		if (f_v) {
			cout << "flag_orbits::freeself before FREE_OBJECTS(Flag_orbit_node)" << endl;
		}
		FREE_OBJECTS(Flag_orbit_node);
	}
	if (Pt) {
		if (f_v) {
			cout << "flag_orbits::freeself before FREE_lint(Pt)" << endl;
		}
		FREE_lint(Pt);
	}
	if (f_v) {
		cout << "flag_orbits::freeself done" << endl;
	}
	null();
}

void flag_orbits::init(actions::action *A, actions::action *A2,
	int nb_primary_orbits_lower, 
	int pt_representation_sz, int nb_flag_orbits, 
	int upper_bound_for_number_of_traces,
	void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level),
	void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level),
	void *free_received_trace_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits::init" << endl;
	}
	flag_orbits::A = A;
	flag_orbits::A2 = A2;
	flag_orbits::nb_primary_orbits_lower = nb_primary_orbits_lower;
	flag_orbits::pt_representation_sz = pt_representation_sz;
	flag_orbits::nb_flag_orbits = nb_flag_orbits;
	flag_orbits::upper_bound_for_number_of_traces = upper_bound_for_number_of_traces;
	flag_orbits::func_to_free_received_trace = func_to_free_received_trace;
	flag_orbits::free_received_trace_data = free_received_trace_data;
	flag_orbits::func_latex_report_trace = func_latex_report_trace;

	Pt = NEW_lint(nb_flag_orbits * pt_representation_sz);
	Flag_orbit_node = NEW_OBJECTS(flag_orbit_node, nb_flag_orbits);
	if (f_v) {
		cout << "flag_orbits::init done" << endl;
	}
}

int flag_orbits::find_node_by_po_so(int po, int so, int &idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits::find_node_by_po_so" << endl;
	}
	int l, r, m, res;
	int f_found = FALSE;
	int len;

	len = nb_flag_orbits;
	if (f_v) {
		cout << "flag_orbits::find_node_by_po_so len=" << len << endl;
	}
	if (len == 0) {
		idx = 0;
		return FALSE;
	}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		if (f_v) {
			cout << "flag_orbits::find_node_by_po_so "
					"l=" << l << " r=" << r << endl;
		}
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle

		if (Flag_orbit_node[m].downstep_primary_orbit < po) {
			res = -1;
		}
		else if (Flag_orbit_node[m].downstep_primary_orbit > po) {
			res = 1;
		}
		else {
			if (Flag_orbit_node[m].downstep_secondary_orbit < so) {
				res = -1;
			}
			else if (Flag_orbit_node[m].downstep_secondary_orbit > so) {
				res = 1;
			}
			else {
				res = 0;
			}
		}
		//res = (*compare_func)(a, v[m], data_for_compare);
		if (f_v) {
			cout << "m=" << m << " res=" << res << endl;
			}
		//res = v[m] - a;
		//cout << "search l=" << l << " m=" << m << " r="
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			r = m;
			}
		}
	// now: l == r;
	// and f_found is set accordingly */
	if (f_found) {
		l--;
	}
	idx = l;
	return f_found;
}

void flag_orbits::write_file(ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "flag_orbits::write_file" << endl;
	}
	fp.write((char *) &nb_primary_orbits_lower, sizeof(int));
	fp.write((char *) &nb_primary_orbits_upper, sizeof(int));
	fp.write((char *) &nb_flag_orbits, sizeof(int));
	fp.write((char *) &pt_representation_sz, sizeof(int));

	if (f_vv) {
		cout << "flag_orbits::write_file Pt matrix:" << endl;
		Lint_matrix_print(Pt, nb_flag_orbits, pt_representation_sz);
	}
	for (i = 0; i < nb_flag_orbits * pt_representation_sz; i++) {
		fp.write((char *) &Pt[i], sizeof(long int));
	}
	if (f_v) {
		cout << "flag_orbits::write_file writing " << nb_flag_orbits << " nodes" << endl;
	}
	for (i = 0; i < nb_flag_orbits; i++) {
		Flag_orbit_node[i].write_file(fp, 0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "flag_orbits::write_file finished" << endl;
	}
}

void flag_orbits::read_file(ifstream &fp,
		actions::action *A, actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "flag_orbits::read_file" << endl;
	}
	flag_orbits::A = A;
	flag_orbits::A2 = A2;
	fp.read((char *) &nb_primary_orbits_lower, sizeof(int));
	fp.read((char *) &nb_primary_orbits_upper, sizeof(int));
	fp.read((char *) &nb_flag_orbits, sizeof(int));
	fp.read((char *) &pt_representation_sz, sizeof(int));

	Pt = NEW_lint(nb_flag_orbits * pt_representation_sz);
	for (i = 0; i < nb_flag_orbits * pt_representation_sz; i++) {
		fp.read((char *) &Pt[i], sizeof(long int));
	}
	if (f_vv) {
		cout << "flag_orbits::read_file Pt matrix:" << endl;
		Lint_matrix_print(Pt, nb_flag_orbits, pt_representation_sz);
	}

	if (f_v) {
		cout << "flag_orbits::read_file reading " << nb_flag_orbits << " nodes" << endl;
	}
	Flag_orbit_node = NEW_OBJECTS(flag_orbit_node, nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {
		if (f_vv) {
			cout << "flag_orbits::read_file "
					"node " << i << " / " << nb_flag_orbits << endl;
		}
		Flag_orbit_node[i].Flag_orbits = this;
		Flag_orbit_node[i].flag_orbit_index = i;
		Flag_orbit_node[i].read_file(fp, 0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "flag_orbits::read_file finished" << endl;
	}
}

void flag_orbits::print_latex(ostream &ost,
	std::string &title, int f_print_stabilizer_gens)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits::print_latex" << endl;
	}

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{" << title << "}" << endl;


	ost << "The number of primary orbits below is " << nb_primary_orbits_lower << "\\\\" << endl;
	ost << "The number of primary orbits above is " << nb_primary_orbits_upper << "\\\\" << endl;
	ost << "The number of flag orbits is " << nb_flag_orbits << "\\\\" << endl;

	int i;

	ost << "The flag orbits are:" << endl;
	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < nb_flag_orbits; i++) {
		flag_orbit_node *F;

		F = &Flag_orbit_node[i];
		ost << "\\item" << endl;
		F->print_latex(this, ost, f_print_stabilizer_gens);
	}
	ost << "\\end{enumerate}" << endl;

}


}}}


