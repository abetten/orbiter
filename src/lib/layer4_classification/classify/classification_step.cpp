// classification.cpp
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

classification_step::classification_step()
{
	A = NULL;
	A2 = NULL;
	max_orbits = 0;
	nb_orbits = 0;
	Orbit = NULL;
	representation_sz = 0;
	Rep = NULL;
}

classification_step::~classification_step()
{
	if (Orbit) {
		FREE_OBJECTS(Orbit);
	}
	if (Rep) {
		FREE_lint(Rep);
	}
}

void classification_step::init(
		actions::action *A,
		actions::action *A2,
	int max_orbits, int representation_sz, 
	ring_theory::longinteger_object &go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_step::init "
				"group order = " << go
				<< " representation_sz = " << representation_sz
				<< " max_orbits = " << max_orbits << endl;
	}
	classification_step::A = A;
	classification_step::A2 = A2;
	go.assign_to(classification_step::go);
	classification_step::max_orbits = max_orbits;
	classification_step::representation_sz = representation_sz;
	Orbit = NEW_OBJECTS(orbit_node, max_orbits);
	Rep = NEW_lint(max_orbits * representation_sz);
	if (f_v) {
		cout << "classification_step::init done" << endl;
	}
}

data_structures_groups::set_and_stabilizer
	*classification_step::get_set_and_stabilizer(
		int orbit_index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::set_and_stabilizer *SaS;
	long int *data;
	groups::strong_generators *Strong_gens;

	if (f_v) {
		cout << "classification_step::get_set_and_stabilizer" << endl;
	}

	SaS = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	data = NEW_lint(representation_sz);
	Lint_vec_copy(
			Rep_ith(orbit_index),
			data, representation_sz);
	
	if (f_v) {
		cout << "classification_step::get_set_and_stabilizer "
				"before Orbit[orbit_index].gens->create_copy" << endl;
	}

	Strong_gens = Orbit[orbit_index].gens->create_copy(verbose_level - 2);

	if (f_v) {
		cout << "classification_step::get_set_and_stabilizer "
				"before SaS->init_everything" << endl;
	}

	SaS->init_everything(
		A, A2, data, representation_sz,
		Strong_gens, 0 /* verbose_level */);

	if (f_v) {
		cout << "classification_step::get_set_and_stabilizer done" << endl;
	}

	return SaS;
}


void classification_step::write_file(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classification_step::write_file" << endl;
	}
	if (f_v) {
		cout << "classification_step::write_file nb_orbits=" << nb_orbits << endl;
		cout << "classification_step::write_file representation_sz=" << representation_sz << endl;
	}
	fp.write((char *) &nb_orbits, sizeof(int));
	fp.write((char *) &representation_sz, sizeof(int));


	if (f_vv) {
		cout << "classification_step::write_file Rep matrix:" << endl;
		Lint_matrix_print(Rep, nb_orbits, representation_sz);
	}

	for (i = 0; i < nb_orbits * representation_sz; i++) {
		fp.write((char *) &Rep[i], sizeof(long int));
	}

	if (f_v) {
		cout << "classification_step::write_file writing " << nb_orbits << " orbits" << endl;
	}
	for (i = 0; i < nb_orbits; i++) {
		Orbit[i].write_file(fp, 0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "classification_step::write_file finished" << endl;
	}
}

void classification_step::read_file(
		std::ifstream &fp,
		actions::action *A,
		actions::action *A2,
		ring_theory::longinteger_object &go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "classification_step::read_file" << endl;
	}
	classification_step::A = A;
	classification_step::A2 = A2;
	go.assign_to(classification_step::go);
	fp.read((char *) &nb_orbits, sizeof(int));
	fp.read((char *) &representation_sz, sizeof(int));

	if (f_v) {
		cout << "classification_step::read_file nb_orbits=" << nb_orbits << endl;
		cout << "classification_step::read_file representation_sz=" << representation_sz << endl;
	}

	Rep = NEW_lint(nb_orbits * representation_sz);
	for (i = 0; i < nb_orbits * representation_sz; i++) {
		fp.read((char *) &Rep[i], sizeof(long int));
	}
	
	if (f_vv) {
		cout << "classification_step::read_file Rep matrix:" << endl;
		Lint_matrix_print(Rep, nb_orbits, representation_sz);
	}

	max_orbits = nb_orbits;
	Orbit = NEW_OBJECTS(orbit_node, nb_orbits);
	if (f_v) {
		cout << "classification_step::read_file reading " << nb_orbits << " orbits" << endl;
	}
	for (i = 0; i < nb_orbits; i++) {
		Orbit[i].C = this;
		Orbit[i].orbit_index = i;
		Orbit[i].read_file(fp, 0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "classification_step::read_file finished" << endl;
	}
}

void classification_step::generate_source_code(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	string prefix;

	if (f_v) {
		cout << "classification_step::generate_source_code" << endl;
	}
	fname.assign(fname_base);
	fname.append(".cpp");

	prefix.assign(fname_base);

	{
		ofstream f(fname);

		generate_source_code(f, prefix, verbose_level - 2);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "classification_step::generate_source_code done" << endl;
	}
}


void classification_step::generate_source_code(
		std::ostream &ost, std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_index;

	if (f_v) {
		cout << "classification_step::generate_source_code" << endl;
	}

	ost << "static int " << prefix << "_nb_reps = "
			<< nb_orbits << ";" << endl;
	ost << "static int " << prefix << "_size = "
			<< representation_sz << ";" << endl;



	if (f_v) {
		cout << "classification_step::generate_source_code "
				"preparing reps" << endl;
	}


#if 1

		int i;

		ost << "static long int " << prefix << "_reps[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {


			if (false) {
				cout << "classification_step::generate_source_code orbit_index = " << orbit_index << endl;
			}

			ost << "\t";
			for (i = 0; i < representation_sz; i++) {
				ost << Rep_ith(orbit_index)[i];
				ost << ", ";
			}
			ost << "	// " << orbit_index << endl;


		}
		ost << "};" << endl;
#endif




	if (f_v) {
		cout << "classification_step::generate_source_code preparing stab_order" << endl;
	}
	ost << "// the stabilizer orders:" << endl;
	ost << "static const char *" << prefix << "_stab_order[] = {" << endl;
	for (orbit_index = 0;
			orbit_index < nb_orbits;
			orbit_index++) {

		ring_theory::longinteger_object ago;

		Orbit[orbit_index].gens->group_order(ago);

		ost << "\t\"";

		ago.print_not_scientific(ost);
		ost << "\"," << endl;

	}
	ost << "};" << endl;




	ost << "static int " << prefix << "_make_element_size = "
			<< A->make_element_size << ";" << endl;

	{
		int *stab_gens_first;
		int *stab_gens_len;
		int fst;

		stab_gens_first = NEW_int(nb_orbits);
		stab_gens_len = NEW_int(nb_orbits);
		fst = 0;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			stab_gens_first[orbit_index] = fst;
			stab_gens_len[orbit_index] =
					Orbit[orbit_index].gens->gens->len;
			//stab_gens_len[orbit_index] =
			//The_surface[iso_type]->stab_gens->gens->len;
			fst += stab_gens_len[orbit_index];
		}


		if (f_v) {
			cout << "classification_step::generate_source_code preparing stab_gens_fst" << endl;
		}
		ost << "static int " << prefix << "_stab_gens_fst[] = { " << endl << "\t";
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			ost << stab_gens_first[orbit_index];
			if (orbit_index < nb_orbits - 1) {
				ost << ", ";
			}
			if (((orbit_index + 1) % 10) == 0) {
				ost << endl << "\t";
			}
		}
		ost << "};" << endl;

		if (f_v) {
			cout << "classification_step::generate_source_code preparing stab_gens_len" << endl;
		}
		ost << "static int " << prefix << "_stab_gens_len[] = { " << endl << "\t";
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			ost << stab_gens_len[orbit_index];
			if (orbit_index < nb_orbits - 1) {
				ost << ", ";
			}
			if (((orbit_index + 1) % 10) == 0) {
				ost << endl << "\t";
			}
		}
		ost << "};" << endl;


		if (f_v) {
			cout << "classification_step::generate_source_code preparing stab_gens" << endl;
		}
		ost << "static int " << prefix << "_stab_gens[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_orbits;
				orbit_index++) {
			int j;

			for (j = 0; j < stab_gens_len[orbit_index]; j++) {
				if (false) {
					cout << "classification_step::generate_source_code "
							"before extract_strong_generators_in_order generator " << j << " / "
							<< stab_gens_len[orbit_index] << endl;
				}
				ost << "\t";
				A->Group_element->element_print_for_make_element(
						Orbit[orbit_index].gens->gens->ith(j), ost);
				//A->element_print_for_make_element(
				//The_surface[iso_type]->stab_gens->gens->ith(j), f);
				if (j == 0) {
					ost << " // begin stabilizer generators orbit " << orbit_index;
				}
				ost << endl;
			}
		}
		ost << "};" << endl;


		FREE_int(stab_gens_first);
		FREE_int(stab_gens_len);
	}

	if (f_v) {
		cout << "classification_step::generate_source_code done" << endl;
	}
}


long int *classification_step::Rep_ith(int i)
{
	return Rep + i * representation_sz;
}

#if 0
long int *classification_step::Rep_lint_ith(int i)
{
	return Rep_lint + i * representation_sz;
}
#endif


void classification_step::print_group_orders()
{
	int i;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go1, ol;

	cout << "i : stab order : orbit length" << endl;
	for (i = 0; i < nb_orbits; i++) {
		Orbit[i].gens->group_order(go1);

		D.integral_division_exact(go, go1, ol);


		cout << i << " : " << go1 << " : " << ol << endl;

	}
}

void classification_step::print_summary(std::ostream &ost)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "classification_step::print_summary" << endl;
	}


	ost << "The order of the group is ";
	go.print_not_scientific(ost);
	ost << "\\\\" << endl;

	ost << "\\bigskip" << endl;

	ost << "The group has " << nb_orbits << " orbits. \\\\" << endl;

}


void classification_step::print_latex(std::ostream &ost,
	std::string &title,
	int f_print_stabilizer_gens,
	int f_has_print_function,
	void (*print_function)(std::ostream &ost, int i,
			classification_step *Step, void *print_function_data),
	void *print_function_data)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "classification_step::print_latex" << endl;
	}

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{" << title << "}" << endl;



	{

		ost << "The order of the group is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}

	ost << "The group has " << nb_orbits << " orbits: \\\\" << endl;

	int i;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go1, ol, Ol;
	Ol.create(0, __FILE__, __LINE__);

	ost << "The orbits are:" << endl;
	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < nb_orbits; i++) {

		if (f_v) {
			cout << "orbit " << i << " / " << nb_orbits << ":" << endl;
		}

		Orbit[i].gens->group_order(go1);

		if (f_v) {
			cout << "stab order " << go1 << endl;
		}

		D.integral_division_exact(go, go1, ol);

		if (f_v) {
			cout << "orbit length " << ol << endl;
		}

		ost << "\\item" << endl;
		ost << "$" << i << " / " << nb_orbits << "$ " << endl;


		if (f_has_print_function) {
			(*print_function)(ost, i, this, print_function_data);
		}


		ost << "$" << endl;

		L.lint_set_print_tex_for_inline_text(ost,
				Rep_ith(i),
				representation_sz);

		ost << "_{";
		go1.print_not_scientific(ost);
		ost << "}$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		if (f_print_stabilizer_gens) {
			//ost << "Strong generators are:" << endl;
			Orbit[i].gens->print_generators_tex(ost);
		}




		D.add_in_place(Ol, ol);


	}
	ost << "\\end{enumerate}" << endl;

	ost << "The overall number of objects is: " << Ol << "\\\\" << endl;

	if (f_v) {
		cout << "classification_step::print_latex done" << endl;
	}

}



}}}


