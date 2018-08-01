// classification.C
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "orbiter.h"

classification::classification()
{
	null();
}

classification::~classification()
{
	freeself();
}

void classification::null()
{
	A = NULL;
	A2 = NULL;
	max_orbits = 0;
	nb_orbits = 0;
	Orbit = NULL;
	representation_sz = 0;
	Rep = NULL;
}

void classification::freeself()
{
	if (Orbit) {
		delete [] Orbit;
		}
	if (Rep) {
		FREE_INT(Rep);
		}
	null();
}

void classification::init(action *A, action *A2, 
	INT max_orbits, INT representation_sz, 
	longinteger_object &go, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification::init group order = " << go << " representation_sz = " << representation_sz << " max_orbits = " << max_orbits << endl;
		}
	classification::A = A;
	classification::A2 = A2;
	go.assign_to(classification::go);
	classification::max_orbits = max_orbits;
	classification::representation_sz = representation_sz;
	Orbit = new orbit_node[max_orbits];
	Rep = NEW_INT(max_orbits * representation_sz);
	if (f_v) {
		cout << "classification::init done" << endl;
		}
}

set_and_stabilizer *classification::get_set_and_stabilizer(INT orbit_index, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_and_stabilizer *SaS;
	INT *data;
	strong_generators *Strong_gens;

	if (f_v) {
		cout << "classification::get_set_and_stabilizer" << endl;
		}

	SaS = new set_and_stabilizer;

	data = NEW_INT(representation_sz);
	INT_vec_copy(Rep + orbit_index * representation_sz, data, representation_sz);
	
	Strong_gens = Orbit[orbit_index].gens->create_copy();

	SaS->init_everything(A, A2, data, representation_sz, 
		Strong_gens, 0 /* verbose_level */);

	if (f_v) {
		cout << "classification::get_set_and_stabilizer done" << endl;
		}

	return SaS;
}

void classification::print_latex(ostream &ost, 
	const BYTE *title, INT f_with_stabilizers)
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	
	cout << "classification::print_latex" << endl;
	
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{" << title << "}" << endl;



	{

	ost << "The order of the group is ";
	go.print_not_scientific(ost);
	ost << "\\\\" << endl;

	ost << "\\bigskip" << endl;
	}

	ost << "The group has " << nb_orbits << " orbits: \\\\" << endl;

	INT i;
	longinteger_domain D;
	longinteger_object go1, ol, Ol;
	Ol.create(0);

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
		
		ost << "$" << i << " / " << nb_orbits << "$ $" << endl;
		INT_set_print_tex_for_inline_text(ost, Rep + i * representation_sz, representation_sz);
		ost << "_{";
		go1.print_not_scientific(ost);
		ost << "}$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;
		if (f_with_stabilizers) {
			//ost << "Strong generators are:" << endl;
			Orbit[i].gens->print_generators_tex(ost);
			D.add_in_place(Ol, ol);
			}


		}

	ost << "The overall number of objects is: " << Ol << "\\\\" << endl;
}

void classification::write_file(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "classification::write_file" << endl;
		}
	fp.write((char *) &nb_orbits, sizeof(INT));
	fp.write((char *) &representation_sz, sizeof(INT));

	for (i = 0; i < nb_orbits * representation_sz; i++) {
		fp.write((char *) &Rep[i], sizeof(INT));
		}
	for (i = 0; i < nb_orbits; i++) {
		Orbit[i].write_file(fp, 0/*verbose_level*/);
		}

	if (f_v) {
		cout << "classification::write_file finished" << endl;
		}
}

void classification::read_file(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "classification::read_file" << endl;
		}
	fp.read((char *) &nb_orbits, sizeof(INT));
	fp.read((char *) &representation_sz, sizeof(INT));

	Rep = NEW_INT(nb_orbits * representation_sz);
	for (i = 0; i < nb_orbits * representation_sz; i++) {
		fp.read((char *) &Rep[i], sizeof(INT));
		}
	
	max_orbits = nb_orbits;
	Orbit = new orbit_node[nb_orbits];
	for (i = 0; i < nb_orbits; i++) {
		Orbit[i].C = this;
		Orbit[i].orbit_index = i;
		Orbit[i].read_file(fp, 0 /*verbose_level*/);
		}

	if (f_v) {
		cout << "classification::read_file finished" << endl;
		}
}


