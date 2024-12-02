/*
 * pugixml_interface.cpp
 *
 *  Created on: Feb 19, 2024
 *      Author: betten
 */



#include "foundations.h"

#include <sstream>

#include <pugixml.hpp>

using namespace std;
using namespace pugi;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace l1_interfaces {


pugixml_interface::pugixml_interface()
{
	Record_birth();

}

pugixml_interface::~pugixml_interface()
{
	Record_death();

}

void pugixml_interface::read_file(
		std::string &fname,
		std::vector<std::vector<std::string> > &Classes_parsed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "pugixml_interface::read_file fname = " << fname << endl;
	}

	xml_document doc;

	doc.load_file(fname.c_str());

	//auto root = doc.root();
	//root.print(std::cout); // prints the entire thing

	pugi::xml_node C = doc.child("doxygenindex");


	vector<string> Classes;

		// tag::code[]
		for (pugi::xml_node c: C.children("compound")) {
			//std::cout << "compound:";


#if 0
		        for (pugi::xml_attribute attr: c.attributes())
		        {
		            std::cout << " " << attr.name() << "=" << attr.value();
		        }

		        for (pugi::xml_node child: c.children())
		        {
		            std::cout << ", child=" << child.name() << ", value=" << child.first_child().value();
		        }

		        std::cout << std::endl;
#endif

#if 1
			for (pugi::xml_attribute attr: c.attributes()) {
				//std::cout << " " << attr.name() << "=" << attr.value();
				if (strcmp(attr.name(), "kind") == 0 && strcmp(attr.value(), "class") == 0) {
					cout << "CLASS:" << endl;
					for (pugi::xml_node child: c.children()) {
						if (strcmp(child.name(), "name") == 0) {
							cout << child.first_child().value() << endl;
							string s;
							s = child.first_child().value();
							Classes.push_back(s);
						}
					}
				}
			}
#endif


	    }

		cout << "Orbiter has " << Classes.size() << " classes" << endl;


#if 0
    auto rotary = doc.root();
    // rotary.print(std::cout); // prints the entire thing

    auto name = rotary
        .select_single_node("//UserInformation/Name/text()")
        .node();
    auto age =  rotary
        .select_single_node("//UserInformation/Age/text()")
        .node();

    std::cout << "\nName is " << name.value() << "\n";
    std::cout << "Age is " << age.text().as_double() << "\n";
#endif

    int i;


    for (i = 0; i < Classes.size(); i++) {
    	string val;
    	string delim;
    	vector<string> entry;

    	val = Classes[i];
		delim = "::";
		std::size_t loc_delim = val.find(delim);
		while (loc_delim != std::string::npos) {
			string s1, s2;
			s1 = val.substr(0, loc_delim);
			s2 = val.substr(loc_delim + 2);
			entry.push_back(s1);
			val = s2;
			loc_delim = val.find(delim);
		}
		entry.push_back(val);
		Classes_parsed.push_back(entry);
    }

	if (f_v) {
		cout << "pugixml_interface::read_file done" << endl;
	}
}




}}}}




