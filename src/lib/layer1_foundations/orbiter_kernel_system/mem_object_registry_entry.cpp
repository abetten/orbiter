// mem_object_registry_entry.cpp
//
// Anton Betten
//
// started:  June 25, 2009




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {






mem_object_registry_entry::mem_object_registry_entry()
{
	null();
}

mem_object_registry_entry::~mem_object_registry_entry()
{

}

void mem_object_registry_entry::null()
{
	time_stamp = 0;
	pointer = NULL;
	object_type = POINTER_TYPE_INVALID;
	object_n = 0;
	object_size_of = 0;
	extra_type_info = NULL;
	source_file = NULL;
	source_line = 0;
}

void mem_object_registry_entry::set_type_from_string(char *str)
{
	if (strcmp(str, "int") == 0) {
		object_type = POINTER_TYPE_int;
	}
	else if (strcmp(str, "pint") == 0) {
		object_type = POINTER_TYPE_pint;
	}
	else if (strcmp(str, "lint") == 0) {
		object_type = POINTER_TYPE_lint;
	}
	else if (strcmp(str, "plint") == 0) {
		object_type = POINTER_TYPE_plint;
	}
	else if (strcmp(str, "ppint") == 0) {
		object_type = POINTER_TYPE_ppint;
	}
	else if (strcmp(str, "char") == 0) {
		object_type = POINTER_TYPE_char;
	}
	else if (strcmp(str, "uchar") == 0) {
		object_type = POINTER_TYPE_uchar;
	}
	else if (strcmp(str, "pchar") == 0) {
		object_type = POINTER_TYPE_pchar;
	}
	else if (strcmp(str, "puchar") == 0) {
		object_type = POINTER_TYPE_puchar;
	}
	else if (strcmp(str, "pvoid") == 0) {
		object_type = POINTER_TYPE_PVOID;
	}
	else if (strcmp(str, "OBJECT") == 0) {
		object_type = POINTER_TYPE_OBJECT;
	}
	else if (strcmp(str, "OBJECTS") == 0) {
		object_type = POINTER_TYPE_OBJECTS;
	}
	else {
		object_type = POINTER_TYPE_INVALID;
	}
}

void mem_object_registry_entry::print_type(std::ostream &ost)
{
	if (object_type == POINTER_TYPE_INVALID) {
		ost << "invalid entry";
		}
	else if (object_type == POINTER_TYPE_int) {
		ost << "int";
		}
	else if (object_type == POINTER_TYPE_pint) {
		ost << "pint";
		}
	else if (object_type == POINTER_TYPE_lint) {
		ost << "lint";
		}
	else if (object_type == POINTER_TYPE_plint) {
		ost << "plint";
		}
	else if (object_type == POINTER_TYPE_ppint) {
		ost << "ppint";
		}
	else if (object_type == POINTER_TYPE_char) {
		ost << "char";
		}
	else if (object_type == POINTER_TYPE_uchar) {
		ost << "uchar";
		}
	else if (object_type == POINTER_TYPE_pchar) {
		ost << "pchar";
		}
	else if (object_type == POINTER_TYPE_puchar) {
		ost << "puchar";
		}
	else if (object_type == POINTER_TYPE_PVOID) {
		ost << "pvoid";
		}
	else if (object_type == POINTER_TYPE_OBJECT) {
		ost << "OBJECT";
		}
	else if (object_type == POINTER_TYPE_OBJECTS) {
		ost << "OBJECTS";
		}
	else {
		ost << "unknown" << endl;
		}
}


int mem_object_registry_entry::size_of()
{
	if (object_type == POINTER_TYPE_INVALID) {
		cout << "mem_object_registry_entry::size_of invalid entry" << endl;
		exit(1);
		}
	else if (object_type == POINTER_TYPE_int) {
		return sizeof(int) * object_n;
		}
	else if (object_type == POINTER_TYPE_pint) {
		return sizeof(int *) * object_n;
		}
	else if (object_type == POINTER_TYPE_lint) {
		return sizeof(long int) * object_n;
		}
	else if (object_type == POINTER_TYPE_plint) {
		return sizeof(long int *) * object_n;
		}
	else if (object_type == POINTER_TYPE_ppint) {
		return sizeof(int **) * object_n;
		}
	else if (object_type == POINTER_TYPE_char) {
		return sizeof(char) * object_n;
		}
	else if (object_type == POINTER_TYPE_uchar) {
		return sizeof(uchar) * object_n;
		}
	else if (object_type == POINTER_TYPE_pchar) {
		return sizeof(char *) * object_n;
		}
	else if (object_type == POINTER_TYPE_puchar) {
		return sizeof(uchar *) * object_n;
		}
	else if (object_type == POINTER_TYPE_PVOID) {
		return sizeof(pvoid) * object_n;
		}
	else if (object_type == POINTER_TYPE_OBJECT) {
		return object_size_of;
		}
	else if (object_type == POINTER_TYPE_OBJECTS) {
		return object_n * object_size_of;
		}
	else {
		cout << "mem_object_registry_entry::size_of "
				"unknown object type " << object_type << endl;
		exit(1);
		}
}

void mem_object_registry_entry::print(int line)
{
	data_structures::algorithms Algo;

	cout << line << " : ";
	Algo.print_pointer_hex(cout, pointer);
	cout << " : " << time_stamp << " : ";

	print_type(cout);

	cout << " : "
		<< object_n << " : "
		<< object_size_of << " : "
		<< extra_type_info << " : "
		<< source_file << " : "
		<< source_line << endl;

}


void mem_object_registry_entry::print_csv(std::ostream &ost, int line)
{
	data_structures::algorithms Algo;

	ost << line << ",";
	Algo.print_pointer_hex(ost, pointer);
	ost << "," << time_stamp << ",";

	print_type(ost);

	ost << ","
		<< object_n << ","
		<< object_size_of << ","
		<< extra_type_info << ","
		<< source_file << ","
		<< source_line << endl;

}




}}}

