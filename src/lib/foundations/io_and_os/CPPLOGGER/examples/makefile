CC=g++
CFLAGS=-std=c++17 -Wall -O3
INCLUDE_DIR=-I../
LIBS=-lpthread

all: \
	make_bin_dir\
	basic_usage\
	typographic_example \
	disable_logging\
	time_stamp \
	multi_thread\
	print_thread_id\
	log_type\
	file_name\
	line_number\
	no_file_line\
	verbosity\
	asynchronous_logging\
	log_level\
	print_array\
	scoped_logging\
	get_type
	
run_all: clean all
	@python3 ./run_all.py

make_bin_dir:
	mkdir -p ./bin

basic_usage: basic_usage.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@
	
typographic_example: typographic_example.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@

disable_logging: disable_logging.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@

time_stamp: time_stamp.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@
	
multi_thread: multi_thread.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)
	
print_thread_id: print_thread_id.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)
	
log_type: log_type.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@
	
file_name: file_name.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@
	
line_number: line_number.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@
	
no_file_line: no_file_line.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@

verbosity: verbosity.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@

asynchronous_logging: asynchronous_logging.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)

log_level: log_level.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)

print_array: print_array.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)
	
scoped_logging: scoped_logging.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)

get_type: get_type.cpp ../*.h
	$(CC) $(CFLAGS) $(INCLUDE_DIR) $< -o ./bin/$@ $(LIBS)

clean:
	rm -rf ./bin/
