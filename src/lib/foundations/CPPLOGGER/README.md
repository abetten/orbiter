# CPPLOGGER

This is a simple, lightweight and thread safe logger for C++. 


## Platforms
* Linux
* MacOS (`get_type` function might not work with clang compilers)
	
	
## Features
* Header only, no need for separate compilation and linking
* ANSI unicode support for:
	* For printing logging information in eight colors
		* Black
		* Red
		* Green
		* Blue
		* Yellow
		* Cyan
		* Purple/ Magenta
		* White
	* For **bold**, underline and *italicizing* text, or a combination of any or all three
	* Provides support for three levels of logging: `[INFO]`, `[ERROR]`, `[WARN]`
	* Flag for enabling or disabling the printing of:
		* Time stamps
		* Threads IDs for logging at thread level for Multi/Single threaded applications
		* Logging types
		* File name and Line numbers
* Provides a flag to disable global logging and also logging in a single translation unit.
* Enabling/Disabling logging does not require flags to be passed in at compile time
* Provides verbose level option to limit how much logging information gets printed. Refer to [this](#Verbose-Level) example.
	
	
## Usage examples
#### Basic usage
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info("-------------------------------------------------------------");
	logger_info("Welcome to CPPLOGGER!");
	logger_info("-------------------------------------------------------------");

	logger_info("The following shows some basic usage.");

	logger_error("This is an error message with some %s:", "argument");
	logger_warning("This is a warning message with 2 args: (%.3f, %.4f)", 1.2323745, 2.5);
	logger_info("This is an info message with three arguments: (%03d, %03d, %03d)", 0, 50, 100);

}
```

#### Typographic examples with colors
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info(logger_red("This is a red text"));
	logger_info(
			logger_red(
					logger_italic("This is a red italic text.")
			)
	);
	logger_info(logger_bold(logger_red("This is a red bold text")));
	logger_info(logger_red(logger_bold(logger_italic("This is a red bold italic text"))));


	logger_info("%s: %d",
			logger_bold(logger_red("This is a red bold text with argument")),
			12);

	logger_info("%s: %s",
			logger_bold(logger_red("This is a red bold text with green argument")),
			logger_green("12"));

	logger_info("%s: %s",
			logger_bold(logger_red("This is a red bold text with green bold argument")),
			logger_bold(logger_green("12")));

	logger_info(logger_underline("%s: %s"),
			logger_bold(logger_red("This is a red bold text with green bold argument")),
			logger_bold(logger_green("12")));
}
```

#### Disabling global logging example
```c++
#include "../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This is before disabling global logging.");

	// This disables logging in the current translation unit.
	// To disable global logging use: logger_enable_global(false)
	logger_enable(false);
	logger_info("This should not be printed.");

	logger_enable(true);
	logger_info("This is after enabling global logging.");

}
```

#### Printing logging time stamps
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This is printed without a time stamp.");

	logger_print_timestamps(true);
	logger_info("This is printed with millisecond time stamp resolution.");

	logger_timestamp_resolution(logger_microsecond);
	logger_info("This is printed with microsecond time stamp resolution.");

	logger_timestamp_resolution(logger_nanosecond);
	logger_info("This is printed with nanosecond time stamp resolution.");

}
```

#### Logging with multiple threads
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

void print_thread(uint id) {
	logger_info_mt("%s: %03d", logger_red("Message from thread"), id);
	logger_info_mt("Message from thread: %03d", id);
}

int main() {
	const uint n_threads = 200;
	std::thread threads[n_threads];
	for (uint8_t i=0; i<n_threads; ++i)
		threads[i] = std::thread(print_thread, i+1);
	for (std::thread& t : threads) t.join();
}
```

####  Logging with printing thread IDs
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

void print_thread(uint id) {
	logger_print_thread_id(true);
	logger_info_mt("%s: %03d", logger_red("Message from thread"), id);
	logger_info_mt("Message from thread: %03d", id);
}

int main() {
	const uint n_threads = 200;
	std::thread threads[n_threads];
	for (uint8_t i=0; i<n_threads; ++i)
		threads[i] = std::thread(print_thread, i+1);
	for (std::thread& t : threads) t.join();
}
```

#### Logging without printing the log type
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This message is printed with log type enabled.");

	logger_print_log_type(false);
	logger_info("This message is printed without the log type.");

}
```

#### Logging without printing the file name
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This message is printed with the filename.");

	logger_print_file(false);
	logger_info("This message is printed without printing the filename.");

}
```

#### Logging without printing the line number
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This message is printed with the line number.");

	logger_print_line(false);
	logger_info("This message is printed without the line number.");

}
```

#### Logging without printing either the filename or the line number
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	logger_info(logger_bold("-------------------------------------------------------------"));
	logger_info(logger_bold("Welcome to CPPLOGGER!"));
	logger_info(logger_bold("-------------------------------------------------------------"));

	logger_info("This message has both line number and file");

	logger_print_file(false);
	logger_print_line(false);
	logger_info("This message does not have either a file or a line number.");

}
```

#### Verbose Level 
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {
	int verbose_level = 0;

	logger_info(verbose_level, 1, "This should only be printed at verbose level 1.");

	logger_info("Current verbose level: %d", verbose_level);
	logger_info("Incrementing verbose level by one: %d", ++verbose_level);
	logger_info("Current verbose level: %d", verbose_level);

	logger_info(verbose_level, 1, "This should only be printed at verbose level 1.");
}
```

#### Asynchronous logging
```c++
#include <CPPLOGGER_ASYNC.h>

logger_init();

int main() {
	logger_info("This message is not asynchronously logged.");
	logger_async_info("This message is asynchronously logged");

	logger_error("This error message is not asynchronously logged.");
	logger_async_error("This error message is asynchronously logged");

	logger_warning("This warning message is not asynchronously logged.");
	logger_async_warning("This warning message is asynchronously logged");
}
```

#### Setting the log level
```c++
#include <CPPLOGGER_SYNC.h>

logger_init();

int main() {

	logger_log_level(logger_loglevel_warning);
	logger_info("This message should not be printed.");
	logger_error("This message should not be printed");
	logger_warning("This warning message should be printed");

	logger_log_level(logger_loglevel_error);
	logger_info("This info message should not be printed.");
	logger_error("This error message should be printed");
	logger_warning("This warning message should not be printed");

	logger_log_level(logger_loglevel_info);
	logger_info("This info message should be printed.");
	logger_error("This error message should not be printed");
	logger_warning("This warning message should not be printed");

	logger_log_level(logger_loglevel_all);
	logger_info("This info message should be printed.");
	logger_error("This error message should be printed");
	logger_warning("This warning message should be printed");

}
```

#### Get type of a variable
```c++
#include <CPPLOGGER_SYNC.h>

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

class sample_class {
public:
	sample_class() {}
	void print() {
		logger_info("print method in class: %s", logger_red(logger::get_type(this)));
	}
};

int main(int argc, char** argv) {

	sample_class c;
	c.print();
	logger_info("c: %s", logger::get_type(c));


	int a = 10;
	logger_info("type of a: %s", logger::get_type(a));
}
```

## Benchmarks
Benchmarking done on Ubuntu 20.04 64 bit, Intel® Core™ i7-8750H CPU @ 2.20GHz × 12  

Throughput is the number of messages printed per second. Iterations is the number of log messages.

#### Benchmark result synchronous
In synchronous mode the logging results are immediately displayed on the output accounting for OS buffer time.
```
[INFO]: *******************************************************************
[INFO]: Single threaded benchmark with 5,000,000 iterations
[INFO]: *******************************************************************
[INFO]: basic           | Elapsed: 0.63 secs | Throughput: 7,878,563/sec
[INFO]: disabled        | Elapsed: 0.02 secs | Throughput: 288,832,788/sec
[INFO]: 
[INFO]: *******************************************************************
[INFO]: 100 thread benchmark with 5,000,000 iterations
[INFO]: *******************************************************************
[INFO]: basic           | Elapsed: 5.53 secs | Throughput: 904,463/sec
[INFO]: disabled        | Elapsed: 0.01 secs | Throughput: 482,712,240/sec
```

#### Benchmark result asynchronous
In asynchronous mode the logging results are not immediately displayed on the output. The logging thread creates a log request and then proceeds to do other work. These logs are eventually displayed on the output when the separate logger thread gets to them or when the program is terminated. If, however the logger queue is full, the threads that want to log a message gets halted while the queue is emptied out. By default, the queue size is 8192 lines.
```
[INFO]: *******************************************************************
[INFO]: Single threaded benchmark with 5,000,000 iterations
[INFO]: *******************************************************************
[INFO]: basic           | Elapsed: 1.90 secs | Throughput: 2,628,297/sec
[INFO]: disabled        | Elapsed: 0.02 secs | Throughput: 267,387,085/sec
[INFO]: 
[INFO]: *******************************************************************
[INFO]: 100 thread benchmark with 5,000,000 iterations
[INFO]: *******************************************************************
[INFO]: basic           | Elapsed: 3.00 secs | Throughput: 1,667,926/sec
[INFO]: disabled        | Elapsed: 0.01 secs | Throughput: 656,571,055/sec
```
