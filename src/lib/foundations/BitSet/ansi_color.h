/*----------------------------------------------------------------------------*/
/// Define some ANSI color codes
/*----------------------------------------------------------------------------*/
#if __cplusplus >= 201103L
#define COLOR_UNICODE "\u001b"
#else
#define COLOR_UNICODE "\x1b"
#endif

#define RESET_COLOR_SCHEME COLOR_UNICODE "[0m"
#define BLACK COLOR_UNICODE "[30m"
#define RED COLOR_UNICODE "[31m"
#define GREEN COLOR_UNICODE "[32m"
#define YELLOW COLOR_UNICODE "[33m"
#define BLUE COLOR_UNICODE "[34m"
#define MAGENTA COLOR_UNICODE "[35m"
#define CYAN COLOR_UNICODE "[36m"
#define WHITE COLOR_UNICODE "[37m"

#define BRIGHT_BLACK COLOR_UNICODE "[30;1m"
#define BRIGHT_RED COLOR_UNICODE "[31;1m"
#define BRIGHT_GREEN COLOR_UNICODE "[32;1m"
#define BRIGHT_YELLOW COLOR_UNICODE "[33;1m"
#define BRIGHT_BLUE COLOR_UNICODE "[34;1m"
#define BRIGHT_MAGENTA COLOR_UNICODE "[35;1m"
#define BRIGHT_CYAN COLOR_UNICODE "[36;1m"
#define BRIGHT_WHITE COLOR_UNICODE "[37;1m"
/*----------------------------------------------------------------------------*/