# Orbiter GUI — Dash Code Workbench

A lightweight **desktop GUI** for the [Orbiter combinatorial classification system](https://github.com/abetten/orbiter), built with Python/Tkinter. It launches your existing Orbiter executable as a subprocess; it does not modify or embed the C++ library.

## Requirements

- Python 3.9+ with Tkinter (`python3 -m tkinter` to check)
- A separately built Orbiter executable, as described by Orbiter's README (`make` at the repository root; inspect your build's output location)
- macOS, Linux, or Windows with Python/Tkinter. On Windows, the Orbiter project recommends WSL for building Orbiter; run the GUI in the same environment or use a compatible executable.

## Start

```bash
python3 orbiter_gui.py
```

1. Select the **Orbiter executable** (usually the built `orbiter.out` or equivalent).
2. Select a **working directory** for output files.
3. Choose an example or enter your own Dash Code arguments (omit executable and `-v`).
4. Click **Run Orbiter** to see the combined stdout/stderr log. **Stop** sends termination to the launched process/group.
5. Save a `.json` project, `.txt` command, or `.log` transcript.

## Examples and caveats

The GUI ships with illustrative command templates based on the uploaded Orbiter source's README and documented command-language patterns. Exact command acceptance depends on your build and command parser. The GUI does **not** validate mathematical syntax before execution; Orbiter itself handles validation. Commands are executed using `subprocess.Popen(argv, shell=False)` so they are not interpreted by a shell. Avoid opening untrusted project files without inspecting their command text.

## Implementation notes

- GUI is responsive while Orbiter runs; worker thread streams stdout/stderr to a queue.
- The subprocess runs with the selected working directory.
- A JSON project stores executable, working directory, verbosity, and Dash Code.
- No Orbiter source files are changed.

## Future extensions

Structured editors for finite fields, group actions, designs, graphs, and poset classification; syntax-aware command completion based on the actual parser; output artifact browser; searchable API docs; integration tests against a built Orbiter binary.
