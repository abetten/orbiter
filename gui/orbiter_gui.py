#!/usr/bin/env python3
"""Orbiter GUI: a Tkinter workbench for the Orbiter Dash Code CLI.

Requires Python 3.9+ and Tkinter. Runs an existing Orbiter executable.
No shell is invoked; command arguments are passed as an argv list.
"""
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_NAME = 'Orbiter GUI — Dash Code Workbench'
EXAMPLES = {
    'Finite field GF(4)': '-define F -finite_field -q 4 -end',
    'Symmetric group S5': '-define G -permutation_group -symmetric_group 5 -end',
    'Projective linear group PGL(4,2)': (
        '-define F -finite_field -q 2 -end\n'
        '-define G -linear_group -PGL 4 F -end'
    ),
    'Custom command (blank)': '',
}

class OrbiterWorkbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry('1120x780')
        self.minsize(800, 540)
        self.exe = tk.StringVar(value=os.environ.get('ORBITER_EXE', ''))
        self.cwd = tk.StringVar(value=str(Path.cwd()))
        self.verbosity = tk.StringVar(value='2')
        self.example = tk.StringVar(value='Finite field GF(4)')
        self.status = tk.StringVar(value='Ready — choose the Orbiter executable.')
        self.command = None
        self.events = queue.Queue()
        self._build()
        self.after(100, self._drain_events)
        self.protocol('WM_DELETE_WINDOW', self._close)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        bar = ttk.Frame(self, padding=12)
        bar.grid(row=0, column=0, sticky='ew')
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text='Orbiter executable').grid(row=0, column=0, sticky='w')
        ttk.Entry(bar, textvariable=self.exe).grid(row=0, column=1, sticky='ew', padx=8)
        ttk.Button(bar, text='Browse…', command=self._choose_exe).grid(row=0, column=2)
        ttk.Label(bar, text='Working directory').grid(row=1, column=0, sticky='w', pady=(9, 0))
        ttk.Entry(bar, textvariable=self.cwd).grid(row=1, column=1, sticky='ew', padx=8, pady=(9, 0))
        ttk.Button(bar, text='Browse…', command=self._choose_cwd).grid(row=1, column=2, pady=(9, 0))

        pane = ttk.Panedwindow(self, orient=tk.VERTICAL)
        pane.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 6))
        top = ttk.Frame(pane, padding=4)
        top.columnconfigure(0, weight=1)
        top.rowconfigure(2, weight=1)
        pane.add(top, weight=2)
        tools = ttk.Frame(top)
        tools.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(tools, text='Example').pack(side=tk.LEFT)
        choices = ttk.Combobox(tools, textvariable=self.example, values=list(EXAMPLES), state='readonly', width=34)
        choices.pack(side=tk.LEFT, padx=6)
        ttk.Button(tools, text='Insert example', command=self._insert_example).pack(side=tk.LEFT, padx=3)
        ttk.Label(tools, text='Verbosity (-v)').pack(side=tk.LEFT, padx=(16, 5))
        ttk.Combobox(tools, textvariable=self.verbosity, values=[str(i) for i in range(9)], state='readonly', width=3).pack(side=tk.LEFT)
        ttk.Button(tools, text='Open command…', command=self._open_command).pack(side=tk.RIGHT, padx=3)
        ttk.Button(tools, text='Save command…', command=self._save_command).pack(side=tk.RIGHT, padx=3)
        ttk.Label(top, text='Dash Code arguments (omit executable and -v):').grid(row=1, column=0, sticky='w')
        self.editor = ScrolledText(top, wrap='word', undo=True, font=('Courier', 12), height=10)
        self.editor.grid(row=2, column=0, sticky='nsew', pady=(5, 0))
        self.editor.insert('1.0', EXAMPLES[self.example.get()])

        bottom = ttk.Frame(pane, padding=4)
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)
        pane.add(bottom, weight=3)
        buttons = ttk.Frame(bottom)
        buttons.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        self.run_button = ttk.Button(buttons, text='▶ Run Orbiter', command=self._run)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(buttons, text='■ Stop', command=self._stop, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=7)
        ttk.Button(buttons, text='Clear log', command=lambda: self.log.delete('1.0', tk.END)).pack(side=tk.LEFT)
        ttk.Button(buttons, text='Save log…', command=self._save_log).pack(side=tk.RIGHT)
        ttk.Label(bottom, text='Execution output').grid(row=1, column=0, sticky='nw')
        self.log = ScrolledText(bottom, wrap='none', font=('Courier', 10), state='normal')
        self.log.grid(row=1, column=0, sticky='nsew', pady=(22, 0))
        self.log.configure(state='disabled')
        ttk.Label(self, textvariable=self.status, anchor='w', padding=(12, 5)).grid(row=2, column=0, sticky='ew')
        self._menu()

    def _menu(self):
        m = tk.Menu(self)
        file_menu = tk.Menu(m, tearoff=False)
        file_menu.add_command(label='Open project…', command=self._open_project)
        file_menu.add_command(label='Save project…', command=self._save_project)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self._close)
        m.add_cascade(label='Project', menu=file_menu)
        help_menu = tk.Menu(m, tearoff=False)
        help_menu.add_command(label='About', command=lambda: messagebox.showinfo('About',
            'Orbiter GUI — a Python/Tkinter front end for the Orbiter C++ CLI.\n\n'
            'Source: https://github.com/abetten/orbiter\n'
            'This workbench does not include or modify Orbiter.'))
        m.add_cascade(label='Help', menu=help_menu)
        self.config(menu=m)

    def _choose_exe(self):
        path = filedialog.askopenfilename(title='Select Orbiter executable')
        if path: self.exe.set(path)

    def _choose_cwd(self):
        path = filedialog.askdirectory(title='Select working directory')
        if path: self.cwd.set(path)

    def _insert_example(self):
        self.editor.delete('1.0', tk.END)
        self.editor.insert('1.0', EXAMPLES[self.example.get()])

    def _command_text(self):
        return self.editor.get('1.0', 'end-1c').strip()

    def _open_command(self):
        path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt'), ('All files', '*')])
        if path:
            self.editor.delete('1.0', tk.END)
            self.editor.insert('1.0', Path(path).read_text(encoding='utf-8'))

    def _save_command(self):
        path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files', '*.txt')])
        if path: Path(path).write_text(self._command_text() + '\n', encoding='utf-8')

    def _save_log(self):
        path = filedialog.asksaveasfilename(defaultextension='.log', filetypes=[('Log files', '*.log'), ('Text files', '*.txt')])
        if path: Path(path).write_text(self.log.get('1.0', 'end-1c'), encoding='utf-8')

    def _save_project(self):
        path = filedialog.asksaveasfilename(defaultextension='.orbiter-gui.json', filetypes=[('Orbiter GUI project', '*.json')])
        if path:
            payload = {'executable': self.exe.get(), 'working_directory': self.cwd.get(),
                       'verbosity': self.verbosity.get(), 'dash_code': self._command_text()}
            Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    def _open_project(self):
        path = filedialog.askopenfilename(filetypes=[('Orbiter GUI project', '*.json')])
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            self.exe.set(payload.get('executable', ''))
            self.cwd.set(payload.get('working_directory', str(Path.cwd())))
            self.verbosity.set(str(payload.get('verbosity', '2')))
            self.editor.delete('1.0', tk.END)
            self.editor.insert('1.0', payload.get('dash_code', ''))
        except (OSError, ValueError) as err:
            messagebox.showerror('Cannot open project', str(err))

    def _append(self, message):
        self.log.configure(state='normal')
        self.log.insert(tk.END, message)
        self.log.see(tk.END)
        self.log.configure(state='disabled')

    def _run(self):
        exe = Path(self.exe.get().strip()).expanduser()
        cwd = Path(self.cwd.get().strip()).expanduser()
        if not exe.is_file():
            messagebox.showerror('Executable not found', 'Choose a valid Orbiter executable file.')
            return
        if not cwd.is_dir():
            messagebox.showerror('Working directory not found', 'Choose an existing working directory.')
            return
        if not os.access(exe, os.X_OK):
            messagebox.showerror('Not executable', 'Selected file is not executable.')
            return
        try:
            args = shlex.split(self._command_text(), posix=True)
        except ValueError as err:
            messagebox.showerror('Invalid command', str(err))
            return
        if not args:
            messagebox.showerror('No command', 'Enter Dash Code arguments to execute.')
            return
        argv = [str(exe.resolve()), '-v', self.verbosity.get()] + args
        self._append('\n' + '='*75 + '\n' + 'Working directory: ' + str(cwd.resolve()) +
                     '\nCommand: ' + shlex.join(argv) + '\n' + '='*75 + '\n')
        self.run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status.set('Running Orbiter…')
        threading.Thread(target=self._worker, args=(argv, str(cwd.resolve())), daemon=True).start()

    def _worker(self, argv, cwd):
        try:
            options = {'cwd': cwd, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT,
                       'text': True, 'bufsize': 1, 'errors': 'replace'}
            if os.name != 'nt': options['start_new_session'] = True
            self.command = subprocess.Popen(argv, **options)
            for line in self.command.stdout:
                self.events.put(('output', line))
            code = self.command.wait()
            self.events.put(('finished', code))
        except Exception as err:
            self.events.put(('error', str(err)))
        finally:
            self.command = None

    def _stop(self):
        proc = self.command
        if proc and proc.poll() is None:
            self.status.set('Stopping Orbiter…')
            try:
                if os.name != 'nt': os.killpg(proc.pid, signal.SIGTERM)
                else: proc.terminate()
            except ProcessLookupError:
                pass
            except OSError as err:
                self._append('\nStop error: ' + str(err) + '\n')

    def _drain_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'output': self._append(data)
                elif kind == 'finished':
                    self._append(f'\n[Orbiter exited with code {data}]\n')
                    self.status.set(f'Finished (exit code {data}).')
                    self.run_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                elif kind == 'error':
                    self._append('\n[Execution error] ' + data + '\n')
                    self.status.set('Execution error.')
                    self.run_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _close(self):
        if self.command and self.command.poll() is None:
            if not messagebox.askyesno('Quit?', 'Orbiter is running. Stop it and quit?'):
                return
            self._stop()
        self.destroy()

if __name__ == '__main__':
    OrbiterWorkbench().mainloop()
