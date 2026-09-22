"""V2 project code package."""
"""Project package.

Expose the standard-library interactive-console symbols so tools such as
pytest/pdb remain compatible with this legacy package name.
"""
try:
    from stdlib_code_compat import InteractiveConsole, InteractiveInterpreter, compile_command  # type: ignore
except ImportError:
    import sysconfig
    from pathlib import Path
    _stdlib_code = Path(sysconfig.get_paths()["stdlib"]) / "code.py"
    _ns = {}
    exec(compile(_stdlib_code.read_text(encoding="utf-8"), str(_stdlib_code), "exec"), _ns)
    InteractiveConsole = _ns["InteractiveConsole"]
    InteractiveInterpreter = _ns["InteractiveInterpreter"]
    compile_command = _ns["compile_command"]
