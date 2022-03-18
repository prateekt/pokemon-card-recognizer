import subprocess
from typing import List, Optional, Tuple


def run_os_command(
    command: List[str], squelch_output: bool = False
) -> Tuple[Optional[str], int]:
    """
    Runs a safe operating system command.

    param command: The command to run
    squelch_output: Whether to squelch output

    return:
        trace: The command trace
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return_code = process.returncode
    if return_code != 0:
        raise SystemError(stderr)
    elif not squelch_output:
        if len(stdout) != 0:
            print(stdout)
        if len(stderr) != 0:
            print(stderr)
    if len(stdout) != 0 and len(stderr) != 0:
        return "\n".join([stdout, stderr]), return_code
    elif len(stdout) != 0:
        return stdout, return_code
    elif len(stderr) != 0:
        return stderr, return_code
    else:
        return None, return_code


def check_if_installed(program: str) -> bool:
    """
    Check if program is installed.

    param program: Name of program

    return:
        True if program is found on unix machine
    """
    results = run_os_command(command=["which", program])
    if len(results) == 0:
        return False
    elif "not found" in results:
        return False
    else:
        return True