import os
import sys
import subprocess
import shutil


venv_dir = "venv"


def create_venv():
    """
    If the virtual python environment for this project does not exist yet, create it.
    """
    # Check if the virtual environment already exists
    if not os.path.exists(venv_dir):
        print(f"Virtual environment '{venv_dir}' not found. Creating one...")

        # Check if Python is installed
        python_executable = (
            shutil.which("python3.11")
            or shutil.which("python3")
            or shutil.which("python")
        )
        if not python_executable:
            sys.exit("No Python interpreter found.")

        # Verify the version of the Python executable
        python_version_output = subprocess.run(
            [python_executable, "--version"], capture_output=True, text=True
        )
        version = python_version_output.stdout.strip().split()[1]
        major, minor, _ = map(int, version.split("."))

        if major < 3 or (major == 3 and minor < 11):
            sys.exit("Python 3.11 or higher is required.")

        # Create the virtual environment
        subprocess.check_call([python_executable, "-m", "venv", venv_dir])

    print(f"Virtual environment '{venv_dir}' is ready\n")


def install_requirements():
    """
    Use pip to install all unsatisfied required packages into the virtual environment.
    """
    if os.name == "posix":
        pip_executable = os.path.join(venv_dir, "bin", "pip")
    else:
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")

    if not os.path.exists("requirements.txt"):
        sys.exit("No 'requirements.txt' file found in the project directory.")

    print("Installing requirements...")
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [pip_executable, "install", "-r", "requirements.txt"],
            stdout=devnull,
            stderr=devnull,
        )
    print("All requirements are now installed.\n")


def main():
    create_venv()
    install_requirements()


if __name__ == "__main__":
    main()
