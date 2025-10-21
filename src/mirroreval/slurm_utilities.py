import importlib.resources as pkg_resources
import subprocess
from pathlib import Path

from jinja2 import BaseLoader, Environment

from mirroreval import slurm_templates

from .config import settings


def render_slurm_script(script_name: str) -> str:
    # Load template
    template_content = pkg_resources.read_text(slurm_templates, "template.sh.j2")
    template = Environment(loader=BaseLoader()).from_string(template_content)

    context = settings.slurm_job.to_dict()
    context["script"] = get_script_path(script_name)

    print("SLURM context:", context)

    rendered_script = template.render(context)

    return rendered_script


def get_script_path(script_name: str) -> Path:
    with pkg_resources.as_file(pkg_resources.files("mirroreval") / script_name) as path:
        return path


def submit_slurm_job(rendered_slurm_script: str) -> None:
    result = subprocess.run(
        ["sbatch"], input=rendered_slurm_script, text=True, capture_output=True
    )
    print("SLURM submission result:", result.stdout)
    if result.returncode != 0:
        print("SLURM submission error:", result.stderr)
