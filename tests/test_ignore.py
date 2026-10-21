"""Host files stay off GitHub and out of the image build context.

Curated sample clips and docs/benchmarks/rtx3060-images.json stay available: the README links the clips, and the
image copies the measured profile.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GIT_LINES = (
    ".env", ".env.*", "!.env.example", "compose.nvml.yaml", "compose.override.yaml",
    "docker-compose.override.yml", "scouts/", "zig-engine/", "tf-h3-dit", "*.tf-int8",
    "*.safetensors", "*.ckpt", "*.gguf", "*.pt", "*.onnx", "*.bin", "/*.mp4", "/*.mov",
    "/*.wav", "/*.flac", "id_rsa", "id_ed25519", "*.pem", ".venv/", "venv/", "node_modules/",
    ".playwright-mcp/", "*.log", "studio-gpu.lock.d/",
)
DOCKER_LINES = (
    ".env", ".env.*", "!.env.example", "compose.nvml.yaml", "compose.override.yaml",
    "docker-compose.override.yml", "scouts", "zig-engine", "tf-h3-dit", "**/*.tf-int8",
    "**/*.safetensors", "**/*.ckpt", "**/*.gguf", "**/*.pt", "**/*.onnx", "**/*.bin",
    "**/*.mp4", "**/*.mov", "**/*.wav", "**/*.flac", "id_rsa", "id_ed25519", "*.pem",
    ".venv", "venv", "node_modules", ".playwright-mcp", "**/*.log", "studio-gpu.lock.d",
    "payload", "outputs", "samples",
)
IGNORED = (
    ".env", ".env.local", "compose.nvml.yaml", "compose.override.yaml", "scouts/x/sheet.jpg",
    "models/a.safetensors", "weights/model.ckpt", "zig-engine/tf-h3-dit", "tf-h3-dit",
    "adapter.safetensors.tf-int8", "out.mp4", "take.wav", ".venv/pyvenv.cfg", "id_ed25519",
    "studio-gpu.lock.d/lock", "payload/engine.bin",
)
KEPT = (
    ".env.example", "samples/singer_standard_8s.mp4", "samples/ui_library.png",
    "docs/benchmarks/rtx3060-images.json", "docs/benchmarks/rtx3060-z-image-turbo.json",
    "compose.yaml", "app/server.py", "spark/smi-pin/nvidia-smi",
)


def lines(name):
    return (ROOT / name).read_text(encoding="utf-8").splitlines()


def test_the_ignore_files_list_host_files():
    git, docker = lines(".gitignore"), lines(".dockerignore")
    assert [line for line in GIT_LINES if line not in git] == []
    assert [line for line in DOCKER_LINES if line not in docker] == []
    assert "docs/benchmarks" not in docker


def check_ignore(path):
    done = subprocess.run(["git", "check-ignore", "--no-index", "-q", "--", path], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode in (0, 1), done.stderr
    return done.returncode == 0


def test_git_ignores_host_files_and_keeps_curated_ones():
    assert [path for path in IGNORED if not check_ignore(path)] == []
    assert [path for path in KEPT if check_ignore(path)] == []


def test_documentation_does_not_name_a_lan_address():
    """Loopback and the Docker bridge are the only addresses in the deployment notes."""
    allowed = {"0.0.0.0", "127.0.0.1", "172.17.0.1"}
    address = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    names = ["MEMORY.md", ".env.example", "compose.yaml", "docs/BUILD_PLAN.md", "docs/OPERATIONS.md"]
    hits = []
    for name in names:
        for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            for found in address.findall(line):
                if found not in allowed:
                    hits.append(f"{name}:{number}: {found}")
    assert hits == []
    assert check_ignore("compose.override.yaml")
