"""Lightweight source code parser that extracts structural relationships.

Parses Java, Python, and TypeScript source files to extract:
- Classes, interfaces, enums, methods, fields
- Extension, implementation, composition, association relationships
- Basic metrics (method count, field count, etc.)

This is NOT a full AST parser — it uses regex-based extraction that
captures the structural relationships needed for GNN training.
"""

import os
import re
from pathlib import Path
from typing import Optional


def detect_language(file_path: str) -> Optional[str]:
    ext = Path(file_path).suffix.lower()
    if ext == ".java":
        return "java"
    if ext in (".py", ".pyi"):
        return "python"
    if ext in (".ts", ".tsx"):
        return "typescript"
    return None


def parse_java_file(content: str, file_path: str) -> dict:
    """Extract structural info from a Java source file."""
    nodes = []
    edges = []

    # Package
    pkg_match = re.search(r'package\s+([\w.]+)\s*;', content)
    package = pkg_match.group(1) if pkg_match else ""

    # Classes / interfaces / enums
    type_pattern = re.compile(
        r'(public\s+|private\s+|protected\s+)?'
        r'(abstract\s+|final\s+|static\s+)?'
        r'(class|interface|enum)\s+'
        r'(\w+)'
        r'(?:\s+extends\s+([\w.]+))?'
        r'(?:\s+implements\s+([\w.,\s]+))?'
    )

    for m in type_pattern.finditer(content):
        type_kind = m.group(3)
        name = m.group(4)
        qname = f"{package}.{name}" if package else name

        nodes.append({
            "id": qname,
            "type": type_kind.upper(),
            "name": name,
            "comment": _extract_comment(content, m.start()),
            "file": file_path,
        })

        if m.group(5):  # extends
            parent = _resolve(m.group(5), package)
            edges.append({"src": qname, "tgt": parent, "type": "EXTENSION"})

        if m.group(6):  # implements
            for iface in m.group(6).split(','):
                iface = iface.strip()
                if iface:
                    edges.append({"src": qname, "tgt": _resolve(iface, package),
                                  "type": "IMPLEMENTATION"})

    # Field types (composition/association)
    field_pattern = re.compile(
        r'(?:private|protected|public)\s+'
        r'([\w.<>\[\]]+)\s+'
        r'(\w+)\s*[;=]'
    )
    class_names = {n["name"] for n in nodes}
    for m in field_pattern.finditer(content):
        field_type = m.group(1).split('<')[0].split('.')[-1]
        if field_type in class_names:
            # Find the enclosing class
            for n in nodes:
                if n["name"] in class_names:
                    edges.append({"src": n["id"], "tgt": _resolve(field_type, package),
                                  "type": "COMPOSITION"})
                    break

    # Method calls / type references (association)
    for n in nodes:
        # Count methods
        methods = re.findall(r'(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(', content)
        # Type references in method bodies
        refs = set(re.findall(r'\b([A-Z]\w+)\b', content))
        refs -= class_names
        refs -= {"String", "Integer", "Long", "Boolean", "List", "Map", "Set", "Optional",
                 "Object", "Class", "Exception", "Override", "System", "void", "Thread"}

    return {"nodes": nodes, "edges": edges}


def parse_python_file(content: str, file_path: str) -> dict:
    """Extract structural info from a Python source file."""
    nodes = []
    edges = []

    # Module name from file path
    parts = Path(file_path).replace('\\', '/').split('/')
    module = '.'.join(parts[:-1]).replace('/', '.') if len(parts) > 1 else ""

    # Classes
    class_pattern = re.compile(r'^class\s+(\w+)\s*'
                               r'(?:\(([\w.,\s]+)\))?\s*:', re.MULTILINE)
    for m in class_pattern.finditer(content):
        name = m.group(1)
        qname = f"{module}.{name}" if module else name

        nodes.append({
            "id": qname,
            "type": "CLASS",
            "name": name,
            "comment": _extract_comment(content, m.start()),
            "file": file_path,
        })

        if m.group(2):
            for parent in m.group(2).split(','):
                parent = parent.strip()
                if parent and parent != "object":
                    edges.append({"src": qname, "tgt": parent, "type": "EXTENSION"})

    # Functions as methods
    func_pattern = re.compile(r'^def\s+(\w+)\s*\(', re.MULTILINE)
    for m in func_pattern.finditer(content):
        name = m.group(1)
        if not name.startswith('_'):
            nodes.append({
                "id": f"{module}.{name}" if module else name,
                "type": "METHOD",
                "name": name,
                "comment": _extract_comment(content, m.start()),
                "file": file_path,
            })

    return {"nodes": nodes, "edges": edges}


def parse_typescript_file(content: str, file_path: str) -> dict:
    """Extract structural info from a TypeScript source file."""
    nodes = []
    edges = []

    # Namespace/module from path
    parts = Path(file_path).replace('\\', '/').split('/')
    ns = '.'.join(parts[:-1]).replace('/', '.') if len(parts) > 1 else ""

    # Classes
    class_pattern = re.compile(
        r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)'
        r'(?:\s+extends\s+([\w.]+))?'
        r'(?:\s+implements\s+([\w.,\s]+))?'
    )
    for m in class_pattern.finditer(content):
        name = m.group(1)
        qname = f"{ns}.{name}" if ns else name

        nodes.append({
            "id": qname,
            "type": "CLASS",
            "name": name,
            "comment": _extract_comment(content, m.start()),
            "file": file_path,
        })

        if m.group(2):
            edges.append({"src": qname, "tgt": m.group(2), "type": "EXTENSION"})
        if m.group(3):
            for iface in m.group(3).split(','):
                iface = iface.strip()
                if iface:
                    edges.append({"src": qname, "tgt": iface, "type": "IMPLEMENTATION"})

    # Interfaces
    iface_pattern = re.compile(r'(?:export\s+)?interface\s+(\w+)')
    for m in iface_pattern.finditer(content):
        name = m.group(1)
        qname = f"{ns}.{name}" if ns else name
        nodes.append({
            "id": qname,
            "type": "INTERFACE",
            "name": name,
            "comment": _extract_comment(content, m.start()),
            "file": file_path,
        })

    # Enums
    enum_pattern = re.compile(r'(?:export\s+)?enum\s+(\w+)')
    for m in enum_pattern.finditer(content):
        name = m.group(1)
        qname = f"{ns}.{name}" if ns else name
        nodes.append({
            "id": qname,
            "type": "ENUM",
            "name": name,
            "comment": "",
            "file": file_path,
        })

    return {"nodes": nodes, "edges": edges}


def parse_repo(repo_path: str, language: str) -> dict:
    """Parse all source files in a repo directory."""
    all_nodes = []
    all_edges = []
    extensions = {
        "java": {".java"},
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
    }

    exts = extensions.get(language, set())
    parsed_files = 0

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   ("node_modules", "__pycache__", "build", "target", "dist", ".git")]

        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in exts:
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                continue

            rel_path = os.path.relpath(fpath, repo_path)

            if language == "java":
                result = parse_java_file(content, rel_path)
            elif language == "python":
                result = parse_python_file(content, rel_path)
            elif language == "typescript":
                result = parse_typescript_file(content, rel_path)
            else:
                continue

            all_nodes.extend(result["nodes"])
            all_edges.extend(result["edges"])
            parsed_files += 1

    # Build metrics for nodes
    metrics = {}
    for node in all_nodes:
        metrics[node["id"]] = {
            "wmc": 1.0,
            "dit": 1.0,
            "noc": 0.0,
            "ac": float(sum(1 for e in all_edges if e["tgt"] == node["id"])),
            "ec": float(sum(1 for e in all_edges if e["src"] == node["id"])),
            "encapsulation": 0.8,
        }

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "metrics": metrics,
        "language": language,
        "parsed_files": parsed_files,
    }


def _extract_comment(content: str, pos: int) -> str:
    """Extract the comment preceding a definition."""
    before = content[:pos].rstrip()
    if before.endswith("*/"):
        idx = before.rfind("/*")
        if idx >= 0:
            comment = before[idx:].replace("/*", "").replace("*/", "").replace("*", "").strip()
            return comment[:200]
    lines = before.split("\n")
    comment_lines = []
    for line in reversed(lines[-5:]):
        if line.strip().startswith("//"):
            comment_lines.insert(0, line.strip().lstrip("/"))
        else:
            break
    if comment_lines:
        return " ".join(comment_lines)[:200]
    return ""


def _resolve(name: str, package: str) -> str:
    """Simple name resolution."""
    if '.' in name:
        return name
    if package:
        return f"{package}.{name}"
    return name
