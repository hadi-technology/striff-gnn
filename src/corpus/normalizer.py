"""Normalize clarpse output into a unified graph format for PyG.

The clarpse API returns:
{
  "model": {
    "components": {
      "com.example.MyClass": {
        "componentName": "MyClass",
        "type": "CLASS",
        "name": "MyClass",
        "comment": "...",
        "sourceFile": "...",
        "children": [...],
        "references": [
          {"type": "extension", "invokedComponent": "...", "external": false},
          {"type": "implementation", "invokedComponent": "...", "external": false},
          {"type": "simple", "invokedComponent": "...", "external": false}
        ],
        ...
      }
    }
  },
  "failures": [...],
  "durationMs": 123
}

This normalizer converts that into:
{
    "nodes": [{"id", "type", "name", "comment", "file", "synthetic"}],
    "edges": [{"src", "tgt", "type"}],
    "metrics": {"component_id": {"wmc", "dit", "noc", "ac", "ec", "encapsulation"}}
}

For Python/TypeScript, creates synthetic MODULE nodes that group top-level
functions and fields under a parent representing the source file.
"""

import json
from pathlib import Path
from typing import Optional
from collections import defaultdict


# Map clarpse reference types to edge types for the GNN
REF_TYPE_MAP = {
    "extension": "EXTENSION",
    "implementation": "IMPLEMENTATION",
    "simple": "ASSOCIATION",
}

# Types that indicate top-level items in Python/TS (not inside a class)
TOP_LEVEL_TYPES = {"FUNCTION", "MODULE_FIELD", "FIELD"}

# Languages that need synthetic module nodes
MODULE_LANGUAGES = {"python", "typescript"}


def normalize(parsed: dict, language: str = "java") -> dict:
    """Convert raw clarpse API response to a normalized graph dict.

    Handles the clarpse JSON format where components is a map keyed by
    unique name, and relations are embedded as references inside each
    component.

    For Python/TypeScript, creates synthetic MODULE nodes that group
    top-level functions and fields under a parent representing the source file.
    """
    if not parsed:
        return {"nodes": [], "edges": [], "metrics": {}}

    nodes = []
    edges = []
    metrics = {}

    # Extract components from clarpse response
    model = parsed.get("model", parsed)
    components_map = model.get("components", {})

    # Build node list and collect component data
    for unique_name, comp in components_map.items():
        if not isinstance(comp, dict):
            continue

        comp_type = comp.get("type", "OTHER")

        node = {
            "id": unique_name,
            "type": comp_type,
            "name": comp.get("name", comp.get("componentName", "")),
            "comment": (comp.get("comment") or "")[:200],
            "file": comp.get("sourceFile", ""),
            "synthetic": False,
        }
        nodes.append(node)

        # Compute basic metrics from component properties
        children = comp.get("children", [])
        refs = comp.get("references", [])
        metrics[unique_name] = {
            "wmc": float(comp.get("cyclo", 1)),
            "dit": 1.0,
            "noc": float(len(children)),
            "ac": float(sum(1 for r in refs if r.get("type") == "simple")),
            "ec": float(sum(1 for r in refs
                          if r.get("type") in ("extension", "implementation"))),
            "encapsulation": 0.8,
        }

    # Build edge list from component references
    seen_edges = set()
    for unique_name, comp in components_map.items():
        if not isinstance(comp, dict):
            continue
        refs = comp.get("references", [])
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            ref_type = ref.get("type", "simple")
            target = ref.get("invokedComponent", "")
            if not target:
                continue

            edge_type = REF_TYPE_MAP.get(ref_type, "ASSOCIATION")
            edge_key = (unique_name, target, edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append({
                "src": unique_name,
                "tgt": target,
                "type": edge_type,
            })

    # Create synthetic MODULE nodes for Python/TypeScript
    if language.lower() in MODULE_LANGUAGES:
        _add_synthetic_modules(nodes, edges, metrics)

    return {"nodes": nodes, "edges": edges, "metrics": metrics}


def _add_synthetic_modules(nodes: list, edges: list, metrics: dict) -> None:
    """Create synthetic MODULE nodes for Python/TS source files.

    For each source file that contains top-level items (functions, fields)
    not already inside a class, create a synthetic module parent and add
    COMPOSITION edges from the module to its children.

    Synthetic modules get aggregated metrics from their children.
    """
    node_ids = {n["id"] for n in nodes}

    # Find which nodes are children of existing CLASS nodes
    children_of_class = set()
    for edge in edges:
        if edge["type"] == "COMPOSITION":
            children_of_class.add(edge["tgt"])

    # Group top-level items by source file
    file_children = defaultdict(list)
    for node in nodes:
        if (node["type"] in TOP_LEVEL_TYPES
                and node["id"] not in children_of_class
                and node.get("file")):
            file_children[node["file"]].append(node)

    seen_edges = {(e["src"], e["tgt"], e["type"]) for e in edges}

    for source_file, children in file_children.items():
        if len(children) < 2:
            continue  # Skip files with only 0-1 top-level items

        module_id = f"module:{source_file}"

        # Skip if module already exists (shouldn't happen but be safe)
        if module_id in node_ids:
            continue

        # Create synthetic module node
        module_name = source_file.rsplit("/", 1)[-1].replace(".py", "").replace(".ts", "")
        module_node = {
            "id": module_id,
            "type": "CLASS",  # Typed as CLASS for one-hot compatibility
            "name": module_name,
            "comment": f"Synthetic module for {source_file}",
            "file": source_file,
            "synthetic": True,
        }
        nodes.append(module_node)
        node_ids.add(module_id)

        # Aggregate metrics from children
        child_metrics = [metrics.get(c["id"], {}) for c in children]
        metrics[module_id] = {
            "wmc": sum(m.get("wmc", 0) for m in child_metrics),
            "dit": 1.0,
            "noc": float(len(children)),
            "ac": sum(m.get("ac", 0) for m in child_metrics),
            "ec": sum(m.get("ec", 0) for m in child_metrics),
            "encapsulation": 0.5,  # Modules are less encapsulated than classes
        }

        # Add COMPOSITION edges from module to children
        for child in children:
            edge_key = (module_id, child["id"], "COMPOSITION")
            if edge_key not in seen_edges:
                edges.append({
                    "src": module_id,
                    "tgt": child["id"],
                    "type": "COMPOSITION",
                })
                seen_edges.add(edge_key)


def save_normalized(graph: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)


def load_normalized(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
