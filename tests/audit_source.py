"""Compare source ASTs with the independent baseline, accounting for documented changes."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENAMES = json.loads((ROOT / "tests/reference_names.json").read_text())


class ReferenceNames(ast.NodeTransformer):
    """Map only the documented source identifiers; tensor expressions stay intact."""
    def visit_Name(self, node):
        node.id = RENAMES["class_names"].get(node.id, node.id)
        return node

    def visit_ClassDef(self, node):
        node.name = RENAMES["class_names"].get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in RENAMES["environment_names"].items():
                # Whole env-key literals and the matching validation error prefix.
                if node.value == old or node.value.startswith(old + "="):
                    node.value = new + node.value[len(old):]
        return node


def definitions(path):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def dump(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads((ROOT / "tests/symbol_mapping.json").read_text())
    expected = json.loads((ROOT / "tests/reference_sha256.json").read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((args.baseline / name).read_bytes()).hexdigest() == digest, name
    helpers = {}
    for path in ["flexmec/models/encoders.py", "flexmec/engine/reporting.py"]:
        helpers.update(definitions(ROOT / path))

    class Inline(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in helpers:
                helper = copy.deepcopy(helpers[node.value.func.id])
                body = Normalize().visit(helper).body
                assert isinstance(body[-1], ast.Return)
                assert isinstance(body[-1].value, ast.Name) and isinstance(node.targets[0], ast.Name)
                assert body[-1].value.id == node.targets[0].id
                assert [a.arg for a in helper.args.args] == [a.id for a in node.value.args]
                return body[:-1]
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in helpers:
                helper = Normalize().visit(copy.deepcopy(helpers[node.value.func.id]))
                assert [a.arg for a in helper.args.args] == [a.id for a in node.value.args]
                return helper.body
            return self.generic_visit(node)

    checked = 0
    for row in mapping:
        old = ReferenceNames().visit(definitions(args.baseline / row["source"])[row["symbol"]])
        new = definitions(ROOT / row["destination"])[row.get("destination_symbol", row["symbol"])]
        new = Inline().visit(copy.deepcopy(new))
        if row["symbol"] == "_get_fail_log_path":
            # Explicit relocation-only path adjustment, tested separately.
            continue
        if row["symbol"] == "_apply_builtin_run_config":
            # Public selection is now fixed; all other built-in defaults match.
            for node in ast.walk(old):
                if isinstance(node, ast.Dict):
                    pairs = [(k, v) for k, v in zip(node.keys, node.values)
                             if not (isinstance(k, ast.Constant) and k.value == "CHECKPOINT_METRIC")]
                    node.keys, node.values = map(list, zip(*pairs))
        assert dump(old) == dump(new), row
        checked += 1
    old_tree = ReferenceNames().visit(ast.parse((args.baseline / "main.py").read_text()))
    old_body = next(n for n in old_tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__").body
    new_run = Inline().visit(copy.deepcopy(definitions(ROOT / "flexmec/engine/runner.py")["run"]))
    new_body = Normalize().visit(new_run).body
    def training_body(body):
        fold = next(n for n in body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "fold_run_idx")
        epoch = next(n for n in fold.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "e")
        result = copy.deepcopy(epoch)
        # Keep training, validation, loss reporting and scheduler; checkpoint
        # selection and post-training outputs are intentionally different.
        validation = next(n for n in result.body if isinstance(n, ast.With))
        stop = next(i for i, n in enumerate(validation.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "scheduler")
        validation.body = validation.body[:stop + 1]
        return result
    assert dump(training_body(old_body)) == dump(training_body(new_body)), "training/validation calculations changed"
    print(f"PASS: {checked} definitions plus training/validation/scheduler AST; checkpoint policy intentionally MCC-only")


if __name__ == "__main__":
    main()
