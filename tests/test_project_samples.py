from __future__ import annotations

from pathlib import Path

from agent_loader_bench.corpus import load_skill_document
from agent_loader_bench.runner import load_dataset


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SKILLS = {
    "skills/vllm-benchmark/SKILL.md": "skill.vllm.benchmark",
    "skills/docs-writing/SKILL.md": "skill.docs.writing",
    "skills/troubleshooting/SKILL.md": "skill.troubleshooting",
}

RECOMMENDED_HEADINGS = {
    "When to use",
    "Inputs",
    "Steps",
    "Output",
    "Safety and Constraints",
    "Examples",
}


def test_sample_skills_include_required_frontmatter_and_headings() -> None:
    for relative_path, expected_id in EXPECTED_SKILLS.items():
        document = load_skill_document(ROOT / relative_path)

        assert document.id == expected_id
        assert document.type == "skill"
        assert document.version == "0.1.0"
        assert document.status == "draft"
        assert document.tags
        assert document.activation_keywords
        assert document.priority > 0
        assert RECOMMENDED_HEADINGS <= {section.title for section in document.sections}


def test_sample_dataset_ids_cover_required_request_categories() -> None:
    items = load_dataset(ROOT / "datasets" / "requests.yml")
    ids = {item.id for item in items}

    assert len(ids) == len(items)
    assert {
        "req.vllm.benchmark.exact",
        "req.vllm.benchmark.semantic",
        "req.loader.ambiguous",
        "req.docs.troubleshooting.multi_skill",
        "req.vllm.wrong_skill_trap",
        "req.unrelated.no_match",
        "req.troubleshooting.local_only_conflict",
    } <= ids

    expected_skill_ids = set(EXPECTED_SKILLS.values())
    for item in items:
        assert set(item.expected_skills) <= expected_skill_ids
