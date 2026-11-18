from fetch_engineering_docs import convert, document_key, output_name

PAGE = """---
title: Deployments
description: >-
  A Deployment manages a set of Pods.
content_type: concept
---

<!-- overview -->

A _Deployment_ updates {{< glossary_tooltip text="Pods" term_id="pod" >}} via a
{{< glossary_tooltip term_id="replica-set" >}}.

{{< note >}}
Do not manage ReplicaSets yourself.
{{< /note >}}

## Creating a Deployment

{{< feature-state for_k8s_version="v1.21" state="stable" >}}
{{< feature-state feature_gate_name="JobManagedBy" >}}

{{% code_sample file="controllers/nginx-deployment.yaml" %}}

```shell
kubectl get pods -o go-template='{{.metadata.name}}'
```
"""


def test_front_matter_is_returned_apart_and_its_description_opens_the_text() -> None:
    front, text = convert(PAGE)

    assert front["title"] == "Deployments"
    assert front["content_type"] == "concept"
    assert text.startswith("A Deployment manages a set of Pods.\n\nA _Deployment_")
    assert "---" not in text


def test_shortcodes_become_the_text_they_stand_for() -> None:
    _, text = convert(PAGE)

    assert "updates Pods via a\nreplica set." in text
    assert "**Note:** \nDo not manage ReplicaSets yourself." in text
    assert "FEATURE STATE: Kubernetes v1.21 [stable]" in text
    assert "Feature gate: `JobManagedBy`" in text
    assert "(Example: `controllers/nginx-deployment.yaml`)" in text
    assert "{{<" not in text and "{{%" not in text
    assert "<!--" not in text


def test_go_templates_in_code_are_left_alone() -> None:
    _, text = convert(PAGE)

    assert "go-template='{{.metadata.name}}'" in text


def test_headings_survive_for_structure_detection() -> None:
    _, text = convert(PAGE)

    assert "\n## Creating a Deployment\n" in text


def test_conversion_is_deterministic() -> None:
    assert convert(PAGE) == convert(PAGE.replace("\n", "\r\n"))


def test_names_and_keys_come_from_the_page_path() -> None:
    page = "workloads/controllers/replica_set.md"

    assert output_name(page) == "workloads--controllers--replica_set.md"
    assert document_key(page) == "workloads-controllers-replica-set"
