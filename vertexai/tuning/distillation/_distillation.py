# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Classes for model tuning based on distillation."""

from typing import Optional

from google.cloud import aiplatform
from vertexai.language_models import _distillation as llm_distillation
from vertexai.language_models import _language_models as tuning


def distill_model(
    *,
    teacher_model: str,
    student_model: str,
    dataset: str,
    train_steps: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    evaluation_spec: Optional[tuning.TuningEvaluationSpec] = None,
    accelerator_type: Optional[tuning._ACCELERATOR_TYPE_TYPE] = None,
    model_display_name: Optional[str] = None,
) -> "_DistillationJob":
    pipeline_job = llm_distillation.submit_distillation_pipeline_job(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset=dataset,
        train_steps=train_steps,
        learning_rate_multiplier=learning_rate_multiplier,
        evaluation_spec=evaluation_spec,
        accelerator_type=accelerator_type,
        model_display_name=model_display_name,
    )
    distillation_job = _DistillationJob(pipeline_job=pipeline_job)
    return distillation_job


class _DistillationJob(tuning._TuningJob):

    def __init__(self, pipeline_job: aiplatform.PipelineJob):
        super().__init__(job=pipeline_job)

    def get_tuned_model(self) -> aiplatform.Model:
        """Blocks until the tuning is complete and returns an `aiplatform.Model` object."""
        vertex_model_name = self._get_tuned_vertex_model_name()
        return aiplatform.Model(model_name=vertex_model_name)
