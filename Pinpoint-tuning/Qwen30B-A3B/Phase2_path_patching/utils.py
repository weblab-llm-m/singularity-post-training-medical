#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Path Patching (Enhanced for Medical QA)

Based on: sycophancy-interpretability/path_patching/utils.py
Added: Medical QA specific visualization functions
"""

from typing import List, Dict

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import PreTrainedModel


def show_path_patching_results(
    m,
    xlabel="Head",
    ylabel="Layer",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig


def create_batch(sliced_data: List[Dict], split: str = "xr_toks", pad_token_id: int = None):

    tokens = [item[split] for item in sliced_data]
    max_length = max([len(item) for item in tokens])

    tokens_tensor = torch.LongTensor([[pad_token_id] * (max_length - len(item)) + item for item in tokens])
    mask_tensor = torch.LongTensor([[0] * (max_length - len(item)) + [1] * len(item) for item in tokens])

    return tokens_tensor, mask_tensor


@torch.no_grad()
def compute_metric(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    record_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the metric used to measure the path patching effect.

    Parameters
    ----------
    model : PreTrainedModel
        Model to compute the metric.
    input_ids : torch.Tensor
        Input ids.
    attention_mask : torch.Tensor
        Attention mask.
    target_token_ids : torch.Tensor
        Target token ids.
    record_token_ids : torch.Tensor
        Record token ids, used as regularization.

    Returns
    -------
    torch.Tensor
        Metric value.
    """

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.detach()

    batch_size = logits.size(0)

    logits_target = logits[torch.arange(batch_size), -1, target_token_ids]
    logits_sum = torch.zeros_like(logits_target).to(logits_target.device)

    for i in range(batch_size):
        logits_sum[i] = logits[i, -1, record_token_ids[i]].sum()

    return logits_target / logits_sum


# ========== 【追加】医療QA用の拡張機能 ==========


@torch.no_grad()
def compute_metric_medical(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
    record_token_ids: torch.Tensor,
    medical_term_positions: torch.Tensor = None,
    weight_medical_terms: bool = False
) -> torch.Tensor:
    """
    医療QA用のメトリクス計算

    オプション: 医療用語への注意が高い場合、メトリクスを調整
    （実験的機能、必要に応じて実装）

    Parameters
    ----------
    model : PreTrainedModel
        Model to compute the metric.
    input_ids : torch.Tensor
        Input ids.
    attention_mask : torch.Tensor
        Attention mask.
    target_token_ids : torch.Tensor
        Target token ids.
    record_token_ids : torch.Tensor
        Record token ids.
    medical_term_positions : torch.Tensor, optional
        Positions of medical terms in the input.
    weight_medical_terms : bool, optional
        Whether to weight medical terms differently.

    Returns
    -------
    torch.Tensor
        Metric value.
    """

    # 基本ロジック（既存と同じ）
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits.detach()

    batch_size = logits.size(0)
    logits_target = logits[torch.arange(batch_size), -1, target_token_ids]
    logits_sum = torch.zeros_like(logits_target).to(logits_target.device)

    for i in range(batch_size):
        logits_sum[i] = logits[i, -1, record_token_ids[i]].sum()

    # 【オプション】医療用語位置の重み付け（実験的）
    if weight_medical_terms and medical_term_positions is not None:
        # 医療用語への注意が高い場合、メトリクスを調整
        # （ここでは実装をスキップ、必要に応じて追加）
        pass

    return logits_target / logits_sum


def show_path_patching_results_with_classification(
    m,
    classification_results=None,
    xlabel="Head",
    ylabel="Layer",
    title="Path Patching Results with Head Classification",
    bartitle="Impact (%)",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap with head classification overlay

    Parameters
    ----------
    m : np.ndarray
        Path patching results matrix [num_layers, num_heads]
    classification_results : dict, optional
        Head classification results from Phase 3
        {
            'medical_term_heads': [(layer, head), ...],
            'guideline_indicator_heads': [(layer, head), ...],
            'reasoning_flow_heads': [(layer, head), ...]
        }
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    bartitle : str
        Colorbar title
    return_fig : bool
        Whether to return the figure
    show_fig : bool
        Whether to show the figure

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure (if return_fig=True)
    """

    # 基本のヒートマップ
    fig = px.imshow(
        m,
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        **kwargs,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    # 【追加】ヘッド分類結果を重ねる
    if classification_results is not None:
        # Medical Term Headsを赤丸でマーク
        medical_heads = classification_results.get('medical_term_heads', [])
        if medical_heads:
            medical_x = [h for l, h in medical_heads]
            medical_y = [l for l, h in medical_heads]

            fig.add_scatter(
                x=medical_x,
                y=medical_y,
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle-open', line=dict(width=2)),
                name='Medical Term Heads'
            )

        # Guideline Indicator Headsを青四角でマーク
        guideline_heads = classification_results.get('guideline_indicator_heads', [])
        if guideline_heads:
            guideline_x = [h for l, h in guideline_heads]
            guideline_y = [l for l, h in guideline_heads]

            fig.add_scatter(
                x=guideline_x,
                y=guideline_y,
                mode='markers',
                marker=dict(color='blue', size=10, symbol='square-open', line=dict(width=2)),
                name='Guideline Indicator Heads'
            )

        # Reasoning Flow Headsを緑三角でマーク
        reasoning_heads = classification_results.get('reasoning_flow_heads', [])
        if reasoning_heads:
            reasoning_x = [h for l, h in reasoning_heads]
            reasoning_y = [l for l, h in reasoning_heads]

            fig.add_scatter(
                x=reasoning_x,
                y=reasoning_y,
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up-open', line=dict(width=2)),
                name='Reasoning Flow Heads'
            )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=1.1, y=1.0)
    )

    fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)

    if show_fig:
        fig.show()
    if return_fig:
        return fig
