"""
Visualization Helpers
可視化の共通処理
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class VisualizationHelpers:
    """可視化ヘルパークラス"""

    @staticmethod
    def get_head_type_color(head_type: str) -> str:
        """
        ヘッドタイプごとの色定義

        Args:
            head_type: ヘッドタイプ

        Returns:
            color: 色
        """
        color_map = {
            'medical_term': 'red',
            'medical_term_heads': 'red',
            'guideline_indicator': 'blue',
            'guideline_indicator_heads': 'blue',
            'reasoning_flow': 'green',
            'reasoning_flow_heads': 'green',
            'unclassified': 'gray'
        }
        return color_map.get(head_type, 'gray')

    @staticmethod
    def get_head_type_marker(head_type: str) -> str:
        """
        ヘッドタイプごとのマーカー定義

        Args:
            head_type: ヘッドタイプ

        Returns:
            marker: マーカー
        """
        marker_map = {
            'medical_term': 'o',
            'medical_term_heads': 'o',
            'guideline_indicator': 's',
            'guideline_indicator_heads': 's',
            'reasoning_flow': '^',
            'reasoning_flow_heads': '^',
            'unclassified': '.'
        }
        return marker_map.get(head_type, '.')

    @staticmethod
    def create_legend(head_types: list) -> list:
        """
        凡例生成

        Args:
            head_types: ヘッドタイプのリスト

        Returns:
            patches: 凡例パッチのリスト
        """
        patches = []

        for head_type in head_types:
            color = VisualizationHelpers.get_head_type_color(head_type)
            label = head_type.replace('_heads', '').replace('_', ' ').title()
            patch = mpatches.Patch(color=color, label=label)
            patches.append(patch)

        return patches

    @staticmethod
    def setup_plot_style():
        """プロットスタイルを設定"""
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        plt.rcParams['figure.figsize'] = (12, 10)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    @staticmethod
    def save_figure(fig, output_path: str, dpi: int = 150):
        """
        図を保存

        Args:
            fig: matplotlib figure
            output_path: 出力パス
            dpi: 解像度
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
