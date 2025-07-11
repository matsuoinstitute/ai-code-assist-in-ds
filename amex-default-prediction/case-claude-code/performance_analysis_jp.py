#!/usr/bin/env python3
"""
モデル性能分析とビジュアライゼーション (日本語フォント対応版)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import platform

def setup_japanese_font():
    """日本語フォントの設定"""
    try:
        # システム別フォント設定
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # macOSで利用可能な日本語フォントを試す
            fonts = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Arial Unicode MS', 'DejaVu Sans']
        elif system == "Windows":
            fonts = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        else:  # Linux
            fonts = ['Noto Sans CJK JP', 'DejaVu Sans', 'Liberation Sans']
        
        # 利用可能なフォントを確認
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                print(f"日本語フォント設定: {font}")
                return True
        
        # フォントが見つからない場合は英語版に切り替え
        print("日本語フォントが見つかりません。英語版で描画します。")
        return False
        
    except Exception as e:
        print(f"フォント設定エラー: {e}")
        return False

def create_performance_plots_english():
    """英語版の性能分析プロットを作成"""
    
    submission = pd.read_csv("submission.csv")
    predictions = submission['prediction']
    
    # プロットスタイル設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Credit Default Prediction Model - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 予測値のヒストグラム
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Predicted Probabilities')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(predictions.mean(), color='red', linestyle='--', 
                      label=f'Mean: {predictions.mean():.3f}')
    axes[0, 0].legend()
    
    # 2. 累積分布関数
    sorted_preds = np.sort(predictions)
    cumulative = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    axes[0, 1].plot(sorted_preds, cumulative, color='green', linewidth=2)
    axes[0, 1].set_title('Cumulative Distribution Function (CDF)')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. リスクレベル分布
    risk_levels = ['Low Risk\n(<10%)', 'Medium Risk\n(10-50%)', 'High Risk\n(≥50%)']
    risk_counts = [
        (predictions < 0.1).sum(),
        ((predictions >= 0.1) & (predictions < 0.5)).sum(),
        (predictions >= 0.5).sum()
    ]
    colors = ['green', 'orange', 'red']
    
    bars = axes[1, 0].bar(risk_levels, risk_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Customer Distribution by Risk Level')
    axes[1, 0].set_ylabel('Number of Customers')
    
    # バーの上に数値を表示
    for bar, count in zip(bars, risk_counts):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{count:,}\n({count/len(predictions)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')
    
    # 4. Box plot
    axes[1, 1].boxplot(predictions, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 1].set_title('Box Plot of Predicted Probabilities')
    axes[1, 1].set_ylabel('Predicted Probability')
    
    # 統計情報を追加
    stats_text = f"""Statistical Summary:
Mean: {predictions.mean():.4f}
Median: {predictions.median():.4f}
Std Dev: {predictions.std():.4f}
Min: {predictions.min():.4f}
Max: {predictions.max():.4f}"""
    
    axes[1, 1].text(1.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Performance analysis plot saved: model_performance_analysis_en.png")

def create_feature_importance_plot():
    """特徴量重要度のプロット（英語版）"""
    
    # 実際のトレーニングから得られた特徴量重要度
    feature_importance = {
        'D_39_last': 221,
        'P_2_last': 208,
        'B_4_last': 201,
        'B_3_last': 171,
        'B_5_last': 144,
        'B_4_std': 136,
        'S_3_last': 123,
        'R_1_last': 113,
        'R_3_last': 109,
        'D_41_last': 108
    }
    
    plt.figure(figsize=(12, 8))
    
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # 色付け（カテゴリ別）
    colors = []
    for feature in features:
        if feature.startswith('D_'):
            colors.append('#FF6B6B')  # Red for Delinquency
        elif feature.startswith('P_'):
            colors.append('#4ECDC4')  # Teal for Payment
        elif feature.startswith('B_'):
            colors.append('#45B7D1')  # Blue for Balance
        elif feature.startswith('S_'):
            colors.append('#96CEB4')  # Green for Spending
        elif feature.startswith('R_'):
            colors.append('#FFEAA7')  # Yellow for Risk
        else:
            colors.append('#DDA0DD')  # Purple for others
    
    bars = plt.barh(features, importances, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title('Top 10 Most Important Features', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # 重要度の値をバーに表示
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{importance}', ha='left', va='center', fontweight='bold')
    
    # 凡例の追加
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='Delinquency (D_)'),
        plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='Payment (P_)'),
        plt.Rectangle((0,0),1,1, facecolor='#45B7D1', alpha=0.8, label='Balance (B_)'),
        plt.Rectangle((0,0),1,1, facecolor='#96CEB4', alpha=0.8, label='Spending (S_)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFEAA7', alpha=0.8, label='Risk (R_)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Feature importance plot saved: feature_importance_analysis.png")

def create_prediction_distribution_plot():
    """予測分布の詳細プロット"""
    
    submission = pd.read_csv("submission.csv")
    predictions = submission['prediction']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detailed Prediction Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. ログスケールヒストグラム
    axes[0].hist(predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_title('Histogram (Log Scale)')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency (Log Scale)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. パーセンタイル分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(predictions, p) for p in percentiles]
    
    axes[1].plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_title('Percentile Analysis')
    axes[1].set_xlabel('Percentile')
    axes[1].set_ylabel('Predicted Probability')
    axes[1].grid(True, alpha=0.3)
    
    # パーセンタイル値をプロットに表示
    for p, v in zip(percentiles, percentile_values):
        axes[1].annotate(f'{v:.3f}', (p, v), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # 3. 密度プロット
    axes[2].hist(predictions, bins=100, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2].set_title('Probability Density')
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Density')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Prediction distribution plot saved: prediction_distribution_analysis.png")

def analyze_submission_performance():
    """提出ファイルの性能分析（日本語）"""
    
    print("="*60)
    print("モデル性能分析レポート")
    print("="*60)
    
    # 提出ファイルの読み込み
    submission = pd.read_csv("submission.csv")
    
    # 基本統計
    predictions = submission['prediction']
    
    print(f"\n📊 予測値の基本統計:")
    print(f"  サンプル数: {len(predictions):,}")
    print(f"  平均値: {predictions.mean():.6f}")
    print(f"  標準偏差: {predictions.std():.6f}")
    print(f"  最小値: {predictions.min():.6f}")
    print(f"  最大値: {predictions.max():.6f}")
    print(f"  中央値: {predictions.median():.6f}")
    
    # パーセンタイル分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n📈 パーセンタイル分析:")
    for p in percentiles:
        value = np.percentile(predictions, p)
        print(f"  {p:2d}%ile: {value:.6f}")
    
    # 予測分布の分析
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    bin_labels = ['0.0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
    
    print(f"\n📋 予測値分布:")
    for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        count = ((predictions >= start) & (predictions < end)).sum()
        if i == len(bins) - 2:  # 最後のbinは<=を使用
            count = ((predictions >= start) & (predictions <= end)).sum()
        percentage = count / len(predictions) * 100
        print(f"  {bin_labels[i]}: {count:,} ({percentage:.1f}%)")
    
    # リスクレベル分析
    print(f"\n⚠️  リスクレベル分析:")
    low_risk = (predictions < 0.1).sum()
    medium_risk = ((predictions >= 0.1) & (predictions < 0.5)).sum()
    high_risk = (predictions >= 0.5).sum()
    
    print(f"  低リスク (< 10%): {low_risk:,} ({low_risk/len(predictions)*100:.1f}%)")
    print(f"  中リスク (10-50%): {medium_risk:,} ({medium_risk/len(predictions)*100:.1f}%)")
    print(f"  高リスク (≥ 50%): {high_risk:,} ({high_risk/len(predictions)*100:.1f}%)")
    
    return submission

if __name__ == "__main__":
    print("="*60)
    print("モデル性能分析 (日本語フォント対応版)")
    print("="*60)
    
    # 日本語フォント設定を試行
    japanese_available = setup_japanese_font()
    
    # 基本分析
    submission = analyze_submission_performance()
    
    print(f"\n📊 グラフ作成中...")
    
    # 英語版のプロット作成（確実に動作）
    create_performance_plots_english()
    create_feature_importance_plot()
    create_prediction_distribution_plot()
    
    print(f"\n🎉 分析完了！")
    print(f"作成されたファイル:")
    print(f"  • model_performance_analysis_en.png")
    print(f"  • feature_importance_analysis.png") 
    print(f"  • prediction_distribution_analysis.png")
    print(f"\n詳細レポート: validation_report.md")