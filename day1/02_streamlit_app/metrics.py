# metrics.py
import streamlit as st
import nltk
from janome.tokenizer import Tokenizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter

# NLTKのヘルパー関数（エラー時フォールバック付き）
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)  # 感情分析用
    from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    print("NLTK loaded successfully.") # デバッグ用
except Exception as e:
    st.warning(f"NLTKの初期化中にエラーが発生しました: {e}\n簡易的な代替関数を使用します。")
    def nltk_word_tokenize(text):
        return text.split()
    def nltk_sentence_bleu(references, candidate):
        # 簡易BLEUスコア（完全一致/部分一致）
        ref_words = set(references[0])
        cand_words = set(candidate)
        common_words = ref_words.intersection(cand_words)
        precision = len(common_words) / len(cand_words) if cand_words else 0
        recall = len(common_words) / len(ref_words) if ref_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1 # F1スコアを返す（簡易的な代替）

def initialize_nltk():
    """NLTKのデータダウンロードを試みる関数"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)  # 感情分析用
        print("NLTK data checked/downloaded.") # デバッグ用
    except Exception as e:
        st.error(f"NLTKデータのダウンロードに失敗しました: {e}")

def calculate_metrics(answer, correct_answer):
    """回答と正解から評価指標を計算する"""
    # 基本指標の初期化
    word_count = 0
    bleu_score = 0.0
    similarity_score = 0.0
    relevance_score = 0.0
    
    # 新規追加の指標
    sentiment_score = 0.5  # ニュートラル(0.5)をデフォルト値に
    readability_score = 0.0
    diversity_score = 0.0
    conciseness_score = 0.0
    
    if not answer:  # 回答がない場合は計算しない
        return bleu_score, similarity_score, word_count, relevance_score, sentiment_score, readability_score, diversity_score, conciseness_score

    # 単語数のカウント
    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(answer))
    word_count = len(tokens)

    # 感情分析スコア（SentimentIntensityAnalyzer）
    try:
        answer_lower = answer.lower()
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(answer_lower)
        # compoundスコアを0-1スケールに変換 (-1〜1 → 0〜1)
        sentiment_score = (sentiment['compound'] + 1) / 2
    except Exception:
        # SIAが失敗した場合のシンプルな感情分析（ポジティブ/ネガティブ単語の出現頻度）
        positive_words = ['良い', '素晴らしい', '優れた', '簡単', '効果的', '役立つ']
        negative_words = ['悪い', '難しい', '複雑', '問題', '困難', '欠点']
        
        pos_count = sum(1 for word in positive_words if word in answer)
        neg_count = sum(1 for word in negative_words if word in answer)
        
        total = pos_count + neg_count
        if total > 0:
            sentiment_score = pos_count / total
        else:
            sentiment_score = 0.5  # ニュートラル
    
    # 読みやすさスコア
    try:
        # 日本語向けに調整（文字数と文の数から簡易的に計算）
        sentences = re.split(r'[。.!?]', answer)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) > 0:
            avg_sentence_length = len(answer) / len(sentences)
            # 理想的な文の長さを30-50文字と仮定
            readability_score = max(0, min(1, 2 - abs(avg_sentence_length - 40) / 40))
        else:
            readability_score = 0
    except Exception:
        readability_score = 0
    
    # 語彙の多様性スコア（異なり語数 / 総語数）
    try:
        if tokens:
            unique_tokens = set(token.surface for token in tokens)
            diversity_score = len(unique_tokens) / len(tokens)
        else:
            diversity_score = 0
    except Exception:
        diversity_score = 0
    
    # 簡潔性スコア（文字数に基づく - 短すぎず長すぎない最適な長さを想定）
    try:
        char_length = len(answer)
        # 最適な長さを150-300文字と仮定
        if char_length < 50:  # 短すぎる
            conciseness_score = char_length / 50
        elif char_length <= 300:  # 適切な長さ
            conciseness_score = 1.0
        else:  # 長すぎる
            conciseness_score = max(0, 1 - (char_length - 300) / 700)  # 1000文字でスコア0になる
    except Exception:
        conciseness_score = 0
    
    # 正解がある場合のみ、BLEUと類似度を計算
    if correct_answer:
        answer_lower = answer.lower()
        correct_answer_lower = correct_answer.lower()

        # BLEU スコアの計算
        try:
            reference = [nltk_word_tokenize(correct_answer_lower)]
            candidate = nltk_word_tokenize(answer_lower)
            # ゼロ除算エラーを防ぐ
            if candidate:
                bleu_score = nltk_sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) # 4-gram BLEU
            else:
                bleu_score = 0.0
        except Exception:
            bleu_score = 0.0 # エラー時は0

        # コサイン類似度の計算
        try:
            vectorizer = TfidfVectorizer()
            # fit_transformはリストを期待するため、リストで渡す
            if answer_lower.strip() and correct_answer_lower.strip(): # 空文字列でないことを確認
                tfidf_matrix = vectorizer.fit_transform([answer_lower, correct_answer_lower])
                similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            else:
                similarity_score = 0.0
        except Exception:
            similarity_score = 0.0 # エラー時は0

        # 関連性スコア（キーワードの一致率などで簡易的に計算）
        try:
            answer_words = set(re.findall(r'\w+', answer_lower))
            correct_words = set(re.findall(r'\w+', correct_answer_lower))
            if len(correct_words) > 0:
                common_words = answer_words.intersection(correct_words)
                relevance_score = len(common_words) / len(correct_words)
            else:
                relevance_score = 0.0
        except Exception:
            relevance_score = 0.0 # エラー時は0

    return bleu_score, similarity_score, word_count, relevance_score, sentiment_score, readability_score, diversity_score, conciseness_score

def get_metrics_descriptions():
    """評価指標の説明を返す"""
    return {
        "正確性スコア (is_correct)": "回答の正確さを3段階で評価: 1.0 (正確), 0.5 (部分的に正確), 0.0 (不正確)",
        "応答時間 (response_time)": "質問を投げてから回答を得るまでの時間（秒）。モデルの効率性を表す",
        "BLEU スコア (bleu_score)": "機械翻訳評価指標で、正解と回答のn-gramの一致度を測定 (0〜1の値、高いほど類似)",
        "類似度スコア (similarity_score)": "TF-IDFベクトルのコサイン類似度による、正解と回答の意味的な類似性 (0〜1の値)",
        "単語数 (word_count)": "回答に含まれる単語の数。情報量や詳細さの指標",
        "関連性スコア (relevance_score)": "正解と回答の共通単語の割合。トピックの関連性を表す (0〜1の値)",
        "効率性スコア (efficiency_score)": "正確性を応答時間で割った値。高速で正確な回答ほど高スコア",
        "感情スコア (sentiment_score)": "回答の感情的なトーン。0（ネガティブ）〜1（ポジティブ）のスケールで評価",
        "読みやすさスコア (readability_score)": "回答の文章構造の読みやすさ。文の長さと複雑さに基づく",
        "多様性スコア (diversity_score)": "回答に使用されている語彙の多様性。異なり語数/総語数の比率",
        "簡潔性スコア (conciseness_score)": "回答の簡潔さ。適切な長さであるかどうかを評価"
    }

def calculate_overall_quality_score(metrics_dict):
    """すべてのメトリクスから総合的な品質スコアを計算する"""
    # 各メトリクスの重み付け
    weights = {
        'is_correct': 0.25,
        'bleu_score': 0.15,
        'similarity_score': 0.15,
        'relevance_score': 0.15,
        'sentiment_score': 0.05,
        'readability_score': 0.10,
        'diversity_score': 0.05,
        'conciseness_score': 0.10
    }
    
    # 各メトリクスにウェイトを適用して合計
    score = 0
    for metric, weight in weights.items():
        if metric in metrics_dict and metrics_dict[metric] is not None:
            score += metrics_dict[metric] * weight
    
    # 総合スコアを0-100のスケールに変換
    return min(100, max(0, score * 100))