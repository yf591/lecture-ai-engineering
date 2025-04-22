# ui.py
import streamlit as st
import pandas as pd
import time
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions

# モダンなUIのためのカスタムCSS
def load_css():
    st.markdown("""
    <style>
    /* 全体のスタイル */
    .main {
        background-color: #f5f7fa;
        padding: 1rem;
    }
    
    /* カードスタイル */
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* チャットメッセージスタイル */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .message {
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 10px;
        position: relative;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-bottom-right-radius: 2px;
        margin-left: 1rem;
    }
    
    .bot-message {
        background-color: #f1f3f4;
        border-bottom-left-radius: 2px;
        margin-right: 1rem;
    }
    
    /* ダッシュボードスタイル */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        flex: 1;
        min-width: 120px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* セパレータースタイル */
    hr {
        margin: 1.5rem 0;
    }
    
    /* ヘッダースタイル */
    h1, h2, h3 {
        color: #333;
    }
    
    /* フィードバックフォームスタイル */
    .feedback-form {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

# チャットメッセージの表示
def display_message(is_user, content):
    """チャットメッセージを表示する"""
    message_class = "user-message" if is_user else "bot-message"
    sender = "You" if is_user else "AI"
    
    st.markdown(f"""
    <div class="message {message_class}">
        <p><strong>{sender}</strong></p>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# --- チャットページのUI ---
def display_chat_page(pipe):
    """モダンなチャットページUIを表示する"""
    load_css()
    
    # チャットUIのコンテナ
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # モダンなヘッダー
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="background-color: #4285F4; width: 40px; height: 40px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px;">
            <span style="color: white; font-size: 20px;">🤖</span>
        </div>
        <h2 style="margin: 0; color: #333;">AIチャットアシスタント</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # チャット履歴の初期化
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # セッション状態の初期化
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    
    # チャット履歴の表示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            display_message(msg["is_user"], msg["content"])
    
    # 入力欄内にボタンを配置するためのカスタムCSS
    st.markdown("""
    <style>
    .input-container {
        position: relative;
        margin-top: 20px;
    }
    .input-field {
        width: 100%;
        padding: 12px;
        padding-right: 60px;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-size: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .input-field:focus {
        outline: none;
        border-color: #4285F4;
    }
    .send-button {
        position: absolute;
        bottom: 8px;
        right: 10px;
        background-color: #4285F4;
        color: white;
        border: none;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .send-button:hover {
        background-color: #3367D6;
    }
    .send-icon {
        width: 16px;
        height: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 常に質問入力フォームを表示する（修正点：フィードバック状態に関わらず表示）
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("質問を入力", placeholder="AIに質問してみましょう...")
        submit_button = st.form_submit_button("送信")
        
        # 質問が送信された場合（フォーム内で処理）
        if submit_button and user_question:
            # 新しい質問が入力された場合、前の会話の状態をリセット（修正点：ここを追加）
            if st.session_state.current_answer and not st.session_state.feedback_given:
                # フィードバックが未提供の場合でも新しい会話を開始できるようにする
                st.session_state.current_answer = ""
                st.session_state.feedback_given = False
            
            st.session_state.current_question = user_question
            
            # チャット履歴に質問を追加
            st.session_state.chat_messages.append({
                "is_user": True,
                "content": user_question
            })
            
            # UIを更新して質問を表示
            st.rerun()
    
    # 質問が送信済みで、まだ回答が生成されていない場合
    if st.session_state.current_question and not st.session_state.current_answer:
        with st.spinner("回答を生成中..."):
            # 回答の生成
            answer, response_time = generate_response(pipe, st.session_state.current_question)
            st.session_state.current_answer = answer
            st.session_state.response_time = response_time
            
            # チャット履歴に回答を追加
            st.session_state.chat_messages.append({
                "is_user": False,
                "content": f"{answer}<br><small>応答時間: {response_time:.2f}秒</small>"
            })
            
            # フィードバックを取得するためにrerun
            st.rerun()
    
    # 回答生成後、まだフィードバックが提供されていない場合
    if st.session_state.current_answer and not st.session_state.feedback_given:
        st.markdown('<div class="feedback-form">', unsafe_allow_html=True)
        st.info("👇 フィードバックを提供するか、上の入力欄から次の質問を入力できます")
        display_feedback_form()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_feedback_form():
    """フィードバック入力フォームを表示する"""
    st.subheader("フィードバック")
    st.write("この回答はどうでしたか？")
    
    with st.form("feedback_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accurate = st.checkbox("正確", key="feedback_accurate")
        with col2:
            partially = st.checkbox("部分的に正確", key="feedback_partially")
        with col3:
            inaccurate = st.checkbox("不正確", key="feedback_inaccurate")
        
        correct_answer = st.text_area(
            "より正確な回答（任意）", 
            height=100,
            placeholder="より良い回答があれば、こちらに記入してください"
        )
        
        feedback_comment = st.text_area(
            "コメント（任意）", 
            height=100,
            placeholder="AIの改善点や良かった点など、自由にコメントしてください"
        )
        
        submitted = st.form_submit_button("送信")
        
        if submitted:
            # フィードバック値の決定
            if inaccurate:
                feedback = "不正確"
                is_correct = 0.0
            elif partially:
                feedback = "部分的に正確"
                is_correct = 0.5
            elif accurate:
                feedback = "正確"
                is_correct = 1.0
            else:
                feedback = "未評価"
                is_correct = None
                
            # フィードバックが未評価の場合はエラーメッセージを表示
            if is_correct is None:
                st.error("評価を選択してください（正確/部分的に正確/不正確）")
                return
                
            # コメントがある場合のみ結合
            combined_feedback = f"{feedback}"
            if feedback_comment:
                combined_feedback += f": {feedback_comment}"
            
            # データベースに保存
            save_to_db(
                st.session_state.current_question,
                st.session_state.current_answer,
                combined_feedback,
                correct_answer,
                is_correct,
                st.session_state.response_time
            )
            
            st.session_state.feedback_given = True
            st.success("フィードバックが送信されました！ありがとうございます。")
            
            # 次の質問ボタンを表示するため再読み込み
            st.rerun()
            
    # フィードバック送信済みの場合、次の質問ボタン
    if st.session_state.feedback_given:
        if st.button("次の質問へ", use_container_width=True):
            # 次の質問のために状態をリセット
            st.session_state.current_question = ""
            st.session_state.current_answer = ""
            st.session_state.response_time = 0.0
            st.session_state.feedback_given = False
            st.rerun()

# --- 履歴閲覧ページのUI ---
def display_history_page():
    """改良された履歴閲覧ページのUIを表示する"""
    load_css()
    
    st.header("チャット履歴と評価分析")
    
    history_df = get_chat_history()
    
    if history_df.empty:
        st.info("まだチャット履歴がありません。")
        return
    
    # タブでセクションを分ける
    tab1, tab2 = st.tabs(["📊 ダッシュボード", "💬 履歴閲覧"])
    
    with tab1:
        display_dashboard(history_df)
    
    with tab2:
        display_history_list(history_df)

def display_dashboard(history_df):
    """シンプルなダッシュボード表示"""
    st.subheader("AI パフォーマンスダッシュボード")
    
    # クリーンなデータセット（NaN値を処理）
    clean_df = history_df.copy()
    
    # カラムの存在確認と追加
    required_columns = {
        'is_correct': 0, 
        'response_time': 0, 
        'bleu_score': 0,
        'similarity_score': 0,
        'readability_score': 0,
        'sentiment_score': 0.5,
        'diversity_score': 0,
        'conciseness_score': 0,
        'quality_score': 0
    }
    
    # 存在しないカラムを追加
    for col, default_val in required_columns.items():
        if col not in clean_df.columns:
            clean_df[col] = default_val
    
    # NaN値を適切なデフォルト値で埋める
    clean_df = clean_df.fillna(required_columns)
    
    # 主要指標をカード表示
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_quality = clean_df['quality_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_quality:.1f}</div>
            <div class="metric-label">平均品質スコア</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        avg_response_time = clean_df['response_time'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_response_time:.2f}秒</div>
            <div class="metric-label">平均応答時間</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        accuracy_rate = (clean_df['is_correct'] >= 0.5).mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy_rate:.1f}%</div>
            <div class="metric-label">正確度</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        total_qa = len(clean_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_qa}</div>
            <div class="metric-label">総質問数</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 正確性の分布
    st.subheader("回答精度の分布")
    accuracy_counts = clean_df['is_correct'].map({1.0: '正確', 0.5: '部分的に正確', 0.0: '不正確'}).value_counts()
    st.bar_chart(accuracy_counts)
    
    # 評価指標の分布
    st.subheader("評価指標の分析")
    
    metrics_to_plot = ['bleu_score', 'similarity_score', 'relevance_score', 
                       'sentiment_score', 'readability_score', 'diversity_score', 
                       'conciseness_score']
    
    # 利用可能な指標を選択肢に含める
    valid_metrics = [m for m in metrics_to_plot if m in clean_df.columns and clean_df[m].notna().any()]
    
    if valid_metrics:
        metric_option = st.selectbox(
            "表示する評価指標を選択",
            valid_metrics,
            format_func=lambda x: {
                'bleu_score': 'BLEU',
                'similarity_score': '類似度',
                'relevance_score': '関連性',
                'sentiment_score': '感情分析',
                'readability_score': '読みやすさ',
                'diversity_score': '語彙の多様性',
                'conciseness_score': '簡潔性'
            }.get(x, x)
        )
        
        st.bar_chart(clean_df[metric_option])
    else:
        st.info("表示可能な評価指標データがありません。")

def display_history_list(history_df):
    """改良された履歴リストを表示する"""
    st.subheader("チャット履歴")
    
    # フィルター
    filter_options = {
        "すべて表示": None,
        "正確なもののみ": 1.0,
        "部分的に正確なもののみ": 0.5,
        "不正確なもののみ": 0.0
    }
    
    display_option = st.selectbox(
        "フィルター:",
        options=list(filter_options.keys()),
    )
    
    # フィルタリングとソート
    filter_value = filter_options[display_option]
    if filter_value is not None:
        filtered_df = history_df[history_df["is_correct"].notna() & (history_df["is_correct"] == filter_value)]
    else:
        filtered_df = history_df
    
    if filtered_df.empty:
        st.info("選択した条件に一致する履歴はありません。")
        return
    
    # ページネーション
    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    current_page = st.slider("ページ", min_value=1, max_value=max(1, total_pages), value=1)
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    st.markdown(f"**{total_items}件中 {start_idx+1}-{end_idx}件を表示**")
    
    # 履歴リスト表示
    for i, row in filtered_df.iloc[start_idx:end_idx].iterrows():
        # 正確性によって色を変更
        if row['is_correct'] == 1.0:
            header_color = "#4CAF50"  # 緑
            accuracy_label = "正確"
        elif row['is_correct'] == 0.5:
            header_color = "#FFC107"  # 黄
            accuracy_label = "部分的に正確"
        else:
            header_color = "#F44336"  # 赤
            accuracy_label = "不正確"
        
        # カード内表示用の短い質問テキスト
        short_question = row['question'][:100] + "..." if len(row['question']) > 100 else row['question']
        
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: #666;">{row['timestamp']}</span>
                <span style="background-color: {header_color}; color: white; padding: 2px 8px; border-radius: 10px;">{accuracy_label}</span>
            </div>
            <p style="font-weight: 500;">{short_question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("詳細を表示"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**質問:**")
                st.write(row['question'])
                
                st.markdown("**回答:**")
                st.write(row['answer'])
                
                if pd.notna(row['correct_answer']) and row['correct_answer']:
                    st.markdown("**正しい回答:**")
                    st.write(row['correct_answer'])
                
                if pd.notna(row['feedback']) and row['feedback']:
                    st.info(f"**フィードバック:** {row['feedback']}")
            
            with col2:
                st.markdown("**評価指標:**")
                st.markdown(f"**品質スコア:** {row.get('quality_score', 0):.1f}/100")
                st.markdown(f"**応答時間:** {row.get('response_time', 0):.2f}秒")
                st.markdown(f"**単語数:** {row.get('word_count', 0)}")
                st.markdown(f"**BLEU:** {row.get('bleu_score', 0):.2f}")
                st.markdown(f"**類似度:** {row.get('similarity_score', 0):.2f}")

# --- サンプルデータ管理ページのUI ---
def display_data_page():
    """改良されたサンプルデータ管理ページのUIを表示する"""
    load_css()
    
    st.header("サンプルデータ & 評価指標管理")
    
    # データ管理セクション
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("データ管理")
    
    count = get_db_count()
    
    # プログレスバーでデータ量を視覚化（最大50件と仮定）
    progress_value = min(count / 50, 1.0)
    
    st.progress(progress_value)
    st.markdown(f"現在のデータベースには **{count}件** のレコードがあります。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("サンプルデータを追加", use_container_width=True):
            create_sample_evaluation_data()
            st.rerun()
    
    with col2:
        if st.button("データベースをクリア", use_container_width=True):
            if clear_db():
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 評価指標の解説
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("評価指標の解説")
    
    metrics_info = get_metrics_descriptions()
    
    # カードUIで評価指標を表示
    metrics_cols = st.columns(2)
    
    for i, (metric, description) in enumerate(metrics_info.items()):
        col_idx = i % 2
        
        with metrics_cols[col_idx]:
            with st.expander(f"{metric}"):
                st.write(description)
    
    st.markdown('</div>', unsafe_allow_html=True)