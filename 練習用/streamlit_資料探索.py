# 導入必要的庫
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 側邊欄

with st.sidebar:
    st.title("CSV 資料探索")
    uploaded_file = st.file_uploader("選擇 CSV file", type=["csv"])
    st.write("資料探索是理解資料的特性和分佈的一個過程，這個網頁可上傳 CSV 檔案，接著選著感興趣的目標變數，即可探索。這包括：")
    st.write("- 直方圖和箱形圖")
    st.write("- 時間序列圖(如果有時間欄位)")
    st.write("- 散佈圖")
    st.write("- 數據統計概述")
    st.write("- 相關性分析")
    st.write("或是使用內建的 iris 資料集，可勾選下方的選項。")
    use_example_dataset = st.checkbox("使用範例資料集：iris")  # 新增勾選框

    if use_example_dataset:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        # df['target'] = pd.Series(iris.target)
        use_example_data = True #新增變數，確認使用者使用內建資料集
        st.write("已使用 Iris 範例資料集。")  # 確認訊息
    else:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            use_example_data = False #確認使用者使用上傳資料集
        else:
            use_example_data = False

    if use_example_data or (uploaded_file is not None):  # 只有在上傳或使用範例資料集時才顯示後續選項
        selected_variable = st.selectbox("選擇要探索的變數", df.columns)
        st.write("Required packages:")
        st.markdown('+ streamlit, numpy, pandas, plotly, matplotlib, seaborn, scikit-learn', unsafe_allow_html=True)
        if st.button("GO"):
            run_analysis = True
        else:
            run_analysis = False
    else:
        run_analysis = False
# 主區域
if run_analysis:
    st.subheader(f"目標變數：{selected_variable} ")

    # 1. 直方圖和箱形圖
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("直方圖")
        try:
            fig_hist = px.histogram(df, x=selected_variable)
            st.plotly_chart(fig_hist)
        except Exception as e:
            st.write(f"繪製直方圖時發生錯誤：{e}")
    with col2:
        st.subheader("箱形圖")
        fig_box = px.box(df, y=selected_variable)
        st.plotly_chart(fig_box)

    # 1.5 時趨圖
    time_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "Date", "DATE", "time", "Time", "TIME", "timestamp", "Timestamp", "TIMESTAMP"]):
            try:
                pd.to_datetime(df[col])
                time_col = col
                break
            except:
                pass

    # 處理時間格式並繪製時趨圖
    if time_col:
        st.subheader("時趨圖")
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            fig_time_series = px.line(df, x=time_col, y=selected_variable)
            st.plotly_chart(fig_time_series)
        except Exception as e:
            st.write(f"繪製時趨圖時發生錯誤：{e}")
    else:
        st.subheader("時趨圖")
        st.write("未偵測到時間欄位，無法繪製時趨圖。")

    # 2. 散佈圖（每三個水平陳列）
    st.subheader("散佈圖 (目標 vs 其他數值變數)")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    scatter_cols = [col for col in numeric_cols if col != selected_variable]
    num_scatter_plots = len(scatter_cols)
    for i in range(0, num_scatter_plots, 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < num_scatter_plots:
                with cols[j]:
                    # 檢查變數是否為數值型
                    if pd.api.types.is_numeric_dtype(df[scatter_cols[i + j]]) and pd.api.types.is_numeric_dtype(df[selected_variable]):
                        fig_scatter = go.Figure(data=go.Scatter(x=df[scatter_cols[i + j]], y=df[selected_variable], mode='markers') )
                        fig_scatter.update_layout(
                            xaxis_title=scatter_cols[i + j],
                            yaxis_title=selected_variable
                        )
                        st.plotly_chart(fig_scatter)
                    else:
                        st.write(f"{scatter_cols[i+j]} 或 {selected_variable} 不是數值型變數.")
    
    # 3. 交叉表數據和統計概述
    st.subheader("原始數據")
    st.dataframe(df)

    st.subheader("數據之統計概述")

    # 檢查資料類型
    st.write("資料類型：")
    st.write(df.dtypes)

    # 檢查資料內容
    # st.write("資料前幾行：")
    # st.write(df.head())

    # 確保所有數值欄位都是數值類型
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 排除非數值欄位
    numeric_df = df.select_dtypes(include=['number'])

    # 檢查數值欄位
    if not numeric_df.empty:
        stats = pd.DataFrame({
            # "數據類型": numeric_df.dtypes,
            "缺失值數量": numeric_df.isnull().sum(),
            "最小值": numeric_df.min(),
            "第一四分位數": numeric_df.quantile(0.25),
            "中位數": numeric_df.median(),
            "第三四分位數": numeric_df.quantile(0.75),
            "最大值": numeric_df.max()
        })
        st.dataframe(stats)
    else:
        st.write("No Numeric Variable Found")

    # 4. 相關性圖
    st.subheader("相關性分析")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr()

        # 創建遮罩，只顯示高相關性的數字
        # mask = np.abs(corr) < 0.8
        # mask = np.triu(mask) # 只顯示上半部分

        fig_corr, ax = plt.subplots()
        #sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, mask=mask, fmt=".2f") # fmt控制小數位數
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f") # fmt控制小數位數
        st.pyplot(fig_corr)

        # 突出顯示選定的變數
        if selected_variable in numeric_df.columns:
            st.subheader(f"{selected_variable} 與其他變數之間的相關係數")
            selected_corr = corr[selected_variable].sort_values(ascending=False)
            st.write(selected_corr)
